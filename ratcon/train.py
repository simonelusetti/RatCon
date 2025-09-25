# train.py
import os, sys, logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import RationaleSelectorModel, nt_xent
from .data import get_dataset, collate
from .losses import sparsity_loss, total_variation_1d
from .inference import evaluate
from dora import get_xp, hydra_main

torch.set_num_threads(os.cpu_count())
torch.autograd.set_detect_anomaly(True)

# -------------------------
# Logging (kept your style)
# -------------------------
class TqdmLoggingHandler(logging.Handler):
    """A logging handler that plays nice with tqdm progress bars."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)   # avoids breaking the bar
            sys.stdout.flush()
        except Exception:
            self.handleError(record)

def get_logger(logfile="train.log"):
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)  # capture all, filtering is per-handler
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()

    ch = TqdmLoggingHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

# -----------------------------------------
# Training epoch (adversarial min–max loop)
# -----------------------------------------
def train_epoch_adv(
    model, loader, opt_sel, opt_adv, device, epoch,
    verbose=False, tau=0.07, l_comp=0.5, l_s=0.01, l_tv=0.02,
    grad_clip=1.0, logger=None
):
    """
    Train one epoch with adversarial complement predictor.
    - opt_sel: optimizer for selector/encoder/projections
    - opt_adv: optimizer for complement adversary
    """
    model.train()
    total = 0.0

    for batch in tqdm(loader, desc=f"Epoch: {epoch+1}"):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        # forward: model returns h_anchor, h_rat, h_c_adv, gates, ...
        out = model(**batch, verbose=verbose, logger=logger)
        h_a, h_r, h_c_adv, g = out["h_anchor"], out["h_rat"], out["h_c_adv"], out["gates"]

        # ---- losses ----
        # rationale InfoNCE (selector wants this LOW)
        L_nce  = nt_xent(h_a, h_r, temperature=tau)

        # adversary InfoNCE: adversary wants this HIGH, selector penalizes it
        # detaching anchor improves stability of the adversary step
        L_comp = nt_xent(h_a.detach(), h_c_adv, temperature=tau)

        # regularizers on gates (mask-aware)
        L_s  = sparsity_loss(g, batch["attention_mask"])
        L_tv = total_variation_1d(g, batch["attention_mask"])

        # ---- 1) Update adversary: maximize L_comp ----
        for p in model.comp_adv.parameters():
            p.requires_grad_(True)
        opt_adv.zero_grad(set_to_none=True)
        (-L_comp).backward(retain_graph=True)           # ascent on adversary
        torch.nn.utils.clip_grad_norm_(model.comp_adv.parameters(), grad_clip)
        opt_adv.step()

        # ---- 2) Update selector (and everything except adversary): minimize total ----
        for p in model.comp_adv.parameters():
            p.requires_grad_(False)
        opt_sel.zero_grad(set_to_none=True)
        loss = L_nce + l_comp * L_comp + l_s * L_s + l_tv * L_tv
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            grad_clip
        )
        opt_sel.step()

        # ---- logging & accounting ----
        bs = batch["embeddings"].size(0) if "embeddings" in batch else h_a.size(0)
        loss_step = loss.item() * bs
        if verbose and logger is not None:
            logger.debug(
                f"step loss: {loss_step:.4f} "
                f"(nce {L_nce.item():.4f}, comp {L_comp.item():.4f}, "
                f"s {L_s.item():.4f}, tv {L_tv.item():.4f})"
            )
        total += loss_step

    return total / len(loader.dataset)

# ----------------
# Eval wrapper
# ----------------
def eval_once(xp, model, eval_ds, tok, cfg, logger):
    dataset_name = cfg.data.eval.dataset
    metrics, samples = evaluate(model, eval_ds, tok, cfg=cfg, logger=logger)
    xp.link.push_metrics({"dataset": dataset_name, "metrics": metrics})
    logger.info(f"dataset: {dataset_name}\n metrics: {metrics}")
    if cfg.eval.samples.show:
        for s in samples:
            logger.info(s)
    return metrics

# -----------
# Main
# -----------
@hydra_main(config_path="conf", config_name="default", version_base="1.1")
def main(cfg):
    logger = get_logger("train.log")
    xp = get_xp()
    logger.info(f"Exp signature: {xp.sig}")
    logger.info(repr(cfg))
    logger.info(f"Work dir: {os.getcwd()}")
    logger.info(f"Exec file: {__file__}")

    # ----------------------------
    # Data
    # ----------------------------
    train_ds, tok = get_dataset(
        name=cfg.data.train.dataset,
        subset=cfg.data.train.subset,
        rebuild=cfg.data.rebuild_ds,
        shuffle=cfg.data.train.shuffle
    )
    eval_ds, _ = get_dataset(
        split="validation",
        name=cfg.data.eval.dataset,
        subset=cfg.data.eval.subset,
        rebuild=cfg.data.rebuild_ds,
        shuffle=cfg.data.eval.shuffle
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.data.train.batch_size,
        collate_fn=collate,
        num_workers=cfg.data.train.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.data.train.num_workers > 0)
    )

    # ----------------------------
    # Device
    # ----------------------------
    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("No GPU available, switching to CPU")
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"

    # ----------------------------
    # Model
    # ----------------------------
    model = RationaleSelectorModel(
        proj_dim=cfg.model.proj_dim
    ).to(cfg.device)

    # Optionally load checkpoint
    if os.path.exists("model.pth") and not getattr(cfg.train, "retrain", False):
        logger.info("Loading model from model.pth")
        model.load_state_dict(torch.load("model.pth", map_location=cfg.device))
    else:
        logger.info("Retraining model from scratch")

    # ----------------------------
    # Optimizers (split: adversary vs the rest)
    # ----------------------------
    # Collect adversary params explicitly
    adv_params = list(model.comp_adv.parameters())
    adv_ids = {id(p) for p in adv_params}
    sel_params = [p for p in model.parameters() if id(p) not in adv_ids]

    opt_sel = torch.optim.AdamW(
        sel_params,
        lr=cfg.model.optim.lr,
        weight_decay=cfg.model.optim.weight_decay,
        betas=getattr(cfg.model.optim, "betas", (0.9, 0.999))
    )
    opt_adv = torch.optim.AdamW(
        adv_params,
        lr=getattr(cfg.model.optim, "lr_adv", cfg.model.optim.lr),
        weight_decay=getattr(cfg.model.optim, "weight_decay_adv", cfg.model.optim.weight_decay),
        betas=getattr(cfg.model.optim, "betas", (0.9, 0.999))
    )

    # ----------------------------
    # Eval-only?
    # ----------------------------
    if cfg.eval.eval_only:
        eval_once(xp, model, eval_ds, tok, cfg, logger)
        return

    # ----------------------------
    # Train
    # ----------------------------
    best_f1 = 0.0
    tau   = getattr(cfg.model.loss, "tau", 0.07)
    l_comp = cfg.model.loss.l_comp
    l_s    = cfg.model.loss.l_s
    l_tv   = cfg.model.loss.l_tv
    grad_clip = getattr(cfg.train, "grad_clip", 1.0)

    for epoch in range(cfg.train.epochs):
        avg = train_epoch_adv(
            model=model,
            loader=train_dl,
            opt_sel=opt_sel,
            opt_adv=opt_adv,
            device=cfg.device,
            epoch=epoch,
            tau=tau,
            l_comp=l_comp,
            l_s=l_s,
            l_tv=l_tv,
            grad_clip=grad_clip,
            verbose=cfg.eval.verbose,
            logger=logger
        )

        logger.info(f"epoch {epoch+1}: loss {avg:.4f}")

        # Eval with adversary in eval mode (no special handling needed)
        model.eval()
        with torch.no_grad():
            metrics = eval_once(xp, model, eval_ds, tok, cfg, logger)

        if metrics.get("f1", 0.0) > best_f1:
            best_f1 = metrics["f1"]
            logger.info(f"New best F1: {best_f1:.4f}, saving model to model.pth")
            torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
