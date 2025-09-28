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
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------
# Logging
# -------------------------
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            sys.stdout.flush()
        except Exception:
            self.handleError(record)

def get_logger(logfile="train.log"):
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
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

# -------------------------
# Training epoch
# -------------------------
def train_epoch(
    model, loader, optimizer, epoch, cfg, logger=None
):
    tau   = getattr(cfg.model.loss, "tau", 0.07)
    device = cfg.device
    l_comp = cfg.model.loss.l_comp
    l_s    = cfg.model.loss.l_s
    l_tv   = cfg.model.loss.l_tv
    grad_clip = getattr(cfg.train, "grad_clip", 1.0)
    total = 0.0
    
    model.train()

    incoming = outgoing = None
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        embeddings = batch["embeddings"]
        attention_mask = batch["attention_mask"]
        if cfg.model.attention_augment:
            incoming = batch["incoming"]
            outgoing = batch["outgoing"]
        out = model(embeddings, attention_mask, incoming, outgoing)  # returns h_anchor, h_rat, h_comp, gates, ...

        h_a, h_r, h_c, g = out["h_anchor"], out["h_rat"], out["h_comp"], out["gates"]

        # losses
        L_rat = nt_xent(h_a, h_r, temperature=tau)
        L_comp = nt_xent(h_a, h_c, temperature=tau)
        L_s = sparsity_loss(g, batch["attention_mask"])
        L_tv = total_variation_1d(g, batch["attention_mask"])

        loss = L_rat - l_comp * L_comp + l_s * L_s + l_tv * L_tv

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # logging
        bs = batch["embeddings"].size(0)
        loss_step = loss.item() * bs
        if cfg.eval.verbose and logger is not None:
            logger.debug(
                f"step loss: {loss_step:.4f} "
                f"(rat {L_rat.item():.4f}, comp {L_comp.item():.4f}, "
                f"s {L_s.item():.4f}, tv {L_tv.item():.4f})"
            )
        total += loss_step

    return total / len(loader.dataset)

# -------------------------
# Eval wrapper
# -------------------------
def eval_once(xp, model, eval_dl, tok, cfg, logger):
    dataset_name = cfg.data.eval.dataset
    metrics, samples = evaluate(model, eval_dl, tok, cfg=cfg, logger=logger)
    xp.link.push_metrics({"dataset": dataset_name, "metrics": metrics})
    if cfg.eval.samples.show:
        for s in samples:
            logger.info(s)
    logger.info(f"dataset: {dataset_name}\n metrics: {metrics}")
    return metrics

# -------------------------
# Main
# -------------------------
@hydra_main(config_path="conf", config_name="default", version_base="1.1")
def main(cfg):
    logger = get_logger("train.log")
    xp = get_xp()
    logger.info(f"Exp signature: {xp.sig}")
    logger.info(repr(cfg))
    logger.info(f"Work dir: {os.getcwd()}")
    logger.info(f"Exec file: {__file__}")

    # --- Data
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
        persistent_workers=(cfg.data.train.num_workers > 0),
        shuffle=cfg.data.train.shuffle
    )
    
    eval_dl = DataLoader(
        eval_ds,
        batch_size=cfg.data.eval.batch_size,
        collate_fn=collate,
        num_workers=cfg.data.eval.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.data.eval.num_workers > 0),
        shuffle=cfg.data.eval.shuffle
    )


    # --- Device
    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("No GPU available, switching to CPU")
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"

    # --- Model
    model = RationaleSelectorModel(attention_augment=cfg.model.attention_augment).to(cfg.device)

    if os.path.exists("model.pth") and not cfg.train.retrain:
        logger.info("Loading model from model.pth")
        state = torch.load("model.pth", map_location=cfg.device)
        model.load_state_dict(state)  # tolerate new/missing keys
    else:
        logger.info("Training model from scratch")

    # --- Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.model.optim.lr,
        weight_decay=cfg.model.optim.weight_decay,
        betas=getattr(cfg.model.optim, "betas", (0.9, 0.999))
    )

    # --- Eval-only?
    if cfg.eval.eval_only:
        eval_once(xp, model, eval_dl, tok, cfg, logger)
        return

    # --- Training loop
    best_f1 = 0.0

    for epoch in range(cfg.train.epochs):
        avg = train_epoch(
            model=model,
            loader=train_dl,
            optimizer=optimizer,
            epoch=epoch,
            cfg=cfg,
            logger=logger,
        )
        logger.info(f"epoch {epoch+1}: loss {avg:.4f}")

        # Eval
        model.eval()
        with torch.no_grad():
            metrics = eval_once(xp, model, eval_dl, tok, cfg, logger)

        if metrics.get("f1", 0.0) > best_f1:
            best_f1 = metrics["f1"]
            logger.info(f"New best F1: {best_f1:.4f}, saving model to model.pth")
            torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
