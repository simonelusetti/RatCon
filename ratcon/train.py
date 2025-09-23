# train.py
import torch, os, logging, sys
from tqdm import tqdm
from torch.utils.data import DataLoader
from .models import RationaleSelectorModel, nt_xent
from .data import get_datasets, collate, get_wikiann
from .losses import complement_margin_loss, sparsity_loss, total_variation_1d
from .inference import sample_inference, evaluate
from dora import get_xp, hydra_main

torch.set_num_threads(os.cpu_count())
torch.autograd.set_detect_anomaly(True)

import logging
from tqdm import tqdm
import sys

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

    # Console handler (INFO and above)
    ch = TqdmLoggingHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # File handler (DEBUG and above → includes verbose)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def train_epoch(model, loader, optimizer, device, epoch, verbose=False,
                    tau=0.07, l_comp=0.5, l_s=0.01, l_tv=0.02, grad_clip=1.0, logger=None):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc=f"Epoch: {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch, verbose=verbose, logger=logger)

        L_nce  = nt_xent(out["h_anchor"], out["h_rat"], temperature=tau)
        L_comp = complement_margin_loss(out["h_anchor"], out["h_comp"], margin=0.3)
        L_s    = sparsity_loss(out["gates"], batch["attention_mask"])
        L_tv   = total_variation_1d(out["gates"], batch["attention_mask"])

        loss = L_nce + l_comp*L_comp + l_s*L_s + l_tv*L_tv

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_step = loss.item() * batch["embeddings"].size(0)
        if verbose:
            logger.debug(f"step loss: {loss_step:.4f} (nce {L_nce.item():.4f}, comp {L_comp.item():.4f}, s {L_s.item():.4f}, tv {L_tv.item():.4f})")
        total += loss_step
        
    return total / len(loader.dataset)

def eval(xp, model, sample_ds, eval_ds, tok, cfg, logger):
    metrics, samples = evaluate(model, eval_ds, tok, cfg.device, thresh=cfg.eval.thresh, logger=logger)
    xp.link.push_metrics(metrics)
    logger.info(f"Metrics: {metrics}")
    if cfg.eval.examples:
        for s in samples:
            logger.info(f"--- {s}")
        logger.info(sample_inference(model, tok, sample_ds, cfg.device, verbose=cfg.eval.verbose, thresh=cfg.eval.thresh, logger=logger))
            

@hydra_main(config_path="conf", config_name="default", version_base="1.1")
def main(cfg):
    logger = get_logger("train.log")
    xp = get_xp()
    logger.info(f"Exp signature: {xp.sig}")
    logger.info(repr(cfg))
    logger.info(f"Work dir: {os.getcwd()}")
    logger.info(f"Exec file: {__file__}")
    
    train_ds, tok = get_datasets(subset=cfg.train.data.subset,rebuild=cfg.eval.rebuild_ds)
    sample_ds, _   = get_datasets(split="validation", subset=cfg.eval.data.subset, rebuild=cfg.eval.rebuild_ds)
    eval_ds, _     = get_wikiann(split="validation", subset=cfg.eval.data.subset, rebuild=cfg.eval.rebuild_ds)
        
    train_dl = DataLoader(
        train_ds, 
        batch_size=cfg.train.data.batch_size, 
        shuffle=cfg.train.data.shuffle,
        collate_fn=collate, 
        num_workers=cfg.train.data.num_workers
    )

    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("No GPU available, switching to CPU")
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"
    
    model = RationaleSelectorModel().to(cfg.device)
    if os.path.exists("model.pth") and not cfg.train.retrain:
        logger.info("Loading model from model.pth")
        model.load_state_dict(torch.load("model.pth", map_location=cfg.device))
    else:
        logger.info("Retraining model from scratch")
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    if cfg.eval.eval_only:
        eval(xp,model, sample_ds, eval_ds, tok, cfg, logger)
        return

    for epoch in range(cfg.train.epochs):
        avg = train_epoch(
            model,
            train_dl,
            optim, 
            cfg.device, 
            epoch, 
            l_comp=cfg.train.loss.l_comp,
            l_s=cfg.train.loss.l_s,
            l_tv=cfg.train.loss.l_tv,
            verbose=cfg.eval.verbose, 
            logger=logger
        )
        
        logger.info(f"epoch {epoch+1}: loss {avg:.4f}")
        eval(xp, model, sample_ds, eval_ds, tok, cfg, logger)
        torch.save(model.state_dict(), "model.pth")
        logger.info("Model saved to model.pth")

if __name__ == "__main__":
    main()
