import os
import sys
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import RationaleSelectorModel, nt_xent
from .data import get_dataset, collate
from .losses import sparsity_loss, total_variation_1d
from .evaluate import evaluate
from dora import get_xp, hydra_main

# -------------------------------------------------------------------
# Torch setup
# -------------------------------------------------------------------
torch.set_num_threads(os.cpu_count())
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# Loss helpers
# -------------------------------------------------------------------


def complement_null_loss(h_comp, h_anchor):
    """Encourage complements to collapse to a null embedding while remaining orthogonal to anchors."""
    null_loss = h_comp.pow(2).sum(dim=-1).mean()
    cosine = F.cosine_similarity(h_comp, h_anchor, dim=-1)
    orth_loss = cosine.pow(2).mean()
    return null_loss + orth_loss


# -------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.use_dual = cfg.model.dual.use
        self.model1, self.model2, self.optimizer = self.build_models_and_optimizer()

    def build_models_and_optimizer(self):
        model1 = RationaleSelectorModel(cfg=self.cfg.model).to(self.cfg.device)
        model2 = RationaleSelectorModel(cfg=self.cfg.model).to(self.cfg.device) if self.use_dual else None

        if self.use_dual:
            # Load or train both models
            for idx, model in enumerate([model1, model2], start=1):
                path = f"model{idx}.pth"
                if os.path.exists(path) and (not self.cfg.train.retrain or self.cfg.eval.eval_only):
                    self.logger.info(f"Loading model{idx} from {path}")
                    state = torch.load(path, map_location=self.cfg.device)
                    model.load_state_dict(state)
                else:
                    self.logger.info(f"Training model{idx} from scratch")

            optimizer = torch.optim.AdamW(
                list(model1.parameters()) + list(model2.parameters()),
                lr=self.cfg.model.optim.lr,
                weight_decay=self.cfg.model.optim.weight_decay,
                betas=self.cfg.model.optim.betas
            )
        else:
            # Single model
            if os.path.exists("model.pth") and (not self.cfg.train.retrain or self.cfg.eval.eval_only):
                self.logger.info("Loading model from model.pth")
                state = torch.load("model.pth", map_location=self.cfg.device)
                model1.load_state_dict(state)
            else:
                self.logger.info("Training model from scratch")

            optimizer = torch.optim.AdamW(
                model1.parameters(),
                lr=self.cfg.model.optim.lr,
                weight_decay=self.cfg.model.optim.weight_decay,
                betas=self.cfg.model.optim.betas
            )

        return model1, model2, optimizer

    def train_epoch(self, loader, epoch):
        tau = self.cfg.model.loss.tau
        l_comp, l_s, l_tv = (
            self.cfg.model.loss.l_comp,
            self.cfg.model.loss.l_s,
            self.cfg.model.loss.l_tv,
        )
        grad_clip = self.cfg.train.grad_clip
        device = self.cfg.device

        total = 0.0
        self.model1.train()
        if self.model2:
            self.model2.train()

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            embeddings, attention_mask = batch["embeddings"], batch["attention_mask"]
            incoming = batch["incoming"] if self.cfg.model.attention_augment else None
            outgoing = batch["outgoing"] if self.cfg.model.attention_augment else None

            # Forward pass model1
            out1 = self.model1(embeddings, attention_mask, incoming, outgoing)
            h_a1, h_r1, h_c1, g1 = out1["h_anchor"], out1["h_rat"], out1["h_comp"], out1["gates"]

            L_rat1 = nt_xent(h_r1, h_a1, temperature=tau)
            L_comp1 = complement_null_loss(h_c1, h_a1)
            L_s1 = sparsity_loss(g1, attention_mask)
            L_tv1 = total_variation_1d(g1, attention_mask)

            if self.model2:
                out2 = self.model2(embeddings, attention_mask, incoming, outgoing)
                h_a2, h_r2, h_c2, g2 = out2["h_anchor"], out2["h_rat"], out2["h_comp"], out2["gates"]

                L_rat2 = nt_xent(h_r2, h_a2, temperature=tau)
                L_comp2 = complement_null_loss(h_c2, h_a2)
                L_s2 = sparsity_loss(g2, attention_mask)
                L_tv2 = total_variation_1d(g2, attention_mask)

                # Symmetric KL loss
                kl_weight = self.cfg.model.dual.kl_weight
                g1_soft, g2_soft = torch.clamp(g1, 1e-6, 1 - 1e-6), torch.clamp(g2, 1e-6, 1 - 1e-6)
                kl_1_2 = torch.sum(g1_soft * (g1_soft.log() - g2_soft.log()), dim=1).mean()
                kl_2_1 = torch.sum(g2_soft * (g2_soft.log() - g1_soft.log()), dim=1).mean()
                kl_loss = 0.5 * (kl_1_2 + kl_2_1)
                
                ls_1 = self.cfg.model.dual.ls_1
                ls_2 = self.cfg.model.dual.ls_2

                loss = (
                    (L_rat1 + l_comp * L_comp1 + ls_1 * L_s1 + l_tv * L_tv1)
                    + (L_rat2 + l_comp * L_comp2 + ls_2 * L_s2 + l_tv * L_tv2)
                    + kl_weight * kl_loss
                )
                params = list(self.model1.parameters()) + list(self.model2.parameters())
            else:
                loss = L_rat1 + l_comp * L_comp1 + l_s * L_s1 + l_tv * L_tv1
                params = self.model1.parameters()

            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            self.optimizer.step()

            bs = embeddings.size(0)
            total += loss.item() * bs

        return total / len(loader.dataset)

    def eval(self, xp, model, eval_dl, tok, per_sentence_stats=False):
        import datetime

        dataset_name = self.cfg.data.eval.dataset
        metrics, samples, word_stats = evaluate(model, eval_dl, tok, cfg=self.cfg, logger=self.logger)
        timestamp = datetime.datetime.now().isoformat()
        xp.link.push_metrics(
            {"dataset": dataset_name, "metrics": metrics, "timestamp": timestamp, "word_stats": word_stats}
        )

        if self.cfg.eval.samples.show:
            for i, s in enumerate(samples):
                extra = f"\nWord stats: {word_stats[i]}\n" if i < len(word_stats) and per_sentence_stats else ""
                self.logger.info(f"{s}{extra}")

        self.logger.info(f"dataset: {dataset_name}\nmetrics: {metrics}")

        # Aggregate avg word stats
        if word_stats:
            total_words = sum(d["total"] for d in word_stats if d["total"] > 0)
            if total_words > 0:
                avg_props = {
                    k: sum(d.get(k, 0) for d in word_stats if d["total"] > 0) / total_words
                    for k in ["nouns", "proper_nouns", "verbs", "conjugations", "stopwords"]
                }
                self.logger.info(f"Average highlighted word proportions: {avg_props}")
            else:
                self.logger.info("No highlighted words to compute average proportions.")
        return metrics

    def train(self, train_dl, eval_dl, tok, xp, per_sentence_stats=False):
        best_f1 = 0.0
        for epoch in range(self.cfg.train.epochs):
            avg = self.train_epoch(train_dl, epoch)
            self.logger.info(f"epoch {epoch+1}: loss {avg:.4f}")

            if self.use_dual:
                self.model1.eval()
                self.model2.eval()
                with torch.no_grad():
                    metrics1 = self.eval(xp, self.model1, eval_dl, tok, per_sentence_stats)
                    metrics2 = self.eval(xp, self.model2, eval_dl, tok, per_sentence_stats)
                f1_1, f1_2 = metrics1.get("f1", 0.0), metrics2.get("f1", 0.0)
                if max(f1_1, f1_2) > best_f1:
                    best_f1 = max(f1_1, f1_2)
                    self.logger.info(
                        f"New best F1: {best_f1:.4f}, saving models to model1.pth and model2.pth"
                    )
                    torch.save(self.model1.state_dict(), "model1.pth")
                    torch.save(self.model2.state_dict(), "model2.pth")
            else:
                self.model1.eval()
                with torch.no_grad():
                    metrics = self.eval(xp, self.model1, eval_dl, tok, per_sentence_stats)
                if metrics.get("f1", 0.0) > best_f1:
                    best_f1 = metrics["f1"]
                    self.logger.info(f"New best F1: {best_f1:.4f}, saving model to model.pth")
                    torch.save(self.model1.state_dict(), "model.pth")

        self.logger.info(f"Best F1: {best_f1:.4f}")

    def eval_only(self, eval_dl, tok, xp, per_sentence_stats=False):
        if self.use_dual:
            self.logger.info("Eval-only mode: evaluating model1")
            self.eval(xp, self.model1, eval_dl, tok, per_sentence_stats)
            self.logger.info("Eval-only mode: evaluating model2")
            self.eval(xp, self.model2, eval_dl, tok, per_sentence_stats)
        else:
            self.eval(xp, self.model1, eval_dl, tok, per_sentence_stats)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
@hydra_main(config_path="conf", config_name="default", version_base="1.1")
def main(cfg):
    logger = get_logger("train.log")
    xp = get_xp()
    logger.info(f"Exp signature: {xp.sig}")
    logger.info(repr(cfg))
    logger.info(f"Work dir: {os.getcwd()}")
    logger.info(f"Exec file: {__file__}")

    # Device setup
    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("No GPU available, switching to CPU")
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"

    # Data
    train_ds, tok = get_dataset(
        name=cfg.data.train.dataset,
        subset=cfg.data.train.subset,
        rebuild=cfg.data.rebuild_ds,
        shuffle=cfg.data.train.shuffle,
    )
    eval_ds, _ = get_dataset(
        split="validation",
        name=cfg.data.eval.dataset,
        subset=cfg.data.eval.subset,
        rebuild=cfg.data.rebuild_ds,
        shuffle=cfg.data.eval.shuffle,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.data.train.batch_size,
        collate_fn=collate,
        num_workers=cfg.data.train.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.data.train.num_workers > 0),
        shuffle=cfg.data.train.shuffle,
    )

    eval_dl = DataLoader(
        eval_ds,
        batch_size=cfg.data.eval.batch_size,
        collate_fn=collate,
        num_workers=cfg.data.eval.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.data.eval.num_workers > 0),
        shuffle=cfg.data.eval.shuffle,
    )

    # Train or eval
    trainer = Trainer(cfg, logger)
    if cfg.eval.eval_only:
        trainer.eval_only(eval_dl, tok, xp, eval.per_sentence_stats)
    else:
        trainer.train(train_dl, eval_dl, tok, xp, eval.per_sentence_stats)
