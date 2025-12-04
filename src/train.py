import copy
import logging
import os
from typing import Dict
from sentence_transformers import SentenceTransformer
from sklearn import base
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from dora import get_xp, hydra_main

from luse.log import (
    get_logger,
    should_disable_tqdm,
    make_loss_table,
    make_slot_table,
)
from luse.utils import (
    configure_runtime,
    prepare_batch,
    sbert_encode,
    metrics_from_counts,
    counts as count_masks,
)
from luse.data import initialize_dataloaders
from luse.selector import RationaleSelectorModel


class SelectorTrainer:
    def __init__(self, cfg, train_dl: DataLoader, eval_dl: DataLoader, logger, xp, device):
        self.cfg = cfg
        self.logger = logger
        self.train_dl = train_dl
        self.eval_dl = eval_dl
        self.device = device
        self.xp = xp

        self.disable_progress = should_disable_tqdm()
        self.checkpoint_path = cfg.train.checkpoint
        self.grad_clip = cfg.train.grad_clip
        
        base = SentenceTransformer(cfg.model.sbert_name)
        self.sbert_pooler = copy.deepcopy(base[1]).to(device).eval()
        for p in self.sbert_pooler.parameters():
            p.requires_grad = False


        d_model = torch.tensor(self.train_dl.dataset[0]["embeddings"]).shape[-1]

        self.model = RationaleSelectorModel(d_model).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay)
        )

    def _save_checkpoint(self):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "meta": {"sig": self.xp.sig},
        }
        torch.save(state, self.checkpoint_path, _use_new_zipfile_serialization=False)
        self.logger.info("Saved checkpoint to %s", os.path.abspath(self.checkpoint_path))

    def _load_checkpoint(self):
        state = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model"], strict=False)
        try:
            self.optimizer.load_state_dict(state["optimizer"])
        except Exception:
            pass
        self.logger.info("Loaded checkpoint from %s", self.checkpoint_path)
        
    def _run_batch(self, embeddings: torch.Tensor, attention_mask: torch.Tensor, sent_repr: torch.Tensor, train: bool) -> Dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        gates = self.model(sent_repr, embeddings, attention_mask)
        loss = recon_loss_simple(gates, sent_repr)
        if train:
            loss.backward()
            if self.grad_clip > 0.0:
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        return {k: float(v.detach()) for k, v in losses.items()}

    def _train_epoch(self, epoch_idx):
        total_loss = 0.0
        example_count = 0

        iterator = tqdm(self.train_dl, desc=f"Training {epoch_idx+1}: ", disable=self.disable_progress)
        for batch in iterator:
            embeddings, attention_mask, _, _ = prepare_batch(batch, self.device)
            sent_repr = sbert_encode(self.sbert_pooler, embeddings, attention_mask)

            loss, _ = self._run_batch(embeddings, attention_mask, sent_repr, train=True)
            batch_size = embeddings.size(0)
            example_count += batch_size
            total_loss += loss * batch_size

        return {"loss": total_loss / example_count}

    @torch.no_grad()
    def _evaluate(self):
        counts = {}
        example_count = 0

        iterator = tqdm(self.eval_dl, desc="Eval: ", disable=self.disable_progress)
        for batch in iterator:
            embeddings, attention_mask, _, labels = prepare_batch(batch, self.device)
            out = self.model(embeddings, attention_mask)
            gates = out["gates"]

            if gates.dim() == 2:
                gates = gates.unsqueeze(1)

            hard = torch.argmax(gates, dim=1)  # [B,T]
            gold = (labels > 0) & attention_mask.bool()

            B, K, T = gates.size()
            for k in range(K):
                pred = (hard == k) & attention_mask.bool()
                key = f"slot:{k}"
                tp, fp, fn, total = counts.get(key, (0, 0, 0, 0))
                d_tp, d_fp, d_fn = count_masks(pred, gold)
                counts[key] = (
                    tp + d_tp,
                    fp + d_fp,
                    fn + d_fn,
                    total + pred.sum().item(),
                )

        metrics = []
        for key, (tp, fp, fn, total) in counts.items():
            f1, p, r = metrics_from_counts(tp, fp, fn)
            metrics.append((key, f1, p, r, tp, fp, fn, total))

        metrics_sorted = sorted(metrics, key=lambda x: x[1], reverse=True)
        best_key, best_f1, best_p, best_r, _, _, _, _ = metrics_sorted[0]
        table_str = make_slot_table(metrics_sorted)

        return best_key, best_f1, best_p, best_r, table_str

    def train(self):
        epochs = self.cfg.train.epochs
        for epoch in range(epochs):
            metrics = self._train_epoch(epoch)
            table = make_loss_table(metrics, ["loss"])
            self.logger.info("Epoch %d/%d train:\n%s", epoch+1, epochs, table)

            if self.eval_dl is not None:
                best_key, best_f1, best_p, best_r, table_str = self._evaluate()
                if table_str:
                    self.logger.info("Eval:\n%s", table_str)
                    self.logger.info(
                        "Best slot=%s f1=%.4f precision=%.4f recall=%.4f",
                        best_key, best_f1, best_p, best_r
                    )

            self._save_checkpoint()

@hydra_main(config_path="conf", config_name="default", version_base="1.1")
def main(cfg):
    logger = get_logger()
    xp = get_xp()
    logger.info(f"Exp signature: {xp.sig}")
    logger.info(repr(cfg))
    logger.info(f"Work dir: {os.getcwd()}")

    configure_runtime(cfg)
    if cfg.runtime.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable, using CPU.")
    device = torch.device(cfg.runtime.device if torch.cuda.is_available() else "cpu")
    cfg.runtime.device = device.type

    train_dl, eval_dl, _ = initialize_dataloaders(
        cfg.data, logger, cfg.model.sbert_name, device=device)
    trainer = SelectorTrainer(cfg, train_dl, eval_dl, logger, xp, device)

    if cfg.train.eval_only:
        trainer._load_checkpoint()
        best_key, best_f1, best_p, best_r, table_str = trainer._evaluate()
        if table_str:
            logger.info("Eval-only slots:\n%s", table_str)
            logger.info(
                "Eval-only best slot=%s f1=%.4f precision=%.4f recall=%.4f",
                best_key,
                best_f1,
                best_p,
                best_r,
            )
    else:
        trainer.train()


if __name__ == "__main__":
    main()
