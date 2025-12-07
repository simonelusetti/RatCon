import os
from typing import Dict

import torch
from sentence_transformers import SentenceTransformer
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from dora import get_xp, hydra_main
import torch.nn.functional as F

from luse.log import (
    get_logger,
    should_disable_tqdm,
    dict_to_table,
)
from luse.utils import (
    configure_runtime,
    prepare_batch,
    sbert_encode,
    format_dict,
)
from luse.data import initialize_dataloaders, CATH_TO_ID, PART_TO_ID
from luse.selector import RationaleSelectorModel


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def recon_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Reconstruction loss: maximize cosine similarity between prediction and target."""
    cos_sim = F.cosine_similarity(pred, target, dim=-1)  # returns [B]
    return 1.0 - cos_sim.mean()


def sparsity_loss(gates: torch.Tensor, attention_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Encourage few tokens to be selected.
    gates: [B, L], attention_mask: [B, L]
    """
    valid = attention_mask.sum(dim=1).clamp_min(1.0)          # [B]
    mean_sel = gates.sum(dim=1) / valid    # [B]
    return mean_sel.mean() + eps


def total_variation_loss(gates: torch.Tensor, attention_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Encourage contiguous selections by penalising rapid changes."""
    g_cur = gates[:, 1:]
    g_prev = gates[:, :-1]
    m_cur = attention_mask[:, 1:]
    m_prev = attention_mask[:, :-1]
    overlap = m_cur * m_prev
    valid_pairs = overlap.sum(dim=1).clamp_min(1.0)
    smoothness = (torch.abs(g_cur - g_prev) * overlap).sum(dim=1) / valid_pairs
    contiguity = ((g_cur * g_prev) * overlap).sum(dim=1) / valid_pairs
    return contiguity.mean() + eps * smoothness.mean()


def compute_losses(
    gates: torch.Tensor,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    sbert: SentenceTransformer,
    cfg,
):
    full_rep = sbert_encode(sbert, embeddings, attention_mask).detach()
    pred_rep = sbert_encode(
        sbert, embeddings * gates.unsqueeze(-1), attention_mask
    )
    comp_rep = sbert_encode(
        sbert, embeddings * ((1 - gates) * attention_mask).unsqueeze(-1), attention_mask
    )
    recon_l = recon_loss(pred_rep, full_rep) * cfg.l_rec
    empty_rep = torch.zeros_like(full_rep)
    comp_l = recon_loss(comp_rep, empty_rep) * cfg.l_comp
    sparse_l = sparsity_loss(gates, attention_mask) * cfg.l_s
    tv_l = total_variation_loss(gates, attention_mask) * cfg.l_tv

    total = recon_l + comp_l + sparse_l + tv_l
    return {"total": total, "recon": recon_l, "comp": comp_l,
            "sparse": sparse_l, "tv": tv_l}


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
        self.tau = float(cfg.model.loss.tau)

        self.sbert = SentenceTransformer(cfg.model.sbert_name)
        for p in self.sbert.parameters():
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
        self.optimizer.load_state_dict(state["optimizer"])
        self.logger.info("Loaded checkpoint from %s", self.checkpoint_path)
        
    def _run_batch(self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        tokens: list[list[str]],
        train: bool
    ) -> Dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        gates = self.model(embeddings, attention_mask, hard=not train)
        losses = compute_losses(
            gates,
            embeddings,
            attention_mask,
            self.sbert,
            self.cfg.model.loss,
        )
        loss_total = losses["total"]
        if train:
            loss_total.backward()
            if self.grad_clip > 0.0:
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        return {k: float(v.detach()) for k, v in losses.items()}

    def _train_epoch(self, epoch_idx):
        totals = {"total": 0.0, "recon": 0.0, "comp": 0.0, "sparse": 0.0, "tv": 0.0}
        example_count = 0

        iterator = tqdm(self.train_dl, desc=f"Training {epoch_idx+1}: ", disable=self.disable_progress)
        for batch in iterator:
            embeddings, attention_mask, tokens, _ = prepare_batch(batch, self.device)
            losses = self._run_batch(embeddings, attention_mask, tokens, train=True)
            batch_size = embeddings.size(0)
            example_count += batch_size
            for k in totals:
                totals[k] += losses[k] * batch_size

        return {k: v / example_count for k, v in totals.items()}

    def _update_counts(
        self,
        gates: torch.Tensor,
        attention_mask: torch.Tensor,
        extra: Dict,
        counts_cath_pred: Dict,
        counts_cath_gold: Dict,
        counts_part_pred: Dict,
        counts_part_gold: Dict,
        total_selected: int,
    ):

        # flatten
        attn = attention_mask.bool().view(-1)
        preds = (gates > 0.5).bool().view(-1)

        # integer tensors, shape (B, L)
        cath_tags = extra["cath_tags"].to(self.device)
        part_tags = extra["part_tags"].to(self.device)

        flat_cath = cath_tags.view(-1)
        flat_part = part_tags.view(-1)

        PAD_CATH = CATH_TO_ID["pad"]
        PAD_PART = PART_TO_ID["pad"]

        # initialize dicts for all classes
        for cid in CATH_TO_ID.values():
            counts_cath_gold.setdefault(cid, 0)
            counts_cath_pred.setdefault(cid, 0)

        for pid in PART_TO_ID.values():
            counts_part_gold.setdefault(pid, 0)
            counts_part_pred.setdefault(pid, 0)

        # --------------------------
        # UPDATE CATH COUNTS
        # --------------------------
        for cid in CATH_TO_ID.values():
            if cid == PAD_CATH:
                continue  # don't evaluate PAD class

            gold_mask = (flat_cath == cid) & attn
            pred_mask = gold_mask & preds

            counts_cath_gold[cid] += gold_mask.sum().item()
            counts_cath_pred[cid] += pred_mask.sum().item()

        # --------------------------
        # UPDATE PART COUNTS
        # --------------------------
        for pid in PART_TO_ID.values():
            if pid == PAD_PART:
                continue  # skip PAD

            gold_mask = (flat_part == pid) & attn
            pred_mask = gold_mask & preds

            counts_part_gold[pid] += gold_mask.sum().item()
            counts_part_pred[pid] += pred_mask.sum().item()

        total_selected += (preds & attn).sum().item()

        return (
            total_selected,
            counts_cath_pred,
            counts_cath_gold,
            counts_part_pred,
            counts_part_gold,
        )
        
    def _rates(
        self,
        counts_cath_pred: Dict,
        counts_cath_gold: Dict,
        counts_part_pred: Dict,
        counts_part_gold: Dict,
    ):
        rev_cath = {v: k for k, v in CATH_TO_ID.items()}

        caths_rates = {}
        for cid, name in rev_cath.items():
            if cid == CATH_TO_ID["pad"]:
                continue
            gold = counts_cath_gold.get(cid, 0)
            pred = counts_cath_pred.get(cid, 0)
            if gold == 0: gold = 1
            caths_rates[name] = pred / gold

        rev_part = {v: k for k, v in PART_TO_ID.items()}

        part_rates = {}
        for pid, name in rev_part.items():
            if pid == PART_TO_ID["pad"]:
                continue
            gold = counts_part_gold.get(pid, 0)
            pred = counts_part_pred.get(pid, 0)
            if gold == 0: gold = 1
            part_rates[name] = pred / gold
            
        return caths_rates, part_rates
    
    def _preferences(
        self,
        counts_cath_pred: Dict,
        counts_part_pred: Dict,
        total_selected: int,
        ):
        if total_selected == 0:
            return {}, {}
        rev_cath = {v: k for k, v in CATH_TO_ID.items()}
        cath_prefs = {}
        for cid, name in rev_cath.items():
            if cid == CATH_TO_ID["pad"]:
                continue
            pred = counts_cath_pred.get(cid, 0)
            cath_prefs[name] = pred / total_selected
        rev_part = {v: k for k, v in PART_TO_ID.items()}
        part_prefs = {}
        for pid, name in rev_part.items():
            if pid == PART_TO_ID["pad"]:
                continue
            pred = counts_part_pred.get(pid, 0)
            part_prefs[name] = pred / total_selected
        return cath_prefs, part_prefs

    @torch.no_grad()
    def _evaluate(self):

        total_tokens = 0
        total_selected = 0
        counts_cath_pred = {}
        counts_cath_gold = {}
        counts_part_pred = {}
        counts_part_gold = {}

        for batch in tqdm(self.eval_dl, desc="Eval: ", disable=self.disable_progress):
            embeddings, attention_mask, _, extra = prepare_batch(batch, self.device)
            gates = self.model(embeddings, attention_mask)

            total_tokens += attention_mask.sum().item()

            (
                total_selected,
                counts_cath_pred,
                counts_cath_gold,
                counts_part_pred,
                counts_part_gold,
            ) = self._update_counts(
                gates,
                attention_mask,
                extra,
                counts_cath_pred,
                counts_cath_gold,
                counts_part_pred,
                counts_part_gold,
                total_selected,
            )

        selection_rate = total_selected / max(total_tokens, 1)
        
        caths_rates, part_rates = self._rates(
            counts_cath_pred,
            counts_cath_gold,
            counts_part_pred,
            counts_part_gold,
        )
        cath_prefs, part_prefs = self._preferences(
            counts_cath_pred,
            counts_part_pred,
            total_selected,
        )
        
        cath_rates = {
            **{f"{k}_%": v for k, v in caths_rates.items()},
        }
        part_rates = {
            **{f"{k}_%": v for k, v in part_rates.items()},
        }
        cath_prefs = {
            **{f"{k}_f": v for k, v in cath_prefs.items()},
        }
        part_prefs = {  
            **{f"{k}_f": v for k, v in part_prefs.items()},
        }

        cath_rates =  dict(sorted(cath_rates.items(), key=lambda x: x[1], reverse=True))
        cath_prefs = dict(sorted(cath_prefs.items(), key=lambda x: x[1], reverse=True))
        
        part_rates = dict(sorted(part_rates.items(), key=lambda x: x[1], reverse=True))
        part_prefs = dict(sorted(part_prefs.items(), key=lambda x: x[1], reverse=True))
        
        return selection_rate, cath_rates, part_rates, cath_prefs, part_prefs


    def train(self):
        epochs = self.cfg.train.epochs
        for epoch in range(epochs):
            metrics = self._train_epoch(epoch)
            table = dict_to_table(metrics)
            self.logger.info("Epoch %d/%d train:\n%s", epoch+1, epochs, table)
            selection_rate, cath_rates, part_rates, cath_prefs, part_prefs = self._evaluate()
            self.logger.info("Epoch %d/%d selection rate: %.5f", epoch+1, epochs, selection_rate)
            self.logger.info("Epoch %d/%d eval:\n%s", epoch+1, epochs, format_dict(cath_rates))
            self.logger.info("Epoch %d/%d eval prefs:\n%s", epoch+1, epochs, format_dict(cath_prefs))
            self.logger.info("Epoch %d/%d eval:\n%s", epoch+1, epochs, format_dict(part_rates))
            self.logger.info("Epoch %d/%d eval prefs:\n%s", epoch+1, epochs, format_dict(part_prefs))
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
        cath_table_str, part_table_str, cath_prefs_table_str, part_prefs_table_str = trainer._evaluate()
        logger.info("Eval-only cath:\n%s", cath_table_str)
        logger.info("Eval-only part:\n%s", part_table_str)
        logger.info("Eval-only cath prefs:\n%s", cath_prefs_table_str)
        logger.info("Eval-only part prefs:\n%s", part_prefs_table_str)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
