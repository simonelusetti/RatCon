import copy
import os
from typing import Dict

import torch
from sentence_transformers import SentenceTransformer
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from dora import get_xp, hydra_main

from luse.log import (
    get_logger,
    should_disable_tqdm,
    dict_to_table,
)
from luse.utils import (
    configure_runtime,
    prepare_batch,
    sbert_encode_texts,
    wp_to_text,
)
from luse.data import initialize_dataloaders
from luse.selector import RationaleSelectorModel


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def recon_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Reconstruction loss: maximize cosine similarity between prediction and target."""
    eps = 1e-8
    pred_norm = pred.norm(dim=-1).clamp_min(eps)
    target_norm = target.norm(dim=-1).clamp_min(eps)
    cos_sim = (pred * target).sum(dim=-1) / (pred_norm * target_norm)
    return 1.0 - cos_sim.mean()


def complement_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Repulsion term: penalise reconstruction from complement selections."""
    return -recon_loss(pred, target)


def sparsity_loss(gates: torch.Tensor, attention_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Encourage few tokens to be selected.
    gates: [B, L], attention_mask: [B, L]
    """
    valid = attention_mask.sum(dim=1).clamp_min(1.0)          # [B]
    mean_sel = (gates * attention_mask).sum(dim=1) / valid    # [B]
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
    tokens: list[list[str]],
    sbert: SentenceTransformer,
    cfg,
    device: torch.device,
):
    full_texts = [wp_to_text(tok_list) for tok_list in tokens]  # list[str]
    full_rep = sbert_encode_texts(sbert, full_texts, device).clone().detach()    # [B, D], no grad
    weighted = embeddings * gates.unsqueeze(-1)  # [B, L, D]

    denom = (gates * attention_mask).sum(dim=1, keepdim=True).clamp_min(1e-6)  # [B, 1]
    pred_rep = weighted.sum(dim=1) / denom                              # [B, D], requires_grad=True
    comp_weights = (1.0 - gates) * attention_mask               # [B, L]
    comp_weighted = embeddings * comp_weights.unsqueeze(-1)
    comp_denom = comp_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    comp_rep = comp_weighted.sum(dim=1) / comp_denom    # [B, D]

    recon_l = recon_loss(pred_rep, full_rep) * cfg.l_rec
    comp_l = complement_loss(comp_rep, full_rep) * cfg.l_comp
    sparse_l = sparsity_loss(gates, attention_mask) * cfg.l_s
    tv_l = total_variation_loss(gates, attention_mask) * cfg.l_tv

    total = recon_l + comp_l + sparse_l + tv_l

    return {
        "total": total,
        "recon": recon_l,
        "comp": comp_l,
        "sparse": sparse_l,
        "tv": tv_l,
    }


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
            tokens,
            self.sbert,
            self.cfg.model.loss,
            self.device,
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

    @torch.no_grad()
    def _evaluate(self):
        
        true_total = 0
        gold_total = 0
        pred_total = 0
        counts_pred = (0,0,0)
        counts_gold = (0,0,0)
        
        for batch in tqdm(self.eval_dl, desc="Eval: ", disable=self.disable_progress):
            embeddings, attention_mask, _, extra = prepare_batch(batch, self.device)
            gates = self.model(embeddings, attention_mask)
            
            tot_gold_thing, tot_gold_action, tot_gold_other = counts_gold
            gold_thing = (extra["factor_tags"] == 0) 
            gold_action = (extra["factor_tags"] == 1) 
            gold_other = (extra["factor_tags"] == 2)
            counts_gold = (
                tot_gold_thing + gold_thing.sum().item(),
                tot_gold_action + gold_action.sum().item(),
                tot_gold_other + gold_other.sum().item(),
            )
            true_total += attention_mask.sum().item()
            gold_total += gold_thing.sum().item() + gold_action.sum().item() + gold_other.sum().item()
            
            tot_pred_thing, tot_pred_action, tot_pred_other = counts_pred
            preditions = (gates > 0.5).long()
            pred_total += preditions.sum().item()
            pred_thing = gold_thing & preditions.bool()
            pred_action = gold_action & preditions.bool()
            pred_other = gold_other & preditions.bool()
            counts_pred = (
                tot_pred_thing + pred_thing.sum().item(),
                tot_pred_action + pred_action.sum().item(),
                tot_pred_other + pred_other.sum().item(),
            )
        
        tot_gold_thing, tot_gold_action, tot_gold_other = counts_gold
        tot_pred_thing, tot_pred_action, tot_pred_other = counts_pred
        thing_percentage = tot_pred_thing / max(tot_gold_thing, 1)
        action_percentage = tot_pred_action / max(tot_gold_action, 1)
        other_percentage = tot_pred_other / max(tot_gold_other, 1)
        
        metrics = {
            "thing_%": thing_percentage,
            "action_%": action_percentage,
            "other_%": other_percentage,
        }
        table_str = dict_to_table(metrics).get_string()
                
        return table_str

    def train(self):
        epochs = self.cfg.train.epochs
        for epoch in range(epochs):
            metrics = self._train_epoch(epoch)
            table = dict_to_table(metrics)
            self.logger.info("Epoch %d/%d train:\n%s", epoch+1, epochs, table)
            eval_table = self._evaluate()
            self.logger.info("Epoch %d/%d eval:\n%s", epoch+1, epochs, eval_table)
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
