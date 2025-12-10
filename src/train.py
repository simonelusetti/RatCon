import os, torch, torch.nn.functional as F, shutil

from typing import Dict
from sentence_transformers import SentenceTransformer
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from dora import get_xp, hydra_main, to_absolute_path
import matplotlib.pyplot as plt
from pathlib import Path


from luse.log import (
    get_logger,
    should_disable_tqdm,
    dict_to_table,
)
from luse.utils import (
    configure_runtime,
    prepare_batch,
    sbert_encode,
    Counts,
    new_counts,
)
from luse.data import (
    initialize_dataloaders,
    CATH_TO_ID,
    PART_TO_ID_BY_SYSTEM,
    canonical_name,
    ID_TO_CATH,
    ID_TO_PART_BY_SYSTEM,
)
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

        self.disable_progress = should_disable_tqdm(cfg.runtime)
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
        
        self.plot_dir = to_absolute_path("outputs/plots")
        self.plot_dir = Path(os.path.join(self.plot_dir, xp.sig))

        self.selection_history = []
        self.cath_history = []
        self.cath_rates_history = []
        self.cath_share_history = []
        self.part_history = []
        self.part_rates_history = []
        self.part_share_history = []
        self.plot_step = 0

    def save_final_plots(self):
        def collect_series(history):
            """Convert list[dict] → {key: [v1, v2, ...]}"""
            out = {}
            for d in history:
                for k, v in d.items():
                    out.setdefault(k, []).append(v)
            return out

        def plot_series(ax, series_dict, title, ylabel, label_map=None):
            """Plot multiple timeseries on a single axis."""
            for k, series in series_dict.items():
                label = label_map[k] if label_map else str(k)
                ax.plot(series, marker="o", label=label)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.2)
            ax.legend()

        fig, axs = plt.subplots(3, 2, figsize=(12, 12))
        axs = axs.flatten()
        plots = [
            {
                "type": "simple",
                "data": self.selection_history,
                "title": "Selection Rate Across Epochs",
                "ylabel": "Selection Rate",
            },
            {
                "type": "dict_history",
                "history": self.cath_history,
                "title": "Category Preferences Across Epochs",
                "ylabel": "Preference Value",
                "label_map": ID_TO_CATH,
            },
            {
                "type": "dict_history",
                "history": self.cath_rates_history,
                "title": "Cath Rates Across Epochs",
                "ylabel": "Rate",
                "label_map": ID_TO_CATH,
            },
            {
                "type": "dict_history",
                "history": self.part_history,
                "title": "Part Preferences Across Epochs",
                "ylabel": "Preference Value",
            },
            {
                "type": "dict_history",
                "history": self.part_rates_history,
                "title": "Part Rates Across Epochs",
                "ylabel": "Rate",
            },
            {
                "type": "dict_history",
                "history": self.cath_share_history,
                "title": "Cath Share Across Epochs",
                "ylabel": "Share",
                "label_map": ID_TO_CATH,
            },
        ]
        for ax, cfg in zip(axs, plots):
            if cfg["type"] == "simple":
                ax.plot(cfg["data"], marker="o")
                ax.set_title(cfg["title"])
                ax.set_xlabel("Epoch")
                ax.set_ylabel(cfg["ylabel"])
                ax.set_ylim(0, 1)
            elif cfg["type"] == "dict_history":
                series = collect_series(cfg["history"])
                plot_series(ax, series, cfg["title"], cfg["ylabel"], cfg.get("label_map"))

        fig.tight_layout()
        out_all = self.plot_dir / "summary_plots.png"
        fig.savefig(out_all)
        plt.close(fig)
        self.logger.info(f"Saved combined plot file: {out_all}")

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
            embeddings, attention_mask, _, _ = prepare_batch(batch, self.device)
            losses = self._run_batch(embeddings, attention_mask, train=True)
            batch_size = embeddings.size(0)
            example_count += batch_size
            for k in totals:
                totals[k] += losses[k] * batch_size

        return {k: v / example_count for k, v in totals.items()}

    @torch.no_grad()
    def _evaluate(self):
        pos_system = canonical_name(self.cfg.data.eval.dataset)
        local_part_to_id = PART_TO_ID_BY_SYSTEM.get(pos_system, None)

        total_tokens = 0
        total_selected = 0
        extras_present = self.eval_dl[0]["part_tags"] is not None
        if extras_present:
            counts_cath_pred = Counts(CATH_TO_ID.values(), CATH_TO_ID["pad"], classes_str=ID_TO_CATH)
            counts_cath_gold = Counts(CATH_TO_ID.values(), CATH_TO_ID["pad"], classes_str=ID_TO_CATH)
            counts_part_pred = Counts(local_part_to_id.values(), local_part_to_id["pad"], classes_str=ID_TO_PART_BY_SYSTEM[pos_system])
            counts_part_gold = Counts(local_part_to_id.values(), local_part_to_id["pad"], classes_str=ID_TO_PART_BY_SYSTEM[pos_system])

        for batch in tqdm(self.eval_dl, desc="Eval: ", disable=self.disable_progress):
            embeddings, attention_mask, _, extra = prepare_batch(batch, self.device)
            gates = self.model(embeddings, attention_mask)

            total_tokens += attention_mask.sum().item()
            if extras_present:
                new_pred_cath, new_gold_cath, new_pred_part, new_gold_part, new_sel \
                = new_counts(
                    self.cfg.data.eval.dataset,
                    gates,
                    attention_mask,
                    extra,
                    local_part_to_id,
                    self.device,
                    cath_str=ID_TO_CATH,
                    part_str=ID_TO_PART_BY_SYSTEM.get(pos_system, None),
                )
                counts_cath_pred += new_pred_cath
                counts_cath_gold += new_gold_cath
                counts_part_pred += new_pred_part
                counts_part_gold += new_gold_part
            total_selected += new_sel
        
        if not extras_present:
            return total_selected, total_tokens, None, None, None, None
        
        return total_selected, total_tokens, counts_cath_gold, counts_cath_pred, counts_part_gold, counts_part_pred

    def log_eval(self):
        total_selected, total_tokens, \
        counts_cath_gold, counts_cath_pred, \
        counts_part_gold, counts_part_pred = self._evaluate()
        
        selection_rate = total_selected / max(total_tokens, 1)
        self.selection_history.append(selection_rate)
        if counts_cath_pred is None:
            return
        
        cath_rates = counts_cath_pred / counts_cath_gold
        part_rates = counts_part_pred / counts_part_gold
        
        cath_pref = counts_cath_pred.preferences()
        part_pref = counts_part_pred.preferences()
        
        cath_share = counts_cath_pred.preferences_over_total(total_selected) / counts_cath_gold.preferences_over_total(total_tokens)
        part_share = counts_part_pred.preferences_over_total(total_selected) / counts_part_gold.preferences_over_total(total_tokens)
                     
        self.cath_history.append(cath_pref.data.copy())
        self.cath_rates_history.append(cath_rates.data.copy())
        self.cath_share_history.append(cath_share.data.copy())
        self.part_history.append(part_pref.top_k(5).copy())
        self.part_rates_history.append(part_rates.top_k(5).copy())
        self.part_share_history.append(part_share.top_k(5).copy())
                     
        if self.disable_progress:
            return
        
        self.logger.info("Eval selection rate: %.5f", selection_rate)
        self.logger.info("Eval cath:\n%s", str(cath_rates))
        self.logger.info("Eval cath prefs:\n%s", str(cath_pref))
        self.logger.info("Eval part:\n%s", str(part_rates))
        self.logger.info("Eval part prefs:\n%s", str(part_pref))
        self.logger.info("Eval cath share:\n%s", str(cath_share))
        self.logger.info("Eval part share:\n%s", str(part_share))

    def train(self):
        epochs = self.cfg.train.epochs
        for epoch in range(epochs):
            metrics = self._train_epoch(epoch)
            table = dict_to_table(metrics)
            if not self.disable_progress:
                self.logger.info("Epoch %d/%d train:\n%s", epoch+1, epochs, table)
            self.log_eval()
            self._save_checkpoint()
        self.save_final_plots()


@hydra_main(config_path="conf", config_name="default", version_base="1.1")
def main(cfg):
    logger = get_logger()
    xp = get_xp()
    logger.info(f"Exp signature: {xp.sig}")
    logger.info(repr(cfg))
    logger.info(f"Work dir: {os.getcwd()}")
    
    src = xp.folder / ".argv.json"
    plot_dir = to_absolute_path("outputs/plots")
    plot_dir = Path(os.path.join(plot_dir, xp.sig))
    plot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, f"{plot_dir}/.argv.json")

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
        trainer.log_eval()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
