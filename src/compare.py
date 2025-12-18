import os, torch, torch.nn.functional as F, shutil

from typing import Dict
from logging import Logger
from sentence_transformers import SentenceTransformer
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from dora import get_xp, hydra_main, to_absolute_path, XP
import matplotlib.pyplot as plt
from pathlib import Path

from luse.log import (
    get_logger,
    should_disable_tqdm,
    dict_to_table,
)
from luse.utils import (
    configure_runtime,
    sbert_encode,
    spectral_filter,
    Counts
)
from luse.data import (
    initialize_data,
    PAD_TAG,
)
from luse.selector import RationaleSelectorModel


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def recon_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Reconstruction loss: maximize cosine similarity between prediction and target."""
    cos_sim = F.cosine_similarity(pred, target, dim=-1)
    return 1.0 - cos_sim.mean()


def sparsity_loss(gates: torch.Tensor, attention_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Encourage few tokens to be selected.
    gates: [B, L], attention_mask: [B, L]
    """
    valid = attention_mask.sum(dim=1).clamp_min(1.0)
    mean_sel = gates.sum(dim=1) / valid   
    return mean_sel.mean() + eps


def compute_losses(
    gates: torch.Tensor,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    full_rep: torch.Tensor,
    sbert: SentenceTransformer,
    loss_cfg: Dict,
):
    pred_rep = sbert_encode(
        sbert, embeddings * gates.unsqueeze(-1), attention_mask
    )
    comp_rep = sbert_encode(
        sbert, embeddings * ((1 - gates) * attention_mask).unsqueeze(-1), attention_mask
    )
    
    if loss_cfg.get("spec_mode", "none") != "none":
        mode = loss_cfg.filter_mode
        cutoff = loss_cfg.cutoff
        pred_rep = spectral_filter(pred_rep, mode=mode, cutoff=cutoff)
        comp_rep = spectral_filter(comp_rep, mode=mode, cutoff=cutoff)
        full_rep = spectral_filter(full_rep, mode=mode, cutoff=cutoff)
    
    recon_l = recon_loss(pred_rep, full_rep) * loss_cfg.l_rec
    empty_rep = torch.zeros_like(full_rep)
    comp_l = recon_loss(comp_rep, empty_rep) * loss_cfg.l_comp
    sparse_l = sparsity_loss(gates, attention_mask) * loss_cfg.l_s

    total = recon_l + comp_l + sparse_l
    return {"total": total, "recon": recon_l, "comp": comp_l, "sparse": sparse_l}


class SelectorTrainer:
    def __init__(self, cfg: Dict, train_dl: DataLoader,
        eval_dl: DataLoader, logger: Logger, xp: XP, tag_map = None,
    ):
        self.cfg = cfg
        self.logger = logger
        self.train_dl = train_dl
        self.eval_dl = eval_dl
        self.device = cfg.runtime.device
        self.xp = xp
        self.tag_map = tag_map
        
        self.disable_progress = should_disable_tqdm(cfg.runtime)
        self.checkpoint_path = cfg.train.checkpoint
        self.grad_clip = cfg.train.grad_clip
        self.loss_cfg = cfg.model.loss

        self.sbert = SentenceTransformer(cfg.model.sbert_name).to(self.device)
        self.sbert.eval()
        for p in self.sbert.parameters():
            p.requires_grad = False

        first_emb = self.train_dl.dataset[0]["embeddings"]
        model_dim = torch.as_tensor(first_emb).shape[-1]
        self.model = RationaleSelectorModel(model_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay)
        )
        
        self.plot_dir = to_absolute_path("outputs/plots")
        self.plot_dir = Path(os.path.join(self.plot_dir, xp.sig))

        self.selection_history = []
        self.part_history = []
        self.part_rates_history = []
        self.part_share_history = []
        self.labels_present = "labels" in self.eval_dl.dataset[0] and self.eval_dl.dataset[0]["labels"] is not None

    def _to_device(self, batch: Dict):
        out = {}
        for k, v in batch.items():
            if k in {"embeddings", "attention_mask", "sentence_reps", "labels", "input_ids"} and v is not None:
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def save_final_plots(self):
        def collect_series(history):
            """Convert list[dict] → {key: [v1, v2, ...]}"""
            out = {}
            for d in history:
                for k, v in d.items():
                    out.setdefault(k, []).append(v)
            return out

        def plot_series(ax, series_dict, title, ylabel, top_k=5):
            """Plot top-K time series on a single axis."""
            # Select top-K by mean value
            means = {
                k: sum(v) / len(v)
                for k, v in series_dict.items()
                if len(v) > 0
            }
            top_keys = sorted(means, key=means.get, reverse=True)[:top_k]

            all_vals = []

            for k in top_keys:
                series = series_dict[k]
                ax.plot(series, label=str(k))  # no markers
                all_vals.extend(series)

            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)

            # Dynamic y-limits for visibility
            if all_vals:
                ymin = min(all_vals)
                ymax = max(all_vals)
                pad = (ymax - ymin) * 0.05 if ymax > ymin else 0.01
                ax.set_ylim(ymin - pad, ymax + pad)
                #ax.set_ylim(0,2)

            ax.legend(
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize="small",
            )

        if self.labels_present:
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
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
                    "history": self.part_share_history,
                    "title": "Part Share Across Epochs",
                    "ylabel": "Share",
                },
            ]

            for ax, cfg in zip(axs, plots):
                if cfg["type"] == "simple":
                    ax.plot(cfg["data"])
                    ax.set_title(cfg["title"])
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel(cfg["ylabel"])

                    if cfg["data"]:
                        ymin = min(cfg["data"])
                        ymax = max(cfg["data"])
                        pad = (ymax - ymin) * 0.05 if ymax > ymin else 0.01
                        ax.set_ylim(ymin - pad, ymax + pad)
                        #ax.set_ylim(0,1)

                elif cfg["type"] == "dict_history":
                    series = collect_series(cfg["history"])
                    plot_series(ax, series, cfg["title"], cfg["ylabel"])

        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(self.selection_history)
            ax.set_title("Selection Rate Across Epochs")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Selection Rate")

        fig.tight_layout()
        out = self.plot_dir / "summary_plots.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"Saved combined plot file: {out}")

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

    @torch.no_grad()
    def evaluate(self):
        total_tokens = 0
        total_selected = 0
        if self.labels_present:
            counts_part_pred = Counts(self.tag_map, PAD_TAG)
            counts_part_gold = Counts(self.tag_map, PAD_TAG)

        for batch in tqdm(self.eval_dl, desc="Eval: ", disable=self.disable_progress):
            batch = self._to_device(batch)
            embeddings = batch["embeddings"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"] if self.labels_present else None
            gates = self.model(embeddings, attention_mask)

            total_tokens += attention_mask.sum().item()
            preds = (gates > 0.5).bool().view(-1)
            if self.labels_present:
                flat_attn = attention_mask.bool().view(-1)
                flat_labels = labels.view(-1)
                counts_part_pred = counts_part_pred + \
                    Counts(self.tag_map, PAD_TAG, classes_mask=flat_labels, mask=flat_attn & preds)
                counts_part_gold = counts_part_gold + \
                    Counts(self.tag_map, PAD_TAG, classes_mask=flat_labels, mask=flat_attn)
                
            total_selected += preds.sum().item()
        
        if not self.labels_present:
            return total_selected, total_tokens, None, None
        
        return total_selected, total_tokens, counts_part_gold, counts_part_pred

    def log_eval(self):
        total_selected, total_tokens, \
        counts_part_gold, counts_part_pred = self.evaluate()
        
        selection_rate = total_selected / max(total_tokens, 1)
        self.selection_history.append(selection_rate)
        
        if not self.labels_present:
            if self.disable_progress:
                return
            self.logger.info("Eval selection rate: %.5f", selection_rate)
            return

        part_rates = counts_part_pred / counts_part_gold
        part_pref = counts_part_pred.preferences()
        part_share = counts_part_pred.preferences_over_total(total_selected) / counts_part_gold.preferences_over_total(total_tokens)
                     
        self.part_history.append(part_pref.data.copy())
        self.part_rates_history.append(part_rates.data.copy())
        self.part_share_history.append(part_share.data.copy())
                     
        if self.disable_progress:
            return
        
        self.logger.info("Eval selection rate: %.5f", selection_rate)
        self.logger.info("Eval part:\n%s", str(part_rates))
        self.logger.info("Eval part prefs:\n%s", str(part_pref))
        self.logger.info("Eval part share:\n%s", str(part_share))

    def train(self):
        epochs = self.cfg.train.epochs
        for epoch in range(epochs):
            totals = {k: 0.0 for k in ("total", "recon", "comp", "sparse")}
            example_count = 0

            iterator = tqdm(self.train_dl, desc=f"Training {epoch+1}: ", disable=self.disable_progress)
            for batch in iterator:
                batch = self._to_device(batch)
                embeddings = batch["embeddings"]
                attention_mask = batch["attention_mask"]
                sentence_reps = batch["sentence_reps"]
                self.optimizer.zero_grad(set_to_none=True)
                
                gates = self.model(embeddings, attention_mask, hard=True)

                losses = compute_losses(gates,embeddings,attention_mask, \
                    sentence_reps,self.sbert,self.loss_cfg)

                loss_total = losses["total"]
                loss_total.backward()
                
                if self.grad_clip > 0.0:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                batch_size = embeddings.size(0)
                example_count += batch_size
                for k, v in losses.items():
                    totals[k] = totals.get(k, 0.0) + float(v.detach()) * batch_size

            metrics = {k: v / example_count for k, v in totals.items() if v != 0.0}
            if not self.disable_progress:
                self.logger.info("Epoch %d/%d train:\n%s", epoch+1, epochs, dict_to_table(metrics))

            self.log_eval()
            self._save_checkpoint()

        self.save_final_plots()


@hydra_main(config_path="conf", config_name="double", version_base="1.1")
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

    train_dl1, eval_dl1, tag_map1 = initialize_data(cfg.data1, logger, device=device)
    train_dl2, eval_dl2, tag_map2 = initialize_data(cfg.data2, logger, device=device)
    
    trainer1 = SelectorTrainer(cfg, train_dl1, eval_dl1, logger, xp, tag_map1)
    trainer2 = SelectorTrainer(cfg, train_dl2, eval_dl2, logger, xp, tag_map2)
    
    

if __name__ == "__main__":
    main()
