import os, torch, shutil, matplotlib.pyplot as plt

from typing import Dict, Any
from logging import Logger
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from dora import get_xp, hydra_main, to_absolute_path, XP
from pathlib import Path

from luse.metrics import Counts
from luse.sentence import (
    build_sentence_encoder
)
from luse.log import (
    get_logger,
    dict_to_table,
)
from luse.utils import (
    recon_loss,
    sparsity_loss,
    configure_runtime,
    spectral_filter,
)
from luse.data import (
    initialize_data,
    PAD_TAG,
)
from luse.selector import RationaleSelectorModel


def compute_losses(
    gates: torch.Tensor,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    sent_encoder: Any,
    loss_cfg: Dict,
):
    full_rep = sent_encoder.encode(embeddings, attention_mask)
    pred_rep = sent_encoder.encode(embeddings * gates.unsqueeze(-1), attention_mask)
    comp_rep = sent_encoder.encode(embeddings * ((1 - gates) * attention_mask).unsqueeze(-1), attention_mask)
    
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
        
        self.checkpoint_path = cfg.train.checkpoint
        self.grad_clip = cfg.train.grad_clip
        self.loss_cfg = cfg.model.loss

        model_dim = torch.as_tensor(self.train_dl.dataset[0]["embeddings"]).shape[-1]  # model dim dynamically set without extra copy
        
        self.sent_encoder = build_sentence_encoder(
            cfg.model.encoder_family,
            hidden_dim=model_dim,
        ).to(self.device)

        self.sent_encoder.eval()
        for p in self.sent_encoder.parameters():
            p.requires_grad = False

        self.model = RationaleSelectorModel(model_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay)
        )
        
        self.selection_history = []
        self.label_history = []
        self.labels_present = "labels" in self.eval_dl.dataset[0] and self.eval_dl.dataset[0]["labels"] is not None

    def _to_device(self, batch: Dict):
        out = {}
        for k, v in batch.items():
            if k in {"embeddings", "attention_mask", "labels", "input_ids"} and v is not None:
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def save_final_plots(self):
        fig, axs = plt.subplots(2, 1, figsize=(16, 16))
        axs = axs.flatten()
        plots = [
            {
                "type": "simple",
                "data": self.selection_history,
                "title": "Selection Rate Across Epochs",
                "ylabel": "Selection Rate",
            },
            {
                "type": "dict",
                "data": self.label_history,
                "title": "Label Across Epochs",
                "ylabel": "Share",
            },
        ]
        for ax, plot_cfg in zip(axs, plots):
            ax.set_title(plot_cfg["title"])
            ax.set_xlabel("Epoch")
            ax.set_ylabel(plot_cfg["ylabel"])
            ax.set_ylim(0,1)
            
            if plot_cfg["type"] == "simple":
                ax.plot(plot_cfg["data"])
            elif plot_cfg["type"] == "dict":
                epochs = range(1, len(plot_cfg["data"]) + 1)
                labels = plot_cfg["data"][0].keys()
                for label in labels:
                    values = [epoch[label] for epoch in plot_cfg["data"]]
                    ax.plot(epochs, values, label=label)
                ax.legend()
                ax.legend(
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    fontsize="small",
                )

        fig.tight_layout()
        out = "summary_plots.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"Saved combined plot file: {out} for experiment {self.xp.sig}")

    def _save_checkpoint(self):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "meta": {"sig": self.xp.sig},
        }
        torch.save(state, self.checkpoint_path, _use_new_zipfile_serialization=False)
        if not (self.cfg.runtime.global_log or self.cfg.runtime.epoch_log):
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

        for batch in tqdm(self.eval_dl, desc="Eval: ", disable=not self.cfg.runtime.epoch_log):
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
            if not self.cfg.runtime.epoch_log:
                return
            self.logger.info("Eval selection rate: %.5f", selection_rate)
            return

        label_share = counts_part_pred / counts_part_gold
        self.label_history.append(label_share.data)
        if not self.cfg.runtime.epoch_log:
            return
        
        self.logger.info("Eval selection rate: %.5f", selection_rate)
        self.logger.info("Eval labels:\n%s", str(label_share))

    def train(self):
        if self.cfg.runtime.global_log:
            iterator = tqdm(range(self.cfg.train.epochs), desc="Training: ")
        else: 
            iterator = range(self.cfg.train.epochs)
            
        for epoch in iterator:
            totals = {k: 0.0 for k in ("total", "recon", "comp", "sparse")}
            example_count = 0

            iterator = tqdm(self.train_dl, desc=f"Training {epoch+1}: ", disable=not self.cfg.runtime.epoch_log)
            for batch in iterator:
                batch = self._to_device(batch)
                embeddings = batch["embeddings"]
                attention_mask = batch["attention_mask"]
                self.optimizer.zero_grad(set_to_none=True)
                
                gates = self.model(embeddings, attention_mask, hard=True)

                losses = compute_losses(gates,embeddings,attention_mask,self.sent_encoder,self.loss_cfg)

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
            if self.cfg.runtime.epoch_log:
                self.logger.info("Epoch %d/%d train:\n%s", epoch+1, self.cfg.train.epochs, dict_to_table(metrics))

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

    train_dl, eval_dl, tag_map = initialize_data(cfg.data, logger, device=device)
    trainer = SelectorTrainer(cfg, train_dl, eval_dl, logger, xp, tag_map)

    if cfg.train.eval_only:
        trainer._load_checkpoint()
        trainer.log_eval()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
