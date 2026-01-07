import os, torch, shutil, matplotlib.pyplot as plt, json, textwrap

from typing import Dict
from logging import Logger
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from dora import get_xp, hydra_main, to_absolute_path, XP
from pathlib import Path

from luse.metrics import Counts
from luse.sentence import (
    build_sentence_encoder,
    SentenceEncoderBase,
)
from luse.log import (
    get_logger,
    dict_to_table,
)
from luse.utils import (
    configure_runtime,
    spectral_filter,
)
from luse.losses import (
    recon_loss,
    sparsity_loss,
    certainty_loss,
)
from luse.data import (
    initialize_data,
)
from luse.selector import RationaleSelectorModel


def compute_losses(
    gates: torch.Tensor,
    input: torch.Tensor,
    attn_mask: torch.Tensor,
    full_rep: torch.Tensor,
    sent_encoder: SentenceEncoderBase,
    loss_cfg: Dict,
):
    pred_rep = sent_encoder.encode(input, attn_mask, gates)
    
    if loss_cfg.get("spec_mode", "none") != "none":
        mode = loss_cfg.filter_mode
        cutoff = loss_cfg.cutoff
        pred_rep = spectral_filter(pred_rep, mode=mode, cutoff=cutoff)
        full_rep = spectral_filter(full_rep, mode=mode, cutoff=cutoff)
    
    recon_l = recon_loss(pred_rep, full_rep) * loss_cfg.l_rec
    sparse_l = sparsity_loss(gates, attn_mask, loss_cfg.sparsity_target) * loss_cfg.l_s
    certainty_l = certainty_loss(gates, attn_mask) * loss_cfg.l_c
    
    total = recon_l + sparse_l + certainty_l
    return {"total": total, "recon": recon_l, "sparse": sparse_l, "certainty": certainty_l}


class SelectorTrainer:
    def __init__(self, cfg: Dict, train_dl: DataLoader,
        test_dl: DataLoader, logger: Logger, xp: XP,
    ):
        self.cfg = cfg
        self.logger = logger
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.device = cfg.runtime.device
        self.xp = xp
        self.examples_to_log = cfg.train.eval.examples
        
        self.checkpoint_path = cfg.train.checkpoint
        self.grad_clip = cfg.train.grad_clip
        self.loss_cfg = cfg.model.loss

        model_dim = torch.as_tensor(self.train_dl.dataset[0]["tkns_embd"]).shape[-1]  # model dim dynamically set without extra copy
        
        self.sent_encoder, _ = build_sentence_encoder(
            cfg.model.encoder.family,
            encoder_name=cfg.model.encoder.name,
            device=self.device,
        )

        self.model = RationaleSelectorModel(model_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay)
        )
        
        self.total_tokens_history = []
        self.total_selected_history = []
        self.label_key = "scnd_labels" if cfg.model.scnd_labels else "labels"
        self.labels_present = (
            self.label_key in self.test_dl.dataset[0]
            and self.test_dl.dataset[0][self.label_key] is not None
        )
        if self.labels_present:
            self.counts_gold_history = []
            self.counts_pred_history = []

    def _to_device(self, batch: Dict):
        out = {}
        for k, v in batch.items():
            if k in {"tkns_embd", "attn_mask", "ids"} and v is not None:
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out
        
    def full_conf_matrix(self):
        txt = ""
        new_txt = ""
        for epoch, (gold, pred) in enumerate(zip(self.counts_gold_history, self.counts_pred_history)):
            new_txt = gold.conf_matrix(pred, epoch)
            if new_txt is None:
                return
            txt += new_txt
        
        out = "confusion_matrix.txt"
        with open(out, "w") as f:
            f.write(txt)
        self.logger.info("Last epoch confusion matrix:\n%s", new_txt)
        self.logger.info("Saved confusion matrix to %s", out)
        

    def save_final_plots(self):
        fig, axs = plt.subplots(2, 1, figsize=(16, 16))
        axs = axs.flatten()
        plots = [
            {
                "type": "simple",
                "data": [s / max(t, 1) for s, t in zip(self.total_selected_history, self.total_tokens_history)],
                "title": "Selection Rate Across Epochs",
                "ylabel": "Selection Rate",
            },
            {
                "type": "dict",
                "data": [
                    (pred / gold).data for pred, gold in \
                        zip(self.counts_pred_history, self.counts_gold_history)
                ],
                "title": "Label Across Epochs",
                "ylabel": "Share",
            },
        ]
        for ax, plot_cfg in zip(axs, plots):
            ax.set_title(plot_cfg["title"])
            ax.set_xlabel("Epoch")
            ax.set_ylabel(plot_cfg["ylabel"])
            
            if plot_cfg["type"] == "simple":
                ax.plot(plot_cfg["data"])
                values = plot_cfg["data"]
            elif plot_cfg["type"] == "dict" and self.labels_present:
                epochs = range(1, len(plot_cfg["data"]) + 1)
                labels = {k for d in plot_cfg["data"] for k in d}
                values = []
                for label in labels:
                    epoch_values = [epoch.get(label, 0) for epoch in plot_cfg["data"]]              
                    values.extend(epoch_values)     
                    ax.plot(epochs, epoch_values, label=label)
                ax.legend()
                ax.legend(
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    fontsize="small",
                )
            ax.set_ylim(min(values), max(values))
            
        argv_path = self.xp.folder / ".argv.json"
        if argv_path.exists():
            with open(argv_path) as f:
                args = json.load(f)

            argv_text = " | ".join(args)
            argv_text = textwrap.fill(argv_text, width=140)

            fig.suptitle(
                argv_text,
                fontsize=14,
                y=0.98,
                ha="center",
            )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        out = "summary_plots.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"Saved combined plot file: {out} for experiment {self.xp.sig}")
        self.full_conf_matrix()

    def save_checkpoint(self):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "meta": {"sig": self.xp.sig},
        }
        torch.save(state, self.checkpoint_path, _use_new_zipfile_serialization=False)
        if not (self.cfg.runtime.global_log or self.cfg.runtime.epoch_log):
            self.logger.info("Saved checkpoint to %s", os.path.abspath(self.checkpoint_path))

    def load_checkpoint(self):
        state = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model"], strict=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.logger.info("Loaded checkpoint from %s", self.checkpoint_path)

    @torch.no_grad()
    def evaluate(self, examples: int = 0):
        total_tokens, total_selected, examples_used = 0, 0, 0
        
        examples_str = []
        counts_pred = None
        counts_gold = None
        counts_init = False

        for batch in tqdm(self.test_dl, desc="Eval: ", disable=not self.cfg.runtime.epoch_log):
            batch = self._to_device(batch)
            embeddings = batch["tkns_embd"]
            attention_mask = batch["attn_mask"]
            labels = batch[self.label_key] if self.labels_present else None
            gates = self.model(embeddings, attention_mask, deterministic=True)
            
            if examples_used < examples:
                tokens = batch["tokens"]
                for i in range(embeddings.size(0)):
                    if examples_used >= examples:
                        break
                    example_gates = gates[i, : attention_mask[i].sum().item()]
                    example_tokens = tokens[i][: attention_mask[i].sum().item()]
                    selected_tokens = [
                        token for token, gate in zip(example_tokens, example_gates)
                        if gate.item() > 0.5
                    ]
                    examples_str.append((example_tokens, selected_tokens))
                    examples_used += 1

            total_tokens += attention_mask.sum().item()
            preds = (gates > 0.5).bool().view(-1)
            flat_attn = attention_mask.bool().view(-1)
            if self.labels_present:
                flat_labels = [x for sub in labels for x in sub]
                if not counts_init:
                    counts_pred = Counts(flat_labels, flat_attn, preds)
                    counts_gold = Counts(flat_labels, flat_attn)
                    counts_init = True
                else:
                    counts_pred = counts_pred + Counts(flat_labels, flat_attn, preds)
                    counts_gold = counts_gold + Counts(flat_labels, flat_attn)
                
            total_selected += (preds * flat_attn).sum().item()
        
        return total_selected, total_tokens, counts_gold, counts_pred, examples_str
    
    @torch.no_grad()
    def log_eval(self, examples: int = 0):
        total_selected, total_tokens, \
        counts_gold, counts_pred, examples_str = self.evaluate(examples=examples)
        
        self.total_tokens_history.append(total_tokens)
        self.total_selected_history.append(total_selected)
        
        if self.labels_present:
            self.counts_gold_history.append(counts_gold)
            self.counts_pred_history.append(counts_pred)
        
        if not self.cfg.runtime.epoch_log:
            return
        
        self.logger.info("Eval selection rate: %.5f", total_selected / max(total_tokens, 1))
        
        if examples > 0 and examples_str:
            for i, (tokens, selected_tokens) in enumerate(examples_str):
                self.logger.info("Eval Example %d:", i+1)
                self.logger.info("Tokens: %s", " ".join(tokens))
                self.logger.info("Selected Tokens: %s\n", " ".join(selected_tokens))
                
        if self.labels_present:
            self.logger.info("Eval labels:\n%s", str(counts_pred / counts_gold))
            new_txt = counts_gold.conf_matrix(counts_pred, epoch=None)
            if new_txt is None:
                return
            self.logger.info("Confusion Matrix:\n%s", new_txt)

    def train(self):
        if self.cfg.runtime.global_log:
            iterator = tqdm(range(self.cfg.train.epochs), desc="Training: ")
        else: 
            iterator = range(self.cfg.train.epochs)
            
        for epoch in iterator:
            totals = {k: 0.0 for k in ("total", "recon", "sparse", "certainty")}
            example_count = 0

            iterator = tqdm(self.train_dl, desc=f"Training {epoch+1}: ", disable=not self.cfg.runtime.epoch_log)
            for batch in iterator:
                batch = self._to_device(batch)
                tkns_embd = batch["tkns_embd"]
                attn_mask = batch["attn_mask"]
                ids = batch["ids"]
                sent_embd = batch["sent_embd"]
                self.optimizer.zero_grad(set_to_none=True)
                
                gates = self.model(tkns_embd, attn_mask)

                losses = compute_losses(gates, ids, attn_mask, sent_embd, self.sent_encoder, self.loss_cfg)

                loss_total = losses["total"]
                loss_total.backward()
                
                if self.grad_clip > 0.0:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                batch_size = tkns_embd.size(0)
                example_count += batch_size
                for k, v in losses.items():
                    totals[k] = totals.get(k, 0.0) + float(v.detach()) * batch_size
                    
            metrics = {k: v / example_count for k, v in totals.items()}
            if self.cfg.runtime.epoch_log:
                self.logger.info("Epoch %d/%d train:\n%s", epoch+1, self.cfg.train.epochs, dict_to_table(metrics))

            self.log_eval(self.examples_to_log)
            self.save_checkpoint()

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

    train_dl, test_dl = initialize_data(cfg.data, logger, device=device)
    trainer = SelectorTrainer(cfg, train_dl, test_dl, logger, xp)

    if cfg.train.eval.eval_only:
        trainer.load_checkpoint()
        trainer.log_eval()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
