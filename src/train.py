import os, torch, shutil, json, math

from typing import Dict
from logging import Logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from dora import get_xp, hydra_main, to_absolute_path, XP
from pathlib import Path
from transformers import AutoTokenizer

from luse.metrics import Counts
from luse.sentence import SentenceEncoder
from luse.selector import RationaleSelectorModel
from luse.data import initialize_data
from luse.log import get_logger, dict_to_table
from luse.utils import (
    configure_runtime,
    open_selection_writer,
    to_device,
    save_final_plots,
)
from luse.losses import (
    contrastive_loss_sym,
    sparsity_loss,
    entropy_loss,
)


def compute_losses(
    g: torch.Tensor,      
    z: torch.Tensor,     
    ids: torch.Tensor,
    attn_mask: torch.Tensor,
    sent_encoder,
    loss_cfg,
):
    pred_rep = sent_encoder.encode(ids, attn_mask, gates=g)
    full_rep = sent_encoder.encode(ids, attn_mask)

    recon_l = contrastive_loss_sym(pred_rep, full_rep, loss_cfg.tau) * loss_cfg.l_rec
    sparse_l = sparsity_loss(z, loss_cfg.sparsity_target) * loss_cfg.l_s
    ent_l = entropy_loss(z) * loss_cfg.l_e

    total = recon_l + sparse_l + ent_l

    return {
        "total": total,
        "recon": recon_l,
        "sparse": sparse_l,
        "entropy": ent_l,
    }, pred_rep


class SelectorTrainer:
    def __init__(
        self,
        cfg: Dict,
        train_dl: DataLoader,
        test_dl: DataLoader,
        sent_encoder: SentenceEncoder,
        tokenizer: AutoTokenizer,
        logger: Logger,
        xp: XP,
    ):
        self.cfg = cfg
        self.logger = logger
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.device = cfg.runtime.device
        self.xp = xp
        self.examples_to_log = cfg.train.eval.examples

        self.checkpoint_path = cfg.train.checkpoint
        self.loss_cfg = cfg.model.loss
        self.train_deterministic = cfg.train.deterministic_gates

        self.sent_encoder, self.tokenizer = sent_encoder, tokenizer
        
        # Infer selector input dim from encoder output (first batch)
        with torch.no_grad():
            first = next(iter(self.train_dl))
            ids = first["ids"].to(self.device)
            attn = first["attn_mask"].to(self.device)
            tkns_embd = self.sent_encoder.token_embeddings(ids, attn)
            model_dim = tkns_embd.shape[-1]

        self.model = RationaleSelectorModel(model_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.model.optim.lr),
            weight_decay=float(cfg.model.optim.weight_decay),
            betas=tuple(cfg.model.optim.betas),
        )

        self.total_tokens_history = []
        self.total_selected_history = []

        self.label_key = "scnd_labels" if cfg.model.scnd_labels else "labels"
        self.labels_present = (
            self.label_key in self.test_dl.dataset[0]
            and self.test_dl.dataset[0][self.label_key] is not None
        )
        self.counts_gold_history = []
        self.counts_pred_history = []
    
    def save_checkpoint(self):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "meta": {
                "sig": self.xp.sig,
            },
        }
        torch.save(
            state,
            self.checkpoint_path,
            _use_new_zipfile_serialization=False,
        )
        if not (self.cfg.runtime.global_log or self.cfg.runtime.epoch_log):
            self.logger.info(
                "Saved checkpoint to %s",
                os.path.abspath(self.checkpoint_path),
            )
            
    def load_checkpoint(self):
        state = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model"], strict=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.logger.info(
            "Loaded checkpoint from %s",
            os.path.abspath(self.checkpoint_path),
        )

    @torch.no_grad()
    def evaluate(self, examples: int = 0):
        total_tokens, total_selected = 0, 0
        all_tokens, all_selected, all_gates = [], [], []
        examples_str = []

        counts_pred = None
        counts_gold = None
        counts_init = False

        for batch in tqdm(self.test_dl, desc="Eval: ", disable=not self.cfg.runtime.epoch_log):
            batch = to_device(self.device, batch)
            ids = batch["ids"]
            attn = batch["attn_mask"]

            tkns_embd = self.sent_encoder.token_embeddings(ids, attn)
            g, z = self.model(tkns_embd, attn, deterministic=True)

            total_tokens += attn.sum().item()
            total_selected += g.sum().item()

            if self.labels_present:
                labels = batch[self.label_key]
                flat_labels = [x for seq in labels for x in seq]
                flat_attn = attn.bool().view(-1)
                flat_preds = g.view(-1)

                if not counts_init:
                    counts_pred = Counts(flat_labels, flat_attn, flat_preds)
                    counts_gold = Counts(flat_labels, flat_attn)
                    counts_init = True
                else:
                    counts_pred = counts_pred + Counts(flat_labels, flat_attn, flat_preds)
                    counts_gold = counts_gold + Counts(flat_labels, flat_attn)

            tokens_batch = batch["tokens"]
            for i in range(ids.size(0)):
                length = attn[i].sum().item()
                toks = tokens_batch[i][:length]
                gate_vals = g[i, :length].detach().cpu().tolist()
                selected = [int(g > 0.5) for g in gate_vals]

                all_tokens.append(toks)
                all_selected.append(selected)
                all_gates.append(gate_vals)

                if len(examples_str) < examples:
                    sel_tokens = [t for t, s in zip(toks, selected) if s]
                    examples_str.append((toks, sel_tokens))

        return (
            total_selected,
            total_tokens,
            counts_gold,
            counts_pred,
            examples_str,
            {"tokens": all_tokens, "selected": all_selected, "gates": all_gates},
            g, z,
        )

    # ------------------------------------------------------------------

    @torch.no_grad()
    def log_eval(self, epoch: int = -1, examples: int = 0):
        (
            total_selected,
            total_tokens,
            counts_gold,
            counts_pred,
            examples_str,
            record, g, z
        ) = self.evaluate(examples)

        if epoch >= 0:
            writer = open_selection_writer(self.xp, epoch)
            writer.write(json.dumps(record) + "\n")
            writer.close()

        self.total_tokens_history.append(total_tokens)
        self.total_selected_history.append(total_selected)

        if self.labels_present:
            self.counts_gold_history.append(counts_gold)
            self.counts_pred_history.append(counts_pred)

        if not self.cfg.runtime.epoch_log:
            return

        self.logger.info(
            "Eval selection rate: %.5f",
            total_selected / max(total_tokens, 1),
        )

        for i, (tokens, selected) in enumerate(examples_str):
            self.logger.info("Eval Example %d:", i + 1)
            self.logger.info("Tokens: %s", " ".join(tokens))
            self.logger.info("Selected Tokens: %s\n", " ".join(selected))

        if self.labels_present:
            self.logger.info("Eval labels:\n%s", str(counts_pred / counts_gold))
            
        with torch.no_grad():
            g = g.detach()
            p1 = g.mean().item()
            p0 = 1.0 - p1
            entropy = 0.0
            if p0 > 0: entropy -= p0 * math.log(p0)
            if p1 > 0: entropy -= p1 * math.log(p1)
            self.logger.info(f"p1={p1:.4f} entropy={entropy:.4f}")
            z = z.detach().view(-1)
            hist = torch.histc(z, bins=10, min=0.0, max=1.0)
            hist = (hist / hist.sum()).cpu().tolist()
            self.logger.info(f"z_hist {hist}")


    # ------------------------------------------------------------------

    def train(self):
        if self.cfg.runtime.global_log:
            epoch_iter = tqdm(range(self.cfg.train.epochs), desc="Training: ")
        else:
            epoch_iter = range(self.cfg.train.epochs)

        for epoch in epoch_iter:
            totals = None
            example_count = 0
            batch_iter = tqdm(
                self.train_dl,
                desc=f"Training {epoch+1}: ",
                disable=not self.cfg.runtime.epoch_log,
            )

            for batch in batch_iter:
                batch = to_device(self.device, batch)
                ids = batch["ids"]
                attn = batch["attn_mask"]

                tkns_embd = self.sent_encoder.token_embeddings(ids, attn)
                assert torch.isnan(tkns_embd).sum().item() == 0, "NaN in token embeddings"
                gates, z = self.model(tkns_embd, attn, deterministic=self.train_deterministic)
                assert torch.isnan(gates).sum().item() == 0, "NaN in gates"
                self.optimizer.zero_grad()
                losses, _ = compute_losses(gates, z, ids, attn, self.sent_encoder, self.loss_cfg)

                for name, value in losses.items():
                    if not torch.isfinite(value):
                        raise RuntimeError(f"Non-finite loss {name}: {value.item()}")

                losses["total"].backward()

                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        continue
                    if not torch.isfinite(param.grad).all():
                        raise RuntimeError(f"Non-finite gradient in {name}")
                
                self.optimizer.step()

                bs = ids.size(0)
                example_count += bs
                for k, v in losses.items():
                    if totals is None:
                        totals = {key: 0.0 for key in losses.keys()}
                    totals[k] += float(v.detach()) * bs

            metrics = {k: v / example_count for k, v in totals.items()}
            if self.cfg.runtime.epoch_log:
                self.logger.info(
                    "Epoch %d/%d train:\n%s",
                    epoch + 1,
                    self.cfg.train.epochs,
                    dict_to_table(metrics),
                )

            self.log_eval(epoch=epoch + 1, examples=self.examples_to_log)
            self.save_checkpoint()

        save_final_plots(
            self.total_selected_history, self.total_tokens_history,
            self.counts_pred_history, self.counts_gold_history, 
            self.labels_present, self.logger, self.xp
        )


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

    configure_runtime(cfg.runtime.threads, cfg.runtime.interop_threads)
    if cfg.runtime.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable, using CPU.")
    device = torch.device(cfg.runtime.device if torch.cuda.is_available() else "cpu")
    cfg.runtime.device = device.type

    train_dl, test_dl, encoder, tokenizer = initialize_data(cfg.data, logger, device=device)
    trainer = SelectorTrainer(cfg, train_dl, test_dl, encoder, tokenizer, logger, xp)

    if cfg.train.eval.eval_only:
        trainer.load_checkpoint()
        trainer.log_eval()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
