import os, torch, json

from typing import Dict
from logging import Logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from dora import get_xp, hydra_main, XP
from transformers import AutoTokenizer

from .metrics import Counts
from .sentence import SentenceEncoder
from .selector import RationaleSelectorModel
from .data import initialize_data
from .losses import recon_loss
from .utils import (
    get_logger,
    configure_runtime,
    open_selection_writer,
    to_device,
    save_final_plots,
    tkns_to_words,
    dict_to_table
)

def compute_losses(ids, attn, g, z, e_loss_raw, sent_encoder, loss_cfg):
    e_loss = e_loss_raw * loss_cfg.l_e
    gates = g if loss_cfg.hard else z
    pred_rep = sent_encoder.encode(ids, attn * gates)
    full_rep = sent_encoder.encode(ids, attn)
    r_loss = recon_loss(pred_rep, full_rep)
    return {
        "recon": float(r_loss.detach()),
        "entropy": float(e_loss.detach()),
        "total": float((r_loss + e_loss).detach())
    }, r_loss + e_loss, gates

class SelectorTrainer:
    def __init__(self, cfg: Dict, train_dl: DataLoader, test_dl: DataLoader, \
        sent_encoder: SentenceEncoder,tokenizer: AutoTokenizer, logger: Logger, xp: XP):
        
        self.loss_cfg = cfg.model.loss

        self.logger = logger
        self.train_dl = train_dl
        self.test_dl = test_dl
        
        self.epochs = cfg.train.epochs
        self.short_log = cfg.runtime.eval.short_log
        self.examples = cfg.runtime.eval.log_examples
        self.device = cfg.runtime.device
        
        self.xp = xp
        self.checkpoint_path = "model.pth"

        self.sent_encoder, self.tokenizer = sent_encoder, tokenizer
        
        with torch.no_grad():
            first = next(iter(self.train_dl))
            model_dim = self.sent_encoder.token_embeddings(first["ids"].to(self.device), \
                first["attn_mask"].to(self.device)).shape[-1]

        self.model = RationaleSelectorModel(model_dim, rho=cfg.model.loss.s_target, tau=cfg.model.loss.tau).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.model.optim.lr),
            weight_decay=float(cfg.model.optim.weight_decay),
            betas=tuple(cfg.model.optim.betas),
        )

        self.label_key = "scnd_labels" if cfg.data.scnd_labels else "labels"
        self.labels_present = (
            self.label_key in self.test_dl.dataset[0]
            and self.test_dl.dataset[0][self.label_key] is not None
        )
        
        self.counts_gold_history, self.counts_pred_history  = [], []
        self.total_tokens_history, self.total_selected_history = [], []
        self.loss_history = []
        
    
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
        if not self.short_log:
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
        self.model.eval()
        total_tokens, total_selected, examples_count = 0, 0, 0
        total_losses = {"total": 0.0, "recon": 0.0, "entropy": 0.0}
        all_tokens, all_selected, all_gates = [], [], []
        examples_str = []

        counts_pred = None
        counts_gold = None
        counts_init = False

        for batch in tqdm(
            self.test_dl,
            desc="Eval",
            position=2,
            leave=False,
            dynamic_ncols=True,
            disable=self.short_log,
        ):
            batch = to_device(self.device, batch)
            ids = batch["ids"]
            attn = batch["attn_mask"]
            word_ids = batch["word_ids"]
            bs = ids.size(0)

            tkns_embd = self.sent_encoder.token_embeddings(ids, attn)
            g, z, e_loss = self.model(tkns_embd, attn, deterministic=True)
            losses, _, gates = compute_losses(ids, attn, g, z, e_loss, self.sent_encoder, self.loss_cfg)
            bs = ids.size(0)
            examples_count += bs
            
            total_losses = {k: v+losses[k] for k, v in losses.items()}
            
            word_pred, word_attn, word_lbls = tkns_to_words(
                gates=g,
                attn_mask=attn,
                labels=batch[self.label_key],
                word_ids=batch["word_ids"],
                threshold=0.5,
            )

            total_tokens += attn.sum().item()
            total_selected += (gates * attn).sum().item()

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
            for i in range(bs):
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
                    
        total_losses = {k: v / examples_count for k, v in total_losses.items()}
        self.loss_history.append(losses)
        
        return (
            total_selected,
            total_tokens,
            counts_gold,
            counts_pred,
            examples_str,
            {"tokens": all_tokens, "selected": all_selected, "gates": all_gates},
            total_losses
        )

    @torch.no_grad()
    def log_eval(self, epoch: int = -1):
        (
            total_selected,
            total_tokens,
            counts_gold,
            counts_pred,
            examples_str,
            record,
            losses
        ) = self.evaluate(self.examples)

        if epoch >= 0:
            writer = open_selection_writer(self.xp, epoch)
            writer.write(json.dumps(record) + "\n")
            writer.close()

        self.total_tokens_history.append(total_tokens)
        self.total_selected_history.append(total_selected)

        if self.labels_present:
            self.counts_gold_history.append(counts_gold)
            self.counts_pred_history.append(counts_pred)

        if self.short_log:
            return
        
        self.logger.info(f"Selection rate: {total_selected / max(total_tokens, 1)}")
        self.logger.info(f"\n{dict_to_table(losses)}")

        for i, (tokens, selected) in enumerate(examples_str):
            self.logger.info("Eval Example %d:", i + 1)
            self.logger.info("Tokens: %s", " ".join(tokens))
            self.logger.info("Selected Tokens: %s\n", " ".join(selected))

        if self.labels_present:
            self.logger.info("Eval labels:\n%s", (counts_pred / counts_gold).to_table())

    def train(self):
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Training",
            position=0,
            leave=True,
            dynamic_ncols=True,
            disable=not self.short_log,
        )

        for epoch in epoch_bar:
            batch_bar = tqdm(
                self.train_dl,
                desc=f"Training Epoch {epoch+1}",
                position=1,
                leave=False,
                dynamic_ncols=True,
                disable=self.short_log,
            )

            total_losses = {"total": 0.0, "recon": 0.0, "entropy": 0.0}
            examples_count = 0
            
            for batch in batch_bar:
                self.optimizer.zero_grad(set_to_none=True)
                batch = to_device(self.device, batch)
                ids = batch["ids"]
                attn = batch["attn_mask"]

                tkns_embd = self.sent_encoder.token_embeddings(ids, attn)
                g, z, e_loss = self.model(tkns_embd, attn, deterministic=False)
                losses, loss, _ = compute_losses(ids, attn, g, z, e_loss, self.sent_encoder, self.loss_cfg)
                bs = ids.size(0)
                examples_count += bs
                
                loss.backward()
                self.optimizer.step()
                total_losses = {k: v+losses[k] for k, v in losses.items()}
                
            if not self.short_log:
                total_losses = {k: v/examples_count for k, v in total_losses.items()}
                self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
                self.logger.info(f"\n{dict_to_table(total_losses)}")
            self.log_eval(epoch=epoch + 1)
            self.model.train()
            self.save_checkpoint()

        save_final_plots(
            self.counts_pred_history, self.counts_gold_history, 
            self.loss_history,
            self.labels_present, self.logger, self.xp
        )


@hydra_main(config_path="conf", config_name="default", version_base="1.1")
def main(cfg):
    logger = get_logger()
    xp = get_xp()
    logger.info(f"Exp signature: {xp.sig}")
    logger.info(repr(cfg))
    logger.info(f"Work dir: {os.getcwd()}")

    cfg.runtime, changed_device = configure_runtime(cfg.runtime)
    if changed_device:
        logger.warning("CUDA requested but unavailable, using CPU.")
    
    train_dl, test_dl, encoder, tokenizer = \
        initialize_data(cfg.data, cfg.runtime.data, logger, device=cfg.runtime.device)
    trainer = SelectorTrainer(cfg, train_dl, test_dl, encoder, tokenizer, logger, xp)
    
    if cfg.train.no_train:
        trainer.load_checkpoint()
        trainer.log_eval()
    else:
        trainer.train()


if __name__ == "__main__":
    main()