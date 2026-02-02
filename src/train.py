import os
import sys
import json
import torch
import logging

from logging import Logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dora import get_xp, hydra_main, XP
from omegaconf import DictConfig
from transformers import AutoTokenizer

from .metrics import Counts
from .sentence import SentenceEncoder, FrozenLLMEncoder
from .selector import RationaleSelectorModel
from .data import initialize_data
from .losses import recon_loss
from .utils import (
    get_logger,
    configure_runtime,
    open_selection_writer,
    to_device,
    save_final_plots,
    dict_to_table,
)


# --------------------------------------------------
# Multi-process safety (ONLY rank 0 prints bars/logs)
# --------------------------------------------------

def _env_rank() -> int | None:
    """Try to infer process rank from common environment variables."""
    keys = [
        "RANK",
        "LOCAL_RANK",
        "SLURM_PROCID",
        "PMI_RANK",
        "OMPI_COMM_WORLD_RANK",
        "MV2_COMM_WORLD_RANK",
        "DORA_RANK",
    ]
    for k in keys:
        v = os.environ.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except ValueError:
            continue
    return None


def is_main_process() -> bool:
    r = _env_rank()
    return (r is None) or (r == 0)


def should_disable_tqdm(short_log: bool) -> bool:
    # If the user asked for short_log, follow it.
    if short_log:
        return True
    # If we aren't the main process, NEVER print progress bars.
    if not is_main_process():
        return True
    # If tqdm can't do cursor movement (not a TTY), don't spam lines.
    if not sys.stderr.isatty():
        return True
    return False


# --------------------------------------------------
# Loss
# --------------------------------------------------

def compute_losses(
    ids: torch.Tensor,
    attn: torch.Tensor,
    z: torch.Tensor,
    sent_encoder: SentenceEncoder,
    reg_term: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor]:

    if isinstance(sent_encoder, FrozenLLMEncoder):
        pred_rep = sent_encoder.encode(ids, attn * z, original_attn=attn)
    else:
        pred_rep = sent_encoder.encode(ids, attn * z)

    full_rep = sent_encoder.encode(ids, attn)

    recon = recon_loss(pred_rep, full_rep)
    total = recon + reg_term

    logs = {
        "recon": recon.item(),
        "reg": reg_term.item(),
        "total": total.item(),
    }

    return logs, total


# --------------------------------------------------
# Trainer
# --------------------------------------------------

class SelectorTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        train_dl: DataLoader,
        test_dl: DataLoader,
        sent_encoder: SentenceEncoder,
        tokenizer: AutoTokenizer,
        logger: Logger,
        xp: XP,
    ) -> None:

        self.cfg = cfg
        self.logger = logger
        self.train_dl = train_dl
        self.test_dl = test_dl

        self.epochs = cfg.train.epochs
        self.short_log = cfg.runtime.eval.short_log
        self.examples = cfg.runtime.eval.log_examples
        self.device = cfg.runtime.device

        # Robust tqdm enable/disable
        self._tqdm_disabled = should_disable_tqdm(self.short_log)

        self.xp = xp
        self.checkpoint_path = "model.pth"

        self.sent_encoder = sent_encoder
        self.tokenizer = tokenizer

        with torch.no_grad():
            first = next(iter(self.train_dl))
            model_dim = self.sent_encoder.token_embeddings(
                first["ids"].to(self.device),
                first["attn_mask"].to(self.device),
            ).shape[-1]

        self.model = RationaleSelectorModel(model_dim, hard_type=cfg.model.selector.hard_type).to(self.device)

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

        self.counts_gold_history = []
        self.counts_pred_history = []
        self.total_tokens_history = []
        self.total_selected_history = []
        self.loss_history = []

    # --------------------------------------------------

    def save_checkpoint(self) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "meta": {"sig": self.xp.sig},
            },
            self.checkpoint_path,
            _use_new_zipfile_serialization=False,
        )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    @torch.no_grad()
    def evaluate(self, examples: int = 0):

        self.model.eval()

        total_tokens = 0
        total_selected = 0
        examples_count = 0
        total_losses = {"recon": 0.0, "reg": 0.0, "total": 0.0}

        all_tokens, all_selected, all_gates = [], [], []
        examples_str = []

        counts_pred = None
        counts_gold = None
        counts_init = False

        for batch in tqdm(
            self.test_dl,
            desc="Eval",
            leave=False,
            dynamic_ncols=True,
            disable=self._tqdm_disabled,
            file=sys.stderr,
        ):
            batch = to_device(self.device, batch)

            ids = batch["ids"]
            attn = batch["attn_mask"]
            bs = ids.size(0)

            tkns_embd = self.sent_encoder.token_embeddings(ids, attn)
            z, g, reg_term = self.model(tkns_embd, attn)
            logs, _ = compute_losses(ids, attn, g, self.sent_encoder, reg_term)

            examples_count += bs
            for k in total_losses:
                total_losses[k] += logs[k]

            total_tokens += attn.sum().item()
            total_selected += (g * attn).sum().item()

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
                    counts_pred += Counts(flat_labels, flat_attn, flat_preds)
                    counts_gold += Counts(flat_labels, flat_attn)

            tokens_batch = batch["tokens"]
            for i in range(bs):
                length = attn[i].sum().item()
                toks = tokens_batch[i][:length]
                gate_vals = g[i, :length].cpu().tolist()
                selected = [int(v > 0.5) for v in gate_vals]

                all_tokens.append(toks)
                all_selected.append(selected)
                all_gates.append(gate_vals)

                if len(examples_str) < examples:
                    sel_tokens = [t for t, s in zip(toks, selected) if s]
                    examples_str.append((toks, sel_tokens))

        for k in total_losses:
            total_losses[k] /= max(examples_count, 1)

        self.loss_history.append(total_losses)

        return (
            total_selected,
            total_tokens,
            counts_gold,
            counts_pred,
            examples_str,
            {"tokens": all_tokens, "selected": all_selected, "gates": all_gates},
            total_losses,
        )

    @torch.no_grad()
    def log_eval(self, epoch: int = -1) -> None:
        (
            total_selected,
            total_tokens,
            counts_gold,
            counts_pred,
            examples_str,
            record,
            losses,
        ) = self.evaluate(self.examples)

        # Only main process writes artifacts
        if is_main_process() and epoch >= 0:
            writer = open_selection_writer(self.xp, epoch)
            writer.write(json.dumps(record) + "\n")
            writer.close()

        self.total_tokens_history.append(total_tokens)
        self.total_selected_history.append(total_selected)

        if self.labels_present:
            self.counts_gold_history.append(counts_gold)
            self.counts_pred_history.append(counts_pred)

        # Only main process logs info
        if (not is_main_process()) or self.short_log:
            return

        self.logger.info(f"Selection rate: {total_selected / max(total_tokens, 1)}")
        self.logger.info(f"\n{dict_to_table(losses)}")

        for i, (tokens, selected) in enumerate(examples_str):
            self.logger.info("Eval Example %d:", i + 1)
            self.logger.info("Tokens: %s", " ".join(tokens))
            self.logger.info("Selected Tokens: %s\n", " ".join(selected))

        if self.labels_present:
            self.logger.info("Eval labels:\n%s", (counts_pred / counts_gold).to_table())

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    def train(self) -> None:
        # If bars are disabled anyway, no need for redirect
        ctx = logging_redirect_tqdm() if not self._tqdm_disabled else nullcontext()

        with ctx:
            epoch_bar = tqdm(
                range(self.epochs),
                desc="Training",
                leave=True,
                dynamic_ncols=True,
                disable=self._tqdm_disabled,
                file=sys.stderr,
            )

            for epoch in epoch_bar:
                self.model.train()
                total_losses = {"recon": 0.0, "reg": 0.0, "total": 0.0}
                examples_count = 0

                for batch in tqdm(
                    self.train_dl,
                    desc=f"Training Epoch {epoch + 1}",
                    leave=False,
                    dynamic_ncols=True,
                    disable=self._tqdm_disabled,
                    file=sys.stderr,
                ):
                    self.optimizer.zero_grad(set_to_none=True)

                    batch = to_device(self.device, batch)
                    ids = batch["ids"]
                    attn = batch["attn_mask"]

                    tkns_embd = self.sent_encoder.token_embeddings(ids, attn)
                    z, g, reg_term = self.model(tkns_embd, attn)
                    logs, loss = compute_losses(ids, attn, g, self.sent_encoder, reg_term)

                    loss.backward()
                    self.optimizer.step()

                    bs = ids.size(0)
                    examples_count += bs
                    for k in total_losses:
                        total_losses[k] += logs[k]

                for k in total_losses:
                    total_losses[k] /= max(examples_count, 1)

                if is_main_process() and (not self.short_log):
                    self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
                    self.logger.info(f"\n{dict_to_table(total_losses)}")

                self.log_eval(epoch + 1)

                # Only main process should checkpoint (avoid races / corruption)
                if is_main_process():
                    self.save_checkpoint()

        # Only main process makes plots
        if is_main_process():
            save_final_plots(
                self.counts_pred_history,
                self.counts_gold_history,
                self.loss_history,
                self.labels_present,
                self.logger,
                self.xp,
            )


# --------------------------------------------------
# Main
# --------------------------------------------------

# Python 3.7+ friendly nullcontext replacement if needed
try:
    from contextlib import nullcontext
except ImportError:
    class nullcontext:
        def __enter__(self): return None
        def __exit__(self, *args): return False


@hydra_main(config_path="conf", config_name="default", version_base="1.1")
def main(cfg: DictConfig) -> None:
    logger = get_logger()
    xp = get_xp()

    # Silence non-main process logging (prevents clutter + tqdm corruption)
    if not is_main_process():
        logger.setLevel(logging.ERROR)
        logger.propagate = False

    logger.info(f"Exp signature: {xp.sig}")
    logger.info(repr(cfg))
    logger.info(f"Work dir: {os.getcwd()}")

    cfg.runtime, changed_device = configure_runtime(cfg.runtime)
    if changed_device:
        logger.warning("CUDA requested but unavailable, using CPU.")

    train_dl, test_dl, encoder, tokenizer = initialize_data(
        cfg.data,
        cfg.runtime.data,
        logger,
        device=cfg.runtime.device,
    )

    trainer = SelectorTrainer(cfg, train_dl, test_dl, encoder, tokenizer, logger, xp)

    if cfg.train.no_train:
        trainer.log_eval()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
