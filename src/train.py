import os
import sys
import torch

from logging import Logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from dora import get_xp, hydra_main, XP
from omegaconf import DictConfig
from transformers import AutoTokenizer
from numpy import linspace

from .metrics import Counts
from .sentence import SentenceEncoder
from .selector import RationaleSelectorModel
from .data import initialize_data
from .utils import (
    get_logger,
    configure_runtime,
    to_device,
    final_plots,
    dict_to_table,
)


def should_disable_tqdm(short_log: bool) -> bool:
    if short_log:
        return True
    if not sys.stderr.isatty():
        return True
    return False


class SelectorTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        train_dl: DataLoader,
        test_dl: DataLoader,
        sent_encoder: SentenceEncoder,
        tokenizer: AutoTokenizer,
        labels_set: set | None,
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

        self._tqdm_disabled = should_disable_tqdm(self.short_log)

        self.xp = xp
        self.checkpoint_path = "model.pth"

        self.tokenizer = tokenizer
        self.labels_set = labels_set

        self.sent_encoder = sent_encoder.to(self.device)
        self.sent_encoder.eval()
        for p in self.sent_encoder.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            first = next(iter(self.train_dl))
            model_dim = self.sent_encoder.token_embeddings(
                first["ids"].to(self.device),
                first["attn_mask"].to(self.device),
            ).shape[-1]

        self.model = RationaleSelectorModel(
            model_dim, loss_cfg=cfg.model.loss, sent_encoder=self.sent_encoder
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.model.optim.lr),
            weight_decay=float(cfg.model.optim.weight_decay),
            betas=tuple(cfg.model.optim.betas),
        )

        self.loss_history = []

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

    def forward_pass(self, batch: dict, examples_count: int, total_losses: dict | None):
        ids = batch["ids"]
        attn = batch["attn_mask"]
        bs = ids.size(0)

        with torch.no_grad():
            tkns_embd = self.sent_encoder.token_embeddings(ids, attn)

        z, g_sweep, loss_tensor, losses_log, loss_sweep, rho_eff_sweep = \
            self.model(ids, tkns_embd, attn)

        if total_losses is None:
            total_losses = {k: 0.0 for k in losses_log}
        for k in total_losses:
            total_losses[k] += losses_log[k]

        return (
            attn,
            z,
            g_sweep,
            loss_tensor,
            losses_log,
            loss_sweep,
            rho_eff_sweep,
            examples_count + bs,
            total_losses,
        )

    @torch.no_grad()
    def evaluate(self, short: bool = False):
        self.model.eval()
        examples_count = 0
        total_losses = None
        counts_pred, counts_gold = None, None

        rho_eff_accum = None
        rho_eff_batches = 0

        if not short:
            start, end, steps = self.cfg.model.loss.sweep_range
            sweep_range = len(linspace(start, end, steps))
            counts_pred = [Counts(labels=self.labels_set) for _ in range(sweep_range)]
            counts_gold = [Counts(labels=self.labels_set) for _ in range(sweep_range)]

        for batch in tqdm(
            self.test_dl,
            desc="Eval",
            leave=False,
            dynamic_ncols=True,
            disable=self._tqdm_disabled,
            file=sys.stderr,
        ):
            batch = to_device(self.device, batch)

            (
                attn,
                z,
                g_sweep,
                loss_tensor,
                losses_log,
                loss_sweep,
                rho_eff_sweep,
                examples_count,
                total_losses,
            ) = self.forward_pass(batch, examples_count, total_losses)

            # ---- accumulate rho_eff across batches ----
            if not short:
                if rho_eff_accum is None:
                    rho_eff_accum = [0.0 for _ in rho_eff_sweep]

                for i, r in enumerate(rho_eff_sweep):
                    rho_eff_accum[i] += r.mean().item()

                rho_eff_batches += 1

            # ---- accumulate counts ----
            if not short and self.labels_set is not None:
                labels = batch["labels"]
                flat_labels = [x for seq in labels for x in seq]

                flat_attn = attn.bool().view(-1).cpu()

                for i, g in enumerate(g_sweep):
                    flat_preds = g.view(-1)  # already CPU
                    counts_pred[i] += Counts(flat_labels, flat_attn, flat_preds)
                    counts_gold[i] += Counts(flat_labels, flat_attn)

        # ---- log effective rho (properly averaged) ----
        if not short and rho_eff_accum is not None:
            rho_eff_mean = [
                x / max(rho_eff_batches, 1)
                for x in rho_eff_accum
            ]

            self.logger.info("Target ρ → effective ρ:")
            for rho_t, rho_e in zip(
                linspace(*self.cfg.model.loss.sweep_range),
                rho_eff_mean,
            ):
                self.logger.info(f"{rho_t:.3f} → {rho_e:.3f}")

        # ---- average losses ----
        for k in total_losses:
            total_losses[k] /= max(examples_count, 1)

        # ---- print metrics ----
        if self.labels_set is not None and not short:
            self.logger.info("\nLabels:\n")

            total_tokens = 0
            for batch in self.test_dl:
                total_tokens += batch["attn_mask"].sum().item()

            for i in range(len(counts_pred)):
                # Correct batch-level selection rate
                selected_mass = sum(g.sum().item() for g in g_sweep)
                rate = selected_mass / total_tokens

                print(f"rate {rate:.3f} → ", end="")
                self.logger.info(f"{(counts_pred[i] / counts_gold[i]).to_table()}")

            return total_losses, [cp / cg for cp, cg in zip(counts_pred, counts_gold)]

        return total_losses, None

    @torch.no_grad()
    def final_eval(self) -> None:
        eval_losses, counts = self.evaluate()
        self.loss_history.append(eval_losses)
        self.logger.info(f"\nFinal evaluation:\n{dict_to_table(eval_losses)}")
        final_plots(self.loss_history, counts, self.logger, self.xp)

    def train(self) -> None:
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
            total_losses = None
            examples_count = 0

            for batch in tqdm(
                self.train_dl,
                desc=f"Training Epoch {epoch + 1}",
                leave=False,
                dynamic_ncols=True,
                disable=self._tqdm_disabled,
                file=sys.stderr,
            ):
                batch = to_device(self.device, batch)
                self.optimizer.zero_grad(set_to_none=True)

                (
                    attn,
                    z,
                    g_sweep,
                    loss_tensor,
                    losses_log,
                    loss_sweep,
                    rho_eff_sweep,
                    examples_count,
                    total_losses,
                ) = self.forward_pass(batch, examples_count, total_losses)

                loss_tensor.backward()
                self.optimizer.step()

            for k in total_losses:
                total_losses[k] /= max(examples_count, 1)

            if not self.short_log:
                self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
                self.logger.info(f"\n{dict_to_table(total_losses)}")

            eval_losses, _ = self.evaluate()
            self.loss_history.append(eval_losses)
            self.save_checkpoint()

        self.final_eval()


@hydra_main(config_path="conf", config_name="default", version_base="1.1")
def main(cfg: DictConfig) -> None:
    logger = get_logger()
    xp = get_xp()

    logger.info(f"Exp signature: {xp.sig}")
    logger.info(repr(cfg))
    logger.info(f"Work dir: {os.getcwd()}")

    cfg.runtime, changed_device = configure_runtime(cfg.runtime)
    if changed_device:
        logger.warning("CUDA requested but unavailable, using CPU.")

    train_dl, test_dl, encoder, tokenizer, labels_set = initialize_data(
        cfg.data,
        cfg.runtime.data,
        logger,
        device=cfg.runtime.device,
    )

    trainer = SelectorTrainer(
        cfg, train_dl, test_dl, encoder, tokenizer, labels_set, logger, xp
    )

    if cfg.train.no_train:
        trainer.final_eval()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
