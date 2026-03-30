import os
import re
import sys
import torch

from datasets import DatasetDict
from logging import Logger
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from dora import get_xp, hydra_main, XP
from omegaconf import DictConfig
from transformers import AutoTokenizer
from numpy import linspace

from .metrics import Counts
from .dynamic_batch import DynamicBatchController
from .sentence import SentenceEncoder
from .selector import RationaleSelectorModel
from .data import PAD_TAG, SPECIAL_TAG, initialize_data
from .utils import (
    get_logger,
    configure_runtime,
    to_device,
    dict_to_table,
    load_loss_history,
    save_label_plots,
    save_loss_history,
    save_loss_plot,
    selection_rate_matrix_to_table,
)
from .retrival_fun import run_stsb_sweep


def should_disable_tqdm(short_log: bool, grid_mode: bool = False) -> bool:
    if short_log:
        return True
    if grid_mode:
        return True
    if not sys.stderr.isatty():
        return True
    return False


class SelectorTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        ds: DatasetDict,
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
        self.grid_mode = bool(cfg.runtime.grid)
        self.examples = cfg.runtime.eval.log_examples
        self.device = cfg.runtime.device
        self.rhos = linspace(cfg.model.loss.sweep_range[0], cfg.model.loss.sweep_range[1], cfg.model.loss.sweep_range[2])
        self.bf16 = bool(cfg.runtime.get("bf16", False))

        self._tqdm_disabled = should_disable_tqdm(self.short_log, self.grid_mode)

        self.xp = xp
        self.resume_training = bool(cfg.train.get("continue", False))
        self.checkpoint_dir = Path(os.getcwd())
        self.state_dir = self.checkpoint_dir / "state"
        self.models_dir = self.state_dir / "models"
        self.plots_dir = self.checkpoint_dir / "plots"
        self.loss_history_path = self.state_dir / "loss_history.json"
        self.legacy_checkpoint_path = self.checkpoint_dir / "model.pth"
        self.legacy_models_glob = "model_*.pth"

        self.tokenizer = tokenizer
        self.labels_set = None if labels_set is None else set(labels_set) | {SPECIAL_TAG}
        selector_cfg = cfg.model.selector
        self.keep_special = bool(selector_cfg.get("keep_special", True))

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

        if cfg.runtime.get("compile", False):
            self.model = torch.compile(self.model, dynamic=True)
            logger.info("torch.compile enabled (first epoch will be slower).")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.model.optim.lr),
            weight_decay=float(cfg.model.optim.weight_decay),
            betas=tuple(cfg.model.optim.betas),
        )

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.loss_history = load_loss_history(self.loss_history_path)
        self.batch_controller = DynamicBatchController(
            cfg.runtime.data,
            ds=ds,
            logger=logger,
            device=self.device,
            shuffle=bool(cfg.data.shuffle),
        )

    def checkpoint_path(self, epoch: int) -> Path:
        return self.models_dir / f"model_{epoch}.pth"

    def checkpoint_epoch(self, checkpoint_path: Path) -> int | None:
        match = re.fullmatch(r"model_(\d+)\.pth", checkpoint_path.name)
        if match is None:
            return None
        return int(match.group(1))

    def latest_checkpoint(self) -> tuple[int, Path] | None:
        candidates: list[tuple[int, Path]] = []

        for checkpoint_path in self.models_dir.glob("model_*.pth"):
            epoch = self.checkpoint_epoch(checkpoint_path)
            if epoch is not None:
                candidates.append((epoch, checkpoint_path))

        for checkpoint_path in self.checkpoint_dir.glob(self.legacy_models_glob):
            epoch = self.checkpoint_epoch(checkpoint_path)
            if epoch is not None:
                candidates.append((epoch, checkpoint_path))

        if candidates:
            return max(candidates, key=lambda item: item[0])

        if self.legacy_checkpoint_path.exists():
            return 0, self.legacy_checkpoint_path

        return None

    def save_checkpoint(self, epoch: int) -> None:
        checkpoint_path = self.checkpoint_path(epoch)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "meta": {
                    "sig": self.xp.sig,
                    "epoch": epoch,
                    **self.batch_controller.checkpoint_meta(),
                },
                "loss_history": self.loss_history,
            },
            checkpoint_path,
            _use_new_zipfile_serialization=False,
        )
        save_loss_history(self.loss_history, self.loss_history_path)
        self.logger.info("Saved checkpoint to %s", checkpoint_path)
        
    def load_checkpoint(self, checkpoint_path: Path | None = None) -> int:
        resolved_path = checkpoint_path
        resolved_epoch = None

        if resolved_path is None:
            latest = self.latest_checkpoint()
            if latest is None:
                raise FileNotFoundError("No checkpoint found in the current experiment folder.")
            resolved_epoch, resolved_path = latest
        else:
            resolved_epoch = self.checkpoint_epoch(resolved_path)

        checkpoint = torch.load(resolved_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        meta = checkpoint.get("meta", {})
        if self.loss_history_path.exists():
            self.loss_history = load_loss_history(self.loss_history_path)
        else:
            self.loss_history = [
                {str(k): float(v) for k, v in item.items()}
                for item in checkpoint.get("loss_history", [])
            ]
        restored_loaders = self.batch_controller.load_checkpoint_meta(meta)
        if restored_loaders is not None:
            self.train_dl, self.test_dl = restored_loaders
        loaded_epoch = int(meta.get("epoch", resolved_epoch or 0))
        self.logger.info(
            "Loaded checkpoint from %s with signature %s at epoch %d (batch size %d)",
            resolved_path,
            meta.get("sig", "unknown"),
            loaded_epoch,
            self.batch_controller.batch_size,
        )
        return loaded_epoch

    def record_eval_losses(self, eval_losses: dict) -> None:
        self.loss_history.append({str(k): float(v) for k, v in eval_losses.items()})
        save_loss_history(self.loss_history, self.loss_history_path)

    def write_loss_plot(self) -> None:
        if not self.loss_history:
            self.logger.info("Skipping loss plot because no loss history is available.")
            return

        loss_path = self.plots_dir / "loss.png"
        save_loss_plot(self.loss_history, str(loss_path))
        self.logger.info("Saved loss plot to %s", loss_path)

    def selection_candidate_mask(self, batch: dict) -> torch.Tensor:
        attn = batch["attn_mask"]
        if self.keep_special:
            return attn

        word_ids = batch.get("word_ids")
        if word_ids is None:
            return attn

        return attn * word_ids.ge(0).to(dtype=attn.dtype)

    def flatten_eval_labels(self, batch: dict, attn: torch.Tensor) -> list[str]:
        labels = batch["labels"]
        flat_labels = [label for seq in labels for label in seq]
        word_ids = batch.get("word_ids")
        if word_ids is None:
            return flat_labels

        flat_attn = attn.bool().view(-1).cpu()
        flat_word_ids = word_ids.view(-1).cpu().tolist()

        adjusted_labels: list[str] = []
        for label, is_attended, word_id in zip(flat_labels, flat_attn.tolist(), flat_word_ids):
            if is_attended and word_id < 0 and label == PAD_TAG:
                adjusted_labels.append(SPECIAL_TAG)
            else:
                adjusted_labels.append(label)

        return adjusted_labels

    def forward_pass(self, batch: dict, examples_count: int, total_losses: dict | None, rhos: list | None = None):
        ids = batch["ids"]
        attn = batch["attn_mask"]
        bs = ids.size(0)
        
        if rhos is None:
            rhos = self.rhos

        # Keep selector optimization in FP32 for stability; BF16 is only used for
        # the frozen first-pass token embedding extraction.
        if self.bf16:
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
                with torch.no_grad():
                    tkns_embd = self.sent_encoder.token_embeddings(ids, attn)
        else:
            with torch.no_grad():
                tkns_embd = self.sent_encoder.token_embeddings(ids, attn)

        selection_mask = self.selection_candidate_mask(batch)
        z, g_sweep, loss_tensor, losses_log, loss_sweep, rho_eff_sweep = \
            self.model(ids, tkns_embd, attn, rhos=rhos, selection_mask=selection_mask)

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
    def short_evaluate(
        self
    ) -> dict:
        self.model.eval()
        examples_count = 0
        total_losses = None

        for batch in tqdm(
            self.test_dl,
            desc="Eval",
            leave=False,
            dynamic_ncols=True,
            disable=self._tqdm_disabled,
            file=sys.stderr,
        ):
            batch = to_device(self.device, batch)
            _, _, _, _, _, _, _, examples_count, total_losses = self.forward_pass(batch, examples_count, total_losses)
        
        for k in total_losses:
            total_losses[k] /= max(examples_count, 1)
        
        return total_losses

    @torch.no_grad()
    def evaluate(
        self
    ) -> tuple[dict, list | None, list | None, list[float] | None]:
        self.model.eval()
        
        examples_count = 0
        total_losses = None

        counts_pred, counts_gold = None, None
        start, end, steps = self.cfg.model.loss.sweep_range
        sweep_range = len(linspace(start, end, steps))
        rho_eff_sum = [0.0 for _ in range(sweep_range)]
        rho_eff_weighted_num = [0.0 for _ in range(sweep_range)]
        t_eff_total = 0.0
        n_samples = 0
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

            attn, _, g_sweep, _, _, _, rho_eff_sweep, examples_count, total_losses \
                = self.forward_pass(batch, examples_count, total_losses, rhos=self.rhos)

            t_eff = self.selection_candidate_mask(batch).sum(dim=1).float()
            t_eff_total += t_eff.sum().item()
            n_samples += int(t_eff.numel())

            for i, r in enumerate(rho_eff_sweep):
                r_float = r.float()
                rho_eff_sum[i] += r_float.sum().item()
                rho_eff_weighted_num[i] += (r_float * t_eff).sum().item()

            if self.labels_set is not None:
                flat_labels = self.flatten_eval_labels(batch, attn)
                flat_attn = attn.bool().view(-1).cpu()

                for i, g in enumerate(g_sweep):
                    flat_preds = g.view(-1)  # already CPU
                    counts_pred[i] += Counts(flat_labels, flat_attn, flat_preds)
                    counts_gold[i] += Counts(flat_labels, flat_attn)

        for k in total_losses:
            total_losses[k] /= max(examples_count, 1)

        if self.labels_set is not None:
            selection_rates: list[float] = list(self.rhos)
            selection_rates = [value / max(n_samples, 1) for value in rho_eff_sum]

            for i, (pred, gold) in enumerate(zip(counts_pred, counts_gold)):
                total_gold = sum(int(v) for v in gold.data.values())
                total_pred = sum(int(v) for v in pred.data.values())
                overall_keep = (total_pred / total_gold) if total_gold > 0 else 0.0

                special_gold = int(gold.data.get(SPECIAL_TAG, 0))
                special_pred = int(pred.data.get(SPECIAL_TAG, 0))
                non_special_gold = total_gold - special_gold
                non_special_pred = total_pred - special_pred
                non_special_keep = (
                    non_special_pred / non_special_gold if non_special_gold > 0 else 0.0
                )

                effective_keep = overall_keep if self.keep_special else non_special_keep
                model_weighted_keep = (
                    rho_eff_weighted_num[i] / t_eff_total if t_eff_total > 0 else 0.0
                )
                theoretical_gap = abs(float(selection_rates[i]) - float(model_weighted_keep))
                quantization_eps = 1.0 / max(
                    total_gold if self.keep_special else non_special_gold,
                    1,
                )
                gap = abs(float(selection_rates[i]) - float(effective_keep))
                assert gap <= theoretical_gap + quantization_eps, (
                    f"Selection-rate mismatch at rho[{i}]={selection_rates[i]:.3f}: "
                    f"effective_keep={effective_keep:.3f}, gap={gap:.3f}, "
                    f"bound={theoretical_gap + quantization_eps:.3f}"
                )

            self.logger.info(
                "\nLabel selection matrix (columns = mean effective selection rate):\n%s",
                selection_rate_matrix_to_table(counts_pred, counts_gold, selection_rates),
            )

            return total_losses, counts_pred, counts_gold, selection_rates

        return total_losses, None, None, None

    @torch.no_grad()
    def final_eval(self, record_eval_history: bool = True) -> None:
        eval_losses, counts_pred, counts_gold, selection_rates = self.evaluate()
        if record_eval_history:
            self.record_eval_losses(eval_losses)
        self.logger.info(f"\nFinal evaluation:\n{dict_to_table(eval_losses)}")
        self.write_loss_plot()
        save_label_plots(
            counts_pred,
            counts_gold,
            selection_rates if selection_rates is not None else self.rhos,
            self.plots_dir,
            self.logger,
        )
        if bool(self.cfg.runtime.eval.get("skip_stsb", False)):
            self.logger.info("Skipping STS-B sweep (runtime.eval.skip_stsb=true).")
            return

        stsb_plot_path = self.plots_dir / "spearman_vs_rho.png"
        _, _, _ = run_stsb_sweep(
            cfg=self.cfg,
            device=self.device,
            encoder=self.sent_encoder,
            tokenizer=self.tokenizer,
            selector=self.model,
            out_path=str(stsb_plot_path),
        )
        self.logger.info("Saved STS-B plot to: %s", stsb_plot_path)

    def train(self) -> None:
        start_epoch = 0

        if self.resume_training:
            latest = self.latest_checkpoint()
            if latest is None:
                self.logger.info("No checkpoint found to resume from. Starting from epoch 1.")
            else:
                _, checkpoint_path = latest
                start_epoch = self.load_checkpoint(checkpoint_path)
                if start_epoch >= self.epochs:
                    self.logger.info(
                        "Checkpoint epoch %d already reaches the target of %d epochs. Running final evaluation only.",
                        start_epoch,
                        self.epochs,
                    )
                    self.final_eval()
                    return
                self.logger.info(
                    "Resuming training from epoch %d of %d.",
                    start_epoch + 1,
                    self.epochs,
                )

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Training",
            leave=True,
            dynamic_ncols=True,
            disable=self._tqdm_disabled,
            file=sys.stderr,
        )

        for epoch in epoch_bar:
            epoch_start_stats = self.batch_controller.sample_memory()
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

                _, _, _, loss_tensor, _, _, _, examples_count, total_losses \
                    = self.forward_pass(batch, examples_count, total_losses)

                loss_tensor.backward()
                self.optimizer.step()

            for k in total_losses:
                total_losses[k] /= max(examples_count, 1)

            if self.grid_mode:
                self.logger.info("GRID_EPOCH %d/%d", epoch + 1, self.epochs)
            elif not self.short_log:
                self.logger.info(
                    "Epoch %d/%d (batch size %d)",
                    epoch + 1,
                    self.epochs,
                    self.batch_controller.batch_size,
                )
                self.logger.info(f"\n{dict_to_table(total_losses)}")

            eval_losses = self.short_evaluate()
            self.record_eval_losses(eval_losses)
            epoch_end_stats = self.batch_controller.sample_memory()
            new_loaders = self.batch_controller.maybe_reduce_after_epoch(
                epoch_start_stats,
                epoch_end_stats,
            )
            if new_loaders is not None:
                self.train_dl, self.test_dl = new_loaders
            self.save_checkpoint(epoch + 1)

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

    train_dl, test_dl, encoder, tokenizer, labels_set, ds = initialize_data(
        cfg.data,
        cfg.runtime.data,
        logger,
        device=cfg.runtime.device,
    )

    trainer = SelectorTrainer(
        cfg, ds, train_dl, test_dl, encoder, tokenizer, labels_set, logger, xp
    )

    if cfg.train.no_train:
        trainer.load_checkpoint()
        trainer.final_eval(record_eval_history=False)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
