import os
import json
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
    save_train_eval_loss_plot,
    save_combined_loss_history,
    load_combined_loss_history,
    selection_rate_matrix_to_table,
    build_chi_square_payload,
    start_run_metrics_capture,
    write_metrics_artifacts,
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
        start_capture,
    ) -> None:

        self.cfg = cfg
        self.logger = logger
        self.train_dl = train_dl
        self.test_dl = test_dl

        self.epochs = cfg.train.epochs
        self.short_log = cfg.runtime.eval.short_log
        self.grid_mode = bool(cfg.runtime.grid)
        self.device = cfg.runtime.device
        self.rhos = linspace(cfg.model.loss.sweep_range[0], cfg.model.loss.sweep_range[1], cfg.model.loss.sweep_range[2])
        self.bf16 = bool(cfg.runtime.get("bf16", False))

        self._tqdm_disabled = should_disable_tqdm(self.short_log, self.grid_mode)

        self.xp = xp
        self.start_capture = start_capture
        self.resume_training = bool(cfg.train.get("continue", False))
        self.completed_epochs = 0
        self.checkpoint_dir = Path(os.getcwd())
        self.state_dir = self.checkpoint_dir / "state"
        self.models_dir = self.state_dir / "models"
        self.plots_dir = self.checkpoint_dir / "plots"
        self.data_dir = self.checkpoint_dir / "data"
        self.loss_history_path = self.data_dir / "loss_history.json"
        self.legacy_checkpoint_path = self.checkpoint_dir / "model.pth"
        self.legacy_models_glob = "model_*.pth"

        self.tokenizer = tokenizer
        self.labels_set = None if labels_set is None else set(labels_set) | {SPECIAL_TAG}

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
            model_dim,
            loss_cfg=cfg.model.loss,
            selector_cfg=cfg.model.get("selector", None),
            sent_encoder=self.sent_encoder,
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
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.train_loss_history, self.eval_loss_history = load_combined_loss_history(self.loss_history_path)
        # Backward-compatible alias used by older checkpoint payloads.
        self.loss_history = self.eval_loss_history
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
                "loss_history": self.eval_loss_history,
                "eval_loss_history": self.eval_loss_history,
                "train_loss_history": self.train_loss_history,
            },
            checkpoint_path,
            _use_new_zipfile_serialization=False,
        )
        save_combined_loss_history(self.train_loss_history, self.eval_loss_history, self.loss_history_path)
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
            self.train_loss_history, self.eval_loss_history = load_combined_loss_history(self.loss_history_path)
        else:
            # Backward compatibility for old checkpoints/history format.
            self.eval_loss_history = [
                {str(k): float(v) for k, v in item.items()}
                for item in checkpoint.get("eval_loss_history", checkpoint.get("loss_history", []))
            ]
            self.train_loss_history = [
                {str(k): float(v) for k, v in item.items()}
                for item in checkpoint.get("train_loss_history", [])
            ]
        self.loss_history = self.eval_loss_history
        restored_loaders = self.batch_controller.load_checkpoint_meta(meta)
        if restored_loaders is not None:
            self.train_dl, self.test_dl = restored_loaders
        loaded_epoch = int(meta.get("epoch", resolved_epoch or 0))
        self.completed_epochs = max(self.completed_epochs, loaded_epoch)
        self.logger.info(
            "Loaded checkpoint from %s with signature %s at epoch %d (batch size %d)",
            resolved_path,
            meta.get("sig", "unknown"),
            loaded_epoch,
            self.batch_controller.batch_size,
        )
        return loaded_epoch

    def record_eval_losses(self, eval_losses: dict) -> None:
        self.eval_loss_history.append({str(k): float(v) for k, v in eval_losses.items()})
        self.loss_history = self.eval_loss_history
        save_combined_loss_history(self.train_loss_history, self.eval_loss_history, self.loss_history_path)

    def record_train_losses(self, train_losses: dict) -> None:
        self.train_loss_history.append({str(k): float(v) for k, v in train_losses.items()})
        save_combined_loss_history(self.train_loss_history, self.eval_loss_history, self.loss_history_path)

    def write_loss_plot(self) -> None:
        if not self.train_loss_history and not self.eval_loss_history:
            self.logger.info("Skipping loss plot because no loss history is available.")
            return

        loss_path = self.plots_dir / "loss.png"
        save_train_eval_loss_plot(self.train_loss_history, self.eval_loss_history, str(loss_path))
        self.logger.info("Saved loss plot to %s", loss_path)

    def _save_eval_selections(self, counts_pred: list, counts_gold: list, selection_rates: list) -> None:
        """Save per-example selections from evaluation."""
        if counts_pred is None or counts_gold is None:
            return
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        selections_path = self.data_dir / "selections.json"
        
        # Convert Counts objects to serializable format
        selections_data = {
            "selection_rates": [float(r) for r in selection_rates],
            "selections_by_rho": []
        }
        
        for rate, pred, gold in zip(selection_rates, counts_pred, counts_gold):
            selections_data["selections_by_rho"].append({
                "rho": float(rate),
                "pred_counts": dict(pred.data),
                "gold_counts": dict(gold.data),
            })
        
        selections_path.write_text(json.dumps(selections_data, indent=2), encoding="utf-8")
        self.logger.info("Saved eval selections to %s", selections_path)

    def _save_eval_chi_square(self, counts_pred: list, counts_gold: list, selection_rates: list) -> None:
        """Save chi-square and Cramer's V data from evaluation counts."""
        if counts_pred is None or counts_gold is None:
            return

        self.data_dir.mkdir(parents=True, exist_ok=True)
        chi_square_path = self.data_dir / "chi_square.json"
        chi_square_data = build_chi_square_payload(counts_pred, counts_gold, selection_rates)
        chi_square_path.write_text(json.dumps(chi_square_data, indent=2), encoding="utf-8")
        self.logger.info("Saved eval chi-square data to %s", chi_square_path)

    def forward_pass(self, batch: dict, examples_count: int, total_loss: float , rhos: list | None = None):
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

        _, g, loss = self.model(ids, tkns_embd, attn, rhos=rhos)

        return (
            attn,
            g,
            loss,
            examples_count + bs,
            total_loss + loss.item() * bs,
        )
        
    @torch.no_grad()
    def short_evaluate(
        self
    ) -> dict:
        self.model.eval()
        examples_count = 0
        total_loss = 0.0

        for batch in tqdm(
            self.test_dl,
            desc="Eval",
            leave=False,
            dynamic_ncols=True,
            disable=self._tqdm_disabled,
            file=sys.stderr,
        ):
            batch = to_device(self.device, batch)
            _, _, _, examples_count, total_loss \
                = self.forward_pass(batch, examples_count, total_loss)
        
        total_loss /= max(1, examples_count)

        return {"eval_loss": total_loss}

    @torch.no_grad()
    def evaluate(
        self
    ) -> tuple[dict, list | None, list | None, list[float] | None]:
        self.model.eval()
        
        examples_count = 0
        total_loss = 0.0

        counts_pred, counts_gold = None, None
        sweep_range = len(self.rhos)
        rho_eff_sum = [0.0 for _ in range(sweep_range)]
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

            attn, g_masks, _, examples_count, total_loss \
                = self.forward_pass(batch, examples_count, total_loss, rhos=self.rhos)

            L_eff = attn.float().sum(-1)                        # [B]
            for i, g_i in enumerate(g_masks):
                rho_eff_sum[i] += (g_i.sum(-1) / L_eff.clamp(min=1)).sum().item()

            if self.labels_set is not None:
                labels = batch["labels"]
                flat_labels = [label for seq in labels for label in seq]
                flat_attn = attn.bool().view(-1).cpu()

                for i, g_i in enumerate(g_masks):
                    flat_preds = g_i.cpu().view(-1)
                    counts_pred[i] += Counts(flat_labels, flat_attn, flat_preds)
                    counts_gold[i] += Counts(flat_labels, flat_attn)

        total_loss /= max(1, examples_count)

        if self.labels_set is not None:
            selection_rates: list[float] = [value / max(examples_count, 1) for value in rho_eff_sum]

            self.logger.info(
                "\nLabel selection matrix (columns = mean effective selection rate):\n%s",
                selection_rate_matrix_to_table(counts_pred, counts_gold, selection_rates),
            )

            return {"eval_loss": total_loss}, counts_pred, counts_gold, selection_rates

        return {"eval_loss": total_loss}, None, None, None

    @torch.no_grad()
    def final_eval(self, record_eval_history: bool = True) -> None:
        eval_losses, counts_pred, counts_gold, selection_rates = self.evaluate()
        if record_eval_history:
            self.record_eval_losses(eval_losses)
        self.logger.info("Final evaluation: eval_loss=%.4f", eval_losses.get("eval_loss", 0.0))
        self.write_loss_plot()
        
        # Save selections data (per-example selections from eval)
        self._save_eval_selections(counts_pred, counts_gold, selection_rates)
        self._save_eval_chi_square(counts_pred, counts_gold, selection_rates)
        
        if bool(self.cfg.runtime.eval.get("skip_stsb", False)):
            self.logger.info("Skipping STS-B sweep (runtime.eval.skip_stsb=true).")
            return

        # Create data and plots directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        stsb_plot_path = self.plots_dir / "spearman_vs_rho.png"
        base, ours, rand = run_stsb_sweep(
            cfg=self.cfg,
            device=self.device,
            encoder=self.sent_encoder,
            tokenizer=self.tokenizer,
            selector=self.model,
            out_path=str(stsb_plot_path),
        )
        stsb_path = self.data_dir / "stsb.json"
        stsb_data = {
            "base": float(base),
            "ours_by_rho": {str(float(k)): float(v) for k, v in ours.items()},
            "random_by_rho": {str(float(k)): float(v) for k, v in rand.items()},
            "rhos": [float(k) for k in ours.keys()],
        }
        stsb_path.write_text(json.dumps(stsb_data, indent=2), encoding="utf-8")
        self.logger.info("Saved STS-B plot to: %s", stsb_plot_path)
        self.logger.info("Saved STS-B data to: %s", stsb_path)

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
                    self.completed_epochs = self.epochs
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
            total_loss = 0.0
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

                _, _, loss, examples_count, total_loss \
                    = self.forward_pass(batch, examples_count, total_loss)

                loss.backward()
                self.optimizer.step()

            total_loss /= max(1, examples_count)
            
            train_losses = {"train_loss": total_loss}
            self.record_train_losses(train_losses)

            eval_losses = self.short_evaluate()
            self.record_eval_losses(eval_losses)

            # Always log epoch details (for audit trail)
            self.logger.info(
                "Epoch %d/%d | train_loss=%.4f | eval_loss=%.4f | batch_size=%d",
                epoch + 1,
                self.epochs,
                train_losses.get("train_loss", 0.0),
                eval_losses.get("eval_loss", 0.0),
                self.batch_controller.batch_size,
            )
            epoch_end_stats = self.batch_controller.sample_memory()
            new_loaders = self.batch_controller.maybe_reduce_after_epoch(
                epoch_start_stats,
                epoch_end_stats,
            )
            if new_loaders is not None:
                self.train_dl, self.test_dl = new_loaders
            self.completed_epochs = epoch + 1
            print("\a", end="", flush=True, file=sys.stderr)
            self.save_checkpoint(epoch + 1)
            write_metrics_artifacts(
                cfg=self.cfg,
                xp=self.xp,
                train_loss_history=self.train_loss_history,
                eval_loss_history=self.eval_loss_history,
                start_capture=self.start_capture,
                epochs_completed=self.completed_epochs,
                epochs_target=int(self.epochs),
                training_completed=False,
            )

        self.final_eval()


@hydra_main(config_path="conf", config_name="default", version_base="1.1")
def main(cfg: DictConfig) -> None:
    start_capture = start_run_metrics_capture()

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
        keep_special=bool(cfg.model.get("keep_special", True)),
    )

    trainer = SelectorTrainer(
        cfg, ds, train_dl, test_dl, encoder, tokenizer, labels_set, logger, xp, start_capture
    )

    if cfg.train.no_train:
        trainer.load_checkpoint(Path(os.getcwd()) / "state/models/" / str(cfg.train.checkpoint_path))
        trainer.final_eval(record_eval_history=False)
    else:
        trainer.train()
        trainer.completed_epochs = int(cfg.train.epochs)
        write_metrics_artifacts(
            cfg=cfg,
            xp=xp,
            train_loss_history=trainer.train_loss_history,
            eval_loss_history=trainer.eval_loss_history,
            start_capture=start_capture,
            epochs_completed=trainer.completed_epochs,
            epochs_target=int(cfg.train.epochs),
            training_completed=True,
        )


if __name__ == "__main__":
    main()
