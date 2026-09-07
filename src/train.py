import json
import math
import os
import sys
import torch

from logging import Logger
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from forge import start_run
from forge.core import ExperimentRun
from omegaconf import DictConfig
from transformers import AutoTokenizer
from numpy import linspace

from .metrics import Counts, SelectionLog, build_token_frequency_table
from .sentence import SentenceEncoder
from .selector import RationaleSelectorModel
from .data import SPECIAL_TAG, initialize_data
from .eval import save_eval_artifacts
from .retrival_fun import run_stsb_sweep
from .utils import (
    get_logger,
    build_final_metrics,
    checkpoint_epoch,
    configure_runtime,
    find_resume_checkpoint,
    to_device,
    save_combined_loss_history,
    load_combined_loss_history,
    start_run_metrics_capture,
    write_metrics_details,
    should_disable_tqdm,
    remap_checkpoint_state_dict,
)
from .view import save_train_eval_loss_plot, save_eval_plots


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
        run: ExperimentRun,
        start_capture,
    ) -> None:

        torch.manual_seed(int(cfg.runtime.get("seed", 42)))

        self.cfg = cfg
        self.logger = logger
        self.train_dl = train_dl
        self.test_dl = test_dl

        self.epochs = cfg.train.epochs
        self.short_log = cfg.runtime.eval.short_log
        self.skip_eval = bool(cfg.runtime.eval.get("skip", False))
        self.grid_mode = bool(cfg.runtime.grid)
        self.device = cfg.runtime.device
        self.rhos = linspace(cfg.model.loss.sweep_range[0], cfg.model.loss.sweep_range[1], cfg.model.loss.sweep_range[2])

        rhos_per_step = cfg.train.get("rhos_per_step", None)
        self.rhos_per_step = len(self.rhos) if rhos_per_step is None else min(int(rhos_per_step), len(self.rhos))

        self._tqdm_disabled = should_disable_tqdm(self.short_log, self.grid_mode)

        self.run = run
        self.start_capture = start_capture
        self.resume_training = bool(cfg.train.get("continue", False))
        self.completed_epochs = 0
        # forge.start_run() has already chdir'd into this run's directory.
        self.checkpoint_dir = Path(os.getcwd())
        self.state_dir = self.checkpoint_dir / "state"
        self.models_dir = self.state_dir / "models"
        self.plots_dir = self.checkpoint_dir / "plots"
        self.data_dir = self.checkpoint_dir / "data"
        self.loss_history_path = self.data_dir / "loss_history.json"

        self.tokenizer = tokenizer
        self.labels_set = None if labels_set is None else set(labels_set) | {SPECIAL_TAG}
        self.label_order = sorted(self.labels_set) if self.labels_set is not None else []

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

        # No training loop to amortize the compile warmup over in eval-only
        # (train.no_train) runs, so skip it there regardless of runtime.compile.
        if cfg.runtime.get("compile", False) and not bool(cfg.train.get("no_train", False)):
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

    def checkpoint_path(self, epoch: int) -> Path:
        return self.models_dir / f"model_{epoch}.pth"

    def latest_checkpoint(self) -> tuple[int, Path] | None:
        return find_resume_checkpoint(self.run)

    def save_checkpoint(self, epoch: int) -> None:
        checkpoint_path = self.checkpoint_path(epoch)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "meta": {
                    "sig": self.run.signature,
                    "epoch": epoch,
                },
            },
            checkpoint_path,
        )
        save_combined_loss_history(self.train_loss_history, self.eval_loss_history, self.loss_history_path)
        self.logger.info("Saved checkpoint to %s", checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path | None = None) -> int:
        if checkpoint_path is None:
            latest = self.latest_checkpoint()
            if latest is None:
                raise FileNotFoundError("No checkpoint found in the current experiment folder.")
            _, checkpoint_path = latest

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = remap_checkpoint_state_dict(checkpoint["model"], self.model.state_dict())
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        meta = checkpoint.get("meta", {})

        # The checkpoint may live in a sibling run directory (see
        # find_resume_checkpoint), so the loss history is read from beside it
        # rather than from this run's own -- otherwise resuming would restart
        # the loss curves from empty. state/models/model_N.pth -> run dir.
        self.train_loss_history, self.eval_loss_history = load_combined_loss_history(
            checkpoint_path.parents[2] / "data" / "loss_history.json"
        )

        loaded_epoch = int(meta.get("epoch", checkpoint_epoch(checkpoint_path) or 0))
        self.completed_epochs = max(self.completed_epochs, loaded_epoch)
        self.logger.info(
            "Loaded checkpoint from %s with signature %s at epoch %d",
            checkpoint_path,
            meta.get("sig", "unknown"),
            loaded_epoch,
        )
        return loaded_epoch

    def record_eval_losses(self, eval_losses: dict) -> None:
        self.eval_loss_history.append({str(k): float(v) for k, v in eval_losses.items()})
        save_combined_loss_history(self.train_loss_history, self.eval_loss_history, self.loss_history_path)

    def record_train_losses(self, train_losses: dict) -> None:
        self.train_loss_history.append({str(k): float(v) for k, v in train_losses.items()})
        save_combined_loss_history(self.train_loss_history, self.eval_loss_history, self.loss_history_path)

    def write_loss_plot(self) -> None:
        if not self.train_loss_history and not self.eval_loss_history:
            self.logger.info("Skipping loss plot because no loss history is available.")
            return

        loss_path = self.plots_dir / "loss.pdf"
        save_train_eval_loss_plot()
        self.logger.info("Saved loss plot to %s", loss_path)

    def forward_pass(self, batch: dict, examples_count: int, total_loss: float , rhos: list | None = None):
        ids = batch["ids"]
        attn = batch["attn_mask"]
        bs = ids.size(0)
        
        if rhos is None:
            rhos = self.rhos

        with torch.no_grad():
            tkns_embd = self.sent_encoder.token_embeddings(ids, attn)

        z, g, loss = self.model(ids, tkns_embd, attn, rhos=rhos)

        return (
            attn,
            z,
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
            _, _, _, _, examples_count, total_loss \
                = self.forward_pass(batch, examples_count, total_loss)

        total_loss /= max(1, examples_count)

        return {"eval_loss": total_loss}

    @torch.no_grad()
    def evaluate(
        self, soft: bool = False, selection_log: SelectionLog | None = None
    ) -> tuple[dict, list | None, list | None]:
        self.model.eval()

        examples_count = 0
        total_loss = 0.0

        counts_pred = [Counts(labels=self.labels_set) for _ in self.rhos]
        counts_gold = [Counts(labels=self.labels_set) for _ in self.rhos]

        for batch in tqdm(
            self.test_dl,
            desc="Eval",
            leave=False,
            dynamic_ncols=True,
            disable=self._tqdm_disabled,
            file=sys.stderr,
        ):
            batch = to_device(self.device, batch)

            attn, z, g, _, examples_count, total_loss \
                = self.forward_pass(batch, examples_count, total_loss, rhos=self.rhos)

            pred_mask = z if soft else g

            if self.labels_set is not None:
                labels = batch["labels"]
                flat_labels = [label for seq in labels for label in seq]
                flat_attn = attn.bool().view(-1).cpu()

                word_ids = batch.get("word_ids")
                if word_ids is not None:
                    flat_word_ids = word_ids.view(-1).cpu().tolist()
                    flat_labels = [
                        SPECIAL_TAG if (is_att and wid < 0 and lbl == "-100") else lbl
                        for lbl, is_att, wid in zip(flat_labels, flat_attn.tolist(), flat_word_ids)
                    ]

                for i, pred_mask_i in enumerate(pred_mask):
                    flat_preds = pred_mask_i.cpu().view(-1)
                    counts_pred[i] += Counts(flat_labels, flat_attn, flat_preds)
                    counts_gold[i] += Counts(flat_labels, flat_attn)

                if selection_log is not None:
                    selection_log.add_batch(
                        batch["ids"].view(-1).cpu().tolist(),
                        flat_labels,
                        flat_attn.tolist(),
                        pred_mask.reshape(len(self.rhos), -1),
                        seq_len=attn.shape[1],
                    )

        total_loss /= max(1, examples_count)

        if self.labels_set is not None:
            return {"eval_loss": total_loss}, counts_pred, counts_gold

        return {"eval_loss": total_loss}, None, None

    @torch.no_grad()
    def final_eval(self, record_eval_history: bool = True) -> None:
        if self.skip_eval:
            self.logger.info("Skipping full evaluation (runtime.eval.skip=true).")
            return

        selection_log = None
        if self.labels_set is not None:
            freq_table = build_token_frequency_table(self.train_dl.dataset["ids"])
            selection_log = SelectionLog(self.rhos, freq_table)

        # Section 5 counts use the hard top-k mask g_pi(rho) per the paper's
        # inference convention; evaluate(soft=True) would use the soft mask z.
        eval_losses, counts_pred, counts_gold = self.evaluate(
            soft=not bool(self.cfg.runtime.eval.get("hard", True)),
            selection_log=selection_log,
        )

        if record_eval_history:
            self.record_eval_losses(eval_losses)
        self.logger.info("Final evaluation: eval_loss=%.4f", eval_losses.get("eval_loss", 0.0))
        self.write_loss_plot()

        # Create data and plots directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Runs unconditionally, regardless of what cfg.data.dataset was used
        # to train -- this is what makes cross-corpus generalization (every
        # run gets STS-B-evaluated, whatever it trained on) possible without
        # any extra configuration.
        base, ours, rand = run_stsb_sweep(
            cfg=self.cfg,
            device=self.device,
            encoder=self.sent_encoder,
            tokenizer=self.tokenizer,
            selector=self.model,
        )
        stsb = {
            "base": float(base),
            "ours_by_rho": {str(float(k)): float(v) for k, v in ours.items()},
            "random_by_rho": {str(float(k)): float(v) for k, v in rand.items()},
        }

        artifact_paths = save_eval_artifacts(
            counts_pred=counts_pred,
            counts_gold=counts_gold,
            rhos=self.rhos,
            label_order=self.label_order,
            selection_log=selection_log,
            stsb=stsb,
        )
        for metric_name, artifact_path in artifact_paths.items():
            self.logger.info("Saved %s artifact to: %s", metric_name, artifact_path)

        if "exact_test_summary" in artifact_paths:
            with artifact_paths["exact_test_summary"].open("r", encoding="utf-8") as f:
                summary = json.load(f)
            z_critical = summary.get("z_critical")
            self.logger.info(
                "Exact test (Bonferroni, per-dataset): m=%d alpha=%.3g threshold=%.6g z_critical=%s",
                summary.get("m", 0),
                summary.get("alpha", 0.0),
                summary.get("threshold", 0.0),
                f"{z_critical:.4f}" if isinstance(z_critical, (int, float)) else "n/a",
            )

        plot_paths = save_eval_plots(artifact_paths.keys(), dataset_name=self.cfg.data.dataset)
        for metric_name, plot_path in plot_paths.items():
            self.logger.info("Saved %s plot to: %s", metric_name, plot_path)

    def train(self) -> None:
        start_epoch = 0
        
        if self.cfg.train.untrained:
            self.logger.info("No train run, only initializated model evaluation")
            self.final_eval()
            return

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

                # The reconstruction-loss forward+backward is the dominant training
                # cost and scales linearly with the number of rhos evaluated per
                # step, so a random subset is used here (full sweep still covers
                # every rho over the course of training via resampling). Eval
                # always uses the full sweep (self.rhos) via forward_pass's default.
                if self.rhos_per_step < len(self.rhos):
                    step_rho_idx = torch.randperm(len(self.rhos))[: self.rhos_per_step]
                    step_rhos = [self.rhos[i] for i in step_rho_idx.tolist()]
                else:
                    step_rhos = self.rhos

                _, _, _, loss, examples_count, total_loss \
                    = self.forward_pass(batch, examples_count, total_loss, rhos=step_rhos)

                loss.backward()
                self.optimizer.step()

            total_loss /= max(1, examples_count)
            
            train_losses = {"train_loss": total_loss}
            self.record_train_losses(train_losses)

            if self.skip_eval:
                eval_losses = {"eval_loss": float("nan")}
            else:
                eval_losses = self.short_evaluate()
                self.record_eval_losses(eval_losses)

            self.logger.info(
                "Epoch %d/%d | train_loss=%.4f | eval_loss=%.4f",
                epoch + 1,
                self.epochs,
                train_losses.get("train_loss", 0.0),
                eval_losses.get("eval_loss", 0.0),
            )
            self.completed_epochs = epoch + 1
            self.save_checkpoint(epoch + 1)
            # NaN placeholders (runtime.eval.skip=true) are dropped rather than
            # written out -- JSON has no NaN literal, and an absent key reads
            # more honestly than a null in `forge` output.
            self.run.push_log(
                {
                    k: float(v)
                    for k, v in {**train_losses, **eval_losses}.items()
                    if math.isfinite(float(v))
                },
                step=epoch + 1,
            )
            write_metrics_details(
                cfg=self.cfg,
                run=self.run,
                start_capture=self.start_capture,
                epochs_completed=self.completed_epochs,
                epochs_target=int(self.epochs),
                training_completed=False,
            )

        self.final_eval()


def main(cfg: DictConfig) -> int:
    start_capture = start_run_metrics_capture()

    # start_run() registers the run and chdirs into its output directory, so
    # it has to come before get_logger() -- otherwise train.log is opened
    # against the launch directory instead of this run's own.
    run = start_run(cfg)
    logger = get_logger()

    logger.info(f"Exp signature: {run.signature}")
    logger.info(repr(cfg))
    logger.info(f"Work dir: {os.getcwd()}")

    cfg.runtime, changed_device = configure_runtime(cfg.runtime)
    if changed_device:
        logger.warning("CUDA requested but unavailable, using CPU.")

    train_dl, test_dl, encoder, tokenizer, labels_set, _ = initialize_data(
        cfg.data,
        cfg.runtime.data,
        logger,
        device=cfg.runtime.device,
        keep_special=bool(cfg.model.get("keep_special", True)),
    )

    trainer = SelectorTrainer(
        cfg, train_dl, test_dl, encoder, tokenizer, labels_set, logger, run, start_capture
    )

    if cfg.train.no_train:
        # Resolved across the experiment's runs, not just this one: a no-train
        # eval re-run starts in a fresh, empty run directory, so the named
        # checkpoint lives in whichever sibling run actually trained it.
        checkpoint_name = str(cfg.train.checkpoint_path)
        candidates = sorted(run.experiment.path.glob(f"*/state/models/{checkpoint_name}"))
        if not candidates:
            raise FileNotFoundError(
                f"No {checkpoint_name} found in any run of experiment {run.experiment.signature}."
            )
        trainer.load_checkpoint(candidates[-1])
        # Falls through to run.finish() either way: skipping the eval is a
        # complete run, and returning early here would leave the run marked
        # "running" for forge's atexit handler to record as "failed".
        if trainer.skip_eval:
            logger.info("Skipping no-train evaluation (runtime.eval.skip=true).")
        else:
            trainer.final_eval(record_eval_history=False)
    else:
        trainer.train()
        trainer.completed_epochs = int(cfg.train.epochs)
        write_metrics_details(
            cfg=cfg,
            run=run,
            start_capture=start_capture,
            epochs_completed=trainer.completed_epochs,
            epochs_target=int(cfg.train.epochs),
            training_completed=True,
        )

    run.finish(build_final_metrics(
        trainer.eval_loss_history, start_capture, trainer.completed_epochs
    ))
    return 0
