import json
import math
import os
import sys
import torch

from collections import Counter
from logging import Logger
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from forge import start_run
from forge.core import ExperimentRun
from omegaconf import DictConfig
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report as sk_classification_report

from .sentence import SentenceEncoder
from .tagger import MLPTagger, gather_word_level
from .data import PAD_TAG, LABEL_DISPLAY_NAMES, canonical_name, initialize_data
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
from .view import save_train_eval_loss_plot


class NERTrainer:
    """Word-level per-token MLP NER probe on top of a frozen sentence encoder.

    Downstream validation for the selection-bias test: holds the probe
    architecture fixed and swaps the encoder, tracking precision/recall/F1
    per epoch so it can be compared against the bias-test results.
    """

    def __init__(
        self,
        cfg: DictConfig,
        train_dl: DataLoader,
        test_dl: DataLoader,
        sent_encoder: SentenceEncoder,
        tag_names: list[str],
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
        self.ner_report_history_path = self.data_dir / "ner_report_history.json"
        self.tag_names = tag_names
        # Label values come from the dataset's own raw label strings: for
        # ClassLabel-encoded datasets (wikiann/conll2003) these are already
        # stringified indices ("0","1",...); for conll2000 (no upstream
        # ClassLabel -- NLTK yields the raw tag string) they're the tag name
        # itself ("B-NP"). Handle both without needing int() to work.
        self.label_to_idx = {str(i): i for i in range(len(tag_names))}
        self.label_to_idx.update({name: i for i, name in enumerate(tag_names)})

        # Inverse-frequency class weights (sklearn's "balanced" formula:
        # total / (num_classes * count[c])), computed once from the
        # training label distribution. Off by default -- see ner.class_weighted
        # in conf/config.yaml. Guards against severe imbalance (e.g.
        # movie_rationales' ~5.4:1 skew) collapsing the probe to always
        # predicting the majority class, but the wikiann/conll2003 probes
        # reported in the paper were trained without it.
        ner_cfg = cfg.get("ner", {})
        self.class_weights = None
        if bool(ner_cfg.get("class_weighted", False)):
            counts = Counter()
            for seq in self.train_dl.dataset["labels"]:
                for v in seq:
                    if v != PAD_TAG:
                        counts[self.label_to_idx[v]] += 1
            total = sum(counts.values())
            num_tags = len(tag_names)
            self.class_weights = torch.tensor(
                [total / (num_tags * counts.get(i, 1)) for i in range(num_tags)],
                dtype=torch.float32,
                device=self.device,
            )

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

        self.model = MLPTagger(
            model_dim,
            num_tags=len(tag_names),
            hidden=int(ner_cfg.get("hidden", 256)),
            dropout=float(ner_cfg.get("dropout", 0.1)),
        ).to(self.device)

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
        self.ner_report_history: list[dict] = (
            json.loads(self.ner_report_history_path.read_text(encoding="utf-8"))
            if self.ner_report_history_path.exists()
            else []
        )

    def checkpoint_path(self, epoch: int) -> Path:
        return self.models_dir / f"model_{epoch}.pth"

    def _build_ner_reports(self, y_true: list[list[str]], y_pred: list[list[str]]) -> tuple[dict, dict, dict]:
        span_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        flat_true = [tag for seq in y_true for tag in seq]
        flat_pred = [tag for seq in y_pred for tag in seq]
        token_report = sk_classification_report(flat_true, flat_pred, output_dict=True, zero_division=0)
        # Binary "is this token part of any entity" (B-*/I-* collapsed to
        # one class vs O) -- needs the joint confusion matrix (e.g. a
        # B-PER predicted as B-ORG is a true positive here but wrong for
        # token_report), so it can't be derived from the per-tag reports
        # after the fact and has to be computed from the raw sequences.
        binarize = lambda tags: ["entity" if t != "O" else "O" for t in tags]
        binary_report = sk_classification_report(
            binarize(flat_true), binarize(flat_pred), output_dict=True, zero_division=0
        )
        return span_report, token_report, binary_report

    def record_ner_report(self, epoch: int, span_report: dict, token_report: dict, binary_report: dict) -> None:
        self.ner_report_history.append({
            "epoch": epoch,
            "span_level": span_report,
            "token_level": token_report,
            "binary_entity_level": binary_report,
        })
        self.ner_report_history_path.write_text(
            json.dumps(self.ner_report_history, indent=2, default=lambda o: o.item()), encoding="utf-8"
        )

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

        # Read from beside the checkpoint, which may be in a sibling run
        # directory -- see SelectorTrainer.load_checkpoint for the full note.
        self.train_loss_history, self.eval_loss_history = load_combined_loss_history(
            checkpoint_path.parents[2] / "data" / "loss_history.json"
        )

        loaded_epoch = int(meta.get("epoch", checkpoint_epoch(checkpoint_path) or 0))
        reports_path = checkpoint_path.parents[2] / "data/ner_report_history.json"
        reports = json.loads(reports_path.read_text()) if reports_path.exists() else []
        self.ner_report_history = [r for r in reports if r["epoch"] <= loaded_epoch]
        if reports_path.exists():
            self.ner_report_history_path.write_text(json.dumps(self.ner_report_history, indent=2))
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

    def _label_ids_tensor(self, labels: list[list[str]]) -> torch.Tensor:
        return torch.tensor(
            [[-1 if v == PAD_TAG else self.label_to_idx[v] for v in seq] for seq in labels],
            dtype=torch.long,
            device=self.device,
        )

    def forward_pass(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        label_ids = self._label_ids_tensor(batch["labels"])

        with torch.no_grad():
            tkns_embd = self.sent_encoder.token_embeddings(batch["ids"], batch["attn_mask"])

        word_emb, word_mask, word_labels = gather_word_level(tkns_embd, batch["word_ids"], label_ids)
        emissions = self.model(word_emb, word_mask)
        loss = self.model.loss(emissions, word_labels, word_mask, weight=self.class_weights)

        return emissions, word_mask, word_labels, loss

    @torch.no_grad()
    def evaluate(self) -> tuple[dict, list[list[str]], list[list[str]]]:
        self.model.eval()
        examples_count = 0
        total_loss = 0.0

        y_true: list[list[str]] = []
        y_pred: list[list[str]] = []

        for batch in tqdm(
            self.test_dl,
            desc="Eval",
            leave=False,
            dynamic_ncols=True,
            disable=self._tqdm_disabled,
            file=sys.stderr,
        ):
            batch = to_device(self.device, batch)
            emissions, word_mask, word_labels, loss = self.forward_pass(batch)

            bs = batch["ids"].size(0)
            examples_count += bs
            total_loss += loss.item() * bs

            decoded = self.model.decode(emissions, word_mask)
            lengths = word_mask.sum(dim=1).tolist()
            gold = word_labels.cpu().tolist()
            for i, seq_len in enumerate(lengths):
                y_true.append([self.tag_names[gold[i][t]] for t in range(seq_len)])
                y_pred.append([self.tag_names[tag] for tag in decoded[i]])

        total_loss /= max(1, examples_count)
        metrics = {
            "eval_loss": total_loss,
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        }
        return metrics, y_true, y_pred

    @torch.no_grad()
    def final_eval(self, record_eval_history: bool = True) -> None:
        if self.skip_eval:
            self.logger.info("Skipping full evaluation (runtime.eval.skip=true).")
            return

        eval_losses, y_true, y_pred = self.evaluate()
        if record_eval_history:
            self.record_eval_losses(eval_losses)
        self.logger.info(
            "Final evaluation: eval_loss=%.4f precision=%.4f recall=%.4f f1=%.4f",
            eval_losses["eval_loss"], eval_losses["precision"], eval_losses["recall"], eval_losses["f1"],
        )
        self.write_loss_plot()

        self.data_dir.mkdir(parents=True, exist_ok=True)
        span_report, token_report, binary_report = self._build_ner_reports(y_true, y_pred)
        report = {"span_level": span_report, "token_level": token_report, "binary_entity_level": binary_report}
        report_path = self.data_dir / "ner_classification_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=lambda o: o.item()), encoding="utf-8")
        self.logger.info("Saved NER classification report to %s", report_path)

    def train(self) -> None:
        start_epoch = 0

        if self.cfg.train.get("untrained", False):
            self.logger.info("No train run, only initialized model evaluation")
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

                _, _, _, loss = self.forward_pass(batch)

                loss.backward()
                self.optimizer.step()

                bs = batch["ids"].size(0)
                examples_count += bs
                total_loss += loss.item() * bs

            total_loss /= max(1, examples_count)

            train_losses = {"train_loss": total_loss}
            self.record_train_losses(train_losses)

            if self.skip_eval:
                eval_losses = {
                    "eval_loss": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                }
            else:
                eval_losses, y_true, y_pred = self.evaluate()
                self.record_eval_losses(eval_losses)
                span_report, token_report, binary_report = self._build_ner_reports(y_true, y_pred)
                self.record_ner_report(epoch + 1, span_report, token_report, binary_report)

            self.logger.info(
                "Epoch %d/%d | train_loss=%.4f | eval_loss=%.4f | precision=%.4f | recall=%.4f | f1=%.4f",
                epoch + 1,
                self.epochs,
                train_losses.get("train_loss", 0.0),
                eval_losses.get("eval_loss", 0.0),
                eval_losses.get("precision", 0.0),
                eval_losses.get("recall", 0.0),
                eval_losses.get("f1", 0.0),
            )
            self.completed_epochs = epoch + 1
            self.save_checkpoint(epoch + 1)
            # NaN placeholders (runtime.eval.skip=true) are dropped -- see the
            # matching note in SelectorTrainer.train.
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


def _resolve_ner_tag_names(dataset_name: str) -> list[str]:
    name = canonical_name(dataset_name)
    if name not in {"wikiann", "conll2003", "conll2000", "movie_rationales"}:
        raise ValueError(
            f"task=ner requires a per-token-labeled dataset (wikiann, "
            f"conll2003, conll2000, or movie_rationales), got dataset={dataset_name!r}."
        )
    tag_map = LABEL_DISPLAY_NAMES[name]
    return [tag_map[str(i)] for i in range(len(tag_map))]


def main(cfg: DictConfig) -> int:
    if cfg.task != "ner":
        raise ValueError("The probe entry point requires task=ner.")
    start_capture = start_run_metrics_capture()

    # start_run() chdirs into the run directory, so it must precede
    # get_logger() -- see the matching note in src/train.py's main.
    run = start_run(cfg)
    logger = get_logger()

    logger.info(f"Exp signature: {run.signature}")
    logger.info(repr(cfg))
    logger.info(f"Work dir: {os.getcwd()}")

    cfg.runtime, changed_device = configure_runtime(cfg.runtime)
    if changed_device:
        logger.warning("CUDA requested but unavailable, using CPU.")

    train_dl, test_dl, encoder, _, _, _ = initialize_data(
        cfg.data,
        cfg.runtime.data,
        logger,
        device=cfg.runtime.device,
        keep_special=bool(cfg.model.get("keep_special", True)),
    )

    tag_names = _resolve_ner_tag_names(cfg.data.dataset)
    trainer = NERTrainer(cfg, train_dl, test_dl, encoder, tag_names, logger, run, start_capture)

    if cfg.train.no_train:
        _, checkpoint = find_resume_checkpoint(run, str(cfg.train.checkpoint_path))
        trainer.load_checkpoint(checkpoint)
        # Falls through to run.finish() either way -- see src/train.py's main.
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
