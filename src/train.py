import os, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import (
    should_disable_tqdm,
    get_logger,
    resolve_sparsity_weights,
    collect_joint_samples,
    log_joint_samples,
)

from .models import RationaleSelectorModel
from .data import get_dataset, collate
from .losses import compute_training_objectives
from .evaluate import evaluate
from .metrics import (
    build_report,
    log_report,
    render_reports_table,
)
from dora import get_xp, hydra_main

# -------------------------------------------------------------------
# Torch setup
# -------------------------------------------------------------------
_slurm_threads = os.getenv("SLURM_CPUS_PER_TASK") or os.getenv("SLURM_CPUS_PER_GPU")
try:
    _slurm_threads = int(_slurm_threads) if _slurm_threads else 0
except ValueError:
    _slurm_threads = 0
_threads = _slurm_threads or (os.cpu_count() or 1)
torch.set_num_threads(max(1, _threads))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.use_null_target = cfg.model.loss.use_null_target
        self.num_models = self._determine_num_models()
        self.models, self.model_labels, self.model_paths, self.optimizer = self.build_models_and_optimizer()
        self._all_model_params = [param for model in self.models for param in model.parameters()]
        self.sparsity_weights = resolve_sparsity_weights(self.cfg.model, self.num_models, logger=self.logger)
        dual_cfg = self.cfg.model.dual
        self.use_kl = False
        self.kl_weight = 0.0
        self.use_mutual_info = False
        self.mutual_info_weight = 0.0
        if dual_cfg is not None:
            kl_cfg = dual_cfg.kl
            self.use_kl = bool(kl_cfg.use)
            self.kl_weight = float(kl_cfg.weight)

            mi_cfg = dual_cfg.mutual_info
            self.use_mutual_info = bool(mi_cfg.use)
            self.mutual_info_weight = float(mi_cfg.weight)
        self.overlap_threshold = float(self.cfg.eval.thresh)

    def _determine_num_models(self):
        dual_cfg = self.cfg.model.dual
        if dual_cfg is None or not dual_cfg.use:
            return 1
        num_models = int(dual_cfg.num)
        if num_models < 1:
            self.logger.warning("Configured num_models < 1; defaulting to 1.")
            return 1
        return num_models

    def _load_or_initialize_model(self, model, path, label):
        if model is None:
            return
        if os.path.exists(path) and (not self.cfg.train.retrain or self.cfg.eval.eval_only):
            self.logger.info(f"Loading {label} from {path}")
            state = torch.load(path, map_location=self.cfg.device)
            model.load_state_dict(state)
        else:
            self.logger.info(f"Training {label} from scratch")

    def _iter_models(self):
        for label, model in zip(self.model_labels, self.models):
            yield label, model

    def _iter_model_specs(self):
        for label, model in zip(self.model_labels, self.models):
            yield label, model, self.model_paths[label]

    def _format_checkpoint_names(self):
        paths = sorted(self.model_paths.values())
        if not paths:
            return ""
        if len(paths) == 1:
            return paths[0]
        if len(paths) == 2:
            return " and ".join(paths)
        return ", ".join(paths[:-1]) + f", and {paths[-1]}"

    def build_models_and_optimizer(self):
        labels = ["model"] if self.num_models == 1 else [f"model{i + 1}" for i in range(self.num_models)]
        models = []
        specs = []

        for label in labels:
            model = RationaleSelectorModel(cfg=self.cfg.model).to(self.cfg.device)
            path = f"{label}.pth"
            self._load_or_initialize_model(model, path, label)
            models.append(model)
            specs.append((label, model, path))

        params = [param for _, model, _ in specs for param in model.parameters()]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.model.optim.lr,
            weight_decay=self.cfg.model.optim.weight_decay,
            betas=self.cfg.model.optim.betas,
        )

        self.model_paths = {label: path for label, _, path in specs}
        return models, labels, self.model_paths, optimizer

    def train_epoch(self, loader, epoch):
        tau = self.cfg.model.loss.tau
        grad_clip = self.cfg.train.grad_clip
        device = self.cfg.device

        total = 0.0
        total_kl_loss = 0.0
        kl_present = False
        total_mi_loss = 0.0
        mi_present = False
        total_overlap = 0.0
        overlap_present = False

        for model in self.models:
            model.train()

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}", disable=should_disable_tqdm()):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            embeddings, attention_mask = batch["embeddings"], batch["attention_mask"]
            incoming = batch["incoming"] if self.cfg.model.attention_augment else None
            outgoing = batch["outgoing"] if self.cfg.model.attention_augment else None

            outputs = [
                model(embeddings, attention_mask, incoming, outgoing)
                for model in self.models
            ]

            loss, kl_value, mi_value, overlap_fraction = compute_training_objectives(
                self.models,
                outputs,
                attention_mask,
                self.cfg.model,
                self.sparsity_weights,
                temperature=tau,
                use_null_target=self.use_null_target,
                use_kl=self.use_kl,
                kl_weight=self.kl_weight,
                use_mutual_info=self.use_mutual_info,
                mutual_info_weight=self.mutual_info_weight,
                overlap_threshold=self.overlap_threshold,
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._all_model_params, grad_clip)
            self.optimizer.step()

            batch_size = embeddings.size(0)
            total += loss.item() * batch_size
            if kl_value is not None:
                total_kl_loss += kl_value.item() * batch_size
                kl_present = True
            if mi_value is not None:
                total_mi_loss += mi_value.item() * batch_size
                mi_present = True
            if overlap_fraction is not None:
                total_overlap += overlap_fraction * batch_size
                overlap_present = True

        denom = max(1, len(loader.dataset))
        avg_loss = total / denom
        avg_kl = (total_kl_loss / denom) if kl_present else None
        avg_mi = (total_mi_loss / denom) if mi_present else None
        avg_overlap = (total_overlap / denom) if overlap_present else None
        return avg_loss, avg_kl, avg_mi, avg_overlap


    def make_report(self, model, eval_dl, tok, label, report_cfg):
        logging_cfg = self.cfg.logging
        disable_progress = should_disable_tqdm(metrics_only=logging_cfg.metrics_only)

        samples_cfg = report_cfg.samples
        samples_num = samples_cfg.num

        partition_cfg = None
        if self.num_models == 1:
            candidate = self.cfg.eval.partition_dual
            if candidate.use:
                partition_cfg = candidate

        reference_cfg = self.cfg.eval.reference_sentence
        reference_sentence = None
        reference_threshold = reference_cfg.threshold
        if reference_cfg.use:
            reference_sentence = reference_cfg.sentence
            if reference_sentence is None:
                self.logger.warning(
                    "reference_sentence.use is True but no sentence provided; skipping reference filtering."
                )

        evaluation = evaluate(
            model,
            eval_dl,
            tok,
            self.cfg.eval.thresh,
            disable_progress=disable_progress,
            attention_augment=self.cfg.model.attention_augment,
            thresh=self.cfg.eval.thresh,
            samples_num=samples_num,
            spacy_model=self.cfg.eval.spacy_model,
            logger=self.logger,
            partition_cfg=partition_cfg,
            reference_sentence=reference_sentence,
            reference_threshold=reference_threshold,
        )

        report = build_report(evaluation)

        partition_reports = []
        for idx, part_eval in enumerate(evaluation.get("partitions") or []):
            base_label = part_eval.get("label") or f"partition_{idx + 1}"
            part_label = f"{label}_{base_label}"
            part_report = build_report(part_eval)
            partition_reports.append((part_label, part_report))

        return report, partition_reports


    def train(self, train_dl, eval_dl, tok, xp):
        best_f1 = float('-inf')
        best_epoch = None
        best_model_label = self.model_labels[0]
        best_report = None
        epoch_samples_cfg = self.cfg.eval.report.epoch.samples
        epoch_show_samples = bool(epoch_samples_cfg.show)
        final_samples_cfg = self.cfg.eval.report.final.samples
        final_show_samples = bool(final_samples_cfg.show)

        for epoch in range(self.cfg.train.epochs):
            avg, avg_kl, avg_mi, avg_overlap = self.train_epoch(train_dl, epoch)
            log_msg = f"epoch {epoch + 1}: train_loss={avg:.4f}"
            if avg_kl is not None:
                log_msg += f", kl_loss={avg_kl:.4f}"
            if avg_mi is not None:
                log_msg += f", mi_loss={avg_mi:.4f}"
            if avg_overlap is not None:
                log_msg += f", overlap={avg_overlap * 100:.2f}%"
            self.logger.info(log_msg)

            reports = {}
            display_reports = {}
            with torch.no_grad():
                for label, model in self._iter_models():
                    report, partition_reports = self.make_report(model, eval_dl, tok, label, self.cfg.eval.report.epoch)
                    log_report(
                        self.logger,
                        report,
                        report_cfg=self.cfg.eval.report.epoch,
                        report_name=f"Epoch {epoch + 1} {label}",
                        show_samples=epoch_show_samples,
                    )
                    xp.link.push_metrics({f"eval/{epoch}/{label}/{self.cfg.data.eval.dataset}": report})
                    reports[label] = report
                    display_reports[label] = report

                    for part_label, part_report in partition_reports:
                        log_report(
                            self.logger,
                            part_report,
                            report_cfg=self.cfg.eval.report.epoch,
                            report_name=f"Epoch {epoch + 1} {part_label}",
                            show_samples=epoch_show_samples,
                        )
                        xp.link.push_metrics({f"eval/{epoch}/{part_label}/{self.cfg.data.eval.dataset}": part_report})
                        display_reports[part_label] = part_report

            table = render_reports_table(
                display_reports,
                precision=4
            )
            self.logger.info(table)

            joint_samples = collect_joint_samples(reports, self.model_labels, epoch_samples_cfg)
            if joint_samples:
                log_joint_samples(self.logger, joint_samples, self.model_labels, f"Epoch {epoch + 1} joint samples:")

            best_label_epoch = None
            best_f1_epoch = float('-inf')
            for label, report in reports.items():
                metrics = report.get("metrics") or {}
                f1 = metrics.get('f1')
                if f1 is None:
                    continue

                if f1 > best_f1_epoch:
                    best_f1_epoch = f1
                    best_label_epoch = label

            if best_label_epoch is not None and best_f1_epoch > best_f1:
                best_f1 = best_f1_epoch
                best_epoch = epoch + 1
                best_model_label = best_label_epoch
                best_report = reports[best_model_label]
                self.logger.info(
                    f"New best F1: {best_f1:.4f}, saving model(s) to {self._format_checkpoint_names()}"
                )
                for _, model, path in self._iter_model_specs():
                    torch.save(model.state_dict(), path)

        if best_report is not None:
            self.logger.info(f"Best model: {best_model_label} at epoch {best_epoch} with F1 {best_f1:.4f}")
            self.logger.info(f"Best report: {best_report}")
            log_report(
                self.logger,
                best_report,
                report_cfg=self.cfg.eval.report.final,
                report_name="Final best " + best_model_label,
                show_samples=final_show_samples,
            )
            xp.link.push_metrics({f"best_eval/{best_epoch}/{best_model_label}/{self.cfg.data.eval.dataset}": best_report})
        else:
            self.logger.info("No best report recorded; skipping checkpoint summary.")
            
        joint_samples = collect_joint_samples(reports, self.model_labels, final_samples_cfg)
        if joint_samples:
            log_joint_samples(self.logger, joint_samples, self.model_labels, "Evaluation joint samples:")

        return best_report, best_model_label


    def evaluate(self, eval_dl, tok):
        final_samples_cfg = self.cfg.eval.report.final.samples
        final_show_samples = bool(final_samples_cfg.show)

        reports = {}
        display_reports = {}
        for label, model in self._iter_models():
            report, partition_reports = self.make_report(model, eval_dl, tok, label, self.cfg.eval.report.final)
            reports[label] = report
            display_reports[label] = report

            log_report(
                self.logger,
                report,
                report_cfg=self.cfg.eval.report.final,
                report_name="Eval " + label,
                show_samples=final_show_samples,
            )

            for part_label, part_report in partition_reports:
                display_reports[part_label] = part_report

        table = render_reports_table(
            display_reports,
            precision=4
        )
        

        self.logger.info(table)

        joint_samples = collect_joint_samples(reports, self.model_labels, final_samples_cfg)
        if joint_samples:
            log_joint_samples(self.logger, joint_samples, self.model_labels, "Evaluation joint samples:")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
@hydra_main(config_path="conf", config_name="default", version_base="1.1")
def main(cfg):
    logger = get_logger("train.log")
    xp = get_xp()
    logger.info(f"Exp signature: {xp.sig}")
    logger.info(repr(cfg))
    logger.info(f"Work dir: {os.getcwd()}")
    logger.info(f"Exec file: {__file__}")

    # Device setup
    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("No GPU available, switching to CPU")
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"

    # Data
    train_ds, tok = get_dataset(
        name=cfg.data.train.dataset,
        subset=cfg.data.train.subset,
        rebuild=cfg.data.rebuild_ds,
        shuffle=cfg.data.train.shuffle,
    )
    eval_shuffle = bool(cfg.data.eval.shuffle)
    if eval_shuffle:
        logger.warning("Disabling shuffle for evaluation loader to preserve ordering across models.")
        eval_shuffle = False

    eval_ds, _ = get_dataset(
        split="validation",
        name=cfg.data.eval.dataset,
        subset=cfg.data.eval.subset,
        rebuild=cfg.data.rebuild_ds,
        shuffle=eval_shuffle,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.data.train.batch_size,
        collate_fn=collate,
        num_workers=cfg.data.train.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.data.train.num_workers > 0),
        shuffle=cfg.data.train.shuffle,
    )

    eval_dl = DataLoader(
        eval_ds,
        batch_size=cfg.data.eval.batch_size,
        collate_fn=collate,
        num_workers=cfg.data.eval.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.data.eval.num_workers > 0),
        shuffle=eval_shuffle,
    )

    # Train or eval
    trainer = Trainer(cfg, logger)
    if cfg.eval.eval_only:
        trainer.evaluate(eval_dl, tok)
    else:
        trainer.train(train_dl, eval_dl, tok, xp)   
