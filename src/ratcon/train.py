import os, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import should_disable_tqdm, get_logger

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
        self.model_label = "model"
        self.model_path = f"{self.model_label}.pth"
        self.model = RationaleSelectorModel(cfg=self.cfg.model).to(self.cfg.device)
        self._load_or_initialize_model(self.model, self.model_path, self.model_label)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.model.optim.lr,
            weight_decay=self.cfg.model.optim.weight_decay,
            betas=self.cfg.model.optim.betas,
        )
        self._all_model_params = list(self.model.parameters())

    def _load_or_initialize_model(self, model, path, label):
        if model is None:
            return
        if os.path.exists(path) and (not self.cfg.train.retrain or self.cfg.eval.eval_only):
            self.logger.info(f"Loading {label} from {path}")
            state = torch.load(path, map_location=self.cfg.device)
            model.load_state_dict(state)
        else:
            self.logger.info(f"Training {label} from scratch")

    def train_epoch(self, loader, epoch):
        tau = self.cfg.model.loss.tau
        grad_clip = self.cfg.train.grad_clip
        device = self.cfg.device

        total = 0.0

        self.model.train()

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}", disable=should_disable_tqdm()):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            embeddings, attention_mask = batch["embeddings"], batch["attention_mask"]
            output = self.model(embeddings, attention_mask)

            loss = compute_training_objectives(
                self.model,
                output,
                attention_mask,
                self.cfg.model,
                temperature=tau,
                use_null_target=self.use_null_target,
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._all_model_params, grad_clip)
            self.optimizer.step()

            batch_size = embeddings.size(0)
            total += loss.item() * batch_size

        denom = max(1, len(loader.dataset))
        avg_loss = total / denom
        return avg_loss


    def make_report(self, model, eval_dl, tok, label, report_cfg):
        logging_cfg = self.cfg.logging
        disable_progress = should_disable_tqdm(metrics_only=logging_cfg.metrics_only)

        samples_cfg = report_cfg.samples
        samples_num = samples_cfg.num

        evaluation = evaluate(
            model,
            eval_dl,
            tok,
            self.cfg.eval.thresh,
            disable_progress=disable_progress,
            thresh=self.cfg.eval.thresh,
            samples_num=samples_num,
            logger=self.logger,
        )

        report = build_report(evaluation)

        return report, []


    def train(self, train_dl, eval_dl, tok, xp):
        best_f1 = float('-inf')
        best_epoch = None
        best_model_label = self.model_label
        best_report = None
        epoch_show_samples = bool(self.cfg.eval.report.epoch.samples.show)
        final_samples_cfg = self.cfg.eval.report.final.samples
        final_show_samples = bool(final_samples_cfg.show)

        for epoch in range(self.cfg.train.epochs):
            avg = self.train_epoch(train_dl, epoch)
            self.logger.info(f"epoch {epoch + 1}: train_loss={avg:.4f}")

            display_reports = {}
            with torch.no_grad():
                label = self.model_label
                report, _ = self.make_report(self.model, eval_dl, tok, label, self.cfg.eval.report.epoch)
                log_report(
                    self.logger,
                    report,
                    report_cfg=self.cfg.eval.report.epoch,
                    report_name=f"Epoch {epoch + 1} {label}",
                    show_samples=epoch_show_samples,
                )
                xp.link.push_metrics({f"eval/{epoch}/{label}/{self.cfg.data.eval.dataset}": report})
                display_reports[label] = report

            table = render_reports_table(
                display_reports,
                precision=4
            )
            self.logger.info(table)

            metrics = display_reports[self.model_label].get("metrics") or {}
            f1 = metrics.get('f1')
            if f1 is not None and f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch + 1
                best_model_label = self.model_label
                best_report = display_reports[self.model_label]
                self.logger.info(
                    f"New best F1: {best_f1:.4f}, saving model to {self.model_path}"
                )
                torch.save(self.model.state_dict(), self.model_path)

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

        return best_report, best_model_label


    def evaluate(self, eval_dl, tok):
        final_samples_cfg = self.cfg.eval.report.final.samples
        final_show_samples = bool(final_samples_cfg.show)

        label = self.model_label
        report, _ = self.make_report(self.model, eval_dl, tok, label, self.cfg.eval.report.final)
        display_reports = {label: report}

        log_report(
            self.logger,
            report,
            report_cfg=self.cfg.eval.report.final,
            report_name="Eval " + label,
            show_samples=final_show_samples,
        )

        table = render_reports_table(
            display_reports,
            precision=4
        )
        

        self.logger.info(table)



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
