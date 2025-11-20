import os, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import (
    should_disable_tqdm,
    get_logger,
    compute_training_objectives
)
from .metrics import log_report, make_report

from .models import RationaleSelectorModel
from .data import get_dataset, collate
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
    def __init__(self, cfg, logger, tok, xp):
        self.cfg = cfg
        self.logger = logger
        self.model_path = "model.pth"
        self.model = RationaleSelectorModel(cfg=self.cfg.model).to(self.cfg.device)
        self._load_or_initialize_model(self.model, self.model_path, "model")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.model.optim.lr,
            weight_decay=self.cfg.model.optim.weight_decay,
            betas=self.cfg.model.optim.betas,
        )
        self._all_model_params = list(self.model.parameters())
        self.tok = tok
        self.xp = xp

    def _load_or_initialize_model(self, model, path):
        if model is None:
            return
        if os.path.exists(path) and (not self.cfg.train.retrain or self.cfg.eval.eval_only):
            self.logger.info(f"Loading model from {path}")
            state = torch.load(path, map_location=self.cfg.device)
            model.load_state_dict(state)
        else:
            self.logger.info(f"Training model from scratch")
            
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
                output,
                attention_mask,
                self.cfg.model,
                temperature=tau,
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._all_model_params, grad_clip)
            self.optimizer.step()

            batch_size = embeddings.size(0)
            total += loss.item() * batch_size

        avg_loss = total / max(1, len(loader.dataset))
        return avg_loss        

    def train(self, train_dl, eval_dl):
        best_f1 = float('-inf')
        best_epoch = None
        best_report = None
        epoch_show_samples = bool(self.cfg.eval.report.epoch.samples.show)
        final_samples_cfg = self.cfg.eval.report.final.samples
        final_show_samples = bool(final_samples_cfg.show)

        for epoch in range(self.cfg.train.epochs):
            avg = self.train_epoch(train_dl, epoch)
            self.logger.info(f"epoch {epoch + 1}: train_loss={avg:.4f}")

            report = self.evaluate(
                eval_dl,
                self.tok,
                num_samples=final_samples_cfg.num if final_show_samples else 0,
                show_samples=epoch_show_samples,
            )
            self.xp.link.push_metrics({f"eval/{epoch}/{self.cfg.data.eval.dataset}": report})

            metrics = report["metrics"]
            f1 = metrics["f1"]
            if f1 is not None and f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch + 1
                best_report = report
                self.logger.info(
                    f"New best F1: {best_f1:.4f}, saving model to {self.model_path}"
                )
                torch.save(self.model.state_dict(), self.model_path)

        if best_report is not None:
            self.logger.info(f"Best Epoch {best_epoch} with F1 {best_f1:.4f}")
            log_report(
                self.logger,
                best_report,
                report_cfg=self.cfg.eval.report.final,
                report_name="Training Best",
                show_samples=final_show_samples,
            )
            self.xp.link.push_metrics({f"best_eval/{best_epoch}/{self.cfg.data.eval.dataset}": best_report})
        else:
            self.logger.info("No best report recorded; skipping checkpoint summary.")

        return best_report
    
    def _inference(self, data, tok, disable_progress):
        inf = []
        with torch.no_grad():
            for batch in tqdm(data, desc="Evaluating", disable=disable_progress):
                embeddings, attention_mask, input_ids = batch["embeddings"], batch["attention_mask"], batch["input_ids"]
                ner_tags = None if not "ner_tags" in batch else batch["ner_tags"]

                output = self.model(embeddings, attention_mask)
                gates_tensor = output["gates"]

                for i in range(embeddings.size(0)):
                    ids = input_ids[i].cpu().tolist()
                    tokens = tok.convert_ids_to_tokens(ids)
                    mask = attention_mask[i].cpu().tolist()
                    gates = gates_tensor[i].detach().cpu().tolist()

                    inf.append(
                        {
                            "ids": ids,
                            "tokens": tokens,
                            "mask": mask,
                            "gates": gates,
                            "gold": None if ner_tags is None else ner_tags[i].cpu().tolist(),
                        }
                    )
        return inf

    def evaluate(self, eval_dl, report_name="Evaluation", report_cfg=None):
        disable_progress = should_disable_tqdm()

        report = make_report(
            self.model,
            eval_dl,
            self.tok,
            self.cfg.eval.thresh,
            disable_progress=disable_progress,
            thresh=self.cfg.eval.thresh,
            num_samples=report_cfg.samples.num if report_cfg and report_cfg.samples.show else 0,
            logger=self.logger,
        )
    
        report.log_report(
            self.logger,
            report_cfg=report_cfg,
            report_name=report_name,
            show_samples=report_cfg.samples.show if report_cfg and report_cfg.samples else False,
        )

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
    trainer = Trainer(cfg, logger, tok, xp)
    if cfg.eval.eval_only:
        trainer.evaluate(eval_dl)
    else:
        trainer.train(train_dl, eval_dl)   
