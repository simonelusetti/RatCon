import os, torch, datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import (
    should_disable_tqdm,
    get_logger,
)

from .models import RationaleSelectorModel, nt_xent
from .data import get_dataset, collate
from .losses import complement_loss, kl_loss, sparsity_loss, total_variation_1d
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
        self.use_null_target = getattr(cfg.model.loss, "use_null_target", False)
        self.num_models = self._determine_num_models()
        self.models, self.model_labels, self.model_paths, self.optimizer = self.build_models_and_optimizer()
        self._all_model_params = [param for model in self.models for param in model.parameters()]
        self.sparsity_weights = self._resolve_sparsity_weights()
        dual_cfg = getattr(self.cfg.model, "dual", None)
        self.kl_weight = float(getattr(dual_cfg, "kl_weight", 0.0)) if dual_cfg is not None else 0.0

    def _determine_num_models(self):
        dual_cfg = getattr(self.cfg.model, "dual", None)
        if dual_cfg is None or not getattr(dual_cfg, "use", False):
            return 1
        num_models = int(getattr(dual_cfg, "num_models", 2))
        if num_models < 1:
            self.logger.warning("Configured num_models < 1; defaulting to 1.")
            return 1
        return num_models

    def _resolve_sparsity_weights(self):
        base = float(self.cfg.model.loss.l_s)
        if self.num_models == 1:
            return [base]

        dual_cfg = getattr(self.cfg.model, "dual", None)
        weights = []

        if dual_cfg is not None:
            configured = getattr(dual_cfg, "sparsity_weights", None)
            if configured is not None:
                configured = list(configured)
                if len(configured) == 1:
                    configured = configured * self.num_models
                if len(configured) == self.num_models:
                    return [float(w) for w in configured]
                self.logger.warning(
                    "sparsity_weights length does not match num_models; falling back to ls_1, ls_2, ..."
                )

        for idx in range(self.num_models):
            attr = f"ls_{idx + 1}"
            value = base
            if dual_cfg is not None and hasattr(dual_cfg, attr):
                value = float(getattr(dual_cfg, attr))
            weights.append(float(value))

        return weights

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

    def _compute_shared_embeddings(self, outputs, attention_mask):
        gates = [out["gates"] for out in outputs]
        complement_product = torch.ones_like(gates[0])
        for g in gates:
            complement_product = complement_product * (1.0 - g)

        shared_gate = 1.0 - complement_product
        shared_gate = torch.clamp(shared_gate, 1e-6, 1.0)
        shared_mask = attention_mask * shared_gate

        shared_comp_gate = torch.clamp(complement_product, 1e-6, 1.0)
        shared_comp_mask = attention_mask * shared_comp_gate

        shared_rationales = []
        shared_complements = []
        for model, out in zip(self.models, outputs):
            pooled_rat = model.pooler({
                "token_embeddings": out["token_embeddings"],
                "attention_mask": shared_mask,
            })["sentence_embedding"]
            pooled_comp = model.pooler({
                "token_embeddings": out["token_embeddings"],
                "attention_mask": shared_comp_mask,
            })["sentence_embedding"]
            if hasattr(model, "fourier"):
                pooled_rat = model.fourier(pooled_rat)
                pooled_comp = model.fourier(pooled_comp)
            shared_rationales.append(pooled_rat)
            shared_complements.append(pooled_comp)

        return shared_rationales, shared_complements

    def _compute_losses(self, outputs, attention_mask, tau):
        anchors = [out["h_anchor"] for out in outputs]
        device = anchors[0].device
        loss = anchors[0].new_zeros(())

        shared_rats, shared_comps = self._compute_shared_embeddings(outputs, attention_mask)

        for idx, out in enumerate(outputs):
            gates = out["gates"]
            null_vec = out["null"] if self.use_null_target else None

            l_rat = nt_xent(shared_rats[idx], anchors[idx], temperature=tau)
            l_comp = complement_loss(shared_comps[idx], anchors[idx], null_vec, self.use_null_target, tau)
            l_s = sparsity_loss(gates, attention_mask)
            l_tv = total_variation_1d(gates, attention_mask)

            loss = loss + l_rat
            loss = loss + self.cfg.model.loss.l_comp * l_comp
            loss = loss + self.sparsity_weights[idx] * l_s
            loss = loss + self.cfg.model.loss.l_tv * l_tv

        kl_total = anchors[0].new_zeros(())
        pair_count = 0
        if self.num_models > 1 and self.kl_weight != 0.0:
            for i in range(self.num_models):
                for j in range(i + 1, self.num_models):
                    out_i = outputs[i]
                    out_j = outputs[j]
                    kl_val = kl_loss(
                        out_i["gates"],
                        out_j["gates"],
                        out_i["alpha"],
                        out_i["beta"],
                        out_j["alpha"],
                        out_j["beta"],
                    )
                    kl_total = kl_total + kl_val
                    pair_count += 1

        avg_kl = None
        if pair_count > 0:
            avg_kl = kl_total / pair_count
            loss = loss + self.kl_weight * avg_kl

        return loss, avg_kl

    def train_epoch(self, loader, epoch):
        tau = self.cfg.model.loss.tau
        grad_clip = self.cfg.train.grad_clip
        device = self.cfg.device

        total = 0.0
        total_kl_loss = 0.0
        kl_present = False

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

            loss, kl_value = self._compute_losses(outputs, attention_mask, tau)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._all_model_params, grad_clip)
            self.optimizer.step()

            batch_size = embeddings.size(0)
            total += loss.item() * batch_size
            if kl_value is not None:
                total_kl_loss += kl_value.item() * batch_size
                kl_present = True

        denom = max(1, len(loader.dataset))
        avg_loss = total / denom
        avg_kl = (total_kl_loss / denom) if kl_present else None
        return avg_loss, avg_kl


    def _fit_cluster_filter(self, model, train_dl, label):
        model.fit_cluster_filter_from_loader(
            train_dl,
            self.cfg.model.clustering,
            logger=self.logger,
            label=label,
        )
        
        
    def make_report(self, model, eval_dl, tok, label, report_cfg):
        logging_cfg = getattr(self.cfg, "logging", None)
        disable_progress = should_disable_tqdm(
            metrics_only=getattr(logging_cfg, "metrics_only", False) if logging_cfg is not None else False
        )

        samples_cfg = getattr(report_cfg, "samples", None)
        samples_num = getattr(samples_cfg, "num", 0) if samples_cfg is not None else 0

        partition_cfg = None
        if self.num_models == 1:
            candidate = getattr(self.cfg.eval, "partition_dual", None)
            if candidate is not None and getattr(candidate, "use", False):
                partition_cfg = candidate

        reference_cfg = getattr(self.cfg.eval, "reference_sentence", None)
        reference_sentence = None
        reference_threshold = 0.5
        if reference_cfg is not None:
            reference_threshold = getattr(reference_cfg, "threshold", 0.5)
            if getattr(reference_cfg, "use", False):
                reference_sentence = getattr(reference_cfg, "sentence", None)
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
            attention_augment=getattr(self.cfg.model, "attention_augment", False),
            thresh=self.cfg.eval.thresh,
            samples_num=samples_num,
            spacy_model=getattr(self.cfg.eval, "spacy_model", "en_core_web_sm"),
            logger=self.logger,
            partition_cfg=partition_cfg,
            reference_sentence=reference_sentence,
            reference_threshold=reference_threshold,
        )

        cluster_info = None
        clustering_cfg = getattr(self.cfg.model, "clustering", None)
        if clustering_cfg and getattr(clustering_cfg, "use", False):
            cluster_info = model.get_cluster_info()

        report = build_report(evaluation, cluster_info)

        partition_reports = []
        for idx, part_eval in enumerate(evaluation.get("partitions") or []):
            base_label = part_eval.get("label") or f"partition_{idx + 1}"
            part_label = f"{label}_{base_label}"
            part_report = build_report(part_eval, None)
            partition_reports.append((part_label, part_report))

        return report, partition_reports


    def train(self, train_dl, eval_dl, tok, xp):
        best_f1 = float('-inf')
        best_epoch = None
        best_model_label = self.model_labels[0]
        best_report = None

        for epoch in range(self.cfg.train.epochs):
            avg, avg_kl = self.train_epoch(train_dl, epoch)
            if avg_kl is not None:
                self.logger.info(f"epoch {epoch + 1}: train_loss={avg:.4f}, kl_loss={avg_kl:.4f}")
            else:
                self.logger.info(f"epoch {epoch + 1}: train_loss={avg:.4f}")

            reports = {}
            display_reports = {}
            with torch.no_grad():
                for label, model in self._iter_models():
                    if self.cfg.model.clustering.use:
                        self._fit_cluster_filter(model, train_dl, label)
                    report, partition_reports = self.make_report(model, eval_dl, tok, label, self.cfg.eval.report.epoch)
                    log_report(self.logger, report, report_cfg=self.cfg.eval.report.epoch, report_name=f"Epoch {epoch + 1} {label}")
                    xp.link.push_metrics({f"eval/{epoch}/{label}/{self.cfg.data.eval.dataset}": report})
                    reports[label] = report
                    display_reports[label] = report

                    for part_label, part_report in partition_reports:
                        log_report(
                            self.logger,
                            part_report,
                            report_cfg=self.cfg.eval.report.epoch,
                            report_name=f"Epoch {epoch + 1} {part_label}",
                        )
                        xp.link.push_metrics({f"eval/{epoch}/{part_label}/{self.cfg.data.eval.dataset}": part_report})
                        display_reports[part_label] = part_report

            table = render_reports_table(
                display_reports,
                eval_cfg=self.cfg.eval,
                precision=4,
            )
            self.logger.info(table)

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
            log_report(self.logger, best_report, report_cfg=self.cfg.eval.report.final, report_name="Final best "+best_model_label)
            xp.link.push_metrics({f"best_eval/{best_epoch}/{best_model_label}/{self.cfg.data.eval.dataset}": best_report})
        else:
            self.logger.info("No best report recorded; skipping checkpoint summary.")
        
        return best_report, best_model_label


    def evaluate(self, eval_dl, tok):
        reports = {}
        display_reports = {}
        for label, model in self._iter_models():
            report, partition_reports = self.make_report(model, eval_dl, tok, label, self.cfg.eval.report.final)
            reports[label] = report
            display_reports[label] = report

            for part_label, part_report in partition_reports:
                display_reports[part_label] = part_report

        table = render_reports_table(
            display_reports,
            eval_cfg=self.cfg.eval,
            precision=4,
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
    eval_ds, _ = get_dataset(
        split="validation",
        name=cfg.data.eval.dataset,
        subset=cfg.data.eval.subset,
        rebuild=cfg.data.rebuild_ds,
        shuffle=cfg.data.eval.shuffle,
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
        shuffle=cfg.data.eval.shuffle,
    )

    # Train or eval
    trainer = Trainer(cfg, logger)
    if cfg.eval.eval_only:
        trainer.evaluate(eval_dl, tok)
    else:
        trainer.train(train_dl, eval_dl, tok, xp)   
