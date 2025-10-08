import os, torch, datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import (
    should_disable_tqdm,
    get_logger,
    shared_distribution,
    shared_complement_distribution,
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
        self.use_dual = cfg.model.dual.use
        self.use_null_target = getattr(cfg.model.loss, "use_null_target", False)
        self.model1, self.model2, self.optimizer = self.build_models_and_optimizer()

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
        if self.use_dual and self.model2 is not None:
            yield "model1", self.model1
            yield "model2", self.model2
        else:
            yield "model", self.model1

    def _iter_model_specs(self):
        for label, model in self._iter_models():
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
        model1 = RationaleSelectorModel(cfg=self.cfg.model).to(self.cfg.device)
        model2 = RationaleSelectorModel(cfg=self.cfg.model).to(self.cfg.device) if self.use_dual else None

        if self.use_dual:
            specs = [
                ("model1", model1, "model1.pth"),
                ("model2", model2, "model2.pth"),
            ]
            params = list(model1.parameters()) + list(model2.parameters())
        else:
            specs = [("model", model1, "model.pth")]
            params = model1.parameters()

        for label, model, path in specs:
            self._load_or_initialize_model(model, path, label)

        self.model_paths = {label: path for label, _, path in specs}

        optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.model.optim.lr,
            weight_decay=self.cfg.model.optim.weight_decay,
            betas=self.cfg.model.optim.betas,
        )

        return model1, model2, optimizer

    def train_dual(self, embeddings, attention_mask, incoming, outgoing):
        # Params
        l_comp, l_tv = self.cfg.model.loss.l_comp, self.cfg.model.loss.l_tv,
        ls_1 = self.cfg.model.dual.ls_1
        ls_2 = self.cfg.model.dual.ls_2
        l_kl = self.cfg.model.dual.kl_weight
        tau = self.cfg.model.loss.tau
        
        # Forward pass
        out1 = self.model1(embeddings, attention_mask, incoming, outgoing)
        h_a1, _, h_c1, g1 = out1["h_anchor"], out1["h_rat"], out1["h_comp"], out1["gates"]
        null_vec1 = out1["null"] if self.use_null_target else None
        token_emb1 = out1["token_embeddings"]

        out2 = self.model2(embeddings, attention_mask, incoming, outgoing)
        h_a2, _, h_c2, g2 = out2["h_anchor"], out2["h_rat"], out2["h_comp"], out2["gates"]
        null_vec2 = out2["null"] if self.use_null_target else None
        token_emb2 = out2["token_embeddings"]

        h_shared_rat1, h_shared_rat2, shared_mask = shared_distribution(
            g1, g2, token_emb1, token_emb2, attention_mask, self.model1, self.model2
        )

        h_shared_comp1, h_shared_comp2, shared_comp_mask = shared_complement_distribution(
            g1, g2, token_emb1, token_emb2, attention_mask, self.model1, self.model2,
        )
        
        L_shared_comp1 = complement_loss(h_shared_comp1, h_a1, null_vec1, self.use_null_target, tau)

        # Losses
        L_s1 = sparsity_loss(g1, attention_mask)
        L_tv1 = total_variation_1d(g1, attention_mask)

        L_s2 = sparsity_loss(g2, attention_mask)
        L_tv2 = total_variation_1d(g2, attention_mask)

        L_rat1 = nt_xent(h_shared_rat1, h_a1, temperature=tau)
        #L_rat2 = nt_xent(h_shared_rat_2, h_a2, temperature=tau)

        # Symmetric KL loss that pushes gates toward complementary behaviour
        alpha1, beta1 = out1["alpha"], out1["beta"]
        alpha2, beta2 = out2["alpha"], out2["beta"]
        L_kl = kl_loss(g1, g2, alpha1, beta1, alpha2, beta2)
        
        loss = L_rat1 + l_comp * L_shared_comp1 + \
            ls_1 * L_s1 + l_tv * L_tv1 \
            + ls_2 * L_s2 + l_tv * L_tv2 \
            + l_kl * L_kl

        #loss = L_rat1 + l_comp * L_shared_comp1 + ls_1 * L_s1 + l_tv * L_tv1 + + kl_weight * kl_loss \
        #    + (L_rat2 + l_comp * L_comp2 + ls_2 * L_s2 + l_tv * L_tv2)

        return loss, L_kl

    def train_epoch(self, loader, epoch):
        tau = self.cfg.model.loss.tau
        l_comp, l_s, l_tv = (
            self.cfg.model.loss.l_comp,
            self.cfg.model.loss.l_s,
            self.cfg.model.loss.l_tv,
        )
        grad_clip = self.cfg.train.grad_clip
        device = self.cfg.device

        total = 0.0
        total_kl_loss = 0.0
        self.model1.train()
        if self.model2:
            self.model2.train()

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}", disable=should_disable_tqdm()):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            embeddings, attention_mask = batch["embeddings"], batch["attention_mask"]
            incoming = batch["incoming"] if self.cfg.model.attention_augment else None
            outgoing = batch["outgoing"] if self.cfg.model.attention_augment else None

            # Forward pass model1
            out1 = self.model1(embeddings, attention_mask, incoming, outgoing)
            h_a1, h_r1, h_c1, g1 = out1["h_anchor"], out1["h_rat"], out1["h_comp"], out1["gates"]

            null_vec1 = out1["null"] if self.use_null_target else None
            L_comp1 = complement_loss(h_c1, h_a1, null_vec1, self.use_null_target, tau)
            L_s1 = sparsity_loss(g1, attention_mask)
            L_tv1 = total_variation_1d(g1, attention_mask)

            if self.model2:
                loss, L_kl = self.train_dual(embeddings, attention_mask, incoming, outgoing)
                params = list(self.model1.parameters()) + list(self.model2.parameters())
                total_kl_loss += L_kl.item() * embeddings.size(0)
            else:
                L_rat1 = nt_xent(h_r1, h_a1, temperature=tau)
                loss = L_rat1 + l_comp * L_comp1 + l_s * L_s1 + l_tv * L_tv1
                params = self.model1.parameters()

            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            self.optimizer.step()

            total += loss.item() * embeddings.size(0)

        return total / len(loader.dataset), total_kl_loss / len(loader.dataset) if self.model2 else None

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
        )

        cluster_info = None
        clustering_cfg = getattr(self.cfg.model, "clustering", None)
        if clustering_cfg and getattr(clustering_cfg, "use", False):
            cluster_info = model.get_cluster_info()

        return build_report(evaluation, cluster_info)


    def train(self, train_dl, eval_dl, tok, xp):
        best_f1 = float('-inf')
        best_epoch = None
        best_model_label = "model"
        best_report = None

        for epoch in range(self.cfg.train.epochs):
            avg, avg_kl = self.train_epoch(train_dl, epoch)
            if avg_kl is not None:
                self.logger.info(f"epoch {epoch + 1}: train_loss={avg:.4f}, kl_loss={avg_kl:.4f}")
            else:
                self.logger.info(f"epoch {epoch + 1}: train_loss={avg:.4f}")

            reports = {}
            with torch.no_grad():
                for label, model in self._iter_models():
                    if self.cfg.model.clustering.use:
                        self._fit_cluster_filter(model, train_dl, label)
                    report = self.make_report(model, eval_dl, tok, label, self.cfg.eval.report.epoch)
                    log_report(self.logger, report, report_cfg=self.cfg.eval.report.epoch, report_name=f"Epoch {epoch + 1} {label}")
                    xp.link.push_metrics({f"eval/{epoch}/{label}/{self.cfg.data.eval.dataset}": report})
                    reports[label] = report

            table = render_reports_table(
                reports,
                eval_cfg=self.cfg.eval,
                precision=4,
            )
            self.logger.info(table)

            best_label_epoch = None
            best_f1_epoch = float('-inf')
            for label, report in reports.items():
                f1 = report.get("metrics").get('f1')

                if f1 > best_f1_epoch:
                    best_f1_epoch = f1
                    best_label_epoch = label

            if best_f1_epoch > best_f1:
                best_f1 = best_f1_epoch
                best_epoch = epoch + 1
                best_model_label = best_label_epoch
                best_report = reports[best_model_label]
                self.logger.info(
                    f"New best F1: {best_f1:.4f}, saving model(s) to {self._format_checkpoint_names()}"
                )
                for _, model, path in self._iter_model_specs():
                    torch.save(model.state_dict(), path)

        self.logger.info(f"Best model: {best_model_label} at epoch {best_epoch} with F1 {best_f1:.4f}")
        self.logger.info(f"Best report: {best_report}")
        log_report(self.logger, best_report, report_cfg=self.cfg.eval.report.final, report_name="Final best "+best_model_label)
        xp.link.push_metrics({f"best_eval/{best_epoch}/{best_model_label}/{self.cfg.data.eval.dataset}": best_report})
        
        return best_report, best_model_label


    def evaluate(self, eval_dl, tok):
        reports = {}
        for label, model in self._iter_models():
            report = self.make_report(model, eval_dl, tok, label, self.cfg.eval.report.final)
            reports[label] = report

        table = render_reports_table(
            reports,
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
