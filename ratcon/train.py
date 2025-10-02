import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import should_disable_tqdm, get_logger

from .models import RationaleSelectorModel, nt_xent
from .data import get_dataset, collate
from .losses import complement_loss, kumaraswamy_log_pdf, sparsity_loss, total_variation_1d
from .evaluate import evaluate
from .metrics import (
    EvaluationResult,
    build_evaluation_payload,
    log_evaluation_result,
    render_metrics_table,
)
from dora import get_xp, hydra_main

# -------------------------------------------------------------------
# Torch setup
# -------------------------------------------------------------------
torch.set_num_threads(os.cpu_count())
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

    def _evaluate_models(self, xp, eval_dl, tok, per_sentence_stats, avg, avg_kl, epoch_idx):
        results = {}
        with torch.no_grad():
            for label, model in self._iter_models():
                results[label] = self.eval(
                    xp,
                    model,
                    eval_dl,
                    tok,
                    per_sentence_stats,
                    model_label=label,
                    avg=avg,
                    avg_kl=avg_kl,
                    log_prefix=f"epoch {epoch_idx}",
                )
        return results

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
            L_rat1 = nt_xent(h_r1, h_a1, temperature=tau)
            L_comp1 = complement_loss(h_c1, h_a1, null_vec1, self.use_null_target, tau)
            L_s1 = sparsity_loss(g1, attention_mask)
            L_tv1 = total_variation_1d(g1, attention_mask)

            if self.model2:
                out2 = self.model2(embeddings, attention_mask, incoming, outgoing)
                h_a2, h_r2, h_c2, g2 = out2["h_anchor"], out2["h_rat"], out2["h_comp"], out2["gates"]

                null_vec2 = out2["null"] if self.use_null_target else None
                L_rat2 = nt_xent(h_r2, h_a2, temperature=tau)
                L_comp2 = complement_loss(h_c2, h_a2, null_vec2, self.use_null_target, tau)
                L_s2 = sparsity_loss(g2, attention_mask)
                L_tv2 = total_variation_1d(g2, attention_mask)

                # Symmetric KL loss that pushes gates toward complementary behaviour
                kl_weight = self.cfg.model.dual.kl_weight
                g1_soft = torch.clamp(g1, 1e-6, 1 - 1e-6)
                g2_soft = torch.clamp(g2, 1e-6, 1 - 1e-6)
                g1_comp = torch.clamp(1.0 - g1, 1e-6, 1 - 1e-6)
                g2_comp = torch.clamp(1.0 - g2, 1e-6, 1 - 1e-6)

                alpha1, beta1 = out1["alpha"], out1["beta"]
                alpha2, beta2 = out2["alpha"], out2["beta"]

                # KL(K(a1,b1) || distribution of 1 - g2)
                log_p1 = kumaraswamy_log_pdf(g1_soft, alpha1, beta1)
                log_q1 = kumaraswamy_log_pdf(g1_comp, alpha2, beta2)
                kl_1_2 = (log_p1 - log_q1).sum(dim=1).mean()

                # KL(K(a2,b2) || distribution of 1 - g1)
                log_p2 = kumaraswamy_log_pdf(g2_soft, alpha2, beta2)
                log_q2 = kumaraswamy_log_pdf(g2_comp, alpha1, beta1)
                kl_2_1 = (log_p2 - log_q2).sum(dim=1).mean()

                kl_loss = 0.5 * (kl_1_2 + kl_2_1)
                total_kl_loss += kl_loss.item() * embeddings.size(0)
                
                ls_1 = self.cfg.model.dual.ls_1
                ls_2 = self.cfg.model.dual.ls_2

                loss = (
                    (L_rat1 + l_comp * L_comp1 + ls_1 * L_s1 + l_tv * L_tv1)
                    + (L_rat2 + l_comp * L_comp2 + ls_2 * L_s2 + l_tv * L_tv2)
                    + kl_weight * kl_loss
                )
                params = list(self.model1.parameters()) + list(self.model2.parameters())
            else:
                loss = L_rat1 + l_comp * L_comp1 + l_s * L_s1 + l_tv * L_tv1
                params = self.model1.parameters()

            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            self.optimizer.step()

            total += loss.item() * embeddings.size(0)

        return total / len(loader.dataset), total_kl_loss / len(loader.dataset) if self.model2 else None

    def eval(
        self,
        xp,
        model,
        eval_dl,
        tok,
        per_sentence_stats=False,
        model_label="model",
        avg=None,
        avg_kl=None,
        log_prefix=None,
    ) -> EvaluationResult:
        import datetime

        dataset_name = self.cfg.data.eval.dataset
        result = evaluate(
            model,
            eval_dl,
            tok,
            cfg=self.cfg,
            logger=self.logger,
        )

        timestamp = datetime.datetime.now().isoformat()
        extra = {}
        if avg is not None:
            extra["avg_train_loss"] = avg
        if avg_kl is not None:
            extra["avg_train_kl"] = avg_kl
        payload = build_evaluation_payload(
            dataset_name,
            model_label,
            result,
            timestamp=timestamp,
            extra=extra or None,
        )
        xp.link.push_metrics(payload)

        log_evaluation_result(
            self.logger,
            dataset_name,
            model_label,
            result,
            prefix=log_prefix,
            show_samples=self.cfg.eval.samples.show,
            per_sentence_stats=per_sentence_stats,
        )

        return result

    def train(self, train_dl, eval_dl, tok, xp, per_sentence_stats=False):
        best_f1 = float('-inf')
        best_epoch = None
        best_metrics = {}
        best_model_name = None

        for epoch in range(self.cfg.train.epochs):
            avg, avg_kl = self.train_epoch(train_dl, epoch)
            if avg_kl is not None:
                self.logger.info(f"epoch {epoch + 1}: train_loss={avg:.4f}, kl_loss={avg_kl:.4f}")
            else:
                self.logger.info(f"epoch {epoch + 1}: train_loss={avg:.4f}")

            eval_results = self._evaluate_models(
                xp, eval_dl, tok, per_sentence_stats, avg, avg_kl, epoch + 1
            )

            table = render_metrics_table(
                eval_results,
                dataset=self.cfg.data.eval.dataset,
                precision=4,
            )
            for line in table.splitlines():
                self.logger.info(line)

            best_label_epoch = None
            best_metrics_epoch = None
            best_f1_epoch = float('-inf')
            for label, result in eval_results.items():
                metrics = result.metrics or {}
                f1_value = metrics.get('f1')
                if f1_value is None:
                    continue
                if f1_value > best_f1_epoch:
                    best_f1_epoch = f1_value
                    best_label_epoch = label
                    best_metrics_epoch = metrics

            if best_label_epoch is None:
                continue

            if best_f1_epoch > best_f1:
                best_f1 = best_f1_epoch
                best_epoch = epoch + 1
                best_metrics = best_metrics_epoch or {}
                if self.use_dual:
                    best_model_name = best_label_epoch
                message = (
                    f"New best F1: {best_f1:.4f}, saving model(s) to {self._format_checkpoint_names()}"
                )
                self.logger.info(message)
                for _, model, path in self._iter_model_specs():
                    torch.save(model.state_dict(), path)

        if best_f1 == float('-inf'):
            best_f1 = float('nan')
        self.logger.info(
            f"Best metrics at epoch {best_epoch if best_epoch is not None else 'n/a'}: {best_metrics}"
        )
        if self.use_dual and best_model_name is not None:
            self.logger.info(f"Best-performing model: {best_model_name}")

        summary = {
            "best_epoch": best_epoch,
            "best_f1": best_f1,
            "best_metrics": best_metrics,
        }
        if self.use_dual:
            summary["best_model"] = best_model_name

        try:
            xp.link.push_metrics({"summary": summary})
        except Exception as ex:
            self.logger.warning(f"Could not push summary metrics: {ex}")

        return summary

    def eval_only(self, eval_dl, tok, xp, per_sentence_stats=False):
        results = {}
        if self.use_dual:
            for label, model in self._iter_models():
                self.logger.info(f"Eval-only mode: evaluating {label}")
                results[label] = self.eval(
                    xp,
                    model,
                    eval_dl,
                    tok,
                    per_sentence_stats,
                    model_label=label,
                    log_prefix="eval-only",
                )
        else:
            self.logger.info("Eval-only mode: evaluating model")
            results["model"] = self.eval(
                xp,
                self.model1,
                eval_dl,
                tok,
                per_sentence_stats,
                model_label="model",
                log_prefix="eval-only",
            )

        table = render_metrics_table(
            results,
            dataset=self.cfg.data.eval.dataset,
            precision=4,
        )
        for line in table.splitlines():
            self.logger.info(line)



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
        trainer.eval_only(eval_dl, tok, xp, cfg.eval.per_sentence_stats)
    else:
        summary = trainer.train(train_dl, eval_dl, tok, xp, cfg.eval.per_sentence_stats)
        logger.info(f"Training summary: {summary}")
