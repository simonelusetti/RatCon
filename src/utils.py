import os, sys, torch
from typing import Any, Mapping, MutableSequence, Sequence

def should_disable_tqdm(*, metrics_only=False):
    """Return True when tqdm progress bars should be disabled."""
    if metrics_only:
        return True

    override = os.environ.get("RATCON_DISABLE_TQDM")
    if override is not None:
        return override.strip().lower() not in {"0", "false", "no", "off"}

    try:
        return not sys.stderr.isatty()
    except Exception:
        return True

import logging
import sys
from pathlib import Path
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            sys.stdout.flush()
        except Exception:  # pragma: no cover - logging fallback
            self.handleError(record)


def get_logger(logfile="train.log"):
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()

    ch = TqdmLoggingHandler()
    ch.setLevel(logging.INFO)
    ch_format = "%(asctime)s - %(levelname)s - %(message)s"
    ch.setFormatter(logging.Formatter(ch_format))

    fh = logging.FileHandler(Path(logfile))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def shared_distribution(g1, g2, token_emb1, token_emb2, attention_mask, model1, model2):
    """Pool union rationales from two gate distributions."""

    shared_gate = 1.0 - (1.0 - g1) * (1.0 - g2)
    shared_gate = torch.clamp(shared_gate, 1e-6, 1.0)
    shared_mask = attention_mask * shared_gate

    h_shared1 = model1.pooler({
        "token_embeddings": token_emb1,
        "attention_mask": shared_mask,
    })["sentence_embedding"]
    h_shared2 = model2.pooler({
        "token_embeddings": token_emb2,
        "attention_mask": shared_mask,
    })["sentence_embedding"]

    return h_shared1, h_shared2, shared_mask


def shared_complement_distribution(g1, g2, token_emb1, token_emb2, attention_mask, model1, model2):
    """Pool intersection complements from two gate distributions."""

    shared_comp_gate = (1.0 - g1) * (1.0 - g2)
    shared_comp_gate = torch.clamp(shared_comp_gate, 1e-6, 1.0)
    shared_comp_mask = attention_mask * shared_comp_gate

    h_shared_comp1 = model1.pooler({
        "token_embeddings": token_emb1,
        "attention_mask": shared_comp_mask,
    })["sentence_embedding"]
    h_shared_comp2 = model2.pooler({
        "token_embeddings": token_emb2,
        "attention_mask": shared_comp_mask,
    })["sentence_embedding"]

    return h_shared_comp1, h_shared_comp2, shared_comp_mask


def resolve_sparsity_weights(model_cfg, num_models, *, logger=None):
    """Determine sparsity weights for each model based on configuration."""
    loss_cfg = model_cfg.loss
    base = float(loss_cfg.l_s)
    if num_models <= 1:
        return [base]

    dual_cfg = model_cfg.dual
    weights: MutableSequence[float] = []

    configured = dual_cfg.sparsity_weights
    if configured is not None:
        configured = list(configured)
        if len(configured) == 1:
            configured = configured * num_models
        if len(configured) == num_models:
            return [float(w) for w in configured]
        if logger is not None:
            logger.warning(
                "sparsity_weights length (%d) does not match num_models (%d); falling back to base weight.",
                len(configured),
                num_models,
            )

    return [base for _ in range(num_models)]


def collect_joint_samples(reports, model_labels, samples_cfg):
    """Collect aligned samples across multiple models for side-by-side comparison."""
    if samples_cfg is None or len(model_labels) <= 1:
        return []

    if not bool(samples_cfg.show):
        return []

    num_limit = int(samples_cfg.num or 0)
    samples_per_label = {}
    for label in model_labels:
        samples = list(reports.get(label, {}).get("samples", []) or [])
        if not samples:
            return []
        if num_limit > 0:
            samples = samples[:num_limit]
        samples_per_label[label] = samples

    max_len = min(len(samples_per_label[label]) for label in model_labels)
    joint_samples = []
    for idx in range(max_len):
        base_sample = samples_per_label[model_labels[0]][idx]
        entry = {"original": base_sample.get("original", "")}
        for label in model_labels:
            entry[label] = samples_per_label[label][idx].get("predicted", "")
        joint_samples.append(entry)
    return joint_samples


def log_joint_samples(logger, samples, model_labels, header):
    """Log joint samples in a compact, readable format."""
    if not samples:
        return
    lines = [header]
    for sample in samples:
        lines.append("")
        lines.append(f"Original: {sample.get('original', '')}")
        for label in model_labels:
            lines.append(f"  Predicted ({label}): {sample.get(label, '')}")
    logger.info("\n".join(lines))
