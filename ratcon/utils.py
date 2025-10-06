import os, sys, torch

def should_disable_tqdm(*, metrics_only: bool = False) -> bool:
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
    if hasattr(model1, "fourier"):
        h_shared1 = model1.fourier(h_shared1)
    if hasattr(model2, "fourier"):
        h_shared2 = model2.fourier(h_shared2)

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
    if hasattr(model1, "fourier"):
        h_shared_comp1 = model1.fourier(h_shared_comp1)
    if hasattr(model2, "fourier"):
        h_shared_comp2 = model2.fourier(h_shared_comp2)

    return h_shared_comp1, h_shared_comp2, shared_comp_mask
