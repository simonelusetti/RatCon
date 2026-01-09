from __future__ import annotations

import logging
from omegaconf import OmegaConf
from pathlib import Path

from luse.data import get_dataset
from luse.sentence import build_sentence_encoder

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATASETS = [
    "conll2003",
    "conll2000",
    "movie_rationales",
    "treebank",
    "ud",
    "parasci",
    "parasci_concat",
    "tweet_sentiment",
]

ENCODER_FAMILIES = [
    "sbert",
    "e5",
    "bge",
    "llm",
]

DEVICE = "cpu"  # always CPU for dataset rebuilds

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rebuild")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def rebuild():
    for family in ENCODER_FAMILIES:
        logger.info("=" * 80)
        logger.info(f"Rebuilding datasets for tokenizer family: {family}")
        logger.info("=" * 80)

        # Minimal config needed for tokenization
        data_cfg = OmegaConf.create(
            {
                "dataset": None,
                "subset": 1.0,
                "max_length": 512,
                "encoder": {
                    "family": family,
                    "name": None,  # use default
                },
                "runtime": {
                    "rebuild": True,
                },
                "config": None,
            }
        )

        # Build encoder once to get tokenizer
        _, tokenizer = build_sentence_encoder(
            family=family,
            encoder_name=None,
            device=DEVICE,
        )

        for dataset in DATASETS:
            logger.info(f"→ Rebuilding dataset: {dataset}")
            data_cfg.dataset = dataset

            try:
                get_dataset(
                    data_cfg=data_cfg,
                    tokenizer=tokenizer,
                    logger=logger,
                )
            except Exception as e:
                logger.error(f"Failed rebuilding {dataset} ({family}): {e}")

    logger.info("All rebuilds completed.")

# ---------------------------------------------------------------------

if __name__ == "__main__":
    rebuild()
