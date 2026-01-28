import logging
from omegaconf import OmegaConf
from pathlib import Path

from src.data import get_dataset
from src.sentence import build_sentence_encoder

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

DEVICE = "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rebuild")

def rebuild() -> None:
    for family in ENCODER_FAMILIES:
        logger.info("=" * 80)
        logger.info(f"Rebuilding datasets for tokenizer family: {family}")
        logger.info("=" * 80)

        data_cfg = OmegaConf.create(
            {
                "dataset": None,
                "subset": 1.0,
                "max_length": 512,
                "encoder": {
                    "family": family,
                    "name": None,
                },
                "runtime": {
                    "rebuild": True,
                },
                "config": None,
            }
        )

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

if __name__ == "__main__":
    rebuild()
