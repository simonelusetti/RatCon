"""Compatibility entry point for the brute-force oracle mask search.

    forge -M oracle run task=oracle data.dataset=wikiann data.encoder.family=bert

New launches can use `forge run task=oracle` through train.py. This shim
keeps existing commands working and still requires task=oracle.
"""
from src.oracle import main  # noqa: F401  -- forge calls main(cfg)
