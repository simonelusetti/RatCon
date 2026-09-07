"""Entry point for the brute-force oracle mask search.

    forge -M oracle run task=oracle data.dataset=wikiann data.encoder.family=bert

Shim for the same reason as train.py -- see its docstring. `task=oracle` is
required; src/oracle.py refuses to run without it, so an oracle run can never
collide with the selector experiment it is meant to be compared against.
"""
from src.oracle import main  # noqa: F401  -- forge calls main(cfg)
