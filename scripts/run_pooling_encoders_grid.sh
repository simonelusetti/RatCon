#!/usr/bin/env bash
# Pooling sweep over the two additional raw token encoders on WikiAnn:
# {electra, roberta} x {mean, max, min} x 3 seeds = 18 runs, i.e. 6
# experiments of 3 runs each. The --sweep replaces the grid file's own
# data.encoder.family axis rather than adding to it, so bert is not re-run.
set -u
cd "$(dirname "$0")/.."
.venv/bin/forge grid --file utils/grid_pooling_wikiann.yaml runtime.device=cuda \
  --sweep data.encoder.family=electra,roberta
echo "GRID_EXIT=$?"
