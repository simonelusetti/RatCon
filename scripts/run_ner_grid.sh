#!/usr/bin/env bash
# NER MLP probe sweep on WikiAnn: one probe per token encoder x 3 seeds.
#
#   tmux new-session -d -s iolex-ner "bash scripts/run_ner_grid.sh 2>&1 | tee outputs/grid_ner_wikiann.log"
#
# -M ner_probe is what selects the probe entry point over the selector.
set -u
cd "$(dirname "$0")/.."
.venv/bin/forge -M ner_probe grid --file utils/grid_ner.yaml runtime.device=cuda
echo "GRID_EXIT=$?"
