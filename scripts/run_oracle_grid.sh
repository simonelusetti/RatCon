#!/usr/bin/env bash
# Oracle mask search across the remaining token encoders on wikiann, mean
# pooling. bert is omitted: 59db8be3 is already done.
#
#   tmux new-session -d -s iolex-oracle-grid \
#     "bash scripts/run_oracle_grid.sh 2>&1 | tee outputs/oracle_grid.log"
#
# --run entries execute in the order given, so electra and roberta land
# first -- those are the two that decide whether the selector-vs-oracle
# divergence is specific to BERT. train.continue is irrelevant here (there
# is nothing to resume: the search is deterministic and stateless), so an
# interrupted family simply re-runs from scratch.
set -u
cd "$(dirname "$0")/.."
.venv/bin/forge grid task=oracle \
    data.dataset=wikiann data.encoder.pooling=mean \
    runtime.device=cuda runtime.data.batch_size=512 \
    --run data.encoder.family=electra \
    --run data.encoder.family=roberta \
    --run data.encoder.family=sbert \
    --run data.encoder.family=e5 \
    --run data.encoder.family=llm
echo "ORACLE_GRID_EXIT=$?"
