#!/usr/bin/env bash
# Pooling-strategy sweep on WikiAnn: bert token encoder x {mean,max,min} x 3 seeds.
#
#   tmux new-session -d -s iolex-pooling "bash scripts/run_pooling_grid.sh 2>&1 | tee outputs/grid_pooling_wikiann.log"
#
# Lives in the repo, not a scratchpad: an agent scratchpad is session-scoped,
# and a run that outlives the session should not depend on one.
set -u
cd "$(dirname "$0")/.."
.venv/bin/forge grid --file utils/grid_pooling_wikiann.yaml runtime.device=cuda
echo "GRID_EXIT=$?"
