#!/usr/bin/env bash
# Waits for the running grid to finish phase 3 (the wikiann oracles), stops it
# before phase 4 starts, then runs what was actually asked for:
#
#   wikiann   bert/mean  seeds 3,4,5   (0,1,2 already done -> 6 total)
#   conll2003 bert/mean  seeds 0..5
#
# Phase 4 of the old script (conll2003 x 5 families x 3 seeds, ~13h) is
# deliberately abandoned. Anything it managed to start in the gap between
# detecting P3B_EXIT and killing the parent is purged, so the store does not
# keep half-finished conll2003 runs.
#
# src/metrics.py now FFT-convolves and groups identical (n,c,k) PMFs, verified
# identical to the previous implementation to ~1e-16 and 215x faster overall,
# so a run should cost ~20 min rather than ~53: the exact test used to be the
# larger half of a run and is now seconds.
set -u
cd "$(dirname "$0")/.."
LOG=outputs/word_level_grid.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date -Is)] waiting for phase 3 (oracles) to finish..."
for _ in $(seq 1 1080); do            # 6h ceiling
    grep -aq 'P3B_EXIT=' "$LOG" && break
    sleep 20
done
if ! grep -aq 'P3B_EXIT=' "$LOG"; then
    echo "[$(date -Is)] phase 3 did not finish within 6h -- stopping anyway"
fi
echo "[$(date -Is)] phase 3 done; stopping the old grid before phase 4"
pkill -f 'run_word_level_grid.sh' 2>/dev/null
sleep 2
pkill -f 'forge grid data.dataset=conll2003' 2>/dev/null
sleep 5

.venv/bin/python - <<'PY'
import forge
from forge.core import ExperimentStore, Selection
from omegaconf import OmegaConf
store = ExperimentStore(root="outputs")
drop = []
for s in store.all_selections():
    c = s.experiment.config
    if (str(OmegaConf.select(c, "task")) == "rationale"
            and str(OmegaConf.select(c, "data.dataset")) == "conll2003"):
        drop.append(Selection(experiment=s.experiment, runs=None))
        print("  purging partial phase-4 experiment", s.experiment.signature, flush=True)
forge.purge(drop)
PY

echo "[$(date -Is)] === wikiann bert/mean, seeds 3,4,5 ==="
.venv/bin/forge grid data.dataset=wikiann data.encoder.family=bert data.encoder.pooling=mean \
    runtime.device=cuda runtime.grid=true runtime.data.batch_size=16 train.continue=true \
    --sweep runtime.seed=3,4,5
echo "[$(date -Is)] WIKIANN_EXIT=$?"

echo "[$(date -Is)] === conll2003 bert/mean, seeds 0..5 ==="
.venv/bin/forge grid data.dataset=conll2003 data.encoder.family=bert data.encoder.pooling=mean \
    runtime.device=cuda runtime.grid=true runtime.data.batch_size=16 train.continue=true \
    --sweep runtime.seed=0,1,2,3,4,5
echo "[$(date -Is)] CONLL_EXIT=$?"
echo "[$(date -Is)] === done ==="
