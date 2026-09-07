#!/usr/bin/env bash
# Runs the NER probe sweep for the two new token encoders once the
# pooling-encoder grid has exited. The wait is done by this detached shell,
# not by an agent polling -- nothing outside this process is involved.
#
#   tmux new-session -d -s iolex-ner2 \
#     "bash scripts/run_ner_probes_after_grid.sh 2>&1 | tee outputs/grid_ner_encoders.log"
#
# Waits on the grid PROCESS (pgrep) rather than on a tmux session: one fewer
# dependency, and it still works if the grid's session is renamed or the
# tmux server is restarted. The heartbeat line matters -- a previous attempt
# at this chaining died mid-wait and left a zero-byte log, so there was no
# way to tell how far it had got. Now there always is.
set -u
cd "$(dirname "$0")/.."

GRID_PATTERN="forge grid --file utils/grid_pooling_wikiann"
echo "[$(date -Is)] waiting for the pooling-encoder grid to finish"
while pgrep -f "$GRID_PATTERN" >/dev/null; do
    sleep 300
    echo "[$(date -Is)] still waiting"
done

echo "[$(date -Is)] grid finished -- starting NER probes for electra + roberta"
.venv/bin/forge -M ner_probe grid --file utils/grid_ner.yaml runtime.device=cuda \
    --sweep data.encoder.family=electra,roberta
echo "[$(date -Is)] GRID_EXIT=$?"
