#!/usr/bin/env bash
# Oracle search for pythia, the one family the grid could not complete.
#
# pooling=last, NOT mean: pythia's selector runs predate the pooling axis and
# used last-token pooling for their reconstruction target -- the standard way
# to embed a causal LM, since only the final position has attended to the
# whole sentence. Running the oracle under mean would make the pair
# not-like-for-like and the selector/oracle comparison meaningless.
#
# It OOM'd at the default runtime.oracle.chunk_tokens=262144: pythia-410m is
# 1024-dim x 24 layers against bert-base's 768 x 12, so the same token budget
# needs roughly three times the activation memory. Quartering the budget
# trades a little Python-loop overhead for headroom; expandable_segments
# additionally limits allocator fragmentation, which the OOM message itself
# suggested.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/forge run task=oracle \
    data.dataset=wikiann data.encoder.family=llm data.encoder.pooling=last \
    runtime.device=cuda runtime.data.batch_size=256 \
    runtime.oracle.chunk_tokens=65536
echo "ORACLE_LLM_EXIT=$?"
