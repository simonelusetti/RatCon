# Iolex

Code for **"No Checked Baggage: Forcing Sentence Encoders to Keep Only
Essential Tokens"** — a study of whether sentence encoders (SBERT, E5,
Pythia) have structural token-selection biases. A differentiable top-ρ
token selector is trained on top of a frozen sentence encoder; its
selections are then checked for label-dependent bias, tested for
downstream-task sufficiency, and correlated against an independent NER
probe.

## Setup

Experiments are managed by [forge](https://github.com/simonelusetti/forge),
which is not on PyPI — clone it and install it editable:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel

git clone https://github.com/simonelusetti/forge.git ../forge
pip install -e ../forge

pip install -r requirements.txt
```

Python >= 3.10. A virtualenv is not optional in practice: a distro
`pip`/`setuptools` older than setuptools 64 cannot install forge editable
at all (`build backend is missing the 'build_editable' hook`), because the
system `setuptools` shadows the modern one pip fetches for the build.

GPU is optional (`runtime.device=cuda`); every experiment in the paper was
produced on CPU. This is a standalone project, not an installable library —
everything runs from the repo root.

## How an experiment is identified

Forge splits identity into two levels:

- an **experiment** — the sha1 of the resolved config, an 8-hex
  **signature** like `f733d54d`
- a **run** — one launch of that experiment, its own 8-hex id

so outputs land under `outputs/xps/<experiment>/<run>/`:

```
outputs/xps/<xp>/
  config.yaml               # the resolved config that defines this experiment
  <run>/
    meta.json               # launched_on, finished_on, status
    metrics.json            # final metrics (run.finish)
    metrics_details.json    # resource usage, resolved runtime
    logs.jsonl              # per-epoch metrics (run.push_log)
    runtime.yaml            # runtime config snapshot for this run
    data/                   # JSON artifacts -- what the plot scripts read
    plots/                  # PNGs saved by the run
    state/models/           # checkpoints
    train.log
```

Keys listed under `forge.exclude` in `conf/config.yaml` are left out of the
signature — all of `runtime.*` (which includes **`runtime.seed`**) and the
`train.*` resume switches.
Excluding the seed is what makes multi-seed sweeps convenient: three seeds
of one configuration are three *runs of one experiment*, not three unrelated
experiments, so the plot scripts take a single signature per curve and pick
up every seed under it.

The seed lives under `runtime` rather than `train` for a second reason
besides the exclusion: forge snapshots the `runtime` block per run into
`runtime.yaml`, while the experiment-level `config.yaml` is rewritten by
every launch. Anywhere else in the config, a multi-seed sweep would leave no
per-run record of which seed a given run used. To see them:

```bash
forge artifact runtime.yaml     # locate each run's snapshot
grep '^seed:' outputs/xps/*/*/runtime.yaml
```

Both entry points share `conf/config.yaml`:

| Task | Entry point | Invocation |
|---|---|---|
| Rationale selector (the bias test) | `train.py` | `forge run <overrides>` |
| Oracle mask search | `train.py` | `forge run task=oracle <overrides>` |

The NER probe is deliberately *not* here — see "The NER probe" below.

Oracle runs share setup, evaluation and artifact generation with the selector,
but skip training, compilation and checkpoint loading. They also save
`data/oracle_masks.npz` and `data/oracle_summary.json`. STS-B search remains
opt-in with `runtime.oracle.stsb=true`. Existing `forge -M oracle ... task=oracle`
commands still work; keeping `task=oracle` preserves experiment signatures.

Key config overrides:

- `data.dataset` — one of `wikiann`, `conll2003`, `conll2000`,
  `movie_rationales`, `stsb`
- `data.encoder.family` — one of `sbert`, `e5`, `llm` (Pythia)
- `runtime.seed` — seed (excluded from the signature, recorded per run in
  `runtime.yaml`; see above)

A plain `forge run data.dataset=wikiann data.encoder.family=sbert` trains
the selector, then automatically runs the bias-test evaluation
(`runtime.eval.skip=false` is the default) and writes every `data/*.json`
artifact the plot scripts need — no separate eval step required.

For `data.encoder.pooling=min` or `max`, each token vector is multiplied by
its selection weight before taking the coordinate-wise minimum or maximum.
Weights are not clipped: zero-weight tokens contribute zero vectors, while
batch padding is excluded. Training differentiates this operation directly.

## The NER probe

The probe is the study's *evaluation object*, not one of its experiments, so
it does not live in the forge store at all. It answers one fixed question
about a corpus — "how well can a small MLP recover each tag from this frozen
encoder's token embeddings?" — and that answer depends on nothing else: the
probe reads word-level token embeddings and never pools, so neither the
pooling strategy, nor ρ, nor the selector can change it. One probe therefore
serves every selector built on the same token encoder.

It is a cache, keyed by exactly what identifies that question:

```
outputs/ner/<dataset>/<family>/seed<k>/
  model.pth      # one model per key -- no per-epoch checkpoints
  report.json    # span_level, token_level, binary_entity_level, + per-epoch history
```

Seed is in the key only so the grounding figures can show a spread across
independently trained probes; drop to one seed if you do not need the band.

```bash
python3 -m ner wikiann --family bert --seeds 0,1,2 --device cuda
python3 -m ner movie_rationales --family bert --seeds 0,1,2 --class-weighted
```

Entries that already exist are loaded, not retrained, so re-running is free
and safe; pass `--retrain` to force. From Python it is one call, which trains
whatever is missing and returns one report per seed:

```python
from ner import performances, load
reports = performances(cfg, dataset="conll2003", seeds=(0, 1, 2))
reports = load("wikiann", "bert")     # read-only, what the plot scripts use
```

Because a dataset and an encoder family name the entry completely, the plot
scripts take no `--*-ner-sig` arguments any more.

### Inspecting results

```bash
forge info                       # every experiment, its config and runs
forge info -S                    # signatures only
forge metrics                    # final metrics, one column per run
forge metrics -l                 # + launched/status columns
forge metrics data.encoder.family=e5   # filter by config override
forge clean                      # drop runs that died without finishing
```

Note that `purge`, `store` and `metrics` infer their selection mode from the
pattern itself (hex → signature, `k=v` → override, otherwise → tag), so
signatures are passed positionally: `forge purge f733d54d`, not `-S`.

## Reproducing the paper's figures

Each row below trains whatever the figure needs (skip if you already have
matching signatures — `forge info -S` lists them) and then calls the plot
script. Every `--*-sig` flag accepts either an experiment signature or a
specific `<xp>/<run>`.

**Figure 4 — per-encoder bias heatmap (WikiAnn / CoNLL-2003)**
```bash
forge grid --file utils/grid_wikiann.yaml

python3 utils/plot_bias_heatmaps.py --dataset wikiann --ncols 1 \
    --strategies sbert,e5,llm
```

**Figure 5 — grounding test (bias rate vs. NER F1, + B/I convergence)**
```bash
forge grid --file utils/grid_conll2003.yaml
python3 -m ner conll2003 --family sbert --seeds 0,1,2
python3 -m ner conll2003 --family e5    --seeds 0,1,2
python3 -m ner conll2003 --family llm   --seeds 0,1,2

python3 utils/plot_paper_figure_conll2003.py \
    --sbert-bias-sig <sig> --e5-bias-sig <sig> --llm-bias-sig <sig> \
    --rho 0.8
```

**Figure 6 — SBERT bias across 3 corpora**

CoNLL-2003 is deliberately not a panel here (WikiAnn already covers NER, and
Figure 4 provides the WikiAnn/CoNLL-2003 comparison) — see the script's
docstring.

```bash
for d in wikiann conll2000 movie_rationales; do
    forge grid --file utils/grid_$d.yaml --sweep data.encoder.family=sbert
done

python3 utils/plot_signed_heatmap_sbert_corpora.py \
    --wikiann-sig <sig> --conll2000-sig <sig> --movie-rationales-sig <sig>
```

**Figure 3 — STS-B sufficiency test (3 panels)**
```bash
# panels 1 + 2: trained+evaluated on STS-B, 3 encoders x 3 seeds
forge grid --file utils/grid_stsb.yaml

# panel 3: SBERT trained on each of the other 4 corpora, evaluated on STS-B
# (STS-B evaluation runs on every training run regardless of data.dataset)
for d in wikiann conll2003 conll2000 movie_rationales; do
    forge grid --file utils/grid_$d.yaml \
        --sweep data.encoder.family=sbert --sweep runtime.seed=0,1,2
done

python3 utils/plot_stsb_sufficiency.py \
    --panel1-sigs <sig> \
    --panel2-sbert-sigs <sig> --panel2-e5-sigs <sig> --panel2-llm-sigs <sig> \
    --panel3-stsb-sigs <sig> --panel3-wikiann-sigs <sig> \
    --panel3-conll2003-sigs <sig> --panel3-conll2000-sigs <sig> \
    --panel3-movie-rationales-sigs <sig>
```
One signature per curve here, not one per seed: the seeds are runs of the
same experiment.

## Sweeps

`forge grid` runs a set of experiments and prints a metrics table at the end.
`utils/grid_*.yaml` hold this repo's sweeps:

```bash
forge grid --file utils/grid_conll2003.yaml         # 3 encoders
```

CLI arguments layer on top of the file, which composes usefully —
`--sweep runtime.seed=0,1,2` adds a seed axis to any grid, and
`--sweep data.encoder.family=sbert` narrows one. There is no probe grid:
`python3 -m ner <datasets...>` takes the place of one.

Two things to know about `forge grid`:

- it runs every entry **in the same process** (dora used a subprocess per
  run), so once-per-process setup has to be idempotent — see
  `_interop_threads_configured` in `src/utils.py`
- it catches and swallows exceptions per entry, marking the run `failed`
  without printing a traceback. When a grid entry fails silently, re-run that
  one config with plain `forge run` to see the error.

`train.continue=true` (set in every grid file) resumes from the highest-epoch
checkpoint across the experiment's runs, so an interrupted grid can be
relaunched as-is. Resume is scoped to runs sharing the same `runtime.seed`:
since the seed is excluded from the signature, every seed is a run of the
same experiment, and an unscoped search would let a seed sweep find an
already-finished run under a *different* seed, decide it had reached the
target epoch, and skip training — quietly yielding identical runs instead of
independent seeds.

Evaluation-only runs (`train.no_train=true`) use the same seed filter when
locating `train.checkpoint_path`. Probe resume also restores per-epoch
classification reports through the loaded checkpoint's epoch.

## Analysis scripts

The figure scripts above reproduce the paper. Three more cover the analysis
this repo has grown since, and take no signatures at all — they discover
what exists in the forge store and in the probe cache:

```bash
python3 utils/plot_ner_grounding.py --grouped-latest --variant both  # grounding, every strategy
python3 utils/plot_bias_heatmaps.py --dataset wikiann                # bias heatmap, every strategy
python3 utils/plot_ner_convergence.py --dataset wikiann              # probe B/I convergence
python3 utils/mask_optimality.py <selector-sig> --rho 0.5            # how good is the selector's mask?
```

Each `utils/*.py` file has a module docstring describing what it produces and
its exact CLI usage — run `python3 utils/<script>.py --help` for the
authoritative, current argument list. `utils/_common.py` holds what they
share: the encoder palette, the heatmap colormap, and the per-tag loaders
that read a bias run and the probe cache.

## Known issues

The old script-based `conll2003` Hub loader is incompatible with
`datasets==4.0.0`. `build_conll2003` therefore reads the Hub's converted
parquet revision directly, preserving the original splits and ClassLabel
metadata without executing a dataset script.
