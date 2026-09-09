"""Everything the utils/plot_*.py scripts share.

Colormap + norm for the signed heatmaps, the encoder palette, and the
per-tag loaders that read a bias run and the tagger cache.

--- colormap + norm ---------------------------------------------------

Unlike src/view.py's _make_signed_effect_norm (which fades gradually inside
its significance threshold and uses a saturating gamma curve outside it),
this pair reserves a fixed, visible slice of the colorbar for the
non-significant zone (|value| <= thresh) and renders it as a genuinely flat
grey -- not a fade -- then maps everything outside thresh linearly (no
saturation curve) so real magnitude differences stay visible.

Getting a truly flat colour (not just reserved space) needs both pieces
together: the norm positions |value| <= thresh within a fixed band of
normalized colorbar space, and the colormap has an actual constant-colour
segment across that same band (two control points at the same colour --
LinearSegmentedColormap interpolates two identical colours to a constant).
A norm alone can't do it: imshow/colorbar always compute cmap(norm(value)),
so if norm(value) still varies across the band, a 3-point red-grey-blue
colormap keeps interpolating through it regardless of how the norm is
shaped.

Kept local to these analysis scripts rather than changing the shared
core-pipeline norm/colormap.
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import FuncNorm, LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tagger  # noqa: E402
from src.data import LABEL_DISPLAY_NAMES  # noqa: E402

# Fraction of the [0,1] colorbar each side of center devoted to the
# non-significant zone -- shared by the norm and the colormap so their
# bands line up exactly. Totals 2x this fraction of the bar's full height
# (0.05 -> 10%).
GREY_HALF_WIDTH = 0.05
_GREY = "#d9d9d9"
_RED = "#b2182b"
_BLUE = "#2166ac"


def make_flat_grey_cmap(grey_half_width: float = GREY_HALF_WIDTH) -> LinearSegmentedColormap:
    GH = grey_half_width
    return LinearSegmentedColormap.from_list(
        "RdGreyBu_flat",
        [(0.0, _RED), (0.5 - GH, _GREY), (0.5 + GH, _GREY), (1.0, _BLUE)],
    )


def make_flat_grey_norm(vmax: float, thresh: float, grey_half_width: float = GREY_HALF_WIDTH) -> FuncNorm:
    GH = grey_half_width
    safe_v = max(vmax - thresh, 1e-8)

    def forward(x):
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)

        pm = x > thresh
        nm = x < -thresh
        dm = ~pm & ~nm

        out[dm] = 0.5 + (x[dm] / thresh) * GH if thresh > 0 else 0.5
        u = np.clip((x[pm] - thresh) / safe_v, 0.0, 1.0)
        out[pm] = (0.5 + GH) + (0.5 - GH) * u
        u = np.clip((-x[nm] - thresh) / safe_v, 0.0, 1.0)
        out[nm] = (0.5 - GH) - (0.5 - GH) * u

        return np.clip(out, 0.0, 1.0)

    def inverse(y):
        y = np.asarray(y, dtype=float)
        out = np.zeros_like(y)

        g_lo, g_hi = 0.5 - GH, 0.5 + GH
        pm = y > g_hi
        nm = y < g_lo
        dm = ~pm & ~nm

        out[dm] = (y[dm] - 0.5) / GH * thresh if GH > 0 else 0.0
        u = np.clip((y[pm] - g_hi) / (1.0 - g_hi), 0.0, 1.0)
        out[pm] = thresh + safe_v * u
        u = np.clip((g_lo - y[nm]) / g_lo, 0.0, 1.0)
        out[nm] = -(thresh + safe_v * u)

        return out

    return FuncNorm((forward, inverse), vmin=-vmax, vmax=vmax)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
#
# Okabe-Ito, colourblind-safe, fixed assignment (never cycled). Colour means
# encoder everywhere in this repo; where a second axis is needed (pooling
# strategy) it is carried by line style, not hue.
ENCODER_COLOR = {
    "sbert":   "#0072B2",
    "e5":      "#E69F00",
    "llm":     "#009E73",
    "bert":    "#D55E00",
    "electra": "#CC79A7",
    "roberta": "#56B4E9",
}
ENCODER_MARKER = {"sbert": "o", "e5": "s", "llm": "^",
                  "bert": "v", "electra": "D", "roberta": "P"}
ENCODER_DISPLAY = {"sbert": "sbert", "e5": "e5", "llm": "pythia",
                   "bert": "bert", "electra": "electra", "roberta": "roberta"}

SENTENCE_ENCODERS = {"sbert", "e5", "llm"}
TOKEN_ENCODERS = {"bert", "electra", "roberta"}

# Not entity types: O's magnitude would dwarf every real tag and flatten the
# scale, and "special" is not a class at all.
EXCLUDE_LABELS = {"O", "special"}


def entity_tags(dataset: str) -> list[str]:
    label_map = LABEL_DISPLAY_NAMES.get(dataset, {})
    return [name for name in label_map.values() if name not in EXCLUDE_LABELS]


# ---------------------------------------------------------------------------
# Per-tag loaders
# ---------------------------------------------------------------------------

def load_bias(run_path: Path, dataset: str, metric: str, rho: float | None) -> dict[str, float]:
    """Selection bias per entity tag for a single bias-test run.

    metric="selection": the observed selection rate minus rho. Under the null
    every token is kept with probability rho regardless of label, so this is
    the raw over/under-selection in probability units -- an effect size,
    unscaled by how variable that tag's count could be by chance.
    metric="division": the same effect divided by the exact null's standard
    deviation, i.e. the signed z-score from the hypergeometric-convolution
    test. Standardising this way lets a rare tag's small raw excess count for
    as much as a common tag's large one. It is an effect size, not a
    significance value -- the p-value is a separate artifact
    (pvalue_curves.json).

    rho=None averages over every rho except 1.0, where keeping everything
    forces the effect to 0 by construction rather than by measurement.
    """
    label_map = LABEL_DISPLAY_NAMES.get(dataset, {})
    filename = "selection_rate_curves.json" if metric == "selection" else "effect_size_curves.json"
    payload = json.loads((run_path / "data" / filename).read_text())
    rhos = np.array(payload["rho"], dtype=float)
    offset = rhos if metric == "selection" else np.zeros_like(rhos)

    # label_map.get(idx, idx), not label_map[idx]: datasets with an upstream
    # ClassLabel (wikiann, conll2003) key their curves by stringified index,
    # but conll2000 has none -- NLTK yields the tag string itself, so its
    # artifacts are keyed "B-NP" directly. Requiring a map hit silently
    # dropped every conll2000 tag and produced an empty result rather than
    # an error.
    if rho is None:
        keep = ~np.isclose(rhos, 1.0)
        return {
            label_map.get(idx, idx): float(np.mean(np.array(curve, dtype=float)[keep] - offset[keep]))
            for idx, curve in payload["curves"].items()
        }
    i = int(np.argmin(np.abs(rhos - rho)))
    return {
        label_map.get(idx, idx): float(curve[i] - offset[i])
        for idx, curve in payload["curves"].items()
    }


def probe_reports(dataset: str, family: str) -> list[dict]:
    """Cached tagger reports for a token encoder, one per seed."""
    reports = tagger.load(dataset, family)
    if not reports:
        raise SystemExit(
            f"No cached tagger for {dataset}/{family}. "
            f"Build one with: python3 -m tagger {dataset} --family {family}")
    return reports


def tagger_f1(report: dict, tags: list[str]) -> dict[str, float]:
    return {tag: float(report["token_level"][tag]["f1-score"]) for tag in tags}


def binary_entity_f1(report: dict) -> float | None:
    """Token-level F1 of "is this token part of any entity" (B-*/I-* collapsed
    against O) -- a real detection metric rather than an average of per-tag F1s.

    None when the corpus has no "O" class (UD upos, deprel, GUM discourse: every
    token carries a real label, so "is this a token" is vacuous). Callers skip
    the aggregate view rather than plotting a constant.
    """
    level = report.get("binary_entity_level")
    return float(level["entity"]["f1-score"]) if level else None


def mean_spread(values: list[float], spread: str) -> tuple[float, float]:
    """Mean and its shadow radius over seeds. ddof=1: these are a sample of
    seeds, not the population of them."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float(arr.mean()), 0.0
    sd = float(arr.std(ddof=1))
    return float(arr.mean()), sd / np.sqrt(arr.size) if spread == "sem" else sd
