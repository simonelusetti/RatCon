"""Signed effect heatmap (entity tags x rho) for all three encoders on one
dataset, in the paper's original Figure 4 layout (one panel per encoder,
shared colorbar). wikiann by default; conll2003 also available.

Two metrics, since they trade off differently:
  --metric zscore  (default): raw signed z-score. Doesn't saturate, so it
      shows real gradation -- the recommended choice.
  --metric pvalue: sign(z) * -log10(p), matching the original chi-square-era
      figure's convention exactly. At this scale these p-values are so
      extreme they hit the exact test's floating-point precision floor
      (~1e-14), so most cells saturate to the same color regardless of how
      much stronger one effect is than another -- kept for continuity with
      the original figure, not because it's the more informative view.

Usage: python3 utils/plot_signed_heatmap.py [--dataset wikiann|conll2003] [--metric zscore|pvalue] [--output PATH]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import LABEL_DISPLAY_NAMES  # noqa: E402
from utils._heatmap_common import make_flat_grey_cmap, make_flat_grey_norm  # noqa: E402

# Bias-test (rationale task) signatures, per dataset -- used only as CLI
# argument DEFAULTS (see __main__), not a hardcoded lookup; every sig is a
# real, overridable --*-sig argument.
_DEFAULT_SIGS = {
    "wikiann": {"sbert": "97d170e1", "e5": "479e2061", "llm": "55a3e229"},
    "conll2003": {"sbert": "273e869c", "e5": "84efae79", "llm": "d4fae7a4"},
}
DATASET_DISPLAY = {"wikiann": "WikiAnn", "conll2003": "CoNLL2003"}
ENCODER_DISPLAY = {"sbert": "SBERT", "e5": "E5", "llm": "Pythia 400m"}

# Two-sided threshold at p=0.05 (uncorrected) -- a visual "is this even
# nominally significant" control, marked on the colorbar, distinct from the
# per-dataset Bonferroni threshold used for the actual significance decision.
Z_CRIT_P05 = 1.9599639845400545
NEGLOGP_CRIT_P05 = -math.log10(0.05)

# Background / non-entity classes to drop from every panel's rows.
EXCLUDE_LABELS = {"O", "special"}


def entity_tags(dataset: str) -> list[str]:
    label_map = LABEL_DISPLAY_NAMES.get(dataset, {})
    return sorted(name for name in label_map.values() if name not in EXCLUDE_LABELS)


def load_matrix(dataset: str, sig: str, metric: str) -> tuple[list[float], dict[str, list[float]]]:
    label_map = LABEL_DISPLAY_NAMES.get(dataset, {})
    effect = json.loads((ROOT / "outputs/xps" / sig / "data/effect_size_curves.json").read_text())
    rho = [float(r) for r in effect["rho"]]

    matrix = {}
    if metric == "pvalue":
        pvalue = json.loads((ROOT / "outputs/xps" / sig / "data/pvalue_curves.json").read_text())
    for label_idx, name in label_map.items():
        if name in EXCLUDE_LABELS:
            continue
        z_curve = np.array(effect["curves"][label_idx])
        if metric == "zscore":
            matrix[name] = z_curve.tolist()
        else:
            neglogp_curve = np.array(pvalue["curves"][label_idx])
            matrix[name] = (np.sign(z_curve) * neglogp_curve).tolist()
    return rho, matrix


def main(dataset: str, sigs: dict[str, str], metric: str, output: Path) -> None:
    sigs = {ENCODER_DISPLAY[enc]: sig for enc, sig in sigs.items()}
    rows = entity_tags(dataset)
    display_name = DATASET_DISPLAY.get(dataset, dataset)
    p05 = Z_CRIT_P05 if metric == "zscore" else NEGLOGP_CRIT_P05
    cbar_label = "Preference score" if metric == "zscore" else "sign × -log10(p)"

    panels = {name: load_matrix(dataset, sig, metric) for name, sig in sigs.items()}

    all_abs = [abs(v) for _, matrix in panels.values() for curve in matrix.values() for v in curve if np.isfinite(v)]
    vmax = float(np.percentile(all_abs, 95)) if all_abs else 1.0
    norm = make_flat_grey_norm(vmax, thresh=p05)
    cmap = make_flat_grey_cmap()

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(6.5, 2.3 * n), sharex=True)
    axes = np.atleast_1d(axes)

    im = None
    for ax, (name, (rho, matrix)) in zip(axes, panels.items()):
        data = np.array([matrix[r] for r in rows])
        im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm, origin="upper")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=8)
        ax.set_title(f"{name} ({display_name})", fontsize=10)
        ax.set_xticks(range(len(rho)))
        ax.set_xticklabels([f"{r:.2f}" for r in rho], fontsize=7, rotation=45, ha="right")

    axes[-1].text(1.02, 0, "ρ", transform=axes[-1].transAxes, fontsize=9, ha="left", va="center")
    fig.tight_layout()
    cbar = fig.colorbar(im, ax=list(axes), shrink=0.85, pad=0.03)
    cbar.set_label(cbar_label, fontsize=9)
    for sign in (1, -1):
        cbar.ax.axhline(sign * p05, color="#333333", linestyle="--", linewidth=1)
    default_ticks = list(cbar.get_ticks())
    all_ticks = sorted(default_ticks + [p05])
    labels = [("p = 0.05" if t == p05 else f"{t:g}") for t in all_ticks]
    cbar.set_ticks(all_ticks)
    cbar.set_ticklabels(labels)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved plot to {output} (+ .pdf)")


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dataset", default="wikiann", choices=list(_DEFAULT_SIGS))
    pre_args, _ = pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[pre])
    defaults = _DEFAULT_SIGS[pre_args.dataset]
    for enc in ("sbert", "e5", "llm"):
        parser.add_argument(f"--{enc}-sig", default=defaults.get(enc),
                             help=f"bias-test dora sig for {ENCODER_DISPLAY[enc]}")
    parser.add_argument("--metric", choices=["zscore", "pvalue"], default="zscore")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    sigs = {enc: getattr(args, f"{enc}_sig") for enc in ("sbert", "e5", "llm")}
    missing = [f"--{enc}-sig" for enc, s in sigs.items() if not s]
    if missing:
        raise ValueError(f"Missing required signature argument(s): {', '.join(missing)}")
    output = args.output or ROOT / f"outputs/analysis/signed_heatmap_{args.dataset}_{args.metric}.png"
    main(args.dataset, sigs, args.metric, output)
