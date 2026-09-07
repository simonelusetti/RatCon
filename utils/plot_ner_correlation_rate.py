"""Correlate the raw over/under-selection rate (per BIO tag, at a chosen
retention rate rho) against the NER MLP probe's token-level F1 for the same
tag, across the three encoder families. wikiann by default; any dataset with
both a bias-test and an NER-probe run for all three encoders works.

Same idea as plot_ner_correlation.py, but the x-axis is a genuine effect
size instead of a significance-scaled statistic: selection_rate_curves.json
already records, per label per rho, the empirical fraction of that label's
tokens the selector retained. Under the null every token is retained with
equal probability regardless of label, so expected rate = rho exactly, and
over/under-selection = observed_rate - rho is the raw, uncalibrated
magnitude of the bias -- not confounded by the label's sample size/variance
the way the z-score (effect_size_curves.json) is. See the paper's discussion
of why this is the preferred axis for this specific plot.

Reports the same two correlations as plot_ner_correlation.py, on this new
scale:
  - within-encoder: each encoder's own per-tag SIGNED over/under-selection
    rate vs per-tag F1.
  - across-encoder: each encoder's mean |over/under-selection rate| vs its
    binary (any entity vs O) detection F1 -- n=3, read directionally.

Usage: python3 utils/plot_ner_correlation_rate.py [--dataset wikiann] [--rho 0.3 | --rho_average] [--output PATH]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import LABEL_DISPLAY_NAMES  # noqa: E402
from utils.forge_paths import run_dir  # noqa: E402
from utils.plot_ner_correlation import (  # noqa: E402
    ENCODER_COLOR,
    ENCODER_DISPLAY,
    ENCODER_MARKER,
    add_sig_args,
    entity_tags,
    load_binary_entity_f1,
    load_ner_f1,
    sigs_from_args,
)


def load_bias_rate(dataset: str, sig: str, rho: float | None) -> dict[str, float]:
    """rho=None averages the over/under-selection rate over every available
    rho except 1.0 (keep-everything -> rate=1.0, over-selection=0 by
    construction there, not a real data point)."""
    label_map = LABEL_DISPLAY_NAMES.get(dataset, {})
    payload = json.loads((run_dir(sig) / "data/selection_rate_curves.json").read_text())
    rhos = np.array(payload["rho"])
    if rho is None:
        keep = ~np.isclose(rhos, 1.0)
        return {
            label_map[label_idx]: float(np.mean(np.array(curve)[keep] - rhos[keep]))
            for label_idx, curve in payload["curves"].items()
            if label_idx in label_map
        }
    idx = int(np.argmin(np.abs(rhos - rho)))
    return {
        label_map[label_idx]: curve[idx] - rhos[idx]
        for label_idx, curve in payload["curves"].items()
        if label_idx in label_map
    }


def main(dataset: str, bias_sigs: dict[str, str], ner_sigs: dict[str, str], rho: float | None, pooled: bool, output: Path) -> None:
    tags = entity_tags(dataset)
    points = []  # (encoder, tag, over_under_rate, f1)
    for encoder in bias_sigs:
        rate_by_tag = load_bias_rate(dataset, bias_sigs[encoder], rho)
        f1_by_tag = load_ner_f1(ner_sigs[encoder], tags)
        for tag in tags:
            points.append((encoder, tag, rate_by_tag[tag], f1_by_tag[tag]))

    # Within-encoder: correlate each encoder's own per-tag signed
    # over/under-selection rate vs per-tag F1 separately, so cross-encoder
    # baseline differences can't confound it.
    within = {}
    for encoder in bias_sigs:
        enc_points = [p for p in points if p[0] == encoder]
        er = np.array([p[2] for p in enc_points])
        ef1 = np.array([p[3] for p in enc_points])
        r_p, _ = pearsonr(er, ef1)
        within[encoder] = r_p

    # Across-encoder: collapse each encoder's per-tag |over/under-selection
    # rate| to one unweighted mean, paired with the genuine per-encoder
    # detection metric (binary entity F1), n=3, read directionally.
    agg_rate_abs = np.array([np.mean([abs(p[2]) for p in points if p[0] == e]) for e in bias_sigs])
    agg_f1 = np.array([load_binary_entity_f1(ner_sigs[e]) for e in bias_sigs])
    across_pearson, _ = pearsonr(agg_rate_abs, agg_f1)

    # Naive pooled: every (encoder, tag) pair as its own independent point.
    pooled_rates = np.array([p[2] for p in points])
    pooled_f1s = np.array([p[3] for p in points])
    pooled_pearson, _ = pearsonr(pooled_rates, pooled_f1s)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.grid(True, color="#DDDDDD", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")
    ax.axvline(0, color="#999999", linewidth=1, linestyle=":", zorder=1)

    for encoder in bias_sigs:
        enc_points = [p for p in points if p[0] == encoder]
        ax.scatter(
            [p[2] for p in enc_points],
            [p[3] for p in enc_points],
            color=ENCODER_COLOR[encoder],
            marker=ENCODER_MARKER[encoder],
            s=70,
            edgecolors="white",
            linewidths=0.8,
            label=ENCODER_DISPLAY[encoder],
            zorder=3,
        )
        for _, tag, rate, f1 in enc_points:
            ax.annotate(
                tag,
                (rate, f1),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7.5,
                color="#555555",
            )

    rho_desc = "averaged over all rho" if rho is None else f"at rho={rho:g}"
    ax.set_xlabel(f"Over/under-selection rate (observed − ρ), {rho_desc}")
    ax.set_ylabel("NER token-level F1")
    ax.set_title(f"{dataset}: encoder over/under-selection vs. NER token F1, by entity tag")
    ax.legend(title="encoder", frameon=False, loc="lower right")
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.22 * (yhi - ylo))

    if pooled:
        box = FancyBboxPatch(
            (0.03, 0.885), 0.34, 0.075,
            transform=ax.transAxes,
            boxstyle="round,pad=0.02",
            facecolor="white", edgecolor="#999999", linewidth=1.2,
            zorder=4,
        )
        ax.add_patch(box)
        ax.text(0.20, 0.9225, f"Pearson r = {pooled_pearson:.2f}", transform=ax.transAxes, ha="center", va="center",
                 fontsize=14, color="#222222", zorder=5)
    else:
        row_labels = [ENCODER_DISPLAY[e] for e in bias_sigs] + ["across"]
        cell_text = [[f"{within[e]:.2f}"] for e in bias_sigs] + [[f"{across_pearson:.2f}"]]
        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=["Pearson"],
            cellLoc="center",
            rowLoc="left",
            bbox=[0.14, 0.78, 0.32, 0.20],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#CCCCCC")
            if row == 0 or col == -1:
                cell.set_text_props(color="#555555")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"Saved plot to {output}")
    for encoder, r_p in within.items():
        print(f"Within {ENCODER_DISPLAY[encoder]}: r={r_p:.4f}  (n={len(tags)})")
    print(f"Across-encoder: r={across_pearson:.4f}  (n=3, low power)")
    print(f"Pooled (naive, every entity type as its own point): r={pooled_pearson:.4f}  (n={len(points)})")


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dataset", default="wikiann", choices=["wikiann", "conll2003"])
    pre_args, _ = pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[pre])
    add_sig_args(parser, dataset_for_defaults=pre_args.dataset)
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--rho_average", action="store_true", help="average the over/under-selection rate over all available rho instead of using a single one")
    parser.add_argument("--pooled", action="store_true", help="show a single large naive-pooled Pearson r instead of the within/across table")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    bias_sigs, ner_sigs = sigs_from_args(args)
    rho = None if args.rho_average else args.rho
    rho_tag = "rhoavg" if args.rho_average else f"rho{args.rho:g}"
    pooled_tag = "_pooled" if args.pooled else ""
    output = args.output or ROOT / "outputs/analysis/ner_correlation_rate" / f"ner_correlation_rate_{args.dataset}_{rho_tag}{pooled_tag}.pdf"
    main(args.dataset, bias_sigs, ner_sigs, rho, args.pooled, output)
