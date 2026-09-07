"""Correlate the bias-test effect size (z-score, per BIO tag, at a chosen
retention rate rho) against the NER MLP probe's token-level F1 for the same
tag, across the three encoder families. wikiann by default; any dataset
with both a bias-test and an NER-probe run for all three encoders works.

Reports two different correlations (see the paper's discussion of within-
vs across-encoder pooling), each on the scale that showed the real signal:
  - within-encoder: each encoder's own per-tag SIGNED preference vs
    per-tag F1 -- direction is exactly what this axis is testing.
  - across-encoder: each encoder's mean |preference| vs its binary (any
    entity vs O) detection F1 -- n=3, so read directionally, not as a
    test. Magnitude, not direction, is what tracked overall detection
    ability here.

Usage: python3 utils/plot_ner_correlation.py [--dataset wikiann] [--rho 0.3 | --rho_average] [--output PATH]
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

# Optional CLI argument DEFAULTS (see add_sig_args below) -- every sig is a
# real, overridable --*-sig argument, not a hardcoded lookup.
#
# Deliberately empty since the move from dora to forge: forge hashes the
# config differently, so none of the old dora signatures resolve any more,
# and a default that always fails is worse than no default (sigs_from_args
# raises naming exactly which flags are missing). Fill a dataset back in once
# its sweep has been re-run under forge -- `forge info -S` lists signatures.
_DEFAULT_SIGS: dict[str, dict[str, dict[str, str]]] = {}

# Background / non-entity classes to drop -- not an entity type, and its
# magnitude would dwarf the real entity tags' z-scores and distort the plot.
EXCLUDE_LABELS = {"O", "special"}

# Okabe-Ito colorblind-safe palette, fixed assignment (never cycled).
ENCODER_COLOR = {"sbert": "#0072B2", "e5": "#E69F00", "llm": "#009E73"}
ENCODER_MARKER = {"sbert": "o", "e5": "s", "llm": "^"}
ENCODER_DISPLAY = {"sbert": "sbert", "e5": "e5", "llm": "pythia"}


def add_sig_args(parser: argparse.ArgumentParser, dataset_for_defaults: str = "wikiann") -> None:
    """Adds --{enc}-bias-sig / --{enc}-ner-sig for enc in sbert/e5/llm.
    Defaults come from _DEFAULT_SIGS[dataset_for_defaults] so existing
    invocations keep working unchanged; pass any sig explicitly to override."""
    defaults = _DEFAULT_SIGS.get(dataset_for_defaults, {"bias": {}, "ner": {}})
    for enc in ("sbert", "e5", "llm"):
        parser.add_argument(
            f"--{enc}-bias-sig", default=defaults["bias"].get(enc),
            help=f"bias-test (rationale) forge sig for {ENCODER_DISPLAY[enc]}",
        )
        parser.add_argument(
            f"--{enc}-ner-sig", default=defaults["ner"].get(enc),
            help=f"NER-probe forge sig for {ENCODER_DISPLAY[enc]}",
        )


def sigs_from_args(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, str]]:
    bias_sigs = {enc: getattr(args, f"{enc}_bias_sig") for enc in ("sbert", "e5", "llm")}
    ner_sigs = {enc: getattr(args, f"{enc}_ner_sig") for enc in ("sbert", "e5", "llm")}
    missing = [f"--{enc}-bias-sig" for enc, s in bias_sigs.items() if not s]
    missing += [f"--{enc}-ner-sig" for enc, s in ner_sigs.items() if not s]
    if missing:
        raise ValueError(f"Missing required signature argument(s): {', '.join(missing)}")
    return bias_sigs, ner_sigs


def entity_tags(dataset: str) -> list[str]:
    label_map = LABEL_DISPLAY_NAMES.get(dataset, {})
    return [name for name in label_map.values() if name not in EXCLUDE_LABELS]


def load_bias_z(dataset: str, sig: str, rho: float | None) -> dict[str, float]:
    """rho=None averages the z-score curve over every available rho except
    1.0 (keep-everything -> z=0 by construction there, not a real data point)."""
    label_map = LABEL_DISPLAY_NAMES.get(dataset, {})
    payload = json.loads((run_dir(sig) / "data/effect_size_curves.json").read_text())
    if rho is None:
        rhos = np.array(payload["rho"])
        keep = ~np.isclose(rhos, 1.0)
        return {
            label_map[label_idx]: float(np.mean(np.array(curve)[keep]))
            for label_idx, curve in payload["curves"].items()
            if label_idx in label_map
        }
    rhos = np.array(payload["rho"])
    idx = int(np.argmin(np.abs(rhos - rho)))
    return {
        label_map[label_idx]: curve[idx]
        for label_idx, curve in payload["curves"].items()
        if label_idx in label_map
    }


def load_ner_f1(sig: str, tags: list[str]) -> dict[str, float]:
    report = json.loads((run_dir(sig) / "data/ner_classification_report.json").read_text())
    return {tag: report["token_level"][tag]["f1-score"] for tag in tags}


def load_binary_entity_f1(sig: str) -> float:
    """Token-level F1 of "is this token part of any entity" (all B-*/I-*
    tags collapsed to one class vs O) -- a genuine detection metric, not an
    aggregate of the per-tag F1s used for the within-encoder analysis."""
    report = json.loads((run_dir(sig) / "data/ner_classification_report.json").read_text())
    return report["binary_entity_level"]["entity"]["f1-score"]


def main(dataset: str, bias_sigs: dict[str, str], ner_sigs: dict[str, str], rho: float | None, pooled: bool, output: Path) -> None:
    tags = entity_tags(dataset)
    points = []  # (encoder, tag, signed_z, f1)
    for encoder in bias_sigs:
        z_by_tag = load_bias_z(dataset, bias_sigs[encoder], rho)
        f1_by_tag = load_ner_f1(ner_sigs[encoder], tags)
        for tag in tags:
            points.append((encoder, tag, z_by_tag[tag], f1_by_tag[tag]))

    # Within-encoder: correlate each encoder's own per-tag signed preference
    # vs per-tag F1 separately, so cross-encoder baseline differences can't
    # confound it.
    within = {}
    for encoder in bias_sigs:
        enc_points = [p for p in points if p[0] == encoder]
        ez = np.array([p[2] for p in enc_points])
        ef1 = np.array([p[3] for p in enc_points])
        r_p, _ = pearsonr(ez, ef1)
        within[encoder] = r_p

    # Across-encoder: collapse each encoder's per-tag |preference| to one
    # unweighted mean, and pair it with a genuine per-encoder detection
    # metric -- token-level F1 of "is this token part of any entity"
    # (binary, all tags collapsed to one class vs O) -- rather than
    # averaging the per-tag F1 values used for the within-encoder analysis,
    # since that would just be another aggregation choice, not a
    # differently-defined metric. |preference| (not signed) is what showed
    # the real across-encoder signal.
    agg_z_abs = np.array([np.mean([abs(p[2]) for p in points if p[0] == e]) for e in bias_sigs])
    agg_f1 = np.array([load_binary_entity_f1(ner_sigs[e]) for e in bias_sigs])
    across_pearson, _ = pearsonr(agg_z_abs, agg_f1)

    # Naive pooled: every (encoder, tag) pair as its own independent point,
    # no within/across split -- the original approach, shown as a single
    # highlighted number when --pooled is set.
    pooled_zs = np.array([p[2] for p in points])
    pooled_f1s = np.array([p[3] for p in points])
    pooled_pearson, _ = pearsonr(pooled_zs, pooled_f1s)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.grid(True, color="#DDDDDD", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")

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
        for _, tag, z, f1 in enc_points:
            ax.annotate(
                tag,
                (z, f1),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7.5,
                color="#555555",
            )

    rho_desc = "averaged over all rho" if rho is None else f"at rho={rho:g}"
    ax.set_xlabel(f"Preference score, {rho_desc}")
    ax.set_ylabel("NER token-level F1")
    ax.set_title(f"{dataset}: encoder preference vs. NER token F1, by entity tag")
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
    # --dataset picked first (separately) since it selects the *defaults*
    # for the --*-sig arguments below -- argparse can't cross-reference one
    # argument's value into another's default within a single parse.
    pre = argparse.ArgumentParser(add_help=False)
    # Listed explicitly rather than derived from _DEFAULT_SIGS: that dict is
    # empty (no signature survived the dora->forge move), which would leave
    # --dataset with no valid choice at all. These are the datasets carrying
    # BIO tags in LABEL_DISPLAY_NAMES, which is what this figure needs.
    pre.add_argument("--dataset", default="wikiann", choices=["wikiann", "conll2003"])
    pre_args, _ = pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[pre])
    add_sig_args(parser, dataset_for_defaults=pre_args.dataset)
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--rho_average", action="store_true", help="average the z-score over all available rho instead of using a single one")
    parser.add_argument("--pooled", action="store_true", help="show a single large naive-pooled Pearson r instead of the within/across table")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    bias_sigs, ner_sigs = sigs_from_args(args)
    rho = None if args.rho_average else args.rho
    rho_tag = "rhoavg" if args.rho_average else f"rho{args.rho:g}"
    pooled_tag = "_pooled" if args.pooled else ""
    output = args.output or ROOT / "outputs/analysis/ner_correlation" / f"ner_correlation_{args.dataset}_{rho_tag}{pooled_tag}.pdf"
    main(args.dataset, bias_sigs, ner_sigs, rho, args.pooled, output)
