"""Paper Figure 3: STS-B sufficiency test, 3 panels.
  left:   SBERT trained+evaluated on STS-B vs. a random-selection baseline.
  center: comparison across the 3 encoder backbones (SBERT/E5/Pythia),
          each trained+evaluated on STS-B.
  right:  cross-corpus generalization -- SBERT selectors trained on STS-B,
          WikiANN, CoNLL-2003, Movie Reviews, and CoNLL-2000, all evaluated
          on STS-B.

Each curve is the mean over several training seeds, with a shaded ±std band
(paper: "mean over three runs"). Reads outputs/xps/<sig>/data/spearman_curves.json,
written by SelectorTrainer.final_eval() -> src/retrival_fun.py's run_stsb_sweep.

NO DEFAULT SIGNATURES -- this repo has not yet run the STS-B training sweep
(dora run data.dataset=stsb/wikiann/conll2003/movie_rationales/conll2000
data.encoder.family=sbert/e5/llm, x3 seeds each). Every curve's sig list must
be passed explicitly via CLI (see --help); this script raises a clear error
rather than silently plotting nothing if a sig list is empty.

Usage: python3 utils/plot_stsb_sufficiency.py \\
    --panel1-sigs SIG SIG SIG \\
    --panel2-sbert-sigs SIG SIG SIG --panel2-e5-sigs SIG SIG SIG --panel2-llm-sigs SIG SIG SIG \\
    --panel3-stsb-sigs SIG SIG SIG --panel3-wikiann-sigs SIG SIG SIG \\
    --panel3-conll2003-sigs SIG SIG SIG --panel3-movie-rationales-sigs SIG SIG SIG \\
    --panel3-conll2000-sigs SIG SIG SIG \\
    [--output PATH]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.plot_ner_correlation import ENCODER_COLOR, ENCODER_DISPLAY  # noqa: E402

DATASET_DISPLAY = {
    "stsb": "STS-B", "wikiann": "WikiANN", "conll2003": "CoNLL-03",
    "movie_rationales": "Movie Review", "conll2000": "CoNLL-00",
}
DATASET_COLOR = {
    "stsb": "#0072B2", "wikiann": "#E69F00", "conll2003": "#009E73",
    "movie_rationales": "#D55E00", "conll2000": "#CC79A7",
}


def load_curves(sig: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    path = ROOT / "outputs/xps" / sig / "data/spearman_curves.json"
    payload = json.loads(path.read_text())
    rho = np.array(payload["rho"], dtype=float)
    selector = np.array(payload["curves"]["selector"], dtype=float)
    random = np.array(payload["curves"]["random"], dtype=float)
    baseline = float(payload["baseline"]["value"])
    return rho, selector, random, baseline


def mean_std_over_seeds(sigs: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not sigs:
        raise ValueError(
            "No signatures provided for one of the curves -- pass real dora run "
            "signatures from the STS-B training sweep via the --panel*-sigs CLI "
            "arguments (see --help) before running this plot."
        )
    curves = [load_curves(sig) for sig in sigs]
    rho = curves[0][0]
    selectors = np.stack([c[1] for c in curves])
    randoms = np.stack([c[2] for c in curves])
    baselines = np.array([c[3] for c in curves])
    return rho, selectors.mean(0), selectors.std(0), randoms.mean(0), randoms.std(0)


def plot_with_band(ax, x, mean, std, label, color=None, linestyle="-", marker="o", alpha=0.18):
    line, = ax.plot(x, mean, marker=marker, linewidth=2.0, linestyle=linestyle, label=label, color=color)
    used_color = line.get_color()
    ax.fill_between(x, mean - std, mean + std, alpha=alpha, color=used_color)
    return used_color


def style_ax(ax) -> None:
    ax.grid(True, color="#DDDDDD", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")
    ax.set_xlabel("ρ")
    ax.set_ylabel("spearman")


def main(
    panel1_sigs: list[str],
    panel2_sigs: dict[str, list[str]],
    panel3_sigs: dict[str, list[str]],
    output: Path,
) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: SBERT on STS-B vs random
    style_ax(ax1)
    rho, sel_mean, sel_std, rand_mean, rand_std = mean_std_over_seeds(panel1_sigs)
    plot_with_band(ax1, rho, sel_mean, sel_std, "SBERT", color=ENCODER_COLOR["sbert"])
    plot_with_band(ax1, rho, rand_mean, rand_std, "SBERT random", color="#999999", linestyle="--", marker="x")
    ax1.set_title("STS-B Sufficiency Test")
    ax1.legend(frameon=False, fontsize=8)

    # Panel 2: encoder comparison, each trained+evaluated on STS-B
    style_ax(ax2)
    for family in ("sbert", "e5", "llm"):
        rho, sel_mean, sel_std, rand_mean, rand_std = mean_std_over_seeds(panel2_sigs[family])
        color = plot_with_band(ax2, rho, sel_mean, sel_std, ENCODER_DISPLAY[family], color=ENCODER_COLOR[family])
        plot_with_band(ax2, rho, rand_mean, rand_std, f"{ENCODER_DISPLAY[family]} random",
                        color=color, linestyle="--", marker="x", alpha=0.10)
    ax2.set_title("Different Encoders")
    ax2.legend(frameon=False, fontsize=7, ncol=2)

    # Panel 3: cross-corpus generalization (SBERT trained elsewhere, eval'd on STS-B)
    style_ax(ax3)
    random_curve_plotted = False
    for dataset in ("stsb", "wikiann", "conll2003", "movie_rationales", "conll2000"):
        rho, sel_mean, sel_std, rand_mean, rand_std = mean_std_over_seeds(panel3_sigs[dataset])
        plot_with_band(ax3, rho, sel_mean, sel_std, DATASET_DISPLAY[dataset], color=DATASET_COLOR[dataset])
        if not random_curve_plotted:
            # One shared random baseline (random selection doesn't depend on
            # which corpus the selector was trained on) rather than 5 near-
            # identical dashed lines cluttering the panel.
            plot_with_band(ax3, rho, rand_mean, rand_std, "random", color="#999999", linestyle="--", marker="x")
            random_curve_plotted = True
    ax3.set_title("Different Datasets (tested on STS-B)")
    ax3.legend(frameon=False, fontsize=7, ncol=2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    print(f"Saved plot to {output} (+ .pdf)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel1-sigs", nargs="+", default=[], metavar="SIG",
                         help="Panel 1: SBERT trained on STS-B, one sig per seed (paper uses 3)")
    for family in ("sbert", "e5", "llm"):
        parser.add_argument(f"--panel2-{family}-sigs", nargs="+", default=[], metavar="SIG",
                             help=f"Panel 2: {ENCODER_DISPLAY[family]} trained on STS-B, one sig per seed")
    for dataset in ("stsb", "wikiann", "conll2003", "movie_rationales", "conll2000"):
        parser.add_argument(f"--panel3-{dataset.replace('_', '-')}-sigs", nargs="+", default=[], metavar="SIG",
                             help=f"Panel 3: SBERT trained on {DATASET_DISPLAY[dataset]}, one sig per seed")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    panel1_sigs = args.panel1_sigs
    panel2_sigs = {family: getattr(args, f"panel2_{family}_sigs") for family in ("sbert", "e5", "llm")}
    panel3_sigs = {dataset: getattr(args, f"panel3_{dataset}_sigs") for dataset in
                    ("stsb", "wikiann", "conll2003", "movie_rationales", "conll2000")}

    output = args.output or ROOT / "outputs/analysis/stsb_sufficiency.png"
    main(panel1_sigs, panel2_sigs, panel3_sigs, output)
