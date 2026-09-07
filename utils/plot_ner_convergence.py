"""NER MLP probe training-progress curves, one line per (encoder, B/I) pair,
for a single dataset -- token-level F1 macro-averaged separately across all
B-* tags and all I-* tags per epoch (span-level has no B/I distinction, since
it collapses B/I into whole entity spans). Independent of the bias-test rho
-- rho is a retention-rate parameter for the exact preference test on the
rationale-selection task and plays no role in training the NER probe, which
has its own separate per-epoch training loop.

Usage: python3 utils/plot_ner_convergence.py [--dataset conll2003] [--output PATH]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.forge_paths import run_dir  # noqa: E402
from utils.plot_ner_correlation import _DEFAULT_SIGS, ENCODER_COLOR, ENCODER_DISPLAY, ENCODER_MARKER  # noqa: E402

BIO_LINESTYLE = {"B": "-", "I": "--"}


def add_ner_sig_args(parser: argparse.ArgumentParser, dataset_for_defaults: str = "conll2003") -> None:
    defaults = _DEFAULT_SIGS.get(dataset_for_defaults, {"ner": {}})["ner"]
    for enc in ("sbert", "e5", "llm"):
        parser.add_argument(
            f"--{enc}-sig", default=defaults.get(enc),
            help=f"NER-probe forge sig for {ENCODER_DISPLAY[enc]}",
        )


def ner_sigs_from_args(args: argparse.Namespace) -> dict[str, str]:
    ner_sigs = {enc: getattr(args, f"{enc}_sig") for enc in ("sbert", "e5", "llm")}
    missing = [f"--{enc}-sig" for enc, s in ner_sigs.items() if not s]
    if missing:
        raise ValueError(f"Missing required signature argument(s): {', '.join(missing)}")
    return ner_sigs


def load_history(sig: str) -> tuple[list[int], list[float], list[float]]:
    """Returns (epochs, macro-avg F1 over B-* tags, macro-avg F1 over I-* tags)."""
    history = json.loads((run_dir(sig) / "data/ner_report_history.json").read_text())
    epochs, b_f1s, i_f1s = [], [], []
    for entry in history:
        token = entry["token_level"]
        b_tags = [v["f1-score"] for k, v in token.items() if k.startswith("B-")]
        i_tags = [v["f1-score"] for k, v in token.items() if k.startswith("I-")]
        epochs.append(entry["epoch"])
        b_f1s.append(sum(b_tags) / len(b_tags))
        i_f1s.append(sum(i_tags) / len(i_tags))
    return epochs, b_f1s, i_f1s


def main(dataset: str, ner_sigs: dict[str, str], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.grid(True, color="#DDDDDD", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")

    for encoder, sig in ner_sigs.items():
        epochs, b_f1s, i_f1s = load_history(sig)
        for prefix, f1s in (("B", b_f1s), ("I", i_f1s)):
            ax.plot(
                epochs, f1s,
                color=ENCODER_COLOR[encoder],
                linestyle=BIO_LINESTYLE[prefix],
                marker=ENCODER_MARKER[encoder],
                markersize=5,
                linewidth=1.6,
                zorder=3,
            )

    ax.set_xlabel("epoch")
    ax.set_ylabel("token-level macro F1")
    ax.set_title(f"{dataset}: NER probe training progress, by encoder and B/I")

    encoder_handles = [
        Line2D([0], [0], color=ENCODER_COLOR[e], marker=ENCODER_MARKER[e], linewidth=1.6, label=ENCODER_DISPLAY[e])
        for e in ner_sigs
    ]
    bio_handles = [
        Line2D([0], [0], color="#555555", linestyle=BIO_LINESTYLE[p], linewidth=1.6, label=f"{p}-*")
        for p in ("B", "I")
    ]
    legend1 = ax.legend(handles=encoder_handles, title="encoder", frameon=False, loc="lower right")
    ax.add_artist(legend1)
    ax.legend(handles=bio_handles, title="tag type", frameon=False, loc="lower center")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dataset", default="conll2003", choices=["wikiann", "conll2003"])
    pre_args, _ = pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[pre])
    add_ner_sig_args(parser, dataset_for_defaults=pre_args.dataset)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    ner_sigs = ner_sigs_from_args(args)
    output = args.output or ROOT / "outputs/analysis" / f"conv_{args.dataset}.pdf"
    main(args.dataset, ner_sigs, output)
