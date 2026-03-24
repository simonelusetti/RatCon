from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Configure here
# ──────────────────────────────────────────────────────────────────────────────

EXPERIMENTS: list[str] = [
]

# Keys to strip from the overrides label (operational / irrelevant noise)
_SKIP_PREFIXES = (
    "runtime.",
    "array=",
    "train.no_train=",
    "train.continue=",
    "slurm.",
)

# ──────────────────────────────────────────────────────────────────────────────

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import yaml


XPS_DIR = Path(__file__).parent.parent / "outputs" / "xps"
PLOT_FILE = Path("plots/chi_square.png")
OVERRIDES_FILE = Path(".hydra/overrides.yaml")


def load_overrides(sig_dir: Path) -> list[str]:
    path = sig_dir / OVERRIDES_FILE
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        items = yaml.safe_load(f) or []
    return [str(x) for x in items]


def clean_label(sig: str, overrides: list[str]) -> str:
    kept = [
        o for o in overrides
        if not any(o.startswith(p) for p in _SKIP_PREFIXES)
    ]
    body = "\n".join(kept) if kept else "<default>"
    return f"{sig}\n{body}"


def collect_experiments(sigs: list[str]) -> list[tuple[str, Path, str]]:
    """Return list of (sig, plot_path, label) for experiments with a chi_square plot."""
    if sigs:
        candidates = [(s, XPS_DIR / s) for s in sigs]
    else:
        candidates = sorted(
            [(d.name, d) for d in XPS_DIR.iterdir() if d.is_dir()],
            key=lambda x: x[0],
        )

    results = []
    missing = []
    for sig, sig_dir in candidates:
        plot_path = sig_dir / PLOT_FILE
        if not plot_path.exists():
            missing.append(sig)
            continue
        overrides = load_overrides(sig_dir)
        label = clean_label(sig, overrides)
        results.append((sig, plot_path, label))

    if missing:
        print(f"Skipped (no chi_square plot): {', '.join(missing)}")

    return results


def make_grid(entries: list[tuple[str, Path, str]], out_path: Path, ncols: int) -> None:
    n = len(entries)
    if n == 0:
        print("No experiments to display.")
        return

    nrows = math.ceil(n / ncols)
    cell_size = 4.5          # inches per cell
    label_height = 1.0       # extra inches at the bottom of each cell for text

    fig_w = ncols * cell_size
    fig_h = nrows * (cell_size + label_height)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

    # Normalise axes to a flat list
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    elif nrows == 1 or ncols == 1:
        axes_flat = list(axes.flat if hasattr(axes, "flat") else axes)
    else:
        axes_flat = [ax for row in axes for ax in row]

    for ax, (sig, plot_path, label) in zip(axes_flat, entries):
        img = mpimg.imread(str(plot_path))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label, fontsize=7, fontfamily="monospace", loc="left",
                     pad=4, wrap=True)

    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout(pad=0.8)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}  ({n} experiments, {nrows}×{ncols} grid)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid view of chi_square plots across experiments.")
    ap.add_argument("--out", type=Path, default=Path("outputs/chi_square_overview.png"))
    ap.add_argument("--ncols", type=int, default=4, help="Number of columns in the grid")
    args = ap.parse_args()

    sigs = EXPERIMENTS
    entries = collect_experiments(sigs)
    if not entries:
        print("Nothing to show.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_grid(entries, args.out, ncols=args.ncols)


if __name__ == "__main__":
    main()
