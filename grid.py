"""Run simple override sweeps defined in `grid.yaml`.

The YAML file must expose two top-level keys:

```
baseline:
  - data.train.subset=0.1
  - train.epochs=5

sweep:
  - - model.loss.l_s=0.01
    - model.loss.l_tv=5.0
  - - model.loss.l_s=0.05
    - model.loss.l_tv=2.0
```

Each list under `sweep` is appended to the baseline overrides and executed
once via `dora run`. The script prints a compact summary of
successes/failures and the run signatures for each sweep setting. You can
also call `--combine-plots` to build a single grid PNG from existing
`outputs/plots/<sig>/rates_cath.png` files using `.argv.json` as titles.
"""

import json
import math
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import yaml
from tabulate import tabulate
from tqdm import tqdm


CONFIG_PATH = Path("grid.yaml")
RUNS_DIR = Path("outputs/grid_runs")


def load_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    baseline = data.get("baseline", []) or []
    sweep = data.get("sweep", []) or []
    if not sweep:
        raise ValueError("No sweep entries defined in config")
    return baseline, sweep


def run_and_capture_signature(cmd, pbar):
    signature = None
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            pbar.write(line)
        if "Exp signature:" in line:
            signature = line.split("Exp signature:")[-1].strip()
    if not signature:
        raise RuntimeError("Experiment signature not found in output")
    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return signature


# ---------------------------------------------------------------------------
# Plot aggregation
# ---------------------------------------------------------------------------
def combine_cath_plots(
    plots_root: Path = Path("outputs/plots"),
    output_path: Path = Path("outputs/plots/combined_rates_cath.png"),
    max_cols: int = 3,
):
    """Combine all rates_cath.png plots into a single grid PNG with titles from .argv.json."""
    entries = []
    for exp_dir in sorted(plots_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        img_path = exp_dir / "rates_cath.png"
        argv_path = exp_dir / ".argv.json"
        if img_path.exists() and argv_path.exists():
            try:
                with argv_path.open("r", encoding="utf-8") as fh:
                    args = json.load(fh)
                title = " ".join(str(a) for a in args) if isinstance(args, (list, tuple)) else str(args)
            except Exception:
                title = exp_dir.name
            entries.append((img_path, title))

    if not entries:
        print(f"No plots found under {plots_root}")
        return

    cols = min(max_cols, len(entries))
    rows = math.ceil(len(entries) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]

    for ax in axes:
        ax.axis("off")

    for ax, (img_path, title) in zip(axes, entries):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(textwrap.fill(title, 60), fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved combined plot to {output_path}")


def main():
    try:
        baseline, sweep = load_config(CONFIG_PATH)
    except (FileNotFoundError, ValueError) as exc:
        print(f"⚠️ {exc}")
        return

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    total_runs = len(sweep)
    with tqdm(total=total_runs, desc="Grid sweep", unit="run") as pbar:
        for overrides in sweep:
            setting_overrides = list(baseline) + list(overrides)
            setting_label = " ".join(overrides) if overrides else "<no overrides>"

            pbar.write(f"\n=== Setting: {setting_label} ===")
            pbar.write(f"Baseline overrides: {baseline}")
            pbar.write(f"Sweep overrides   : {overrides}")

            cmd = ["dora", "run"] + setting_overrides
            signatures = []
            failures = 0
            try:
                sig = run_and_capture_signature(cmd, pbar)
                signatures.append(sig)
                pbar.write(f"    ✓ Run signature: {sig}")
            except subprocess.CalledProcessError as exc:
                failures += 1
                pbar.write(f"    ⚠️ Run failed: {exc}")
            except RuntimeError as exc:
                failures += 1
                pbar.write(f"    ⚠️ {exc}")
            finally:
                pbar.update(1)

            summary_rows.append(
                {
                    "setting": setting_label,
                    "runs": len(signatures),
                    "failures": failures,
                    "signatures": ", ".join(signatures) if signatures else "",
                }
            )

    if not summary_rows:
        print("No runs executed.")
        return

    table = tabulate(summary_rows, headers="keys", tablefmt="github", floatfmt=".4f")
    print("\nSummary table:\n")
    print(table)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = RUNS_DIR / f"grid_results_{timestamp}.md"
    output_content = f"Baseline overrides: {', '.join(baseline) if baseline else '<none>'}\n\n{table}\n"
    output_path.write_text(output_content, encoding="utf-8")
    print(f"\nSaved table view to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grid sweep runner and plot combiner.")
    parser.add_argument("--combine-plots", action="store_true", help="Combine rates_cath.png plots into one image.")
    parser.add_argument("--plots-root", type=Path, default=Path("outputs/plots"), help="Root directory containing plot subfolders.")
    parser.add_argument("--output", type=Path, default=Path("outputs/plots/combined_rates_cath.png"), help="Output path for combined plot.")
    parser.add_argument("--max-cols", type=int, default=3, help="Maximum columns in the combined plot grid.")
    args = parser.parse_args()

    if args.combine_plots:
        combine_cath_plots(args.plots_root, args.output, max_cols=args.max_cols)
    else:
        main()
