from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tabulate import tabulate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
XPS_DIR = PROJECT_ROOT / "outputs" / "xps"
OUT_DIR = PROJECT_ROOT / "outputs" / "aggregates"
OVERRIDES_FILE = Path(".argv.json")
METRICS_FILE = Path("metrics.json")
DETAILS_FILE = Path("metrics_details.json")
DEFAULT_EVAL_SWEEP_RANGE = "[0.1, 1.0, 10]"

_LABEL_SKIP_PREFIXES = (
    "runtime.",
    "array=",
    "train.no_train=",
    "train.continue=",
    "slurm.",
)


@dataclass(frozen=True)
class RunRecord:
    sig: str
    sig_dir: Path
    overrides: tuple[str, ...]
    run_id: int | None
    metrics: dict
    details: dict


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_overrides(sig_dir: Path) -> list[str]:
    path = sig_dir / OVERRIDES_FILE
    if not path.exists():
        return []
    data = load_json(path)
    if isinstance(data, list):
        return [str(item) for item in data]
    return []


def parse_run_id(overrides: list[str]) -> int | None:
    for item in reversed(overrides):
        if item.startswith("run="):
            try:
                return int(item.split("=", 1)[1])
            except ValueError:
                return None
    return None


def grouped_key(overrides: list[str]) -> tuple[str, ...]:
    return tuple(sorted(item for item in overrides if not item.startswith("run=")))


def display_label(overrides: list[str]) -> str:
    kept = [item for item in overrides if not item.startswith("run=") and not item.startswith(_LABEL_SKIP_PREFIXES)]
    return "\n".join(kept) if kept else "<default>"


def slugify(text: str, max_len: int = 120) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_.-")
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        slug = "group"
    return slug[:max_len]


def group_hash(key: tuple[str, ...]) -> str:
    digest = hashlib.sha1("\n".join(key).encode("utf-8")).hexdigest()
    return digest[:8]


def list_selected_signatures(selected: list[str] | None) -> list[Path]:
    if selected:
        return [XPS_DIR / sig for sig in selected]
    if not XPS_DIR.exists():
        return []
    return sorted([p for p in XPS_DIR.iterdir() if p.is_dir()], key=lambda p: p.name)


def load_run_record(sig_dir: Path) -> RunRecord | None:
    metrics_path = sig_dir / METRICS_FILE
    details_path = sig_dir / DETAILS_FILE
    argv_path = sig_dir / OVERRIDES_FILE
    if not metrics_path.exists() or not details_path.exists() or not argv_path.exists():
        return None

    metrics = load_json(metrics_path)
    details = load_json(details_path)
    overrides = load_overrides(sig_dir)
    return RunRecord(
        sig=sig_dir.name,
        sig_dir=sig_dir,
        overrides=tuple(overrides),
        run_id=parse_run_id(overrides),
        metrics=metrics if isinstance(metrics, dict) else {},
        details=details if isinstance(details, dict) else {},
    )


def latest_checkpoint_path(sig_dir: Path) -> Path | None:
    checkpoints = sorted(
        sig_dir.glob("state/models/model_*.pth"),
        key=lambda path: int(re.search(r"model_(\d+)\.pth$", path.name).group(1)) if re.search(r"model_(\d+)\.pth$", path.name) else -1,
    )
    if not checkpoints:
        return None
    return checkpoints[-1]


def run_eval(sig: str, sig_dir: Path, skip_stsb: bool = True, dry_run: bool = False) -> None:
    checkpoint = latest_checkpoint_path(sig_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {sig_dir / 'state/models'}")

    cmd = [
        "dora",
        "run",
        "--from_sig",
        sig,
        "train.no_train=true",
        f"train.checkpoint_path={checkpoint.name}",
        "runtime.grid=false",
        f"runtime.eval.sweep_range={DEFAULT_EVAL_SWEEP_RANGE}",
    ]
    if skip_stsb:
        cmd.append("runtime.eval.skip_stsb=true")

    if dry_run:
        print("DRY-RUN eval:", " ".join(cmd))
        return

    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, check=False)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def aggregate_series(series_list: list[list[float]]) -> tuple[list[float], list[float], list[int]]:
    if not series_list:
        return [], [], []

    max_len = max(len(series) for series in series_list)
    if max_len == 0:
        return [], [], []

    arr = torch.full((len(series_list), max_len), float("nan"), dtype=torch.float64)
    for row_idx, series in enumerate(series_list):
        if not series:
            continue
        arr[row_idx, : len(series)] = torch.tensor(series, dtype=torch.float64)

    finite = torch.isfinite(arr)
    counts = finite.sum(dim=0)
    safe = torch.where(finite, arr, torch.zeros_like(arr))
    denom = counts.clamp_min(1).to(torch.float64)
    mean = safe.sum(dim=0) / denom

    centered = torch.where(finite, arr - mean.unsqueeze(0), torch.zeros_like(arr))
    var = (centered**2).sum(dim=0) / denom
    std = torch.sqrt(var)

    mean = torch.where(counts > 0, mean, torch.full_like(mean, float("nan")))
    std = torch.where(counts > 0, std, torch.full_like(std, float("nan")))

    return mean.tolist(), std.tolist(), [int(x) for x in counts.tolist()]


def plot_history(
    out_path: Path,
    title: str,
    train_mean: list[float],
    train_std: list[float],
    eval_mean: list[float],
    eval_std: list[float],
    n_runs: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    panels = [
        (axes[0], train_mean, train_std, "Train loss", "train_loss"),
        (axes[1], eval_mean, eval_std, "Eval loss", "eval_loss"),
    ]

    for ax, mean, std, panel_title, ylabel in panels:
        if mean:
            x = torch.arange(1, len(mean) + 1, dtype=torch.float64)
            y = torch.tensor(mean, dtype=torch.float64)
            s = torch.tensor(std, dtype=torch.float64)
            valid = torch.isfinite(y)
            if bool(valid.any().item()):
                ax.plot(x[valid].tolist(), y[valid].tolist(), marker="o", linewidth=2.0, label=f"mean ({n_runs} runs)")
                band = valid & torch.isfinite(s)
                if bool(band.any().item()):
                    xb = x[band]
                    yb = y[band]
                    sb = s[band]
                    ax.fill_between(xb.tolist(), (yb - sb).tolist(), (yb + sb).tolist(), alpha=0.18)
        ax.set_title(panel_title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize="small")

    axes[1].set_xlabel("history step")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_scalar_bars(out_path: Path, title: str, stats: list[tuple[str, float, float, int]]) -> None:
    if not stats:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(stats) * 1.25), 6))
    labels = [item[0] for item in stats]
    means = [item[1] for item in stats]
    stds = [item[2] for item in stats]
    xs = range(len(stats))
    ax.bar(xs, means, yerr=stds, capsize=6, alpha=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("final eval loss")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def summarize_group(label: str, records: list[RunRecord], group_dir: Path) -> dict[str, object]:
    eval_histories = [record.details.get("eval_loss_history", []) for record in records]
    train_histories = [record.details.get("train_loss_history", []) for record in records]
    final_losses = []
    best_losses = []
    durations = []

    for record in records:
        final_loss = record.metrics.get("final_losses", {}).get("eval_loss")
        if isinstance(final_loss, (int, float)):
            final_losses.append(float(final_loss))
        eval_hist = record.details.get("eval_loss_history", [])
        if eval_hist:
            best = min((entry.get("eval_loss") for entry in eval_hist if isinstance(entry, dict) and "eval_loss" in entry), default=None)
            if best is not None:
                best_losses.append(float(best))
        duration = record.metrics.get("run_statistics", {}).get("duration_seconds")
        if isinstance(duration, (int, float)):
            durations.append(float(duration))

    eval_series = [
        [float(entry["eval_loss"]) for entry in history if isinstance(entry, dict) and "eval_loss" in entry]
        for history in eval_histories
    ]
    train_series = [
        [float(entry["train_loss"]) for entry in history if isinstance(entry, dict) and "train_loss" in entry]
        for history in train_histories
    ]

    train_mean, train_std, train_counts = aggregate_series(train_series)
    eval_mean, eval_std, eval_counts = aggregate_series(eval_series)

    def mean_std(values: list[float]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        tensor = torch.tensor(values, dtype=torch.float64)
        return float(tensor.mean().item()), float(tensor.std(unbiased=False).item())

    final_mean, final_std = mean_std(final_losses)
    best_mean, best_std = mean_std(best_losses)
    duration_mean, duration_std = mean_std(durations)

    group_dir.mkdir(parents=True, exist_ok=True)
    history_path = group_dir / "history.png"
    plot_history(
        history_path,
        title=label,
        train_mean=train_mean,
        train_std=train_std,
        eval_mean=eval_mean,
        eval_std=eval_std,
        n_runs=len(records),
    )

    scalar_path = group_dir / "final_eval_loss.png"
    plot_scalar_bars(
        scalar_path,
        title=f"Final eval loss: {label}",
        stats=[(label, final_mean, final_std, len(records))],
    )

    summary = {
        "label": label,
        "n_runs": len(records),
        "runs": [record.sig for record in records],
        "final_eval_loss_mean": final_mean,
        "final_eval_loss_std": final_std,
        "best_eval_loss_mean": best_mean,
        "best_eval_loss_std": best_std,
        "duration_seconds_mean": duration_mean,
        "duration_seconds_std": duration_std,
        "train_history_len": len(train_mean),
        "eval_history_len": len(eval_mean),
        "train_history_counts": train_counts,
        "eval_history_counts": eval_counts,
    }

    summary_path = group_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    label_inline = label.replace("\n", " | ")

    markdown = tabulate(
        [
            [
                label_inline,
                len(records),
                ", ".join(record.sig for record in records),
                f"{final_mean:.6f}",
                f"{final_std:.6f}",
                f"{best_mean:.6f}",
                f"{best_std:.6f}",
                f"{duration_mean:.1f}",
                f"{duration_std:.1f}",
            ]
        ],
        headers=[
            "label",
            "runs",
            "signatures",
            "final_eval_mean",
            "final_eval_std",
            "best_eval_mean",
            "best_eval_std",
            "duration_mean_s",
            "duration_std_s",
        ],
        tablefmt="github",
    )
    (group_dir / "summary.md").write_text(markdown + "\n", encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Dora experiment repeats grouped by overrides minus run=.")
    parser.add_argument(
        "--sigs",
        nargs="*",
        default=None,
        help="Optional list of signatures to aggregate. Defaults to all signatures in outputs/xps.",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=2,
        help="Only aggregate groups with at least this many runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write aggregated plots and summaries.",
    )
    parser.add_argument(
        "--rerun-eval",
        action="store_true",
        help="Re-run a Dora no-train eval pass for each selected signature before aggregating.",
    )
    parser.add_argument(
        "--dry-run-eval",
        action="store_true",
        help="Print eval commands without executing them (implies --rerun-eval semantics).",
    )
    args = parser.parse_args()

    selected_dirs = list_selected_signatures(args.sigs)
    if not selected_dirs:
        print("No experiment signatures found.")
        return

    groups: dict[tuple[str, ...], list[RunRecord]] = defaultdict(list)
    label_map: dict[tuple[str, ...], str] = {}

    for sig_dir in selected_dirs:
        if args.rerun_eval:
            run_eval(sig_dir.name, sig_dir, skip_stsb=True, dry_run=args.dry_run_eval)

        record = load_run_record(sig_dir)
        if record is None:
            continue
        key = grouped_key(list(record.overrides))
        label_map.setdefault(key, display_label(list(record.overrides)))
        groups[key].append(record)

    eligible_groups = [
        (key, sorted(records, key=lambda rec: rec.run_id if rec.run_id is not None else 10**9))
        for key, records in groups.items()
        if len(records) >= max(1, args.min_runs)
    ]
    eligible_groups.sort(key=lambda item: (label_map.get(item[0], ""), len(item[1]), item[0]))

    if not eligible_groups:
        print("No groups matched the requested minimum run count.")
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = args.output_dir or (OUT_DIR / f"aggregate_{timestamp}")
    out_root.mkdir(parents=True, exist_ok=True)

    overview_rows = []
    group_summaries = []

    for idx, (key, records) in enumerate(eligible_groups, start=1):
        label = label_map.get(key, "<default>")
        label_inline = label.replace("\n", " | ")
        group_id = f"{slugify(label)}-{group_hash(key)}"
        group_dir = out_root / group_id
        summary = summarize_group(label, records, group_dir)
        group_summaries.append(summary)
        overview_rows.append(
            [
                idx,
                label_inline,
                len(records),
                ", ".join(summary["runs"]),
                f"{summary['final_eval_loss_mean']:.6f}",
                f"{summary['final_eval_loss_std']:.6f}",
                f"{summary['best_eval_loss_mean']:.6f}",
                f"{summary['best_eval_loss_std']:.6f}",
                f"{summary['duration_seconds_mean']:.1f}",
            ]
        )

    overview_table = tabulate(
        overview_rows,
        headers=[
            "#",
            "label",
            "runs",
            "signatures",
            "final_eval_mean",
            "final_eval_std",
            "best_eval_mean",
            "best_eval_std",
            "duration_mean_s",
        ],
        tablefmt="github",
    )

    overview_path = out_root / "overview.md"
    overview_path.write_text(
        "# Aggregated experiment repeats\n\n"
        f"Source: {XPS_DIR}\n\n"
        f"Minimum runs per group: {max(1, args.min_runs)}\n\n"
        f"{overview_table}\n",
        encoding="utf-8",
    )

    print(f"Saved aggregated results to {out_root}")
    print(f"Overview: {overview_path}")
    print(overview_table)


if __name__ == "__main__":
    main()