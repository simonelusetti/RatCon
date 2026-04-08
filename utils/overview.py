from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
XPS_DIR = PROJECT_ROOT / "outputs" / "xps"
OUT_ROOT = PROJECT_ROOT / "outputs" / "utils" / "overview"
DEFAULT_CFG = PROJECT_ROOT / "src" / "conf" / "default.yaml"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_run_id(overrides: list[str]) -> int | None:
    for item in reversed(overrides):
        if item.startswith("run="):
            return int(item.split("=", 1)[1])
    return None


def override_key(item: str) -> str:
    key = item.split("=", 1)[0]
    return key.lstrip("+")


def matches_exclude(key: str, pattern: str) -> bool:
    if pattern.endswith(".*"):
        base = pattern[:-2]
        return key == base or key.startswith(f"{base}.")
    return key == pattern


def load_dora_exclude() -> list[str]:
    if not DEFAULT_CFG.exists():
        return []
    with DEFAULT_CFG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dora_cfg = cfg.get("dora", {}) if isinstance(cfg, dict) else {}
    excludes = dora_cfg.get("exclude", []) if isinstance(dora_cfg, dict) else []
    return [str(x) for x in excludes]


def filtered_overrides(overrides: list[str], exclude_patterns: list[str]) -> list[str]:
    kept: list[str] = []
    for item in overrides:
        key = override_key(item)
        if key == "run":
            continue
        if any(matches_exclude(key, pattern) for pattern in exclude_patterns):
            continue
        kept.append(item)
    return kept


def group_key(overrides: list[str], exclude_patterns: list[str]) -> tuple[str, ...]:
    return tuple(sorted(filtered_overrides(overrides, exclude_patterns)))


def label_from_overrides(overrides: list[str], exclude_patterns: list[str]) -> str:
    kept = filtered_overrides(overrides, exclude_patterns)
    return "\n".join(kept) if kept else "<default>"


def default_train_epochs() -> int:
    if not DEFAULT_CFG.exists():
        return 10
    with DEFAULT_CFG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    try:
        return int(train_cfg.get("epochs", 10))
    except (TypeError, ValueError):
        return 10


def expected_checkpoint(sig_dir: Path) -> Path | None:
    metrics_path = sig_dir / "metrics_details.json"
    epoch = default_train_epochs()

    if metrics_path.exists():
        try:
            payload = load_json(metrics_path)
            training = payload.get("training", {}) if isinstance(payload, dict) else {}
            epoch = int(training.get("epochs_target", epoch))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    ckpt = sig_dir / "state" / "models" / f"model_{epoch}.pth"
    if not ckpt.exists():
        return None
    return ckpt


def needs_eval(sig_dir: Path) -> bool:
    """Check if a run needs an eval rerun based on eval-generated artifacts."""
    data_dir = sig_dir / "data"
    stsb_path = sig_dir / "data" / "stsb.json"
    selections_path = data_dir / "selections.json"
    chi_square_path = data_dir / "chi_square.json"

    # STS-B should always be produced when eval is enabled.
    if not stsb_path.exists():
        return True

    # Chi-square exists only for labeled datasets where selections are produced.
    if selections_path.exists() and not chi_square_path.exists():
        return True

    return False


def rerun_eval(sig: str, sig_dir: Path, ckpt: Path | None, dry_run: bool) -> bool:
    """Re-run evaluation for a signature. Returns True if eval ran successfully."""
    if ckpt is None:
        print(f"Skipping eval rerun for {sig}: expected final checkpoint not found in {sig_dir / 'state/models'}")
        return False
    cmd = [
        "dora",
        "run",
        "--from_sig",
        sig,
        "train.no_train=true",
        f"train.checkpoint_path={ckpt.name}",
        "runtime.grid=false",
        "runtime.eval.skip_stsb=false",
        "runtime.eval.sweep_range=[0.1,1.0,10]",
    ]
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return False
    print(f"Running evaluation for {sig}...")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, check=False)
    if proc.returncode != 0:
        print(f"WARNING: Eval rerun failed for {sig}")
        return False
    print(f"  ✓ Evaluation completed for {sig}")
    return True


@dataclass
class RunData:
    sig: str
    sig_dir: Path
    overrides: list[str]
    run_id: int | None
    train_history: list[dict[str, float]]
    eval_history: list[dict[str, float]]
    chi_square: dict[str, Any] | None
    stsb: dict[str, Any] | None
    selections: dict[str, Any] | None


def load_run(sig_dir: Path) -> RunData:
    argv_path = sig_dir / ".argv.json"
    loss_history_path = sig_dir / "data" / "loss_history.json"
    chi_square_path = sig_dir / "data" / "chi_square.json"
    stsb_path = sig_dir / "data" / "stsb.json"
    selections_path = sig_dir / "data" / "selections.json"
    if not argv_path.exists():
        raise FileNotFoundError(f"Missing {argv_path}")

    overrides = [str(x) for x in load_json(argv_path)]
    loss_data = load_json(loss_history_path) if loss_history_path.exists() else {}
    chi_square = load_json(chi_square_path) if chi_square_path.exists() else None
    stsb = load_json(stsb_path) if stsb_path.exists() else None
    selections = load_json(selections_path) if selections_path.exists() else None

    # Load loss histories from dedicated loss_history.json
    train_history = loss_data.get("train", []) if isinstance(loss_data, dict) else []
    eval_history = loss_data.get("eval", []) if isinstance(loss_data, dict) else []
    if not isinstance(train_history, list) or not isinstance(eval_history, list):
        raise ValueError(f"Invalid history format in {loss_history_path}")

    return RunData(
        sig=sig_dir.name,
        sig_dir=sig_dir,
        overrides=overrides,
        run_id=parse_run_id(overrides),
        train_history=[{str(k): float(v) for k, v in d.items()} for d in train_history],
        eval_history=[{str(k): float(v) for k, v in d.items()} for d in eval_history],
        chi_square=chi_square,
        stsb=stsb,
        selections=selections,
    )


def mean_std_curves(curves: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    max_len = max(len(c) for c in curves)
    arr = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, c in enumerate(curves):
        arr[i, : len(c)] = np.asarray(c, dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return mean, std


def plot_with_band(
    ax,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    label: str,
    linestyle: str = "-",
    alpha: float = 0.18,
) -> None:
    valid = np.isfinite(mean)
    if not np.any(valid):
        return
    xv = x[valid]
    yv = mean[valid]
    sv = std[valid]
    line, = ax.plot(xv, yv, marker="o", linewidth=2.0, linestyle=linestyle, label=label)
    band_valid = np.isfinite(sv)
    if np.any(band_valid):
        xb = xv[band_valid]
        yb = yv[band_valid]
        sb = sv[band_valid]
        ax.fill_between(xb, yb - sb, yb + sb, alpha=alpha, color=line.get_color())


@dataclass
class GroupData:
    key: tuple[str, ...]
    label: str
    runs: list[RunData]


def make_axes(n_items: int, ncols: int) -> tuple[Any, list[Any]]:
    nrows = max(1, math.ceil(n_items / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.2, nrows * 4.2))
    axes_arr = np.asarray(axes).reshape(-1)
    return fig, list(axes_arr)


def style_group_axis(ax, label: str, n_runs: int) -> None:
    ax.set_title(f"{label}\nn={n_runs}", fontsize=8, loc="left", fontfamily="monospace")
    ax.grid(True, alpha=0.2)


def plot_loss_overview(groups: list[GroupData], out_path: Path, ncols: int) -> None:
    fig, axes = make_axes(len(groups), ncols)
    for ax, group in zip(axes, groups):
        eval_curves = [[x["eval_loss"] for x in r.eval_history if "eval_loss" in x] for r in group.runs]
        eval_curves = [curve for curve in eval_curves if curve]
        if not eval_curves:
            style_group_axis(ax, group.label, len(group.runs))
            ax.set_xlabel("history step")
            ax.set_ylabel("eval loss")
            ax.text(0.5, 0.5, "no eval loss history", transform=ax.transAxes, ha="center", va="center")
            continue
        mean, std = mean_std_curves(eval_curves)
        x = np.arange(1, len(mean) + 1, dtype=float)
        plot_with_band(ax, x, mean, std, "eval mean+-std")
        style_group_axis(ax, group.label, len(group.runs))
        ax.set_xlabel("history step")
        ax.set_ylabel("eval loss")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)

    for ax in axes[len(groups):]:
        ax.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)





def extract_spearman_curve(stsb: dict[str, Any], keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    values = None
    for key in keys:
        if key in stsb:
            values = stsb[key]
            break
    if values is None:
        raise ValueError(f"Missing one of {keys} in stsb.json")

    if isinstance(values, dict):
        items = sorted(((float(k), float(v)) for k, v in values.items()), key=lambda kv: kv[0])
        x = np.asarray([k for k, _ in items], dtype=float)
        y = np.asarray([v for _, v in items], dtype=float)
        return x, y

    if isinstance(values, list):
        rhos = stsb.get("rhos")
        if not isinstance(rhos, list) or len(rhos) != len(values):
            raise ValueError("Invalid list spearman format in stsb.json")
        x = np.asarray([float(v) for v in rhos], dtype=float)
        y = np.asarray([float(v) for v in values], dtype=float)
        return x, y

    raise ValueError(f"Unsupported format for {keys} in stsb.json")


def plot_spearman_overview(groups: list[GroupData], out_path: Path, ncols: int) -> None:
    fig, axes = make_axes(len(groups), ncols)
    for ax, group in zip(axes, groups):
        selector_curves: list[np.ndarray] = []
        random_curves: list[np.ndarray] = []
        x_ref: np.ndarray | None = None

        for run in group.runs:
            if run.stsb is None:
                print(f"Skipping STS-B for {run.sig}: missing {run.sig_dir / 'data' / 'stsb.json'}")
                continue
            x, y_selector = extract_spearman_curve(run.stsb, ("ours_by_rho", "ours"))
            _, y_random = extract_spearman_curve(run.stsb, ("random_by_rho", "random"))
            if x_ref is None:
                x_ref = x
            elif x_ref.shape != x.shape or not np.allclose(x_ref, x, atol=1e-8, rtol=1e-8):
                raise ValueError(f"Spearman rho grid mismatch inside group: {group.label}")
            selector_curves.append(y_selector)
            random_curves.append(y_random)

        if not selector_curves or x_ref is None:
            style_group_axis(ax, group.label, len(group.runs))
            ax.set_xlabel("selection rate (rho)")
            ax.set_ylabel("spearman")
            ax.text(0.5, 0.5, "no stsb data", transform=ax.transAxes, ha="center", va="center")
            continue

        selector_mean, selector_std = mean_std_curves([c.tolist() for c in selector_curves])
        random_mean, random_std = mean_std_curves([c.tolist() for c in random_curves])
        plot_with_band(ax, x_ref, selector_mean, selector_std, "selector mean+-std", linestyle="-")
        plot_with_band(ax, x_ref, random_mean, random_std, "random mean+-std", linestyle="--", alpha=0.14)
        style_group_axis(ax, group.label, len(group.runs))
        ax.set_xlabel("selection rate (rho)")
        ax.set_ylabel("spearman")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)

    for ax in axes[len(groups):]:
        ax.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def extract_chi_square_curves(chi_square: dict[str, Any], metric: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows = chi_square.get("rows") if isinstance(chi_square, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Invalid chi_square.json format: expected non-empty rows list")

    rho_values: list[float] = []
    label_points: dict[str, list[float]] = {}
    for row in rows:
        rho = float(row.get("rho"))
        rho_values.append(rho)
        labels = row.get("labels", [])
        if not isinstance(labels, list):
            continue
        for item in labels:
            label = str(item.get("label"))
            if metric == "chi_square":
                p_value = float(item.get("p_value", 1.0))
                p_value = max(p_value, 1e-300)
                value = -math.log10(p_value)
            else:
                value = float(item.get("cramers_v", 0.0))
            label_points.setdefault(label, []).append(value)

    x = np.asarray(rho_values, dtype=float)
    curves = {label: np.asarray(values, dtype=float) for label, values in label_points.items()}
    return x, curves


def extract_selection_rate_curves(selections: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows = selections.get("selections_by_rho") if isinstance(selections, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Invalid selections.json format: expected non-empty selections_by_rho list")

    labels: set[str] = set()
    for row in rows:
        pred_counts = row.get("pred_counts", {})
        gold_counts = row.get("gold_counts", {})
        if isinstance(pred_counts, dict):
            labels.update(str(label) for label in pred_counts.keys())
        if isinstance(gold_counts, dict):
            labels.update(str(label) for label in gold_counts.keys())

    sorted_labels = sorted(labels)
    rho_values: list[float] = []
    label_points: dict[str, list[float]] = {label: [] for label in sorted_labels}

    for row in rows:
        rho_values.append(float(row.get("rho")))
        pred_counts_raw = row.get("pred_counts", {})
        gold_counts_raw = row.get("gold_counts", {})

        pred_counts = {str(k): float(v) for k, v in pred_counts_raw.items()} if isinstance(pred_counts_raw, dict) else {}
        gold_counts = {str(k): float(v) for k, v in gold_counts_raw.items()} if isinstance(gold_counts_raw, dict) else {}

        for label in sorted_labels:
            kept = pred_counts.get(label, 0.0)
            total = gold_counts.get(label, 0.0)
            value = (kept / total) if total > 0 else 0.0
            label_points[label].append(value)

    x = np.asarray(rho_values, dtype=float)
    curves = {label: np.asarray(values, dtype=float) for label, values in label_points.items()}
    return x, curves


def is_negative_label(label: str) -> bool:
    normalized = label.strip().lower()
    if normalized in {"o", "0", "false", "non_entity", "none", "negative", "neg"}:
        return True
    return False


def filter_negative_label_curves(chi_square: dict[str, Any], curves: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    negative_label_raw = chi_square.get("negative_label") if isinstance(chi_square, dict) else None
    negative_label = str(negative_label_raw).strip() if negative_label_raw is not None else None

    filtered: dict[str, np.ndarray] = {}
    for label, curve in curves.items():
        if negative_label is not None and label.strip() == negative_label:
            continue
        if is_negative_label(label):
            continue
        filtered[label] = curve
    return filtered


def plot_chi_square_overview(groups: list[GroupData], out_path: Path, ncols: int, metric: str) -> None:
    ylabel = "-log10(p)" if metric == "chi_square" else "Cramer's V"
    fig, axes = make_axes(len(groups), ncols)
    for ax, group in zip(axes, groups):
        x_ref: np.ndarray | None = None
        per_label_runs: dict[str, list[np.ndarray]] = {}

        for run in group.runs:
            if run.chi_square is None:
                continue
            x, label_curves = extract_chi_square_curves(run.chi_square, metric=metric)
            label_curves = filter_negative_label_curves(run.chi_square, label_curves)
            if x_ref is None:
                x_ref = x
            elif x_ref.shape != x.shape or not np.allclose(x_ref, x, atol=1e-8, rtol=1e-8):
                raise ValueError(f"Chi-square rho grid mismatch inside group: {group.label}")
            for label, curve in label_curves.items():
                per_label_runs.setdefault(label, []).append(curve)

        style_group_axis(ax, group.label, len(group.runs))
        ax.set_xlabel("selection rate")
        ax.set_ylabel(ylabel)

        if x_ref is None or not per_label_runs:
            ax.text(0.5, 0.5, "no chi-square data", transform=ax.transAxes, ha="center", va="center")
            continue

        for label, curves in sorted(per_label_runs.items(), key=lambda kv: kv[0]):
            mean, std = mean_std_curves([c.tolist() for c in curves])
            plot_with_band(ax, x_ref, mean, std, f"{label} (n={len(curves)})")

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)

    for ax in axes[len(groups):]:
        ax.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_selection_rates_overview(groups: list[GroupData], out_path: Path, ncols: int) -> None:
    fig, axes = make_axes(len(groups), ncols)
    for ax, group in zip(axes, groups):
        x_ref: np.ndarray | None = None
        per_label_runs: dict[str, list[np.ndarray]] = {}

        for run in group.runs:
            if run.selections is None:
                continue
            x, label_curves = extract_selection_rate_curves(run.selections)
            if x_ref is None:
                x_ref = x
            elif x_ref.shape != x.shape or not np.allclose(x_ref, x, atol=1e-8, rtol=1e-8):
                raise ValueError(f"Selection-rate rho grid mismatch inside group: {group.label}")

            for label, curve in label_curves.items():
                per_label_runs.setdefault(label, []).append(curve)

        style_group_axis(ax, group.label, len(group.runs))
        ax.set_xlabel("effective selection rate (rho)")
        ax.set_ylabel("selection rate")
        ax.set_ylim(0.0, 1.05)

        if x_ref is None or not per_label_runs:
            ax.text(0.5, 0.5, "no selections data", transform=ax.transAxes, ha="center", va="center")
            continue

        for label, curves in sorted(per_label_runs.items(), key=lambda kv: kv[0]):
            mean, std = mean_std_curves([c.tolist() for c in curves])
            plot_with_band(ax, x_ref, mean, std, f"{label} (n={len(curves)})")

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)

    for ax in axes[len(groups):]:
        ax.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_groups(runs: list[RunData], exclude_patterns: list[str]) -> list[GroupData]:
    grouped: dict[tuple[str, ...], list[RunData]] = {}
    labels: dict[tuple[str, ...], str] = {}
    for run in runs:
        key = group_key(run.overrides, exclude_patterns)
        grouped.setdefault(key, []).append(run)
        labels.setdefault(key, label_from_overrides(run.overrides, exclude_patterns))

    out: list[GroupData] = []
    for key, group_runs in grouped.items():
        sorted_runs = sorted(group_runs, key=lambda r: r.run_id if r.run_id is not None else 10**9)
        out.append(GroupData(key=key, label=labels[key], runs=sorted_runs))
    out.sort(key=lambda g: (g.label, len(g.runs)))
    return out


def load_overrides_for_sig(sig_dir: Path) -> list[str] | None:
    argv_path = sig_dir / ".argv.json"
    if not argv_path.exists():
        return None
    payload = load_json(argv_path)
    if not isinstance(payload, list):
        return None
    return [str(item) for item in payload]


def filter_sig_dirs_by_group_size(
    sig_dirs: list[Path],
    exclude_patterns: list[str],
    min_group_runs: int,
) -> list[Path]:
    if min_group_runs <= 1:
        return sig_dirs

    grouped: dict[tuple[str, ...], list[Path]] = {}
    skipped = 0
    for sig_dir in sig_dirs:
        overrides = load_overrides_for_sig(sig_dir)
        if overrides is None:
            skipped += 1
            continue
        key = group_key(overrides, exclude_patterns)
        grouped.setdefault(key, []).append(sig_dir)

    kept: list[Path] = []
    for members in grouped.values():
        if len(members) >= min_group_runs:
            kept.extend(members)

    kept_set = set(kept)
    ordered_kept = [sig_dir for sig_dir in sig_dirs if sig_dir in kept_set]

    print(
        "Signature filter by group size: "
        f"kept {len(ordered_kept)}/{len(sig_dirs)} signatures "
        f"with min_group_runs={min_group_runs}" +
        (f" (skipped {skipped} without .argv.json)" if skipped else "")
    )
    return ordered_kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render grouped overview figures (loss, chi-square, spearman) as mean+-std across runs."
    )
    parser.add_argument("--sigs", nargs="*", default=None, help="Optional list of signatures to include.")
    parser.add_argument("--rerun-eval", action="store_true", help="Force re-run eval on each selected signature.")
    parser.add_argument("--dry-run-eval", action="store_true", help="Print eval commands only.")
    parser.add_argument("--skip-auto-eval", action="store_true", help="Skip automatic eval for missing data.")
    parser.add_argument("--ncols", type=int, default=4, help="Grid columns for overview figures.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Custom output directory.")
    parser.add_argument(
        "--min-group-runs",
        type=int,
        default=1,
        help="Keep only groups with at least this many runs (after Dora excludes and run key filtering).",
    )
    parser.add_argument(
        "--only-multi-run-groups",
        action="store_true",
        help="Convenience flag for --min-group-runs=2.",
    )
    args = parser.parse_args()

    if args.dry_run_eval and not args.rerun_eval:
        raise ValueError("--dry-run-eval requires --rerun-eval")

    if args.min_group_runs < 1:
        raise ValueError("--min-group-runs must be >= 1")

    min_group_runs = max(args.min_group_runs, 2) if args.only_multi_run_groups else args.min_group_runs

    if args.sigs:
        sig_dirs = [XPS_DIR / sig for sig in args.sigs]
    else:
        sig_dirs = sorted([p for p in XPS_DIR.iterdir() if p.is_dir()], key=lambda p: p.name)

    for sig_dir in sig_dirs:
        if not sig_dir.exists():
            raise FileNotFoundError(f"Missing signature directory: {sig_dir}")

    exclude_patterns = load_dora_exclude()
    sig_dirs = filter_sig_dirs_by_group_size(sig_dirs, exclude_patterns, min_group_runs)
    if not sig_dirs:
        raise ValueError("No signatures left after group-size filtering.")

    # Auto-trigger eval for runs with missing data (unless skipped)
    if not args.skip_auto_eval:
        missing_eval = [sd for sd in sig_dirs if needs_eval(sd)]
        if missing_eval:
            print(f"Found {len(missing_eval)} runs with missing evaluation data.")
            print(f"Auto-triggering evaluation...")
            for sig_dir in missing_eval:
                rerun_eval(sig_dir.name, sig_dir, expected_checkpoint(sig_dir), dry_run=args.dry_run_eval)
            print()

    # Explicit eval rerun (if requested)
    if args.rerun_eval:
        print(f"Force re-running evaluation for all {len(sig_dirs)} selected signatures...")
        for sig_dir in sig_dirs:
            rerun_eval(sig_dir.name, sig_dir, expected_checkpoint(sig_dir), dry_run=args.dry_run_eval)
        print()

    # Load all runs, skip any that still fail
    runs: list[RunData] = []
    for sig_dir in sig_dirs:
        try:
            runs.append(load_run(sig_dir))
        except FileNotFoundError as exc:
            print(f"Skipping {sig_dir.name}: {exc}")
            continue

    if not runs:
        raise ValueError("No valid runs found after loading signatures.")

    groups = build_groups(runs, exclude_patterns)
    if not groups:
        raise ValueError("No experiment groups found.")

    out_root = args.output_dir or OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    # Keep overview output flat: remove legacy subfolders from older scripts.
    for child in out_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)

    plot_loss_overview(groups, out_root / "loss_overview.png", ncols=args.ncols)
    plot_selection_rates_overview(groups, out_root / "selection_rates_overview.png", ncols=args.ncols)
    plot_chi_square_overview(groups, out_root / "chi_square_overview.png", ncols=args.ncols, metric="chi_square")
    plot_chi_square_overview(groups, out_root / "cramers_v_overview.png", ncols=args.ncols, metric="cramers_v")
    plot_spearman_overview(groups, out_root / "spearman_overview.png", ncols=args.ncols)

    print(f"Saved overview figures to {out_root}")


if __name__ == "__main__":
    main()
