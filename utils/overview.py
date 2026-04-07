from __future__ import annotations

import argparse
import json
import math
import re
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
    """Check if a run needs evaluation (missing required data files)."""
    loss_history_path = sig_dir / "data" / "loss_history.json"
    stsb_path = sig_dir / "data" / "stsb.json"
    
    # Run needs eval if either critical file is missing
    return not loss_history_path.exists() or not stsb_path.exists()


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
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        print(f"WARNING: Eval rerun failed for {sig}")
        print(f"  stdout: {proc.stdout}")
        print(f"  stderr: {proc.stderr}")
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
    stsb: dict[str, Any] | None


def load_run(sig_dir: Path) -> RunData:
    argv_path = sig_dir / ".argv.json"
    details_path = sig_dir / "metrics_details.json"
    loss_history_path = sig_dir / "data" / "loss_history.json"
    stsb_path = sig_dir / "data" / "stsb.json"
    if not argv_path.exists():
        raise FileNotFoundError(f"Missing {argv_path}")
    if not details_path.exists():
        raise FileNotFoundError(f"Missing {details_path}")
    if not loss_history_path.exists():
        raise FileNotFoundError(f"Missing {loss_history_path} (loss history not found)")

    overrides = [str(x) for x in load_json(argv_path)]
    details = load_json(details_path)
    loss_data = load_json(loss_history_path)
    stsb = load_json(stsb_path) if stsb_path.exists() else None

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
        stsb=stsb,
    )


def mean_std_curves(curves: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    max_len = max(len(c) for c in curves)
    arr = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, c in enumerate(curves):
        arr[i, : len(c)] = np.asarray(c, dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return mean, std


def plot_with_band(ax, x: np.ndarray, mean: np.ndarray, std: np.ndarray, label: str) -> None:
    valid = np.isfinite(mean)
    if not np.any(valid):
        return
    xv = x[valid]
    yv = mean[valid]
    sv = std[valid]
    ax.plot(xv, yv, marker="o", linewidth=2.0, label=label)
    band_valid = np.isfinite(sv)
    if np.any(band_valid):
        xb = xv[band_valid]
        yb = yv[band_valid]
        sb = sv[band_valid]
        ax.fill_between(xb, yb - sb, yb + sb, alpha=0.18)


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
        mean, std = mean_std_curves(eval_curves)
        x = np.arange(1, len(mean) + 1, dtype=float)
        plot_with_band(ax, x, mean, std, "eval mean+-std")
        style_group_axis(ax, group.label, len(group.runs))
        ax.set_xlabel("history step")
        ax.set_ylabel("eval loss")
        ax.legend(fontsize=7)

    for ax in axes[len(groups):]:
        ax.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)





def extract_spearman_curve(stsb: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    ours = stsb.get("ours_by_rho", stsb.get("ours"))
    if ours is None:
        raise ValueError("Missing `ours_by_rho` in stsb.json")

    if isinstance(ours, dict):
        items = sorted(((float(k), float(v)) for k, v in ours.items()), key=lambda kv: kv[0])
        x = np.asarray([k for k, _ in items], dtype=float)
        y = np.asarray([v for _, v in items], dtype=float)
        return x, y

    if isinstance(ours, list):
        rhos = stsb.get("rhos")
        if not isinstance(rhos, list) or len(rhos) != len(ours):
            raise ValueError("Invalid list spearman format in stsb.json")
        x = np.asarray([float(v) for v in rhos], dtype=float)
        y = np.asarray([float(v) for v in ours], dtype=float)
        return x, y

    raise ValueError("Unsupported `ours_by_rho` format in stsb.json")


def plot_spearman_overview(groups: list[GroupData], out_path: Path, ncols: int) -> None:
    fig, axes = make_axes(len(groups), ncols)
    for ax, group in zip(axes, groups):
        curves: list[np.ndarray] = []
        x_ref: np.ndarray | None = None

        for run in group.runs:
            if run.stsb is None:
                raise FileNotFoundError(
                    f"Missing {run.sig_dir / 'data' / 'stsb.json'}; rerun eval with STS-B enabled to plot spearman overview."
                )
            x, y = extract_spearman_curve(run.stsb)
            if x_ref is None:
                x_ref = x
            elif x_ref.shape != x.shape or not np.allclose(x_ref, x, atol=1e-8, rtol=1e-8):
                raise ValueError(f"Spearman rho grid mismatch inside group: {group.label}")
            curves.append(y)

        mean, std = mean_std_curves([c.tolist() for c in curves])
        plot_with_band(ax, x_ref, mean, std, "selector mean+-std")
        style_group_axis(ax, group.label, len(group.runs))
        ax.set_xlabel("selection rate (rho)")
        ax.set_ylabel("spearman")
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
    args = parser.parse_args()

    if args.dry_run_eval and not args.rerun_eval:
        raise ValueError("--dry-run-eval requires --rerun-eval")

    if args.sigs:
        sig_dirs = [XPS_DIR / sig for sig in args.sigs]
    else:
        sig_dirs = sorted([p for p in XPS_DIR.iterdir() if p.is_dir()], key=lambda p: p.name)

    for sig_dir in sig_dirs:
        if not sig_dir.exists():
            raise FileNotFoundError(f"Missing signature directory: {sig_dir}")

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

    exclude_patterns = load_dora_exclude()
    groups = build_groups(runs, exclude_patterns)
    if not groups:
        raise ValueError("No experiment groups found.")

    out_root = args.output_dir or OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    plot_loss_overview(groups, out_root / "loss_overview.png", ncols=args.ncols)
    plot_spearman_overview(groups, out_root / "spearman_overview.png", ncols=args.ncols)

    print(f"Saved overview figures to {out_root}")


if __name__ == "__main__":
    main()
