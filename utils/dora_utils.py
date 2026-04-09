from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_json


XPS_DIR = PROJECT_ROOT / "outputs" / "xps"
DEFAULT_CFG = PROJECT_ROOT / "src" / "conf" / "default.yaml"


# ---------------------------------------------------------------------------
# Default config helpers
# ---------------------------------------------------------------------------

def _load_default_cfg() -> dict:
    if not DEFAULT_CFG.exists():
        return {}
    with DEFAULT_CFG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def default_train_epochs() -> int:
    cfg = _load_default_cfg()
    try:
        return int(cfg.get("train", {}).get("epochs", 10))
    except (TypeError, ValueError):
        return 10


def load_dora_exclude() -> list[str]:
    cfg = _load_default_cfg()
    excludes = cfg.get("dora", {}).get("exclude", [])
    return [str(x) for x in excludes]


# ---------------------------------------------------------------------------
# Override parsing and grouping
# ---------------------------------------------------------------------------

def parse_run_id(overrides: list[str]) -> int | None:
    for item in reversed(overrides):
        if item.startswith("run="):
            return int(item.split("=", 1)[1])
    return None


def override_key(item: str) -> str:
    return item.split("=", 1)[0].lstrip("+")


def matches_exclude(key: str, pattern: str) -> bool:
    if pattern.endswith(".*"):
        base = pattern[:-2]
        return key == base or key.startswith(f"{base}.")
    return key == pattern


def filtered_overrides(overrides: list[str], exclude_patterns: list[str]) -> list[str]:
    return [
        item for item in overrides
        if (key := override_key(item)) != "run"
        and not any(matches_exclude(key, p) for p in exclude_patterns)
    ]


def group_key(overrides: list[str], exclude_patterns: list[str]) -> tuple[str, ...]:
    return tuple(sorted(filtered_overrides(overrides, exclude_patterns)))


def label_from_overrides(overrides: list[str], exclude_patterns: list[str]) -> str:
    kept = filtered_overrides(overrides, exclude_patterns)
    return "\n".join(kept) if kept else "<default>"


# ---------------------------------------------------------------------------
# Signature directory helpers
# ---------------------------------------------------------------------------

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

    kept_set = {sd for members in grouped.values() if len(members) >= min_group_runs for sd in members}
    ordered_kept = [sig_dir for sig_dir in sig_dirs if sig_dir in kept_set]

    print(
        "Signature filter by group size: "
        f"kept {len(ordered_kept)}/{len(sig_dirs)} signatures "
        f"with min_group_runs={min_group_runs}" +
        (f" (skipped {skipped} without .argv.json)" if skipped else "")
    )
    return ordered_kept


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
    return ckpt if ckpt.exists() else None


def needs_eval(sig_dir: Path) -> bool:
    if expected_checkpoint(sig_dir) is None:
        return False  # no final checkpoint — training incomplete, skip

    overrides = load_overrides_for_sig(sig_dir)
    if overrides is not None and any(str(item).strip() == "runtime.eval.skip=true" for item in overrides):
        return False  # explicitly configured to skip evaluation

    data_dir = sig_dir / "data"
    spearman_artifact = data_dir / "spearman_curves.json"
    if not spearman_artifact.exists():
        return True

    count_artifacts = (
        data_dir / "selection_rate_curves.json",
        data_dir / "chi_square_curves.json",
        data_dir / "cramers_v_curves.json",
    )
    count_exists = [path.exists() for path in count_artifacts]

    # Unlabeled datasets may not produce count-based artifacts at all.
    if not any(count_exists):
        return False

    # If one exists, all should exist; otherwise the run is partially evaluated.
    return not all(count_exists)


def rerun_eval(sig: str, sig_dir: Path, ckpt: Path | None, dry_run: bool) -> bool:
    if ckpt is None:
        print(f"Skipping eval rerun for {sig}: expected final checkpoint not found in {sig_dir / 'state/models'}")
        return False
    cmd = [
        "dora", "run", "--from_sig", sig,
        "train.no_train=true",
        f"train.checkpoint_path={ckpt.name}",
        "runtime.grid=false",
        "runtime.eval.skip=false",
        "runtime.eval.sweep_range=[0.1,1.0,10]",
    ]
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return False
    print(f"Running evaluation for {sig} (checkpoint: {ckpt.name})...")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, check=False)
    if proc.returncode != 0:
        print(f"WARNING: Eval rerun failed for {sig}")
        return False
    print(f"  ✓ Evaluation completed for {sig}")
    return True


# ---------------------------------------------------------------------------
# Run / Group data model
# ---------------------------------------------------------------------------

@dataclass
class RunData:
    sig: str
    sig_dir: Path
    overrides: list[str]
    run_id: int | None


def load_run(sig_dir: Path) -> RunData:
    argv_path = sig_dir / ".argv.json"
    if not argv_path.exists():
        raise FileNotFoundError(f"Missing {argv_path}")

    overrides = [str(x) for x in load_json(argv_path)]

    return RunData(
        sig=sig_dir.name,
        sig_dir=sig_dir,
        overrides=overrides,
        run_id=parse_run_id(overrides),
    )


@dataclass
class GroupData:
    key: tuple[str, ...]
    label: str
    runs: list[RunData]


def build_groups(runs: list[RunData], exclude_patterns: list[str]) -> list[GroupData]:
    grouped: dict[tuple[str, ...], list[RunData]] = {}
    for run in runs:
        grouped.setdefault(group_key(run.overrides, exclude_patterns), []).append(run)

    out = [
        GroupData(
            key=key,
            label=label_from_overrides(group_runs[0].overrides, exclude_patterns),
            runs=sorted(group_runs, key=lambda r: r.run_id if r.run_id is not None else 10**9),
        )
        for key, group_runs in grouped.items()
    ]
    out.sort(key=lambda g: (g.label, len(g.runs)))
    return out
