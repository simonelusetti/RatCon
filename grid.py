"""Run configurable sweeps defined in `grid.yaml` and summarise metrics.

The YAML file must expose two top-level keys:

```
baseline:
  - data.train.subset=0.1
  - model.fourier.use=True

sweep:
  - - model.fourier.mode=lowpass
    - model.fourier.keep_ratio=0.3
  - - model.fourier.mode=highpass
    - model.fourier.keep_ratio=0.5
```

Each list under `sweep` is appended to the baseline overrides and executed
`NUM_RUNS` times via `dora run`. After every run the script identifies the newly
created experiment directory inside `outputs/xps`, extracts metrics from its
`history.json`, and accumulates them. Once all runs finish, per-setting averages
are rendered as a Markdown table printed to stdout and saved to
`outputs/grid_runs/grid_results_<timestamp>.md`.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import yaml
from tabulate import tabulate


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("grid.yaml")
NUM_RUNS: int = 3
XP_ROOT = Path("outputs/xps")
RUNS_DIR = Path("outputs/grid_runs")

# Seconds to wait between attempts when polling for a freshly written history.
HISTORY_POLL_DELAY = 2.0
HISTORY_POLL_RETRIES = 5


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def _ensure_str_list(values: Sequence[object]) -> List[str]:
    tokens: List[str] = []
    for item in values:
        if item is None:
            continue
        if isinstance(item, str):
            token = item.strip()
            if token:
                tokens.append(token)
        else:
            token = str(item).strip()
            if token:
                tokens.append(token)
    return tokens


def load_config(path: Path) -> tuple[List[str], List[List[str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    baseline_raw = data.get("baseline", [])
    sweep_raw = data.get("sweep", [])

    baseline = _ensure_str_list(baseline_raw if isinstance(baseline_raw, Sequence) else [baseline_raw])

    sweep: List[List[str]] = []
    for entry in sweep_raw:
        if isinstance(entry, (list, tuple)):
            tokens = _ensure_str_list(entry)
        elif isinstance(entry, str):
            tokens = _ensure_str_list(entry.split())
        else:
            tokens = _ensure_str_list([entry])
        if tokens:
            sweep.append(tokens)

    if not sweep:
        raise ValueError("No sweep entries defined in config")

    return baseline, sweep


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def list_xp_dirs() -> set[str]:
    if not XP_ROOT.exists():
        return set()
    return {p.name for p in XP_ROOT.iterdir() if p.is_dir()}


def load_metrics_from_history(xp_dir: str) -> Optional[Dict[str, float]]:
    history_path = XP_ROOT / xp_dir / "history.json"
    for _ in range(HISTORY_POLL_RETRIES):
        if history_path.exists():
            with history_path.open("r", encoding="utf-8") as fh:
                try:
                    history = json.load(fh)
                except json.JSONDecodeError:
                    history = None
            if isinstance(history, list) and history:
                metrics = history[-1].get("metrics", {})
                if metrics:
                    return metrics
        time.sleep(HISTORY_POLL_DELAY)
    print(f"⚠️ Unable to read metrics for experiment {xp_dir}")
    return None


def average_metrics(metric_dicts: Iterable[Dict[str, float]]) -> Dict[str, Optional[float]]:
    metric_dicts = list(metric_dicts)
    if not metric_dicts:
        return {}
    keys = sorted({k for d in metric_dicts for k in d})
    averages: Dict[str, Optional[float]] = {}
    for key in keys:
        values = [d[key] for d in metric_dicts if key in d and d[key] is not None]
        averages[key] = sum(values) / len(values) if values else None
    return averages


def format_setting(overrides: Sequence[str]) -> str:
    return " ".join(overrides) if overrides else "<no overrides>"


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        baseline, sweep = load_config(CONFIG_PATH)
    except (FileNotFoundError, ValueError) as exc:
        print(f"⚠️ {exc}")
        return

    XP_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    initial_dirs = list_xp_dirs()
    seen_dirs = set(initial_dirs)

    summary_rows: List[Dict[str, object]] = []

    total_settings = len(sweep)

    for idx, overrides in enumerate(sweep, start=1):
        setting_overrides = list(baseline) + list(overrides)
        setting_label = format_setting(overrides)

        print(f"\n=== Setting {idx}/{total_settings}: {setting_label} ===")
        print(f"Baseline overrides: {baseline}")
        print(f"Sweep overrides   : {overrides}")

        per_run_metrics: List[Dict[str, float]] = []
        failures = 0

        for run_idx in range(1, NUM_RUNS + 1):
            print(f"  → Run {run_idx}/{NUM_RUNS}")
            before_run = set(seen_dirs)

            cmd = ["dora", "run"] + setting_overrides
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                failures += 1
                print(f"    ⚠️ Run failed: {exc}")
                continue

            after_run = list_xp_dirs()
            new_dirs = sorted(after_run - before_run)
            seen_dirs.update(after_run)

            if not new_dirs:
                print("    ⚠️ No new experiment directory detected; skipping metric collection.")
                continue

            for xp_name in new_dirs:
                metrics = load_metrics_from_history(xp_name)
                if metrics:
                    per_run_metrics.append(metrics)
                    print("    ✓ Collected metrics", {k: round(v, 4) for k, v in metrics.items()})

        avg_metrics = average_metrics(per_run_metrics)
        row: Dict[str, object] = {
            "setting": setting_label,
            "runs": len(per_run_metrics),
            "failures": failures,
        }
        for metric_name, value in avg_metrics.items():
            row[f"avg_{metric_name}"] = value
        summary_rows.append(row)

    if not summary_rows:
        print("No successful runs to summarise.")
        return

    df = pd.DataFrame(summary_rows)
    display_df = df.fillna("")
    table = tabulate(display_df, headers="keys", tablefmt="github", floatfmt=".4f")
    print("\nSummary table:\n")
    print(table)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = RUNS_DIR / f"grid_results_{timestamp}.md"
    output_content = f"Baseline overrides: {', '.join(baseline) if baseline else '<none>'}\n\n{table}\n"
    output_path.write_text(output_content, encoding="utf-8")
    print(f"\nSaved table view to {output_path}")


if __name__ == "__main__":
    main()
