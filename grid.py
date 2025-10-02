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
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import yaml
from tabulate import tabulate
from tqdm import tqdm
from ratcon.utils import should_disable_tqdm


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


def load_metrics_from_history(signature: str) -> Optional[Dict[str, float]]:
    history_path = XP_ROOT / signature / "history.json"
    for _ in range(HISTORY_POLL_RETRIES):
        if history_path.exists():
            with history_path.open("r", encoding="utf-8") as fh:
                try:
                    history = json.load(fh)
                except json.JSONDecodeError:
                    history = None
            if isinstance(history, list) and history:
                summary_metrics: Dict[str, float] = {}
                last_metrics: Dict[str, float] = {}

                for entry in reversed(history):
                    if not isinstance(entry, dict):
                        continue
                    if not last_metrics and "metrics" in entry:
                        metrics = entry.get("metrics", {})
                        if isinstance(metrics, dict) and metrics:
                            last_metrics = metrics
                    if not summary_metrics and "summary" in entry:
                        summary = entry.get("summary", {})
                        if isinstance(summary, dict):
                            best_metrics = summary.get("best_metrics", {})
                            if isinstance(best_metrics, dict):
                                for key, value in best_metrics.items():
                                    summary_metrics[f"best_{key}"] = value
                            if "best_epoch" in summary:
                                summary_metrics["best_epoch"] = summary.get("best_epoch")
                            if "best_f1" in summary and "best_f1" not in summary_metrics:
                                summary_metrics["best_f1"] = summary.get("best_f1")
                            if "best_model" in summary:
                                summary_metrics["best_model"] = summary.get("best_model")

                    if summary_metrics and last_metrics:
                        break

                combined: Dict[str, float] = {}
                if last_metrics:
                    combined.update({f"final_{k}": v for k, v in last_metrics.items()})
                combined.update(summary_metrics)

                if combined:
                    return combined
        time.sleep(HISTORY_POLL_DELAY)
    print(f"⚠️ Unable to read metrics for experiment {signature}")
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


def run_and_capture_signature(cmd: List[str], pbar: tqdm) -> str:
    signature: Optional[str] = None
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert process.stdout is not None
    signature_pattern = re.compile(r"Exp signature:\s*([a-fA-F0-9]+)")

    for line in process.stdout:
        line = line.rstrip()
        if line:
            pbar.write(line)
        match = signature_pattern.search(line)
        if match:
            signature = match.group(1)

    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)

    if not signature:
        raise RuntimeError("Experiment signature not found in output")

    return signature


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

    summary_rows: List[Dict[str, object]] = []

    total_runs = len(sweep) * NUM_RUNS
    disable_progress = should_disable_tqdm()
    with tqdm(total=total_runs, desc="Grid sweep", unit="run", disable=disable_progress) as pbar:
        for overrides in sweep:
            setting_overrides = list(baseline) + list(overrides)
            if not any(str(ov).startswith("logging.metrics_only") for ov in setting_overrides):
                setting_overrides.append("logging.metrics_only=true")
            setting_label = format_setting(overrides)

            pbar.write(f"\n=== Setting: {setting_label} ===")
            pbar.write(f"Baseline overrides: {baseline}")
            pbar.write(f"Sweep overrides   : {overrides}")

            per_run_metrics: List[Dict[str, float]] = []
            failures = 0

            for _ in range(NUM_RUNS):
                cmd = ["dora", "run"] + setting_overrides
                signature = None
                try:
                    signature = run_and_capture_signature(cmd, pbar)
                except subprocess.CalledProcessError as exc:
                    failures += 1
                    pbar.write(f"    ⚠️ Run failed: {exc}")
                    pbar.update(1)
                    continue
                except RuntimeError as exc:
                    failures += 1
                    pbar.write(f"    ⚠️ {exc}")
                    pbar.update(1)
                    continue

                metrics = load_metrics_from_history(signature)
                if metrics:
                    per_run_metrics.append(metrics)
                    pbar.write("    ✓ Collected metrics " + str({k: round(v, 4) for k, v in metrics.items()}))
                else:
                    pbar.write(f"    ⚠️ Metrics unavailable for signature {signature}")

                pbar.update(1)

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
