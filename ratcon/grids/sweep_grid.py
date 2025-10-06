"""Dora grid definition that mirrors the local YAML-driven sweep."""

from pathlib import Path

import treetable as tt
import yaml
from dora import Explorer, Launcher

CONFIG_PATH = Path(__file__).resolve().parents[2] / "grid.yaml"

def _ensure_str_list(values):
    tokens = []
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
def load_yaml_sweep(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    baseline_raw = data.get("baseline", [])
    sweep_raw = data.get("sweep", [])

    baseline = _ensure_str_list(
        baseline_raw if isinstance(baseline_raw, (list, tuple)) else [baseline_raw]
    )

    sweep = []
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
        raise ValueError("No sweep entries defined in grid.yaml")

    return baseline, sweep


class YAMLExplorer(Explorer):
    metric_names = ("precision", "recall", "f1")

    def get_grid_metrics(self):
        return [
            tt.group(
                "best",
                [
                    tt.leaf("best_epoch"),
                    *(tt.leaf(name, ".4f") for name in self.metric_names),
                ],
                align=">")
        ]

    def process_history(self, history):
        best_epoch = None
        best_metrics = {name: None for name in self.metric_names}

        for entry in history:
            if isinstance(entry, dict) and "summary" in entry:
                summary = entry["summary"]
                best_epoch = summary.get("best_epoch")
                metrics = summary.get("best_metrics", {})
                for name in self.metric_names:
                    if name in metrics:
                        best_metrics[name] = metrics[name]
                break

        result = {"best": {"best_epoch": best_epoch}}
        result["best"].update(best_metrics)
        return result


@YAMLExplorer
def explorer(launcher):
    baseline, sweep = load_yaml_sweep(CONFIG_PATH)

    configured_launcher = launcher.bind(baseline) if baseline else launcher

    for overrides in sweep:
        overrides = list(overrides)
        if not any(str(ov).startswith("logging.metrics_only") for ov in overrides + baseline):
            overrides.append("logging.metrics_only=true")
        configured_launcher(overrides)
