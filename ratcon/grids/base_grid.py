"""Dora grid definition for RatCon.

Runs hyper-parameter sweeps via Slurm using the defaults declared in the Hydra
configuration (`slurm` section).
"""

from __future__ import annotations

from dora import Explorer, Launcher
import treetable as tt


class RatConExplorer(Explorer):
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


@RatConExplorer
def explorer(launcher: Launcher):
    launcher = launcher.bind({"data.train.subset": 0.1})
    for l_comp in [0.01, 0.1, 1.0]:
        for l_s in [0.01, 0.1, 1.0]:
            for l_tv in [0.01, 0.1, 1.0]:
                launcher({
                    "model.loss.l_comp": l_comp,
                    "model.loss.l_s": l_s,
                    "model.loss.l_tv": l_tv,
                })
