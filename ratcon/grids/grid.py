"""Dora grid definition for RatCon.

Runs hyper-parameter sweeps via Slurm using the defaults declared in the Hydra
configuration (`slurm` section).
"""

from __future__ import annotations

from dora import Explorer, Launcher

@Explorer
def explorer(launcher: Launcher):
    launcher = launcher.bind({"data.train.subset":0.1})
    for l_comp in [0.01, 0.1, 1.0]:
        for l_s in [0.01, 0.1, 1.0]:
            for l_tv in [0.01, 0.1, 1.0]:
                launcher({
                    "model.loss.l_comp": l_comp,
                    "model.loss.l_s": l_s,
                    "model.loss.l_tv": l_tv,
                })
