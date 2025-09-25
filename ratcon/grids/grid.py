# grid.py
import ratcon.grids.patch_local  # applies monkeypatch
from dora import Explorer, Launcher

@Explorer
def explorer(launcher: Launcher):
    for l_comp in [0.01, 0.1, 1.0]:
        for l_s in [0.01, 0.1, 1.0]:
            for l_tv in [0.01, 0.1, 1.0]:
                launcher({
                    "train.loss.l_comp": l_comp,
                    "train.loss.l_s": l_s,
                    "train.loss.l_tv": l_tv,
                })
