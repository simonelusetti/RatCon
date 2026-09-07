"""Resolve a forge signature to the run directory the plot scripts read from.

Under dora every experiment owned exactly one directory, so a signature was
a path: `outputs/xps/<sig>/data/foo.json`. Forge separates the two levels --
the *experiment* (the config hash) from the *runs* launched under it
(`outputs/xps/<sig>/<run-id>/`) -- so a signature no longer names a single
directory and the plot scripts have to go through this.

That split is what makes multi-seed sweeps convenient here: `runtime.seed` is
in `forge.exclude` (see conf/config.yaml), so three seeds of one
configuration are three *runs* of one experiment rather than three unrelated
experiments. Pass the experiment signature and `run_dirs` returns all three.

Every `--*-sig` flag accepts either level, and prefixes work the same way
they do in the forge CLI:

    1c3d45ee            the experiment -- all of its completed runs
    1c3d45ee/c8f79e25   one specific run
    1c3d           /    1c3d45ee/c8f      prefixes of either
"""
from __future__ import annotations

import sys
from pathlib import Path

from forge.core import ExperimentStore
from forge.matching import select_signatures

ROOT = Path(__file__).resolve().parent.parent


def _store() -> ExperimentStore:
    # Anchored to the repo root rather than the cwd, so the plot scripts work
    # from anywhere -- forge's own default is cwd-relative.
    return ExperimentStore(root=ROOT / "outputs")


def _known_signatures() -> list[str]:
    return sorted(
        f"{run.signature}  ({run.status})"
        for selection in _store().all_selections()
        for run in (selection.runs or [])
    )


def run_dirs(sig: str) -> list[Path]:
    """Run directories for *sig*, oldest launch first.

    An experiment-level signature resolves to its completed runs; naming a
    run explicitly (`<xp>/<run>`) returns that run whatever its status, since
    asking for it by name is unambiguous about intent.
    """
    store = _store()
    selections = select_signatures([sig], store=store)

    runs = [
        run
        for selection in selections
        for run in (
            selection.runs
            if selection.runs is not None
            else store.list_runs(selection.experiment.signature)
        )
    ]

    named_run = "/" in sig
    if not named_run:
        runs = [run for run in runs if run.status == "done"] or runs

    if not runs:
        known = _known_signatures()
        raise SystemExit(
            f"No run found for signature {sig!r} under {store.xps_dir}.\n"
            + ("Available runs:\n  " + "\n  ".join(known) if known
               else "No runs exist yet -- launch one with `forge run ...`.")
        )

    return [run.path for run in sorted(runs, key=lambda r: r.launched_on)]


def run_dir(sig: str) -> Path:
    """The single run directory for *sig*.

    Scripts that plot one curve per signature use this. If *sig* names an
    experiment with several runs (e.g. a multi-seed sweep), the most recent
    one is used and the choice is reported on stderr rather than made
    silently -- pass `<xp>/<run>` to pin it.
    """
    dirs = run_dirs(sig)
    if len(dirs) > 1:
        chosen = dirs[-1]
        print(
            f"note: signature {sig!r} matches {len(dirs)} runs; using the most "
            f"recent ({chosen.parent.name}/{chosen.name}). "
            f"Pass <xp>/<run> to select a different one.",
            file=sys.stderr,
        )
        return chosen
    return dirs[0]


def discover_strategies(dataset: str, tasks: tuple[str, ...] = ("rationale", "oracle")) -> list[dict]:
    """Every selection experiment for *dataset*, newest-config first.

    Shared by the grounding scatter and the bias heatmaps so they can never
    disagree about what exists. Returns one dict per experiment with its
    label, family, pooling, task, signature and completed run directories.

    The label carries a `[task]` suffix for anything that is not the plain
    trained selector: the brute-force oracle shares a family and pooling with
    the selector it is the ceiling for, so without it the two would collide
    on one key.
    """
    from forge.core import ExperimentStore
    from omegaconf import OmegaConf

    store = ExperimentStore(root=ROOT / "outputs")
    found = []
    for selection in store.all_selections():
        cfg = selection.experiment.config
        if str(OmegaConf.select(cfg, "data.dataset")) != dataset:
            continue
        task = str(OmegaConf.select(cfg, "task"))
        if task not in tasks:
            continue
        done = [r.path for r in (selection.runs or []) if r.status == "done"]
        if not done:
            continue
        family = str(OmegaConf.select(cfg, "data.encoder.family"))
        pooling = OmegaConf.select(cfg, "data.encoder.pooling")
        pooling = None if pooling is None else str(pooling)
        base = f"{family}/{pooling}" if pooling else family
        found.append({
            "label": base if task == "rationale" else f"{base} [{task}]",
            "family": family,
            "pooling": pooling,
            "task": task,
            "signature": selection.experiment.signature,
            "runs": done,
        })
    return found
