"""Re-key stored experiments after `ner:` became `tagger:` and left the signature.

The tagger's hyperparameters were part of every selector and oracle signature
even though nothing outside tagger/ reads them -- so bumping the tagger's
hidden size forked experiments that never used it. Adding `tagger.*` to
forge.exclude fixes that, but it also changes every signature.

Nothing a stored run *computed* depended on those keys, so the results stay
valid; only their address changes. This rewrites each experiment's config.yaml
into the new form and moves the directory to the recomputed signature, rather
than orphaning the runs and recomputing them.

Skips any experiment with a `running` run -- migrating a directory out from
under a live process would break it. Safe to re-run: an experiment already in
its correct location is left alone.
"""
import json
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from forge.core import canonical_signature  # noqa: E402

XPS = ROOT / "outputs" / "xps"
NEW_EXCLUDE = ["forge.*", "runtime.*", "tagger.*",
               "train.no_train", "train.continue", "train.checkpoint_path"]


def migrated(cfg):
    cfg = cfg.copy()
    if "ner" in cfg:
        cfg.tagger = cfg.ner
        del cfg["ner"]
    cfg.forge.exclude = list(NEW_EXCLUDE)
    return cfg


def main(apply: bool) -> int:
    moved = skipped = already = 0
    for cfg_path in sorted(XPS.glob("*/config.yaml")):
        xp = cfg_path.parent
        running = [m for m in xp.glob("*/meta.json")
                   if json.loads(m.read_text()).get("status") == "running"]
        cfg = OmegaConf.load(cfg_path)
        new_cfg = migrated(cfg)
        new_sig = canonical_signature(new_cfg)
        if new_sig == xp.name:
            already += 1
            continue
        if running:
            print(f"  SKIP {xp.name}: {len(running)} run(s) still running")
            skipped += 1
            continue
        dest = XPS / new_sig
        print(f"  {xp.name} -> {new_sig}"
              f"  ({len(list(xp.glob('*/meta.json')))} runs)"
              + ("" if apply else "   [dry run]"))
        if not apply:
            continue
        if dest.exists():   # merge: same experiment reached from both spellings
            for run in [d for d in xp.iterdir() if d.is_dir()]:
                shutil.move(str(run), str(dest / run.name))
            OmegaConf.save(new_cfg, dest / "config.yaml")
            shutil.rmtree(xp)
        else:
            OmegaConf.save(new_cfg, cfg_path)
            xp.rename(dest)
        moved += 1
    print(f"\n{moved} migrated, {already} already current, {skipped} skipped (running)")
    return 0


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
