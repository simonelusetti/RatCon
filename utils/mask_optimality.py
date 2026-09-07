"""How good is the selector's mask, compared to every other mask it could
have picked?

Everything else in this repo measures what the selector *selects* -- which
tags, how often. Nothing checks whether those tokens are actually a good
answer to the objective it was trained on. This does, directly: for each
sentence it enumerates the alternative masks of the same size, scores every
one of them, and reports the percentile the selector's own mask lands in.

The score is the training objective itself: cosine similarity between the
sentence embedding rebuilt from the kept tokens and the encoder's embedding
of the whole sentence. Higher is better. A selector that had learnt nothing
would sit near the 50th percentile; a perfect one at the 100th.

Cost
----
The expensive part is not the combinatorics -- itertools emits masks far
faster than they can be scored. It is that the mask enters the *attention*,
not just the pooling (see RationaleSelectorModel.forward), so every candidate
needs its own transformer forward. Masks are therefore scored in batches.

C(n, k) is tiny for a typical sentence (wikiann's median is 11 tokens, so 165
masks at rho=0.3) but explodes in the tail (n=32, rho=0.5 is 6e8). So:
exhaustive whenever C(n, k) <= --max-combinations, and a uniform sample of
--samples masks above that. Sampling is not a fallback to apologise for --
drawing uniformly from the same space makes the percentile an unbiased
estimate of the exhaustive one, with a standard error of at most
0.5/sqrt(samples) (~1.1% at the default 2000).

Control
-------
--control-draws random masks per sentence are scored through the *same*
function as the selector's, and their percentiles averaged. This must come
out at 0.5: it is the null the whole measurement rests on. Scoring them
through the same path (rather than just re-indexing the candidate scores)
is what makes it a real check -- it would catch the selector's mask being
scored inconsistently with its alternatives, which mere re-indexing cannot.

Averaging many draws matters. One draw per sentence has a standard deviation
of ~0.29, so over a couple of hundred sentences the control still wobbles by
several points and cannot distinguish "fine" from "subtly broken"; at 20
draws it pins down to well under a point.

Usage:
  python3 utils/mask_optimality.py --sig <experiment-or-run> [--rho 0.3]
      [--sentences 200] [--max-combinations 20000] [--samples 2000]
      [--device cpu] [--output PATH]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forge.core import ExperimentStore  # noqa: E402

from src.data import initialize_data  # noqa: E402
from src.selector import RationaleSelectorModel  # noqa: E402
from src.utils import checkpoint_epoch, remap_checkpoint_state_dict  # noqa: E402
from utils.forge_paths import run_dirs  # noqa: E402


def resolve_run(sig: str) -> tuple[Path, OmegaConf]:
    """Run directory plus the config of the experiment that owns it."""
    path = run_dirs(sig)[-1]
    store = ExperimentStore(root=ROOT / "outputs")
    cfg = store.load_run(path.parent.name, path.name).experiment.config
    return path, cfg


def load_selector(run_path: Path, cfg, encoder, device: str) -> RationaleSelectorModel:
    """Rebuild the selector for this run and load its final checkpoint."""
    checkpoints = sorted(
        (p for p in (run_path / "state" / "models").glob("model_*.pth")
         if checkpoint_epoch(p) is not None),
        key=checkpoint_epoch,
    )
    if not checkpoints:
        raise SystemExit(f"No checkpoint under {run_path / 'state/models'}.")

    with torch.no_grad():
        probe = torch.zeros(1, 4, dtype=torch.long, device=device)
        model_dim = encoder.token_embeddings(probe, torch.ones_like(probe)).shape[-1]

    model = RationaleSelectorModel(
        model_dim,
        loss_cfg=cfg.model.loss,
        selector_cfg=cfg.model.get("selector", None),
        sent_encoder=encoder,
    ).to(device)
    state = torch.load(checkpoints[-1], map_location=device)
    model.load_state_dict(remap_checkpoint_state_dict(state["model"], model.state_dict()))
    model.eval()
    print(f"loaded {checkpoints[-1].relative_to(ROOT)}")
    return model


def budget(rho: float, n: int) -> int:
    """Token budget k, matching selector.py's forward() exactly -- the whole
    comparison is void if the alternatives are a different size."""
    return max(1, int(round(rho * n))) if n > 0 else 0


def candidate_masks(n: int, k: int, max_combinations: int, samples: int, rng: random.Random):
    """Every k-subset of n positions, or a uniform sample of them.

    Returns (list of index tuples, exhaustive?).
    """
    total = math.comb(n, k)
    if total <= max_combinations:
        return list(itertools.combinations(range(n), k)), True
    return [tuple(rng.sample(range(n), k)) for _ in range(samples)], False


def percentile(scores: torch.Tensor, value: torch.Tensor) -> float:
    """Fraction of `scores` that `value` beats, counting ties as half.

    Midrank rather than a strict `<`: with ties (repeated tokens in a
    sentence can make two different masks score identically) a strict
    comparison systematically under-credits, and it would bias the control
    below 0.5 even when nothing is wrong.
    """
    return float((scores < value).float().mean() + 0.5 * (scores == value).float().mean())


@torch.no_grad()
def score_masks(encoder, ids_row: torch.Tensor, masks: torch.Tensor, chunk: int) -> torch.Tensor:
    """Cosine of each masked reconstruction against the full-sentence target.

    Re-encodes per mask on purpose: the mask is an attention mask, so the
    token states themselves change with it -- pooling the unmasked states
    would measure a different (and easier) problem.
    """
    length = ids_row.shape[0]
    full_attn = torch.ones(1, length, device=ids_row.device)
    target = encoder.pool_full(
        encoder.token_embeddings(ids_row.unsqueeze(0), full_attn), full_attn
    )

    out = []
    for start in range(0, masks.shape[0], chunk):
        block = masks[start:start + chunk]
        ids_rep = ids_row.unsqueeze(0).expand(block.shape[0], length)
        pooled = encoder.pool(encoder.token_embeddings(ids_rep, block), block)
        out.append(torch.nn.functional.cosine_similarity(pooled, target, dim=-1))
    return torch.cat(out)


def main(sig: str, rho: float, n_sentences: int, max_combinations: int, samples: int,
         chunk: int, control_draws: int, device: str, seed: int, output: Path) -> None:
    run_path, cfg = resolve_run(sig)
    family = str(OmegaConf.select(cfg, "data.encoder.family"))
    pooling = str(OmegaConf.select(cfg, "data.encoder.pooling"))
    print(f"run {run_path.parent.name}/{run_path.name}  ({family} x {pooling})")

    cfg = cfg.copy()
    cfg.runtime.device = device
    _, _, encoder, _, _, ds = initialize_data(
        cfg.data, cfg.runtime.data, None, device=device,
        keep_special=bool(cfg.model.get("keep_special", True)),
    )
    model = load_selector(run_path, cfg, encoder, device)

    rng = random.Random(seed)
    torch.manual_seed(seed)
    test = ds["test"]
    rows = min(n_sentences, len(test))

    selector_pct, random_pct, exhaustive_flags, sizes = [], [], [], []
    for i in range(rows):
        item = test[i]
        attn = torch.tensor(item["attn_mask"], device=device, dtype=torch.float)
        keep = attn > 0
        n = int(keep.sum())
        k = budget(rho, n)
        if n < 2 or k >= n:
            continue  # nothing to compare against: only one mask of that size

        ids_row = torch.tensor(item["ids"], device=device, dtype=torch.long)[keep]
        combos, exhaustive = candidate_masks(n, k, max_combinations, samples, rng)

        masks = torch.zeros(len(combos), n, device=device)
        idx = torch.tensor(combos, device=device, dtype=torch.long)
        masks.scatter_(1, idx, 1.0)

        scores = score_masks(encoder, ids_row, masks, chunk)

        # The selector's own mask, under the same hard top-k convention the
        # rest of the pipeline evaluates with.
        emb = encoder.token_embeddings(ids_row.unsqueeze(0), torch.ones(1, n, device=device))
        _, g, _ = model(ids_row.unsqueeze(0), emb, torch.ones(1, n, device=device), rhos=[rho])
        chosen = g[0, 0].unsqueeze(0)
        chosen_score = score_masks(encoder, ids_row, chosen, chunk)[0]

        selector_pct.append(percentile(scores, chosen_score))

        # Control: independently drawn masks, scored through the same path.
        control = [tuple(rng.sample(range(n), k)) for _ in range(control_draws)]
        cm = torch.zeros(len(control), n, device=device)
        cm.scatter_(1, torch.tensor(control, device=device, dtype=torch.long), 1.0)
        cs = score_masks(encoder, ids_row, cm, chunk)
        random_pct.append(float(np.mean([percentile(scores, v) for v in cs])))
        exhaustive_flags.append(exhaustive)
        sizes.append(len(combos))

    if not selector_pct:
        raise SystemExit("No sentence had more than one candidate mask at this rho.")

    sel = np.array(selector_pct)
    rnd = np.array(random_pct)
    n_exh = sum(exhaustive_flags)
    print(f"\n{len(sel)} sentences  |  rho={rho}  |  "
          f"{n_exh} exhaustive, {len(sel) - n_exh} sampled  |  "
          f"{np.mean(sizes):.0f} candidate masks/sentence on average")
    print(f"\n  selector beats {sel.mean() * 100:.2f}% of alternative masks (median "
          f"{np.median(sel) * 100:.2f}%)")
    se = rnd.std(ddof=1) / math.sqrt(len(rnd)) if len(rnd) > 1 else float("nan")
    verdict = "OK" if abs(rnd.mean() - 0.5) < max(3 * se, 0.01) else "OFF -- investigate before trusting the result above"
    print(f"  random control {rnd.mean() * 100:.2f}% (+/-{se * 100:.2f} SE, {control_draws} draws/sentence)  <- must be ~50%: {verdict}")
    for thr in (0.5, 0.9, 0.99, 1.0):
        label = "== best possible" if thr == 1.0 else f">= {thr:.0%}"
        frac = (sel >= thr).mean() if thr < 1.0 else (sel >= 1.0).mean()
        print(f"    sentences where selector is {label:16s} {frac * 100:5.1f}%")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.grid(True, color="#DDDDDD", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")
    bins = np.linspace(0, 1, 41)
    ax.hist(rnd, bins=bins, color="#999999", alpha=0.55, label="random mask (control)", zorder=2)
    ax.hist(sel, bins=bins, color="#0072B2", alpha=0.75, label="selector's mask", zorder=3)
    ax.axvline(sel.mean(), color="#0072B2", linestyle="--", linewidth=1.6, zorder=4)
    ax.axvline(0.5, color="#666666", linestyle=":", linewidth=1.2, zorder=4)
    ax.set_xlabel(f"fraction of alternative masks beaten (rho={rho:g})")
    ax.set_ylabel("sentences")
    ax.set_title(f"{family} x {pooling}: is the chosen mask actually a good one?\n"
                 f"selector mean {sel.mean():.1%} vs random {rnd.mean():.1%}, "
                 f"n={len(sel)} sentences", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"\nSaved plot to {output}")

    summary = {
        "signature": f"{run_path.parent.name}/{run_path.name}",
        "family": family, "pooling": pooling, "rho": rho,
        "sentences": len(sel),
        "exhaustive_sentences": int(n_exh),
        "mean_candidates_per_sentence": float(np.mean(sizes)),
        "selector_mean_percentile": float(sel.mean()),
        "selector_median_percentile": float(np.median(sel)),
        "random_control_mean_percentile": float(rnd.mean()),
        "random_control_se": float(se),
        "fraction_optimal": float((sel >= 1.0).mean()),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {output.with_suffix('.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sig", required=True, help="experiment or <xp>/<run> signature")
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--sentences", type=int, default=200)
    parser.add_argument("--max-combinations", type=int, default=20000,
                        help="enumerate exhaustively up to this many masks, sample above it")
    parser.add_argument("--samples", type=int, default=2000,
                        help="uniform samples per sentence when not exhaustive")
    parser.add_argument("--chunk", type=int, default=256, help="masks scored per forward")
    parser.add_argument("--control-draws", type=int, default=20,
                        help="random masks per sentence for the control (more = tighter null)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    out = args.output or (ROOT / "outputs/analysis" /
                          f"maskopt_{args.sig.replace('/', '_')}_r{args.rho:g}.pdf")
    main(args.sig, args.rho, args.sentences, args.max_combinations, args.samples,
         args.chunk, args.control_draws, args.device, args.seed, out)
