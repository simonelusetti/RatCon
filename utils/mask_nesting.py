"""
mask_nesting.py  —  measure how nested the selector's hard masks are across rho values.

For each pair of rho values (rho_low < rho_high), computes per-sentence:

    overlap(low, high) = |g_low ∩ g_high| / |g_low|

= fraction of tokens selected at the lower rate that are *also* selected at the
higher rate.  A value of 1.0 means perfect nesting (every smaller mask is a
strict subset of the larger one).

Usage:
    python utils/mask_nesting.py --sig 024ef2a7
    python utils/mask_nesting.py --sig 024ef2a7 --split train --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from numpy import linspace
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import initialize_data, collate
from src.selector import RationaleSelectorModel
from utils.dora_utils import XPS_DIR, expected_checkpoint, load_overrides_for_sig


# ---------------------------------------------------------------------------
# Config reconstruction from a signature
# ---------------------------------------------------------------------------

def load_xp_cfg(sig_dir: Path) -> OmegaConf:
    """Merge default.yaml with the overrides stored in the xp's .argv.json."""
    default_path = PROJECT_ROOT / "src" / "conf" / "default.yaml"
    base = OmegaConf.load(default_path)

    overrides = load_overrides_for_sig(sig_dir) or []
    # Filter out runtime.* and train.* keys that don't affect model/data structure
    skip_prefixes = ("runtime.", "train.", "run=")
    filtered = [o for o in overrides if not any(o.startswith(p) for p in skip_prefixes)]

    patch = OmegaConf.from_dotlist(filtered)
    return OmegaConf.merge(base, patch)


# ---------------------------------------------------------------------------
# Welford online mean/variance accumulator (per-pair)
# ---------------------------------------------------------------------------

class OnlineMeanVar:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update_batch(self, values: np.ndarray) -> None:
        """Accept a 1-D array of scalar observations."""
        for v in values:
            self.n += 1
            delta = v - self.mean
            self.mean += delta / self.n
            delta2 = v - self.mean
            self.M2 += delta * delta2

    @property
    def std(self) -> float:
        return float(np.sqrt(self.M2 / self.n)) if self.n > 1 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Measure mask nesting across rho values.")
    p.add_argument("--sig", required=True, help="Experiment signature (e.g. 024ef2a7)")
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None, help="Limit dataset to N examples")
    p.add_argument("--sweep-range", nargs=3, type=float, metavar=("START", "END", "STEPS"),
                   help="Override rho sweep (default: from model config)")
    args = p.parse_args()

    sig_dir = XPS_DIR / args.sig
    if not sig_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {sig_dir}")

    # ---- config ----
    cfg = load_xp_cfg(sig_dir)
    device = torch.device(args.device)

    sweep = cfg.model.loss.sweep_range
    rhos = list(linspace(sweep[0], sweep[1], int(sweep[2])))
    if args.sweep_range:
        rhos = list(linspace(args.sweep_range[0], args.sweep_range[1], int(args.sweep_range[2])))

    print(f"Signature : {args.sig}")
    print(f"Dataset   : {cfg.data.dataset}  split={args.split}")
    print(f"Encoder   : {cfg.data.encoder.family}")
    print(f"Rhos ({len(rhos)}): {[round(r, 3) for r in rhos]}")

    runtime_data_cfg = OmegaConf.create({
        "rebuild": False,
        "test_subset": None,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    })
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
    data_cfg.subset = 1.0
    _, _, encoder, _, _, ds = initialize_data(
        data_cfg,
        runtime_data_cfg,
        device=args.device,
        keep_special=bool(cfg.model.get("keep_special", True)),
    )
    encoder.eval()

    available_splits = list(ds.keys())
    split = args.split
    if split not in available_splits:
        fallback = next((s for s in ("test", "train") if s in available_splits), available_splits[0])
        print(f"Warning: split '{split}' not found, falling back to '{fallback}' (available: {available_splits})")
        split = fallback
    split_ds = ds[split]
    if args.max_samples is not None:
        split_ds = split_ds.select(range(min(args.max_samples, len(split_ds))))

    loader = DataLoader(
        split_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate,
        shuffle=False,
    )

    # ---- model ----
    ckpt_path = expected_checkpoint(sig_dir)
    if ckpt_path is None:
        # Fallback: look for any model checkpoint
        candidates = sorted((sig_dir / "state" / "models").glob("model_*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found in {sig_dir}")
        ckpt_path = candidates[-1]
    print(f"Checkpoint: {ckpt_path.relative_to(PROJECT_ROOT)}")

    # Infer embedding dim from first batch
    first_batch = next(iter(loader))
    ex_ids = first_batch["ids"][:1].to(device)
    ex_attn = first_batch["attn_mask"][:1].to(device)
    with torch.no_grad():
        ex_emb = encoder.token_embeddings(ex_ids, ex_attn)
    emb_dim = ex_emb.shape[-1]

    selector_cfg = OmegaConf.to_container(cfg.model.get("selector", OmegaConf.create({})))
    loss_cfg = OmegaConf.to_container(cfg.model.loss)
    model = RationaleSelectorModel(
        embedding_dim=emb_dim,
        sent_encoder=encoder,
        loss_cfg=loss_cfg,
        selector_cfg=selector_cfg,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()

    # ---- accumulate per-pair overlap ----
    R = len(rhos)
    # stats[(i, j)] where i < j (rho[i] < rho[j], mask[i] selects fewer tokens)
    stats: dict[tuple[int, int], OnlineMeanVar] = {
        (i, j): OnlineMeanVar()
        for i in range(R)
        for j in range(i + 1, R)
    }

    total_examples = 0
    for batch in tqdm(loader, desc="Batches"):
        ids = batch["ids"].to(device)
        attn = batch["attn_mask"].to(device)

        with torch.no_grad():
            token_emb = encoder.token_embeddings(ids, attn)
            _, g, _ = model(ids, token_emb, attn, rhos=rhos)
        # g: [R, B, L]  —  hard binary masks

        g_np = g.cpu().numpy()  # [R, B, L]
        k_per_rho = g_np.sum(axis=2)  # [R, B]  number selected per (rho, example)

        for i in range(R):
            for j in range(i + 1, R):
                intersection = (g_np[i] * g_np[j]).sum(axis=1)  # [B]
                k_i = k_per_rho[i]                               # [B]

                # Only count examples where mask_i selects at least one token
                valid = k_i > 0
                if not valid.any():
                    continue

                overlap = intersection[valid] / k_i[valid]
                stats[(i, j)].update_batch(overlap)

        total_examples += ids.shape[0]

    # ---- report ----
    print(f"\n{'='*60}")
    print(f"Mask nesting overlap  (|g_low ∩ g_high| / |g_low|)")
    print(f"Examples: {total_examples}  |  1.0 = perfect nesting")
    print(f"{'='*60}")

    # Consecutive pairs (most interpretable)
    print("\nConsecutive rho pairs:")
    print(f"  {'rho_low':>8}  {'rho_high':>9}  {'mean':>7}  {'std':>7}  {'n':>7}")
    print(f"  {'-'*46}")
    consecutive_means = []
    for i in range(R - 1):
        acc = stats[(i, i + 1)]
        consecutive_means.append(acc.mean)
        print(f"  {rhos[i]:>8.3f}  {rhos[i+1]:>9.3f}  {acc.mean:>7.4f}  {acc.std:>7.4f}  {acc.n:>7d}")

    # Global summary
    all_means = [stats[(i, j)].mean for i in range(R) for j in range(i + 1, R) if stats[(i, j)].n > 0]
    all_stds  = [stats[(i, j)].std  for i in range(R) for j in range(i + 1, R) if stats[(i, j)].n > 0]

    print(f"\nAll pairs summary:")
    print(f"  mean overlap = {np.mean(all_means):.4f}  (std of means = {np.std(all_means):.4f})")
    print(f"  consecutive  = {np.mean(consecutive_means):.4f}")

    if len(rhos) <= 10:
        # Print full lower-triangular matrix of means
        print(f"\nFull pair matrix (mean overlap, row=rho_low, col=rho_high):")
        header = "         " + "".join(f"  {r:.2f}" for r in rhos[1:])
        print(f"  {header}")
        for i in range(R - 1):
            row = f"  {rhos[i]:.2f}  |"
            for j in range(R):
                if j <= i:
                    row += "      "
                else:
                    row += f"  {stats[(i,j)].mean:.3f}"
            print(row)


if __name__ == "__main__":
    main()
