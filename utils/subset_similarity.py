import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from numpy import linspace
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    canonical_name,
    encode_examples,
    resolve_dataset,
    TEXT_FIELD,
    collate,
)
from src.selector import RationaleSelectorModel
from src.sentence import build_sentence_encoder, DEFAULT_MODEL_NAMES


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

FRACS = (0.1, 0.3, 0.5)             # must match selector sweep_range linspace
ENCODERS = ("sbert", "e5", "bge", "llm")


# ---------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------

def configure_runtime_simple(num_threads: int, interop_threads: int | None = None) -> None:
    torch.set_num_threads(int(num_threads))
    if interop_threads is not None:
        torch.set_num_interop_threads(int(interop_threads))


def build_data_cfg(args: argparse.Namespace, family: str) -> tuple[DictConfig, DictConfig]:
    """
    Returns (data_cfg, runtime_cfg) compatible with src.data.encode_examples / DataLoader.
    """
    data_cfg = OmegaConf.create({
        "dataset": args.dataset,
        "subset": 1.0,
        "max_length": args.max_length,
        "encoder": {"family": family, "name": DEFAULT_MODEL_NAMES.get(family)},
        "config": json.loads(args.config_json) if args.config_json else None,
    })
    runtime_cfg = OmegaConf.create({
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": args.device,
    })
    return data_cfg, runtime_cfg


# ---------------------------------------------------------------------
# Masking logic (selectable vs special vs padding)
# ---------------------------------------------------------------------

def selectable_masks(
    ids: torch.Tensor,
    attn: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    selectable: valid & not special
    special:    valid & special
    n_select:   number of selectable tokens per example
    """
    special_ids = [
        i for i in (
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        ) if i is not None
    ]
    if special_ids:
        special_tensor = torch.tensor(special_ids, device=ids.device)
        is_special = (ids[..., None] == special_tensor).any(-1)
    else:
        is_special = torch.zeros_like(attn, dtype=torch.bool)

    valid = attn.bool()
    selectable = valid & ~is_special
    special = valid & is_special
    n_select = selectable.sum(1)
    return selectable, special, n_select


# ---------------------------------------------------------------------
# Dataset / loader
# ---------------------------------------------------------------------

def build_dataloader(
    args: argparse.Namespace,
    data_cfg: DictConfig,
    runtime_cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[DataLoader, Dataset]:
    name = canonical_name(data_cfg.dataset)
    text_field = TEXT_FIELD.get(name, "tokens")
    ds = resolve_dataset(name, text_field=text_field, config=data_cfg.get("config"))

    # drop labels if present (we only need ids/attn/tokens here)
    drop_cols = {"labels", "scnd_labels"} & set(ds["train"].column_names)
    if drop_cols:
        ds = ds.remove_columns(list(drop_cols))

    ds = encode_examples(data_cfg, ds, tokenizer)
    split = ds[args.split]

    if args.index is not None:
        split = split.select([args.index])
    elif args.subset is not None:
        split = split.select(range(int(len(split) * args.subset)))
    elif args.max_samples is not None:
        split = split.select(range(args.max_samples))

    loader = DataLoader(
        split,
        batch_size=int(runtime_cfg.batch_size),
        num_workers=int(runtime_cfg.num_workers),
        collate_fn=collate,
        pin_memory=(runtime_cfg.device == "cuda"),
        shuffle=False,
    )
    return loader, split


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mr")
    p.add_argument("--split", default="train")
    p.add_argument("--index", type=int)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--best-percentile", type=float, default=0.95)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--keep-special", action="store_true", default=True)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--subset", type=float)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--num-threads", type=int, default=48)
    p.add_argument("--interop-threads", type=int)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--config-json")
    p.add_argument("--selector-ckpt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    configure_runtime_simple(args.num_threads, args.interop_threads)
    device = torch.device(args.device)

    # We align selector sweep_range with FRACS exactly:
    # linspace(start, end, steps) = FRACS if FRACS are evenly spaced.
    # For (0.1, 0.3, 0.5) it's perfect.
    sweep_start, sweep_end, sweep_steps = min(FRACS), max(FRACS), len(FRACS)
    selector_loss_cfg = OmegaConf.create({"sweep_range": [sweep_start, sweep_end, sweep_steps]})

    for family in ENCODERS:
        encoder, tokenizer = build_sentence_encoder(
            family, DEFAULT_MODEL_NAMES.get(family), device.type
        )

        data_cfg, runtime_cfg = build_data_cfg(args, family)
        loader, split_ds = build_dataloader(args, data_cfg, runtime_cfg, tokenizer)

        selector = None
        if args.selector_ckpt:
            # infer embedding dim from one example
            with torch.no_grad():
                ex = split_ds[0]
                ex_ids = torch.tensor(ex["ids"], device=device)[None]
                ex_attn = torch.tensor(ex["attn_mask"], device=device)[None]
                ex_token_emb = encoder.token_embeddings(ex_ids, ex_attn)

            selector = RationaleSelectorModel(
                embedding_dim=ex_token_emb.size(-1),
                sent_encoder=encoder,
                loss_cfg=selector_loss_cfg,
            ).to(device)

            ckpt = torch.load(args.selector_ckpt, map_location=device)
            selector.load_state_dict(ckpt.get("model", ckpt), strict=False)
            selector.eval()

        # Running stats per frac
        sums = {frac: dict(mean=0.0, p_best=0.0, best_ratio=0.0) for frac in FRACS}
        count = 0
        token_sum = 0
        degenerate = {frac: 0 for frac in FRACS}
        clamp = {frac: 0 for frac in FRACS}
        trials_total = {frac: 0 for frac in FRACS}

        # selector stats per frac
        sel_cos = {frac: 0.0 for frac in FRACS}
        sel_rate = {frac: 0.0 for frac in FRACS}
        sel_n = 0

        for batch in tqdm(loader, desc=f"{family}"):
            ids = batch["ids"].to(device)
            attn = batch["attn_mask"].to(device)

            # heavy forward ONCE
            with torch.no_grad():
                token_emb_full = encoder.token_embeddings(ids, attn)
                full = encoder.pool(token_emb_full, attn)

            selectable, special, n_select = selectable_masks(ids, attn, tokenizer)
            T_eff = attn.sum(1)  # per example

            # ----- selector outputs (hard masks per rho) -----
            # Note: your selector returns hard masks in g_sweep (cpu).
            # We move them back to device for pooling.
            if selector is not None:
                with torch.no_grad():
                    _, g_sweep, *_ = selector(ids, token_emb_full, attn)
                # g_sweep is list[Tensor[B,T]] on CPU
                g_sweep = [g.to(device) for g in g_sweep]

            # ----- random baseline per frac -----
            for frac in FRACS:
                sims = []
                for _ in range(args.trials):
                    k = torch.clamp((n_select.float() * frac).round().long(), min=1)
                    clamp[frac] += (k == 1).sum().item()
                    trials_total[frac] += ids.size(0)

                    # random scores over selectable positions
                    scores = torch.rand_like(attn.float()).masked_fill(~selectable, -1e9)
                    max_k = int(k.max().item())

                    topk = torch.topk(scores, k=max_k, dim=1).indices  # [B, max_k]
                    mask = torch.zeros_like(attn, dtype=torch.long)     # pool_mask (0/1)

                    r = torch.arange(ids.size(0), device=device)[:, None]
                    # keep only first k[b] indices in each row
                    mask[r, topk] = (torch.arange(max_k, device=device)[None, :] < k[:, None]).long()

                    if args.keep_special:
                        mask = mask | special.long()

                    # IMPORTANT: keep contextual token embeddings from full sentence, only change pooling mask
                    rep = encoder.pool(token_emb_full, mask)
                    sims.append(F.cosine_similarity(rep, full, dim=-1))

                sims = torch.stack(sims)  # [trials, B]
                sims_sorted = sims.sort(0).values
                p_idx = int(args.best_percentile * (sims.size(0) - 1))
                p_best = sims_sorted[p_idx]
                mean = sims.mean(0)
                ratio = torch.where(mean != 0, p_best / mean, torch.inf)

                sums[frac]["mean"] += mean.sum().item()
                sums[frac]["p_best"] += p_best.sum().item()
                sums[frac]["best_ratio"] += ratio.sum().item()
                degenerate[frac] += (sims.std(0) == 0).sum().item()

            # ----- selector metrics per frac (hard top-k masks) -----
            if selector is not None:
                # IMPORTANT: g_sweep order matches linspace(sweep_start, sweep_end, sweep_steps)
                # which we set to align with FRACS.
                for i, frac in enumerate(FRACS):
                    hard_mask = g_sweep[i].long()
                    if args.keep_special:
                        hard_mask = hard_mask | special.long()

                    rep_sel = encoder.pool(token_emb_full, hard_mask)
                    sel_cos[frac] += F.cosine_similarity(rep_sel, full, dim=-1).sum().item()

                    # selection rate measured on pool_mask vs full valid tokens
                    sel_rate[frac] += (hard_mask.sum(1).float() / T_eff.clamp(min=1).float()).sum().item()

                sel_n += ids.size(0)

            token_sum += attn.sum().item()
            count += ids.size(0)

        # ----- print results -----
        avg_tokens = token_sum / max(count, 1)
        print(f"\n{args.dataset} {args.split} | {family}")
        print(f"Avg tokens: {avg_tokens:.2f}  trials={args.trials}")

        for frac in FRACS:
            print(f"\nfrac={frac}")
            print(f"Best/mean@p{int(100*args.best_percentile)} "
                  f"{sums[frac]['p_best']/count:.4f} / {sums[frac]['mean']/count:.4f} "
                  f"ratio={sums[frac]['best_ratio']/count:.4f}")
            print(f"Degenerate: {100*degenerate[frac]/count:.2f}%")
            print(f"Clamp k<1: {100*clamp[frac]/max(1,trials_total[frac]):.2f}%")

            if sel_n:
                print(f"Selector: cos={sel_cos[frac]/sel_n:.4f} rate={sel_rate[frac]/sel_n:.4f}")

        # Note: for FrozenBGE your pool() ignores pool_mask (CLS pooling),
        # so selection has essentially no effect and cosine will be ~1.0 for all masks.

    print("\nDone.")


if __name__ == "__main__":
    main()
