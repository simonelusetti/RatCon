from __future__ import annotations

import argparse, json, sys, torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import canonical_name, encode_examples, resolve_dataset, TEXT_FIELD, collate
from src.sentence import build_sentence_encoder
from src.utils import configure_runtime


# -------------------------
# configs
# -------------------------

def build_cfgs(args):
    data_cfg = OmegaConf.create({
        "dataset": args.dataset,
        "subset": 1.0,
        "max_length": args.max_length,
        "encoder": {
            "family": "sbert",
            "name": "all-MiniLM-L6-v2",
        },
        "config": json.loads(args.config_json) if args.config_json else None,
    })

    runtime_cfg = OmegaConf.create({
        "threads": args.threads,
        "interop_threads": args.interop_threads,
        "device": args.device,
        "token_parallelism": False,
        "data": {
            "rebuild": False,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
    })
    return data_cfg, runtime_cfg


def build_dataloader(args, data_cfg, runtime_cfg, tokenizer):
    name = canonical_name(data_cfg.dataset)
    text_field = TEXT_FIELD.get(name, "tokens")
    ds = resolve_dataset(name, text_field, config=data_cfg.get("config"))

    if "labels" not in ds["train"].column_names:
        raise ValueError("Dataset must provide token-level string labels.")

    ds = encode_examples(data_cfg, ds, tokenizer)
    split = ds[args.split]

    if args.index is not None:
        split = split.select([args.index])
    elif args.subset is not None:
        split = split.select(range(int(len(split) * args.subset)))
    elif args.max_samples is not None:
        split = split.select(range(args.max_samples))

    return DataLoader(
        split,
        batch_size=runtime_cfg.data.batch_size,
        num_workers=runtime_cfg.data.num_workers,
        collate_fn=collate,
        pin_memory=(args.device == "cuda"),
    )


# -------------------------
# token helpers
# -------------------------

def selectable_masks(ids, attn, tokenizer):
    special_ids = {
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
    }
    special_ids = {i for i in special_ids if i is not None}

    special = torch.zeros_like(attn, dtype=torch.bool)
    for sid in special_ids:
        special |= (ids == sid)

    valid = attn.bool()
    selectable = valid & ~special
    n_select = selectable.sum(1)
    return selectable, special, n_select


def sample_random_subset(selectable, special, k):
    """
    selectable: [B, L] bool
    special:    [B, L] bool
    k:          [B] long
    """
    B, L = selectable.shape
    scores = torch.rand(B, L, device=selectable.device)
    scores = scores.masked_fill(~selectable, -1e9)

    kmax = int(k.max().item())
    topk = scores.topk(kmax, dim=1).indices

    mask = torch.zeros(B, L, device=selectable.device, dtype=torch.bool)
    r = torch.arange(B, device=selectable.device)[:, None]
    mask[r, topk] = (torch.arange(kmax, device=selectable.device)[None, :] < k[:, None])

    mask |= special
    return mask


# -------------------------
# main
# -------------------------

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", default="conll2003")
    p.add_argument("--split", default="train")
    p.add_argument("--index", type=int)

    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=128)

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--interop-threads", type=int, default=8)

    p.add_argument("--subset", type=float)
    p.add_argument("--max-samples", type=int)

    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--frac", type=float, default=0.30)
    p.add_argument("--best-frac", type=float, default=0.05)

    p.add_argument("--config-json")

    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)

    encoder, tokenizer = build_sentence_encoder(
        "sbert", "all-MiniLM-L6-v2", device.type
    )

    data_cfg, runtime_cfg = build_cfgs(args)
    loader = build_dataloader(args, data_cfg, runtime_cfg, tokenizer)

    configure_runtime(runtime_cfg)

    selected_counts = defaultdict(float)
    total_counts = defaultdict(float)

    for batch in tqdm(loader, desc="Sampling subsets"):
        ids = batch["ids"].to(device)
        attn = batch["attn_mask"].to(device)
        labels = batch["labels"]  # list[list[str]]

        with torch.no_grad():
            full = encoder.encode(ids, attn)

        selectable, special, n_select = selectable_masks(ids, attn, tokenizer)
        k = torch.clamp((n_select.float() * args.frac).round().long(), min=1)

        B, L = ids.shape
        T = args.trials

        sims = torch.empty(T, B, device=device)
        masks = torch.empty(T, B, L, device=device, dtype=torch.bool)

        for t in range(T):
            mask = sample_random_subset(selectable, special, k)
            masks[t] = mask
            with torch.no_grad():
                rep = encoder.encode(ids, mask)
            sims[t] = F.cosine_similarity(rep, full, dim=-1)

        M = max(1, int(round(args.best_frac * T)))
        best_idx = sims.topk(M, dim=0).indices  # [M, B]

        for b in range(B):
            sel_pos = selectable[b]
            if sel_pos.sum() == 0:
                continue

            idx = torch.where(sel_pos)[0].tolist()
            token_labels = [labels[b][i] for i in idx]

            best_masks = masks[best_idx[:, b], b][:, sel_pos]
            selected_per_token = best_masks.sum(0).cpu().tolist()

            for lab, count in zip(token_labels, selected_per_token):
                selected_counts[lab] += count
                total_counts[lab] += M

    print("\n==== P(token selected | label) on best subsets ====")
    for lab in sorted(total_counts):
        p_sel = selected_counts[lab] / total_counts[lab]
        print(f"{lab:8s} : {p_sel:.4f}")


if __name__ == "__main__":
    main()
