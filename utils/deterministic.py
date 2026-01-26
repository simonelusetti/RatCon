from __future__ import annotations

import argparse, json, sys, torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

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
            "name": "all-MiniLM-L6-v2",  # or whatever your default is
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

    return DataLoader(
        split,
        batch_size=runtime_cfg.data.batch_size,
        num_workers=runtime_cfg.data.num_workers,
        collate_fn=collate,
        pin_memory=(args.device == "cuda"),
    )


# -------------------------
# partition logic
# -------------------------

def non_special_masks(ids, attn, tokenizer):
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
    return selectable, special


def random_partition(selectable):
    rand = torch.rand_like(selectable.float())
    a = selectable & (rand < 0.5)
    b = selectable & ~a

    # ensure both non-empty
    a_empty = a.sum(1) == 0
    b_empty = b.sum(1) == 0

    for i in torch.where(a_empty)[0]:
        j = torch.where(b[i])[0][0]
        a[i, j] = True
        b[i, j] = False

    for i in torch.where(b_empty)[0]:
        j = torch.where(a[i])[0][0]
        b[i, j] = True
        a[i, j] = False

    return a, b


# -------------------------
# main
# -------------------------

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", default="mr")
    p.add_argument("--split", default="train")
    p.add_argument("--index", type=int)

    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=128)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--threads", type=int, default=48)
    p.add_argument("--interop-threads", type=int, default=1)

    p.add_argument("--subset", type=float)
    p.add_argument("--max-samples", type=int)

    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--normalize-sum", action="store_true")

    p.add_argument("--config-json")

    args = p.parse_args()

    torch.manual_seed(0)

    device = torch.device(args.device)

    # ---- SBERT only ----
    encoder, tokenizer = build_sentence_encoder(
        "sbert", "all-MiniLM-L6-v2", device.type
    )

    data_cfg, runtime_cfg = build_cfgs(args)
    loader = build_dataloader(args, data_cfg, runtime_cfg, tokenizer)
    configure_runtime(runtime_cfg)

    all_means = []
    all_stds = []

    for batch in tqdm(loader, desc="SBERT"):
        ids = batch["ids"].to(device)
        attn = batch["attn_mask"].to(device)

        with torch.no_grad():
            full = encoder.encode(ids, attn)

        selectable, special = non_special_masks(ids, attn, tokenizer)

        B = ids.size(0)
        sims = torch.zeros(args.trials, B, device=device)

        for t in range(args.trials):
            mask_a, mask_b = random_partition(selectable)
            mask_a |= special
            mask_b |= special

            with torch.no_grad():
                a = encoder.encode(ids, mask_a)
                b = encoder.encode(ids, mask_b)

            summed = a + b
            if args.normalize_sum:
                summed = 0.5 * summed

            sims[t] = F.cosine_similarity(summed, full, dim=-1)

        # per-sentence stats
        mean = sims.mean(0)
        std = sims.std(0)

        all_means.append(mean.cpu())
        all_stds.append(std.cpu())

    all_means = torch.cat(all_means)
    all_stds = torch.cat(all_stds)

    print("==== Additivity distribution (SBERT) ====")
    print(f"Sentences: {all_means.numel()}")
    print(f"Mean cosine: {all_means.mean():.4f} ± {all_means.std():.4f}")
    print(f"Mean intra-sentence std: {all_stds.mean():.4f}")
    print(f"Median intra-sentence std: {all_stds.median():.4f}")
    print(f"Max intra-sentence std: {all_stds.max():.4f}")


if __name__ == "__main__":
    main()
