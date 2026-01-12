from __future__ import annotations

import argparse, json, math, random
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader

from luse.data import canonical_name, encode_examples, resolve_dataset, TEXT_FIELD, collate
from luse.selector import RationaleSelectorModel
from luse.sentence import build_sentence_encoder, DEFAULT_MODEL_NAMES
from luse.utils import configure_runtime

FRACS = [0.1, 0.3, 0.5]
ENCODERS = ["sbert", "e5", "bge", "llm"]


def build_data_cfg(args: argparse.Namespace, encoder_family: str | None = None) -> Any:
    encoder_family = encoder_family or ENCODERS[0]
    cfg: Dict[str, Any] = {
        "dataset": args.dataset,
        "subset": 1.0,
        "max_length": args.max_length,
        "encoder": {
            "family": encoder_family,
            "name": DEFAULT_MODEL_NAMES.get(encoder_family, None),
        },
        "runtime": {
            "rebuild": False,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
        "config": None,
    }
    if args.config_json:
        cfg["config"] = json.loads(args.config_json)
    return OmegaConf.create(cfg)


def get_special_tokens(tokenizer) -> set[str]:
    return {
        tokenizer.cls_token,
        tokenizer.sep_token,
        tokenizer.pad_token,
        tokenizer.bos_token,
        tokenizer.eos_token,
    }


def precompute_selectable_masks(ids_batch: torch.Tensor, attn_batch: torch.Tensor, tokenizer):
    special_ids = [
        idx for idx in [
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        ]
        if idx is not None
    ]
    if special_ids:
        special_ids_t = torch.tensor(special_ids, device=ids_batch.device)
        is_special = (ids_batch.unsqueeze(-1) == special_ids_t).any(dim=-1)
    else:
        is_special = torch.zeros_like(attn_batch, dtype=torch.bool)
    valid = attn_batch == 1
    selectable_mask = valid & ~is_special
    special_mask = valid & is_special
    selectable_counts = selectable_mask.sum(dim=1)
    return selectable_mask, special_mask, selectable_counts

def print_debug_sample(
    size,
    debug_printed,
    max_samples,
    tokens_batch,
    mask_batch,
    attn,
    selectable_mask,
    special_mask,
    k_per_row,
    tokenizer,
    frac,
) -> None:
    for i in range(size):
        if debug_printed >= max_samples:
            break
        subset_tokens = [
            tok for tok, keep in zip(tokens_batch[i], mask_batch[i].tolist()) if keep
        ]
        attn_row = attn[i].tolist()
        filtered_tokens = [
            tok for tok, keep in zip(tokens_batch[i], attn_row) if keep == 1
        ]
        full_text = tokenizer.convert_tokens_to_string(filtered_tokens)
        subset_text = tokenizer.convert_tokens_to_string(subset_tokens)
        selectable_count = int(selectable_mask[i].sum().item())
        special_count = int(special_mask[i].sum().item())
        subset_count = int(mask_batch[i].sum().item())
        k_target = int(k_per_row[i].item())
        print(
            f"[debug {debug_printed}] counts selectable={selectable_count} "
            f"special={special_count} k={k_target} subset={subset_count} "
            f"frac={frac}"
        )
        print(f"[debug {debug_printed}] text={full_text}")
        print(f"[debug {debug_printed}] subset={subset_text}")
        debug_printed += 1
    return debug_printed

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate how random token subsets compare to the full sentence embedding.",
    )
    parser.add_argument("--dataset", default="mr")
    parser.add_argument("--split", default="train")
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--best-percentile", type=float, default=0.95)
    parser.add_argument("--debug-samples", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--keep-special", action="store_true", default=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--subset", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=48)
    parser.add_argument("--config-json", default=None, help="Dataset config JSON (for custom datasets).")
    parser.add_argument("--selector-ckpt", default=None, help="Optional selector checkpoint to compare.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    configure_runtime(threads=args.num_threads)

    device = torch.device(args.device)
    for encoder_family in ENCODERS:
        encoder, tokenizer = build_sentence_encoder(
            family=encoder_family,
            encoder_name=DEFAULT_MODEL_NAMES.get(encoder_family, None),
            device=device.type,
        )

        data_cfg = build_data_cfg(args, encoder_family=encoder_family)
        name = canonical_name(data_cfg.dataset)
        text_field = TEXT_FIELD.get(name, "tokens")
        raw_ds = resolve_dataset(name, text_field, config=data_cfg.get("config", None))
        if "labels" in raw_ds["train"].column_names:
            raw_ds = raw_ds.remove_columns(["labels"])
        if "scnd_labels" in raw_ds["train"].column_names:
            raw_ds = raw_ds.remove_columns(["scnd_labels"])
        ds = encode_examples(data_cfg, raw_ds, tokenizer, scnd_labels_map=None)
        split_ds = ds[args.split]
        if args.index is not None:
            split_ds = split_ds.select([args.index])
        if args.subset is not None:
            split_ds = split_ds.select(range(int(len(split_ds) * args.subset)))
        elif args.max_samples is not None:
            split_ds = split_ds.select(range(args.max_samples))
        dataloader = DataLoader(
            split_ds,
            batch_size=data_cfg.runtime.batch_size,
            num_workers=data_cfg.runtime.num_workers,
            collate_fn=collate,
            shuffle=False,
            pin_memory=(device == "cuda"),
        )

        if args.selector_ckpt:
            with torch.no_grad():
                example = split_ds[0]
                ids = torch.tensor(example["ids"], dtype=torch.long, device=device).unsqueeze(0)
                attn = torch.tensor(example["attn_mask"], dtype=torch.long, device=device).unsqueeze(0)
                tkns_embd = encoder.token_embeddings(ids, attn)
            model_dim = tkns_embd.shape[-1]
            selector = RationaleSelectorModel(model_dim).to(device)
            state = torch.load(args.selector_ckpt, map_location=device)
            selector.load_state_dict(state["model"], strict=False)
            selector.eval()
        else:
            selector = None

        for frac in FRACS:
            count = 0
            token_count_sum = 0
            sums = {
                "mean": 0.0,
                "p_best": 0.0,
                "best_ratio": 0.0,
            }
            degenerate_count = 0
            clamped_count = 0
            trial_count = 0
            selector_cos_sum = 0.0
            selector_rate_sum = 0.0
            selector_count = 0
            debug_samples = args.debug_samples if (encoder_family == ENCODERS[0] and frac == FRACS[0]) else 0
            debug_printed = 0

            for batch in tqdm(dataloader, desc=f"{encoder_family} frac={frac}"):
                ids = batch["ids"].to(device)
                attn = batch["attn_mask"].to(device)
                tokens_batch = batch["tokens"]

                with torch.no_grad():
                    full_rep = encoder.encode(ids, attn)

                batch_size = ids.size(0)
                selectable_mask, special_mask, selectable_counts = precompute_selectable_masks(
                    ids,
                    attn,
                    tokenizer,
                )
                cos_trials = []

                for trial_idx in range(args.trials):
                    k_per_row = torch.round(selectable_counts.float() * frac).to(torch.long)
                    clamped_mask = k_per_row < 1
                    if clamped_mask.any():
                        clamped_count += int(clamped_mask.sum().item())
                        k_per_row = torch.clamp(k_per_row, min=1)
                    trial_count += batch_size

                    max_k = int(k_per_row.max().item())
                    scores = torch.rand_like(attn, dtype=torch.float, device=device)
                    scores = scores.masked_fill(~selectable_mask, float("-inf"))
                    _, topk_idx = torch.topk(scores, k=max_k, dim=1)

                    mask_batch = torch.zeros_like(attn)
                    keep_mask = torch.arange(max_k, device=device).unsqueeze(0) < k_per_row.unsqueeze(1)
                    row_ids = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, max_k)
                    mask_batch[row_ids[keep_mask], topk_idx[keep_mask]] = 1

                    if args.keep_special:
                        mask_batch = mask_batch | special_mask
                    if debug_samples and trial_idx == 0 and debug_printed < debug_samples:
                        debug_printed += print_debug_sample(
                            debug_printed=debug_printed,
                            size=batch_size,
                            max_samples=debug_samples - debug_printed,
                            tokens_batch=tokens_batch,
                            mask_batch=mask_batch,
                            attn=attn,
                            selectable_mask=selectable_mask,
                            special_mask=special_mask,
                            k_per_row=k_per_row,
                            tokenizer=tokenizer,
                            frac=frac,
                        )

                    with torch.no_grad():
                        pred_rep = encoder.encode(ids, mask_batch)
                        cos = F.cosine_similarity(pred_rep, full_rep, dim=-1)
                    cos_trials.append(cos)

                sims_tensor = torch.stack(cos_trials, dim=0)
                sims_sorted, _ = sims_tensor.sort(dim=0)
                best_idx = int(args.best_percentile * (sims_tensor.size(0) - 1))
                p_best = sims_sorted[best_idx]
                mean = sims_tensor.mean(dim=0)
                best_ratio = torch.where(mean != 0, p_best / mean, torch.full_like(mean, float("inf")))
                degenerate_mask = sims_tensor.std(dim=0) == 0

                sums["mean"] += float(mean.sum().item())
                sums["p_best"] += float(p_best.sum().item())
                sums["best_ratio"] += float(best_ratio.sum().item())
                degenerate_count += int(degenerate_mask.sum().item())
                token_count_sum += int(attn.sum().item())
                count += batch_size

                if selector:
                    with torch.no_grad():
                        tkns_embd = encoder.token_embeddings(ids, attn)
                        gates, _ = selector(tkns_embd, attn, deterministic=True)
                        sel_rep = encoder.encode(ids, attn * gates)
                        sel_cos = F.cosine_similarity(sel_rep, full_rep, dim=-1)
                        sel_rate = gates.sum(dim=1) / attn.sum(dim=1)
                    selector_cos_sum += float(sel_cos.sum().item())
                    selector_rate_sum += float(sel_rate.sum().item())
                    selector_count += int(sel_cos.numel())

            print(f"Dataset: {args.dataset} split={args.split} samples={count} encoder={encoder_family} frac={frac}")
            print(f"Avg tokens: {token_count_sum / count:.2f}  trials={args.trials}")
            print(
                "Best/mean @ p{:.0f}: p_best={:.4f} mean={:.4f} ratio={:.4f}".format(
                    100.0 * args.best_percentile,
                    sums["p_best"] / count,
                    sums["mean"] / count,
                    sums["best_ratio"] / count,
                )
            )
            print(
                "Degenerate stats: std=0 count={} ({:.2f}%)".format(
                    degenerate_count,
                    100.0 * degenerate_count / count,
                )
            )
            if trial_count > 0:
                print(
                    "Subset clamp: k<1 count={} ({:.2f}%)".format(
                        clamped_count,
                        100.0 * clamped_count / trial_count,
                    )
                )

            if selector_count > 0:
                print(
                    "Selector avg: cosine={:.4f} selection_rate={:.4f}".format(
                        selector_cos_sum / selector_count,
                        selector_rate_sum / selector_count,
                    )
                )


if __name__ == "__main__":
    main()
