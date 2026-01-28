import argparse, json, torch, sys, torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
from transformers import PreTrainedTokenizerBase
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import canonical_name, encode_examples, resolve_dataset, TEXT_FIELD, collate
from src.selector import RationaleSelectorModel
from src.sentence import build_sentence_encoder, DEFAULT_MODEL_NAMES
from src.utils import configure_runtime


FRACS = (0.1, 0.3, 0.5)
ENCODERS = ("sbert", "e5", "bge", "llm")


def build_data_cfg(args: argparse.Namespace, family: str) -> tuple[DictConfig, DictConfig]:
    return OmegaConf.create({
        "dataset": args.dataset,
        "subset": 1.0,
        "max_length": args.max_length,
        "encoder": {"family": family, "name": DEFAULT_MODEL_NAMES.get(family)},
        "config": json.loads(args.config_json) if args.config_json else None,
    }), OmegaConf.create({
        "threads": args.threads,
        "interop_threads": args.interop_threads,
        "device": args.device,
        "token_parallelism": False,
        "data": {
            "rebuild": False,
            "batch_size": 256,
            "num_workers": args.num_workers,
        },
        "eval": {"short_log": False, "log_examples": 0}
    })


def selectable_masks(
    ids: torch.Tensor,
    attn: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    special_ids = [
        i for i in (
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        ) if i is not None
    ]
    is_special = (
        (ids[..., None] == torch.tensor(special_ids, device=ids.device)).any(-1)
        if special_ids else torch.zeros_like(attn, dtype=torch.bool)
    )
    valid = attn.bool()
    selectable = valid & ~is_special
    return selectable, valid & is_special, selectable.sum(1)


def build_dataloader(
    args: argparse.Namespace,
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[DataLoader, Dataset]:
    name = canonical_name(cfg.dataset)
    text_field = TEXT_FIELD.get(name, "tokens")
    ds = resolve_dataset(name, text_field, config=cfg.get("config"))

    drop_cols = {"labels", "scnd_labels"} & set(ds["train"].column_names)
    if drop_cols:
        ds = ds.remove_columns(list(drop_cols))

    ds = encode_examples(cfg, ds, tokenizer)
    split = ds[args.split]

    if args.index is not None:
        split = split.select([args.index])
    elif args.subset:
        split = split.select(range(int(len(split) * args.subset)))
    elif args.max_samples:
        split = split.select(range(args.max_samples))

    return DataLoader(
        split,
        batch_size=cfg.runtime.batch_size,
        num_workers=cfg.runtime.num_workers,
        collate_fn=collate,
        pin_memory=(args.device == "cuda"),
    ), split

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mr")
    p.add_argument("--split", default="train")
    p.add_argument("--index", type=int)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--best-percentile", type=float, default=0.95)
    p.add_argument("--debug-samples", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--keep-special", action="store_true", default=True)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--subset", type=float)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--num-threads", type=int, default=48)
    p.add_argument("--config-json")
    p.add_argument("--selector-ckpt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    configure_runtime(args.num_threads)
    device = torch.device(args.device)

    for family in ENCODERS:
        encoder, tokenizer = build_sentence_encoder(
            family, DEFAULT_MODEL_NAMES.get(family), device.type
        )

        runtime_cfg, data_cfg = build_data_cfg(args, family)
        loader, split_ds = build_dataloader(args, tokenizer)

        selector = None
        if args.selector_ckpt:
            with torch.no_grad():
                ex = split_ds[0]
                emb = encoder.token_embeddings(
                    torch.tensor(ex["ids"], device=device)[None],
                    torch.tensor(ex["attn_mask"], device=device)[None],
                )
            selector = RationaleSelectorModel(emb.size(-1)).to(device)
            selector.load_state_dict(torch.load(args.selector_ckpt, map_location=device)["model"], strict=False)
            selector.eval()

        for frac in FRACS:
            sums = dict(mean=0.0, p_best=0.0, best_ratio=0.0)
            count = token_sum = degenerate = clamp = trials = 0
            sel_cos = sel_rate = sel_n = 0

            for batch in tqdm(loader, desc=f"{family} frac={frac}"):
                ids, attn = batch["ids"].to(device), batch["attn_mask"].to(device)

                with torch.no_grad():
                    full = encoder.encode(ids, attn)

                selectable, special, n_select = selectable_masks(ids, attn, tokenizer)
                sims = []

                for _ in range(args.trials):
                    k = torch.clamp((n_select.float() * frac).round().long(), min=1)
                    clamp += (k == 1).sum().item()
                    trials += ids.size(0)

                    scores = torch.rand_like(attn.float()).masked_fill(~selectable, -1e9)
                    topk = scores.topk(int(k.max()), dim=1).indices
                    mask = torch.zeros_like(attn)
                    r = torch.arange(ids.size(0), device=device)[:, None]
                    mask[r, topk] = (torch.arange(topk.size(1), device=device) < k[:, None])

                    if args.keep_special:
                        mask |= special

                    with torch.no_grad():
                        sims.append(F.cosine_similarity(
                            encoder.encode(ids, mask), full, dim=-1
                        ))

                sims = torch.stack(sims)
                sims_sorted = sims.sort(0).values
                p_idx = int(args.best_percentile * (sims.size(0) - 1))
                p_best, mean = sims_sorted[p_idx], sims.mean(0)
                ratio = torch.where(mean != 0, p_best / mean, torch.inf)

                sums["mean"] += mean.sum().item()
                sums["p_best"] += p_best.sum().item()
                sums["best_ratio"] += ratio.sum().item()
                degenerate += (sims.std(0) == 0).sum().item()

                token_sum += attn.sum().item()
                count += ids.size(0)

                if selector:
                    with torch.no_grad():
                        gates, _ = selector(encoder.token_embeddings(ids, attn), attn, deterministic=True)
                        rep = encoder.encode(ids, attn * gates)
                        sel_cos += F.cosine_similarity(rep, full, -1).sum().item()
                        sel_rate += (gates.sum(1) / attn.sum(1)).sum().item()
                        sel_n += ids.size(0)

            print(f"{args.dataset} {args.split} | {family} frac={frac}")
            print(f"Avg tokens: {token_sum / count:.2f}  trials={args.trials}")
            print(f"Best/mean@p{int(100*args.best_percentile)} "
                  f"{sums['p_best']/count:.4f} / {sums['mean']/count:.4f} "
                  f"ratio={sums['best_ratio']/count:.4f}")
            print(f"Degenerate: {100*degenerate/count:.2f}%")
            print(f"Clamp k<1: {100*clamp/max(1,trials):.2f}%")
            if sel_n:
                print(f"Selector: cos={sel_cos/sel_n:.4f} rate={sel_rate/sel_n:.4f}")


if __name__ == "__main__":
    main()
