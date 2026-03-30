import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from scipy.stats import spearmanr
from tqdm import tqdm
from numpy import linspace


def should_disable_tqdm() -> bool:
    if os.environ.get("DISABLE_TQDM"):
        return True
    if not sys.stderr.isatty():
        return True
    return False


def build_non_special_mask(tokenizer, input_ids, attention_mask, device):
    special_tokens_mask = torch.tensor(
        [
            tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
            for ids in input_ids.detach().cpu().tolist()
        ],
        dtype=attention_mask.dtype,
        device=device,
    )
    return attention_mask * (1 - special_tokens_mask)


def get_rhos(cfg):
    return list(linspace(
        cfg.sweep_range[0],
        cfg.sweep_range[1],
        cfg.sweep_range[2],
    ))


class STSBDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "sentence1": item["sentence1"],
            "sentence2": item["sentence2"],
            "label": float(item["label"]),
        }


def build_stsb_collate(tokenizer, device, max_length):
    def collate_fn(batch):
        s1 = [x["sentence1"] for x in batch]
        s2 = [x["sentence2"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.float)

        t1 = tokenizer(
            s1,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        t2 = tokenizer(
            s2,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "t1": {k: v.to(device) for k, v in t1.items()},
            "t2": {k: v.to(device) for k, v in t2.items()},
            "labels": labels,
        }

    return collate_fn


@torch.no_grad()
def eval_baseline(loader, encoder):
    sims, labels = [], []

    for batch in tqdm(
        loader,
        desc="STS-B baseline",
        leave=False,
        dynamic_ncols=True,
        disable=should_disable_tqdm(),
        file=sys.stderr,
    ):
        t1, t2 = batch["t1"], batch["t2"]

        e1 = encoder.token_embeddings(t1["input_ids"], t1["attention_mask"])
        e2 = encoder.token_embeddings(t2["input_ids"], t2["attention_mask"])

        z1 = encoder.pool(e1, t1["attention_mask"])
        z2 = encoder.pool(e2, t2["attention_mask"])

        sims.extend(F.cosine_similarity(z1, z2).cpu().tolist())
        labels.extend(batch["labels"].tolist())

    return float(spearmanr(sims, labels)[0])


@torch.no_grad()
def eval_sweep(
    loader,
    encoder,
    mask_generator,
    eval_cfg,
    desc: str = "STS-B selector",
    progress_bar=None,
):
    rhos = get_rhos(eval_cfg)

    sims = {rho: [] for rho in rhos}
    labels = []

    iterator = loader
    if progress_bar is None:
        iterator = tqdm(
            loader,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            disable=should_disable_tqdm(),
            file=sys.stderr,
        )

    for batch in iterator:
        t1, t2 = batch["t1"], batch["t2"]

        a1 = t1["attention_mask"]
        a2 = t2["attention_mask"]

        e2_full = encoder.token_embeddings(t2["input_ids"], a2)
        z2 = encoder.pool(e2_full, a2)

        new_a1_sweep = mask_generator(t1, a1, rhos)

        for rho, new_a1 in zip(rhos, new_a1_sweep):
            e1_masked = encoder.token_embeddings(t1["input_ids"], new_a1)
            z1 = encoder.pool(e1_masked, new_a1)
            sims[rho].extend(F.cosine_similarity(z1, z2).cpu().tolist())

        labels.extend(batch["labels"].tolist())
        if progress_bar is not None:
            progress_bar.update(1)

    return {rho: float(spearmanr(sims[rho], labels)[0]) for rho in rhos}


def build_selector_mask_generator(
    selector,
    encoder,
    tokenizer,
    device,
    keep_special: bool = True,
):
    @torch.no_grad()
    def mask_generator(t1, a1, rhos):
        e1_full = encoder.token_embeddings(t1["input_ids"], a1)
        selection_mask = a1
        if not keep_special:
            selection_mask = build_non_special_mask(tokenizer, t1["input_ids"], a1, device)
        _, g_sweep, *_ = selector(
            t1["input_ids"],
            e1_full,
            a1,
            rhos=rhos,
            selection_mask=selection_mask,
        )

        new_a1_sweep = []
        for g in g_sweep:
            g = g.detach().to(device).float()
            new_a1_sweep.append(g * a1)

        return new_a1_sweep

    return mask_generator


def build_random_mask_generator(cfg, tokenizer, device, keep_special: bool = True):
    @torch.no_grad()
    def mask_generator(t1, a1, rhos):
        candidate_mask = a1
        if not keep_special:
            candidate_mask = build_non_special_mask(tokenizer, t1["input_ids"], a1, device)

        T1 = candidate_mask.sum(1)
        new_a1_sweep = []

        for rho in rhos:
            k1 = (T1.float() * rho).round().long()
            k1 = torch.where(T1 > 0, torch.clamp(k1, min=1), torch.zeros_like(k1))
            hard1 = torch.zeros_like(a1, dtype=torch.float, device=device)

            for b in range(a1.size(0)):
                valid = (candidate_mask[b] == 1).nonzero(as_tuple=False).squeeze(1)
                if valid.numel() == 0:
                    continue

                kb = min(int(k1[b].item()), valid.numel())
                rvals = torch.rand(valid.numel(), device=device)
                topk = torch.topk(rvals, kb).indices
                hard1[b, valid[topk]] = 1.0

            new_a1_sweep.append(hard1 * candidate_mask)

        return new_a1_sweep

    return mask_generator


@torch.no_grad()
def eval_random_sweep(loader, encoder, tokenizer, eval_cfg, device, keep_special: bool = True):
    rhos = get_rhos(eval_cfg)
    runs = eval_cfg.random_selector.runs

    acc = {rho: [] for rho in rhos}

    with tqdm(
        total=runs * len(loader),
        desc="STS-B random selector",
        leave=False,
        dynamic_ncols=True,
        disable=should_disable_tqdm(),
        file=sys.stderr,
    ) as pbar:
        for run in range(runs):
            torch.manual_seed(eval_cfg.random_selector.seed + run)
            rand_mask_gen = build_random_mask_generator(
                eval_cfg,
                tokenizer,
                device,
                keep_special=keep_special,
            )

            out = eval_sweep(
                loader,
                encoder,
                rand_mask_gen,
                eval_cfg,
                progress_bar=pbar,
            )

            for rho in rhos:
                acc[rho].append(out[rho])

    return {rho: sum(v) / len(v) for rho, v in acc.items()}


def plot(base, ours, rand, out_path: str = "spearman_vs_rho.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rhos = list(ours.keys())

    plt.figure(figsize=(7, 5))
    plt.plot(rhos, [base] * len(rhos), "--", label="Baseline")
    plt.plot(rhos, [ours[r] for r in rhos], "o-", label="Trained selector")
    plt.plot(rhos, [rand[r] for r in rhos], "x-", label="Random selector")
    plt.xlabel("Selection rate (ρ)")
    plt.ylabel("Spearman correlation (STS-B)")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


@torch.no_grad()
def run_stsb_sweep(cfg, device, encoder, tokenizer, selector, out_path: str = "spearman_vs_rho.png"):
    hf_ds = load_dataset("glue", "stsb", split=cfg.runtime.eval.split)
    ds = STSBDataset(hf_ds)

    collate_fn = build_stsb_collate(
        tokenizer,
        device,
        cfg.data.max_length,
    )

    loader = DataLoader(
        ds,
        batch_size=cfg.runtime.data.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    base = eval_baseline(loader, encoder)

    selector_cfg = cfg.model.selector
    keep_special = bool(selector_cfg.get("keep_special", True))
    selector_mask_gen = build_selector_mask_generator(
        selector,
        encoder,
        tokenizer,
        device,
        keep_special=keep_special,
    )
    
    ours = eval_sweep(loader, encoder, selector_mask_gen, cfg.runtime.eval)
    rand = eval_random_sweep(
        loader,
        encoder,
        tokenizer,
        cfg.runtime.eval,
        device,
        keep_special=keep_special,
    )

    plot(base, ours, rand, out_path=out_path)

    return base, ours, rand
