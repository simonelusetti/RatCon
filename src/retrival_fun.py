# src/metrics_stsb.py
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import spearmanr
from tqdm import tqdm
from numpy import linspace


def get_rhos(cfg):
    return list(linspace(
        cfg.model.loss.sweep_range[0],
        cfg.model.loss.sweep_range[1],
        cfg.model.loss.sweep_range[2],
    ))


def batch_tokenize(tokenizer, sentences, device, max_length):
    out = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in out.items()}


@torch.no_grad()
def eval_baseline(ds, encoder, tokenizer, cfg, device):
    sims, labels = [], []
    bs = cfg.eval.batch_size
    max_len = cfg.eval.max_length

    for i in tqdm(range(0, len(ds), bs), desc="STS-B baseline"):
        batch = ds[i:i + bs]

        t1 = batch_tokenize(tokenizer, batch["sentence1"], device, max_len)
        t2 = batch_tokenize(tokenizer, batch["sentence2"], device, max_len)

        e1 = encoder.token_embeddings(t1["input_ids"], t1["attention_mask"])
        e2 = encoder.token_embeddings(t2["input_ids"], t2["attention_mask"])

        z1 = encoder.pool(e1, t1["attention_mask"])
        z2 = encoder.pool(e2, t2["attention_mask"])

        sims.extend(F.cosine_similarity(z1, z2).cpu().tolist())
        labels.extend(batch["label"])

    return float(spearmanr(sims, labels)[0])


@torch.no_grad()
def eval_sweep(ds, encoder, tokenizer, cfg, device, mask_generator, desc: str):
    rhos = get_rhos(cfg)
    bs = cfg.eval.batch_size
    max_len = cfg.eval.max_length

    sims = {rho: [] for rho in rhos}
    labels = []

    for i in tqdm(range(0, len(ds), bs), desc=desc):
        batch = ds[i:i + bs]

        t1 = batch_tokenize(tokenizer, batch["sentence1"], device, max_len)  # query
        t2 = batch_tokenize(tokenizer, batch["sentence2"], device, max_len)  # target

        a1 = t1["attention_mask"]
        a2 = t2["attention_mask"]

        # target once
        e2_full = encoder.token_embeddings(t2["input_ids"], a2)
        z2 = encoder.pool(e2_full, a2)

        # query masks once per batch
        new_a1_sweep = mask_generator(t1, a1, rhos)

        for rho, new_a1 in zip(rhos, new_a1_sweep):
            e1_masked = encoder.token_embeddings(t1["input_ids"], new_a1)
            z1 = encoder.pool(e1_masked, new_a1)
            sims[rho].extend(F.cosine_similarity(z1, z2).cpu().tolist())

        labels.extend(batch["label"])

    return {rho: float(spearmanr(sims[rho], labels)[0]) for rho in rhos}


def build_selector_mask_generator(selector, encoder, device):
    @torch.no_grad()
    def mask_generator(t1, a1, rhos):
        e1_full = encoder.token_embeddings(t1["input_ids"], a1)
        _, g_sweep, *_ = selector(t1["input_ids"], e1_full, a1)

        new_a1_sweep = []
        for g in g_sweep:
            g = g.detach().to(device).float()
            new_a1_sweep.append(g * a1)
        return new_a1_sweep

    return mask_generator


def build_random_mask_generator(cfg, device):
    @torch.no_grad()
    def mask_generator(t1, a1, rhos):
        T1 = a1.sum(1)
        new_a1_sweep = []

        for rho in rhos:
            k1 = torch.clamp((T1.float() * rho).round().long(), min=1)
            hard1 = torch.zeros_like(a1, dtype=torch.float, device=device)

            for b in range(a1.size(0)):
                valid = (a1[b] == 1).nonzero(as_tuple=False).squeeze(1)
                if valid.numel() == 0:
                    continue

                kb = min(int(k1[b].item()), valid.numel())
                rvals = torch.rand(valid.numel(), device=device)
                topk = torch.topk(rvals, kb).indices
                hard1[b, valid[topk]] = 1.0

            new_a1_sweep.append(hard1 * a1)

        return new_a1_sweep

    return mask_generator


@torch.no_grad()
def eval_random_sweep(ds, encoder, tokenizer, cfg, device):
    rhos = get_rhos(cfg)
    runs = cfg.eval.random_selector.runs
    acc = {rho: [] for rho in rhos}

    for run in range(runs):
        torch.manual_seed(cfg.eval.random_selector.seed + run)
        rand_mask_gen = build_random_mask_generator(cfg, device)

        out = eval_sweep(
            ds, encoder, tokenizer, cfg, device,
            rand_mask_gen,
            desc=f"STS-B random {run+1}/{runs}",
        )
        for rho in rhos:
            acc[rho].append(out[rho])

    return {rho: sum(v) / len(v) for rho, v in acc.items()}


def plot(base, ours, rand):
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

    plt.savefig("spearman_vs_rho.png", dpi=300)
    plt.close()


@torch.no_grad()
def run_stsb_sweep(cfg, device, encoder, tokenizer, selector):
    ds = load_dataset("glue", "stsb", split=cfg.eval.split)

    base = eval_baseline(ds, encoder, tokenizer, cfg, device)

    selector_mask_gen = build_selector_mask_generator(selector, encoder, device)
    ours = eval_sweep(ds, encoder, tokenizer, cfg, device, selector_mask_gen, desc="STS-B selector")
    rand = eval_random_sweep(ds, encoder, tokenizer, cfg, device)

    plot(base, ours, rand)

    return base, ours, rand