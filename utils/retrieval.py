#!/usr/bin/env python3
from __future__ import annotations

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

from dora import hydra_main
from omegaconf import DictConfig

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Path setup (Hydra-safe)
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.selector import RationaleSelectorModel
from src.sentence import build_sentence_encoder

# ---------------------------------------------------------------------
# Dataset (Retrieval-style)
# ---------------------------------------------------------------------

class RetrievalDataset(Dataset):
    def __init__(self, tokenizer, queries, docs):
        self.tokenizer = tokenizer
        self.queries = queries
        self.docs = docs

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        tq = self.tokenizer(self.queries[idx], truncation=True, return_tensors="pt")
        td = self.tokenizer(self.docs[idx], truncation=True, return_tensors="pt")

        return {
            "q_ids": tq["input_ids"].squeeze(0),
            "q_attn": tq["attention_mask"].squeeze(0),
            "d_ids": td["input_ids"].squeeze(0),
            "d_attn": td["attention_mask"].squeeze(0),
        }


def collate_fn(batch):
    def pad(key, dtype):
        return pad_sequence(
            [x[key].to(dtype) for x in batch],
            batch_first=True,
            padding_value=0,
        )

    return {
        "q_ids": pad("q_ids", torch.long),
        "q_attn": pad("q_attn", torch.float),
        "d_ids": pad("d_ids", torch.long),
        "d_attn": pad("d_attn", torch.float),
    }

# ---------------------------------------------------------------------
# Selector utilities
# ---------------------------------------------------------------------

@torch.no_grad()
def compute_gates(encoder, selector, ids, attn):
    token_emb = encoder.token_embeddings(ids, attn)

    if isinstance(selector, RationaleSelectorModel):
        g, _ = selector(token_emb, attn, deterministic=True)
        return g
    else:
        return selector(token_emb, attn)


def random_selector(p: float):
    @torch.no_grad()
    def _selector(token_emb, attn):
        return (torch.rand_like(attn) < p).float() * attn
    return _selector

# ---------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------

def recall_at_k(ranks, k):
    return float(np.mean([1.0 if r < k else 0.0 for r in ranks]))


def mean_reciprocal_rank(ranks):
    return float(np.mean([1.0 / (r + 1) for r in ranks]))


def compute_metrics(ranks):
    return {
        "R@1": recall_at_k(ranks, 1),
        "R@5": recall_at_k(ranks, 5),
        "R@10": recall_at_k(ranks, 10),
        "MRR": mean_reciprocal_rank(ranks),
    }


def aggregate_metrics(metrics_list):
    keys = metrics_list[0].keys()
    mean = {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}
    std = {k: float(np.std([m[k] for m in metrics_list])) for k in keys}
    return mean, std

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_retrieval(encoder, selector, dataloader, device):
    ranks = []

    for batch in tqdm(dataloader, desc="Retrieval eval"):
        q_ids = batch["q_ids"].to(device)
        q_attn = batch["q_attn"].to(device)
        d_ids = batch["d_ids"].to(device)
        d_attn = batch["d_attn"].to(device)

        if selector is None:
            q_emb = encoder.encode(q_ids, q_attn)
            d_emb = encoder.encode(d_ids, d_attn)
        else:
            q_g = compute_gates(encoder, selector, q_ids, q_attn)
            d_g = compute_gates(encoder, selector, d_ids, d_attn)

            q_emb = encoder.encode(q_ids, q_attn * q_g)
            d_emb = encoder.encode(d_ids, d_attn * d_g)

        q_emb = torch.nn.functional.normalize(q_emb, dim=1)
        d_emb = torch.nn.functional.normalize(d_emb, dim=1)

        sim = q_emb @ d_emb.T
        sorted_idx = torch.argsort(sim, dim=1, descending=True)

        for i in range(sim.size(0)):
            rank = (sorted_idx[i] == i).nonzero(as_tuple=False).item()
            ranks.append(rank)

    return ranks


@torch.no_grad()
def estimate_selection_rate(encoder, selector, dataloader, device):
    sel = 0.0
    tot = 0.0

    for batch in tqdm(dataloader, desc="Estimate sparsity"):
        q_ids = batch["q_ids"].to(device)
        q_attn = batch["q_attn"].to(device)
        d_ids = batch["d_ids"].to(device)
        d_attn = batch["d_attn"].to(device)

        q_g = compute_gates(encoder, selector, q_ids, q_attn)
        d_g = compute_gates(encoder, selector, d_ids, d_attn)

        sel += float((q_g * q_attn).sum() + (d_g * d_attn).sum())
        tot += float(q_attn.sum() + d_attn.sum())

    return sel / max(tot, 1.0)

# ---------------------------------------------------------------------
# Main (Hydra)
# ---------------------------------------------------------------------

@hydra_main(config_path="retrieval_conf", config_name="default", version_base="1.1")
def main(cfg: DictConfig):

    device = cfg.runtime.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        device = "cpu"

    encoder, tokenizer = build_sentence_encoder(
        family=cfg.model.encoder.family,
        encoder_name=cfg.model.encoder.name,
        device=device,
    )

    ds = load_dataset("glue", "stsb", split=cfg.eval.split)
    dataset = RetrievalDataset(tokenizer, ds["sentence1"], ds["sentence2"])
    loader = DataLoader(
        dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------
    # Base encoder
    # --------------------------------------------------
    print("\nEvaluating base encoder...")
    base_ranks = evaluate_retrieval(encoder, None, loader, device)
    base_metrics = compute_metrics(base_ranks)
    print(base_metrics)

    # --------------------------------------------------
    # Selector
    # --------------------------------------------------
    with torch.no_grad():
        sample = dataset[0]
        ids = sample["q_ids"].unsqueeze(0).to(device)
        attn = sample["q_attn"].unsqueeze(0).to(device)
        model_dim = encoder.token_embeddings(ids, attn).shape[-1]

    selector = RationaleSelectorModel(
        embedding_dim=model_dim,
        rho=cfg.model.selector.rho,
        tau=cfg.model.selector.tau,
    ).to(device)

    state = torch.load(cfg.eval.checkpoint, map_location=device)
    selector.load_state_dict(state["model"], strict=False)
    selector.eval()
    for p in selector.parameters():
        p.requires_grad = False

    # --------------------------------------------------
    # Sparsity
    # --------------------------------------------------
    print("\nEstimating selector sparsity...")
    p_sel = estimate_selection_rate(encoder, selector, loader, device)
    print(f"Mean selection rate: {p_sel:.4f}")

    # --------------------------------------------------
    # Trained selector
    # --------------------------------------------------
    print("\nEvaluating trained selector...")
    trained_ranks = evaluate_retrieval(encoder, selector, loader, device)
    trained_metrics = compute_metrics(trained_ranks)
    print(trained_metrics)

    # --------------------------------------------------
    # Random baselines (aggregated)
    # --------------------------------------------------
    print("\nRandom selector baselines...")
    rand_metrics_all = []

    for seed in cfg.eval.random_seeds:
        torch.manual_seed(seed)
        sel_rand = random_selector(p_sel)
        ranks = evaluate_retrieval(encoder, sel_rand, loader, device)
        rand_metrics_all.append(compute_metrics(ranks))

    rand_mean, rand_std = aggregate_metrics(rand_metrics_all)

    print("\nRandom selector (mean ± std):")
    for k in rand_mean:
        print(f"{k}: {rand_mean[k]:.4f} ± {rand_std[k]:.4f}")

    # --------------------------------------------------
    # Comparisons
    # --------------------------------------------------
    print("\nΔ (trained - random mean):")
    for k in trained_metrics:
        print(f"{k}: {trained_metrics[k] - rand_mean[k]:+.4f}")

    print("\nΔ (base - trained):")
    for k in trained_metrics:
        print(f"{k}: {base_metrics[k] - trained_metrics[k]:+.4f}")


if __name__ == "__main__":
    main()
