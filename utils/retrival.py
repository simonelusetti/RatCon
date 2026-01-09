#!/usr/bin/env python3
from __future__ import annotations

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

from luse.selector import RationaleSelectorModel
from luse.sentence import build_sentence_encoder

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

MODEL_FAMILY = "sbert"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHECKPOINT_PATH = "/home/simonelusetti/RatCon/outputs/xps/47776c72/model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
SPLIT = "validation"

RECALL_K = (1, 5, 10)
RANDOM_SEEDS = [0, 1, 2, 3, 4]

# ---------------------------------------------------------------------
# Dataset (Retrieval-style)
# ---------------------------------------------------------------------

class RetrievalDataset(Dataset):
    """
    Each item:
      query  = sentence1
      doc    = sentence2
      label  = implicit (same index is the positive)
    """
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

def load_selector(checkpoint_path, model_dim):
    state = torch.load(checkpoint_path, map_location=DEVICE)
    selector = RationaleSelectorModel(model_dim).to(DEVICE)
    selector.load_state_dict(state["model"], strict=True)
    selector.eval()
    for p in selector.parameters():
        p.requires_grad = False
    return selector


@torch.no_grad()
def compute_gates(encoder, selector, ids, attn):
    token_emb = encoder.token_embeddings(ids, attn)
    return selector(token_emb, attn)


def random_selector(p: float):
    @torch.no_grad()
    def _selector(token_emb, attn):
        # attn is float {0,1}. Keep with prob p only on valid tokens.
        return (torch.rand_like(attn) < p).float() * attn
    return _selector

# ---------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------

def recall_at_k(ranks, k):
    return float(np.mean([1.0 if r < k else 0.0 for r in ranks]))


def mean_reciprocal_rank(ranks):
    return float(np.mean([1.0 / (r + 1) for r in ranks]))


def fmt_metrics(m: dict) -> str:
    keys = [f"R@{k}" for k in RECALL_K] + ["MRR"]
    return " | ".join(f"{k}: {m[k]:.4f}" for k in keys)

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_retrieval(encoder, selector, dataloader):
    """
    In-batch retrieval:
      - each query retrieves over docs in the same batch
      - positive is the diagonal element
    """
    ranks = []

    for batch in tqdm(dataloader, desc="Retrieval eval"):
        q_ids = batch["q_ids"].to(DEVICE)
        q_attn = batch["q_attn"].to(DEVICE)
        d_ids = batch["d_ids"].to(DEVICE)
        d_attn = batch["d_attn"].to(DEVICE)

        if selector is None:
            q_emb = encoder.encode(q_ids, q_attn)
            d_emb = encoder.encode(d_ids, d_attn)
        else:
            if isinstance(selector, RationaleSelectorModel):
                q_g = compute_gates(encoder, selector, q_ids, q_attn)
                d_g = compute_gates(encoder, selector, d_ids, d_attn)
            else:
                # random selector expects (token_emb, attn)
                q_te = encoder.token_embeddings(q_ids, q_attn)
                d_te = encoder.token_embeddings(d_ids, d_attn)
                q_g = selector(q_te, q_attn)
                d_g = selector(d_te, d_attn)

            q_emb = encoder.encode(q_ids, q_attn * q_g)
            d_emb = encoder.encode(d_ids, d_attn * d_g)

        q_emb = torch.nn.functional.normalize(q_emb, dim=1)
        d_emb = torch.nn.functional.normalize(d_emb, dim=1)
        sim = q_emb @ d_emb.T  # [B, B]

        sorted_idx = torch.argsort(sim, dim=1, descending=True)
        for i in range(sim.size(0)):
            rank = (sorted_idx[i] == i).nonzero(as_tuple=False).item()
            ranks.append(rank)

    metrics = {}
    for k in RECALL_K:
        metrics[f"R@{k}"] = recall_at_k(ranks, k)
    metrics["MRR"] = mean_reciprocal_rank(ranks)
    return metrics


@torch.no_grad()
def estimate_selection_rate(encoder, selector_model, dataloader):
    sel = 0.0
    tot = 0.0

    for batch in tqdm(dataloader, desc="Estimate sparsity"):
        q_ids = batch["q_ids"].to(DEVICE)
        q_attn = batch["q_attn"].to(DEVICE)
        d_ids = batch["d_ids"].to(DEVICE)
        d_attn = batch["d_attn"].to(DEVICE)

        q_g = compute_gates(encoder, selector_model, q_ids, q_attn)
        d_g = compute_gates(encoder, selector_model, d_ids, d_attn)

        sel += float((q_g * q_attn).sum().item() + (d_g * d_attn).sum().item())
        tot += float(q_attn.sum().item() + d_attn.sum().item())

    return sel / max(tot, 1.0)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ds = load_dataset("glue", "stsb", split=SPLIT)

    encoder, tokenizer = build_sentence_encoder(
        family=MODEL_FAMILY,
        encoder_name=MODEL_NAME,
        device=DEVICE,
    )

    dataset = RetrievalDataset(tokenizer, ds["sentence1"], ds["sentence2"])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print("\nEvaluating base encoder (retrieval)...")
    base_metrics = evaluate_retrieval(encoder, None, loader)
    print(fmt_metrics(base_metrics))

    # infer selector input dim from token embeddings
    with torch.no_grad():
        sample = dataset[0]
        ids = sample["q_ids"].unsqueeze(0).to(DEVICE)
        attn = sample["q_attn"].unsqueeze(0).to(DEVICE)
        model_dim = encoder.token_embeddings(ids, attn).shape[-1]

    selector = load_selector(CHECKPOINT_PATH, model_dim)

    print("\nEstimating trained selector sparsity...")
    p = estimate_selection_rate(encoder, selector, loader)
    print(f"Mean selection rate p: {p:.4f}")

    print("\nEvaluating trained selector (retrieval)...")
    trained_metrics = evaluate_retrieval(encoder, selector, loader)
    print(fmt_metrics(trained_metrics))

    print("\nRandom baselines (same sparsity p)...")
    rand_metrics_all = []
    for seed in RANDOM_SEEDS:
        torch.manual_seed(seed)
        sel_rand = random_selector(p)
        m = evaluate_retrieval(encoder, sel_rand, loader)
        rand_metrics_all.append(m)
        print(f"  seed={seed} | {fmt_metrics(m)}")

    # aggregate random
    keys = [f"R@{k}" for k in RECALL_K] + ["MRR"]
    rand_mean = {k: float(np.mean([m[k] for m in rand_metrics_all])) for k in keys}
    rand_std = {k: float(np.std([m[k] for m in rand_metrics_all])) for k in keys}

    print("\nRandom selector (mean ± std):")
    for k in keys:
        print(f"  {k}: {rand_mean[k]:.4f} ± {rand_std[k]:.4f}")

    print("\nΔ (base - trained):")
    for k in keys:
        print(f"  {k}: {base_metrics[k] - trained_metrics[k]:+.4f}")

    print("\nΔ (trained - random mean):")
    for k in keys:
        print(f"  {k}: {trained_metrics[k] - rand_mean[k]:+.4f}")


if __name__ == "__main__":
    main()
