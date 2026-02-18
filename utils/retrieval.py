import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf
from datasets import load_dataset
from scipy.stats import spearmanr
from numpy import linspace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_CFG_PATH = os.path.join(
    os.path.dirname(__file__),
    "retrieval_conf",
    "default.yaml",
)

from src.sentence import build_sentence_encoder
from src.selector import RationaleSelectorModel


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def batch_tokenize(tokenizer, sentences, device, max_length):
    out = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in out.items()}


def get_rhos(cfg):
    return list(linspace(
        cfg.eval.rho_sweep.start,
        cfg.eval.rho_sweep.end,
        cfg.eval.rho_sweep.steps,
    ))


def mask_special_tokens(input_ids, tokenizer):
    """
    Returns a float mask [B,T] that is 0 for special tokens
    (CLS/SEP/PAD/BOS/EOS if defined), 1 otherwise.
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

    if not special_ids:
        return torch.ones_like(input_ids, dtype=torch.float)

    mask = torch.ones_like(input_ids, dtype=torch.float)
    for sid in special_ids:
        mask = mask * (input_ids != sid).float()
    return mask


# ------------------------------------------------------------
# Baseline (no selection)
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Selector Sweep
# Selection applied:
#   1) at embedding level (zeroing token embeddings)
#   2) at pooling level (mask)
# ------------------------------------------------------------
@torch.no_grad()
def eval_selector_sweep(ds, encoder, tokenizer, selector, cfg, device):
    rhos = get_rhos(cfg)
    bs = cfg.eval.batch_size
    max_len = cfg.eval.max_length

    sims = {rho: [] for rho in rhos}
    labels = []

    for i in tqdm(range(0, len(ds), bs), desc="STS-B selector"):
        batch = ds[i:i + bs]

        t1 = batch_tokenize(tokenizer, batch["sentence1"], device, max_len)
        t2 = batch_tokenize(tokenizer, batch["sentence2"], device, max_len)

        a1 = t1["attention_mask"]
        a2 = t2["attention_mask"]

        # 1) FULL embeddings for selector scoring (unmasked attention)
        e1_full = encoder.token_embeddings(t1["input_ids"], a1)
        e2_full = encoder.token_embeddings(t2["input_ids"], a2)

        # selector returns hard masks (CPU)
        _, g1_sweep, *_ = selector(t1["input_ids"], e1_full, a1)
        _, g2_sweep, *_ = selector(t2["input_ids"], e2_full, a2)

        g1_sweep = [g.to(device).float() for g in g1_sweep]
        g2_sweep = [g.to(device).float() for g in g2_sweep]

        for i_rho, rho in enumerate(rhos):
            if rho >= 1.0 - 1e-12:
                # MUST match baseline exactly
                new_a1 = a1
                new_a2 = a2
            else:
                # Treat special tokens like any other: just apply selector mask over valid tokens
                hard1 = g1_sweep[i_rho]
                hard2 = g2_sweep[i_rho]

                new_a1 = hard1 * a1
                new_a2 = hard2 * a2

            # 2) Recompute embeddings under the masked attention graph
            e1_masked = encoder.token_embeddings(t1["input_ids"], new_a1)
            e2_masked = encoder.token_embeddings(t2["input_ids"], new_a2)

            # 3) Pool using the same mask used in attention
            z1 = encoder.pool(e1_masked, new_a1)
            z2 = encoder.pool(e2_masked, new_a2)

            sims[rho].extend(F.cosine_similarity(z1, z2).cpu().tolist())

        labels.extend(batch["label"])

    return {rho: float(spearmanr(sims[rho], labels)[0]) for rho in rhos}

# ------------------------------------------------------------
# Random Sweep
# Same embedding-level masking as selector
# ------------------------------------------------------------
@torch.no_grad()
def eval_random_sweep(ds, encoder, tokenizer, cfg, device):
    rhos = get_rhos(cfg)
    runs = cfg.eval.random_selector.runs
    bs = cfg.eval.batch_size
    max_len = cfg.eval.max_length

    acc = {rho: [] for rho in rhos}

    for run in range(runs):
        torch.manual_seed(cfg.eval.random_selector.seed + run)

        sims = {rho: [] for rho in rhos}
        labels = []

        for i in tqdm(range(0, len(ds), bs), leave=False,
                      desc=f"STS-B random {run+1}/{runs}"):

            batch = ds[i:i + bs]

            t1 = batch_tokenize(tokenizer, batch["sentence1"], device, max_len)
            t2 = batch_tokenize(tokenizer, batch["sentence2"], device, max_len)

            a1 = t1["attention_mask"]
            a2 = t2["attention_mask"]

            T1 = a1.sum(1)
            T2 = a2.sum(1)

            for rho in rhos:
                if rho >= 1.0 - 1e-12:
                    # MUST match baseline exactly
                    new_a1 = a1
                    new_a2 = a2
                else:
                    k1 = torch.clamp((T1.float() * rho).round().long(), min=1)
                    k2 = torch.clamp((T2.float() * rho).round().long(), min=1)

                    hard1 = torch.zeros_like(a1, dtype=torch.float, device=device)
                    hard2 = torch.zeros_like(a2, dtype=torch.float, device=device)

                    for b in range(a1.size(0)):
                        valid1 = (a1[b] == 1).nonzero(as_tuple=False).squeeze(1)
                        valid2 = (a2[b] == 1).nonzero(as_tuple=False).squeeze(1)

                        if valid1.numel() > 0:
                            kb = min(int(k1[b].item()), valid1.numel())
                            rvals = torch.rand(valid1.numel(), device=device)
                            topk = torch.topk(rvals, kb).indices
                            hard1[b, valid1[topk]] = 1.0

                        if valid2.numel() > 0:
                            kb = min(int(k2[b].item()), valid2.numel())
                            rvals = torch.rand(valid2.numel(), device=device)
                            topk = torch.topk(rvals, kb).indices
                            hard2[b, valid2[topk]] = 1.0

                    new_a1 = hard1 * a1
                    new_a2 = hard2 * a2

                # Recompute embeddings under masked attention
                e1_masked = encoder.token_embeddings(t1["input_ids"], new_a1)
                e2_masked = encoder.token_embeddings(t2["input_ids"], new_a2)

                z1 = encoder.pool(e1_masked, new_a1)
                z2 = encoder.pool(e2_masked, new_a2)

                sims[rho].extend(F.cosine_similarity(z1, z2).cpu().tolist())

            labels.extend(batch["label"])

        for rho in rhos:
            acc[rho].append(float(spearmanr(sims[rho], labels)[0]))

    return {rho: sum(v) / len(v) for rho, v in acc.items()}

# ------------------------------------------------------------
# Selector Loader
# ------------------------------------------------------------

def load_selector(signature, encoder, cfg, device):
    ckpt = os.path.join(PROJECT_ROOT, "outputs", "xps", signature, "model.pth")

    sweep_cfg = OmegaConf.create({
        "sweep_range": [
            cfg.eval.rho_sweep.start,
            cfg.eval.rho_sweep.end,
            cfg.eval.rho_sweep.steps,
        ]
    })

    sample_ids = torch.randint(0, 100, (1, 8), device=device)
    sample_attn = torch.ones_like(sample_ids, device=device)
    emb = encoder.token_embeddings(sample_ids, sample_attn)

    selector = RationaleSelectorModel(
        embedding_dim=emb.size(-1),
        sent_encoder=encoder,
        loss_cfg=sweep_cfg,
    ).to(device)

    state = torch.load(ckpt, map_location=device)
    selector.load_state_dict(state["model"], strict=False)
    selector.eval()

    for p in selector.parameters():
        p.requires_grad_(False)

    return selector


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

def print_results(base, ours, rand):
    print(f"\nBaseline Spearman ρ: {base:.4f}\n")
    print("rho\tours\tΔ(base)\trandom\tΔ(base)\tΔ(ours-rand)")
    for rho in ours:
        o, r = ours[rho], rand[rho]
        print(f"{rho:.4f}\t{o:.4f}\t{o-base:+.4f}\t{r:.4f}\t{r-base:+.4f}\t{o-r:+.4f}")


def plot(base, ours, rand, path):
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

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    xp_dir = os.path.join(PROJECT_ROOT, "outputs", "xps", cfg.selector.signature)

    device = cfg.runtime.device

    encoder, tokenizer = build_sentence_encoder(
        cfg.model.encoder.family,
        cfg.model.encoder.name,
        device,
    )

    ds = load_dataset("glue", "stsb", split=cfg.eval.split)

    base = eval_baseline(ds, encoder, tokenizer, cfg, device)
    selector = load_selector(cfg.selector.signature, encoder, cfg, device)

    ours = eval_selector_sweep(ds, encoder, tokenizer, selector, cfg, device)
    rand = eval_random_sweep(ds, encoder, tokenizer, cfg, device)

    print_results(base, ours, rand)
    plot(base, ours, rand, os.path.join(xp_dir, "spearman_vs_rho.png"))
    print("\nSaved plot to:", os.path.join(xp_dir, "spearman_vs_rho.png"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) == 2 else DEFAULT_CFG_PATH)
