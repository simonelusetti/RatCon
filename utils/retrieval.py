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
    # Match training paradigm: sweep_range lives under cfg.model.loss
    return list(linspace(
        cfg.model.loss.sweep_range[0],
        cfg.model.loss.sweep_range[1],
        cfg.model.loss.sweep_range[2],
    ))


# ------------------------------------------------------------
# Baseline
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
# Unified sweep engine (QUERY ONLY masked)
# ------------------------------------------------------------

@torch.no_grad()
def eval_sweep(
    ds,
    encoder,
    tokenizer,
    cfg,
    device,
    mask_generator,
    desc: str,
):
    rhos = get_rhos(cfg)
    bs = cfg.eval.batch_size
    max_len = cfg.eval.max_length

    sims = {rho: [] for rho in rhos}
    labels = []

    for i in tqdm(range(0, len(ds), bs), desc=desc):
        batch = ds[i:i + bs]

        # sentence1 = query (masked)
        # sentence2 = target (full)

        t1 = batch_tokenize(tokenizer, batch["sentence1"], device, max_len)
        t2 = batch_tokenize(tokenizer, batch["sentence2"], device, max_len)

        a1 = t1["attention_mask"]
        a2 = t2["attention_mask"]

        # ---- TARGET (compute ONCE per batch) ----
        e2_full = encoder.token_embeddings(t2["input_ids"], a2)
        z2 = encoder.pool(e2_full, a2)

        # ---- QUERY masks for ALL rhos (computed once per batch) ----
        new_a1_sweep = mask_generator(t1, a1, rhos)

        # ---- Sweep: only query changes ----
        for rho, new_a1 in zip(rhos, new_a1_sweep):
            e1_masked = encoder.token_embeddings(t1["input_ids"], new_a1)
            z1 = encoder.pool(e1_masked, new_a1)
            sims[rho].extend(F.cosine_similarity(z1, z2).cpu().tolist())

        labels.extend(batch["label"])

    return {rho: float(spearmanr(sims[rho], labels)[0]) for rho in rhos}


# ------------------------------------------------------------
# Mask generators
# ------------------------------------------------------------

def build_selector_mask_generator(selector, encoder, device):
    def mask_generator(t1, a1, rhos):
        # full embeddings for selector scoring
        e1_full = encoder.token_embeddings(t1["input_ids"], a1)

        _, g_sweep, *_ = selector(t1["input_ids"], e1_full, a1)
        g_sweep = [g.to(device).float() for g in g_sweep]

        new_a1_sweep = []
        for _, g in zip(rhos, g_sweep):
            new_a1_sweep.append(g * a1)
        return new_a1_sweep

    return mask_generator


def build_random_mask_generator(cfg, device):
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


# ------------------------------------------------------------
# Random sweep wrapper (multiple runs)
# ------------------------------------------------------------

@torch.no_grad()
def eval_random_sweep(ds, encoder, tokenizer, cfg, device):
    rhos = get_rhos(cfg)
    runs = cfg.eval.random_selector.runs

    acc = {rho: [] for rho in rhos}

    for run in range(runs):
        torch.manual_seed(cfg.eval.random_selector.seed + run)

        rand_mask_gen = build_random_mask_generator(cfg, device)
        out = eval_sweep(
            ds,
            encoder,
            tokenizer,
            cfg,
            device,
            rand_mask_gen,
            desc=f"STS-B random {run+1}/{runs}",
        )

        for rho in rhos:
            acc[rho].append(out[rho])

    return {rho: sum(v) / len(v) for rho, v in acc.items()}


# ------------------------------------------------------------
# Load Selector (new API: requires rhos)
# ------------------------------------------------------------

def load_selector(signature, encoder, cfg, device):
    ckpt = os.path.join(PROJECT_ROOT, "outputs", "xps", signature, "model.pth")
    rhos = get_rhos(cfg)

    sample_ids = torch.randint(0, 100, (1, 8), device=device)
    sample_attn = torch.ones_like(sample_ids, device=device)
    emb = encoder.token_embeddings(sample_ids, sample_attn)

    selector = RationaleSelectorModel(
        embedding_dim=emb.size(-1),
        loss_cfg=cfg.model.loss,
        sent_encoder=encoder,
        rhos=rhos,
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

    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main(cfg_path):
    cfg = OmegaConf.load(cfg_path)

    # experiment directory (like training script)
    xp_dir = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "xps",
        cfg.selector.signature,
    )

    os.makedirs(xp_dir, exist_ok=True)

    device = cfg.runtime.device

    encoder, tokenizer = build_sentence_encoder(
        cfg.model.encoder.family,
        cfg.model.encoder.name,
        device,
    )

    ds = load_dataset("glue", "stsb", split=cfg.eval.split)

    base = eval_baseline(ds, encoder, tokenizer, cfg, device)

    selector = load_selector(cfg.selector.signature, encoder, cfg, device)
    selector_mask_gen = build_selector_mask_generator(selector, encoder, device)

    ours = eval_sweep(
        ds,
        encoder,
        tokenizer,
        cfg,
        device,
        selector_mask_gen,
        desc="STS-B selector",
    )

    rand = eval_random_sweep(ds, encoder, tokenizer, cfg, device)

    print_results(base, ours, rand)

    plot_path = os.path.join(xp_dir, "spearman_vs_rho.png")
    plot(base, ours, rand, plot_path)

    print("\nSaved plot to:", plot_path)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) == 2 else DEFAULT_CFG_PATH)
