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
from src.selector import RationaleSelectorModel, probabilistic_top_k
from src.utils import configure_runtime


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


@torch.no_grad()
def eval_baseline(encoder, tokenizer, cfg, device):
    ds = load_dataset("glue", "stsb", split=cfg.eval.split)

    sims, labels = [], []
    bs = cfg.eval.batch_size
    max_len = cfg.eval.max_length

    for i in tqdm(range(0, len(ds), bs), desc="STS-B eval (baseline)"):
        batch = ds[i:i + bs]

        t1 = batch_tokenize(tokenizer, batch["sentence1"], device, max_len)
        t2 = batch_tokenize(tokenizer, batch["sentence2"], device, max_len)

        e1 = encoder.token_embeddings(t1["input_ids"], t1["attention_mask"])
        e2 = encoder.token_embeddings(t2["input_ids"], t2["attention_mask"])

        s1 = encoder.pool(e1, t1["attention_mask"])
        s2 = encoder.pool(e2, t2["attention_mask"])

        sims.extend(F.cosine_similarity(s1, s2).cpu().tolist())
        labels.extend(batch["label"])

    return float(spearmanr(sims, labels)[0])


@torch.no_grad()
def eval_selector_sweep(encoder, tokenizer, selector, cfg, device):
    ds = load_dataset("glue", "stsb", split=cfg.eval.split)


    rhos = get_rhos(cfg)
    bs = cfg.eval.batch_size
    max_len = cfg.eval.max_length
    tau = cfg.selector.tau

    sims = {rho: [] for rho in rhos}
    labels = []

    for i in tqdm(range(0, len(ds), bs), desc="STS-B eval (selector sweep)"):
        batch = ds[i:i + bs]

        t1 = batch_tokenize(tokenizer, batch["sentence1"], device, max_len)
        t2 = batch_tokenize(tokenizer, batch["sentence2"], device, max_len)

        a1 = t1["attention_mask"].float()
        a2 = t2["attention_mask"].float()

        e1 = encoder.token_embeddings(t1["input_ids"], a1)
        e2 = encoder.token_embeddings(t2["input_ids"], a2)

        s1 = selector.selector(e1 * a1.unsqueeze(-1))
        s2 = selector.selector(e2 * a2.unsqueeze(-1))

        for rho in rhos:
            g1 = probabilistic_top_k(s1, a1, rho, tau)
            g2 = probabilistic_top_k(s2, a2, rho, tau)

            z1 = encoder.pool(e1, a1 * g1)
            z2 = encoder.pool(e2, a2 * g2)

            sims[rho].extend(F.cosine_similarity(z1, z2).cpu().tolist())

        labels.extend(batch["label"])

    return {rho: float(spearmanr(sims[rho], labels)[0]) for rho in rhos}


@torch.no_grad()
def eval_random_sweep(encoder, tokenizer, cfg, device):
    ds = load_dataset("glue", "stsb", split=cfg.eval.split)

    rhos = get_rhos(cfg)
    runs = cfg.eval.random_selector.runs
    bs = cfg.eval.batch_size
    max_len = cfg.eval.max_length

    acc = {rho: [] for rho in rhos}

    for run in range(runs):
        torch.manual_seed(cfg.eval.random_selector.seed + run)

        sims = {rho: [] for rho in rhos}
        labels = []

        for i in tqdm(range(0, len(ds), bs), leave=False):
            batch = ds[i:i + bs]

            t1 = batch_tokenize(tokenizer, batch["sentence1"], device, max_len)
            t2 = batch_tokenize(tokenizer, batch["sentence2"], device, max_len)

            a1 = t1["attention_mask"].float()
            a2 = t2["attention_mask"].float()

            e1 = encoder.token_embeddings(t1["input_ids"], a1)
            e2 = encoder.token_embeddings(t2["input_ids"], a2)

            for rho in rhos:
                g1 = (torch.rand_like(a1) < rho).float() * a1
                g2 = (torch.rand_like(a2) < rho).float() * a2

                z1 = encoder.pool(e1, a1 * g1)
                z2 = encoder.pool(e2, a2 * g2)

                sims[rho].extend(F.cosine_similarity(z1, z2).cpu().tolist())

            labels.extend(batch["label"])

        for rho in rhos:
            acc[rho].append(float(spearmanr(sims[rho], labels)[0]))

    return {rho: sum(v) / len(v) for rho, v in acc.items()}


def load_selector(signature, encoder, tokenizer, device):
    ckpt = os.path.join(PROJECT_ROOT, "outputs", "xps", signature, "model.pth")

    sample = tokenizer("test", return_tensors="pt")
    sample = {k: v.to(device) for k, v in sample.items()}
    emb = encoder.token_embeddings(sample["input_ids"], sample["attention_mask"])

    selector = RationaleSelectorModel(emb.size(-1)).to(device)
    state = torch.load(ckpt, map_location=device)
    selector.load_state_dict(state["model"], strict=False)
    selector.eval()
    selector.sent_encoder = None
    selector.loss_cfg = None

    for p in selector.parameters():
        p.requires_grad_(False)

    return selector


def print_results(base, ours, rand):
    print(f"\nSBERT baseline Spearman ρ: {base:.4f}\n")
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
    plt.plot(rhos, [base]*len(rhos), "--", label="SBERT baseline")
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


def main(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    xp_dir = os.path.join(PROJECT_ROOT, "outputs", "xps", cfg.selector.signature)

    cfg.runtime, _ = configure_runtime(cfg.runtime)
    device = cfg.runtime.device

    encoder, tokenizer = build_sentence_encoder(
        cfg.model.encoder.family,
        cfg.model.encoder.name,
        device,
    )

    base = eval_baseline(encoder, tokenizer, cfg, device)

    selector = load_selector(cfg.selector.signature, encoder, tokenizer, device)

    ours = eval_selector_sweep(encoder, tokenizer, selector, cfg, device)
    rand = eval_random_sweep(encoder, tokenizer, cfg, device)

    print_results(base, ours, rand)

    plot(base, ours, rand, os.path.join(xp_dir, "spearman_vs_rho.png"))

    print("\nSaved plot to:", os.path.join(xp_dir, "spearman_vs_rho.png"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) == 2 else DEFAULT_CFG_PATH)
