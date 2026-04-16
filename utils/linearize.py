from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import initialize_data
from src.utils import (
    configure_runtime,
    get_logger,
)


torch.set_printoptions(precision=6, sci_mode=False)

# Self-contained defaults for this utility script (no external config file required).
DEFAULT_CFG = {
    "model": {
        "keep_special": True,
        "loss": {"sweep_range": [0.1, 1.0, 10]},
    },
    "data": {
        "shuffle": False,
        "max_length": 512,
        "dataset": "wikiann",
        "config": None,
        "subset": 1.0,
        "encoder": {"family": "sbert", "name": None},
    },
    "runtime": {
        "threads": 24,
        "interop_threads": 1,
        "device": "cpu",
        "data": {
            "rebuild": False,
            "test_subset": None,
            "batch_size": 64,
            "num_workers": 0,
        },
    },
    "linearize": {
        "exclude_special": True,
        "max_masks_per_sentence": 64,
        "seed": 1234,
        "jacobian_vectorize": False,
        "plots_dir": "outputs/linearize",
        "eval_chunk": 256,
    },
}


def load_cfg(overrides: list[str]) -> DictConfig:
    cfg = OmegaConf.create(DEFAULT_CFG)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(cfg).__name__}")
    return cfg


def linspace_rhos(cfg: DictConfig) -> list[float]:
    sweep = cfg.model.loss.sweep_range
    start = float(sweep[0])
    end = float(sweep[1])
    steps = int(sweep[2])
    return torch.linspace(start, end, steps=steps, dtype=torch.float64).tolist()


def spearman_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() != y.numel() or x.numel() < 2:
        return float("nan")

    rx = torch.argsort(torch.argsort(x)).to(torch.float64)
    ry = torch.argsort(torch.argsort(y)).to(torch.float64)

    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    if float(denom.item()) == 0.0:
        return float("nan")
    return float((rx @ ry / denom).item())


def compute_valid_span(attn: torch.Tensor, exclude_special: bool) -> tuple[int, torch.Tensor, torch.Tensor]:
    valid_start = 1 if exclude_special else 0
    valid_end_b = attn.sum(dim=1).to(torch.long)
    if exclude_special:
        valid_end_b = torch.clamp(valid_end_b - 1, min=valid_start)
    n_valid_b = valid_end_b - valid_start
    return valid_start, valid_end_b, n_valid_b


def build_random_candidate_masks_per_rho(
    valid_mask: torch.Tensor,
    valid_start: int,
    valid_end: int,
    k_r: torch.Tensor,
    max_masks: int,
    seed: int,
) -> torch.Tensor:
    seq_len = int(valid_mask.numel())
    device = valid_mask.device
    k_r = k_r.to(dtype=torch.long, device=device)
    n_rhos = int(k_r.numel())

    valid_positions = torch.arange(seq_len, device=device)
    valid_positions = valid_positions[(valid_positions >= valid_start) & (valid_positions < valid_end)]

    if max_masks <= 0 or valid_positions.numel() == 0 or n_rhos == 0:
        return torch.zeros((n_rhos, 0, seq_len), device=device, dtype=valid_mask.dtype)

    k_r = torch.clamp(k_r, min=0, max=int(valid_positions.numel()))
    max_k = int(k_r.max().item())
    if max_k <= 0:
        return torch.zeros((n_rhos, max_masks, seq_len), device=device, dtype=valid_mask.dtype)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    rand_scores = torch.rand((n_rhos, max_masks, valid_positions.numel()), device=device, generator=generator)
    top_idx = torch.topk(rand_scores, k=max_k, dim=2, largest=True, sorted=False).indices

    keep = (
        torch.arange(max_k, device=device).view(1, 1, max_k)
        < k_r.view(n_rhos, 1, 1)
    ).to(valid_mask.dtype).expand(n_rhos, max_masks, max_k)

    masks = torch.zeros((n_rhos, max_masks, seq_len), device=device, dtype=valid_mask.dtype)
    chosen_positions = valid_positions[top_idx]
    masks.scatter_(2, chosen_positions, keep)
    return masks


def compute_sentence_linearization_basis(
    ids_i: torch.Tensor,
    attn_i: torch.Tensor,
    encoder,
    vectorize_jacobian: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask0 = attn_i.float().detach().clone().requires_grad_(True)

    def encode(mask_1d: torch.Tensor) -> torch.Tensor:
        mask_b = mask_1d.unsqueeze(0)
        token_emb = encoder.token_embeddings(ids_i, mask_b)
        sent = encoder.pool(token_emb, mask_b)
        return F.normalize(sent.squeeze(0), dim=-1)

    try:
        jac = torch.autograd.functional.jacobian(encode, mask0, vectorize=vectorize_jacobian)
    except TypeError:
        jac = torch.autograd.functional.jacobian(encode, mask0)

    baseline_hat = encode(mask0).detach()
    orth_component = jac - baseline_hat[:, None] * torch.matmul(baseline_hat, jac).unsqueeze(0)
    return baseline_hat, orth_component


def compute_all_candidate_losses(
    ids_i: torch.Tensor,
    full_rep_i: torch.Tensor,
    encoder,
    candidate_masks: torch.Tensor,
    chunk: int = 256,
) -> torch.Tensor:
    """Compute actual reconstruction loss for every candidate mask.

    Args:
        ids_i:           [1, L] token ids for one sentence
        full_rep_i:      [d] full sentence pooled representation
        encoder:         SentenceEncoder
        candidate_masks: [n_rhos, max_masks, L]
        chunk:           max encoder batch size to avoid OOM

    Returns:
        losses: [n_rhos, max_masks] float64
    """
    n_rhos, max_masks, seq_len = candidate_masks.shape
    total = n_rhos * max_masks
    ids_rep = ids_i.expand(total, seq_len)
    flat_masks = candidate_masks.reshape(total, seq_len)
    full_rep_rep = full_rep_i.unsqueeze(0).expand(total, -1)

    losses = torch.empty(total, dtype=torch.float64, device=ids_i.device)
    with torch.inference_mode():
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            tok = encoder.token_embeddings(ids_rep[start:end], flat_masks[start:end])
            pred = encoder.pool(tok, flat_masks[start:end])
            losses[start:end] = (
                1.0 - F.cosine_similarity(pred, full_rep_rep[start:end], dim=-1)
            ).to(torch.float64)
    return losses.view(n_rhos, max_masks)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def save_spearman_ranking_plot(
    rhos: list[float],
    means: list[float],
    stds: list[float],
    out_path: Path,
) -> None:
    """Plot mean Spearman(approx_ranking, actual_ranking) per rho."""
    x = torch.tensor(rhos, dtype=torch.float64)
    y = torch.tensor(means, dtype=torch.float64)
    s = torch.tensor(stds, dtype=torch.float64)
    valid = torch.isfinite(x) & torch.isfinite(y)

    overall = float("nan")
    if int(valid.sum().item()) >= 1:
        overall = float(y[valid].mean().item())

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Ranking Agreement: Spearman(approx, actual) vs Rho")
    ax.set_xlabel("rho")
    ax.set_ylabel("Spearman correlation")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")

    if bool(valid.any().item()):
        xv, yv, sv = x[valid], y[valid], s[valid]
        ax.plot(xv.tolist(), yv.tolist(), marker="o", linewidth=2.0, label="mean Spearman")
        band = torch.isfinite(sv)
        if bool(band.any().item()):
            ax.fill_between(
                xv[band].tolist(),
                (yv[band] - sv[band]).tolist(),
                (yv[band] + sv[band]).tolist(),
                alpha=0.15,
            )

    ax.grid(True, alpha=0.2)
    ax.legend(fontsize="small")
    label = f"mean={overall:.4f}" if math.isfinite(overall) else "mean=nan"
    ax.text(0.02, 0.98, label, transform=ax.transAxes, va="top", ha="left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_loss_comparison_plot(
    rhos: list[float],
    linear_means: list[float],
    linear_stds: list[float],
    oracle_means: list[float],
    oracle_stds: list[float],
    random_means: list[float],
    random_stds: list[float],
    out_path: Path,
) -> None:
    """Plot actual reconstruction loss for linearized / oracle / random vs rho.

    Random baseline = mean loss over all candidate masks (no selection).
    Oracle          = min loss over all candidates (best possible from the pool).
    Linearized      = loss of the mask chosen by min approx score.
    """
    x = torch.tensor(rhos, dtype=torch.float64)
    curves = [
        (random_means,  random_stds,  "random (mean)",       "tab:gray"),
        (linear_means,  linear_stds,  "linearized selection", "tab:blue"),
        (oracle_means,  oracle_stds,  "oracle (best)",        "tab:green"),
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Reconstruction Loss: Linearized vs Oracle vs Random")
    ax.set_xlabel("rho")
    ax.set_ylabel("1 − cos_sim  (reconstruction loss)")

    for means, stds, label, color in curves:
        y = torch.tensor(means, dtype=torch.float64)
        s = torch.tensor(stds, dtype=torch.float64)
        valid = torch.isfinite(x) & torch.isfinite(y)
        if not bool(valid.any().item()):
            continue
        xv, yv, sv = x[valid], y[valid], s[valid]
        ax.plot(xv.tolist(), yv.tolist(), marker="o", linewidth=2.0, label=label, color=color)
        band = torch.isfinite(sv)
        if bool(band.any().item()):
            ax.fill_between(
                xv[band].tolist(),
                (yv[band] - sv[band]).tolist(),
                (yv[band] + sv[band]).tolist(),
                alpha=0.12,
                color=color,
            )

    ax.grid(True, alpha=0.2)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Linearized mask analysis: "
            "(1) Spearman between approx and actual mask rankings per rho, "
            "(2) reconstruction-loss comparison of linearized / oracle / random selection."
        )
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--subset", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--encoder-family", type=str, default=None)
    parser.add_argument("--encoder-name", type=str, default=None)
    parser.add_argument("--keep-special", action="store_true")
    parser.add_argument("--drop-special", action="store_true")
    parser.add_argument("--max-masks-per-sentence", type=int, default=None,
                        help="Candidate masks per sentence/rho (default 64; more = better Spearman but slower).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--jacobian-vectorize", action="store_true",
                        help="Use vectorized Jacobian (faster but less interrupt-friendly).")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf dotlist overrides, e.g. data.dataset=wikiann",
    )
    args = parser.parse_args()

    override_list = list(args.overrides)
    if args.dataset is not None:
        override_list.append(f"data.dataset={args.dataset}")
    if args.subset is not None:
        override_list.append(f"data.subset={args.subset}")
    if args.batch_size is not None:
        override_list.append(f"runtime.data.batch_size={args.batch_size}")
    if args.device is not None:
        override_list.append(f"runtime.device={args.device}")
    if args.encoder_family is not None:
        override_list.append(f"data.encoder.family={args.encoder_family}")
    if args.encoder_name is not None:
        override_list.append(f"data.encoder.name={args.encoder_name}")
    if args.max_masks_per_sentence is not None:
        override_list.append(f"linearize.max_masks_per_sentence={args.max_masks_per_sentence}")
    if args.seed is not None:
        override_list.append(f"linearize.seed={args.seed}")
    if args.jacobian_vectorize:
        override_list.append("linearize.jacobian_vectorize=true")
    if args.keep_special and args.drop_special:
        raise ValueError("Use only one of --keep-special or --drop-special.")
    if args.keep_special:
        override_list.append("model.keep_special=true")
    if args.drop_special:
        override_list.append("model.keep_special=false")

    cfg = load_cfg(override_list)

    logger = get_logger(logfile="linearize.log")
    logger.info("Linearized mask analysis run")
    logger.info(repr(cfg))
    logger.info("Work dir: %s", Path.cwd())

    cfg.runtime, changed_device = configure_runtime(cfg.runtime)
    if changed_device:
        logger.warning("CUDA requested but unavailable, using CPU.")

    train_dl, test_dl, encoder, tokenizer, labels_set, ds = initialize_data(
        cfg.data,
        cfg.runtime.data,
        logger,
        device=cfg.runtime.device,
        keep_special=bool(cfg.model.get("keep_special", True)),
    )
    del train_dl, tokenizer, labels_set, ds

    device = torch.device(cfg.runtime.device)
    encoder = encoder.to(device)
    encoder.eval()

    rhos = linspace_rhos(cfg)
    n_rhos = len(rhos)
    rho_idx = torch.arange(n_rhos, device=device)
    rhos_t = torch.tensor(rhos, device=device, dtype=torch.float32)

    exclude_special = bool(cfg.linearize.get("exclude_special", True))
    max_masks_per_sentence = max(1, int(cfg.linearize.get("max_masks_per_sentence", 64)))
    seed = int(cfg.linearize.get("seed", 1234))
    jacobian_vectorize = bool(cfg.linearize.get("jacobian_vectorize", False))
    eval_chunk = int(cfg.linearize.get("eval_chunk", 256))

    # Per-rho accumulators ------------------------------------------------
    # Task 1: Spearman between approx and actual rankings
    spearman_sum   = torch.zeros(n_rhos, dtype=torch.float64)
    spearman_sq    = torch.zeros(n_rhos, dtype=torch.float64)
    spearman_n     = torch.zeros(n_rhos, dtype=torch.float64)

    # Task 2: reconstruction loss (actual) for each selection strategy
    linear_sum     = torch.zeros(n_rhos, dtype=torch.float64)
    linear_sumsq   = torch.zeros(n_rhos, dtype=torch.float64)
    oracle_sum     = torch.zeros(n_rhos, dtype=torch.float64)
    oracle_sumsq   = torch.zeros(n_rhos, dtype=torch.float64)
    random_sum     = torch.zeros(n_rhos, dtype=torch.float64)
    random_sumsq   = torch.zeros(n_rhos, dtype=torch.float64)
    count          = torch.zeros(n_rhos, dtype=torch.float64)
    # ---------------------------------------------------------------------

    plots_root = Path(str(cfg.linearize.get("plots_dir", "outputs/linearize")))
    if not plots_root.is_absolute():
        plots_root = Path.cwd() / plots_root
    plots_root.mkdir(parents=True, exist_ok=True)
    logger.info("Artifacts will be written to %s", plots_root)

    evaluated = 0
    pbar = tqdm(test_dl, desc="Dataset batches", dynamic_ncols=True, unit="batch", position=0)

    for batch in pbar:
        ids_b  = batch["ids"].to(device)
        attn_b = batch["attn_mask"].to(device)

        attn_f_b = attn_b.float()
        with torch.inference_mode():
            full_token_emb_b = encoder.token_embeddings(ids_b, attn_b)
            full_rep_b       = encoder.pool(full_token_emb_b, attn_f_b)

        valid_start, valid_end_b, n_valid_b = compute_valid_span(attn_b, exclude_special)
        valid_sentence_mask = n_valid_b > 0
        if not bool(valid_sentence_mask.any().item()):
            continue

        k_br = torch.round(n_valid_b.float().unsqueeze(1) * rhos_t.unsqueeze(0)).long()
        k_br = torch.minimum(k_br, n_valid_b.unsqueeze(1))
        k_br = torch.where(n_valid_b.unsqueeze(1) > 0, k_br.clamp(min=1), torch.zeros_like(k_br))

        valid_idx = torch.nonzero(valid_sentence_mask, as_tuple=False).squeeze(1)

        for bi in valid_idx.tolist():
            ids_i      = ids_b[bi: bi + 1]
            attn_i     = attn_b[bi]
            full_rep_i = full_rep_b[bi]

            _, orth_component = compute_sentence_linearization_basis(
                ids_i, attn_i, encoder, vectorize_jacobian=jacobian_vectorize,
            )

            attn_i_float = attn_i.float()
            valid_end_i  = int(valid_end_b[bi].item())
            k_r          = k_br[bi]

            candidate_masks_rm = build_random_candidate_masks_per_rho(
                attn_i_float, valid_start, valid_end_i, k_r,
                max_masks_per_sentence, seed + evaluated * 1000 + bi,
            )
            if candidate_masks_rm.numel() == 0:
                continue

            # Linearized (approx) scores [n_rhos, max_masks]
            delta         = candidate_masks_rm.float() - attn_i_float.view(1, 1, -1)
            approx_delta  = delta @ orth_component.T
            approx_scores = torch.linalg.vector_norm(approx_delta, dim=2)

            # Actual reconstruction losses for ALL candidates [n_rhos, max_masks]
            actual_losses = compute_all_candidate_losses(
                ids_i, full_rep_i, encoder, candidate_masks_rm, chunk=eval_chunk,
            )  # float64, on device

            # Task 2: per-strategy losses
            best_approx_idx = torch.argmin(approx_scores, dim=1)          # [n_rhos]
            linear_losses   = actual_losses[rho_idx, best_approx_idx]     # [n_rhos] actual loss of linearized pick
            oracle_losses   = actual_losses.min(dim=1).values              # [n_rhos] best achievable from pool
            random_losses   = actual_losses.mean(dim=1)                    # [n_rhos] expected loss of a random pick

            linear_cpu = linear_losses.cpu()
            oracle_cpu = oracle_losses.cpu()
            random_cpu = random_losses.cpu()

            linear_sum   += linear_cpu;  linear_sumsq  += linear_cpu.square()
            oracle_sum   += oracle_cpu;  oracle_sumsq  += oracle_cpu.square()
            random_sum   += random_cpu;  random_sumsq  += random_cpu.square()
            count        += 1.0

            # Task 1: Spearman(approx_ranking, actual_ranking) per rho
            for r in range(n_rhos):
                rho_sp = spearman_torch(
                    approx_scores[r].to(torch.float64).cpu(),
                    actual_losses[r].cpu(),
                )
                if math.isfinite(rho_sp):
                    spearman_sum[r] += rho_sp
                    spearman_sq[r]  += rho_sp * rho_sp
                    spearman_n[r]   += 1.0

        evaluated += int(valid_sentence_mask.sum().item())

    if evaluated == 0:
        raise RuntimeError("No samples evaluated. Try increasing --subset or --max-masks-per-sentence.")

    def _mean_std(s, ssq, n):
        mean = torch.where(n > 0, s / n, torch.full_like(s, float("nan")))
        var  = torch.where(n > 0, ssq / n - mean.square(), torch.full_like(s, float("nan")))
        std  = torch.sqrt(torch.clamp(var, min=0.0))
        return mean.tolist(), std.tolist()

    spearman_means, spearman_stds = _mean_std(spearman_sum, spearman_sq,   spearman_n)
    linear_means,   linear_stds   = _mean_std(linear_sum,   linear_sumsq,  count)
    oracle_means,   oracle_stds   = _mean_std(oracle_sum,   oracle_sumsq,  count)
    random_means,   random_stds   = _mean_std(random_sum,   random_sumsq,  count)

    # Plot 1: Spearman ranking agreement vs rho
    spearman_path = plots_root / "spearman_ranking_vs_rho.png"
    save_spearman_ranking_plot(rhos, spearman_means, spearman_stds, spearman_path)
    logger.info("Saved Spearman ranking plot to %s", spearman_path)

    # Plot 2: loss comparison vs rho
    comparison_path = plots_root / "loss_comparison_vs_rho.png"
    save_loss_comparison_plot(
        rhos,
        linear_means, linear_stds,
        oracle_means,  oracle_stds,
        random_means,  random_stds,
        comparison_path,
    )
    logger.info("Saved loss comparison plot to %s", comparison_path)

    # Summary JSON
    summary = {
        "dataset":                str(cfg.data.dataset),
        "encoder_family":         str(cfg.data.encoder.family),
        "encoder_name":           (
            None if cfg.data.encoder.name in {None, "None", "null", "NULL"}
            else str(cfg.data.encoder.name)
        ),
        "subset":                 float(cfg.data.subset),
        "seed":                   int(seed),
        "max_masks_per_sentence": int(max_masks_per_sentence),
        "exclude_special":        bool(exclude_special),
        "evaluated_sentences":    int(evaluated),
        "spearman_ranking_by_rho":     {f"{rho:.6f}": float(m) for rho, m in zip(rhos, spearman_means)},
        "spearman_ranking_std_by_rho": {f"{rho:.6f}": float(s) for rho, s in zip(rhos, spearman_stds)},
        "linear_loss_by_rho":    {f"{rho:.6f}": float(m) for rho, m in zip(rhos, linear_means)},
        "oracle_loss_by_rho":    {f"{rho:.6f}": float(m) for rho, m in zip(rhos, oracle_means)},
        "random_loss_by_rho":    {f"{rho:.6f}": float(m) for rho, m in zip(rhos, random_means)},
    }
    summary_path = plots_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved summary to %s", summary_path)
    logger.info("Evaluated samples: %d", evaluated)
    logger.info("Done.")


if __name__ == "__main__":
    main()
