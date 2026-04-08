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
from src.metrics import Counts
from src.utils import (
    configure_runtime,
    build_chi_square_payload,
    get_logger,
    selection_rate_matrix_to_table,
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
        "threads": 48,
        "interop_threads": 4,
        "device": "cpu",
        "data": {
            "rebuild": False,
            "test_subset": None,
            "batch_size": 16,
            "num_workers": 0,
        },
    },
    "linearize": {
        "exclude_special": True,
        "max_masks_per_sentence": 1024,
        "seed": 1234,
        "jacobian_vectorize": False,
        "plots_dir": "outputs/utils/linearize",
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

    # Rank transform (average ties not needed for these continuous curves).
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


def save_selection_loss_plot(
    rhos: list[float],
    means: list[float],
    stds: list[float],
    out_path: Path,
) -> None:

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Selected Mask Drift vs Rho")
    ax.set_xlabel("rho")
    ax.set_ylabel("Mean cosine drift")

    x_t = torch.tensor(rhos, dtype=torch.float64)
    y_t = torch.tensor(means, dtype=torch.float64)
    s_t = torch.tensor(stds, dtype=torch.float64)

    ax.plot(x_t.tolist(), y_t.tolist(), marker="o", linewidth=2.0, label="mean selected drift")
    mask = torch.isfinite(y_t) & torch.isfinite(s_t)
    if bool(mask.any().item()):
        xm = x_t[mask]
        ym = y_t[mask]
        sm = s_t[mask]
        ax.fill_between(xm.tolist(), (ym - sm).tolist(), (ym + sm).tolist(), alpha=0.15)

    ax.grid(True, alpha=0.2)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_spearman_plot(rhos: list[float], mean_losses: list[float], out_path: Path) -> float:
    x = torch.tensor(rhos, dtype=torch.float64)
    y = torch.tensor(mean_losses, dtype=torch.float64)
    valid = torch.isfinite(x) & torch.isfinite(y)

    rho_val = float("nan")
    if int(valid.sum().item()) >= 2:
        rho_val = spearman_torch(x[valid], y[valid])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Spearman of Rho vs Selected Drift")
    ax.set_xlabel("rho")
    ax.set_ylabel("Mean cosine drift")
    if bool(valid.any().item()):
        ax.plot(x[valid].tolist(), y[valid].tolist(), marker="o", linewidth=2.0)
    ax.grid(True, alpha=0.2)

    title = f"spearman={rho_val:.4f}" if math.isfinite(rho_val) else "spearman=nan"
    ax.text(0.02, 0.98, title, transform=ax.transAxes, va="top", ha="left")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return rho_val


def save_label_plots(
    counts_pred,
    counts_gold,
    selection_rates: list[float],
    chi_square_data: dict,
    plots_dir: Path,
    logger,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted({label for c in counts_gold for label in c.data.keys()})
    x = torch.tensor(selection_rates, dtype=torch.float64)

    # Plot per-label selection rates across effective rho values.
    selection_rates_path = plots_dir / "selection_rates.png"
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Per-label selection rate vs effective rho")
    ax.set_xlabel("effective rho")
    ax.set_ylabel("selection rate")
    for label in labels:
        ys = []
        for pred, gold in zip(counts_pred, counts_gold):
            tot = float(gold.data.get(label, 0))
            kept = float(pred.data.get(label, 0))
            ys.append((kept / tot) if tot > 0 else 0.0)
        ax.plot(x.tolist(), ys, marker="o", linewidth=1.8, label=str(label))
    ax.grid(True, alpha=0.2)
    if labels:
        ax.legend(fontsize="x-small", ncol=2)
    fig.tight_layout()
    fig.savefig(selection_rates_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved selection rates plot to %s", selection_rates_path)

    # Plot mean chi-square over labels for each effective rho.
    chi_square_path = plots_dir / "chi_square.png"
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Mean chi-square vs effective rho")
    ax.set_xlabel("effective rho")
    ax.set_ylabel("mean chi-square")
    chi_means = []
    rows = chi_square_data.get("rows", []) if isinstance(chi_square_data, dict) else []
    for row in rows:
        label_rows = row.get("labels", [])
        vals = [float(item.get("chi2", 0.0)) for item in label_rows]
        chi_means.append(sum(vals) / len(vals) if vals else 0.0)
    if chi_means:
        ax.plot(x[: len(chi_means)].tolist(), chi_means, marker="o", linewidth=2.0)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(chi_square_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chi-square plot to %s", chi_square_path)


# ---------------------------------------------------------------------------------------


def evaluate_selected_masks_batched(
    ids_b: torch.Tensor,
    full_rep_b: torch.Tensor,
    encoder,
    selected_masks: torch.Tensor,
) -> torch.Tensor:
    # ids_b: [B, L], selected_masks: [B, R, L] -> losses: [B, R]
    if selected_masks.numel() == 0:
        return torch.empty(0, device=ids_b.device, dtype=full_rep_b.dtype)

    bsz, n_rhos, seq_len = selected_masks.shape
    with torch.inference_mode():
        ids_rep = ids_b.unsqueeze(1).expand(bsz, n_rhos, seq_len).reshape(bsz * n_rhos, seq_len)
        eff_attn = selected_masks.reshape(bsz * n_rhos, seq_len)
        tok = encoder.token_embeddings(ids_rep, eff_attn)
        pred = encoder.pool(tok, eff_attn)
        full_rep_rep = full_rep_b.unsqueeze(1).expand(bsz, n_rhos, full_rep_b.size(-1)).reshape(bsz * n_rhos, -1)
        losses = 1.0 - F.cosine_similarity(pred, full_rep_rep, dim=-1)
    return losses.view(bsz, n_rhos)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exploratory linearized mask utility: score random masks from one baseline forward pass, then measure the real drift of the selected mask."
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--subset", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--encoder-family", type=str, default=None)
    parser.add_argument("--encoder-name", type=str, default=None)
    parser.add_argument("--keep-special", action="store_true")
    parser.add_argument("--drop-special", action="store_true")
    parser.add_argument("--max-masks-per-sentence", type=int, default=None, help="Random masks tested per sentence/rho.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for random mask fallback.")
    parser.add_argument("--jacobian-vectorize", action="store_true", help="Use vectorized Jacobian (faster but less interrupt-friendly).")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf dotlist overrides, e.g. data.dataset=wikiann runtime.data.batch_size=64",
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
    logger.info("Exploratory linearized mask utility run (no Dora experiment)")
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
    del train_dl, tokenizer, ds

    device = torch.device(cfg.runtime.device)
    encoder = encoder.to(device)
    encoder.eval()

    rhos = linspace_rhos(cfg)
    n_rhos = len(rhos)
    real_sum = torch.zeros(n_rhos, dtype=torch.float64)
    real_sumsq = torch.zeros(n_rhos, dtype=torch.float64)
    real_count = torch.zeros(n_rhos, dtype=torch.float64)
    approx_sum = torch.zeros(n_rhos, dtype=torch.float64)
    rho_eff_sum = torch.zeros(n_rhos, dtype=torch.float64)

    counts_pred = [Counts(labels=labels_set) for _ in rhos] if labels_set is not None else None
    counts_gold = [Counts(labels=labels_set) for _ in rhos] if labels_set is not None else None

    exclude_special = bool(cfg.linearize.get("exclude_special", True))
    max_masks_per_sentence = max(1, int(cfg.linearize.get("max_masks_per_sentence", 4096)))
    seed = int(cfg.linearize.get("seed", 1234))
    jacobian_vectorize = bool(cfg.linearize.get("jacobian_vectorize", False))
    rhos_t = torch.tensor(rhos, device=device, dtype=torch.float32)
    rho_idx = torch.arange(n_rhos, device=device)

    evaluated = 0

    def _safe_name(value: object) -> str:
        return str(value).replace("/", "_").replace(" ", "_")

    plots_root = Path(str(cfg.linearize.get("plots_dir", "outputs/utils/linearize")))
    if not plots_root.is_absolute():
        plots_root = Path.cwd() / plots_root
    run_dir = plots_root / "__".join(
        [
            _safe_name(cfg.data.dataset),
            _safe_name(cfg.data.encoder.family),
            _safe_name(cfg.data.encoder.name or "default"),
            f"seed={seed}",
        ]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Artifacts will be written to %s", run_dir)

    pbar = tqdm(test_dl, desc="Dataset batches", dynamic_ncols=True, unit="batch", position=0)

    for batch in pbar:
        ids_b = batch["ids"].to(device)
        attn_b = batch["attn_mask"].to(device)
        labels_b = batch.get("labels", None)

        bs = ids_b.size(0)

        seq_len = int(ids_b.size(1))
        attn_f_b = attn_b.float()
        with torch.inference_mode():
            full_token_emb_b = encoder.token_embeddings(ids_b, attn_b)
            full_rep_b = encoder.pool(full_token_emb_b, attn_f_b)

        valid_start, valid_end_b, n_valid_b = compute_valid_span(attn_b, exclude_special)
        valid_sentence_mask = n_valid_b > 0
        if not bool(valid_sentence_mask.any().item()):
            continue

        k_br = torch.round(n_valid_b.to(torch.float32).unsqueeze(1) * rhos_t.unsqueeze(0)).to(torch.long)
        k_br = torch.minimum(k_br, n_valid_b.unsqueeze(1))
        k_br = torch.where(n_valid_b.unsqueeze(1) > 0, torch.clamp(k_br, min=1), torch.zeros_like(k_br))
        selected_masks = torch.zeros((bs, n_rhos, seq_len), device=device, dtype=attn_f_b.dtype)

        valid_idx = torch.nonzero(valid_sentence_mask, as_tuple=False).squeeze(1)
        for bi in valid_idx.tolist():
            ids_i = ids_b[bi : bi + 1]
            attn_i = attn_b[bi]

            _, orth_component = compute_sentence_linearization_basis(
                ids_i,
                attn_i,
                encoder,
                vectorize_jacobian=jacobian_vectorize,
            )

            attn_i_float = attn_i.float()
            valid_end_i = int(valid_end_b[bi].item())
            k_r = k_br[bi]
            candidate_masks_rm = build_random_candidate_masks_per_rho(
                attn_i_float,
                valid_start,
                valid_end_i,
                k_r,
                max_masks_per_sentence,
                seed + evaluated * 1000 + bi,
            )

            if candidate_masks_rm.numel() == 0:
                continue

            delta = candidate_masks_rm.to(dtype=torch.float32) - attn_i_float.view(1, 1, -1)
            approx_delta = delta @ orth_component.T  # [n_rhos, max_masks, d]
            approx_scores = torch.linalg.vector_norm(approx_delta, dim=2)
            best_idx = torch.argmin(approx_scores, dim=1)

            selected_masks[bi] = candidate_masks_rm[rho_idx, best_idx]

            best_approx = approx_scores[rho_idx, best_idx].to(torch.float64).cpu()
            approx_sum += best_approx

        selected_masks = selected_masks * valid_sentence_mask.view(bs, 1, 1).to(attn_f_b.dtype)

        selected_real_losses = evaluate_selected_masks_batched(
            ids_b,
            full_rep_b,
            encoder,
            selected_masks,
        )

        selected_valid = selected_masks[valid_idx]
        losses_valid = selected_real_losses[valid_idx]
        L_eff_valid = attn_b[valid_idx].sum(dim=1).to(torch.float32).clamp(min=1.0)

        losses_valid_cpu = losses_valid.to(torch.float64).cpu()
        real_sum += losses_valid_cpu.sum(dim=0)
        real_sumsq += losses_valid_cpu.square().sum(dim=0)
        real_count += float(losses_valid_cpu.size(0))

        rho_eff_batch = (selected_valid.sum(dim=2).to(torch.float32) / L_eff_valid.unsqueeze(1)).sum(dim=0)
        rho_eff_sum += rho_eff_batch.to(torch.float64).cpu()

        if counts_pred is not None and counts_gold is not None and labels_b is not None:
            for bi in valid_idx.tolist():
                labels_i = labels_b[bi]
                flat_attn = attn_b[bi].bool().cpu()
                gold_counts_i = Counts(labels_i, flat_attn)
                for ridx in range(n_rhos):
                    flat_preds = selected_masks[bi, ridx].bool().cpu()
                    counts_pred[ridx] += Counts(labels_i, flat_attn, flat_preds)
                    counts_gold[ridx] += gold_counts_i

        evaluated += int(valid_sentence_mask.sum().item())

    if evaluated == 0:
        raise RuntimeError(
            "No samples evaluated. Try lowering max_length/subset, or increase --max-masks-per-sentence."
        )

    selection_rates = (rho_eff_sum / float(evaluated)).tolist()

    mean_losses_t = torch.where(real_count > 0, real_sum / real_count, torch.full_like(real_sum, float("nan")))
    var_losses_t = torch.where(
        real_count > 0,
        real_sumsq / real_count - mean_losses_t.square(),
        torch.full_like(real_sum, float("nan")),
    )
    std_losses_t = torch.sqrt(torch.clamp(var_losses_t, min=0.0))

    mean_losses = mean_losses_t.tolist()
    std_losses = std_losses_t.tolist()

    loss_path = run_dir / "selected_mask_drift.png"
    save_selection_loss_plot(rhos, mean_losses, std_losses, loss_path)
    logger.info("Saved selection loss plot to %s", loss_path)

    spearman_path = run_dir / "spearman_vs_rho.png"
    rho_s = save_spearman_plot(rhos, mean_losses, spearman_path)
    logger.info("Saved Spearman plot to %s (rho=%.4f)", spearman_path, rho_s)

    finite_mean = mean_losses_t[torch.isfinite(mean_losses_t)]
    eval_loss = float(finite_mean.mean().item()) if int(finite_mean.numel()) > 0 else float("nan")

    approx_means_t = approx_sum / evaluated if evaluated > 0 else torch.full_like(approx_sum, float("nan"))
    approx_means = approx_means_t.tolist()

    summary = {
        "dataset": str(cfg.data.dataset),
        "encoder_family": str(cfg.data.encoder.family),
        "encoder_name": None if cfg.data.encoder.name in {None, "None", "null", "NULL"} else str(cfg.data.encoder.name),
        "subset": float(cfg.data.subset),
        "seed": int(seed),
        "max_masks_per_sentence": int(max_masks_per_sentence),
        "exclude_special": bool(exclude_special),
        "evaluated_sentences": int(evaluated),
        "selection_rates": [float(rate) for rate in selection_rates],
        "eval_loss": float(eval_loss),
        "eval_loss_by_rho": {f"{rho:.6f}": float(loss) for rho, loss in zip(rhos, mean_losses)},
        "eval_loss_std_by_rho": {f"{rho:.6f}": float(std) for rho, std in zip(rhos, std_losses)},
        "linearized_loss_by_rho": {f"{rho:.6f}": float(loss) for rho, loss in zip(rhos, approx_means)},
        "loss_gap_by_rho": {
            f"{rho:.6f}": (float(loss) - float(approx)) if math.isfinite(loss) and math.isfinite(approx) else float("nan")
            for rho, loss, approx in zip(rhos, mean_losses, approx_means)
        },
        "spearman_rho_vs_loss": float(rho_s),
        "artifacts_dir": str(run_dir),
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved summary to %s", summary_path)

    if counts_pred is not None and counts_gold is not None:
        selections_path = run_dir / "selections.json"
        selections_data = {
            "selection_rates": [float(r) for r in selection_rates],
            "selections_by_rho": [
                {
                    "rho": float(rate),
                    "pred_counts": dict(pred.data),
                    "gold_counts": dict(gold.data),
                }
                for rate, pred, gold in zip(selection_rates, counts_pred, counts_gold)
            ],
        }
        selections_path.write_text(json.dumps(selections_data, indent=2), encoding="utf-8")
        logger.info("Saved selection counts to %s", selections_path)

        chi_square_path = run_dir / "chi_square.json"
        chi_square_data = build_chi_square_payload(counts_pred, counts_gold, selection_rates)
        chi_square_path.write_text(json.dumps(chi_square_data, indent=2), encoding="utf-8")
        logger.info("Saved chi-square data to %s", chi_square_path)

        logger.info("Configured rho grid: %s", ", ".join(f"{rho:.3f}" for rho in rhos))
        logger.info("Effective selected-token rates: %s", ", ".join(f"{rate:.3f}" for rate in selection_rates))
        logger.info(
            "\nLabel selection matrix (selected masks):\n%s",
            selection_rate_matrix_to_table(counts_pred, counts_gold, selection_rates),
        )
        save_label_plots(counts_pred, counts_gold, selection_rates, chi_square_data, run_dir, logger)

    logger.info("Evaluated samples: %d", evaluated)
    logger.info("Done.")


if __name__ == "__main__":
    main()
