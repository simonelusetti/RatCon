from __future__ import annotations

import argparse
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
    get_logger,
    save_label_plots,
    selection_rate_matrix_to_table,
)


torch.set_printoptions(precision=6, sci_mode=False)

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>", "-100"}
MASK_EVAL_BATCH_SIZE = 256

# Self-contained defaults for this utility script (no external config file required).
DEFAULT_CFG = {
    "model": {
        "keep_special": True,
        "loss": {"sweep_range": [0.1, 1.0, 10]},
    },
    "data": {
        "shuffle": False,
        "max_length": 512,
        "dataset": "mr",
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
            "num_workers": 2,
            "dynamic_batch": {
                "enabled": False,
                "min_batch_size": 16,
                "reduce_factor": 0.5,
                "min_available_ratio": 0.10,
                "max_swap_growth_mb": 128,
            },
        },
    },
    "linearize": {
        "exclude_special": True,
        "max_masks_per_sentence": 1024,
        "seed": 1234,
        "plots_dir": "plots",
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


def save_oracle_loss_plot(rhos: list[float], losses_by_rho: dict[float, list[float]], out_path: Path) -> tuple[list[float], list[float]]:
    means, stds = [], []
    for rho in rhos:
        vals = losses_by_rho.get(rho, [])
        if vals:
            v = torch.tensor(vals, dtype=torch.float64)
            means.append(float(v.mean().item()))
            stds.append(float(v.std(unbiased=False).item()))
        else:
            means.append(float("nan"))
            stds.append(float("nan"))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Oracle Minimum Drift vs Rho")
    ax.set_xlabel("rho")
    ax.set_ylabel("Min cosine drift")

    x_t = torch.tensor(rhos, dtype=torch.float64)
    y_t = torch.tensor(means, dtype=torch.float64)
    s_t = torch.tensor(stds, dtype=torch.float64)

    ax.plot(x_t.tolist(), y_t.tolist(), marker="o", linewidth=2.0, label="mean min drift")
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

    return means, stds


def save_spearman_plot(rhos: list[float], mean_losses: list[float], out_path: Path) -> float:
    x = torch.tensor(rhos, dtype=torch.float64)
    y = torch.tensor(mean_losses, dtype=torch.float64)
    valid = torch.isfinite(x) & torch.isfinite(y)

    rho_val = float("nan")
    if int(valid.sum().item()) >= 2:
        rho_val = spearman_torch(x[valid], y[valid])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Spearman of Rho vs Oracle Drift")
    ax.set_xlabel("rho")
    ax.set_ylabel("Mean min cosine drift")
    if bool(valid.any().item()):
        ax.plot(x[valid].tolist(), y[valid].tolist(), marker="o", linewidth=2.0)
    ax.grid(True, alpha=0.2)

    title = f"spearman={rho_val:.4f}" if math.isfinite(rho_val) else "spearman=nan"
    ax.text(0.02, 0.98, title, transform=ax.transAxes, va="top", ha="left")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return rho_val


# ---------------------------------------------------------------------------------------


def evaluate_selected_masks_batched(
    ids_b: torch.Tensor,
    attn_f_b: torch.Tensor,
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
        eff_attn = (attn_f_b.unsqueeze(1) * selected_masks).reshape(bsz * n_rhos, seq_len)
        tok = encoder.token_embeddings(ids_rep, eff_attn)
        pred = encoder.pool(tok, eff_attn)
        full_rep_rep = full_rep_b.unsqueeze(1).expand(bsz, n_rhos, full_rep_b.size(-1)).reshape(bsz * n_rhos, -1)
        losses = 1.0 - F.cosine_similarity(pred, full_rep_rep, dim=-1)
    return losses.view(bsz, n_rhos)


def evaluate_linearized_masks_batched(
    attn_f_b: torch.Tensor,
    full_token_emb_b: torch.Tensor,
    full_rep_b: torch.Tensor,
    encoder,
    all_masks_4d: torch.Tensor,
) -> torch.Tensor:
    # all_masks_4d: [B, R, M, L] -> losses: [B, R, M]
    if all_masks_4d.numel() == 0:
        return torch.empty(0, device=attn_f_b.device, dtype=full_rep_b.dtype)

    bsz, n_rhos, n_masks, seq_len = all_masks_4d.shape
    hidden = full_token_emb_b.size(-1)
    with torch.inference_mode():
        eff_attn = (attn_f_b.unsqueeze(1).unsqueeze(1) * all_masks_4d).reshape(bsz * n_rhos * n_masks, seq_len)
        tok = (
            full_token_emb_b.unsqueeze(1)
            .unsqueeze(1)
            .expand(bsz, n_rhos, n_masks, seq_len, hidden)
            .reshape(bsz * n_rhos * n_masks, seq_len, hidden)
        )
        pred = encoder.pool(tok, eff_attn)
        full_rep_rep = (
            full_rep_b.unsqueeze(1)
            .unsqueeze(1)
            .expand(bsz, n_rhos, n_masks, full_rep_b.size(-1))
            .reshape(bsz * n_rhos * n_masks, -1)
        )
        losses = 1.0 - F.cosine_similarity(pred, full_rep_rep, dim=-1)
    return losses.view(bsz, n_rhos, n_masks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random-mask oracle utility (no training, no checkpoints, no Dora runs)."
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
    if args.keep_special and args.drop_special:
        raise ValueError("Use only one of --keep-special or --drop-special.")
    if args.keep_special:
        override_list.append("model.keep_special=true")
    if args.drop_special:
        override_list.append("model.keep_special=false")

    cfg = load_cfg(override_list)

    logger = get_logger(logfile="linearize.log")
    logger.info("Oracle linearize utility run (no Dora experiment)")
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
    losses_by_rho: dict[float, list[float]] = {rho: [] for rho in rhos}
    rho_eff_sum = [0.0 for _ in rhos]

    counts_pred = [Counts(labels=labels_set) for _ in rhos] if labels_set is not None else None
    counts_gold = [Counts(labels=labels_set) for _ in rhos] if labels_set is not None else None

    exclude_special = bool(cfg.linearize.get("exclude_special", True))
    max_masks_per_sentence = int(cfg.linearize.get("max_masks_per_sentence", 4096))
    seed = int(cfg.linearize.get("seed", 1234))
    rhos_t = torch.tensor(rhos, device=device, dtype=torch.float32)

    evaluated = 0
    masks_evaluated = 0

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

        valid_start = 1 if exclude_special else 0
        valid_end_b = attn_b.sum(dim=1).to(torch.long)
        if exclude_special:
            valid_end_b = torch.clamp(valid_end_b - 1, min=valid_start)
        n_valid_b = valid_end_b - valid_start
        valid_sentence_mask = n_valid_b > 0
        if not bool(valid_sentence_mask.any().item()):
            continue

        k_br = torch.round(n_valid_b.to(torch.float32).unsqueeze(1) * rhos_t.unsqueeze(0)).to(torch.long)
        k_br = torch.minimum(k_br, n_valid_b.unsqueeze(1))
        k_br = torch.where(n_valid_b.unsqueeze(1) > 0, torch.clamp(k_br, min=1), torch.zeros_like(k_br))
        max_k = int(k_br.max().item())
        if max_k <= 0:
            continue

        generator = torch.Generator(device=device)
        generator.manual_seed(seed + evaluated)

        # Global candidate tensor across the full sentence batch.
        rand_scores = torch.rand(
            (bs, len(rhos), max_masks_per_sentence, seq_len),
            device=device,
            generator=generator,
        )

        pos = torch.arange(seq_len, device=device).view(1, 1, 1, seq_len)
        valid_region = (pos >= valid_start) & (pos < valid_end_b.view(bs, 1, 1, 1))
        rand_scores = torch.where(valid_region, rand_scores, torch.full_like(rand_scores, -1e9))

        top_idx = torch.topk(rand_scores, k=max_k, dim=3, largest=True, sorted=False).indices

        keep_topk = (
            torch.arange(max_k, device=device).view(1, 1, 1, max_k)
            < k_br.view(bs, len(rhos), 1, 1)
        ).to(attn_f_b.dtype)
        all_masks_4d = torch.zeros(
            (bs, len(rhos), max_masks_per_sentence, seq_len),
            device=device,
            dtype=attn_f_b.dtype,
        )
        all_masks_4d.scatter_(3, top_idx, keep_topk.expand(bs, len(rhos), max_masks_per_sentence, max_k))

        # Force completely invalid sentences to all-zero masks.
        all_masks_4d = all_masks_4d * valid_sentence_mask.view(bs, 1, 1, 1).to(attn_f_b.dtype)

        masks_evaluated += int(valid_sentence_mask.sum().item()) * len(rhos) * max_masks_per_sentence

        all_linearized_losses = evaluate_linearized_masks_batched(
            attn_f_b,
            full_token_emb_b,
            full_rep_b,
            encoder,
            all_masks_4d,
        )

        best_idx = torch.argmin(all_linearized_losses, dim=2)
        selected_masks = all_masks_4d[
            torch.arange(bs, device=device).view(bs, 1),
            torch.arange(len(rhos), device=device).view(1, len(rhos)),
            best_idx,
        ]

        selected_real_losses = evaluate_selected_masks_batched(
            ids_b,
            attn_f_b,
            full_rep_b,
            encoder,
            selected_masks,
        )

        valid_idx = torch.nonzero(valid_sentence_mask, as_tuple=False).squeeze(1)
        selected_valid = selected_masks[valid_idx]
        losses_valid = selected_real_losses[valid_idx]
        L_eff_valid = attn_b[valid_idx].sum(dim=1).to(torch.float32).clamp(min=1.0)

        for ridx, rho in enumerate(rhos):
            rho_losses = losses_valid[:, ridx].detach().cpu().tolist()
            losses_by_rho[rho].extend(float(v) for v in rho_losses)

            rho_eff_sum[ridx] += float((selected_valid[:, ridx, :].sum(dim=1).to(torch.float32) / L_eff_valid).sum().item())

            if counts_pred is not None and counts_gold is not None and labels_b is not None:
                for bi in valid_idx.tolist():
                    labels_i = labels_b[bi]
                    flat_attn = attn_b[bi].bool().cpu()
                    flat_preds = selected_masks[bi, ridx].bool().cpu()
                    counts_pred[ridx] += Counts(labels_i, flat_attn, flat_preds)
                    counts_gold[ridx] += Counts(labels_i, flat_attn)

        evaluated += int(valid_sentence_mask.sum().item())

    if evaluated == 0:
        raise RuntimeError(
            "No samples evaluated. Try lowering max_length/subset, or increase --max-masks-per-sentence."
        )

    selection_rates = [value / float(evaluated) for value in rho_eff_sum]

    plots_dir = Path(str(cfg.linearize.get("plots_dir", "plots")))
    if not plots_dir.is_absolute():
        plots_dir = Path.cwd() / plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    loss_path = plots_dir / "linearize_loss.png"
    mean_losses, _ = save_oracle_loss_plot(rhos, losses_by_rho, loss_path)
    logger.info("Saved oracle loss plot to %s", loss_path)

    spearman_path = plots_dir / "spearman_vs_rho.png"
    rho_s = save_spearman_plot(rhos, mean_losses, spearman_path)
    logger.info("Saved Spearman plot to %s (rho=%.4f)", spearman_path, rho_s)

    if counts_pred is not None and counts_gold is not None:
        logger.info("Configured rho grid: %s", ", ".join(f"{rho:.3f}" for rho in rhos))
        logger.info("Effective selected-token rates: %s", ", ".join(f"{rate:.3f}" for rate in selection_rates))
        logger.info(
            "\nLabel selection matrix (oracle masks):\n%s",
            selection_rate_matrix_to_table(counts_pred, counts_gold, rhos),
        )
        save_label_plots(counts_pred, counts_gold, selection_rates, plots_dir, logger)

    logger.info("Evaluated samples: %d", evaluated)
    logger.info("Total masks evaluated: %d", masks_evaluated)
    logger.info("Random masks per sentence/rho: %d", max_masks_per_sentence)
    logger.info("Done.")


if __name__ == "__main__":
    main()
