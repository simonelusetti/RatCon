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
            "batch_size": 64,
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
        "max_masks_per_sentence": 4096,
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


def valid_positions(attn: torch.Tensor, tokens: list[str], exclude_special: bool) -> list[int]:
    valid = [i for i, m in enumerate(attn.tolist()) if int(m) == 1]
    if not exclude_special:
        return valid
    return [i for i in valid if tokens[i] not in SPECIAL_TOKENS]


def k_from_rho(rho: float, n_valid: int) -> int:
    if n_valid <= 0:
        return 0
    k = int(round(float(rho) * float(n_valid)))
    return max(1, min(n_valid, k))


def build_masks_from_subsets(
    subsets: torch.Tensor,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    masks = torch.zeros((int(subsets.size(0)), seq_len), device=device, dtype=dtype)
    if subsets.numel() == 0:
        return masks
    masks.scatter_(1, subsets.long(), 1.0)
    return masks


def evaluate_masks_batched(
    ids_b: torch.Tensor,
    attn_f: torch.Tensor,
    full_rep: torch.Tensor,
    encoder,
    masks: torch.Tensor,
    batch_size: int = MASK_EVAL_BATCH_SIZE,
) -> torch.Tensor:
    losses = []
    n = masks.size(0)
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            m = masks[start:end]
            ids_rep = ids_b.expand(m.size(0), -1)
            eff_attn = attn_f.expand(m.size(0), -1) * m

            tok = encoder.token_embeddings(ids_rep, eff_attn)
            pred = encoder.pool(tok, eff_attn)
            full_rep_rep = full_rep.expand(m.size(0), -1)
            loss = 1.0 - F.cosine_similarity(pred, full_rep_rep, dim=-1)
            losses.append(loss)

    if not losses:
        return torch.empty(0, device=ids_b.device, dtype=full_rep.dtype)
    return torch.cat(losses, dim=0)


def random_search_subsets(
    attn: torch.Tensor,
    tokens: list[str],
    rho: float,
    exclude_special: bool,
    candidates: int,
    seed: int,
) -> tuple[torch.Tensor, int]:
    valid = valid_positions(attn, tokens, exclude_special)
    n_valid = len(valid)
    k = k_from_rho(rho, n_valid)
    if k <= 0:
        return torch.empty((0, 0), dtype=torch.long, device=attn.device), 0

    target = int(candidates)
    if target <= 0:
        return torch.empty((0, 0), dtype=torch.long, device=attn.device), 0

    valid_t = torch.tensor(valid, device=attn.device, dtype=torch.long)
    # Torch-native vectorized random subset generation:
    # random scores -> top-k indices per row -> map local positions to token indices.
    generator = torch.Generator(device=attn.device)
    generator.manual_seed(int(seed))
    rand_scores = torch.rand((target, n_valid), device=attn.device, generator=generator)
    local_pick = torch.topk(rand_scores, k=k, dim=1, largest=True, sorted=False).indices
    subsets = valid_t[local_pick]
    subsets, _ = torch.sort(subsets, dim=1)
    subsets = torch.unique(subsets, dim=0)
    return subsets, int(subsets.size(0))


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

    evaluated = 0
    masks_evaluated = 0

    pbar = tqdm(test_dl, desc="Dataset batches", dynamic_ncols=True, unit="batch", position=0)

    mask_pbar = tqdm(
        total=None,
        desc="Mask creation",
        dynamic_ncols=True,
        unit="mask",
        position=1,
        leave=False,
    )

    for batch in pbar:
        ids_b = batch["ids"].to(device)
        attn_b = batch["attn_mask"].to(device)
        tokens_b = batch["tokens"]
        labels_b = batch.get("labels", None)

        bs = ids_b.size(0)
        for bi in range(bs):
            ids = ids_b[bi]
            attn = attn_b[bi]
            tokens = tokens_b[bi]

            # Reuse full-sentence representation across all rho values for this sentence.
            ids_1 = ids.unsqueeze(0)
            attn_1 = attn.unsqueeze(0)
            attn_f_1 = attn_1.float()
            with torch.inference_mode():
                full_token_emb = encoder.token_embeddings(ids_1, attn_1)
                full_rep = encoder.pool(full_token_emb, attn_f_1)

            sample_masks: list[torch.Tensor] = []
            sample_losses: list[float] = []
            invalid_sample = False
            per_rho_masks: list[torch.Tensor] = []
            per_rho_counts: list[int] = []

            for ridx, rho in enumerate(rhos):
                subsets, sampled_count = random_search_subsets(
                    attn=attn,
                    tokens=tokens,
                    rho=rho,
                    exclude_special=exclude_special,
                    candidates=max_masks_per_sentence,
                    seed=seed + ridx + evaluated,
                )

                masks_evaluated += int(sampled_count)
                mask_pbar.update(int(sampled_count))
                mask_pbar.set_postfix_str(f"evaluated={masks_evaluated}")

                if sampled_count <= 0:
                    invalid_sample = True
                    break

                masks = build_masks_from_subsets(subsets, attn_f_1.size(1), attn_f_1.device, attn_f_1.dtype)
                per_rho_masks.append(masks)
                per_rho_counts.append(sampled_count)

            if invalid_sample:
                continue

            # Single batched forward for all rho candidate masks of this sentence.
            all_masks = torch.cat(per_rho_masks, dim=0)
            all_losses = evaluate_masks_batched(ids_1, attn_f_1, full_rep, encoder, all_masks)

            offset = 0
            for ridx in range(len(rhos)):
                count = per_rho_counts[ridx]
                chunk_losses = all_losses[offset: offset + count]
                chunk_masks = all_masks[offset: offset + count]
                idx = int(torch.argmin(chunk_losses).item())
                sample_losses.append(float(chunk_losses[idx].item()))
                sample_masks.append(chunk_masks[idx].clone())
                offset += count

            L_eff = max(float(attn.sum().item()), 1.0)
            for ridx, rho in enumerate(rhos):
                mask = sample_masks[ridx]
                losses_by_rho[rho].append(float(sample_losses[ridx]))
                rho_eff_sum[ridx] += float(mask.sum().item() / L_eff)

                if counts_pred is not None and counts_gold is not None and labels_b is not None:
                    labels_i = labels_b[bi]
                    flat_attn = attn.bool().cpu()
                    flat_preds = mask.bool().cpu()
                    counts_pred[ridx] += Counts(labels_i, flat_attn, flat_preds)
                    counts_gold[ridx] += Counts(labels_i, flat_attn)

            evaluated += 1

    mask_pbar.close()

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
        logger.info(
            "\nLabel selection matrix (oracle masks):\n%s",
            selection_rate_matrix_to_table(counts_pred, counts_gold, selection_rates),
        )
        save_label_plots(counts_pred, counts_gold, selection_rates, plots_dir, logger)

    logger.info("Evaluated samples: %d", evaluated)
    logger.info("Total masks evaluated: %d", masks_evaluated)
    logger.info("Random masks per sentence/rho: %d", max_masks_per_sentence)
    logger.info("Done.")


if __name__ == "__main__":
    main()
