"""Oracle mask search: the brute-force upper bound on the selector.

The trained selector answers "which rho-fraction of tokens should I keep?"
with a learned scoring function. This answers the same question by trying
every possible answer and keeping the best one. There is no model and no
training -- the artifact is the set of winning masks.

That makes it the ceiling the selector is measured against. Any gap between
the two is the selector failing to find masks that exist; anything the
*oracle* cannot do is a limit of the objective rather than of the model.

Drop-in by design
-----------------
OracleSelector exposes exactly RationaleSelectorModel's forward signature
--- (ids, embeddings, attn, rhos) -> (z, g, loss) --- so it flows through
the evaluation stack unchanged: the same Counts/SelectionLog bookkeeping,
the same save_eval_artifacts, and run_stsb_sweep works on it untouched.
Nothing in the training path needed a branch for this.

Efficiency
----------
Three things carry it, all of them tensor work rather than threads:

  * Mask tables are cached per (n, k). Every sentence of the same length
    shares one [C, n] matrix, so the combinatorics are enumerated once for
    the whole run rather than once per sentence.
  * (sentence, rho) pairs are bucketed by (n, k) and solved together. Equal
    length and equal budget means an identical candidate table, so S
    sentences x C candidates collapse into one S*C row forward. A lone
    median-length sentence is an 84-row batch, nowhere near enough to
    occupy a GPU; grouped, the same work fills it. Grouping by exact length
    also means no batch ever carries padding it does not need.
  * Rows are chunked to a token budget rather than a fixed count, so short
    sentences pack more candidates per forward than long ones -- which
    matters on a corpus whose lengths span an order of magnitude.

C(n, k) is small for typical sentences but unbounded in the tail, so
runtime.oracle.max_combinations caps it: above the cap the search falls
back to runtime.oracle.samples uniform draws and reports the best of those. That makes the
result a lower bound on the true oracle for those sentences, never an
overstatement, and the fraction affected is logged.
"""
from __future__ import annotations

import itertools
import json
import os
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from forge import start_run
from numpy import linspace

from .data import SPECIAL_TAG, initialize_data
from .eval import save_eval_artifacts
from .metrics import Counts, SelectionLog, build_token_frequency_table
from .retrival_fun import run_stsb_sweep
from .sentence import SentenceEncoder
from .utils import (
    build_final_metrics,
    configure_runtime,
    get_logger,
    should_disable_tqdm,
    start_run_metrics_capture,
    to_device,
    write_metrics_details,
)
from .view import save_eval_plots


class MaskTables:
    """Cache of [C, n] candidate-mask matrices, keyed by (n, k).

    The point of the cache: masks depend only on the *shape* of the problem,
    not on the sentence, so a corpus of 20k sentences with a median length of
    9 needs a few hundred tables in total instead of 200k enumerations.
    """

    def __init__(self, max_combinations: int, samples: int, seed: int, device: str) -> None:
        self.max_combinations = max_combinations
        self.samples = samples
        self.device = device
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self._tables: dict[tuple[int, int], tuple[torch.Tensor, bool]] = {}

    def get(self, n: int, k: int) -> tuple[torch.Tensor, bool]:
        """Return ([C, n] float mask matrix, exhaustive?)."""
        key = (n, k)
        if key in self._tables:
            return self._tables[key]

        total = math.comb(n, k)
        if total <= self.max_combinations:
            idx = torch.tensor(list(itertools.combinations(range(n), k)), dtype=torch.long)
            exhaustive = True
        else:
            # Uniform k-subsets without a Python loop: argsort of noise gives
            # a random permutation per row, and the first k columns of it are
            # a uniform random subset.
            noise = torch.rand(self.samples, n, generator=self.generator)
            idx = noise.argsort(dim=1)[:, :k]
            exhaustive = False

        masks = torch.zeros(idx.shape[0], n, dtype=torch.float)
        masks.scatter_(1, idx, 1.0)
        table = (masks.to(self.device), exhaustive)
        self._tables[key] = table
        return table


def budget(rho: float, n: int) -> int:
    """Token budget, matching RationaleSelectorModel.forward exactly."""
    return max(1, int(round(rho * n))) if n > 0 else 0


class OracleSelector(nn.Module):
    """Brute-force stand-in for RationaleSelectorModel.

    Has no parameters: forward() searches instead of predicting. The returned
    `z` is the hard mask itself rather than a relaxation, since there is no
    gradient to carry -- nothing here is trained.
    """

    def __init__(
        self,
        sent_encoder: SentenceEncoder,
        max_combinations: int = 10_000,
        samples: int = 10_000,
        chunk_tokens: int = 1 << 18,
        seed: int = 0,
        device: str = "cpu",
        precision: str = "fp16",
        progress: tqdm | None = None,
    ) -> None:
        super().__init__()
        self.sent_encoder = sent_encoder
        self.tables = MaskTables(max_combinations, samples, seed, device)
        self.chunk_tokens = chunk_tokens
        self.device = device
        self.autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
        self.progress = progress
        self.n_exhaustive = 0
        self.n_capped = 0

    @torch.no_grad()
    def _best_for_group(self, ids_stack, masks, targets_stack):
        """Best mask for every sentence in a group sharing the same (n, k).

        The batching that makes this viable: sentences of equal length with
        equal budget share one candidate table, so S sentences x C candidates
        become a single S*C row problem instead of S separate C-row forwards.
        At wikiann's median length a lone sentence is an 84-row batch -- far
        too small to occupy a GPU -- while the grouped form fills it.
        """
        S, n = ids_stack.shape
        C = masks.shape[0]
        ids_rep = ids_stack.repeat_interleave(C, dim=0)
        masks_rep = masks.repeat(S, 1)
        targets_rep = targets_stack.repeat_interleave(C, dim=0)

        rows_per_chunk = max(1, self.chunk_tokens // max(n, 1))
        use_amp = self.autocast_dtype is not None and str(self.device).startswith("cuda")
        scores = []
        for start in range(0, S * C, rows_per_chunk):
            sl = slice(start, start + rows_per_chunk)
            block = masks_rep[sl]
            # Reduced precision is safe here in a way it would not be in
            # training: the search only needs the *ranking* of candidates to
            # survive, and fp16 reproduces fp32 pooled embeddings to a cosine
            # of 0.999999 while running >3x faster. Scores are taken back to
            # fp32 before the comparison so the argmax is not decided in
            # half precision.
            with torch.autocast("cuda", dtype=self.autocast_dtype, enabled=use_amp):
                pooled = self.sent_encoder.pool(
                    self.sent_encoder.token_embeddings(ids_rep[sl], block), block
                )
            scores.append(F.cosine_similarity(pooled.float(), targets_rep[sl].float(), dim=-1))
            if self.progress is not None:
                self.progress.update(block.shape[0])
        return torch.cat(scores).view(S, C)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, embeddings: torch.Tensor, attn: torch.Tensor, rhos):
        B, L = ids.shape
        R = len(rhos)
        device = ids.device
        attn_f = attn.float()

        # Same target the selector reconstructs towards, computed from the
        # already-available full-attention embeddings rather than recomputed.
        targets = self.sent_encoder.pool_full(embeddings, attn_f)

        g = torch.zeros(R, B, L, device=device)
        losses = []

        # Bucket every (sentence, rho) pair by the shape of its search, so
        # that one forward serves all of them.
        groups: dict[tuple[int, int], list[tuple[int, int, torch.Tensor]]] = defaultdict(list)
        for b in range(B):
            valid = attn_f[b] > 0
            n = int(valid.sum())
            if n == 0:
                continue
            positions = valid.nonzero(as_tuple=True)[0]
            for r, rho in enumerate(rhos):
                k = budget(float(rho), n)
                if k >= n:  # keeping everything: only one mask exists
                    g[r, b, positions] = 1.0
                    losses.append(torch.zeros((), device=device))
                    continue
                groups[(n, k)].append((r, b, positions))

        for (n, k), items in groups.items():
            masks, exhaustive = self.tables.get(n, k)
            self.n_exhaustive += len(items) if exhaustive else 0
            self.n_capped += 0 if exhaustive else len(items)

            ids_stack = torch.stack([ids[b, pos] for _, b, pos in items])
            targets_stack = torch.stack([targets[b] for _, b, _ in items])
            scores = self._best_for_group(ids_stack, masks, targets_stack)

            best = scores.argmax(dim=1)
            for i, (r, b, positions) in enumerate(items):
                g[r, b, positions] = masks[best[i]]
                losses.append(1.0 - scores[i, best[i]])

        loss = torch.stack(losses).mean() if losses else torch.zeros((), device=device)
        return g, g, loss


def total_candidate_rows(dataset, rhos, max_combinations: int, samples: int) -> int:
    """Rows the search will score, for a progress bar with a real ETA."""
    total = 0
    for attn in dataset["attn_mask"]:
        n = int(sum(attn))
        if n == 0:
            continue
        for rho in rhos:
            k = budget(float(rho), n)
            if k >= n:
                continue
            c = math.comb(n, k)
            total += c if c <= max_combinations else samples
    return total


def build_progress(dataset, rhos, cfg) -> tqdm:
    oracle_cfg = cfg.runtime.get("oracle", {})
    total = total_candidate_rows(
        dataset, rhos,
        int(oracle_cfg.get("max_combinations", 10_000)),
        int(oracle_cfg.get("samples", 10_000)),
    )
    # Deliberately NOT gated on should_disable_tqdm's isatty check. This is a
    # multi-hour search whose whole point is being able to see how far along
    # it is, and it is normally launched piped into `tee` for a log -- which
    # makes stderr a pipe and would silently switch the bar off exactly when
    # it is most wanted. Off a tty the refresh is throttled instead, so the
    # log gets a readable progress line every 30s rather than thousands.
    on_tty = sys.stderr.isatty()
    return tqdm(
        total=total,
        desc="Oracle mask search",
        unit="mask",
        unit_scale=True,
        dynamic_ncols=on_tty,
        mininterval=0.1 if on_tty else 30.0,
        disable=bool(cfg.runtime.eval.short_log) or bool(cfg.runtime.grid)
                or bool(os.environ.get("DISABLE_TQDM")),
        file=sys.stderr,
    )


def save_masks(path: Path, rhos, per_rho_indices: list[list[np.ndarray]]) -> None:
    """Persist the winning masks as flat indices plus per-sentence offsets.

    Ragged by nature (one variable-length index list per sentence per rho),
    so a flat array with an offsets vector rather than an object array --
    compact, and loadable without pickle.
    """
    payload = {"rho": np.asarray([float(r) for r in rhos], dtype=np.float32)}
    for r, per_sentence in enumerate(per_rho_indices):
        flat = np.concatenate(per_sentence) if per_sentence else np.zeros(0, dtype=np.int32)
        offsets = np.cumsum([0] + [len(a) for a in per_sentence]).astype(np.int64)
        payload[f"indices_{r}"] = flat.astype(np.int32)
        payload[f"offsets_{r}"] = offsets
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg) -> int:
    """Run the oracle search and emit the same artifacts a selector run does.

    Deliberately its own entry point rather than a branch inside
    SelectorTrainer: there is no training loop, no optimiser and no
    checkpoint here, so almost nothing of that class would apply. What IS
    shared -- the counting, the artifacts, the plots -- is shared by calling
    the same functions, not by threading a flag through a second code path.
    """
    start_capture = start_run_metrics_capture()
    run = start_run(cfg)
    logger = get_logger()

    if str(cfg.get("task", "")) != "oracle":
        raise SystemExit(
            "Refusing to run: task must be 'oracle'. forge derives the experiment "
            "signature from the config alone, so without this an oracle run would "
            "hash to the same experiment as the selector run it is meant to be "
            "compared against. Pass task=oracle."
        )

    logger.info(f"Exp signature: {run.signature}")
    logger.info(repr(cfg))

    cfg.runtime, changed_device = configure_runtime(cfg.runtime)
    if changed_device:
        logger.warning("CUDA requested but unavailable, using CPU.")
    device = cfg.runtime.device

    train_dl, test_dl, encoder, tokenizer, labels_set, _ = initialize_data(
        cfg.data, cfg.runtime.data, logger, device=device,
        keep_special=bool(cfg.model.get("keep_special", True)),
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    rhos = linspace(*cfg.model.loss.sweep_range)
    oracle_cfg = cfg.runtime.get("oracle", {})
    progress = build_progress(test_dl.dataset, rhos, cfg)
    selector = OracleSelector(
        encoder,
        max_combinations=int(oracle_cfg.get("max_combinations", 10_000)),
        samples=int(oracle_cfg.get("samples", 10_000)),
        chunk_tokens=int(oracle_cfg.get("chunk_tokens", 1 << 18)),
        seed=int(cfg.runtime.get("seed", 42)),
        device=device,
        precision=str(oracle_cfg.get("precision", "fp16")),
        progress=progress,
    )
    # TF32 doubles fp32 matmul throughput on Ampere-and-later. Set here
    # rather than globally so the selector's training path keeps the exact
    # numerics every existing run was produced with.
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled; search precision=%s", oracle_cfg.get("precision", "fp16"))

    logger.info(
        "Oracle search over %s candidate masks (cap=%s, fallback samples=%s)",
        f"{progress.total:,}", oracle_cfg.get("max_combinations", 10_000),
        oracle_cfg.get("samples", 10_000),
    )

    label_set = None if labels_set is None else set(labels_set) | {SPECIAL_TAG}
    label_order = sorted(label_set) if label_set is not None else []
    counts_pred = [Counts(labels=label_set) for _ in rhos]
    counts_gold = [Counts(labels=label_set) for _ in rhos]
    selection_log = None
    if label_set is not None:
        selection_log = SelectionLog(rhos, build_token_frequency_table(train_dl.dataset["ids"]))

    per_rho_indices: list[list[np.ndarray]] = [[] for _ in rhos]
    total_loss, examples = 0.0, 0

    with torch.no_grad():
        for batch in test_dl:
            batch = to_device(device, batch)
            ids, attn = batch["ids"], batch["attn_mask"]
            emb = encoder.token_embeddings(ids, attn)
            _, g, loss = selector(ids, emb, attn, rhos=rhos)

            bs = ids.size(0)
            examples += bs
            total_loss += float(loss) * bs

            for r in range(len(rhos)):
                for b in range(bs):
                    per_rho_indices[r].append(
                        g[r, b].nonzero(as_tuple=True)[0].cpu().numpy().astype(np.int32)
                    )

            if label_set is not None:
                flat_attn = attn.bool().view(-1).cpu()
                flat_labels = [lab for seq in batch["labels"] for lab in seq]
                word_ids = batch.get("word_ids")
                if word_ids is not None:
                    flat_wids = word_ids.view(-1).cpu().tolist()
                    flat_labels = [
                        SPECIAL_TAG if (is_att and wid < 0 and lbl == "-100") else lbl
                        for lbl, is_att, wid in zip(flat_labels, flat_attn.tolist(), flat_wids)
                    ]
                for i in range(len(rhos)):
                    counts_pred[i] += Counts(flat_labels, flat_attn, g[i].cpu().view(-1))
                    counts_gold[i] += Counts(flat_labels, flat_attn)
                if selection_log is not None:
                    selection_log.add_batch(
                        ids.view(-1).cpu().tolist(), flat_labels, flat_attn.tolist(),
                        g.reshape(len(rhos), -1), seq_len=attn.shape[1],
                    )
    progress.close()

    total_loss /= max(1, examples)
    searched = selector.n_exhaustive + selector.n_capped
    exhaustive_frac = selector.n_exhaustive / max(1, searched)
    logger.info(
        "Oracle reconstruction loss=%.4f over %d sentences | %.1f%% of (sentence, rho) "
        "pairs searched exhaustively, the rest capped (a lower bound on the true oracle there)",
        total_loss, examples, 100 * exhaustive_frac,
    )

    save_masks(Path("data") / "oracle_masks.npz", rhos, per_rho_indices)
    logger.info("Saved winning masks to data/oracle_masks.npz")

    stsb = None
    if bool(oracle_cfg.get("stsb", False)):
        base, ours, rand = run_stsb_sweep(
            cfg=cfg, device=device, encoder=encoder, tokenizer=tokenizer, selector=selector,
        )
        stsb = {
            "base": float(base),
            "ours_by_rho": {str(float(k)): float(v) for k, v in ours.items()},
            "random_by_rho": {str(float(k)): float(v) for k, v in rand.items()},
        }
    else:
        logger.info("Skipping the STS-B sweep (runtime.oracle.stsb=false): brute-forcing it "
                    "means a fresh search per pair per rho on the finer eval grid.")

    artifact_paths = save_eval_artifacts(
        counts_pred=counts_pred, counts_gold=counts_gold, rhos=rhos,
        label_order=label_order, selection_log=selection_log, stsb=stsb,
    )
    for name, path in artifact_paths.items():
        logger.info("Saved %s artifact to: %s", name, path)
    for name, path in save_eval_plots(artifact_paths.keys(), dataset_name=cfg.data.dataset).items():
        logger.info("Saved %s plot to: %s", name, path)

    Path("data/oracle_summary.json").write_text(json.dumps({
        "reconstruction_loss": total_loss,
        "sentences": examples,
        "exhaustive_fraction": exhaustive_frac,
        "searched_pairs": searched,
        "candidate_rows": int(progress.total),
    }, indent=2), encoding="utf-8")

    write_metrics_details(
        cfg=cfg, run=run, start_capture=start_capture,
        epochs_completed=0, epochs_target=0, training_completed=True,
    )
    run.finish({
        "eval_loss": total_loss,
        "exhaustive_fraction": exhaustive_frac,
        **{k: v for k, v in build_final_metrics([], start_capture, 0).items()
           if k in ("duration_seconds", "max_rss_gb")},
    })
    return 0
