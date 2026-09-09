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
src/train.py selects this implementation for task=oracle and owns the shared
setup, evaluation, artifact writing, and Forge run lifecycle.

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
import os
import math
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .sentence import SentenceEncoder
from .words import word_slots


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

    @contextmanager
    def search_precision(self):
        """Enable oracle TF32 without leaking it into later runs in a grid."""
        if not str(self.device).startswith("cuda"):
            yield
            return
        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32

    @torch.no_grad()
    def _best_for_group(self, ids_stack, attn_stack, expand, masks, targets_stack):
        """Best mask for every sentence in a group sharing the same (n_words, k).

        The batching that makes this viable: sentences with equal word count
        and equal budget share one candidate table, so S sentences x C
        candidates become a single S*C row problem instead of S separate
        C-row forwards. At wikiann's median length a lone sentence is a
        20-row batch -- far too small to occupy a GPU -- while the grouped
        form fills it.

        Candidates are over WORDS, but the encoder needs subwords, so `expand`
        ([S, n_words, L]) broadcasts each word decision onto that word's
        token positions. Sentences sharing a word count can still tokenize
        differently, which is exactly why the expansion is per sentence.
        """
        S, L = ids_stack.shape
        C = masks.shape[0]
        ids_rep = ids_stack.repeat_interleave(C, dim=0)
        # [S, C, L] -> [S*C, L]: word mask broadcast to this sentence's subwords
        masks_rep = torch.einsum("cw,swl->scl", masks, expand).reshape(S * C, L)
        valid_rep = attn_stack.repeat_interleave(C, dim=0)
        targets_rep = targets_stack.repeat_interleave(C, dim=0)

        rows_per_chunk = max(1, self.chunk_tokens // max(L, 1))
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
                    self.sent_encoder.token_embeddings(ids_rep[sl], block), block,
                    valid_mask=valid_rep[sl],
                )
            scores.append(F.cosine_similarity(pooled.float(), targets_rep[sl].float(), dim=-1))
            if self.progress is not None:
                self.progress.update(block.shape[0])
        return torch.cat(scores).view(S, C)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, embeddings: torch.Tensor, attn: torch.Tensor, rhos,
                word_ids: torch.Tensor | None = None):
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
        if word_ids is None:  # identity map -- see RationaleSelectorModel.forward
            word_ids = torch.where(attn.bool(),
                                   torch.arange(L, device=device).expand(B, L),
                                   torch.full((B, L), -1, device=device))
        slot, is_word, n_words, _ = word_slots(word_ids)

        # The search is over WORDS, so the budget is a fraction of words and
        # candidate counts collapse: at wikiann's median, C(13,6)=1716 over
        # subwords becomes C(8,4)=70 over words.
        groups: dict[tuple[int, int], list[tuple[int, int, torch.Tensor]]] = defaultdict(list)
        for b in range(B):
            valid = attn_f[b] > 0
            n = int(n_words[b])
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

            width = max(int(p.numel()) for _, b, p in items)
            S = len(items)
            ids_stack = torch.zeros(S, width, dtype=ids.dtype, device=device)
            attn_stack = torch.zeros(S, width, device=device)
            expand = torch.zeros(S, n, width, device=device)
            for i, (_, b, pos) in enumerate(items):
                m = pos.numel()
                ids_stack[i, :m] = ids[b, pos]
                attn_stack[i, :m] = 1.0
                expand[i, slot[b, pos], torch.arange(m, device=device)] = 1.0
            targets_stack = torch.stack([targets[b] for _, b, _ in items])
            scores = self._best_for_group(ids_stack, attn_stack, expand, masks, targets_stack)

            best = scores.argmax(dim=1)
            for i, (r, b, positions) in enumerate(items):
                # winning word mask, broadcast back onto this sentence's subwords
                g[r, b, positions] = masks[best[i]][slot[b, positions]]
                losses.append(1.0 - scores[i, best[i]])

        loss = torch.stack(losses).mean() if losses else torch.zeros((), device=device)
        return g, g, loss


def total_candidate_rows(dataset, rhos, max_combinations: int, samples: int) -> int:
    """Rows the search will score, for a progress bar with a real ETA."""
    total = 0
    # Word count, not token count: the search enumerates over words.
    for attn, wids in zip(dataset["attn_mask"], dataset["word_ids"]):
        n = len({w for w, a in zip(wids, attn) if a and w is not None and w >= 0})
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


def main(cfg) -> int:
    """Compatibility entry point for existing `forge -M oracle` launchers."""
    if str(cfg.get("task", "")) != "oracle":
        raise ValueError("The oracle entry point requires task=oracle.")
    from .train import main as run
    return run(cfg)
