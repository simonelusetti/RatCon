from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import hypergeom
import torch

from .data import PAD_TAG


def bonferroni_threshold(m: int, alpha: float = 0.05) -> float:
    """Bonferroni family-wise-error threshold for m simultaneous tests."""
    return alpha / m if m > 0 else 0.0



def _norm_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convolve two PMFs and renormalise.

    np.convolve is direct, O(len(a)*len(b)). Profiling showed ~99% of a run's
    post-training time went here: the common categories build supports in the
    tens of thousands ("O" reaches 81,710 at rho=1) and the upper levels of the
    reduction tree convolve arrays that large against each other. FFT
    convolution is O(n log n) and agrees with the direct form to ~1e-16 per
    entry, leaving the derived z and p identical to 12+ significant figures.
    Small arrays keep the direct path, where FFT setup would dominate.
    """
    conv = fftconvolve(a, b) if a.size * b.size > 4096 else np.convolve(a, b)
    conv = np.clip(conv, 0.0, None)
    return conv / conv.sum()


def _conv_power(pmf: np.ndarray, m: int) -> np.ndarray:
    """m-fold self-convolution, by repeated squaring: O(log m) convolutions."""
    result, base = None, pmf
    while m:
        if m & 1:
            result = base if result is None else _norm_conv(result, base)
        m >>= 1
        if m:
            base = _norm_conv(base, base)
    return result


def exact_null_distribution(
    n_values: Sequence[int],
    c_values: Sequence[int],
    k_values: Sequence[int],
) -> np.ndarray:
    """Exact null PMF for the corpus-wide count of category tokens selected under
    independent per-sentence uniformly-random k_i-of-n_i sampling without
    replacement (the same budgeted top-k null the real selector's inference
    procedure follows).

    Within one sentence, "how many of the k_i drawn tokens have the category"
    is exactly Hypergeometric(n_i, c_i, k_i) — a fact about sampling without
    replacement, not a modeling choice. Sentences are independent, so the
    corpus-wide total's exact distribution is the convolution of the
    per-sentence hypergeometric PMFs. This replaces both Monte Carlo
    simulation (which has sampling noise and, if read out via tail-counting,
    a hard p-value floor) and a normal approximation (invalid for rare
    categories, where most sentences contribute nothing and the CLT doesn't
    apply) with the exact distribution, valid uniformly for common and rare
    categories alike.

    Sentences with c_i == 0 contribute a point mass at 0 and are skipped —
    for rare categories, that's the overwhelming majority of sentences, and
    is what keeps this tractable.
    """
    # Sentences repeat: (n, c, k) is drawn from a small set -- wikiann averages
    # 8 words per sentence, so 20,000 sentences collapse to a few hundred
    # distinct triples. Building each PMF once, then combining identical ones
    # by repeated squaring (convolution is associative and commutative), turns
    # ~14,000 convolutions into ~2,000 without changing the result.
    groups: Counter = Counter()
    for n, c, k in zip(n_values, c_values, k_values):
        if c > 0 and k > 0:
            groups[(int(n), int(c), int(k))] += 1

    if not groups:
        return np.array([1.0])  # category never occurs: point mass at 0

    pmfs: list[np.ndarray] = []
    for (n, c, k), multiplicity in groups.items():
        lo = max(0, k - n + c)
        hi = min(c, k)
        xs = np.arange(lo, hi + 1)
        support = hypergeom.pmf(xs, n, c, k)
        support = support / support.sum()
        base = np.zeros(hi + 1)
        base[lo:hi + 1] = support
        pmfs.append(_conv_power(base, multiplicity))

    # Pairwise tree reduction keeps intermediate arrays balanced in size,
    # rather than convolving sequentially into one array that grows across
    # thousands of steps (matters for common categories, e.g. "O", where the
    # final support can run into the tens of thousands).
    while len(pmfs) > 1:
        merged = [_norm_conv(pmfs[i], pmfs[i + 1]) for i in range(0, len(pmfs) - 1, 2)]
        if len(pmfs) % 2 == 1:
            merged.append(pmfs[-1])
        pmfs = merged

    return pmfs[0]


def exact_pvalue_z(observed: int, null_pmf: np.ndarray) -> tuple[float, float]:
    """Two-sided exact p-value and z-score from an exact null PMF
    (see exact_null_distribution).

    z uses the PMF's own exact mean/variance (not estimated from samples).
    The p-value sums the probability of every outcome at least as unlikely
    as the observed one — the standard convention for exact tests on
    discrete, possibly-skewed nulls (the same principle R's fisher.test
    uses), rather than a naive doubled one-sided tail, which can misrepresent
    significance for skewed/low-count nulls.
    """
    idx = np.arange(null_pmf.size)
    mean = float(np.dot(idx, null_pmf))
    var = float(np.dot((idx - mean) ** 2, null_pmf))
    std = var ** 0.5
    z = (observed - mean) / std if std > 1e-9 else 0.0

    p_obs = float(null_pmf[observed]) if 0 <= observed < null_pmf.size else 0.0
    p_value = float(null_pmf[null_pmf <= p_obs + 1e-15].sum())
    p_value = min(1.0, max(p_value, float(np.finfo(float).tiny)))
    return p_value, z


def build_token_frequency_table(ids_per_example: Iterable[list[int]]) -> Counter:
    """Token-id counts over a (train) split's tokenized examples."""
    freq = Counter()
    for ids in ids_per_example:
        freq.update(ids)
    return freq


class SelectionLog:
    """Per-token record of (label, log-frequency, selected-at-each-rho) over an eval split.

    Feeds the frequency-controlled effect sizes and the logistic-regression
    check of Section 5 without touching the Counts aggregation path.
    """

    def __init__(self, rhos: Sequence[float], freq_table: Counter) -> None:
        self.rhos = [float(r) for r in rhos]
        self.freq_table = freq_table
        self.label_to_idx: dict[str, int] = {}
        self._labels: list[np.ndarray] = []
        self._log_freqs: list[np.ndarray] = []
        self._selected: list[list[np.ndarray]] = [[] for _ in self.rhos]
        self._sentence_ids: list[np.ndarray] = []
        self._next_sentence_id = 0

    def add_batch(
        self,
        flat_ids: Sequence[int],
        flat_labels: Sequence[str],
        flat_attn: Sequence[bool],
        pred_masks: torch.Tensor,
        seq_len: int,
    ) -> None:
        """pred_masks: [R, N] selection masks aligned with the flat token stream.

        seq_len: L of the [B, L] batch flat_* was flattened from (row-major),
        used to recover each token's sentence id for the post-hoc permutation
        test (build_freq_binned_effects_payload), which needs to regroup the
        buffered flat arrays back into per-sentence spans without re-running
        the encoder.
        """
        valid = np.asarray(flat_attn, dtype=bool) & (np.asarray(flat_labels) != PAD_TAG)
        if not valid.any():
            return

        n_sentences = len(flat_attn) // seq_len
        sentence_ids = (np.arange(len(flat_attn)) // seq_len + self._next_sentence_id)[valid]
        self._next_sentence_id += n_sentences

        label_idx = np.array(
            [
                self.label_to_idx.setdefault(lbl, len(self.label_to_idx))
                for lbl, ok in zip(flat_labels, valid)
                if ok
            ],
            dtype=np.int32,
        )
        ids_arr = np.asarray(flat_ids)[valid]
        log_freqs = np.log1p([self.freq_table.get(int(t), 0) for t in ids_arr]).astype(np.float32)

        self._labels.append(label_idx)
        self._log_freqs.append(log_freqs)
        self._sentence_ids.append(sentence_ids.astype(np.int64))
        pred_np = pred_masks.detach().cpu().numpy() > 0.5
        for r in range(len(self.rhos)):
            self._selected[r].append(pred_np[r][valid])

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Returns (labels [N], log_freqs [N], selected [R, N], sentence_ids [N], label_names)."""
        labels = np.concatenate(self._labels) if self._labels else np.array([], dtype=np.int32)
        log_freqs = np.concatenate(self._log_freqs) if self._log_freqs else np.array([], dtype=np.float32)
        sentence_ids = np.concatenate(self._sentence_ids) if self._sentence_ids else np.array([], dtype=np.int64)
        selected = np.stack([
            np.concatenate(per_rho) if per_rho else np.array([], dtype=bool)
            for per_rho in self._selected
        ]) if self._labels else np.zeros((len(self.rhos), 0), dtype=bool)
        names = [lbl for lbl, _ in sorted(self.label_to_idx.items(), key=lambda kv: kv[1])]
        return labels, log_freqs, selected, sentence_ids, names


class Counts:
    def __init__(
        self,
        labels: list | None = None,
        mask: torch.Tensor | None = None,
        pred_mask: torch.Tensor | None = None,
        pad: str = PAD_TAG,
    ) -> None:
        self.data = {}
        self.pad = pad

        if labels is None:
            return  # Fully empty Counts

        self.data = dict.fromkeys(set(labels) - {self.pad}, 0)

        if mask is None:
            return  # Labels but all set to 0

        if pred_mask is not None:
            mask = mask.bool() * pred_mask # Pred counts

        mask = mask.tolist()
        for c in self.data.keys():
            self.data[c] = sum(1 if yi and xi == c else 0 for xi, yi in zip(labels, mask))

    def __add__(self, other: Counts) -> Counts:
        if not isinstance(other, Counts):
            raise ValueError("Can only add Counts to Counts.")
        result = Counts()
        result.data = {
            c: self.data.get(c, 0) + other.data.get(c, 0)
            for c in self.data.keys() | other.data.keys()
        }
        return result
