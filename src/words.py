"""Word-level primitives shared by the selector, the oracle and the probe.

The encoder runs on subword tokens, but a subword is not a meaningful unit
to keep or discard: half a word is not a rationale. Selection therefore
happens over *words*, and these helpers move between the two layers.

Three reasons the word is the right atom, all measured on wikiann:

  * fragmentation is label-dependent (bert splits B-LOC into 1.85 pieces but
    I-ORG into 1.22), so counting subwords silently weights tags by how they
    tokenize rather than by how they are selected;
  * it is also tokenizer-dependent (pythia splits B-LOC into 2.52), so two
    encoders scored over their own subwords are not being compared on the
    same units -- whereas every tokenizer sees the identical 160,862 words;
  * the tagger is already word-level, so word-level selection removes the
    aggregation rule that otherwise sits between the two.

`word_ids` is the [B, L] tensor collate() builds, -1 wherever a position is
padding or a special token that belongs to no word.
"""
from __future__ import annotations

import torch


def first_subword_mask(word_ids: torch.Tensor) -> torch.Tensor:
    """[B, L] bool: True at the first subword position of each word."""
    valid = word_ids >= 0
    prev = torch.cat([torch.full_like(word_ids[:, :1], -1), word_ids[:, :-1]], dim=1)
    return valid & (word_ids != prev)


def word_slots(word_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """(slot, is_word, n_words, W).

    slot[b, l] is the word index position l belongs to, so every subword of a
    word shares one slot -- which is what makes gather/scatter exact inverses.
    """
    first = first_subword_mask(word_ids)
    is_word = word_ids >= 0
    slot = (first.cumsum(dim=1) - 1).clamp(min=0)
    n_words = first.sum(dim=1)
    return slot, is_word, n_words, int(n_words.max().item()) if n_words.numel() else 0


def gather_word_mean(x: torch.Tensor, word_ids: torch.Tensor) -> torch.Tensor:
    """[B, L, D] subword states -> [B, W, D], each word the mean of its pieces.

    Mean rather than first-subword: the selector is judging how much a word
    contributes to the sentence embedding, and every one of its pieces enters
    that pooling. (The probe uses first-subword instead -- a different model
    answering a different question; see ner/model.py.)
    """
    B, L, D = x.shape
    slot, is_word, _, W = word_slots(word_ids)
    idx = slot.unsqueeze(-1).expand(B, L, D)
    keep = is_word.unsqueeze(-1).to(x.dtype)
    total = torch.zeros(B, max(W, 1), D, device=x.device, dtype=x.dtype)
    total.scatter_add_(1, idx, x * keep)
    count = torch.zeros(B, max(W, 1), 1, device=x.device, dtype=x.dtype)
    count.scatter_add_(1, slot.unsqueeze(-1), keep)
    return total / count.clamp(min=1.0)


def word_valid(word_ids: torch.Tensor) -> torch.Tensor:
    """[B, W] float: 1 where the word slot is a real word."""
    slot, is_word, _, W = word_slots(word_ids)
    count = torch.zeros(word_ids.shape[0], max(W, 1), device=word_ids.device)
    count.scatter_add_(1, slot, is_word.float())
    return (count > 0).float()


def scatter_word_to_tokens(word_vals: torch.Tensor, word_ids: torch.Tensor) -> torch.Tensor:
    """[B, W] word values -> [B, L], broadcast across each word's subwords.

    The inverse of the slot mapping, and what makes a word-level decision
    enforceable on a subword-shaped attention mask: every piece of a word
    receives the same value, so a word can never be half-kept.
    """
    slot, is_word, _, _ = word_slots(word_ids)
    return word_vals.gather(1, slot) * is_word.to(word_vals.dtype)


def subword_counts(word_ids: torch.Tensor) -> torch.Tensor:
    """[B, W] how many subwords each word occupies -- a covariate, since
    fragmentation is exactly the confound word-level selection removes."""
    slot, is_word, _, W = word_slots(word_ids)
    count = torch.zeros(word_ids.shape[0], max(W, 1), device=word_ids.device)
    count.scatter_add_(1, slot, is_word.float())
    return count
