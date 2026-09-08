import torch
import torch.nn as nn


def first_subword_mask(word_ids: torch.Tensor) -> torch.Tensor:
    """[B, L] bool: True at the first subword position of each word.

    word_ids is -1 at special/padding positions (see collate() in data.py).
    """
    valid = word_ids >= 0
    prev = torch.cat([torch.full_like(word_ids[:, :1], -1), word_ids[:, :-1]], dim=1)
    return valid & (word_ids != prev)


def gather_word_level(
    token_emb: torch.Tensor,
    word_ids: torch.Tensor,
    label_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collapse subword-level tensors to word-level ones via each word's first subword.

    token_emb: [B, L, D], word_ids: [B, L] long (-1 = not a word), label_ids: [B, L] long.
    Returns (word_emb [B, W, D], word_mask [B, W] bool, word_labels [B, W] long),
    W = max words in the batch. Padding slots in word_labels are left at -1.
    """
    B, L, D = token_emb.shape
    mask = first_subword_mask(word_ids)
    slot = mask.cumsum(dim=1) - 1
    W = int(mask.sum(dim=1).max().item()) if bool(mask.any()) else 0

    word_emb = token_emb.new_zeros(B, W, D)
    word_mask = torch.zeros(B, W, dtype=torch.bool, device=token_emb.device)
    word_labels = label_ids.new_full((B, W), -1)

    b_idx, l_idx = mask.nonzero(as_tuple=True)
    s_idx = slot[b_idx, l_idx]
    word_emb[b_idx, s_idx] = token_emb[b_idx, l_idx]
    word_mask[b_idx, s_idx] = True
    word_labels[b_idx, s_idx] = label_ids[b_idx, l_idx]

    return word_emb, word_mask, word_labels


class MLPTagger(nn.Module):
    """Per-token MLP probe on top of frozen token embeddings.

    Each word's embedding is classified independently (no cross-token
    mixing, no structured decoding), so performance reflects only what the
    frozen encoder itself makes linearly/nonlinearly recoverable about that
    token, uncontaminated by any sequence-modeling capacity of the probe.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_tags: int,
        hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_tags),
        )

    def forward(self, word_emb: torch.Tensor, word_mask: torch.Tensor) -> torch.Tensor:
        return self.net(word_emb)

    def loss(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        safe_tags = tags.masked_fill(~mask, -1)
        return nn.functional.cross_entropy(
            emissions.reshape(-1, emissions.shape[-1]),
            safe_tags.reshape(-1),
            weight=weight,
            ignore_index=-1,
        )

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        preds = emissions.argmax(dim=-1)
        lengths = mask.sum(dim=1).tolist()
        return [preds[i, :length].tolist() for i, length in enumerate(lengths)]
