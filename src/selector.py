from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sentence import SentenceEncoder
from src.words import (
    gather_word_mean, scatter_word_to_tokens, word_valid,
)


# ------------------------------------------------------------
# Selector MLP
# ------------------------------------------------------------

class SelectorMLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden: int,
        dropout: float,
        condition_on_rho: bool = True,
    ) -> None:
        super().__init__()
        self.condition_on_rho = condition_on_rho
        in_dim = d_model + 1 if condition_on_rho else d_model
        self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, token_emb: torch.Tensor, rho: torch.Tensor | None = None) -> torch.Tensor:
        if self.condition_on_rho:
            if rho is None:
                raise ValueError("rho must be provided when condition_on_rho is enabled")
            rho = rho[:, None, None].to(dtype=token_emb.dtype, device=token_emb.device)
            rho = rho.expand(token_emb.shape[0], token_emb.shape[1], 1)
            x = torch.cat([token_emb, rho], dim=-1)
        else:
            x = token_emb
        x = self.ln(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)


# ------------------------------------------------------------
# Differentiable Soft Ranking
# ------------------------------------------------------------

def soft_rank(
    scores: torch.Tensor,
    attn: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    attn = attn.float()
    scores = scores.masked_fill(attn == 0, 0.0)

    denom = attn.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (scores * attn).sum(dim=1, keepdim=True) / denom
    var = (((scores - mean) ** 2) * attn).sum(dim=1, keepdim=True) / denom
    std = torch.sqrt(var + 1e-6)
    scores = (scores - mean) / std

    diff = scores.unsqueeze(2) - scores.unsqueeze(1)

    p = torch.sigmoid((-diff) / tau)

    pair_mask = attn.unsqueeze(1) * attn.unsqueeze(2)
    p = p * pair_mask

    r = 1.0 + p.sum(dim=2) - 0.5
    r = r.masked_fill(attn == 0, 1e9)

    return r


# ------------------------------------------------------------
# Rationale Selector Model
# ------------------------------------------------------------

class RationaleSelectorModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden: int | None = None,
        dropout: float = 0.1,
        sent_encoder: SentenceEncoder | None = None,
        loss_cfg: dict | None = None,
        selector_cfg: dict | None = None,
    ) -> None:
        super().__init__()

        if hidden is None:
            hidden = 4 * embedding_dim // 3

        selector_cfg = selector_cfg or {}
        self.condition_on_rho = bool(selector_cfg.get("condition_on_rho", True))

        self.selector = SelectorMLP(
            embedding_dim,
            hidden,
            dropout,
            condition_on_rho=self.condition_on_rho,
        )
        self.sent_encoder = sent_encoder
        self.loss_cfg = loss_cfg

        self.tau_rank = float(selector_cfg.get("tau_rank", 0.05))
        self.tau_gate = float(selector_cfg.get("tau_gate", 0.2))

    def _rank(
        self,
        embeddings: torch.Tensor,
        attn: torch.Tensor,
        rhos: Sequence[float],
    ) -> torch.Tensor:
        """Soft ranks over the selection units, shape [R, B, U].

        Called with WORD embeddings and a word validity mask -- the unit of
        selection is the word, not the subword (see src/words.py).
        """
        param_dtype = next(self.parameters()).dtype
        if embeddings.dtype != param_dtype:
            embeddings = embeddings.to(param_dtype)

        attn_f = attn.float()
        B, L = attn.shape
        R = len(rhos)
        device = embeddings.device
        rhos_t = torch.tensor(list(rhos), device=device, dtype=torch.float32)

        emb = embeddings * attn.unsqueeze(-1)
        if self.condition_on_rho:
            emb_rep = emb[None].expand(R, B, L, -1).reshape(R * B, L, emb.shape[-1])
            attn_rep_for_rank = attn_f[None].expand(R, B, L).reshape(R * B, L)
            rho_per_example = rhos_t[:, None].expand(R, B).reshape(R * B)

            scores = self.selector(emb_rep, rho_per_example)
            scores = scores.masked_fill(attn_rep_for_rank == 0, 0.0)

            ranks = soft_rank(scores, attn_rep_for_rank, tau=self.tau_rank).view(R, B, L)
        else:
            scores = self.selector(emb)
            scores = scores.masked_fill(attn_f == 0, 0.0)
            shared_ranks = soft_rank(scores, attn_f, tau=self.tau_rank)
            ranks = shared_ranks.unsqueeze(0).expand(R, -1, -1)

        return ranks

    def forward(
        self,
        ids: torch.Tensor,
        embeddings: torch.Tensor,
        attn: torch.Tensor,
        rhos: Sequence[float],
        word_ids: torch.Tensor | None = None,
    ):
        """Select whole WORDS, then broadcast the decision to their subwords.

        word_ids=None falls back to one word per token, which reproduces the
        old subword-level behaviour exactly -- kept only so single-sentence
        analysis helpers that have no word map still run. Training and
        evaluation always pass it.
        """
        param_dtype = next(self.parameters()).dtype
        if embeddings.dtype != param_dtype:
            embeddings = embeddings.to(param_dtype)

        device = embeddings.device

        attn_f = attn.float()
        with torch.no_grad():
            full_rep = self.sent_encoder.pool_full(embeddings, attn_f)

        B, L = ids.shape
        R = len(rhos)

        if word_ids is None:
            # identity map: every attended position is its own "word"
            word_ids = torch.where(attn.bool(),
                                   torch.arange(L, device=device).expand(B, L),
                                   torch.full((B, L), -1, device=device))
        word_emb = gather_word_mean(embeddings, word_ids)
        selection_f = word_valid(word_ids)
        U = selection_f.shape[1]
        # rho is now a fraction of WORDS: "keep 30% of the words" rather than
        # 30% of the subword pieces, which is both interpretable and identical
        # across tokenizers.
        L_eff = selection_f.sum(dim=1).float()

        rhos_t = torch.tensor(list(rhos), device=device, dtype=torch.float32)

        ranks = self._rank(word_emb, selection_f, rhos)

        k = (rhos_t[:, None] * L_eff[None]).round().long()
        k = torch.where(L_eff[None] > 0, k.clamp(min=1), torch.zeros_like(k))

        gate_raw = torch.sigmoid(
            (k.float()[:, :, None] - ranks) / self.tau_gate
        ) * selection_f[None]

        z_word = gate_raw / gate_raw.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        z_word = z_word * k.float()[:, :, None]

        invalid_ranks = ranks.masked_fill(selection_f[None] == 0, float("inf"))
        _, sorted_idx = torch.sort(invalid_ranks, dim=2)
        pos = torch.arange(U, device=device)
        valid_sel = pos[None, None, :] < k[:, :, None]
        g_word = torch.zeros(R, B, U, device=device)
        g_word.scatter_(2, sorted_idx, valid_sel.float())

        # Broadcast the word decision onto its subwords, so the mask entering
        # attention is subword-shaped but constant within a word.
        z = torch.stack([scatter_word_to_tokens(z_word[r], word_ids) for r in range(R)])
        g = torch.stack([scatter_word_to_tokens(g_word[r], word_ids) for r in range(R)])

        # Training always uses the soft mask z (Eq. 3); the hard top-k mask g
        # is exposed for eval/inference callers (Section 4/5/6), not training.
        effective_attns = attn_f[None] * z

        ids_rep = ids[None].expand(R, B, L).reshape(R * B, L)
        attn_rep = effective_attns.reshape(R * B, L)

        tok = self.sent_encoder.token_embeddings(ids_rep, attn_rep)
        valid_rep = attn_f[None].expand(R, B, L).reshape(R * B, L)
        pred_rep = self.sent_encoder.pool(tok, attn_rep, valid_mask=valid_rep).view(R, B, -1)

        full_rep_exp = full_rep.unsqueeze(0).expand(R, B, -1)
        per_sample = 1.0 - F.cosine_similarity(pred_rep, full_rep_exp, dim=-1)
        loss = per_sample.mean()

        return z, g, loss
