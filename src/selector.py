from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sentence import SentenceEncoder


# ------------------------------------------------------------
# Selector MLP
# ------------------------------------------------------------

class SelectorMLP(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, token_emb: torch.Tensor) -> torch.Tensor:
        x = self.ln(token_emb)
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
    gamma: float = 1.0,
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
    if gamma != 1.0:
        p = p ** gamma

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
    ) -> None:
        super().__init__()

        if hidden is None:
            hidden = 4 * embedding_dim // 3

        self.selector = SelectorMLP(embedding_dim, hidden, dropout)
        self.sent_encoder = sent_encoder
        self.loss_cfg = loss_cfg

        self.tau_rank = 0.5
        self.gamma_rank = 1.0
        self.tau_gate = 1.0

    def forward(
        self,
        ids: torch.Tensor,
        embeddings: torch.Tensor,
        attn: torch.Tensor,
        rhos: Sequence[float],
    ):
        param_dtype = next(self.parameters()).dtype
        if embeddings.dtype != param_dtype:
            embeddings = embeddings.to(param_dtype)

        device = embeddings.device

        emb = embeddings * attn.unsqueeze(-1)
        scores = self.selector(emb)
        scores = scores.masked_fill(attn == 0, 0.0)

        attn_f = attn.float()
        with torch.no_grad():
            full_rep = self.sent_encoder.pool(embeddings, attn_f)

        selection_f = attn_f
        L_eff = selection_f.sum(dim=1).float()

        ranks = soft_rank(
            scores,
            selection_f,
            tau=self.tau_rank,
            gamma=self.gamma_rank,
        )

        B, L = ids.shape
        R = len(rhos)

        rhos_t = torch.tensor(list(rhos), device=device, dtype=torch.float32)

        k = (rhos_t[:, None] * L_eff[None]).round().long()
        k = torch.where(L_eff[None] > 0, k.clamp(min=1), torch.zeros_like(k))

        gate_raw = torch.sigmoid(
            (k.float()[:, :, None] - ranks[None]) / self.tau_gate
        ) * selection_f[None]

        z = gate_raw / gate_raw.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        z = z * k.float()[:, :, None]

        invalid_ranks = ranks.masked_fill(attn == 0, float("inf"))
        _, sorted_idx = torch.sort(invalid_ranks, dim=1)
        pos = torch.arange(L, device=device)
        valid_sel = pos[None, None, :] < k[:, :, None]
        g = torch.zeros(R, B, L, device=device)
        g.scatter_(2, sorted_idx[None].expand(R, -1, -1), valid_sel.float())

        g_st = g + (z - z.detach())

        effective_attns = attn_f[None] * g_st
        ids_rep = ids[None].expand(R, B, L).reshape(R * B, L)
        attn_rep = effective_attns.reshape(R * B, L)

        all_token_emb = self.sent_encoder.token_embeddings(ids_rep, attn_rep)
        pred_rep = self.sent_encoder.pool(all_token_emb, attn_rep)
        pred_rep = pred_rep.view(R, B, -1)

        full_rep_exp = full_rep.unsqueeze(0).expand(R, B, -1)
        per_sample = 1.0 - F.cosine_similarity(pred_rep, full_rep_exp, dim=-1)
        loss = per_sample.mean()

        return z, g, loss