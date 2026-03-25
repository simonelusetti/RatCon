from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import linspace

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

    scores = scores.masked_fill(attn == 0, 0.0)

    mean = (scores * attn).sum(dim=1, keepdim=True) / \
           attn.sum(dim=1, keepdim=True).clamp(min=1.0)

    var = ((scores - mean) ** 2 * attn).sum(dim=1, keepdim=True) / \
          attn.sum(dim=1, keepdim=True).clamp(min=1.0)

    std = torch.sqrt(var + 1e-6)
    scores = (scores - mean) / std

    diff = scores.unsqueeze(1) - scores.unsqueeze(2)
    p = torch.sigmoid(diff / tau)

    p = p ** gamma

    p = p * attn.unsqueeze(1)

    r = 1.0 + p.sum(dim=1)
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

        self.tau_rank = 0.05
        self.gamma_rank = 2.0
        self.tau_gate = 0.2

    def forward(
        self,
        ids: torch.Tensor,
        embeddings: torch.Tensor,
        attn: torch.Tensor,
        rhos: Sequence[float],
        selection_mask: torch.Tensor | None = None,
    ):
        param_dtype = next(self.parameters()).dtype
        if embeddings.dtype != param_dtype:
            embeddings = embeddings.to(param_dtype)
        device = embeddings.device
        if selection_mask is None:
            selection_mask = attn

        emb = embeddings * selection_mask.unsqueeze(-1)
        scores = self.selector(emb)
        scores = scores.masked_fill(selection_mask == 0, 0.0)

        # Reuse the pre-computed token embeddings passed in — avoids one extra BERT forward.
        attn_f = attn.float()
        with torch.no_grad():
            full_rep = self.sent_encoder.pool(embeddings, attn_f)

        selection_f = selection_mask.float()
        T_eff = selection_f.sum(dim=1).float()

        ranks = soft_rank(
            scores,
            selection_f,
            tau=self.tau_rank,
            gamma=self.gamma_rank,
        )

        # ── Phase 1: vectorised gate computation (no Python loops) ──────────
        B, T = ids.shape
        R = len(rhos)

        rhos_t = torch.tensor(list(rhos), device=device, dtype=torch.float32)  # [R]

        # k_all: [R, B] — number of tokens to select per (rho, sample)
        k_all = (rhos_t[:, None] * T_eff[None]).round().long()
        k_all = torch.where(T_eff[None] > 0, k_all.clamp(min=1), torch.zeros_like(k_all))

        # Soft gates: [R, B, T]
        gate_raw = torch.sigmoid(
            (k_all.float()[:, :, None] - ranks[None]) / self.tau_gate
        ) * selection_f[None]
        g_soft = gate_raw / gate_raw.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        g_soft = g_soft * k_all.float()[:, :, None]

        # Hard masks via sorted ranks — no Python loops at all.
        # Assign inf to invalid positions so they sink to the end when sorted.
        invalid_ranks = ranks.masked_fill(selection_mask == 0, float("inf"))  # [B, T]
        _, sorted_idx = torch.sort(invalid_ranks, dim=1)                      # [B, T]
        pos = torch.arange(T, device=device)                                   # [T]
        valid_sel = pos[None, None, :] < k_all[:, :, None]                    # [R, B, T]
        hard_masks = torch.zeros(R, B, T, device=device)
        hard_masks.scatter_(2, sorted_idx[None].expand(R, -1, -1), valid_sel.float())

        g_st_all = hard_masks + (g_soft - g_soft.detach())                    # [R, B, T]

        k_eff = hard_masks.sum(dim=-1)                                        # [R, B]
        rho_eff_sweep = list((k_eff / T_eff[None].clamp(min=1.0)).detach().unbind(0))
        hard_mask_list = list(hard_masks.unbind(0))

        # ── Phase 2: one batched BERT forward for all rhos ───────────────────
        # Stack effective attention masks: [R, B, T] → [R*B, T]
        effective_attns = attn_f[None] * g_st_all                         # [R, B, T]
        ids_rep  = ids[None].expand(R, B, T).reshape(R * B, T)           # [R*B, T]
        attn_rep = effective_attns.reshape(R * B, T)                      # [R*B, T]

        all_token_emb = self.sent_encoder.token_embeddings(ids_rep, attn_rep)  # [R*B, T, D]
        all_pred_rep  = self.sent_encoder.pool(all_token_emb, attn_rep)        # [R*B, D]
        all_pred_rep  = all_pred_rep.view(R, B, -1)                            # [R, B, D]

        # ── Phase 3: vectorised reconstruction loss ───────────────────────────
        full_rep_exp = full_rep.unsqueeze(0).expand(R, B, -1)                  # [R, B, D]
        per_sample   = 1.0 - F.cosine_similarity(all_pred_rep, full_rep_exp, dim=-1)  # [R, B]
        recon_avg    = per_sample.mean()

        g_sweep    = [hm.detach().cpu() for hm in hard_mask_list]  # [R] × [B, T]
        loss_sweep = per_sample.mean(dim=1).tolist()

        losses_log = {
            "recon": float(recon_avg.detach().item()),
            "total": float(recon_avg.detach().item()),
        }

        return g_st_all[-1], g_sweep, recon_avg, losses_log, loss_sweep, rho_eff_sweep
