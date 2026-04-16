from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sentence import SentenceEncoder


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
        selector_cfg: dict | None = None,
    ) -> None:
        super().__init__()

        if hidden is None:
            hidden = 4 * embedding_dim // 3

        selector_cfg = selector_cfg or {}
        self.condition_on_rho = bool(selector_cfg.get("condition_on_rho", True))
        self.use_hard_for_reencode = bool(
            selector_cfg.get("use_hard", selector_cfg.get("use_hard_mask_for_reencode", False))
        )

        self.selector = SelectorMLP(
            embedding_dim,
            hidden,
            dropout,
            condition_on_rho=self.condition_on_rho,
        )
        self.sent_encoder = sent_encoder
        self.loss_cfg = loss_cfg

        self.tau_rank = float(selector_cfg.get("tau_rank", 0.05))
        self.gamma_rank = float(selector_cfg.get("gamma_rank", 2.0))
        self.tau_gate = float(selector_cfg.get("tau_gate", 0.2))

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

        attn_f = attn.float()
        with torch.no_grad():
            full_rep = self.sent_encoder.pool(embeddings, attn_f)

        selection_f = attn_f
        L_eff = selection_f.sum(dim=1).float()

        B, L = ids.shape
        R = len(rhos)

        rhos_t = torch.tensor(list(rhos), device=device, dtype=torch.float32)

        emb = embeddings * attn.unsqueeze(-1)
        if self.condition_on_rho:
            emb_rep = emb[None].expand(R, B, L, -1).reshape(R * B, L, emb.shape[-1])
            attn_rep_for_rank = attn_f[None].expand(R, B, L).reshape(R * B, L)
            rho_per_example = rhos_t[:, None].expand(R, B).reshape(R * B)

            scores = self.selector(emb_rep, rho_per_example)
            scores = scores.masked_fill(attn_rep_for_rank == 0, 0.0)

            ranks = soft_rank(
                scores,
                attn_rep_for_rank,
                tau=self.tau_rank,
                gamma=self.gamma_rank,
            ).view(R, B, L)
        else:
            scores = self.selector(emb)
            scores = scores.masked_fill(attn_f == 0, 0.0)
            shared_ranks = soft_rank(
                scores,
                attn_f,
                tau=self.tau_rank,
                gamma=self.gamma_rank,
            )
            ranks = shared_ranks.unsqueeze(0).expand(R, -1, -1)

        k = (rhos_t[:, None] * L_eff[None]).round().long()
        k = torch.where(L_eff[None] > 0, k.clamp(min=1), torch.zeros_like(k))

        gate_raw = torch.sigmoid(
            (k.float()[:, :, None] - ranks) / self.tau_gate
        ) * selection_f[None]

        z = gate_raw / gate_raw.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        z = z * k.float()[:, :, None]

        invalid_ranks = ranks.masked_fill(attn_f[None] == 0, float("inf"))
        _, sorted_idx = torch.sort(invalid_ranks, dim=2)
        pos = torch.arange(L, device=device)
        valid_sel = pos[None, None, :] < k[:, :, None]
        g = torch.zeros(R, B, L, device=device)
        g.scatter_(2, sorted_idx, valid_sel.float())

        g_st = g + (z - z.detach())

        if self.use_hard_for_reencode:
            effective_attns = attn_f[None] * g_st
        else:
            effective_attns = attn_f[None] * z
        ids_rep = ids[None].expand(R, B, L).reshape(R * B, L)
        attn_rep = effective_attns.reshape(R * B, L)

        def _reencode(ids_r, attn_r):
            with torch.autocast(ids_r.device.type, dtype=torch.bfloat16, enabled=True):
                tok = self.sent_encoder.token_embeddings(ids_r, attn_r)
                return self.sent_encoder.pool(tok, attn_r)

        pred_rep = _reencode(ids_rep, attn_rep)
        pred_rep = pred_rep.view(R, B, -1)

        full_rep_exp = full_rep.unsqueeze(0).expand(R, B, -1)
        per_sample = 1.0 - F.cosine_similarity(pred_rep, full_rep_exp, dim=-1)
        loss = per_sample.mean()

        return z, g, loss
