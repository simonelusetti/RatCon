import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15
from numpy import linspace

from src.losses import recon_loss
from src.sentence import SentenceEncoder


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


def hard_top_k_from_sorted(
    sorted_idx: torch.Tensor,
    attn: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    B, T = attn.shape
    T_eff = attn.sum(dim=1).long()
    h = torch.zeros_like(attn)

    for i in range(B):
        k = max(1, int(rho * T_eff[i].item()))
        idx = sorted_idx[i, :k]
        h[i, idx] = 1.0

    return h * attn


class RationaleSelectorModel(nn.Module):
    """
    OPTION 2 (strict, causal)
    Uses the NEW SentenceEncoder API:
      - token_embeddings(...)
      - pool(...)
    Recomputes embeddings for every g.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden: int | None = None,
        dropout: float = 0.1,
        sent_encoder: SentenceEncoder | None = None,
        loss_cfg: dict | None = None,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden is None:
            hidden = 4 * embedding_dim // 3

        self.selector = SelectorMLP(embedding_dim, hidden, dropout)
        self.sent_encoder = sent_encoder
        self.loss_cfg = loss_cfg
        self.tau = float(tau)

    def forward(
        self,
        ids: torch.Tensor,
        embeddings: torch.Tensor,
        attn: torch.Tensor,
    ):
        # ─────────────────────────
        # Selector (once)
        # ─────────────────────────
        emb = embeddings * attn.unsqueeze(-1)
        scores = self.selector(emb)
        scores = scores.masked_fill(attn == 0, -1e9)

        z = entmax15(scores / self.tau, dim=1) * attn

        sorted_idx = z.argsort(dim=1, descending=True)

        # ─────────────────────────
        # Full representation (baseline)
        # ─────────────────────────
        with torch.no_grad():
            full_token_emb = self.sent_encoder.token_embeddings(ids, attn)
            full_rep = self.sent_encoder.pool(full_token_emb, attn)

        # ─────────────────────────
        # Sweep (expensive but correct)
        # ─────────────────────────
        g_sweep = []
        loss_sweep = []

        recon_sum = torch.zeros((), device=full_rep.device)
        start, end, steps = self.loss_cfg.sweep_range
        rhos = linspace(start, end, steps)

        for rho in rhos:
            h = hard_top_k_from_sorted(sorted_idx, attn, rho=float(rho))

            # straight-through estimator
            g = h.detach() - z.detach() + z
            g_sweep.append(g.detach().cpu())

            effective_attn = attn * g

            with torch.no_grad():
                token_emb = self.sent_encoder.token_embeddings(ids, effective_attn)
                pred_rep = self.sent_encoder.pool(token_emb, effective_attn)

            l_r = recon_loss(pred_rep, full_rep)
            recon_sum = recon_sum + l_r
            loss_sweep.append(l_r.item())

        recon_avg = recon_sum / len(rhos)

        return (
            z,
            g_sweep,
            {
                "recon": recon_avg.item(),
                "total": recon_avg.item(),
            },
            loss_sweep,
        )
