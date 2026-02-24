from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import linspace

from src.losses import recon_loss
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
        rhos: Sequence[float] | None = None,
        sent_encoder: SentenceEncoder | None = None,
        loss_cfg: dict | None = None,
    ) -> None:
        super().__init__()

        if hidden is None:
            hidden = 4 * embedding_dim // 3

        self.selector = SelectorMLP(embedding_dim, hidden, dropout)
        self.sent_encoder = sent_encoder
        self.loss_cfg = loss_cfg
        self.rhos = rhos

        self.tau_rank = 0.05
        self.gamma_rank = 2.0
        self.tau_gate = 0.2

    def forward(
        self,
        ids: torch.Tensor,
        embeddings: torch.Tensor,
        attn: torch.Tensor,
    ):
        device = embeddings.device

        emb = embeddings * attn.unsqueeze(-1)
        scores = self.selector(emb)
        scores = scores.masked_fill(attn == 0, 0.0)

        with torch.no_grad():
            full_token_emb = self.sent_encoder.token_embeddings(ids, attn)
            full_rep = self.sent_encoder.pool(full_token_emb, attn)

        g_sweep = []           # 🔥 now HARD masks
        loss_sweep = []
        rho_eff_sweep = []

        recon_sum = torch.zeros((), device=device)

        start, end, steps = self.loss_cfg.sweep_range

        attn_f = attn.float()
        T_eff = attn_f.sum(dim=1).float()

        ranks = soft_rank(
            scores,
            attn_f,
            tau=self.tau_rank,
            gamma=self.gamma_rank,
        )

        for rho in self.rhos:

            k = torch.clamp((float(rho) * T_eff).round().long(), min=1)

            # ------------------------------------------------------------
            # SOFT GATE (for training / reconstruction)
            # ------------------------------------------------------------

            gate_raw = torch.sigmoid(
                (k.float().unsqueeze(1) - ranks) / self.tau_gate
            ) * attn_f

            g_soft = gate_raw / gate_raw.sum(dim=1, keepdim=True).clamp(min=1e-8)
            g_soft = g_soft * k.unsqueeze(1)

            # ------------------------------------------------------------
            # HARD TOP-K (for evaluation only)
            # ------------------------------------------------------------

            # Use ranks directly for stable top-k
            hard_mask = torch.zeros_like(g_soft)

            # smaller rank = better
            for b in range(ranks.size(0)):
                kb = int(k[b].item())
                valid_idx = attn[b].nonzero(as_tuple=False).squeeze(1)

                if valid_idx.numel() == 0:
                    continue

                # select top-k lowest ranks
                ranks_b = ranks[b, valid_idx]
                topk = torch.topk(-ranks_b, kb).indices  # negative for lowest
                selected = valid_idx[topk]

                hard_mask[b, selected] = 1.0

            # ------------------------------------------------------------

            k_eff = g_soft.sum(dim=1)
            rho_eff = k_eff / T_eff.clamp(min=1.0)

            g_sweep.append(hard_mask.detach().cpu())  # 🔥 HARD mask now
            rho_eff_sweep.append(rho_eff.detach())

            # reconstruction still uses SOFT gate
            effective_attn = attn_f * g_soft

            token_emb = self.sent_encoder.token_embeddings(ids, effective_attn)
            pred_rep = self.sent_encoder.pool(token_emb, effective_attn)

            l_r = recon_loss(pred_rep, full_rep)
            recon_sum = recon_sum + l_r
            loss_sweep.append(float(l_r.detach().item()))

        recon_avg = recon_sum / len(self.rhos)

        losses_log = {
            "recon": float(recon_avg.detach().item()),
            "total": float(recon_avg.detach().item()),
        }

        return g_soft, g_sweep, recon_avg, losses_log, loss_sweep, rho_eff_sweep
