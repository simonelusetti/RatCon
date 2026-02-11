import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import linspace

from src.losses import recon_loss
from src.sentence import SentenceEncoder
from entmax import entmax15


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


def sample_gumbel(shape, device, eps: float = 1e-6) -> torch.Tensor:
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)


def stochastic_topk(
    scores: torch.Tensor,
    attn: torch.Tensor,
    k: torch.Tensor,
    tau: float,
    n_samples: int,
):
    B, T = scores.shape
    device = scores.device

    probs = torch.zeros_like(scores)

    for _ in range(n_samples):
        noise = sample_gumbel(scores.shape, device)
        perturbed = (scores + noise * tau).masked_fill(attn == 0, -1e9)

        h = torch.zeros_like(scores)

        for i in range(B):
            ki = int(k[i].item())
            _, idx = perturbed[i].topk(ki)
            h[i, idx] = 1.0

        probs += h

    probs = probs / n_samples
    return probs


def _normalize_over_valid(x: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    x = x * attn
    return x / x.sum(dim=1, keepdim=True).clamp(min=1e-12)


class RationaleSelectorModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden: int | None = None,
        dropout: float = 0.1,
        sent_encoder: SentenceEncoder | None = None,
        loss_cfg: dict | None = None,
        tau: float = 1.0,
        n_samples: int = 5,
    ) -> None:
        super().__init__()
        if hidden is None:
            hidden = 4 * embedding_dim // 3

        self.selector = SelectorMLP(embedding_dim, hidden, dropout)
        self.sent_encoder = sent_encoder
        self.loss_cfg = loss_cfg
        self.tau = float(tau)
        self.n_samples = n_samples

    def forward(
        self,
        ids: torch.Tensor,
        embeddings: torch.Tensor,
        attn: torch.Tensor,
    ):
        emb = embeddings * attn.unsqueeze(-1)
        scores = self.selector(emb)
        scores = scores.masked_fill(attn == 0, -1e9)

        with torch.no_grad():
            full_token_emb = self.sent_encoder.token_embeddings(ids, attn)
            full_rep = self.sent_encoder.pool(full_token_emb, attn)

        g_sweep = []
        loss_sweep = []
        rho_eff_sweep = []

        recon_sum = torch.zeros((), device=embeddings.device)
        start, end, steps = self.loss_cfg.sweep_range
        rhos = linspace(start, end, steps)

        T_eff = attn.sum(dim=1).float()

        for rho in rhos:
            k = torch.clamp((rho * T_eff).long(), min=1)

            g_hard = stochastic_topk(
                scores=scores,
                attn=attn,
                k=k,
                tau=self.tau,
                n_samples=self.n_samples,
            )

            g_soft = entmax15(scores / self.tau, dim=1) * attn
            g_soft = g_soft / g_soft.sum(dim=1, keepdim=True).clamp(min=1e-6)

            g = g_hard.detach() - g_soft.detach() + g_soft

            # -------------------------
            # Alignment diagnostics
            # -------------------------

            hard_mask = (g_hard > 0)
            soft_mask = (g_soft > 0)

            support_overlap = (hard_mask & soft_mask).sum(dim=1).float()
            support_overlap_mean = support_overlap.mean()

            mass_on_hard = (g_soft * hard_mask.float()).sum(dim=1)
            mass_on_hard_mean = mass_on_hard.mean()

            # -------------------------

            k_eff = g.sum(dim=1)
            rho_eff = k_eff / T_eff

            g_sweep.append(g.detach().cpu())
            rho_eff_sweep.append(rho_eff.detach())

            effective_attn = attn * g
            token_emb = self.sent_encoder.token_embeddings(ids, effective_attn)
            pred_rep = self.sent_encoder.pool(token_emb, effective_attn)

            l_r = recon_loss(pred_rep, full_rep)
            recon_sum = recon_sum + l_r
            loss_sweep.append(float(l_r.detach().item()))

        recon_avg = recon_sum / len(rhos)

        losses_log = {
            "recon": float(recon_avg.detach().item()),
            "total": float(recon_avg.detach().item()),
        }

        return g, g_sweep, recon_avg, losses_log, loss_sweep, rho_eff_sweep
