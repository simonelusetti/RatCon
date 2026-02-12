import torch
import torch.nn as nn
import torch.nn.functional as F
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


def kuma_sample(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    u = torch.rand_like(a).clamp(min=eps, max=1 - eps)
    return (1.0 - (1.0 - u).pow(1.0 / b)).pow(1.0 / a)


def top_k(z: torch.Tensor, attn: torch.Tensor, rho: float) -> torch.Tensor:
    z = z * attn
    T = attn.sum(dim=1).clamp(min=1).long()
    z_hard = torch.zeros_like(z)

    for i in range(z.size(0)):
        k = max(1, int(rho * T[i].item()))
        _, idx = z[i].topk(k)
        z_hard[i, idx] = 1.0

    return z_hard * attn


def sample_gumbel(shape, device, eps: float = 1e-6) -> torch.Tensor:
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)


def probabilistic_top_k(
    scores: torch.Tensor,
    attn: torch.Tensor,
    rho: float,
) -> torch.Tensor:

    scores = scores * attn
    B, T = scores.shape

    gumbel = sample_gumbel(scores.shape, scores.device)
    perturbed = scores + gumbel

    z_hard = torch.zeros_like(scores)
    T_eff = attn.sum(dim=1).long()

    for i in range(B):
        k = round(rho * T_eff[i].item())
        k = max(1, min(int(k), int(T_eff[i].item())))
        _, idx = perturbed[i].topk(k)
        z_hard[i, idx] = 1.0

    return z_hard * attn


def total_variation_1d(z: torch.Tensor, attn: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    z = z * attn
    dz = (z[:, 1:] - z[:, :-1]).abs()
    valid = (attn[:, 1:] * attn[:, :-1]).to(dz.dtype)
    tv_sum = (dz * valid).sum(dim=1)
    denom = valid.sum(dim=1).clamp(min=1.0)
    tv = (tv_sum / denom).mean()
    return tv


class RationaleSelectorModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden: int | None = None,
        dropout: float = 0.1,
        tau: float = 1.0,
        l: float = -0.1,
        r: float = 1.1,
        coupling_k: float = 2.0,
        eps: float = 1e-6,
        rho: float = 0.30,
        hard_type: str = "probabilistic_top_k",
        tv_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden is None:
            hidden = 4 * embedding_dim // 3

        self.selector = SelectorMLP(embedding_dim, hidden, dropout)
        self.tau = float(tau)
        self.l = float(l)
        self.r = float(r)
        self.coupling_k = float(coupling_k)
        self.eps = float(eps)
        self.rho = float(rho)
        self.hard_type = hard_type
        self.tv_weight = float(tv_weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        attn: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        emb = embeddings * attn.unsqueeze(-1)
        scores = self.selector(emb)
        scores = scores.masked_fill(attn == 0, -1e9)

        z = entmax15(scores / self.tau, dim=1) * attn

        if self.hard_type == "hard_kuma":
            eff = scores + self.coupling_k * (2.0 * z - 1.0)
            a = F.softplus(eff) + self.eps
            b = F.softplus(-eff) + self.eps
            x = kuma_sample(a, b, self.eps)
            y = self.l + (self.r - self.l) * x
            h = (y.clamp(0.0, 1.0) > 0.5).float() * attn
        elif self.hard_type == "top_k":
            h = top_k(z, attn, rho=self.rho)
        elif self.hard_type == "sweep":
            h = top_k(z, attn, rho=self.rho)
        elif self.hard_type == "probabilistic_top_k":
            h = probabilistic_top_k(
                scores=scores,
                attn=attn,
                rho=self.rho,
            )
        elif self.hard_type == "sweep":
            h = z
        elif self.hard_type == "none":
            gumbel = sample_gumbel(scores.shape, scores.device)
            h = (z > 0.0).float() * attn + (gumbel / self.tau).sigmoid() * attn
        else:
            raise ValueError(f"Unknown hard_type: {self.hard_type}")

        g = h.detach() - z.detach() + z

        reg = self.tv_weight * total_variation_1d(g, attn, eps=self.eps)

        return z, g, reg
