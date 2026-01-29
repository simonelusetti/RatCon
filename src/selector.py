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

    def forward(
        self,
        embeddings: torch.Tensor,
        attn: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        emb = embeddings * attn.unsqueeze(-1)
        scores = self.selector(emb)
        scores = scores.masked_fill(attn == 0, -1e9)

        z = entmax15(scores / self.tau, dim=1) * attn

        eff = scores + self.coupling_k * (2.0 * z - 1.0)
        a = F.softplus(eff) + self.eps
        b = F.softplus(-eff) + self.eps

        x = kuma_sample(a, b, self.eps)
        y = self.l + (self.r - self.l) * x
        h = (y.clamp(0.0, 1.0) > 0.5).float() * attn

        g = h.detach() - z.detach() + z
        reg = scores.new_zeros(())

        return z, g, reg
