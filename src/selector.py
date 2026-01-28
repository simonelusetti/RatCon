import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15


class SelectorMLP(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.1) -> None:
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


class RationaleSelectorModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        rho: float = 0.2,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.selector = SelectorMLP(
            embedding_dim,
            hidden=4 * embedding_dim // 3,
            dropout=0.1,
        )
        self.rho = rho
        self.tau = tau

    def _hard_topk(self, z: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        B, T = z.shape
        _, idx = torch.sort(z, dim=1, descending=True)
        ranks = torch.arange(T, device=z.device).unsqueeze(0)
        mask = ranks < K
        h = torch.zeros_like(z)
        h.scatter_(1, idx, mask.float())
        return h

    def forward(
        self,
        embeddings: torch.Tensor,
        attn: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        B, T, D = embeddings.shape

        emb = embeddings * attn.unsqueeze(-1)

        scores = self.selector(emb)
        scores = scores.masked_fill(attn == 0, -1e9)

        # sparse normalization
        p = entmax15(scores / self.tau, dim=1)

        T_eff = attn.sum(dim=1, keepdim=True)
        K = (self.rho * T_eff).round().clamp(min=1)

        z = K * p

        if deterministic:
            g = self._hard_topk(z, K)
        else:
            h = self._hard_topk(z, K)
            g = h + (z - z.detach())  # straight-through estimator

        return g, z
