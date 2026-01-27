import torch, torch.nn as nn, torch.nn.functional as F


class Selector(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.out = nn.Linear(d_model, 1)

    def forward(self, token_emb: torch.Tensor) -> torch.Tensor:
        # [B, T, D] → [B, T]
        return self.out(token_emb).squeeze(-1)


class SelectorMLP(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, token_emb: torch.Tensor) -> torch.Tensor:
        # token_emb: [B,T,D] -> scores: [B,T]
        x = self.ln(token_emb)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x).squeeze(-1)
        return x


class RationaleSelectorModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        rho: float = 0.2,   # fraction of available tokens
        tau: float = 1.0,   # softmax temperature
    ):
        super().__init__()
        #self.selector = Selector(embedding_dim)
        self.selector = SelectorMLP(embedding_dim, hidden=4*embedding_dim//3, dropout=0.1)
        self.rho = rho
        self.tau = tau
        
    def _hard_topk(self, z: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        z: [B, T]
        K: [B, 1]  (integer budget per sentence)
        returns h: [B, T] with exactly K[i] ones per row
        """

        B, T = z.shape

        # sort indices descending
        _, idx = torch.sort(z, dim=1, descending=True)   # [B, T]

        # build rank positions: 0,1,2,...
        ranks = torch.arange(T, device=z.device).unsqueeze(0)  # [1, T]

        # mask: rank < K
        mask = ranks < K                                  # [B, T]

        # scatter back to original positions
        h = torch.zeros_like(z)
        h.scatter_(1, idx, mask.float())

        return h

    def forward(
        self,
        embeddings: torch.Tensor,   # [B, T, D]
        attn: torch.Tensor,         # [B, T] (0/1)
        deterministic: bool = False
    ):
        B, T, D = embeddings.shape

        emb = embeddings * attn.unsqueeze(-1)
        scores = self.selector(emb)                     # [B, T]
        scores = scores.masked_fill(attn == 0, -1e9)

        p = torch.softmax(scores / self.tau, dim=1)     # sum p = 1
        entropy = -(p * (p + 1e-12).log()).sum(dim=1)  # [B]
        T_eff = attn.sum(dim=1).clamp(min=1)
        max_entropy = T_eff.log()
        norm_entropy = (entropy / max_entropy).mean()
        T_eff = attn.sum(dim=1, keepdim=True)            # [B, 1]
        K = (self.rho * T_eff).round().clamp(min=1)      # [B, 1]
        z = K * p                                        # sum z = K

        if deterministic:
            g = self._hard_topk(z, K)
        else:
            h = self._hard_topk(z, K)
            g = h + (z - z.detach())                     # ST estimator

        return g, z, norm_entropy