import torch, torch.nn as nn

"""
def kuma_sample(alpha: torch.Tensor, beta: torch.Tensor, u_min: float = 1e-4, eps: float = 1e-6):
    u = torch.rand_like(alpha).clamp(u_min, 1.0 - u_min)
    inv_alpha = 1.0 / (alpha + eps)
    inv_beta = 1.0 / (beta + eps)
    inner = 1.0 - torch.pow(1.0 - u, inv_beta)
    z = torch.pow(inner.clamp(min=eps, max=1.0), inv_alpha)
    return z.clamp(eps, 1.0 - eps)


def kuma_mean(alpha: torch.Tensor, beta: torch.Tensor, eps: float = 1e-8):
    inv_alpha = 1.0 / (alpha + eps)
    log_mean = (
        torch.log(beta + eps)
        + torch.lgamma(1.0 + inv_alpha)
        + torch.lgamma(beta)
        - torch.lgamma(1.0 + beta + inv_alpha)
    )
    return torch.exp(log_mean).clamp(0.0, 1.0)"""


class Selector(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.out = nn.Linear(d_model, 1)

    def forward(self, token_emb: torch.Tensor) -> torch.Tensor:
        # [B, T, D] → [B, T]
        return self.out(token_emb).squeeze(-1)


class RationaleSelectorModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        rho: float = 0.2,   # fraction of available tokens
        tau: float = 1.0,   # softmax temperature
    ):
        super().__init__()
        self.selector = Selector(embedding_dim)
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
        """
        with torch.no_grad():
            p_detached = torch.softmax(scores / self.tau, dim=1)
            entropy = -(p_detached * (p_detached + 1e-12).log()).sum(dim=1)  # [B]
            T_eff = attn.sum(dim=1).clamp(min=1)
            max_entropy = T_eff.log()
            norm_entropy = entropy / max_entropy
            print(
                f"[Selector entropy] mean={norm_entropy.mean().item():.3f} std={norm_entropy.std().item():.3f} \
                    min={norm_entropy.min().item():.3f} max={norm_entropy.max().item():.3f}"
            )"""

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