# models.py
import torch, logging
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

EPS = 1e-8

logger = logging.getLogger(__name__)

class HardKumaSampler(nn.Module):
    """
    Differentiable HardKuma gate per token.
    We use Kumaraswamy reparameterization and clamp to (0,1).
    Log-space version.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def sample(alpha, beta, eps=1e-6, u_min=1e-4):
        # u in (u_min, 1-u_min)
        u = torch.rand_like(alpha)
        u = u.clamp(u_min, 1.0 - u_min)

        # log-space version:
        # t = (1 - u)^(1/beta)  = exp( (1/beta) * log(1-u) )
        log1m_u = torch.log1p(-u)                              # safe log(1-u)
        inv_beta = 1.0 / (beta + eps)
        t = torch.exp(inv_beta * log1m_u)                      # (1-u)^(1/beta)

        # x = (1 - t)^(1/alpha)  = exp( (1/alpha) * log(1 - t) )
        # clamp argument to log1p to avoid log(<=0) numerics
        one_minus_t = (1.0 - t).clamp(min=eps, max=1.0)
        log1m_t = torch.log(one_minus_t)
        inv_alpha = 1.0 / (alpha + eps)
        x = torch.exp(inv_alpha * log1m_t)

        # keep away from exact 0/1 to stabilize backward
        return x.clamp(eps, 1.0 - eps)


class Selector(nn.Module):
    """
    Tiny 1-layer conv + FF to produce (alpha, beta) per token from token embeddings.
    """
    def __init__(self, d_model, hidden=256):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, 2)  # alpha, beta
        self.softplus = nn.Softplus()
        
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, token_emb):  # [B, L, D]
        h = F.gelu(self.proj(token_emb))
        params = self.out(h)                 # [B, L, 2]
        alpha, beta = params[..., 0], params[..., 1]
        # map to (0, +inf)
        alpha = (self.softplus(alpha) + 1.0).clamp(1.0, 10.0)
        beta  = (self.softplus(beta)  + 1.0).clamp(1.0, 10.0)   
        return alpha, beta


class ComplementAdversary(nn.Module):
    """
    Learns a per-token weighting over the complement to form a 'best possible' complement view.
    Outputs a pooled representation that goes through the same view projection.
    """
    def __init__(self, d_model, hidden=256, temp=0.5):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, 1)
        )
        self.temp = temp

    def forward(self, token_emb, comp_mask):  # token_emb: [B,L,D], comp_mask: [B,L] in [0,1]
        # score tokens, masked softmax over complement positions only
        logits = self.scorer(token_emb).squeeze(-1)             # [B,L]
        logits = logits - (1.0 - comp_mask) * 1e9               # -inf where not complement
        w = F.softmax(logits / self.temp, dim=-1)               # [B,L]
        return w                                                # normalized weights


class RationaleSelectorModel(nn.Module):
    def __init__(self, encoder_name="sentence-transformers/all-MiniLM-L6-v2", proj_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        d_model = self.encoder.config.hidden_size
        self.selector = Selector(d_model)
        self.kuma = HardKumaSampler()

        self.proj_anchor = nn.Sequential(nn.Linear(d_model, proj_dim), nn.GELU(),
                                         nn.Linear(proj_dim, proj_dim))
        self.proj_view   = nn.Sequential(nn.Linear(d_model, proj_dim), nn.GELU(),
                                         nn.Linear(proj_dim, proj_dim))

        # NEW: adversarial complement predictor
        self.comp_adv = ComplementAdversary(d_model, hidden=256, temp=0.5)

    @staticmethod
    def weighted_mean_pool(last_hidden, attn_weights):  # [B,L,D], [B,L]
        w = attn_weights.unsqueeze(-1)
        s = (last_hidden * w).sum(dim=1)
        z = w.sum(dim=1).clamp_min(EPS)
        return s / z

    def build_base_views(self, embeddings, attention_mask, gates):
        h_anchor = self.weighted_mean_pool(embeddings, attention_mask.float())
        h_rat    = self.weighted_mean_pool(embeddings, attention_mask.float() * gates)
        h_comp   = self.weighted_mean_pool(embeddings, attention_mask.float() * (1.0 - gates))

        h_anchor = F.normalize(self.proj_anchor(h_anchor), dim=-1, eps=1e-6)
        h_rat    = F.normalize(self.proj_view(h_rat),     dim=-1, eps=1e-6)
        h_comp   = F.normalize(self.proj_view(h_comp),    dim=-1, eps=1e-6)
        return h_anchor, h_rat, h_comp

    def forward(self, embeddings, attention_mask, return_details=False):
        # selector -> gates
        alpha, beta = self.selector(embeddings)
        gates = self.kuma.sample(alpha, beta).clamp(1e-6, 1.0 - 1e-6)

        # base views
        h_anchor, h_rat, h_comp = self.build_base_views(embeddings, attention_mask, gates)

        # adversarial complement: learn a 'best' complement pooling
        comp_mask = attention_mask.float() * (1.0 - gates)
        w_adv = self.comp_adv(embeddings, comp_mask)                  # [B,L]
        h_c_adv = self.weighted_mean_pool(embeddings, w_adv)          # [B,D]
        h_c_adv = F.normalize(self.proj_view(h_c_adv), dim=-1, eps=1e-6)

        out = {
            "h_anchor": h_anchor, "h_rat": h_rat,
            "h_comp": h_comp, "h_c_adv": h_c_adv,
            "gates": gates, "alpha": alpha, "beta": beta
        }
        return out if return_details else out


def nt_xent(anchor, positive, temperature=0.07):
    """
    InfoNCE with in-batch negatives: anchors vs. positives (one-to-one).
    """
    temperature = max(float(temperature), 1e-3)
    B = anchor.size(0)
    logits = anchor @ positive.t() / temperature              # [B,B]
    labels = torch.arange(B, device=anchor.device)
    return F.cross_entropy(logits, labels)
