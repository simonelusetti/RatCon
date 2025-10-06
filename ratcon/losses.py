# losses.py
import torch
import torch.nn.functional as F

from .models import nt_xent

def complement_margin_loss(h_anchor, h_comp, margin=0.3):
    # we want cosine(anchor, comp) to be LOW -> (1 - cos) to be HIGH
    cos = (h_anchor * h_comp).sum(dim=-1)            # [B]
    neg = 1.0 - cos
    return torch.relu(margin - neg).mean()

def sparsity_loss(gates, mask):
    # average gate value over *valid* tokens only (mask==1)
    valid = (mask > 0).float()
    return (gates * valid).sum() / (valid.sum() + 1e-8)

def total_variation_1d(gates, mask):
    # penalize changes across adjacent valid tokens
    valid = (mask > 0).float()
    diff = torch.abs(gates[:, 1:] - gates[:, :-1])
    # only count where both tokens are valid
    both = valid[:, 1:] * valid[:, :-1]
    return (diff * both).sum() / (both.sum() + 1e-8)


def complement_loss(h_comp, h_anchor, null_vec=None, use_null_target=False, temperature=0.07):
    """Push complements toward the null embedding or repel them from anchors."""
    if use_null_target and null_vec is not None:
        return (h_comp - null_vec).pow(2).mean()
    return -nt_xent(h_comp, h_anchor, temperature=temperature)


def kumaraswamy_log_pdf(x, alpha, beta, eps=1e-6):
    """Log-PDF of a Kumaraswamy(a, b) evaluated at x in (0, 1)."""
    x = torch.clamp(x, eps, 1.0 - eps)
    alpha = torch.clamp(alpha, eps)
    beta = torch.clamp(beta, eps)
    x_pow_alpha = torch.clamp(torch.pow(x, alpha), eps, 1.0 - eps)
    return (
        torch.log(alpha)
        + torch.log(beta)
        + (alpha - 1.0) * torch.log(x)
        + (beta - 1.0) * torch.log1p(-x_pow_alpha)
    )

def kl_loss(g1, g2, alpha1, beta1, alpha2, beta2):
    g1_soft = torch.clamp(g1, 1e-6, 1 - 1e-6)
    g2_soft = torch.clamp(g2, 1e-6, 1 - 1e-6)
    g1_comp = torch.clamp(1.0 - g1, 1e-6, 1 - 1e-6)
    g2_comp = torch.clamp(1.0 - g2, 1e-6, 1 - 1e-6)

    # KL(K(a1,b1) || distribution of 1 - g2)
    log_p1 = kumaraswamy_log_pdf(g1_soft, alpha1, beta1)
    log_q1 = kumaraswamy_log_pdf(g1_comp, alpha2, beta2)
    kl_1_2 = (log_p1 - log_q1).sum(dim=1).mean()

    # KL(K(a2,b2) || distribution of 1 - g1)
    log_p2 = kumaraswamy_log_pdf(g2_soft, alpha2, beta2)
    log_q2 = kumaraswamy_log_pdf(g2_comp, alpha1, beta1)
    kl_2_1 = (log_p2 - log_q2).sum(dim=1).mean()

    return 0.5 * (kl_1_2 + kl_2_1)