# losses.py
import torch
import torch.nn.functional as F
from typing import Sequence

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


def mutual_information_penalty(g1, g2, mask, eps=1e-8):
    """
    Approximate mutual information between gated token selections from two models.
    Treat each token as a Bernoulli variable with success probability equal to the soft gate.
    """
    valid = (mask > 0).float()
    valid_counts = valid.sum(dim=1).clamp_min(1.0)

    g1 = torch.clamp(g1, eps, 1.0 - eps)
    g2 = torch.clamp(g2, eps, 1.0 - eps)

    p11 = ((g1 * g2) * valid).sum(dim=1) / valid_counts
    p10 = ((g1 * (1.0 - g2)) * valid).sum(dim=1) / valid_counts
    p01 = (((1.0 - g1) * g2) * valid).sum(dim=1) / valid_counts
    p00 = (((1.0 - g1) * (1.0 - g2)) * valid).sum(dim=1) / valid_counts

    p1 = (g1 * valid).sum(dim=1) / valid_counts
    p0 = 1.0 - p1
    q1 = (g2 * valid).sum(dim=1) / valid_counts
    q0 = 1.0 - q1

    def _term(joint, marg1, marg2):
        joint_clamped = joint.clamp_min(eps)
        denom = (marg1 * marg2).clamp_min(eps)
        return joint_clamped * (torch.log(joint_clamped) - torch.log(denom))

    mi = (
        _term(p11, p1, q1)
        + _term(p10, p1, q0)
        + _term(p01, p0, q1)
        + _term(p00, p0, q0)
    )
    return mi.mean()


def _compute_shared_embeddings(models, outputs, attention_mask):
    shared_rationales = []
    shared_complements = []
    valid_mask = attention_mask

    shared_gate_product = torch.ones_like(outputs[0]["gates"])
    for out in outputs:
        shared_gate_product = shared_gate_product * (1.0 - out["gates"])

    shared_gate = torch.clamp(1.0 - shared_gate_product, 1e-6, 1.0)
    shared_mask = valid_mask * shared_gate

    shared_comp_gate = torch.clamp(shared_gate_product, 1e-6, 1.0)
    shared_comp_mask = valid_mask * shared_comp_gate

    for model, out in zip(models, outputs):
        pooled_rat = model.pooler({
            "token_embeddings": out["token_embeddings"],
            "attention_mask": shared_mask,
        })["sentence_embedding"]
        pooled_comp = model.pooler({
            "token_embeddings": out["token_embeddings"],
            "attention_mask": shared_comp_mask,
        })["sentence_embedding"]
        shared_rationales.append(pooled_rat)
        shared_complements.append(pooled_comp)

    return shared_rationales, shared_complements


def compute_training_objectives(
    models: Sequence,
    outputs,
    attention_mask,
    model_cfg,
    sparsity_weights,
    *,
    temperature,
    use_null_target,
    use_kl,
    kl_weight,
    use_mutual_info,
    mutual_info_weight,
    overlap_threshold,
):
    """Compute total loss and auxiliary statistics for multi-model training."""
    anchors = [out["h_anchor"] for out in outputs]
    device = anchors[0].device
    loss = anchors[0].new_zeros(())

    shared_rats, shared_comps = _compute_shared_embeddings(models, outputs, attention_mask)

    loss_cfg = model_cfg.loss
    l_comp_weight = float(loss_cfg.l_comp)
    l_tv_weight = float(loss_cfg.l_tv)

    for idx, out in enumerate(outputs):
        gates = out["gates"]
        null_vec = out["null"] if use_null_target else None

        l_rat = nt_xent(shared_rats[idx], anchors[idx], temperature=temperature)
        l_comp = complement_loss(shared_comps[idx], anchors[idx], null_vec, use_null_target, temperature)
        l_s = sparsity_loss(gates, attention_mask)
        l_tv = total_variation_1d(gates, attention_mask)

        loss = loss + l_rat
        loss = loss + l_comp_weight * l_comp
        loss = loss + sparsity_weights[idx] * l_s
        loss = loss + l_tv_weight * l_tv

    kl_total = anchors[0].new_zeros(())
    kl_pairs = 0
    if use_kl and len(outputs) > 1:
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                out_i = outputs[i]
                out_j = outputs[j]
                kl_val = kl_loss(
                    out_i["gates"],
                    out_j["gates"],
                    out_i["alpha"],
                    out_i["beta"],
                    out_j["alpha"],
                    out_j["beta"],
                )
                kl_total = kl_total + kl_val
                kl_pairs += 1

    avg_kl = None
    if kl_pairs > 0:
        avg_kl = kl_total / kl_pairs
        if kl_weight != 0.0:
            loss = loss + kl_weight * avg_kl

    mi_total = anchors[0].new_zeros(())
    mi_pairs = 0
    if use_mutual_info and len(outputs) > 1:
        mask_float = attention_mask.to(dtype=anchors[0].dtype)
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                mi_val = mutual_information_penalty(
                    outputs[i]["gates"],
                    outputs[j]["gates"],
                    mask_float,
                )
                mi_total = mi_total + mi_val
                mi_pairs += 1

    avg_mi = None
    if mi_pairs > 0:
        avg_mi = mi_total / mi_pairs
        if mutual_info_weight != 0.0:
            loss = loss + mutual_info_weight * avg_mi

    overlap_fraction = None
    if len(outputs) > 1:
        with torch.no_grad():
            valid_mask = (attention_mask > 0).float()
            valid_counts = valid_mask.sum(dim=1).clamp_min(1.0)
            gate_stack = torch.stack([out["gates"].detach() for out in outputs], dim=0)
            selected_counts = (gate_stack >= overlap_threshold).float().sum(dim=0)
            multi_mask = (selected_counts >= 2).float()
            per_example_overlap = (multi_mask * valid_mask).sum(dim=1) / valid_counts
            if torch.isfinite(per_example_overlap).any():
                overlap_fraction = float(per_example_overlap.mean().item())

    return loss, avg_kl, avg_mi, overlap_fraction
