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


def compute_training_objectives(
    model,
    output,
    attention_mask,
    model_cfg,
    *,
    temperature,
    use_null_target,
):
    """Compute total loss for single-model training."""
    anchors = output["h_anchor"]
    null_vec = output["null"] if use_null_target else None
    gates = output["gates"]

    l_rat = nt_xent(output["h_rat"], anchors, temperature=temperature)
    l_comp = complement_loss(output["h_comp"], anchors, null_vec, use_null_target, temperature)
    l_s = sparsity_loss(gates, attention_mask)
    l_tv = total_variation_1d(gates, attention_mask)

    loss_cfg = model_cfg.loss
    loss = l_rat
    loss = loss + float(loss_cfg.l_comp) * l_comp
    loss = loss + float(loss_cfg.l_s) * l_s
    loss = loss + float(loss_cfg.l_tv) * l_tv

    return loss
