import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# -----------------------------------------------------------------------------
# Aliases / defaults
# -----------------------------------------------------------------------------

ALIASES = {
    # Plain pretrained BERT: a *token* encoder with no sentence-level
    # training on top, unlike sbert/e5 whose weights were tuned to make one
    # particular pooling produce good sentence embeddings. That makes it the
    # control for the pooling experiments -- vary data.encoder.pooling over
    # this family and nothing about the representation is co-adapted to any
    # of the strategies being compared.
    "bert": {"bert", "bert-base"},
    # ELECTRA and RoBERTa are the other raw token encoders -- pretrained
    # stacks with no sentence-level objective on top, so like `bert` they
    # privilege no pooling strategy and can serve as controls.
    #   electra: same architecture AND the same WordPiece vocab as bert, so
    #     its subword segmentation is byte-identical. The bias test counts
    #     subwords per tag, so bert and electra share a null distribution and
    #     their z-scores compare cell for cell. What differs is the
    #     pretraining objective (replaced-token detection, not MLM).
    #   roberta: same architecture, but BPE with a 50k vocab -- it segments
    #     differently, so its per-tag token counts and therefore its null
    #     differ. Still interpretable, but it varies tokenisation as well as
    #     pretraining, and is not directly comparable at the token level.
    "electra": {"electra"},
    "roberta": {"roberta"},
    "sbert": {"sbert"},
    "e5": {"e5", "retrieval", "gte"},
    "llm": {"llm"},
}

ALIAS_TO_CANON = {
    alias: canon
    for canon, aliases in ALIASES.items()
    for alias in aliases
}

DEFAULT_MODEL_NAMES = {
    "bert": "bert-base-uncased",
    "electra": "google/electra-base-discriminator",
    "roberta": "roberta-base",
    "sbert": "sentence-transformers/all-MiniLM-L6-v2",
    "e5": "intfloat/e5-base-v2",
    "llm": "EleutherAI/pythia-410m",
}

TOKENIZER_GROUPS = {
    # electra sits in the bert-base group because its tokenizer is the same
    # WordPiece vocabulary, verified to produce identical ids; roberta needs
    # its own because BPE does not.
    "bert-base": {"bert", "bert-base", "electra", "sbert", "e5", "retrieval", "gte"},
    "roberta": {"roberta"},
    "gpt": {"llm"},
}

CANONICAL_TOKENIZERS = {
    "bert-base": "bert-base-uncased",
    "roberta": "roberta-base",
    "gpt": "EleutherAI/pythia-410m",
}

# Extra from_pretrained kwargs, per tokenizer group. Every dataset here
# arrives as a list of words and is encoded with is_split_into_words=True
# (see encode_examples in src/data.py); RoBERTa's byte-level BPE refuses
# pre-tokenized input unless it is told that each word starts on a space
# boundary. Deliberately NOT applied to the gpt group: pythia's tokenizer
# accepts word lists as-is, and setting the flag there would change its
# segmentation and silently invalidate every existing llm run.
TOKENIZER_KWARGS = {
    "roberta": {"add_prefix_space": True},
}


def resolve_tokenizer_group(family: str) -> str:
    family = family.lower()
    for group, families in TOKENIZER_GROUPS.items():
        if family in families:
            return group
    raise ValueError(f"Unknown encoder family: {family}")


def resolve_tokenizer(family: str) -> AutoTokenizer:
    group = resolve_tokenizer_group(family)
    tokenizer = AutoTokenizer.from_pretrained(
        CANONICAL_TOKENIZERS[group], use_fast=True, **TOKENIZER_KWARGS.get(group, {})
    )

    if tokenizer.pad_token is None:
        fallback = tokenizer.eos_token or tokenizer.bos_token or tokenizer.unk_token
        if fallback is None:
            raise ValueError(
                f"Tokenizer {tokenizer.name_or_path} has no pad/eos/bos/unk token available for padding."
            )
        tokenizer.pad_token = fallback

    tokenizer.padding_side = "right"
    return tokenizer


# -----------------------------------------------------------------------------
# Token embedding backends
# -----------------------------------------------------------------------------

def _sdpa_context(device: torch.device):
    """Pin scaled_dot_product_attention to its reference kernel on CUDA.

    The selector's whole mechanism depends on gradients flowing back through
    the *attention mask* (the soft mask z enters as `attention_mask`), so this
    attention runs with autograd enabled rather than under no_grad. CUDA's
    memory-efficient backward kernel rejects the resulting broadcast mask --
    `RuntimeError: LSE is not correctly aligned (strideH)` -- because the
    bias is [B, 1, 1, T] with a zero stride over the head dimension.

    SDPBackend.MATH is the reference implementation, so this is a kernel
    choice and not an approximation: same values, more memory. Left untouched
    on CPU, which has no such restriction and is the path every result in the
    paper was produced on.

    Only applied when gradients are actually being tracked. The defect is in
    the memory-efficient *backward* kernel, so under torch.no_grad -- every
    evaluation path, and the whole of the brute-force oracle search -- the
    fast kernels are both correct and dramatically quicker. Forcing MATH
    there would buy nothing and cost a large multiple in throughput on
    workloads that are pure forward passes.
    """
    if device.type != "cuda" or not torch.is_grad_enabled():
        return contextlib.nullcontext()
    return sdpa_kernel([SDPBackend.MATH])


def bert_token_embeddings(
    model: AutoModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Returns last hidden states [B, T, D] for BERT-style encoders.
    Adds log(attention_mask) to attention scores, supporting fractional weights.
    """
    hidden_states = model.embeddings(input_ids)
    key_mask = attention_mask[:, None, None, :].type_as(hidden_states)  # [B,1,1,T]

    for layer in model.encoder.layer:
        attn = layer.attention.self
        bsz, seq_len, _ = hidden_states.size()

        q = attn.query(hidden_states)
        k = attn.key(hidden_states)
        v = attn.value(hidden_states)

        q = q.view(bsz, seq_len, attn.num_attention_heads, attn.attention_head_size).transpose(1, 2)
        k = k.view(bsz, seq_len, attn.num_attention_heads, attn.attention_head_size).transpose(1, 2)
        v = v.view(bsz, seq_len, attn.num_attention_heads, attn.attention_head_size).transpose(1, 2)

        attn_bias = torch.log(key_mask.clamp(min=1e-9))  # 0 for valid, ≈-inf for masked
        with _sdpa_context(hidden_states.device):
            context = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)

        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, attn.all_head_size)
        attn_out = layer.attention.output(context, hidden_states)
        hidden_states = layer.output(layer.intermediate(attn_out), attn_out)

    return hidden_states


def gpt_token_embeddings(
    model: AutoModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Returns last hidden states [B, T, D] for GPT-style encoders.
    HuggingFace causal models compute the additive bias as (1 - mask) * -inf,
    so values outside [0, 1] produce incorrect biases (+inf for mask > 1).
    Clamping guards against soft z values that can exceed 1.
    """
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask.clamp(0.0, 1.0),
    ).last_hidden_state


# -----------------------------------------------------------------------------
# Pooling strategies
# -----------------------------------------------------------------------------
#
# A pooling strategy is the second half of an experiment's identity: an
# experiment is a (token encoder, pooling strategy) pair. The token encoder
# decides what each token's hidden state is; the strategy decides how a set
# of those states becomes one sentence vector. Holding the encoder fixed and
# varying the strategy is what isolates whether a token-selection bias is a
# property of the representation or of the reduction applied to it.

POOLING_STRATEGIES = ("mean", "max", "min", "last")

def masked_pool(
    token_emb: torch.Tensor,
    pool_mask: torch.Tensor,
    strategy: str,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce token states [B, L, D] to one vector per sequence [B, D].

    mean: weighted average. min/max: scale each token vector by its score,
    then take the coordinate-wise extremum across the scaled vectors. A
    zero-scored token therefore contributes a zero *vector* that competes in
    the extremum like any other. Scores are not clipped -- the selector
    renormalises z to sum to k, so they can exceed 1, and leaving them
    unclipped is safe because the reduction is scale-equivariant
    (amax(a*x) == a*amax(x) for a > 0) and the reconstruction loss is a
    cosine: inflating every score uniformly cannot move it, so only the
    relative pattern of scores -- actual selection -- carries signal.
    last: last positive-score position; a hard index, so it passes no
    gradient to the scores at all and the selector trains through the
    attention path alone. Empty selections produce zeros.

    valid_mask marks real tokens so batch padding stays out of the extrema.
    It only bites when nothing is deselected (pool_full, or rho = 1): once
    any real token scores zero, its zero vector is already in the pool and
    padding's adds nothing. Callers working on padding-free rows -- the
    oracle's per-(n, k) groups, mask_optimality's trimmed sentences -- can
    omit it. See tests/test_pooling.py, which pins all of this.
    """
    if strategy not in POOLING_STRATEGIES:
        raise ValueError(
            f"Unknown pooling strategy {strategy!r}; expected one of {', '.join(POOLING_STRATEGIES)}."
        )

    mask = pool_mask.unsqueeze(-1).type_as(token_emb)

    if strategy == "mean":
        return (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    if strategy in ("max", "min"):
        weighted = token_emb * mask
        if valid_mask is not None:
            valid = valid_mask.bool().unsqueeze(-1)
            weighted = weighted.masked_fill(~valid, -torch.inf if strategy == "max" else torch.inf)
        pooled = weighted.amax(dim=1) if strategy == "max" else weighted.amin(dim=1)
        if valid_mask is not None:
            pooled = torch.where(valid.any(dim=1), pooled, torch.zeros_like(pooled))
        return pooled

    # strategy == "last"
    occupied = mask.sum(dim=1) > 0
    positions = torch.arange(token_emb.shape[1], device=token_emb.device)
    selected = (mask.squeeze(-1) > 0).to(token_emb.dtype)
    last_idx = (selected * (positions + 1).to(token_emb.dtype)).max(dim=1).values.long() - 1
    pooled = token_emb[torch.arange(token_emb.shape[0], device=token_emb.device), last_idx.clamp(min=0)]
    return torch.where(occupied, pooled, torch.zeros_like(pooled))


# -----------------------------------------------------------------------------
# Encoder interface: token_embeddings (expensive, once) + pool (cheap, many times)
# -----------------------------------------------------------------------------

class SentenceEncoder(nn.Module):
    """
    - token_embeddings(ids, attn) runs the transformer with the supplied mask.
    - pool(token_emb, pool_mask, valid_mask) computes sentence repr for an arbitrary
      pool_mask. pool_mask is where your selector's g lives (e.g.,
      pool_mask = attn * g). valid_mask is the original padding mask, not g.
    - pool_full(token_emb, attn_mask) computes the whole-sentence embedding
      (the *entire* real sequence, not a subset) -- the reconstruction target.

    Both use the SAME configured strategy (data.encoder.pooling). That is
    deliberate: the selector is trained to reproduce pool_full's output from
    a subset via pool, so if the two used different reductions the selector
    would be learning a cross-operator mapping rather than answering "which
    tokens suffice under this pooling strategy". Keeping them identical is
    what makes results comparable across strategies.
    """
    def __init__(self, normalize: bool, pooling: str = "mean") -> None:
        super().__init__()
        self.normalize = normalize
        if pooling not in POOLING_STRATEGIES:
            raise ValueError(
                f"Unknown pooling strategy {pooling!r}; expected one of {', '.join(POOLING_STRATEGIES)}."
            )
        self.pooling = pooling

    def token_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def pool(
        self, token_emb: torch.Tensor, pool_mask: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sent_emb = masked_pool(token_emb, pool_mask, self.pooling, valid_mask)
        return F.normalize(sent_emb, dim=-1) if self.normalize else sent_emb

    def pool_full(self, token_emb: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        return self.pool(token_emb, attn_mask, valid_mask=attn_mask)


# -----------------------------------------------------------------------------
# Concrete encoders
# -----------------------------------------------------------------------------

class FrozenSBERT(SentenceEncoder):
    def __init__(self, model_name: str, normalize: bool, pooling: str = "mean") -> None:
        super().__init__(normalize, pooling)
        self.model = SentenceTransformer(model_name)
        self.backbone = self.model[0].auto_model

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def token_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return bert_token_embeddings(self.backbone, input_ids, attention_mask)


class _FrozenHFEncoder(SentenceEncoder):
    def __init__(self, model_name: str, normalize: bool, pooling: str = "mean") -> None:
        super().__init__(normalize, pooling)
        self.model = AutoModel.from_pretrained(model_name)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False


class _BertStackEncoder(_FrozenHFEncoder):
    """Any BERT-architecture stack loaded through AutoModel.

    Shared by the raw `bert` family and by `e5`: they differ in weights and
    in E5's "query: " input prefix (applied in src/data.py at tokenisation
    time), not in how token states are produced.
    """

    def token_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return bert_token_embeddings(self.model, input_ids, attention_mask)


class FrozenBERT(_BertStackEncoder):
    pass


class FrozenELECTRA(_BertStackEncoder):
    pass


class FrozenRoBERTa(_BertStackEncoder):
    pass


class FrozenE5(_BertStackEncoder):
    pass


class FrozenLLMEncoder(_FrozenHFEncoder):
    """Causal encoder.

    Note on `pooling=last`: causal attention means only the final real token
    has attended to the whole sequence, which is why last-token extraction is
    the standard way to embed a causal LM's sentence (GritLM / LLM2Vec /
    PromptEOL). That convention now lives in the shared `last` strategy
    rather than being hardcoded here, so it applies to whichever encoder is
    configured with it -- and so this encoder can equally be run under mean /
    max / min, which is the point of making pooling a free axis. Assumes
    right-padding, as this repo's tokenizers are configured.
    """

    def token_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return gpt_token_embeddings(self.model, input_ids, attention_mask)


# -----------------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------------

# One entry per canonical family, same keys as DEFAULT_MODEL_NAMES. A table
# rather than a branch chain so that adding an encoder is three dict entries
# and (usually) a `pass`-bodied subclass of an existing stack.
ENCODER_CLASSES = {
    "bert": FrozenBERT,
    "electra": FrozenELECTRA,
    "roberta": FrozenRoBERTa,
    "sbert": FrozenSBERT,
    "e5": FrozenE5,
    "llm": FrozenLLMEncoder,
}


def build_sentence_encoder(
    family: str,
    encoder_name: str | None,
    device: str | None = None,
    pooling: str = "mean",
) -> tuple[SentenceEncoder, AutoTokenizer]:
    """Build the (token encoder, pooling strategy) pair an experiment is defined by.

    `family` picks the token encoder, `pooling` the reduction applied to its
    token states (see POOLING_STRATEGIES). The two are independent by design:
    the research question is whether a selection bias follows the encoder or
    the pooling, which is only answerable if either can be varied alone.
    """
    family = ALIAS_TO_CANON[family.lower()]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if encoder_name in {None, "None", "null", "NULL"}:
        encoder_name = DEFAULT_MODEL_NAMES[family]

    tokenizer = resolve_tokenizer(family)

    encoder = ENCODER_CLASSES[family](encoder_name, normalize=False, pooling=pooling)
    encoder.to(device)
    encoder.eval()
    return encoder, tokenizer
