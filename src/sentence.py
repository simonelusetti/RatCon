from __future__ import annotations

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ALIASES = {
    "sbert": {"sbert"},
    "e5": {"e5", "retrieval", "gte"},
    "bge": {"bge", "late"},
    "llm": {"llm"}
}

ALIAS_TO_CANON = {
    alias: canon
    for canon, aliases in ALIASES.items()
    for alias in aliases
}

DEFAULT_MODEL_NAMES = {
    "sbert": "sentence-transformers/all-MiniLM-L6-v2",
    "e5": "intfloat/e5-base-v2",
    "bge": "BAAI/bge-base-en-v1.5",
    "llm": "EleutherAI/pythia-410m",
}

TOKENIZER_GROUPS = {
    "bert-base": {"sbert", "e5", "retrieval", "gte"},
    "bge": {"bge", "late"},
    "gpt": {"llm"},
}

CANONICAL_TOKENIZERS = {
    "bert-base": "bert-base-uncased",
    "bge": "BAAI/bge-base-en-v1.5",
    "gpt": "EleutherAI/pythia-410m",
}


# ---------------------------------------------------------------------
# Tokenizer resolution
# ---------------------------------------------------------------------

def resolve_tokenizer_group(family: str) -> str:
    family = family.lower()
    for group, families in TOKENIZER_GROUPS.items():
        if family in families:
            return group
    raise ValueError(f"Unknown encoder family: {family}")


def resolve_tokenizer(family: str) -> AutoTokenizer:
    group = resolve_tokenizer_group(family)
    return AutoTokenizer.from_pretrained(CANONICAL_TOKENIZERS[group], use_fast=True)


# ---------------------------------------------------------------------
# Base encoder
# ---------------------------------------------------------------------

class SentenceEncoder(nn.Module):
    def __init__(self, normalize: bool):
        super().__init__()
        self.normalize = normalize

    def token_embeddings(
        self,
        input_ids: torch.Tensor,       # [B, T]
        attention_mask: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:
        raise NotImplementedError()

    def encode(
        self,
        input_ids: torch.Tensor,       # [B, T]
        attention_mask: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:
        raise NotImplementedError()


# ---------------------------------------------------------------------
# SBERT
# ---------------------------------------------------------------------

class FrozenSBERT(SentenceEncoder):
    def __init__(self, model_name: str, normalize: bool, device: str):
        super().__init__(normalize)
        self.model = SentenceTransformer(model_name, device=device)
        self.transformer = self.model[0]

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def _first_fractional(attention_mask: torch.Tensor) -> tuple[int, int, float] | None:
        mask = attention_mask.detach()
        fractional = (mask > 0) & (mask < 1)
        coords = torch.nonzero(fractional, as_tuple=False)
        if coords.numel() == 0:
            return None
        b_idx, t_idx = coords[0].tolist()
        return int(b_idx), int(t_idx), float(mask[b_idx, t_idx].item())

    def token_embeddings(self, input_ids, attention_mask):
        model = self.transformer.auto_model
        hidden_states = model.embeddings(input_ids=input_ids)
        mask = attention_mask.to(dtype=hidden_states.dtype)
        key_mask = mask[:, None, None, :]
        debug_info = self._first_fractional(attention_mask)
        debug_logged = False

        for layer_idx, layer in enumerate(model.encoder.layer):
            attn = layer.attention.self
            bsz, seq_len, _ = hidden_states.size()

            query = attn.query(hidden_states)
            key = attn.key(hidden_states)
            value = attn.value(hidden_states)

            query = query.view(bsz, seq_len, attn.num_attention_heads, attn.attention_head_size).transpose(1, 2)
            key = key.view(bsz, seq_len, attn.num_attention_heads, attn.attention_head_size).transpose(1, 2)
            value = value.view(bsz, seq_len, attn.num_attention_heads, attn.attention_head_size).transpose(1, 2)

            scores = torch.matmul(query, key.transpose(-1, -2))
            scores = scores / math.sqrt(attn.attention_head_size)
            probs = torch.softmax(scores, dim=-1)
            probs = attn.dropout(probs)

            """
            if debug_info is not None and not debug_logged and layer_idx == 0:
                b_idx, t_idx, val = debug_info
                probs_masked = probs * key_mask
                col_raw = probs[b_idx, 0, :, t_idx].sum().item()
                col_masked = probs_masked[b_idx, 0, :, t_idx].sum().item()
                row_raw = probs[b_idx, 0, t_idx, :].sum().item()
                row_masked = probs_masked[b_idx, 0, t_idx, :].sum().item()
                print(
                    "[DEBUG] fractional token "
                    f"batch={b_idx} idx={t_idx} val={val:.4f} "
                    f"col_raw={col_raw:.4f} col_masked={col_masked:.4f} "
                    f"row_raw={row_raw:.4f} row_masked={row_masked:.4f}"
                )
                debug_logged = True"""

            probs = probs * key_mask
            denom = probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            probs = probs / denom

            context = torch.matmul(probs, value)
            context = context.permute(0, 2, 1, 3).contiguous()
            context = context.view(bsz, seq_len, attn.all_head_size)

            attn_output = layer.attention.output(context, hidden_states)
            intermediate_output = layer.intermediate(attn_output)
            hidden_states = layer.output(intermediate_output, attn_output)

        return hidden_states

    def encode(self, input_ids, attention_mask):
        token_emb = self.token_embeddings(input_ids, attention_mask)
        mask = attention_mask.unsqueeze(-1).type_as(token_emb)

        sent_emb = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        if self.normalize:
           return F.normalize(sent_emb, dim=-1)

        return sent_emb


# ---------------------------------------------------------------------
# Generic frozen HF encoder
# ---------------------------------------------------------------------

class _FrozenHFEncoder(SentenceEncoder):
    def __init__(self, model_name: str, normalize: bool):
        super().__init__(normalize)
        self.model = AutoModel.from_pretrained(model_name)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward_hidden(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return out.last_hidden_state  # [B, T, D]

    def token_embeddings(self, input_ids, attention_mask):
        return self.forward_hidden(input_ids, attention_mask)


# ---------------------------------------------------------------------
# E5 / retrieval-style mean pooling
# ---------------------------------------------------------------------

class FrozenE5(_FrozenHFEncoder):
    def encode(self, input_ids, attention_mask):
        token_emb = self.forward_hidden(input_ids, attention_mask)
        mask = attention_mask.unsqueeze(-1).type_as(token_emb)

        sent_emb = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        if self.normalize:
            sent_emb = F.normalize(sent_emb, dim=-1)

        return sent_emb


# ---------------------------------------------------------------------
# BGE (CLS pooling)
# ---------------------------------------------------------------------

class FrozenBGE(_FrozenHFEncoder):
    def encode(self, input_ids, attention_mask):
        token_emb = self.forward_hidden(input_ids, attention_mask)
        sent_emb = token_emb[:, 0]

        if self.normalize:
            sent_emb = F.normalize(sent_emb, dim=-1)

        return sent_emb


# ---------------------------------------------------------------------
# LLM-style last-token pooling
# ---------------------------------------------------------------------

class FrozenLLMEncoder(_FrozenHFEncoder):
    def encode(self, input_ids, attention_mask):
        token_emb = self.forward_hidden(input_ids, attention_mask)

        mask = attention_mask > 0
        idx = mask.sum(dim=1).clamp(min=1) - 1
        sent_emb = token_emb[torch.arange(token_emb.size(0), device=token_emb.device), idx]

        if self.normalize:
            sent_emb = F.normalize(sent_emb, dim=-1)

        return sent_emb


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

def build_sentence_encoder(
    family: str,
    encoder_name: str | None,
    device: str | None = None,
) -> tuple[SentenceEncoder, AutoTokenizer]:

    family = family.lower()
    family = ALIAS_TO_CANON[family]

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if encoder_name in {None, "None", "null", "NULL"}:
        assert family in DEFAULT_MODEL_NAMES, "Unrecognized encoder family"
        encoder_name = DEFAULT_MODEL_NAMES[family]

    tokenizer = resolve_tokenizer(family)

    if family == "sbert":
        encoder = FrozenSBERT(encoder_name, normalize=True, device=device)
    elif family in {"e5", "retrieval", "gte"}:
        encoder = FrozenE5(encoder_name, normalize=True)
    elif family in {"bge", "late"}:
        encoder = FrozenBGE(encoder_name, normalize=True)
    elif family == "llm":
        encoder = FrozenLLMEncoder(encoder_name, normalize=True)
    else:
        raise ValueError(f"Unknown encoder family: {family}")

    encoder.to(device)
    encoder.eval()
    return encoder, tokenizer
