from __future__ import annotations

from tqdm import tqdm
from collections import defaultdict

import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from luse.data import CATH_TO_ID, ID_TO_CATH, get_dataset
from luse.utils import sbert_encode_texts

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 512
SPLIT = "train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    ds = get_dataset(name="ud", shuffle=True, rebuild=True)
    sbert = SentenceTransformer(MODEL_NAME)

    sum_shift = {cat: 0.0 for cat in CATH_TO_ID.keys()}   # sum of normalized shifts
    count_sent = {cat: 0 for cat in CATH_TO_ID.keys()}    # number of sentences that contributed

    cos = torch.nn.CosineSimilarity(dim=-1)

    for example in tqdm(ds):
        tokens = example["tokens"]
        cath_tags = example["cath_tags"]

        if not tokens:
            continue

        # original sentence embedding
        s_emb = sbert_encode_texts(sbert, tokens, DEVICE)[0]

        for (cat, _) in CATH_TO_ID.items():
            idxs = [i for i, c in enumerate(cath_tags) if c == cat]
            n_masked = len(idxs)
            if n_masked == 0:
                continue  # this category not present in this sentence

            masked_tokens = [tok for i, tok in enumerate(tokens) if i not in idxs]

            m_emb = sbert_encode_texts(sbert, masked_tokens, DEVICE)[0]

            shift = 1.0 - cos(s_emb, m_emb).item()

            norm_shift = shift / n_masked

            sum_shift[cat] += norm_shift
            count_sent[cat] += 1


    print(f"Results on UD split = {SPLIT}")
    for cat in CATH_TO_ID.keys():
        if count_sent[cat] == 0:
            avg = float("nan")
        else:
            avg = sum_shift[cat] / count_sent[cat]
        print(
            f"Category: {cat:7s} | "
            f"avg normalized shift = {avg:.6f} | "
            f"sentences contributing = {count_sent[cat]}"
        )


if __name__ == "__main__":
    main()
