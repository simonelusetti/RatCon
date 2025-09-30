# Main Idea

* We do not have labels for entities.
* Hypothesis: if we remove an entity word, the sentence embedding should change disproportionately.
* We use **rationale selection** (selecting subsets of tokens for a task).
* Since this is unsupervised, we treat it as **data augmentation**:
  → Which tokens, if removed, change the embedding the most?

# Model

### Embedding

* Each sentence is tokenized and encoded with a pretrained **Sentence-BERT** encoder.
* Output: contextual embeddings `H = [h_1,h_2, ...]`.

### HardKuma Selector

* Token embeddings are projected to parameters `α_i, β_i`.
* **HardKuma sampler** produces a continuous gate variable `g_i` in `(0,1)`
  * Training: stochastic reparameterization (soft-selection, differentiable).
  * Inference: approximates a Bernoulli mask (hard-selection, non differentiable).

### SBERT Encoding

Three sentence-level vectors are created using SBERT pooling starting from the previous layer:

* **Anchor** (all tokens): `h_anc = Pool(H)`
* **Rationale** (selected tokens): `h_rat = Pool(H ⊙ g)`
* **Complement** (unselected tokens): `h_cmp = Pool(H ⊙ (1 - g))`

### Losses

1. **InfoNCE (NT-Xent):**

   * Contrast anchors vs. rationals (`L_rat`).
   * Contrast anchors vs. complements (`L_cmp`).
2. **Regularization:**

   * **Sparsity loss** → encourages few tokens to be selected.
   * **Total variation loss** → encourages contiguous selections.

**Total loss:**

```
L = L_rat - λ_cmp * L_cmp + λ_s * L_sparsity + λ_tv * L_TV
```

---

# Improvements

### Attention Augment

* **Idea:** Model may struggle to distinguish important nouns, verbs, adjectives, etc.
* **Application:** Add two scalar features per token (`in_i`, `out_i`) representing incoming and outgoing attention strength.

### Fourier Filters

* **Idea:** Remove superficial syntactic info in embeddings via frequency filtering.
* **Application:** Apply FFT, mask frequencies (lowpass, highpass, bandpass), then inverse FFT.

### Dual (+) Models

* **Idea:** Use multiple models in parallel, forcing them to learn different rationales (e.g., nouns vs. verbs).
* **Application:**

  * Align their gate distributions with a symmetric KL divergence.
  * Total loss includes both per-model losses and the KL term.
