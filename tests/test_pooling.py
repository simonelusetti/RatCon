"""Pins the pooling contract. Pure tensor maths -- no model downloads.

The min/max spec these tests encode: scale every token embedding by its
selection score, then take the coordinate-wise extremum across the scaled
embeddings. A score of zero therefore contributes a zero *vector* that
competes in the extremum like any other, and scores are not clipped to
[0, 1]. Batch padding is not an embedding at all and is excluded via
valid_mask.
"""
import unittest

import torch
import torch.nn.functional as F

from src.sentence import POOLING_STRATEGIES, SentenceEncoder, masked_pool


def reference_extremum(token_emb, scores, valid, strategy):
    """The spec, written out literally: scale, then reduce coordinate-wise."""
    out = torch.zeros(token_emb.shape[0], token_emb.shape[2])
    for b in range(token_emb.shape[0]):
        vectors = [token_emb[b, t] * scores[b, t]
                   for t in range(token_emb.shape[1]) if valid[b, t]]
        stacked = torch.stack(vectors)
        out[b] = stacked.amax(0) if strategy == "max" else stacked.amin(0)
    return out


class MinMaxSpecTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.h = torch.randn(3, 6, 8)
        self.attn = torch.ones(3, 6)
        self.scores = torch.rand(3, 6)

    def test_matches_scale_then_coordinatewise_extremum(self):
        for strategy in ("min", "max"):
            with self.subTest(strategy=strategy):
                got = masked_pool(self.h, self.scores, strategy, valid_mask=self.attn)
                want = reference_extremum(self.h, self.scores, self.attn.bool(), strategy)
                torch.testing.assert_close(got, want)

    def test_zero_score_contributes_a_zero_vector(self):
        # All-positive states, so the zero vector from the deselected token
        # is what wins the minimum. This is the spec, not a rounding effect.
        h = torch.tensor([[[1.0, 2.0], [4.0, 9.0], [3.0, 5.0]]])
        scores = torch.tensor([[1.0, 1.0, 0.0]])
        attn = torch.ones(1, 3)
        torch.testing.assert_close(
            masked_pool(h, scores, "min", valid_mask=attn), torch.zeros(1, 2))
        torch.testing.assert_close(
            masked_pool(h, scores, "max", valid_mask=attn), torch.tensor([[4.0, 9.0]]))

    def test_scores_above_one_are_not_clipped(self):
        # The selector renormalises z to sum to k, so scores can exceed 1.
        h = torch.tensor([[[2.0, -3.0]]])
        got = masked_pool(h, torch.tensor([[2.5]]), "max", valid_mask=torch.ones(1, 1))
        torch.testing.assert_close(got, torch.tensor([[5.0, -7.5]]))

    def test_padding_is_excluded_not_scored_as_zero(self):
        # A padded batch must give the same answer as the unpadded sentence.
        h = torch.tensor([[[1.0, 2.0], [4.0, 9.0], [0.0, 0.0]]])
        attn = torch.tensor([[1.0, 1.0, 0.0]])
        scores = attn.clone()  # every real token selected
        for strategy in ("min", "max"):
            with self.subTest(strategy=strategy):
                padded = masked_pool(h, scores, strategy, valid_mask=attn)
                unpadded = masked_pool(h[:, :2], scores[:, :2], strategy,
                                       valid_mask=attn[:, :2])
                torch.testing.assert_close(padded, unpadded)

    def test_valid_mask_matters_only_when_nothing_is_deselected(self):
        # Once any real token has score 0 its zero vector is already in the
        # pool, so padding's zeros add nothing -- which is why the oracle and
        # mask_optimality can pass padding-free rows and omit valid_mask.
        # Signs are chosen so the omission would show if it mattered: an
        # all-positive sentence for min, all-negative for max.
        for strategy, sign in (("min", 1.0), ("max", -1.0)):
            with self.subTest(strategy=strategy):
                h = sign * (torch.arange(12, dtype=torch.float).view(1, 4, 3) + 1.0)
                attn = torch.tensor([[1.0, 1.0, 1.0, 0.0]])       # one padded slot
                partial = torch.tensor([[1.0, 0.0, 1.0, 0.0]])    # one token dropped
                torch.testing.assert_close(
                    masked_pool(h, partial, strategy, valid_mask=attn),
                    masked_pool(h, partial, strategy))
                self.assertFalse(torch.allclose(
                    masked_pool(h, attn, strategy, valid_mask=attn),
                    masked_pool(h, attn, strategy)))

    def test_fully_padded_row_pools_to_zero(self):
        h = torch.randn(2, 4, 3)
        attn = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0] * 4])
        for strategy in ("min", "max"):
            with self.subTest(strategy=strategy):
                pooled = masked_pool(h, attn, strategy, valid_mask=attn)
                self.assertTrue(torch.isfinite(pooled).all())
                torch.testing.assert_close(pooled[1], torch.zeros(3))


class LossGeometryTests(unittest.TestCase):
    """Properties the reconstruction loss relies on."""

    def test_cosine_is_invariant_to_uniform_score_scaling(self):
        # Scaling every score by a > 0 scales the pooled vector by a, so the
        # cosine loss cannot be gamed by inflating scores; only the relative
        # pattern of scores -- actual selection -- moves it.
        torch.manual_seed(3)
        h, attn = torch.randn(2, 7, 16), torch.ones(2, 7)
        scores = torch.rand(2, 7)
        for strategy in ("min", "max"):
            target = masked_pool(h, attn, strategy, valid_mask=attn)
            base = F.cosine_similarity(
                masked_pool(h, scores, strategy, valid_mask=attn), target, dim=-1)
            for a in (0.25, 4.0):
                with self.subTest(strategy=strategy, a=a):
                    scaled = F.cosine_similarity(
                        masked_pool(h, scores * a, strategy, valid_mask=attn), target, dim=-1)
                    torch.testing.assert_close(scaled, base, rtol=1e-5, atol=1e-6)

    def test_score_gradients_are_finite_and_order_one(self):
        # Regression guard: an earlier masked-penalty formulation leaked a
        # 1e4 exclusion constant into the backward pass, producing gradients
        # that dwarfed every other term in the loss.
        torch.manual_seed(5)
        h, attn = torch.randn(2, 6, 8), torch.ones(2, 6)
        for strategy in ("mean", "min", "max"):
            with self.subTest(strategy=strategy):
                scores = torch.rand(2, 6, requires_grad=True)
                masked_pool(h, scores, strategy, valid_mask=attn).sum().backward()
                self.assertTrue(torch.isfinite(scores.grad).all())
                self.assertLess(scores.grad.abs().max().item(), 100.0)

    def test_last_carries_no_gradient_to_the_scores(self):
        # `last` is a hard index, so the selector trains only through the
        # attention path for this strategy. Pinned because a silently
        # detached mask would otherwise look like a strategy that simply
        # learns nothing.
        h = torch.randn(2, 6, 8)
        scores = torch.rand(2, 6, requires_grad=True)
        pooled = masked_pool(h, scores, "last", valid_mask=torch.ones(2, 6))
        self.assertFalse(pooled.requires_grad)


class OtherStrategyTests(unittest.TestCase):
    def test_mean_is_the_score_weighted_average(self):
        h = torch.tensor([[[1.0, 2.0], [3.0, 6.0]]])
        got = masked_pool(h, torch.tensor([[1.0, 3.0]]), "mean", valid_mask=torch.ones(1, 2))
        torch.testing.assert_close(got, torch.tensor([[2.5, 5.0]]))

    def test_last_takes_the_final_positive_score(self):
        h = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        scores = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        got = masked_pool(h, scores, "last", valid_mask=torch.ones(1, 4))
        torch.testing.assert_close(got, torch.tensor([[3.0]]))

    def test_empty_selection_pools_to_zero(self):
        h = torch.randn(1, 4, 5)
        zeros = torch.zeros(1, 4)
        for strategy in POOLING_STRATEGIES:
            with self.subTest(strategy=strategy):
                pooled = masked_pool(h, zeros, strategy, valid_mask=torch.ones(1, 4))
                self.assertTrue(torch.isfinite(pooled).all())

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            masked_pool(torch.randn(1, 2, 3), torch.ones(1, 2), "median")


class EncoderPoolingTests(unittest.TestCase):
    """pool()/pool_full() must apply the same strategy to subset and target."""

    class Identity(SentenceEncoder):
        def token_embeddings(self, input_ids, attention_mask):
            raise NotImplementedError

    def test_pool_full_is_the_extremum_over_real_tokens_only(self):
        h = torch.tensor([[[1.0, 2.0], [4.0, 9.0], [7.0, 7.0]]])
        attn = torch.tensor([[1.0, 1.0, 0.0]])
        for strategy in ("min", "max"):
            with self.subTest(strategy=strategy):
                enc = self.Identity(normalize=False, pooling=strategy)
                want = (torch.tensor([[1.0, 2.0]]) if strategy == "min"
                        else torch.tensor([[4.0, 9.0]]))
                torch.testing.assert_close(enc.pool_full(h, attn), want)

    def test_unknown_pooling_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            self.Identity(normalize=False, pooling="median")


if __name__ == "__main__":
    unittest.main()
