"""
tests/test_transformer.py — Tests for transformer building blocks.

Covers Softmax, LayerNorm, Embedding gradient correctness,
and a Transformer forward-pass smoke test.
"""

import numpy as np
import pytest
from utils import assert_gradient_correct
from deeplygrad import Tensor, xp
from deeplygrad.nn import Softmax, LayerNorm, Embedding, Linear
from deeplygrad.transformer import Transformer, TransformerConfig


# ======================================================================
#  Softmax
# ======================================================================

class TestSoftmax:
    def test_sums_to_one(self):
        x = Tensor(np.random.randn(3, 5))
        sm = Softmax(axis=-1)
        out = sm(x)
        row_sums = out.data.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-7)

    def test_gradient(self):
        np.random.seed(42)
        x = Tensor(np.random.randn(4, 5), requires_grad=True)
        sm = Softmax(axis=-1)
        assert_gradient_correct(lambda: sm(x).sum(), [x])

    def test_numerical_stability(self):
        """Large values should not cause overflow."""
        x = Tensor([[1000.0, 1001.0, 1002.0]])
        sm = Softmax(axis=-1)
        out = sm(x)
        assert np.all(np.isfinite(out.data))
        np.testing.assert_allclose(out.data.sum(), 1.0, atol=1e-7)


# ======================================================================
#  LayerNorm
# ======================================================================

class TestLayerNorm:
    def test_zero_mean_unit_var(self):
        """Output should have approximately zero mean along last dim."""
        np.random.seed(42)
        x = Tensor(np.random.randn(2, 8), requires_grad=False)
        ln = LayerNorm(8)
        out = ln(x)
        means = out.data.mean(axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)

    def test_input_gradient(self):
        np.random.seed(42)
        x = Tensor(np.random.randn(2, 8), requires_grad=True)
        ln = LayerNorm(8)
        assert_gradient_correct(lambda: ln(x).sum(), [x])

    def test_gamma_gradient(self):
        np.random.seed(42)
        x = Tensor(np.random.randn(2, 8))
        ln = LayerNorm(8)
        assert_gradient_correct(lambda: ln(x).sum(), [ln.gamma])

    def test_gamma_beta_shape(self):
        ln = LayerNorm(16)
        assert ln.gamma.shape == (16,)
        assert ln.beta.shape == (16,)


# ======================================================================
#  Embedding
# ======================================================================

class TestEmbedding:
    def test_forward_values(self):
        """Output rows should match the corresponding weight rows."""
        emb = Embedding(5, 3)
        indices = Tensor(xp.array([0, 2, 4]))
        out = emb(indices)
        np.testing.assert_array_equal(out.data[0], emb.W.data[0])
        np.testing.assert_array_equal(out.data[1], emb.W.data[2])
        np.testing.assert_array_equal(out.data[2], emb.W.data[4])

    def test_duplicate_index_gradient(self):
        """When the same index appears twice, gradients should accumulate (add.at)."""
        emb = Embedding(4, 3)
        indices = Tensor(xp.array([1, 1, 2]))
        out = emb(indices)
        loss = out.sum()
        loss.backward()

        grad_row1 = emb.W.grad[1]
        grad_row2 = emb.W.grad[2]
        np.testing.assert_allclose(grad_row1, 2.0 * np.ones(3), atol=1e-7)
        np.testing.assert_allclose(grad_row2, 1.0 * np.ones(3), atol=1e-7)

    def test_requires_grad_from_weight(self):
        """Output requires_grad should come from the weight, not the index tensor."""
        emb = Embedding(5, 3)
        indices = Tensor(xp.array([0, 1]), requires_grad=False)
        out = emb(indices)
        assert out.requires_grad is True


# ======================================================================
#  Transformer smoke test
# ======================================================================

class TestTransformerSmoke:
    @pytest.fixture
    def tiny_model(self):
        cfg = TransformerConfig(
            d_model=16, n_heads=2, d_mlp=32, n_layers=1,
            n_vocab=10, n_ctx=8, max_seq_len=8, dropout=0.0,
        )
        model = Transformer(cfg)
        model.eval()
        return model, cfg

    def test_output_shape(self, tiny_model):
        model, cfg = tiny_model
        x = Tensor(xp.random.randint(0, cfg.n_vocab, (2, 6)))
        logits, loss = model(x)
        assert logits.shape == (2, 6, cfg.n_vocab)
        assert loss is None

    def test_loss_is_scalar(self, tiny_model):
        model, cfg = tiny_model
        x = Tensor(xp.random.randint(0, cfg.n_vocab, (2, 6)))
        targets = Tensor(xp.random.randint(0, cfg.n_vocab, (2, 6)))
        logits, loss = model(x, targets)
        assert logits.shape == (2, 6, cfg.n_vocab)
        assert loss.data.size == 1
        assert loss.requires_grad is True
