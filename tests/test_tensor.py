"""
tests/test_tensor.py — Gradient tests for all Tensor operations.

Each test creates a small computation, runs backward(), and verifies
against numerical gradients. These are the correctness proofs for Part 1.
"""

import numpy as np
import pytest
from utils import numerical_gradient, assert_gradient_correct
from deeplygrad import Tensor, xp, BACKEND_NAME
print(f"Using backend: {BACKEND_NAME}")
from deeplygrad.tensor import where

class TestArithmetic:
    def test_add(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(3, 4), requires_grad=True)
        assert_gradient_correct(lambda: (a + b).sum(), [a, b])
    
    def test_mul(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(3, 4), requires_grad=True)
        assert_gradient_correct(lambda: (a * b).sum(), [a, b])
    
    def test_matmul(self):
        a = Tensor(np.random.randn(5, 6), requires_grad=True)
        b = Tensor(np.random.randn(6, 7), requires_grad=True)
        assert_gradient_correct(lambda: (a @ b).sum(), [a, b])

    def test_div(self):
        np.random.seed(42)
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(3, 4) + 3.0, requires_grad=True)
        assert_gradient_correct(lambda: (a / b).sum(), [a, b])

class TestMatmul:
    """Matrix multiplication with various shapes."""

    def test_2d_matmul(self):
        np.random.seed(42)
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4, 2), requires_grad=True)
        assert_gradient_correct(lambda: (a @ b).sum(), [a, b])

    def test_matmul_vector(self):
        np.random.seed(42)
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4), requires_grad=True)
        assert_gradient_correct(lambda: (a @ b).sum(), [a, b])

class TestUnaryOps:
    """Unary operations: pow, exp, log."""

    def test_pow(self):
        a = Tensor([2.0, 3.0, 4.0], requires_grad=True)
        assert_gradient_correct(lambda: (a ** 3).sum(), [a])

    def test_exp(self):
        a = Tensor([1.0, 2.0, 0.5], requires_grad=True)
        assert_gradient_correct(lambda: a.exp().sum(), [a])

    def test_log(self):
        a = Tensor([1.0, 2.0, 0.5], requires_grad=True)
        assert_gradient_correct(lambda: a.log().sum(), [a])

    def test_exp_log_roundtrip(self):
        """exp(log(x)) should have gradient ≈ 1."""
        a = Tensor([1.0, 2.0, 0.5], requires_grad=True)
        assert_gradient_correct(lambda: a.exp().log().sum(), [a])

class TestReductions:
    """Sum, mean, max."""

    def test_sum(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        assert_gradient_correct(lambda: a.sum(), [a])

    def test_sum_axis(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        assert_gradient_correct(lambda: a.sum(axis=1).sum(), [a])

    def test_mean(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        assert_gradient_correct(lambda: a.mean(), [a])

    def test_max(self):
        a = Tensor([1.0, 5.0, 3.0, 2.0], requires_grad=True)
        assert_gradient_correct(lambda: a.max(), [a])
    
class TestShapeOps:
    """Reshape, transpose, indexing."""

    def test_reshape(self):
        np.random.seed(42)
        a = Tensor(np.random.randn(2, 3), requires_grad=True)
        assert_gradient_correct(lambda: a.reshape(3, 2).sum(), [a])

    def test_transpose(self):
        np.random.seed(42)
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        assert_gradient_correct(lambda: a.T.sum(), [a])

    def test_indexing(self):
        np.random.seed(42)
        a = Tensor(np.random.randn(4, 3), requires_grad=True)
        assert_gradient_correct(lambda: a[1:3, :2].sum(), [a])


class TestBroadcasting:
    """Broadcasting: ops between different-shaped tensors."""

    def test_broadcast_add(self):
        np.random.seed(42)
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4), requires_grad=True)
        assert_gradient_correct(lambda: (a + b).sum(), [a, b])

    def test_broadcast_mul(self):
        np.random.seed(42)
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4), requires_grad=True)
        assert_gradient_correct(lambda: (a * b).sum(), [a, b])


class TestSharedNodes:
    """Gradient accumulation when a tensor is used multiple times."""

    def test_x_squared(self):
        """x * x should give grad = 2x, not x."""
        x = Tensor([3.0], requires_grad=True)
        assert_gradient_correct(lambda: (x * x).sum(), [x])
        # Also verify the actual value
        x.zero_grad()
        loss = (x * x).sum()
        loss.backward()
        np.testing.assert_allclose(x.grad, [6.0], atol=1e-7)

    def test_x_used_three_times(self):
        """x + x + x should give grad = 3."""
        x = Tensor([2.0], requires_grad=True)
        assert_gradient_correct(lambda: (x + x + x).sum(), [x])


class TestComparison:
    """Comparison operators and where()."""

    def test_gt(self):
        a = Tensor([1.0, -2.0, 3.0])
        result = a > 0
        np.testing.assert_array_equal(result.numpy(), [1.0, 0.0, 1.0])

    def test_lt(self):
        a = Tensor([1.0, -2.0, 3.0])
        result = a < 0
        np.testing.assert_array_equal(result.numpy(), [0.0, 1.0, 0.0])

    def test_ge(self):
        a = Tensor([1.0, 0.0, -1.0])
        result = a >= 0
        np.testing.assert_array_equal(result.numpy(), [1.0, 1.0, 0.0])

    def test_eq(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([1.0, 0.0, 3.0])
        result = a == b
        np.testing.assert_array_equal(result.numpy(), [1.0, 0.0, 1.0])

    def test_comparison_not_differentiable(self):
        """Comparison results should not require grad."""
        a = Tensor([1.0, -2.0], requires_grad=True)
        result = a > 0
        assert not result.requires_grad  # Check if requires_grad is False

    def test_where_forward(self):
        cond = Tensor([1.0, 0.0, 1.0])
        x = Tensor([10.0, 20.0, 30.0])
        y = Tensor([-1.0, -2.0, -3.0])
        result = where(cond, x, y)
        np.testing.assert_array_equal(result.numpy(), [10.0, -2.0, 30.0])

    def test_where_gradient(self):
        """Gradient should flow through x where True, y where False."""
        x = Tensor([10.0, 20.0, 30.0], requires_grad=True)
        y = Tensor([-1.0, -2.0, -3.0], requires_grad=True)
        cond = Tensor([1.0, 0.0, 1.0])
        assert_gradient_correct(lambda: where(cond, x, y).sum(), [x, y])


class TestComplexExpressions:
    """Realistic multi-op expressions to test the full chain."""

    def test_linear_mse(self):
        """Linear layer + MSE loss, like a real training step."""
        np.random.seed(42)
        X = Tensor(np.random.randn(5, 3))
        W = Tensor(np.random.randn(3, 2), requires_grad=True)
        b = Tensor(np.random.randn(2), requires_grad=True)
        y = Tensor(np.random.randn(5, 2))

        def f():
            W.zero_grad()
            b.zero_grad()  # Zero gradients for W and b
            pred = X @ W + b
            diff = pred - y
            return (diff * diff).mean()

        assert_gradient_correct(f, [W, b])

    def test_sigmoid_chain(self):
        """Sigmoid approximated from basic ops: 1 / (1 + exp(-x))."""
        np.random.seed(42)
        x = Tensor(np.random.randn(4), requires_grad=True)

        def f():
            x.zero_grad()
            return (Tensor(1.0) / (Tensor(1.0) + (-x).exp())).sum()

        assert_gradient_correct(f, [x])

    def test_relu_via_where(self):
        """ReLU expressed as where(x > 0, x, 0) — autograd composes it."""
        np.random.seed(42)
        x = Tensor(np.random.randn(4, 3), requires_grad=True)

        def f():
            x.zero_grad()
            return where(x > 0, x, Tensor(0.0)).sum()

        assert_gradient_correct(f, [x])
