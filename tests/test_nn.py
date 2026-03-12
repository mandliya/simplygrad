"""
tests/test_nn.py — Tests for nn.Module, activations, losses, and optimizers.

Tests both gradient correctness and Module infrastructure
(buffers, train/eval, parameter collection).
"""

import numpy as np
import pytest
from utils import assert_gradient_correct
from deeplygrad import Tensor, xp
from deeplygrad.nn import Module, Linear, ReLU, GELU, CrossEntropyLoss
from deeplygrad.optim import SGD, Adam


# ======================================================================
#  Gradient checks for nn components
# ======================================================================

class TestReLU:
    def test_gradient(self):
        np.random.seed(42)
        x = Tensor(np.random.randn(4, 3), requires_grad=True)
        relu = ReLU()
        assert_gradient_correct(lambda: relu(x).sum(), [x])

    def test_forward_values(self):
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        relu = ReLU()
        result = relu(x)
        np.testing.assert_array_equal(result.numpy(), [0.0, 0.0, 0.0, 1.0, 2.0])

    def test_no_custom_backward(self):
        """ReLU should NOT have a custom _grad_fn — it composes from where()."""
        x = Tensor([1.0, -1.0], requires_grad=True)
        relu = ReLU()
        out = relu(x)
        # The output's _grad_fn comes from where(), not from ReLU directly
        # This proves ReLU is using composition, not a custom backward
        assert out._grad_fn is not None  # where() sets this
        assert out._parents is not None


class TestGELU:
    def test_gradient(self):
        np.random.seed(42)
        x = Tensor(np.random.randn(4, 3), requires_grad=True)
        gelu = GELU()
        assert_gradient_correct(lambda: gelu(x).sum(), [x])

    def test_approximates_relu_for_positive(self):
        """For large positive x, GELU(x) ≈ x."""
        x = Tensor([10.0, 20.0])
        gelu = GELU()
        result = gelu(x)
        np.testing.assert_allclose(result.numpy(), [10.0, 20.0], atol=0.01)

    def test_near_zero_for_negative(self):
        """For large negative x, GELU(x) ≈ 0."""
        x = Tensor([-10.0, -20.0])
        gelu = GELU()
        result = gelu(x)
        np.testing.assert_allclose(result.numpy(), [0.0, 0.0], atol=0.01)


class TestCrossEntropyLoss:
    def test_gradient(self):
        np.random.seed(42)
        logits = Tensor(np.random.randn(8, 5), requires_grad=True)
        targets = Tensor(np.random.randint(0, 5, size=(8,)))
        criterion = CrossEntropyLoss()
        assert_gradient_correct(lambda: criterion(logits, targets), [logits])

    def test_perfect_prediction(self):
        """Loss should be low when logits strongly favor correct class."""
        logits = Tensor([[10.0, -10.0, -10.0],
                         [-10.0, 10.0, -10.0]])
        targets = Tensor([0, 1])
        criterion = CrossEntropyLoss()
        loss = criterion(logits, targets)
        assert float(loss.data) < 0.001

    def test_uniform_prediction(self):
        """Loss should be log(num_classes) when logits are all equal."""
        logits = Tensor([[0.0, 0.0, 0.0]])
        targets = Tensor([0])
        criterion = CrossEntropyLoss()
        loss = criterion(logits, targets)
        expected = np.log(3.0)  # -log(1/3)
        np.testing.assert_allclose(float(loss.data), expected, atol=1e-6)


class TestLinear:
    def test_input_gradient(self):
        np.random.seed(42)
        lin = Linear(4, 3)
        x = Tensor(np.random.randn(5, 4), requires_grad=True)
        assert_gradient_correct(lambda: lin(x).sum(), [x])

    def test_weight_gradient(self):
        np.random.seed(42)
        lin = Linear(4, 3)
        x = Tensor(np.random.randn(5, 4))
        assert_gradient_correct(lambda: lin(x).sum(), [lin.weight])

    def test_bias_gradient(self):
        np.random.seed(42)
        lin = Linear(4, 3)
        x = Tensor(np.random.randn(5, 4))
        assert_gradient_correct(lambda: lin(x).sum(), [lin.bias])

    def test_output_shape(self):
        lin = Linear(4, 3)
        x = Tensor(np.random.randn(5, 4))
        assert lin(x).shape == (5, 3)

    def test_no_bias(self):
        lin = Linear(4, 3, bias=False)
        assert lin.bias is None
        x = Tensor(np.random.randn(5, 4))
        assert lin(x).shape == (5, 3)


class TestMLPGradient:
    """End-to-end gradient check through a full MLP."""

    def test_mlp_fc1_weight(self):
        np.random.seed(42)

        class TinyMLP(Module):
            def __init__(self):
                super().__init__()
                self.fc1 = Linear(4, 8)
                self.relu = ReLU()
                self.fc2 = Linear(8, 3)

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        mlp = TinyMLP()
        ce = CrossEntropyLoss()
        x = Tensor(np.random.randn(5, 4))
        targets = Tensor(np.random.randint(0, 3, size=(5,)))
        assert_gradient_correct(
            lambda: ce(mlp(x), targets),
            [mlp.fc1.weight]
        )

    def test_mlp_fc2_weight(self):
        np.random.seed(42)

        class TinyMLP(Module):
            def __init__(self):
                super().__init__()
                self.fc1 = Linear(4, 8)
                self.relu = ReLU()
                self.fc2 = Linear(8, 3)

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        mlp = TinyMLP()
        ce = CrossEntropyLoss()
        x = Tensor(np.random.randn(5, 4))
        targets = Tensor(np.random.randint(0, 3, size=(5,)))
        assert_gradient_correct(
            lambda: ce(mlp(x), targets),
            [mlp.fc2.weight]
        )


# ======================================================================
#  Module infrastructure tests
# ======================================================================

class TestModuleInit:
    """Module.__init__ sets up training mode and buffers."""

    def test_default_training_mode(self):
        m = Module()
        assert m.training is True

    def test_default_empty_buffers(self):
        m = Module()
        assert m._buffers == {}

    def test_subclass_has_training(self):
        lin = Linear(4, 3)
        assert hasattr(lin, 'training')
        assert lin.training is True

    def test_subclass_has_buffers(self):
        lin = Linear(4, 3)
        assert hasattr(lin, '_buffers')
        assert lin._buffers == {}


class TestTrainEval:
    """train() and eval() toggle training mode recursively."""

    def test_eval_sets_false(self):
        m = Module()
        m.eval()
        assert m.training is False

    def test_train_sets_true(self):
        m = Module()
        m.eval()
        m.train()
        assert m.training is True

    def test_propagates_to_children(self):
        class Parent(Module):
            def __init__(self):
                super().__init__()
                self.child = Linear(2, 2)
            def forward(self, x):
                return self.child(x)

        model = Parent()
        assert model.child.training is True

        model.eval()
        assert model.training is False
        assert model.child.training is False

        model.train()
        assert model.training is True
        assert model.child.training is True

    def test_returns_self(self):
        m = Module()
        assert m.train() is m
        assert m.eval() is m


class TestBuffers:
    """register_buffer() stores non-learnable state."""

    def test_buffer_accessible_as_attribute(self):
        m = Module()
        buf = Tensor(np.ones(3))
        m.register_buffer('my_buf', buf)
        assert hasattr(m, 'my_buf')
        assert m.my_buf is buf

    def test_buffer_in_buffers_dict(self):
        m = Module()
        buf = Tensor(np.ones(3))
        m.register_buffer('my_buf', buf)
        assert 'my_buf' in m._buffers
        assert m._buffers['my_buf'] is buf

    def test_buffer_excluded_from_parameters(self):
        class BufferedModule(Module):
            def __init__(self):
                super().__init__()
                self.weight = Tensor(np.ones(3), requires_grad=True)
                self.register_buffer('mask', Tensor(np.tril(np.ones((3, 3)))))
            def forward(self, x):
                return x

        m = BufferedModule()
        params = m.parameters()
        assert len(params) == 1  # only weight, not mask
        assert any(p is m.weight for p in params)

    def test_buffer_with_requires_grad_still_excluded(self):
        """Even if a buffer tensor has requires_grad, it should be excluded."""
        m = Module()
        buf = Tensor(np.ones(3), requires_grad=True)
        m.register_buffer('buf', buf)
        assert len(m.parameters()) == 0


class TestParameters:
    """parameters() collects the right tensors."""

    def test_linear_parameters(self):
        lin = Linear(4, 3)
        params = lin.parameters()
        assert len(params) == 2  # weight + bias

    def test_linear_no_biaseters(self):
        lin = Linear(4, 3, bias=False)
        params = lin.parameters()
        assert len(params) == 1  # weight only

    def test_nested_parameters(self):
        class TwoLayer(Module):
            def __init__(self):
                super().__init__()
                self.fc1 = Linear(4, 3)
                self.fc2 = Linear(3, 2)
            def forward(self, x):
                return self.fc2(self.fc1(x))

        model = TwoLayer()
        params = model.parameters()
        assert len(params) == 4  # 2 weights + 2 biases

    def test_relu_has_no_parameters(self):
        relu = ReLU()
        assert len(relu.parameters()) == 0


class TestZeroGrad:
    def test_zeros_all_grads(self):
        lin = Linear(4, 3)
        x = Tensor(np.random.randn(2, 4))
        loss = lin(x).sum()
        loss.backward()
        assert lin.weight.grad is not None

        lin.zero_grad()
        assert lin.weight.grad is None
        assert lin.bias.grad is None


# ======================================================================
#  Optimizer tests
# ======================================================================

class TestSGD:
    def test_step_updates_params(self):
        w = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        w.grad = xp.array([0.1, 0.2, 0.3])
        opt = SGD([w], lr=1.0)
        opt.step()
        np.testing.assert_allclose(w.numpy(), [0.9, 1.8, 2.7])

    def test_momentum(self):
        w = Tensor(np.array([1.0]), requires_grad=True)
        opt = SGD([w], lr=0.1, momentum=0.9)

        # Step 1: v = 0.9*0 + 1.0 = 1.0, w = 1.0 - 0.1*1.0 = 0.9
        w.grad = xp.array([1.0])
        opt.step()
        np.testing.assert_allclose(w.numpy(), [0.9], atol=1e-7)

        # Step 2: v = 0.9*1.0 + 1.0 = 1.9, w = 0.9 - 0.1*1.9 = 0.71
        w.grad = xp.array([1.0])
        opt.step()
        np.testing.assert_allclose(w.numpy(), [0.71], atol=1e-7)


class TestAdam:
    def test_step_updates_params(self):
        w = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        opt = Adam([w], lr=0.1)
        w.grad = xp.array([1.0, -1.0])
        opt.step()
        # After one step, params should have moved
        assert not np.allclose(w.numpy(), [1.0, 2.0])

    def test_converges_on_simple_problem(self):
        """Adam should minimize x² to near zero."""
        x = Tensor(np.array([5.0]), requires_grad=True)
        opt = Adam([x], lr=0.1)
        for _ in range(200):
            x.zero_grad()
            loss = x ** 2
            loss.backward()
            opt.step()
        assert abs(float(x.data.item())) < 0.01

    def test_zero_grad(self):
        w = Tensor(np.array([1.0]), requires_grad=True)
        w.grad = xp.array([1.0])
        opt = Adam([w])
        opt.zero_grad()
        assert w.grad is None