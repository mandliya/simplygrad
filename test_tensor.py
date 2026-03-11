"""
tests/test_tensor.py — Test suite for cutigrad.Tensor

These tests serve double duty:
  1. Verify our autograd is correct (via numerical gradient checking)
  2. Act as living documentation for the blog post

We use numerical gradient checking as our ground truth:
    df/dx ≈ (f(x + ε) - f(x - ε)) / (2ε)
This is the finite-difference method — slow but reliable.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from deeplygrad import Tensor, xp, BACKEND_NAME


def _old_unused():
    pass


def check_gradient(f, tensors, names=None, tol=1e-4):
    """Run backward, then compare analytic grad to numerical grad."""
    if names is None:
        names = [f"tensor_{i}" for i in range(len(tensors))]

    # Step 1: Compute numerical gradients FIRST (these don't call backward)
    numerical_grads = {}
    for t, name in zip(tensors, names):
        if t.requires_grad:
            numerical_grads[name] = numerical_gradient_pure(t, tensors, f)

    # Step 2: Now compute analytic gradients via backward
    for t in tensors:
        t.zero_grad()
    loss = f_no_zero(f, tensors)  # forward without zeroing
    loss.backward()

    all_pass = True
    for t, name in zip(tensors, names):
        if not t.requires_grad:
            continue
        analytic = t.grad
        numerical = numerical_grads[name]

        diff = np.abs(np.array(analytic) - numerical)
        max_diff = float(diff.max())
        if max_diff > tol:
            print(f"  ✗ {name}: max_diff = {max_diff:.6f}")
            print(f"    analytic:  {analytic}")
            print(f"    numerical: {numerical}")
            all_pass = False
        else:
            print(f"  ✓ {name}: max_diff = {max_diff:.2e}")
    return all_pass


def numerical_gradient_pure(tensor, all_tensors, f, eps=1e-4):
    """Compute numerical gradient by perturbing tensor.data directly.
    
    Note: We use float64 for the perturbation and difference computation
    to avoid float32 precision issues in the finite-difference formula.
    """
    grad = np.zeros(tensor.shape, dtype=np.float64)
    it = np.nditer(grad, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old_val = float(tensor.data[idx])

        tensor.data[idx] = old_val + eps
        for t in all_tensors:
            t.zero_grad()
        loss_plus = float(f().data)

        tensor.data[idx] = old_val - eps
        for t in all_tensors:
            t.zero_grad()
        loss_minus = float(f().data)

        grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        tensor.data[idx] = old_val
        it.iternext()
    return grad


def f_no_zero(f, tensors):
    """Call f() — f() already zeros grads internally, just call it."""
    return f()


# ======================================================================
#  Tests
# ======================================================================

def test_add():
    print("\n--- Test: Addition ---")
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    def f():
        a.zero_grad(); b.zero_grad()
        return (a + b).sum()
    assert check_gradient(f, [a, b], ["a", "b"])


def test_mul():
    print("\n--- Test: Multiplication ---")
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    def f():
        a.zero_grad(); b.zero_grad()
        return (a * b).sum()
    assert check_gradient(f, [a, b], ["a", "b"])


def test_matmul():
    print("\n--- Test: Matrix Multiplication ---")
    np.random.seed(42)
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(4, 2), requires_grad=True)
    def f():
        a.zero_grad(); b.zero_grad()
        return (a @ b).sum()
    assert check_gradient(f, [a, b], ["A (3x4)", "B (4x2)"])


def test_pow():
    print("\n--- Test: Power ---")
    a = Tensor([2.0, 3.0, 4.0], requires_grad=True)
    def f():
        a.zero_grad()
        return (a ** 3).sum()
    assert check_gradient(f, [a], ["a"])


def test_exp_log():
    print("\n--- Test: Exp + Log ---")
    a = Tensor([1.0, 2.0, 0.5], requires_grad=True)
    def f():
        a.zero_grad()
        return a.exp().log().sum()  # exp(log(x)) grad should be ~1
    assert check_gradient(f, [a], ["a"])


def test_broadcast_add():
    print("\n--- Test: Broadcast Addition ---")
    # a is (3,4), b is (4,) — b gets broadcast along axis 0
    np.random.seed(42)
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(4), requires_grad=True)
    def f():
        a.zero_grad(); b.zero_grad()
        return (a + b).sum()
    assert check_gradient(f, [a, b], ["a (3x4)", "b (4,)"])


def test_broadcast_mul():
    print("\n--- Test: Broadcast Multiplication ---")
    np.random.seed(42)
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(4), requires_grad=True)
    def f():
        a.zero_grad(); b.zero_grad()
        return (a * b).sum()
    assert check_gradient(f, [a, b], ["a (3x4)", "b (4,)"])


def test_complex_expression():
    """
    A more realistic test: something like a simple linear layer + MSE loss.
    out = ((X @ W + b) - y)^2   [mean squared error]
    """
    print("\n--- Test: Linear Layer + MSE Loss ---")
    np.random.seed(42)
    X = Tensor(np.random.randn(5, 3), requires_grad=False)
    W = Tensor(np.random.randn(3, 2), requires_grad=True)
    b = Tensor(np.random.randn(2), requires_grad=True)
    y = Tensor(np.random.randn(5, 2), requires_grad=False)

    def f():
        W.zero_grad(); b.zero_grad()
        pred = X @ W + b
        diff = pred - y
        loss = (diff * diff).mean()
        return loss
    assert check_gradient(f, [W, b], ["W (3x2)", "b (2,)"])


def test_reshape():
    print("\n--- Test: Reshape ---")
    np.random.seed(42)
    a = Tensor(np.random.randn(2, 3), requires_grad=True)
    def f():
        a.zero_grad()
        return a.reshape(3, 2).sum()
    assert check_gradient(f, [a], ["a"])


def test_transpose():
    print("\n--- Test: Transpose ---")
    np.random.seed(42)
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    def f():
        a.zero_grad()
        return a.T.sum()
    assert check_gradient(f, [a], ["a"])


def test_shared_node():
    """
    When a tensor is used multiple times, gradients must ACCUMULATE.
    y = x * x  →  dy/dx = 2x (not x!)
    This is a common bug in naive autograd implementations.
    """
    print("\n--- Test: Shared Node (x * x) ---")
    x = Tensor([3.0], requires_grad=True)
    def f():
        x.zero_grad()
        return (x * x).sum()
    assert check_gradient(f, [x], ["x"])
    print(f"    x = 3.0, expected grad = 6.0, got = {x.grad}")


def test_chain():
    """Multi-step chain: sigmoid approximation using basic ops."""
    print("\n--- Test: Multi-step Chain ---")
    np.random.seed(42)
    x = Tensor(np.random.randn(4), requires_grad=True)
    def f():
        x.zero_grad()
        # Approximate sigmoid: 1 / (1 + exp(-x))
        return (Tensor(1.0) / (Tensor(1.0) + (-x).exp())).sum()
    assert check_gradient(f, [x], ["x"])


def test_indexing():
    print("\n--- Test: Indexing ---")
    np.random.seed(42)
    a = Tensor(np.random.randn(4, 3), requires_grad=True)
    def f():
        a.zero_grad()
        return a[1:3, :2].sum()
    assert check_gradient(f, [a], ["a"])


def test_max():
    print("\n--- Test: Max ---")
    a = Tensor([1.0, 5.0, 3.0, 2.0], requires_grad=True)
    def f():
        a.zero_grad()
        return a.max()
    assert check_gradient(f, [a], ["a"])


# ======================================================================
#  Run all tests
# ======================================================================

if __name__ == "__main__":
    print(f"Backend: {BACKEND_NAME}")
    print("=" * 50)

    test_add()
    test_mul()
    test_matmul()
    test_pow()
    test_exp_log()
    test_broadcast_add()
    test_broadcast_mul()
    test_complex_expression()
    test_reshape()
    test_transpose()
    test_shared_node()
    test_chain()
    test_indexing()
    test_max()

    print("\n" + "=" * 50)
    print("All tests passed!")