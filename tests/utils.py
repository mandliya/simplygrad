"""
tests/utils.py: Utility functions for testing gradients in Deeplygrad.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, List, Callable
import pytest
import numpy as np
from deeplygrad import Tensor, xp, BACKEND_NAME
print(f"Using backend: {BACKEND_NAME}")


def numerical_gradient(
        tensor: Tensor, f: Callable, all_tensors: Optional[List] = None, eps=1e-4) -> np.ndarray:
    """
    Compute numerical gradient of scalar function f w.r.t. tensor
    using central finite differences:
        df/dx ≈ (f(x+ε) - f(x-ε)) / 2ε

    Uses float64 for the difference computation to avoid float32 precision issues.
    """
    if all_tensors is None:
        all_tensors = [tensor]

    original_value = tensor.data.copy()
    grad = np.zeros_like(tensor.data)

    # Iterate over all indices in the tensor
    it = np.nditer(tensor.data, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index

        # Perturb positively
        tensor.data[idx] = original_value[idx] + eps
        f_plus = float(f())

        # Perturb negatively
        tensor.data[idx] = original_value[idx] - eps
        f_minus = float(f())

        # Restore original value
        tensor.data[idx] = original_value[idx]

        # Compute numerical gradient
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()

    return grad

def assert_gradient_correct(
        f: Callable, tensors: List[Tensor], tol=1e-4):
    """
    Assert that analytic gradients match numerical gradients for all
    tensors with requires_grad=True.

    Parameters
    ----------
    f : callable
        A function that takes no arguments and returns a scalar Tensor.
    tensors : list of Tensor
        The tensors to check gradients for.
    tol : float
        Maximum allowed difference between analytic and numerical gradients.
    """
    # Step 1: Compute numerical gradients FIRST (before calling backward)
    numerical_grads = {}
    for t in tensors:
        if t.requires_grad:
            numerical_grads[id(t)] = numerical_gradient(t, f, tensors)

    # Step 2: Now compute analytic gradients via backward
    for t in tensors:
        t.zero_grad()
    loss = f()  # forward pass
    loss.backward()  # compute gradients

    # Step 3: Compare gradients
    for t in tensors:
        if not t.requires_grad:
            continue
        analytic = t.grad
        numerical = numerical_grads[id(t)]

        diff = np.abs(np.array(analytic) - numerical)
        max_diff = float(diff.max())
        assert max_diff < tol, (
            f"Gradient mismatch: max_diff={max_diff:.2e} > tol={tol:.2e}\n"
            f"  analytic:  {analytic}\n"
            f"  numerical: {numerical}"
        )