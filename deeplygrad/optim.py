"""
deplygrad.optim - Optimizer implementations for training models.

The optimizers will update the parameters of the model based on the computed gradients. 
We will implement:
- Stochastic Gradient Descent (SGD): The simplest optimization algorithm that updates parameters
 in the opposite direction of the gradient. We'll use optional momentum to accelerate convergence.
- Adam: An adaptive learning rate optimization algorithm that computes individual learning rates
    for different parameters based on the first and second moments of the gradients.
- RMSProp: An optimization algorithm that divides the learning rate by an exponentially
 decaying average of squared gradients.
"""

from typing import List
from deeplygrad.tensor import Tensor
from deeplygrad.backend import xp


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, parameters: List[Tensor], lr: float = 0.001):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum
    update rule (with momentum):
        v_t = momentum * v_{t-1} + grad
        param = param - lr * v_t
    """
    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.001,
        momentum: float = 0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocities = [xp.zeros_like(p.data) for p in parameters]
    
    def step(self):
        for p, v in zip(self.parameters, self.velocities):
            if p.grad is None:
                continue
            if self.momentum > 0:
                v[:] = self.momentum * v + p.grad
                p.data -= self.lr * v
            else:
                p.data -= self.lr * p.grad


class Adam(Optimizer):
    """
    Adam Optimizer (Kingma & Ba, 2014)

    Maintains per-parameter first moment (mean) and second moment
    (uncentered variance) estimates, with bias correction

    update rule:
        m_t = β₁ * m_{t-1} + (1-β₁) * grad
        v_t = β₂ * v_{t-1} + (1-β₂) * grad²
        m̂_t = m_t / (1 - β₁^t)      ← bias correction
        v̂_t = v_t / (1 - β₂^t)
        param = param - lr * m̂_t / (√v̂_t + ε)

    """
    def __init__(
            self,
            parameters: List[Tensor],
            lr: float = 0.001,
            beta1: float = 0.9,
            beta2: float = 0.999,
            eps: float = 1e-8):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = [xp.zeros_like(p.data) for p in self.parameters]
        self.v = [xp.zeros_like(p.data) for p in self.parameters]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1)* p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2)* p.grad * p.grad

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)
