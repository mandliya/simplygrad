"""
deeplygrad.nn: Neural network building blocks

Like PyTorch, we will use a Module for all the elementary blocks
like losses, layers, activations. 
"""

from deeplygrad.tensor import Tensor
from deeplygrad.backend import xp
from typing import List, Optional, Dict

class Module:
    """
    Base class for all neural network components
    This gives each component a uniform interface:
        - forward() defines the computation
        - parameters() collects all the trainable parameters
        - zero_grad() resets the gradients of all the parameters of the module
        - train()/eval() toggles the training mode
        - register_buffer() stores non-learnable states
    
    Subclass ideally implements the forward.
    """
    def __init__(self):
        self.training: bool = True
        self._buffers: Dict[str, Optional[Tensor]] = {}

    def parameters(self) -> List[Tensor]:
        """
        Recursively collect all Tensors that requires grad.
        Walks through the attributes, child Modules and list/tuple of Modules
        """
        buffer_names = set(self._buffers.keys()) if hasattr(self, '_buffers') else set()

        params = []
        for attr_name in dir(self):
            if attr_name.startswith('_'):
                continue
            if attr_name in buffer_names:
                continue
            attr = getattr(self, attr_name)
            if isinstance(attr, Tensor) and attr.requires_grad:
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params
    
    def zero_grad(self):
        """Zero all parameter gradients in this module and it's children"""
        for param in self.parameters():
            param.zero_grad()
    
    def register_buffer(self, name: str, tensor: Optional[Tensor]):
        """
        Registers a non-learnable tensor as part of the module's state
        Buffers are tensors which:
            - Are not parameters (won't be returned by parameters())
            - Are part of module's state (e.g. causal masks, running stats)
            - Are accesible as self.name
        Example (we'll do it when we'll implement transformer causal self-attention)
            mask = Tensor(np.tril(np.ones(n_ctx, n_ctx)))
            self.register_buffer('mask', mask)
        """
        self._buffers[name] = tensor
        setattr(self, name, tensor)

    def train(self, mode: bool = True) -> 'Module':
        """
        Sets the training mode. Affects behavior of Dropout, BatchNorm etc
        Recurses into child modules
        Usage:
            model.train()
            model.eval()
        """
        self.training = mode
        for attr_name in dir(self):
            if attr_name.startswith('_'):
                continue
            attr = getattr(self, attr_name)
            if isinstance(attr, Module):
                attr.train(mode)
        return self
    
    def eval(self) -> 'Module':
        return self.train(mode=False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


# ======================================================================
#  2. Linear — the core layer
# ======================================================================

class Linear(Module):
    """
    Fully connected Linear Layer: out = x @ W + b

    Uses Kaiming (He) initialization, which keeps the activation
    variance roughly constant through ReLU layers.

    Parameters:
    -----------
    in_features: int
        size of each input sample.
    out_features: int
        size of each output sample
    bias: bool
        If true, we add a learnable bias
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        scale = xp.sqrt(2.0 / in_features)
        self.weight = Tensor(
            data=xp.random.randn(in_features, out_features) * scale,
            requires_grad=True
        )
        self.bias = Tensor(
            xp.zeros(out_features),
            requires_grad=True
        ) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def __repr__(self):
        return f"Linear({self.in_features}, {self.out_features}, bias={self.bias is not None})"


# ======================================================================
#  3. Activation functions, start with ReLU
# ======================================================================

class ReLU(Module):
    """
    ReLU activation: out = max(0, x)

    Gradient: 1 where x > 0, 0 elsewhere.

    Special Note: We don't need custom backward here because
    the autograd engine composes the backward passes of `>` and `where`
    automatically via the chain rule.
    """
    def __init__(self):
        super().__init__()

    def forward(self, t: Tensor) -> Tensor:
        from deeplygrad.tensor import where
        return where(t > 0, t, Tensor(0.0))
    
class GELU(Module):
    """
    GELU activation: out = x · phi(x) where phi is the standard normal CDF.

    Uses the tanh approximation (same as PyTorch default):
        GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    This is the activation used in GPT-2, BERT, and most modern transformers.
    Unlike ReLU which has a hard cutoff, GELU smoothly gates values.
    """
    def __init__(self):
        super().__init__()

    def forward(self, t: Tensor) -> Tensor:
        c = xp.sqrt(2.0/xp.pi)
        x = t.data
        inner = x + 0.044715 * (x ** 3)
        tanh_inner = xp.tanh(c * inner)
        out_data = 0.5 * x * (1 + tanh_inner)
        out = Tensor(out_data, requires_grad=t.requires_grad)

        if out.requires_grad:
            out._parents = [t]

            def _backward(grad_output):
                sech2 = 1.0 - tanh_inner ** 2
                d_inner = c * (1.0 + 3.0 * 0.044715 * x ** 2)
                g = grad_output * (0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner)
                t.grad = t.grad + g if t.grad is not None else g
            
            out._grad_fn = _backward
        return out
    
# ======================================================================
#  4. Loss functions, start with MSE
# ======================================================================

class MSELoss(Module):
    """
    Mean Squared Error Loss: loss = mean((y_pred - y_true)²)

    This is the standard loss for regression problems.
    Again, we don't need a custom backward because the autograd engine will
    automatically compute the gradients via the chain rule.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        loss = ((y_pred - y_true) ** 2)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError("Invalid reduction. Use 'mean' or 'sum'.")

class CrossEntropyLoss(Module):
    """
    Cross-Entropy Loss for multi-class classification.

    It takes raw logits (unnormalized scores) and integer class labels.
    Returns scalar mean loss over the batch.

    We will implement a fused softmax + cross entrpy loss. 
    It will use log-sum-exp trick for numerical stability and
    will be more efficient than separate softmax + log + mean.
    log(Σ exp(x)) = max(x) + log(Σ exp(x - max(x)))

    This is similar to PyTorch's nn.CrossEntropyLoss, 
    which also combines LogSoftmax and NLLLoss in one single class.
    (note similar not same).

    The gradient w.r.t logits is:
        grad = (softmax(logits) - one_hot(labels)) / batch_size

    We will write a custom _backward for this to avoid computing
    the full softmax in the forward pass.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        batch_size, num_classes = logits.data.shape
        # Compute log-sum-exp for numerical stability
        max_logits = xp.max(logits.data, axis=1, keepdims=True)
        shifted = logits.data - max_logits
        exp_shifted = xp.exp(shifted)
        log_sum_exp = max_logits + xp.log(xp.sum(exp_shifted, axis=1, keepdims=True))
        
        # Compute the log-probabilities
        log_probs = logits.data - log_sum_exp
        # Gather the log-probabilities of the correct classes
        target_log_probs = log_probs[xp.arange(batch_size), targets.data.astype(int)]
        
        # Compute mean loss
        loss_data = -xp.mean(target_log_probs)
        loss = Tensor(loss_data, requires_grad=logits.requires_grad)

        if loss.requires_grad:
            loss._parents = [logits, targets]

            def _backward(grad_output):
                # Compute softmax probabilities
                probs = xp.exp(log_probs)
                # Create one-hot encoding of targets
                one_hot = xp.zeros_like(probs)
                one_hot[xp.arange(batch_size), targets.data.astype(int)] = 1
                # Gradient w.r.t logits
                grad_logits = ((probs - one_hot) / batch_size) * grad_output
                logits.grad = logits.grad + grad_logits if logits.grad is not None else grad_logits
            
            loss._grad_fn = _backward

        return loss