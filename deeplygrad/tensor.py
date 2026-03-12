"""
deeplygrad.tensor - A Tensor with automatic differentiation

This Tensor library is wrapper over numpy/cupy and supports automatic differentiation
for tensor operation like PyTorch. Every Tensor records how it was created
(which operation, what inputs), forming a Directed Acyclic Graph (DAG). When you 
call .backward(), we walk this DAG in reverse topological order, applying chain
rule at each node.

Key concepts:

1. Compute graph contruction (eager mode like pytorch)
2. Reverse-mode autodiff (backpropagation)
3. Topological sort for correct gradient ordering
4. Gradient accumulation for shared nodes
"""

from typing import Any, List, Tuple, Optional, Union, Callable
from deeplygrad.backend import xp, to_numpy
import numpy as np
from contextlib import contextmanager

BackwardFn = Callable[[np.ndarray], None]

class Tensor:
  """
  A multi-dimensional array with automatic differentiation support
  
  Parameters:
  -------------
  data: array-like
    The acutal array data (stored as xp.ndarray - numpy or cupy)
  requires_grad: bool
    If True, operations on this tensor will be tracked for backprop
  """

  def __init__(self, data: xp.ndarray | np.ndarray, requires_grad: bool=False):
    # Coerce to xp array if not already
    if isinstance(data, xp.ndarray):
      self.data = data
    elif isinstance(data, np.ndarray):
      self.data = xp.array(data)
    else:
      self.data = xp.array(data, dtype=xp.float64)
    
    self.requires_grad = requires_grad
    
    # grad will hold dL/d(self) after backward
    self.grad: Optional[xp.ndarray] = None

    # Computational graph metadata
    # _grad_fn: Implements the backward pass for the operation that created
    #   this tensor. Given the gradient of the loss w.r.t. this tensor's output
    #   (the upstream gradient), it computes the gradient w.r.t. each input
    #   (using the chain rule) and accumulates them into the parents' .grad.
    #
    #   For example, if this tensor was created by `out = a * b`, then
    #   _grad_fn(grad_output) computes:
    #       a.grad += grad_output * b   (local derivative w.r.t. a is b)
    #       b.grad += grad_output * a   (local derivative w.r.t. b is a)
    self._grad_fn: Optional[BackwardFn] = None

    # _parents: the tensors that were inputs to the operation that created 
    # this tensor.
    self._parents: List[Tensor] = []

  
  @property
  def shape(self) -> tuple:
    return self.data.shape

  @property
  def dtype(self) -> np.dtype[np.floating]:
    return self.data.dtype

  @property
  def ndim(self) -> int:
    return self.data.ndim

  @property
  def T(self) -> 'Tensor':
    """Transpose (shorthand, like numpy .T)."""
    return self.transpose()
  
  def size(self, dim:Optional[int]=None):
    if dim is None:
      return self.shape
    return self.shape[dim]
  
  def transpose(self, *axes):
    """Transpose dimensions"""
    if len(axes) == 0:
      # Default: reverse all axes (like .T)
      axes_tuple = None
      reverse_axes = None
    elif len(axes) == 2:
      axes_tuple = list(range(self.ndim))
      axes_tuple[axes[0]], axes_tuple[axes[1]] = axes_tuple[axes[1]], axes_tuple[axes[0]]
      reverse_axes = axes_tuple
    else:
      raise ValueError("transpose expects 0 or 2 axis arguments")

    out_data = self.data.transpose(axes_tuple)
    out = Tensor(out_data, requires_grad=self.requires_grad)
    if out.requires_grad:
      out._parents = [self]
      
      def _backward(grad_output):
        if self.requires_grad:
          if reverse_axes is not None:
            g = grad_output.transpose()
          else:
            g = grad_output.transpose(reverse_axes)
          
          self.grad = self.grad + g if self.grad is not None else g
        
      out._grad_fn = _backward
    
    return out

  def backward(self, grad: Optional[xp.ndarray] = None):
    """
    Compute gradients via reverse-mode autodiff

    How it works:
      1. Start from this tensor (typically the loss) with gradient = 1
      2. Topologically sort the computation graph.
      3. Walk in the reverse order. At each node, call its _grad_fn to 
      propagate gradients to it's parents
    
    Why toposort?
    A tensor might be used in multiple downstream operations. We need to make
    sure we all downstream gradients are accumulated before we propagate
    further back. Topological sort guarantees this.
    """ 

    if not self.requires_grad:
      return RuntimeError("backward() called ona tensor that doesn't require grad")
    
    if grad is None:
      if self.data.size != 1:
        raise RuntimeError(
          "backward() without a gradient argument is only valid for scalar tensors. "
          f"Got shape {self.shape}. Pass a gradient tensor explicitly."
        )
      grad = xp.ones_like(self.data)
    
    # Step 1: topological sort via DFS
    topo_order: List[Tensor] = []
    visited = set()

    def _build_topo(tensor: Tensor):
      if id(tensor) not in visited:
        visited.add(id(tensor))
        for parent in tensor._parents:
          _build_topo(parent)
        topo_order.append(tensor)
      
    _build_topo(self)

    # Step 2: Reverse walk - propagate gradients
    self.grad = grad
    for tensor in reversed(topo_order):
      if tensor._grad_fn is not None:
        tensor._grad_fn(tensor.grad)
    
  def zero_grad(self):
    self.grad = None

  
  def detach(self):
    """Return a new Tensor sharing data but detached from the graph."""
    return Tensor(self.data, requires_grad=False)

  def add(self, other: Union['Tensor', float, int]) -> 'Tensor':
    """Element wise addition: out = self + other """
    other = _ensure_tensor(other)
    out_data = self.data + other.data
    out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)

    if out.requires_grad:
      out._parents = [self, other]
    
      def _backward(grad_output):
        if self.requires_grad:
          g = _unbroadcast(grad_output, self.shape)
          self.grad = self.grad + g if self.grad is not None else g
        if other.requires_grad:
          g = _unbroadcast(grad_output, other.shape)
          other.grad = other.grad + g if other.grad is not None else g
      
      out._grad_fn = _backward
    
    return out

  def mul(self, other: Union['Tensor', float, int]) -> 'Tensor':
    """Element-wise multiplication: out = self * other"""
    other = _ensure_tensor(other)
    out_data = self.data * other.data
    out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)

    if out.requires_grad:
      out._parents = [self, other]

      def _backward(grad_output):
        if self.requires_grad:
          g = _unbroadcast(grad_output * other.data, self.shape)
          self.grad = self.grad + g if self.grad is not None else g
        if other.requires_grad:
          g = _unbroadcast(grad_output * self.data, other.shape)
          other.grad = other.grad + g if other.grad is not None else g
      
      out._grad_fn = _backward

    return out

  
  def neg(self) -> 'Tensor':
    """Negation: out = -self"""
    return self.mul(Tensor(xp.array(-1.0)))

  def sub(self, other: Union['Tensor', float, int]) -> 'Tensor':
    """Substraction: out = self - other"""
    other = _ensure_tensor(other)
    return self.add(other.neg())
  
  def pow(self, exponent: float) -> 'Tensor':
    out_data = self.data ** exponent
    out = Tensor(out_data, requires_grad=self.requires_grad)

    if out.requires_grad:
      out._parents = [self]

      def _backward(grad_output):
        g = grad_output * exponent * (self.data ** (exponent - 1))
        self.grad = self.grad + g if self.grad is not None else g
      
      out._grad_fn = _backward
    return out


  def div(self, other: Union['Tensor', float, int]) -> 'Tensor':
    """Division: out = self / other (via self * other^-1 )"""
    other = _ensure_tensor(other)
    return self.mul(other.pow(-1))


  def matmul(self, other: 'Tensor') -> 'Tensor':
    """
    Matrix multiplication: out = self @ other
    Gradient derivation:
      If C = A @ B, then:
        dL/dA = dL/dC @ B^T
        dL/dB = A^T @ dL/dC
    """

    out_data = self.data @ other.data
    out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)

    if out.requires_grad:
      out._parents = [self, other]

      def _backward(grad_output):
        if self.requires_grad:
          if other.data.ndim == 1:
            g = xp.outer(grad_output, other.data)
          else:
            g = grad_output @ xp.swapaxes(other.data, -1, -2)
          
          g = _unbroadcast(g, self.shape)
          self.grad = self.grad + g if self.grad is not None else g

        if other.requires_grad:
          if self.data.ndim == 1:
            g = xp.outer(self.data, grad_output)
          else:
            g = xp.swapaxes(self.data, -1, -2) @ grad_output
          g = _unbroadcast(g, other.shape)
          other.grad = other.grad + g if other.grad is not None else g

      out._grad_fn = _backward    
    return out


  def sum(self, axis=None, keepdims=False) -> 'Tensor':
    """
    Summation over aixs

    Gradient: Sum is a many to one op. The gradient just broadcasts
    the upstream scaler gradient back to the input shape
    """
    out_data = self.data.sum(axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=self.requires_grad)

    if out.requires_grad:
      out._parents = [self]

      def _backward(grad_output):
        if self.requires_grad:
          if axis is not None and not keepdims:
            # Re-insert the reduced axis so broadcasting works
            g = xp.expand_dims(grad_output, axis=axis)
          else:
            g = grad_output
          
          g = xp.broadcast_to(g, self.shape).copy()
          self.grad = self.grad + g if self.grad is not None else g
        
      out._grad_fn = _backward
    return out

  def mean(self, axis=None, keepdims=False) -> 'Tensor':
    """ Mean = sum /count """
    if axis is None:
      n = self.data.size
    elif isinstance(axis, int):
      n = self.shape[axis]
    else:
      n = 1
      for ax in axis:
        n *= self.shape[ax]
    return self.sum(axis=axis, keepdims=keepdims).div(Tensor(xp.array(float(n))))

  
  def reshape(self, *shape) -> 'Tensor':
    """Reshape tensor (gradient reshapes back)"""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
      shape = tuple(shape[0])
    out_data = self.data.reshape(shape)
    out = Tensor(out_data, requires_grad=self.requires_grad)

    if out.requires_grad:
      out._parents = [self]

      def _backward(grad_output):
        if self.requires_grad:
          g = grad_output.reshape(self.shape)
          self.grad = self.grad + g if self.grad is not None else g

      out._grad_fn = _backward
    return out  

  def exp(self) -> 'Tensor':
    out_data = xp.exp(self.data)
    out = Tensor(out_data, requires_grad=self.requires_grad)

    if out.requires_grad:
      out._parents = [self]

      def _backward(grad_output):
        g = grad_output * out_data
        self.grad = self.grad + g if self.grad is not None else g
      
      out._grad_fn = _backward
    return out

  
  def log(self) -> 'Tensor':
    """Natural log: out = ln(self)"""
    out_data = xp.log(self.data)
    out = Tensor(out_data, requires_grad=self.requires_grad)

    if out.requires_grad:
      out._parents = [self]

      def _backward(grad_output):
        g = grad_output * (1 / self.data)
        self.grad = self.grad + g if self.grad is not None else g
      
      out._grad_fn = _backward
    return out

  def max(self, axis=None, keepdims=False) -> 'Tensor':
    """Max along the axis (gradient flow only to the max element)"""
    out_data = self.data.max(axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=self.requires_grad)

    if out.requires_grad:
      out._parents = [self]

      def _backward(grad_output):
        if self.requires_grad:
          if axis is not None and not keepdims:
            max_expandad = xp.expand_dims(out_data, axis=axis)
            grad_expanded = xp.expand_dims(grad_output, axis=axis)
          else:
            max_expandad = out_data
            grad_expanded = grad_output
          mask = (self.data == max_expandad).astype(self.data.dtype)

          # Normalize mask in case of ties
          mask_sum = mask.sum(axis=axis, keepdims=True)
          mask = mask / xp.maximum(mask_sum, 1.0)

          g = xp.broadcast_to(grad_expanded, self.shape) * mask
          self.grad = self.grad + g if self.grad is not None else g
      out._grad_fn = _backward
    return out

  def __getitem__(self, idx) -> 'Tensor':
      out_data = self.data[idx]
      out = Tensor(out_data, requires_grad=self.requires_grad)

      if out.requires_grad:
          out._parents = [self]

          def _backward(grad_output):
              if self.requires_grad:
                  g = xp.zeros_like(self.data)
                  g[idx] = grad_output  # scatter gradient back
                  self.grad = self.grad + g if self.grad is not None else g

          out._grad_fn = _backward
      return out

  def __add__(self, other):       return self.add(other)
  def __radd__(self, other):      return _ensure_tensor(other).add(self)
  def __mul__(self, other):       return self.mul(other)
  def __rmul__(self, other):      return _ensure_tensor(other).mul(self)
  def __sub__(self, other):       return self.sub(other)
  def __rsub__(self, other):      return _ensure_tensor(other).sub(self)
  def __truediv__(self, other):   return self.div(other)
  def __rtruediv__(self, other):  return _ensure_tensor(other).div(self)
  def __neg__(self):              return self.neg()
  def __matmul__(self, other):    return self.matmul(other)
  def __rmatmul__(self, other):   return _ensure_tensor(other).matmul(self)
  def __pow__(self, exp):         return self.pow(exp)
  def __float__(self):            return float(self.data)
  def __int__(self):              return int(self.data)
  def __bool__(self):
    if self.data.size != 1:
      raise ValueError(
        "The truth value of a Tensor with more than one element is ambiguous. "
        "Use .sum(), .any() or .all()."
      )
    return bool(self.data)
  def __str__(self):              return str(self.data)
  def __len__(self):              return len(self.data)

  def __gt__(self, other):
    other = _ensure_tensor(other)
    return Tensor((self.data > other.data).astype(self.data.dtype))

  def __ge__(self, other):
    other = _ensure_tensor(other)
    return Tensor((self.data >= other.data).astype(self.data.dtype))
  
  def __lt__(self, other):
    other = _ensure_tensor(other)
    return Tensor((self.data < other.data).astype(self.data.dtype))
    
  def __le__(self, other):
    other = _ensure_tensor(other)
    return Tensor((self.data <= other.data).astype(self.data.dtype))

  def __eq__(self, other):
    other = _ensure_tensor(other)
    return Tensor((self.data == other.data).astype(self.data.dtype))

  def __repr__(self):
      grad_info = ", requires_grad=True" if self.requires_grad else ""
      return f"Tensor({to_numpy(self.data)}{grad_info})"

  def item(self):
      if self.data.size != 1:
          raise ValueError(
              f"item() can only be called on single-element tensors, got {self.shape}"
          )
      return self.data.item()

  def numpy(self):
      """Return data as a numpy array (moves off GPU if needed)."""
      return to_numpy(self.data)

  @contextmanager
  def no_grad(self):
    """Context manager to temporarily disable gradient tracking."""
    was_requires_grad = self.requires_grad
    self.requires_grad = False
    try:
      yield
    finally:
      self.requires_grad = was_requires_grad

### Helpers

def _ensure_tensor(x: Any) -> Tensor:
  if isinstance(x, Tensor):
    return x
  return Tensor(x, requires_grad=False)


def _unbroadcast(grad: xp.ndarray, target_shape: tuple) -> xp.ndarray:
  """
  Reduce the grad back to target_shape by summing over broadcast dimensions.

  Why?

    When we do `a + b` and a is (3,4) but b is (4,), NumPy broadcasts b.
    The gradient w.r.t. b must sum over the broadcast dimension (axis 0)
    to get back to shape (4,). This function handles that automatically.
  """
  if grad.shape == target_shape:
    return grad

  ndim_diff = grad.ndim - len(target_shape)
  padded_shape = (1, ) * ndim_diff + target_shape

  reduce_axes = []
  for i, (gs, ts) in enumerate(zip(grad.shape, padded_shape)):
    if ts == 1 and gs > 1:
      reduce_axes.append(i)
    elif ts == 1 and gs == 1:
      # Was a leading dimension added by broadcasting — also reduce
      if i < ndim_diff:
        reduce_axes.append(i)
  
  if reduce_axes:
    grad = grad.sum(axis=tuple(reduce_axes), keepdims=True)

  return grad.reshape(target_shape)


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Element-wise selection: out_i = x_i if condition_i else y_i"""
    x = _ensure_tensor(x)
    y = _ensure_tensor(y)
    cond = condition.data.astype(bool)
    out_data = xp.where(cond, x.data, y.data)
    out = Tensor(out_data, requires_grad=x.requires_grad or y.requires_grad)

    if out.requires_grad:
        out._parents = [x, y]

        def _backward(grad_output):
          if x.requires_grad:
            g = xp.where(cond, grad_output, xp.zeros_like(grad_output))
            g = _unbroadcast(g, x.shape)
            x.grad = x.grad + g if x.grad is not None else g
          if y.requires_grad:
            g = xp.where(cond, xp.zeros_like(grad_output), grad_output)
            g = _unbroadcast(g, y.shape)
            y.grad = y.grad + g if y.grad is not None else g
        out._grad_fn = _backward
    return out
