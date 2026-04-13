"""
deeplygrad — A tiny autograd engine built on CuPy/NumPy.

Built from scratch for learning. Every operation records its gradient
function, forming a computation graph that backward() traverses.
"""

from deeplygrad.tensor import Tensor
from deeplygrad.backend import xp, BACKEND_NAME

__version__ = "0.1.0"
__all__ = [
    "Tensor", "xp", "BACKEND_NAME", "Module", "Linear", "ReLU", 
    "CrossEntropyLoss", "Adam", "SGD", "cat", "stack"]