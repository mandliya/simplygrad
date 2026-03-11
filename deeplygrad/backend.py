"""
deeplygrad.backend - Numpy/Cupy backend switching

Depending on the backend (GPU vs CPU), we can seamlessly switch the backend.
By importing our array library through a single module, we can run the SAME 
autograd code on CPU (NumPy) or GPU (CuPy) with zero code changes. 
This is exactly what libraries like Chainer (CuPy's origin) do internally.

Usage:
   from deeplygrad.backend import xp #xp is either cupy or numpy
   a = xp.array([1, 2, 3]) # works on whatever device is available
"""

import os

# User can force a backend via environment variable:
#   CUTIGRAD_BACKEND=numpy  (force CPU)
#   CUTIGRAD_BACKEND=cupy   (force GPU, will error if no GPU)

_requested = os.environ.get("DEEPLYGRAD_BACKEND", "auto").lower()

if _requested == "cupy":
  import cupy as xp
elif _requested == "numpy":
  import numpy as xp
else:
  try:
    import cupy as xp
    xp.array([0])
  except Exception:
    import numpy as xp

BACKEND_NAME = xp.__name__

def to_numpy(arr):
  """Convert an array to numpy"""
  if BACKEND_NAME == "cupy":
    return arr.get()
  return arr




