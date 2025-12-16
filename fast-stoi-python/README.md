# Fast STOI

Python bindings for the fast stoi rust library.

It uses numpy `float32` internally for faster simd computations,
and because this is the default type for `pytorch` data.
Calling its functions with `float64` will trigger conversions which may
lose you some performance.

## Installation

```python
pip install fast_stoi
```

## Usage

Compute STOI from numpy data

```python
import numpy as np
from fast_stoi import stoi

x = np.random.random(24_000).astype(np.float32)
y = np.random.random(24_000).astype(np.float32)

score = stoi(x, y, fs_sig=8_000, extended=False)

```

Compute STOI with the torch wrapper.

```python
import torch
from fast_stoi import STOI

stoi = STOI(fs_sig=8_000, extended=False)

x = torch.randn(24_000)
y = torch.randn(24_000)

score = stoi(x, y)
```
