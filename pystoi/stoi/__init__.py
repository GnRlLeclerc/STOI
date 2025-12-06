"""Python STOI implementation."""

import numpy as np

from .stoi import par_stoi as par_stoi_internal  # type: ignore
from .stoi import stoi as stoi_internal  # type: ignore

__all__ = ["stoi"]


def stoi(x: np.ndarray, y: np.ndarray, fs_sig: int, extended=False) -> float:
    """
    Compute the Short-Time Objective Intelligibility (STOI) measure between two signals.
    Args:
        x: Clean speech signal (1D array).
        y: Processed speech signal (1D array).
        fs_sig: Sampling frequency of the signals (must be positive).
        extended: Whether to use the extended STOI measure (default: False).
    """

    assert fs_sig > 0, "fs_sig must be positive"
    assert x.shape == y.shape, "x and y must be of the same shapes"
    assert len(x.shape) <= 2, "Arrays must be 1D or 2D"

    x = x.astype(np.float64)
    y = y.astype(np.float64)

    if len(x.shape) == 2:
        par_stoi_internal(x, y, fs_sig, extended)

    try:
        return stoi_internal(x, y, fs_sig, extended)
    except Warning as e:
        print(e, "Returning 1e-5")
        return 1e-5
