"""Python STOI implementation."""

import numpy as np
from numpy.typing import NDArray

from .stoi import stoi as stoi_internal  # type: ignore

__all__ = ["stoi"]


def stoi(
    x: NDArray[np.float64], y: NDArray[np.float64], fs_sig: int, extended=False
) -> float:
    """
    Compute the Short-Time Objective Intelligibility (STOI) measure between two signals.
    Args:
        x (NDArray[np.float64]): Clean speech signal (1D array).
        y (NDArray[np.float64]): Processed speech signal (1D array).
        fs_sig (int): Sampling frequency of the signals (must be positive).
        extended (bool): Whether to use the extended STOI measure (default: False).
    """

    assert fs_sig > 0, "fs_sig must be positive"
    assert x.ndim == 1, "x must be 1D"
    assert y.ndim == 1, "y must be 1D"
    assert x.dtype == np.float64, "x must be float64"
    assert y.dtype == np.float64, "y must be float64"

    return stoi_internal(x, y, fs_sig, extended)
