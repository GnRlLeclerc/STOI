"""Fast STOI implementation."""

import warnings

import numpy as np

from .fast_stoi import par_stoi as par_stoi_internal  # type: ignore
from .fast_stoi import stoi as stoi_internal  # type: ignore

__all__ = ["stoi", "STOI"]

ERROR_MESSAGE = (
    "Not enough STFT frames to compute intermediate "
    "intelligibility measure after removing silent "
    "frames. Returning 1e-5. Please check you wav files"
)


def stoi(x: np.ndarray, y: np.ndarray, fs_sig: int, extended=False) -> np.ndarray:
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

    if x.dtype != np.float32:
        x = x.astype(np.float32)
    if y.dtype != np.float32:
        y = y.astype(np.float32)

    if len(x.shape) == 2:
        out = par_stoi_internal(x, y, fs_sig, extended)
        if np.any(out == 1e-5):
            warnings.warn(ERROR_MESSAGE)
        return out

    try:
        out = stoi_internal(x, y, fs_sig, extended)
    except Warning:
        warnings.warn(ERROR_MESSAGE)
        out = 1e-5

    return np.array(out)


try:
    from torch import Tensor, nn, tensor

    class STOI(nn.Module):
        """
        Compute the Short-Time Objective Intelligibility (STOI) measure between two signals.
        Args:
            fs_sig: Sampling frequency of the signals (must be positive).
            extended: Whether to use the extended STOI measure (default: False).
        """

        def __init__(self, fs_sig: int, extended: bool = False):
            super().__init__()
            self.fs_sig = fs_sig
            self.extended = extended

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            """
            Args:
                x: Clean speech signal (1D tensor).
                y: Processed speech signal (1D tensor).
            """
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()

            return tensor(stoi(x_np, y_np, self.fs_sig, self.extended))

except ImportError:
    pass  # declare something that throws when you import it
