import numpy as np
from pystoi import stoi as theirs

from stoi import stoi as ours


def test_standard():
    np.random.seed(42)
    values = []
    for _ in range(100):
        x = np.random.random(24_000)
        y = np.random.random(24_000)

        values.append(theirs(x, y, fs_sig=8_000) - ours(x, y, fs_sig=8_000))

    assert np.all(np.array(values) < 1e-10)
