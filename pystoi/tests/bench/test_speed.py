import numpy as np


def test_pystoi_reference(benchmark):
    from pystoi import stoi

    sr = 8_000
    seconds = 3
    batch_siwe = 16
    size = batch_siwe * seconds * sr
    x = np.random.random(size).astype(np.float32)
    y = np.random.random(size).astype(np.float32)
    benchmark(stoi, x, y, sr, False)


def test_ours(benchmark):
    from stoi import stoi

    sr = 8_000
    seconds = 3
    batch_siwe = 16
    size = batch_siwe * seconds * sr
    x = np.random.random(size).astype(np.float32)
    y = np.random.random(size).astype(np.float32)
    benchmark(stoi, x, y, sr, False)
