import numpy as np


def test_pystoi_reference(benchmark):
    from pystoi import stoi

    sr = 8_000
    seconds = 3
    batch_size = 16
    size = batch_size * seconds * sr
    x = np.random.random(size).astype(np.float32)
    y = np.random.random(size).astype(np.float32)
    benchmark(stoi, x, y, sr, False)


def test_torch_stoi_reference(benchmark):
    from torch_stoi import NegSTOILoss
    from torchaudio import torch

    stoi = NegSTOILoss(8_000)

    sr = 8_000
    seconds = 3
    batch_size = 16
    size = batch_size * seconds * sr
    x = torch.randn(size)
    y = torch.randn(size)
    benchmark(stoi, x, y)


def test_ours(benchmark):
    from stoi import stoi

    sr = 8_000
    seconds = 3
    batch_size = 16
    size = batch_size * seconds * sr
    x = np.random.random(size).astype(np.float32)
    y = np.random.random(size).astype(np.float32)
    benchmark(stoi, x, y, sr, False)
