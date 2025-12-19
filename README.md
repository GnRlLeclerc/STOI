# Fast STOI

Rust fast STOI implementation with python bindings.

Inspired from [pystoi](https://github.com/mpariente/pystoi)
and [torch_stoi](https://github.com/mpariente/pytorch_stoi).

The implementation follows the original `STOI` formula with a maximum
error of `1e-7` on random gaussian noise.
It is much faster than `pystoi`, and even faster than the simplified `torch_stoi`
version (which uses a much lighter resampling).

- `fast-stoi/`: Rust implementation
- `fast-stoi-python/`: python bindings available as the `fast_stoi` package

## Installation

Rust _(TODO: not yet published)_:

```bash
cargo add fast_stoi
```

Python _(TODO: not yet published)_:

```bash
pip install fast_stoi
```

For development, running tests or running benchmarks:

```bash
cd fast-stoi-python
uv sync --all-groups
```

## Optimizations

- use [`faer`](https://github.com/sarah-quinones/faer-rs) for fast operations and **simd**
- use `f32` internally for even faster vectorization than `f64`
  _(`pystoi` uses the default `np.float64` internally)_
- abuse **cache locality** with `faer`'s column-major storage layout
- limit allocations and copies
- use `rayon` for parallelism at `rust` level _(whose low overhead makes
  it actually work compared to python's `multiprocessing` for this relatively
  fast computation)_

## Benchmarks

Run on a plugged-in Lenovo Yoga Slim 7 Pro X laptop
(AMD Ryzen 7 6800HS Creator Edition cpu).

Parameters:

- 3s audio samples at 8000Hz as f32
- batches of 16 elements

The `torch_stoi` and `pystoi` version are run without parallelism
on the batched benchmarks (the overhead of `multiprocessing` is too high).

`torch_stoi` is run on CPU only.

| Implementation | Single |        | Batched  |         |
| -------------- | ------ | ------ | -------- | ------- |
| `fast_stoi`    | 1.8 ms |        | 6.1 ms   |         |
| `torch_stoi`   | 2.2 ms | x 1.22 | 42.4 ms  | x 6.92  |
| `pystoi`       | 9.7 ms | x 5.5  | 144.7 ms | x 23.65 |

TODO: extended STOI benchmark results (once the implementation is optimized)

## Develop

Type checking and linting:

```bash
pyrefly check
ruff check
```

Run tests:

```bash
pytest --benchmark-skip
```

Run benchmarks:

```bash
pytest tests/bench/test_speed_standard.py
pytest tests/bench/test_speed_extended.py
```
