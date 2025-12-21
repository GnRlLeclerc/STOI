# Fast STOI

## Installation

```bash
cargo add fast-stoi
```

## Usage

Compute STOI from `f32` slices:

```rust
let x = vec![0.0; 24_000];
let y = vec![0.0; 24_000];

let stoi = fast_stoi::stoi(&x, &y, 8_000, false).unwrap();

```
