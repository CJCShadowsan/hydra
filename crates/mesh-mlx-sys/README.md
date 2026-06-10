# mesh-mlx-sys

Raw FFI bindings to the [MLX C API](https://github.com/ml-explore/mlx-c) (`mlx-c`)
for the native Rust MLX runtime ([`mesh-mlx`](../mesh-mlx)).

These are hand-written `extern "C"` declarations mirroring the `mlx/c/*.h`
headers (v0.6.x), kept to the focused subset `mesh-mlx` needs: arrays, a handful
of ops, the fused fast kernels (RoPE, SDPA, RMS norm), quantized matmul /
dequantize, safetensors IO, and the distributed collectives (`all_sum`,
`all_gather`, `send`, `recv_like`) + group management.

## Features

- `link-mlx` — build and link the native MLX C++/Metal engine + `mlx-c` (via
  CMake `FetchContent`) on Apple Silicon. **Off by default**: without it the
  crate provides panicking stubs for every FFI symbol so dependents type-check,
  link, and run pure-Rust unit tests on any platform without a Metal build.

## Safety

All MLX C handles are `{ void* ctx }` structs passed by value. Functions
returning `int` return `0` on success. Constructors return a handle the caller
owns and must free. The safe RAII wrappers live in `mesh-mlx`; this crate is the
unsafe boundary.

See [`docs/design/MESH_MLX.md`](../../docs/design/MESH_MLX.md) for the full
architecture.
