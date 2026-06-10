//! Stub implementations of the MLX C API for builds **without** the `link-mlx`
//! feature.
//!
//! These let `mesh-mlx` (and its dependents) compile, link, and run pure-Rust
//! unit tests in CI on any platform without building the native Metal engine.
//! Every stub panics if actually invoked — engine ops require `link-mlx`.
//!
//! Pure-logic code paths (config parsing, pipeline planning, the latency
//! planner, transport/hostfile rendering, loader file selection, shape
//! validation that returns before touching the engine) never call these.
//!
//! # Safety
//! These stubs take the same raw-pointer arguments as the real FFI but never
//! dereference them — they panic immediately. They are `unsafe` only to mirror
//! the real `extern "C"` signatures so call sites compile identically in both
//! configurations.

// Stubs mirror the FFI surface; the safety contract is documented module-wide.
#![allow(clippy::missing_safety_doc)]

use super::*;

macro_rules! stub {
    ($name:ident ( $($arg:ident : $ty:ty),* ) -> $ret:ty) => {
        /// Stub: panics unless built with `link-mlx`. Declared `unsafe` to match
        /// the real `extern "C"` signature so callers' `unsafe` blocks are
        /// necessary in both configurations.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name($(_: $ty),*) -> $ret {
            panic!(concat!(
                stringify!($name),
                ": MLX engine call without the `link-mlx` feature. ",
                "Enable `link-mlx` for real inference."
            ));
        }
    };
    ($name:ident ( $($arg:ident : $ty:ty),* )) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name($(_: $ty),*) {
            panic!(concat!(stringify!($name), ": MLX engine call without `link-mlx`."));
        }
    };
}

// array
stub!(mlx_array_new() -> mlx_array);
stub!(mlx_array_free(a: mlx_array) -> c_int);
stub!(mlx_array_new_data(d: *const c_void, s: *const c_int, n: c_int, t: mlx_dtype) -> mlx_array);
stub!(mlx_array_new_int(v: c_int) -> mlx_array);
stub!(mlx_array_new_float32(v: c_float) -> mlx_array);
stub!(mlx_array_set(a: *mut mlx_array, s: mlx_array) -> c_int);
stub!(mlx_array_itemsize(a: mlx_array) -> size_t);
stub!(mlx_array_size(a: mlx_array) -> size_t);
stub!(mlx_array_nbytes(a: mlx_array) -> size_t);
stub!(mlx_array_ndim(a: mlx_array) -> size_t);
stub!(mlx_array_shape(a: mlx_array) -> *const c_int);
stub!(mlx_array_dim(a: mlx_array, d: c_int) -> c_int);
stub!(mlx_array_dtype(a: mlx_array) -> mlx_dtype);
stub!(mlx_array_eval(a: mlx_array) -> c_int);
stub!(mlx_array_data_float32(a: mlx_array) -> *const c_float);
stub!(mlx_array_data_int32(a: mlx_array) -> *const c_int);
stub!(mlx_array_data_uint32(a: mlx_array) -> *const u32);

// stream
stub!(mlx_stream_free(s: mlx_stream) -> c_int);
stub!(mlx_default_cpu_stream_new() -> mlx_stream);
stub!(mlx_default_gpu_stream_new() -> mlx_stream);

// vector_array
stub!(mlx_vector_array_new() -> mlx_vector_array);
stub!(mlx_vector_array_free(v: mlx_vector_array) -> c_int);
stub!(mlx_vector_array_new_value(v: mlx_array) -> mlx_vector_array);
stub!(mlx_vector_array_append_value(v: mlx_vector_array, a: mlx_array) -> c_int);
stub!(mlx_vector_array_size(v: mlx_vector_array) -> size_t);
stub!(mlx_vector_array_get(r: *mut mlx_array, v: mlx_vector_array, i: size_t) -> c_int);

// map
stub!(mlx_map_string_to_array_new() -> mlx_map_string_to_array);
stub!(mlx_map_string_to_string_new() -> mlx_map_string_to_string);
stub!(mlx_map_string_to_array_free(m: mlx_map_string_to_array) -> c_int);
stub!(mlx_map_string_to_string_free(m: mlx_map_string_to_string) -> c_int);
stub!(mlx_map_string_to_array_iterator_new(m: mlx_map_string_to_array) -> mlx_map_string_to_array_iterator);
stub!(mlx_map_string_to_array_iterator_free(i: mlx_map_string_to_array_iterator) -> c_int);
stub!(mlx_map_string_to_array_iterator_next(k: *mut *const c_char, v: *mut mlx_array, i: mlx_map_string_to_array_iterator) -> c_int);

// io
stub!(mlx_load_safetensors(a: *mut mlx_map_string_to_array, m: *mut mlx_map_string_to_string, f: *const c_char, s: mlx_stream) -> c_int);

// ops
stub!(mlx_matmul(r: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int);
stub!(mlx_add(r: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int);
stub!(mlx_multiply(r: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int);
stub!(mlx_reshape(r: *mut mlx_array, a: mlx_array, sh: *const c_int, n: size_t, s: mlx_stream) -> c_int);
stub!(mlx_transpose_axes(r: *mut mlx_array, a: mlx_array, ax: *const c_int, n: size_t, s: mlx_stream) -> c_int);
stub!(mlx_astype(r: *mut mlx_array, a: mlx_array, t: mlx_dtype, s: mlx_stream) -> c_int);
stub!(mlx_concatenate_axis(r: *mut mlx_array, a: mlx_vector_array, ax: c_int, s: mlx_stream) -> c_int);
stub!(mlx_take_axis(r: *mut mlx_array, a: mlx_array, i: mlx_array, ax: c_int, s: mlx_stream) -> c_int);
stub!(mlx_argmax(r: *mut mlx_array, a: mlx_array, k: bool, s: mlx_stream) -> c_int);
stub!(mlx_sigmoid(r: *mut mlx_array, a: mlx_array, s: mlx_stream) -> c_int);
stub!(mlx_softmax_axis(r: *mut mlx_array, a: mlx_array, ax: c_int, p: bool, s: mlx_stream) -> c_int);

// fast
stub!(mlx_fast_rms_norm(r: *mut mlx_array, x: mlx_array, w: mlx_array, e: c_float, s: mlx_stream) -> c_int);
stub!(mlx_fast_rope(r: *mut mlx_array, x: mlx_array, d: c_int, t: bool, b: mlx_optional_float, sc: c_float, o: c_int, f: mlx_array, s: mlx_stream) -> c_int);
stub!(mlx_fast_scaled_dot_product_attention(r: *mut mlx_array, q: mlx_array, k: mlx_array, v: mlx_array, sc: c_float, m: *const c_char, ma: mlx_array, si: mlx_array, s: mlx_stream) -> c_int);

// distributed group
stub!(mlx_distributed_group_new() -> mlx_distributed_group);
stub!(mlx_distributed_group_free(g: mlx_distributed_group) -> c_int);
stub!(mlx_distributed_init(r: *mut mlx_distributed_group, st: bool, bk: *const c_char) -> c_int);
stub!(mlx_distributed_group_rank(g: mlx_distributed_group) -> c_int);
stub!(mlx_distributed_group_size(g: mlx_distributed_group) -> c_int);
stub!(mlx_distributed_group_split(r: *mut mlx_distributed_group, g: mlx_distributed_group, c: c_int, k: c_int) -> c_int);
stub!(mlx_distributed_is_available(bk: *const c_char) -> bool);

// distributed collectives
stub!(mlx_distributed_all_sum(r: *mut mlx_array, x: mlx_array, g: mlx_distributed_group, s: mlx_stream) -> c_int);
stub!(mlx_distributed_all_gather(r: *mut mlx_array, x: mlx_array, g: mlx_distributed_group, s: mlx_stream) -> c_int);
stub!(mlx_distributed_send(r: *mut mlx_array, x: mlx_array, d: c_int, g: mlx_distributed_group, s: mlx_stream) -> c_int);
stub!(mlx_distributed_recv_like(r: *mut mlx_array, x: mlx_array, sr: c_int, g: mlx_distributed_group, s: mlx_stream) -> c_int);
