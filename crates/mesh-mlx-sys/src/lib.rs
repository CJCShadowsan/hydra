//! Raw FFI bindings to the [MLX C API](https://github.com/ml-explore/mlx-c).
//!
//! These mirror the `mlx/c/*.h` headers (v0.6.x). They are hand-written rather
//! than bindgen-generated to keep the surface small and reviewable — `mesh-mlx`
//! only needs a focused subset (arrays, a handful of ops, fast kernels, IO,
//! random, and the distributed collectives) to implement LLM inference.
//!
//! All MLX C handles are `{ void* ctx }` structs passed by value. Functions
//! returning `int` return `0` on success and non-zero on error. Constructors
//! return the handle by value; the caller owns it and must `*_free` it. The
//! safe wrappers in `mesh-mlx` enforce ownership via RAII.
//!
//! Without the `link-mlx` feature this crate provides the type/function
//! declarations but is not linked against the native engine, so dependents can
//! type-check pure-Rust logic in CI without a Metal build.

#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use std::os::raw::{c_char, c_float, c_int, c_void};

#[cfg(not(feature = "link-mlx"))]
mod stubs;
#[cfg(not(feature = "link-mlx"))]
pub use stubs::*;

pub type size_t = usize;

// ---------------------------------------------------------------------------
// Opaque handle types (all are `{ void* ctx }`)
// ---------------------------------------------------------------------------

macro_rules! mlx_handle {
    ($name:ident) => {
        #[repr(C)]
        #[derive(Copy, Clone)]
        pub struct $name {
            pub ctx: *mut c_void,
        }
    };
}

mlx_handle!(mlx_array);
mlx_handle!(mlx_stream);
mlx_handle!(mlx_device);
mlx_handle!(mlx_vector_array);
mlx_handle!(mlx_map_string_to_array);
mlx_handle!(mlx_map_string_to_string);
mlx_handle!(mlx_map_string_to_array_iterator);
mlx_handle!(mlx_string);
mlx_handle!(mlx_distributed_group);

/// MLX dtype enum (order matches `mlx/c/array.h`).
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum mlx_dtype {
    MLX_BOOL = 0,
    MLX_UINT8,
    MLX_UINT16,
    MLX_UINT32,
    MLX_UINT64,
    MLX_INT8,
    MLX_INT16,
    MLX_INT32,
    MLX_INT64,
    MLX_FLOAT16,
    MLX_FLOAT32,
    MLX_FLOAT64,
    MLX_BFLOAT16,
    MLX_COMPLEX64,
}

/// `mlx_optional_float` from `mlx/c/types.h` — a float with a validity flag.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct mlx_optional_float {
    pub value: c_float,
    pub has_value: bool,
}

// ---------------------------------------------------------------------------
// FFI declarations
// ---------------------------------------------------------------------------

#[cfg(feature = "link-mlx")]
unsafe extern "C" {
    // ---- array ----
    pub fn mlx_array_new() -> mlx_array;
    pub fn mlx_array_free(arr: mlx_array) -> c_int;
    pub fn mlx_array_new_data(
        data: *const c_void,
        shape: *const c_int,
        dim: c_int,
        dtype: mlx_dtype,
    ) -> mlx_array;
    pub fn mlx_array_new_int(val: c_int) -> mlx_array;
    pub fn mlx_array_new_float32(val: c_float) -> mlx_array;
    pub fn mlx_array_set(arr: *mut mlx_array, src: mlx_array) -> c_int;
    pub fn mlx_array_itemsize(arr: mlx_array) -> size_t;
    pub fn mlx_array_size(arr: mlx_array) -> size_t;
    pub fn mlx_array_nbytes(arr: mlx_array) -> size_t;
    pub fn mlx_array_ndim(arr: mlx_array) -> size_t;
    pub fn mlx_array_shape(arr: mlx_array) -> *const c_int;
    pub fn mlx_array_dim(arr: mlx_array, dim: c_int) -> c_int;
    pub fn mlx_array_dtype(arr: mlx_array) -> mlx_dtype;
    pub fn mlx_array_eval(arr: mlx_array) -> c_int;
    pub fn mlx_array_data_float32(arr: mlx_array) -> *const c_float;
    pub fn mlx_array_data_int32(arr: mlx_array) -> *const c_int;
    pub fn mlx_array_data_uint32(arr: mlx_array) -> *const u32;

    // ---- stream / device ----
    pub fn mlx_stream_free(stream: mlx_stream) -> c_int;
    pub fn mlx_default_cpu_stream_new() -> mlx_stream;
    pub fn mlx_default_gpu_stream_new() -> mlx_stream;

    // ---- vector_array ----
    pub fn mlx_vector_array_new() -> mlx_vector_array;
    pub fn mlx_vector_array_free(vec: mlx_vector_array) -> c_int;
    pub fn mlx_vector_array_new_value(val: mlx_array) -> mlx_vector_array;
    pub fn mlx_vector_array_append_value(vec: mlx_vector_array, val: mlx_array) -> c_int;
    pub fn mlx_vector_array_size(vec: mlx_vector_array) -> size_t;
    pub fn mlx_vector_array_get(res: *mut mlx_array, vec: mlx_vector_array, idx: size_t) -> c_int;

    // ---- map (safetensors result) ----
    pub fn mlx_map_string_to_array_new() -> mlx_map_string_to_array;
    pub fn mlx_map_string_to_string_new() -> mlx_map_string_to_string;
    pub fn mlx_map_string_to_array_free(map: mlx_map_string_to_array) -> c_int;
    pub fn mlx_map_string_to_string_free(map: mlx_map_string_to_string) -> c_int;
    pub fn mlx_map_string_to_array_iterator_new(
        map: mlx_map_string_to_array,
    ) -> mlx_map_string_to_array_iterator;
    pub fn mlx_map_string_to_array_iterator_free(it: mlx_map_string_to_array_iterator) -> c_int;
    /// Returns non-zero when iteration is exhausted.
    pub fn mlx_map_string_to_array_iterator_next(
        key: *mut *const c_char,
        value: *mut mlx_array,
        it: mlx_map_string_to_array_iterator,
    ) -> c_int;

    // ---- io ----
    pub fn mlx_load_safetensors(
        res_0: *mut mlx_map_string_to_array,
        res_1: *mut mlx_map_string_to_string,
        file: *const c_char,
        s: mlx_stream,
    ) -> c_int;

    // ---- core ops ----
    pub fn mlx_matmul(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_add(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_multiply(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_reshape(
        res: *mut mlx_array,
        a: mlx_array,
        shape: *const c_int,
        shape_num: size_t,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_transpose_axes(
        res: *mut mlx_array,
        a: mlx_array,
        axes: *const c_int,
        axes_num: size_t,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_astype(res: *mut mlx_array, a: mlx_array, dtype: mlx_dtype, s: mlx_stream) -> c_int;
    pub fn mlx_concatenate_axis(
        res: *mut mlx_array,
        arrays: mlx_vector_array,
        axis: c_int,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_take_axis(
        res: *mut mlx_array,
        a: mlx_array,
        indices: mlx_array,
        axis: c_int,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_argmax(res: *mut mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) -> c_int;
    pub fn mlx_sigmoid(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_softmax_axis(
        res: *mut mlx_array,
        a: mlx_array,
        axis: c_int,
        precise: bool,
        s: mlx_stream,
    ) -> c_int;

    // ---- fast kernels ----
    pub fn mlx_fast_rms_norm(
        res: *mut mlx_array,
        x: mlx_array,
        weight: mlx_array,
        eps: c_float,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_fast_rope(
        res: *mut mlx_array,
        x: mlx_array,
        dims: c_int,
        traditional: bool,
        base: mlx_optional_float,
        scale: c_float,
        offset: c_int,
        freqs: mlx_array,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_fast_scaled_dot_product_attention(
        res: *mut mlx_array,
        queries: mlx_array,
        keys: mlx_array,
        values: mlx_array,
        scale: c_float,
        mask_mode: *const c_char,
        mask_arr: mlx_array,
        sinks: mlx_array,
        s: mlx_stream,
    ) -> c_int;

    // ---- distributed group ----
    pub fn mlx_distributed_group_new() -> mlx_distributed_group;
    pub fn mlx_distributed_group_free(group: mlx_distributed_group) -> c_int;
    /// `strict` forces a real backend (errors if unavailable).
    pub fn mlx_distributed_init(
        res: *mut mlx_distributed_group,
        strict: bool,
        bk: *const c_char,
    ) -> c_int;
    pub fn mlx_distributed_group_rank(group: mlx_distributed_group) -> c_int;
    pub fn mlx_distributed_group_size(group: mlx_distributed_group) -> c_int;
    pub fn mlx_distributed_group_split(
        res: *mut mlx_distributed_group,
        group: mlx_distributed_group,
        color: c_int,
        key: c_int,
    ) -> c_int;
    pub fn mlx_distributed_is_available(bk: *const c_char) -> bool;

    // ---- distributed collectives ----
    pub fn mlx_distributed_all_sum(
        res: *mut mlx_array,
        x: mlx_array,
        group: mlx_distributed_group,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_distributed_all_gather(
        res: *mut mlx_array,
        x: mlx_array,
        group: mlx_distributed_group,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_distributed_send(
        res: *mut mlx_array,
        x: mlx_array,
        dst: c_int,
        group: mlx_distributed_group,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_distributed_recv_like(
        res: *mut mlx_array,
        x: mlx_array,
        src: c_int,
        group: mlx_distributed_group,
        s: mlx_stream,
    ) -> c_int;
}

/// A null handle (`ctx == NULL`), used where the C API documents "may be null".
impl mlx_array {
    pub const fn null() -> Self {
        mlx_array {
            ctx: std::ptr::null_mut(),
        }
    }
}
impl mlx_distributed_group {
    pub const fn null() -> Self {
        mlx_distributed_group {
            ctx: std::ptr::null_mut(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pure type-level checks that compile without linking the engine.
    #[test]
    fn handles_are_pointer_sized() {
        assert_eq!(
            std::mem::size_of::<mlx_array>(),
            std::mem::size_of::<*mut c_void>()
        );
        assert_eq!(mlx_dtype::MLX_FLOAT32 as i32, 10);
        assert_eq!(mlx_dtype::MLX_BFLOAT16 as i32, 12);
        assert!(mlx_array::null().ctx.is_null());
    }
}
