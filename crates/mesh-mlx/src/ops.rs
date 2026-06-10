//! Tensor ops used by transformer forward passes — thin safe wrappers over the
//! `mlx-c` ops. Each builds a graph node; nothing is computed until `eval`.

use crate::array::{Array, Stream, binary, check, cstr, unary};
use crate::{MlxError, Result};
use mesh_mlx_sys as sys;

/// Matrix multiply `a @ b`.
pub fn matmul(a: &Array, b: &Array, s: &Stream) -> Result<Array> {
    binary(sys::mlx_matmul, a, b, s, "matmul")
}

/// Elementwise add.
pub fn add(a: &Array, b: &Array, s: &Stream) -> Result<Array> {
    binary(sys::mlx_add, a, b, s, "add")
}

/// Elementwise multiply.
pub fn multiply(a: &Array, b: &Array, s: &Stream) -> Result<Array> {
    binary(sys::mlx_multiply, a, b, s, "multiply")
}

/// Sigmoid.
pub fn sigmoid(a: &Array, s: &Stream) -> Result<Array> {
    unary(sys::mlx_sigmoid, a, s, "sigmoid")
}

/// SiLU / swish: `x * sigmoid(x)`.
pub fn silu(x: &Array, s: &Stream) -> Result<Array> {
    let sig = sigmoid(x, s)?;
    multiply(x, &sig, s)
}

/// Reshape to the given shape.
pub fn reshape(a: &Array, shape: &[i32], s: &Stream) -> Result<Array> {
    let mut res = unsafe { sys::mlx_array_new() };
    let rc = unsafe { sys::mlx_reshape(&mut res, a.raw, shape.as_ptr(), shape.len(), s.raw) };
    check(rc, "reshape")?;
    Ok(Array::from_raw(res))
}

/// Permute axes.
pub fn transpose(a: &Array, axes: &[i32], s: &Stream) -> Result<Array> {
    let mut res = unsafe { sys::mlx_array_new() };
    let rc = unsafe { sys::mlx_transpose_axes(&mut res, a.raw, axes.as_ptr(), axes.len(), s.raw) };
    check(rc, "transpose")?;
    Ok(Array::from_raw(res))
}

/// Concatenate along `axis`.
pub fn concatenate(arrays: &[&Array], axis: i32, s: &Stream) -> Result<Array> {
    let vec = unsafe { sys::mlx_vector_array_new() };
    for a in arrays {
        let rc = unsafe { sys::mlx_vector_array_append_value(vec, a.raw) };
        if rc != 0 {
            unsafe { sys::mlx_vector_array_free(vec) };
            return Err(MlxError::Engine("vector append failed".into()));
        }
    }
    let mut res = unsafe { sys::mlx_array_new() };
    let rc = unsafe { sys::mlx_concatenate_axis(&mut res, vec, axis, s.raw) };
    unsafe { sys::mlx_vector_array_free(vec) };
    check(rc, "concatenate")?;
    Ok(Array::from_raw(res))
}

/// Gather rows along `axis` (used for embedding lookup with `axis=0`).
pub fn take(a: &Array, indices: &Array, axis: i32, s: &Stream) -> Result<Array> {
    let mut res = unsafe { sys::mlx_array_new() };
    let rc = unsafe { sys::mlx_take_axis(&mut res, a.raw, indices.raw, axis, s.raw) };
    check(rc, "take")?;
    Ok(Array::from_raw(res))
}

/// Argmax over the last axis (`keepdims = false`).
pub fn argmax(a: &Array, s: &Stream) -> Result<Array> {
    let mut res = unsafe { sys::mlx_array_new() };
    let rc = unsafe { sys::mlx_argmax(&mut res, a.raw, false, s.raw) };
    check(rc, "argmax")?;
    Ok(Array::from_raw(res))
}

/// Softmax over `axis`.
pub fn softmax(a: &Array, axis: i32, s: &Stream) -> Result<Array> {
    let mut res = unsafe { sys::mlx_array_new() };
    let rc = unsafe { sys::mlx_softmax_axis(&mut res, a.raw, axis, true, s.raw) };
    check(rc, "softmax")?;
    Ok(Array::from_raw(res))
}

/// Fused RMS norm.
pub fn rms_norm(x: &Array, weight: &Array, eps: f32, s: &Stream) -> Result<Array> {
    let mut res = unsafe { sys::mlx_array_new() };
    let rc = unsafe { sys::mlx_fast_rms_norm(&mut res, x.raw, weight.raw, eps, s.raw) };
    check(rc, "rms_norm")?;
    Ok(Array::from_raw(res))
}

/// Fused rotary positional embedding.
pub fn rope(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
    offset: i32,
    s: &Stream,
) -> Result<Array> {
    let base_opt = sys::mlx_optional_float {
        value: base,
        has_value: true,
    };
    let mut res = unsafe { sys::mlx_array_new() };
    let rc = unsafe {
        sys::mlx_fast_rope(
            &mut res,
            x.raw,
            dims,
            traditional,
            base_opt,
            scale,
            offset,
            sys::mlx_array::null(),
            s.raw,
        )
    };
    check(rc, "rope")?;
    Ok(Array::from_raw(res))
}

/// Fused scaled dot-product attention. `mask_mode` is e.g. `"causal"` or `""`.
pub fn scaled_dot_product_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask_mode: &str,
    s: &Stream,
) -> Result<Array> {
    let mode = cstr(mask_mode)?;
    let mut res = unsafe { sys::mlx_array_new() };
    let rc = unsafe {
        sys::mlx_fast_scaled_dot_product_attention(
            &mut res,
            q.raw,
            k.raw,
            v.raw,
            scale,
            mode.as_ptr(),
            sys::mlx_array::null(),
            sys::mlx_array::null(),
            s.raw,
        )
    };
    check(rc, "sdpa")?;
    Ok(Array::from_raw(res))
}
