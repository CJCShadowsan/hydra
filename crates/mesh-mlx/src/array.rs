//! Safe RAII wrappers over `mlx_array` and `mlx_stream`.
//!
//! Every [`Array`] owns its underlying `mlx_array` handle and frees it on drop.
//! Ops are methods that allocate a fresh result handle and check the C return
//! code, returning [`MlxError`] on failure. Lazy evaluation is MLX's model:
//! ops build a graph; [`Array::eval`] forces materialisation.

use crate::{MlxError, Result};
use mesh_mlx_sys as sys;
use std::ffi::CString;

/// Element type of an [`Array`], mirroring the subset of `mlx_dtype` we use.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    I32,
    U32,
}

impl Dtype {
    pub(crate) fn to_sys(self) -> sys::mlx_dtype {
        match self {
            Dtype::F32 => sys::mlx_dtype::MLX_FLOAT32,
            Dtype::F16 => sys::mlx_dtype::MLX_FLOAT16,
            Dtype::BF16 => sys::mlx_dtype::MLX_BFLOAT16,
            Dtype::I32 => sys::mlx_dtype::MLX_INT32,
            Dtype::U32 => sys::mlx_dtype::MLX_UINT32,
        }
    }

    pub(crate) fn from_sys(d: sys::mlx_dtype) -> Dtype {
        match d {
            sys::mlx_dtype::MLX_FLOAT16 => Dtype::F16,
            sys::mlx_dtype::MLX_BFLOAT16 => Dtype::BF16,
            sys::mlx_dtype::MLX_INT32 => Dtype::I32,
            sys::mlx_dtype::MLX_UINT32 => Dtype::U32,
            _ => Dtype::F32,
        }
    }
}

/// A compute stream (device queue). Defaults to the GPU (Metal) stream.
pub struct Stream {
    pub(crate) raw: sys::mlx_stream,
}

impl Stream {
    /// The default Metal (GPU) stream — what inference runs on.
    pub fn gpu() -> Self {
        Stream {
            raw: unsafe { sys::mlx_default_gpu_stream_new() },
        }
    }

    /// The default CPU stream.
    pub fn cpu() -> Self {
        Stream {
            raw: unsafe { sys::mlx_default_cpu_stream_new() },
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            sys::mlx_stream_free(self.raw);
        }
    }
}

/// An owned MLX array handle.
pub struct Array {
    pub(crate) raw: sys::mlx_array,
}

impl Array {
    /// Wrap a raw handle taking ownership (used internally after C calls).
    pub(crate) fn from_raw(raw: sys::mlx_array) -> Self {
        Array { raw }
    }

    /// Build an array from a flat `f32` buffer and shape (row-major).
    pub fn from_f32(data: &[f32], shape: &[i32]) -> Result<Self> {
        let want: usize = shape.iter().product::<i32>().max(0) as usize;
        if want != data.len() {
            return Err(MlxError::Shape(format!(
                "data len {} does not match shape {:?} ({} elements)",
                data.len(),
                shape,
                want
            )));
        }
        let raw = unsafe {
            sys::mlx_array_new_data(
                data.as_ptr() as *const _,
                shape.as_ptr(),
                shape.len() as i32,
                Dtype::F32.to_sys(),
            )
        };
        Self::check_handle(raw)
    }

    /// Build an `i32` array (e.g. token ids) from a flat buffer and shape.
    pub fn from_i32(data: &[i32], shape: &[i32]) -> Result<Self> {
        let want: usize = shape.iter().product::<i32>().max(0) as usize;
        if want != data.len() {
            return Err(MlxError::Shape(format!(
                "data len {} != shape {:?}",
                data.len(),
                shape
            )));
        }
        let raw = unsafe {
            sys::mlx_array_new_data(
                data.as_ptr() as *const _,
                shape.as_ptr(),
                shape.len() as i32,
                Dtype::I32.to_sys(),
            )
        };
        Self::check_handle(raw)
    }

    fn check_handle(raw: sys::mlx_array) -> Result<Self> {
        if raw.ctx.is_null() {
            Err(MlxError::Engine("mlx returned a null array".into()))
        } else {
            Ok(Array::from_raw(raw))
        }
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        unsafe { sys::mlx_array_ndim(self.raw) }
    }

    /// Shape as a `Vec<i32>`.
    pub fn shape(&self) -> Vec<i32> {
        let n = self.ndim();
        if n == 0 {
            return vec![];
        }
        let ptr = unsafe { sys::mlx_array_shape(self.raw) };
        (0..n).map(|i| unsafe { *ptr.add(i) }).collect()
    }

    /// Total element count.
    pub fn size(&self) -> usize {
        unsafe { sys::mlx_array_size(self.raw) }
    }

    /// Element dtype.
    pub fn dtype(&self) -> Dtype {
        Dtype::from_sys(unsafe { sys::mlx_array_dtype(self.raw) })
    }

    /// Force evaluation of the lazy graph backing this array.
    pub fn eval(&self) -> Result<()> {
        let rc = unsafe { sys::mlx_array_eval(self.raw) };
        check(rc, "eval")
    }

    /// Copy this array's contents out as `f32` (evaluates first). The array is
    /// cast to f32 on the given stream if needed.
    pub fn to_vec_f32(&self, s: &Stream) -> Result<Vec<f32>> {
        let casted = self.astype(Dtype::F32, s)?;
        casted.eval()?;
        let n = casted.size();
        let ptr = unsafe { sys::mlx_array_data_float32(casted.raw) };
        if ptr.is_null() {
            return Err(MlxError::Engine("null f32 data pointer".into()));
        }
        Ok((0..n).map(|i| unsafe { *ptr.add(i) }).collect())
    }

    /// Copy this array's contents out as `i32` (evaluates first).
    pub fn to_vec_i32(&self) -> Result<Vec<i32>> {
        self.eval()?;
        let n = self.size();
        let ptr = unsafe { sys::mlx_array_data_int32(self.raw) };
        if ptr.is_null() {
            return Err(MlxError::Engine("null i32 data pointer".into()));
        }
        Ok((0..n).map(|i| unsafe { *ptr.add(i) }).collect())
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        if !self.raw.ctx.is_null() {
            unsafe {
                sys::mlx_array_free(self.raw);
            }
        }
    }
}

/// Check a C return code, mapping non-zero to an engine error.
pub(crate) fn check(rc: i32, op: &str) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError::Engine(format!("mlx op '{op}' failed (rc={rc})")))
    }
}

/// Run a unary C op of the form `int f(mlx_array* res, mlx_array a, mlx_stream)`.
pub(crate) fn unary(
    f: unsafe extern "C" fn(*mut sys::mlx_array, sys::mlx_array, sys::mlx_stream) -> i32,
    a: &Array,
    s: &Stream,
    op: &str,
) -> Result<Array> {
    let mut res = unsafe { sys::mlx_array_new() };
    let rc = unsafe { f(&mut res, a.raw, s.raw) };
    check(rc, op)?;
    Ok(Array::from_raw(res))
}

/// Run a binary C op of the form `int f(res, a, b, stream)`.
pub(crate) fn binary(
    f: unsafe extern "C" fn(
        *mut sys::mlx_array,
        sys::mlx_array,
        sys::mlx_array,
        sys::mlx_stream,
    ) -> i32,
    a: &Array,
    b: &Array,
    s: &Stream,
    op: &str,
) -> Result<Array> {
    let mut res = unsafe { sys::mlx_array_new() };
    let rc = unsafe { f(&mut res, a.raw, b.raw, s.raw) };
    check(rc, op)?;
    Ok(Array::from_raw(res))
}

impl Array {
    pub(crate) fn astype(&self, dtype: Dtype, s: &Stream) -> Result<Array> {
        let mut res = unsafe { sys::mlx_array_new() };
        let rc = unsafe { sys::mlx_astype(&mut res, self.raw, dtype.to_sys(), s.raw) };
        check(rc, "astype")?;
        Ok(Array::from_raw(res))
    }
}

/// Convert a Rust string to a `CString` for the C API.
pub(crate) fn cstr(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| MlxError::Engine("string contained NUL".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_roundtrip() {
        for d in [Dtype::F32, Dtype::F16, Dtype::BF16, Dtype::I32, Dtype::U32] {
            assert_eq!(Dtype::from_sys(d.to_sys()), d);
        }
    }

    #[test]
    fn shape_element_mismatch_is_rejected() {
        // Pure validation path — does not call the engine.
        let err = Array::from_f32(&[1.0, 2.0], &[3]);
        assert!(err.is_err());
    }
}
