//! Distributed group + collectives — safe wrappers over `mlx-c`'s distributed
//! C API. This is the exact machinery Python's `mx.distributed` binds to; we
//! call the same C functions from Rust. The transport (ring/TCP or
//! jaccl/Thunderbolt-RDMA) is selected by MLX at `init` time from the hostfile
//! and `MLX_DIST_BACKEND` env, not here.

use crate::array::check;
use crate::array::{Array, Stream};
use crate::{MlxError, Result};
use mesh_mlx_sys as sys;
use std::ffi::CString;

/// Which MLX distributed backend to initialise.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Backend {
    /// Pick whatever is available.
    Any,
    /// TCP ring (Ethernet/Wi-Fi/Thunderbolt).
    Ring,
    /// RDMA over Thunderbolt 5.
    Jaccl,
    /// MPI.
    Mpi,
}

impl Backend {
    fn as_cstr(self) -> Option<CString> {
        let name = match self {
            Backend::Any => return None,
            Backend::Ring => "ring",
            Backend::Jaccl => "jaccl",
            Backend::Mpi => "mpi",
        };
        CString::new(name).ok()
    }

    /// Whether MLX reports this backend as available without initialising it.
    pub fn is_available(self) -> bool {
        match self.as_cstr() {
            None => unsafe { sys::mlx_distributed_is_available(std::ptr::null()) },
            Some(c) => unsafe { sys::mlx_distributed_is_available(c.as_ptr()) },
        }
    }
}

/// A distributed process group. Owns the underlying `mlx_distributed_group`.
pub struct Group {
    raw: sys::mlx_distributed_group,
    rank: i32,
    size: i32,
}

impl Group {
    /// Initialise the distributed runtime for `backend`. With `strict`, errors
    /// if no real backend is available (rather than silently single-process).
    pub fn init(backend: Backend, strict: bool) -> Result<Self> {
        let mut raw = unsafe { sys::mlx_distributed_group_new() };
        let rc = match backend.as_cstr() {
            None => unsafe { sys::mlx_distributed_init(&mut raw, strict, std::ptr::null()) },
            Some(c) => unsafe { sys::mlx_distributed_init(&mut raw, strict, c.as_ptr()) },
        };
        check(rc, "distributed_init")?;
        Self::wrap(raw)
    }

    fn wrap(raw: sys::mlx_distributed_group) -> Result<Self> {
        if raw.ctx.is_null() {
            return Err(MlxError::Distributed("null group handle".into()));
        }
        let rank = unsafe { sys::mlx_distributed_group_rank(raw) };
        let size = unsafe { sys::mlx_distributed_group_size(raw) };
        Ok(Group { raw, rank, size })
    }

    /// 0-based rank of this process within the group.
    pub fn rank(&self) -> i32 {
        self.rank
    }

    /// Number of processes in the group.
    pub fn size(&self) -> i32 {
        self.size
    }

    /// Split the group into subgroups by `color`, ordered by `key`.
    pub fn split(&self, color: i32, key: i32) -> Result<Group> {
        let mut raw = unsafe { sys::mlx_distributed_group_new() };
        let rc = unsafe { sys::mlx_distributed_group_split(&mut raw, self.raw, color, key) };
        check(rc, "group_split")?;
        Self::wrap(raw)
    }

    /// All-reduce (sum) `x` across the group. Every rank receives the total.
    pub fn all_sum(&self, x: &Array, s: &Stream) -> Result<Array> {
        let mut res = unsafe { sys::mlx_array_new() };
        let rc = unsafe { sys::mlx_distributed_all_sum(&mut res, x.raw, self.raw, s.raw) };
        check(rc, "all_sum")?;
        Ok(Array::from_raw(res))
    }

    /// All-gather `x` across the group along axis 0.
    pub fn all_gather(&self, x: &Array, s: &Stream) -> Result<Array> {
        let mut res = unsafe { sys::mlx_array_new() };
        let rc = unsafe { sys::mlx_distributed_all_gather(&mut res, x.raw, self.raw, s.raw) };
        check(rc, "all_gather")?;
        Ok(Array::from_raw(res))
    }

    /// Send `x` to `dst`. Returns the (dependency) array MLX produces.
    pub fn send(&self, x: &Array, dst: i32, s: &Stream) -> Result<Array> {
        let mut res = unsafe { sys::mlx_array_new() };
        let rc = unsafe { sys::mlx_distributed_send(&mut res, x.raw, dst, self.raw, s.raw) };
        check(rc, "send")?;
        Ok(Array::from_raw(res))
    }

    /// Receive an array shaped like `template` from `src`.
    pub fn recv_like(&self, template: &Array, src: i32, s: &Stream) -> Result<Array> {
        let mut res = unsafe { sys::mlx_array_new() };
        let rc =
            unsafe { sys::mlx_distributed_recv_like(&mut res, template.raw, src, self.raw, s.raw) };
        check(rc, "recv_like")?;
        Ok(Array::from_raw(res))
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        if !self.raw.ctx.is_null() {
            unsafe {
                sys::mlx_distributed_group_free(self.raw);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_names_are_valid_cstrings() {
        assert!(Backend::Any.as_cstr().is_none());
        assert_eq!(Backend::Ring.as_cstr().unwrap().to_str().unwrap(), "ring");
        assert_eq!(Backend::Jaccl.as_cstr().unwrap().to_str().unwrap(), "jaccl");
        assert_eq!(Backend::Mpi.as_cstr().unwrap().to_str().unwrap(), "mpi");
    }
}
