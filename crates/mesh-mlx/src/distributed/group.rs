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

// SAFETY: an `mlx_distributed_group` is a reference-counted handle; engine
// access is serialised by the runtime. See the note on `array::Array`.
unsafe impl Send for Group {}
unsafe impl Sync for Group {}

impl Group {
    /// Initialise the distributed runtime for `backend`. With `strict`, errors
    /// if no real backend is available (rather than silently single-process).
    pub fn init(backend: Backend, strict: bool) -> Result<Self> {
        let raw = match backend.as_cstr() {
            None => unsafe { sys::mlx_distributed_init(strict, std::ptr::null()) },
            Some(c) => unsafe { sys::mlx_distributed_init(strict, c.as_ptr()) },
        };
        Self::wrap(raw, "distributed_init")
    }

    /// Take ownership of a freshly allocated group handle. mlx-c v0.6.0 returns
    /// a null handle on failure and does not export a public group-free symbol,
    /// so `Drop` is intentionally a no-op until mlx-c grows that API.
    fn wrap(raw: sys::mlx_distributed_group, op: &str) -> Result<Self> {
        if raw.ctx.is_null() {
            return Err(MlxError::Distributed(format!("{op}: null group handle")));
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
        let raw = unsafe { sys::mlx_distributed_group_split(self.raw, color, key) };
        Self::wrap(raw, "group_split")
    }

    /// All-reduce (sum) `x` across the group. Every rank receives the total.
    pub fn all_sum(&self, x: &Array, s: &Stream) -> Result<Array> {
        let mut res = unsafe { sys::mlx_array_new() };
        let rc = unsafe { sys::mlx_distributed_all_sum(&mut res, x.raw, self.raw, s.raw) };
        // Take ownership before the check so the handle is freed on error.
        let res = Array::from_raw(res);
        check(rc, "all_sum")?;
        Ok(res)
    }

    /// All-gather `x` across the group along axis 0.
    pub fn all_gather(&self, x: &Array, s: &Stream) -> Result<Array> {
        let mut res = unsafe { sys::mlx_array_new() };
        let rc = unsafe { sys::mlx_distributed_all_gather(&mut res, x.raw, self.raw, s.raw) };
        let res = Array::from_raw(res);
        check(rc, "all_gather")?;
        Ok(res)
    }

    /// Send `x` to `dst`. Returns the (dependency) array MLX produces.
    pub fn send(&self, x: &Array, dst: i32, s: &Stream) -> Result<Array> {
        let mut res = unsafe { sys::mlx_array_new() };
        let rc = unsafe { sys::mlx_distributed_send(&mut res, x.raw, dst, self.raw, s.raw) };
        let res = Array::from_raw(res);
        check(rc, "send")?;
        Ok(res)
    }

    /// Receive an array shaped like `template` from `src`.
    pub fn recv_like(&self, template: &Array, src: i32, s: &Stream) -> Result<Array> {
        let mut res = unsafe { sys::mlx_array_new() };
        let rc =
            unsafe { sys::mlx_distributed_recv_like(&mut res, template.raw, src, self.raw, s.raw) };
        let res = Array::from_raw(res);
        check(rc, "recv_like")?;
        Ok(res)
    }

    /// Broadcast a byte payload from `root` to every rank, returning the bytes
    /// on all ranks.
    ///
    /// This is the same length-prefixed additive-reduce pattern MLX's own
    /// distributed server uses to deliver a request to worker ranks: only the
    /// root contributes the real values and the others contribute zeros, so the
    /// `all_sum` reduces to the root's payload on every rank. Two reductions are
    /// used — first the length (so workers can size their buffer), then the
    /// bytes. A zero length is a valid empty/sentinel broadcast.
    ///
    /// All ranks must call this in lock-step with the same `root`.
    pub fn broadcast_bytes(&self, root: i32, payload: &[u8], s: &Stream) -> Result<Vec<u8>> {
        let is_root = self.rank == root;

        // Step 1: agree on the length.
        let len_val: i32 = if is_root {
            i32::try_from(payload.len())
                .map_err(|_| MlxError::Distributed("broadcast payload too large".into()))?
        } else {
            0
        };
        let len_arr = Array::from_i32(&[len_val], &[1])?;
        let len_sum = self.all_sum(&len_arr, s)?;
        let len = *len_sum.to_vec_i32()?.first().unwrap_or(&0);
        if len <= 0 {
            return Ok(Vec::new());
        }
        let len = len as usize;

        // Step 2: agree on the bytes. Encode as i32 (the FFI's integer array
        // dtype) so the additive reduce is exact; non-root ranks send zeros.
        let mut buf = vec![0i32; len];
        if is_root {
            for (dst, b) in buf.iter_mut().zip(payload.iter()) {
                *dst = *b as i32;
            }
        }
        let arr = Array::from_i32(&buf, &[len as i32])?;
        let summed = self.all_sum(&arr, s)?;
        let vals = summed.to_vec_i32()?;
        Ok(vals.iter().map(|v| (*v & 0xff) as u8).collect())
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        // mlx-c v0.6.0 exposes a private inline `mlx_distributed_group_free_`
        // helper but no public C symbol for freeing groups. Avoid linking a
        // non-existent symbol; leaking this process-lifetime distributed handle
        // is preferable to a broken native build. Revisit when mlx-c exports a
        // public free function.
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
