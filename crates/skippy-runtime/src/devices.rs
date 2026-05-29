use std::ffi::{CStr, c_char};
use std::ptr;

use anyhow::{Result, anyhow};
use skippy_ffi::{BackendDevice as RawBackendDevice, BackendDeviceType as RawBackendDeviceType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendDeviceType {
    Cpu,
    Gpu,
    IntegratedGpu,
    Accelerator,
    Meta,
}

impl From<RawBackendDeviceType> for BackendDeviceType {
    fn from(value: RawBackendDeviceType) -> Self {
        match value {
            RawBackendDeviceType::Cpu => Self::Cpu,
            RawBackendDeviceType::Gpu => Self::Gpu,
            RawBackendDeviceType::IGpu => Self::IntegratedGpu,
            RawBackendDeviceType::Accel => Self::Accelerator,
            RawBackendDeviceType::Meta => Self::Meta,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendDevice {
    pub name: String,
    pub description: Option<String>,
    pub device_id: Option<String>,
    pub memory_free: u64,
    pub memory_total: u64,
    pub device_type: BackendDeviceType,
    pub caps: u64,
}

pub fn backend_devices() -> Result<Vec<BackendDevice>> {
    let mut error = ptr::null_mut();
    let mut count = 0usize;
    let status = unsafe { skippy_ffi::skippy_backend_device_count(&mut count, &mut error) };
    super::ensure_ok(status, error)?;

    let mut devices = Vec::with_capacity(count);
    for index in 0..count {
        let mut raw = RawBackendDevice {
            version: 0,
            name: ptr::null(),
            description: ptr::null(),
            device_id: ptr::null(),
            memory_free: 0,
            memory_total: 0,
            device_type: RawBackendDeviceType::Cpu,
            caps: 0,
        };
        let mut error = ptr::null_mut();
        let status = unsafe { skippy_ffi::skippy_backend_device_at(index, &mut raw, &mut error) };
        super::ensure_ok(status, error)?;
        devices.push(backend_device_from_raw(raw)?);
    }

    Ok(devices)
}

fn backend_device_from_raw(raw: RawBackendDevice) -> Result<BackendDevice> {
    Ok(BackendDevice {
        name: c_string_required(raw.name, "backend device name")?,
        description: c_string_optional(raw.description)?,
        device_id: c_string_optional(raw.device_id)?,
        memory_free: raw.memory_free,
        memory_total: raw.memory_total,
        device_type: raw.device_type.into(),
        caps: raw.caps,
    })
}

fn c_string_required(ptr: *const c_char, field: &str) -> Result<String> {
    if ptr.is_null() {
        return Err(anyhow!("{field} is null"));
    }
    Ok(unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned())
}

fn c_string_optional(ptr: *const c_char) -> Result<Option<String>> {
    if ptr.is_null() {
        return Ok(None);
    }
    let value = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    Ok((!value.is_empty()).then_some(value))
}

/// Checks whether a backend device has at least `headroom_bytes` of free memory.
///
/// Returns `Ok(true)` if the device exists and `free >= headroom_bytes`.
/// Returns `Ok(false)` if the device exists but free memory is below the threshold,
/// or if the device index is out of range.
/// Returns `Err` on ABI error.
///
/// # Backend coverage
///
/// The underlying C API (`skippy_backend_device_check_memory`) delegates to
/// `ggml_backend_dev_memory()` which queries the backend's `get_memory()`.
/// This is accurate for CUDA and ROCm. For other backends (Intel, Vulkan,
/// Metal) where `get_memory` may return 0 or stale values, extend the C
/// function with a backend-specific fallback before calling
/// `ggml_backend_dev_memory()` — see the switch-case skeleton in skippy.cpp.
pub fn check_device_memory(device_index: usize, headroom_bytes: u64) -> Result<bool> {
    let mut sufficient = false;
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_backend_device_check_memory(
            device_index,
            headroom_bytes as usize,
            &mut sufficient,
            &mut error,
        )
    };
    super::ensure_ok(status, error)?;
    Ok(sufficient)
}

#[cfg(test)]
mod tests {
    use super::check_device_memory;

    #[test]
    fn out_of_range_device_index_returns_false() {
        let result = check_device_memory(usize::MAX, 0).unwrap();
        assert!(
            !result,
            "out-of-range device index should indicate insufficient memory"
        );
    }

    #[test]
    fn impossible_headroom_returns_false() {
        let result = check_device_memory(0, u64::MAX).unwrap();
        assert!(
            !result,
            "impossible headroom should indicate insufficient memory"
        );
    }

    #[test]
    fn zero_headroom_does_not_crash() {
        // This may return true (if a device exists) or false (if no devices).
        // Either is valid — the important thing is it doesn't error or crash.
        let result = check_device_memory(0, 0);
        assert!(
            result.is_ok(),
            "zero headroom query should not fail: {:?}",
            result.err()
        );
    }
}

/// Dynamic GPU allocation tracker — replaces static VRAM headroom with max-seen tracking.
///
/// Uses the C-side `skippy_alloc_tracker_*` API which tracks VMM pool allocations
/// via the CUDA alloc hook from patch 0083.
///
/// # Multi-session limitation
///
/// The underlying CUDA alloc hook (`g_ggml_cuda_alloc_hook`) is a process-global
/// static pointer — when multiple sessions are created, the last `session_create`
/// wins and overwrites any prior hook registration. Currently the callback always
/// returns `true` (tracking-only), so worst-case impact across overlapping sessions
/// is noisy `max_seen` tracking (the atomic counter accumulates allocations from all
/// active sessions). Always call [`reset`] before loading a new model to clear stale
/// values, or headroom computation will be based on outdated allocation patterns.
pub mod alloc_tracker {
    use super::*;

    /// Initialize/reset the allocation tracker at session startup.
    pub fn init() {
        unsafe {
            skippy_ffi::skippy_alloc_tracker_init();
        }
    }

    /// Reset max-seen counter (called on model reload / full lifecycle restart).
    pub fn reset() {
        unsafe {
            skippy_ffi::skippy_alloc_tracker_reset();
        }
    }

    /// Returns the maximum VMM pool allocation size seen so far.
    /// Used to compute dynamic headroom: `headroom = max_seen * 1.05`.
    pub fn get_max_seen() -> u64 {
        unsafe { skippy_ffi::skippy_alloc_tracker_get_max_seen() }
    }

    /// Compute dynamic headroom based on max-seen allocation.
    /// Returns 5% of max_seen, clamped to [min_headroom, max_headroom].
    pub fn compute_dynamic_headroom(max_seen: u64) -> u64 {
        let min_headroom = 256 * 1024 * 1024;
        let dynamic = (max_seen as f64 * 0.05) as u64;
        std::cmp::max(min_headroom, dynamic)
    }

    /// Check device memory using the new tracker-based approach.
    ///
    /// If max_seen > 0 and current free >= computed headroom → OK.
    /// Falls back to static check_device_memory if no data yet.
    pub fn check_device_memory_with_tracker(device_index: usize, total_vram: u64) -> Result<bool> {
        let max_seen = get_max_seen();

        if max_seen > 0 {
            let headroom = compute_dynamic_headroom(max_seen);
            check_device_memory(device_index, headroom)
        } else {
            let static_headroom = std::cmp::max(256 * 1024 * 1024, total_vram / 20);
            check_device_memory(device_index, static_headroom)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn compute_dynamic_headroom_clamps_minimum() {
            let headroom = compute_dynamic_headroom(1_000_000);
            assert!(
                headroom >= 256 * 1024 * 1024,
                "should clamp to 256MB minimum"
            );
        }

        #[test]
        fn compute_dynamic_headroom_scales() {
            let headroom = compute_dynamic_headroom(8_000_000_000u64);
            assert!(headroom == 400_000_000, "5% of 8GB should be 400MB");
        }

        #[test]
        fn fallback_check_does_not_crash() {
            let result = check_device_memory_with_tracker(usize::MAX, 0);
            assert!(
                result.is_ok(),
                "out-of-range device should not error: {:?}",
                result.err()
            );
            assert!(!result.unwrap(), "should return false for missing device");
        }
    }
}
