use crate::system::hardware::GpuFacts;

pub(super) fn local_gpu_capacity_fallback(gpu: &GpuFacts) -> Option<u64> {
    if gpu.vram_bytes > 0 || gpu.backend_device.is_none() {
        return None;
    }
    macos_planning_memory_bytes()
}

#[cfg(target_os = "macos")]
fn macos_planning_memory_bytes() -> Option<u64> {
    macos_metal_recommended_working_set_bytes().or_else(|| {
        macos_total_memory_bytes()
            .filter(|bytes| *bytes > 0)
            .map(|bytes| (bytes as f64 * 0.80) as u64)
    })
}

#[cfg(target_os = "macos")]
fn macos_metal_recommended_working_set_bytes() -> Option<u64> {
    use std::ffi::{c_char, c_void};

    #[link(name = "Metal", kind = "framework")]
    unsafe extern "C" {
        fn MTLCreateSystemDefaultDevice() -> *mut c_void;
    }

    #[link(name = "objc")]
    unsafe extern "C" {
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        fn objc_msgSend(receiver: *mut c_void, selector: *mut c_void, ...) -> usize;
    }

    unsafe {
        let device = MTLCreateSystemDefaultDevice();
        if device.is_null() {
            return None;
        }
        let selector = c"recommendedMaxWorkingSetSize";
        let selector = sel_registerName(selector.as_ptr());
        if selector.is_null() {
            return None;
        }
        let bytes = objc_msgSend(device, selector) as u64;
        (bytes > 0).then_some(bytes)
    }
}

#[cfg(target_os = "macos")]
fn macos_total_memory_bytes() -> Option<u64> {
    let mut bytes = 0_u64;
    let mut len = std::mem::size_of::<u64>();
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            std::ptr::addr_of_mut!(bytes).cast(),
            std::ptr::addr_of_mut!(len),
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && bytes > 0).then_some(bytes)
}

#[cfg(not(target_os = "macos"))]
fn macos_planning_memory_bytes() -> Option<u64> {
    None
}
