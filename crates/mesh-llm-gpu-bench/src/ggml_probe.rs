use crate::{BenchmarkBackend, DecodeKernelProbe};
use anyhow::{Context, Result, anyhow};
use libc::{c_char, c_int, c_void};
use std::ffi::CStr;

unsafe extern "C" {
    fn mesh_llm_gpu_bench_ggml_decode_probe_json(
        backend_kind: c_int,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_decode_probe_free(ptr: *mut c_void);
}

pub fn run(backend: BenchmarkBackend) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };

    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe { mesh_llm_gpu_bench_ggml_decode_probe_json(backend_kind, &mut error) };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML decode probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).context("GGML decode probe returned invalid output")
}
