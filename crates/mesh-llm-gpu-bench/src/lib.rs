mod output;
mod runner;

#[cfg(any(feature = "cuda", feature = "hip", feature = "intel"))]
mod capture;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "hip")]
mod hip;

#[cfg(feature = "intel")]
mod intel;

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
mod ggml_probe;

#[cfg(target_os = "macos")]
mod metal;

pub use output::{BenchmarkOutput, DecodeKernelProbe};
pub use runner::{
    BenchmarkBackend, BenchmarkRunner, parse_benchmark_output, run_benchmark, runner_for,
};
