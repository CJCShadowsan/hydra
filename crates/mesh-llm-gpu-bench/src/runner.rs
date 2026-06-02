use crate::BenchmarkOutput;
use anyhow::Result;
#[cfg(any(
    not(target_os = "macos"),
    not(feature = "cuda"),
    not(feature = "hip"),
    not(feature = "intel")
))]
use anyhow::anyhow;
use std::{hint::black_box, time::Duration};

const SAMPLER_PROBE_PROMPT_TOKENS: usize = 4096;
const SAMPLER_PROBE_VOCAB_TOKENS: usize = 131_072;
const SAMPLER_PROBE_RUNS: usize = 9;

#[derive(Clone, Copy)]
struct SamplerProbe {
    history_us_per_token: f64,
    vocab_us_per_token: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkBackend {
    Metal,
    Cuda,
    Hip,
    Intel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkRunner {
    pub backend: BenchmarkBackend,
}

pub fn runner_for(
    os: &str,
    gpu_count: u8,
    gpu_name: Option<&str>,
    is_soc: bool,
) -> Option<BenchmarkRunner> {
    if gpu_count == 0 {
        tracing::debug!("no GPUs detected; skipping benchmark");
        return None;
    }

    let gpu_upper = gpu_name.unwrap_or("").to_uppercase();

    if os == "macos" && is_soc {
        return Some(BenchmarkRunner {
            backend: BenchmarkBackend::Metal,
        });
    }

    if os == "linux" || os == "windows" {
        if gpu_upper.contains("NVIDIA")
            || gpu_upper.contains("ORIN")
            || gpu_upper.contains("NVGPU")
            || gpu_upper.contains("TEGRA")
        {
            return Some(BenchmarkRunner {
                backend: BenchmarkBackend::Cuda,
            });
        }

        if gpu_upper.contains("AMD") || gpu_upper.contains("RADEON") {
            return Some(BenchmarkRunner {
                backend: BenchmarkBackend::Hip,
            });
        }

        if gpu_upper.contains("INTEL") || gpu_upper.contains("ARC") {
            tracing::info!(
                "Intel GPU benchmark is not supported in standard mesh-llm builds; skipping"
            );
            return None;
        }

        if os == "linux" && is_soc {
            tracing::warn!("Jetson benchmark is unvalidated for ARM CUDA; attempting");
            return Some(BenchmarkRunner {
                backend: BenchmarkBackend::Cuda,
            });
        }
    }

    tracing::warn!("could not identify benchmark runner for GPU platform: {gpu_name:?}");
    None
}

pub fn parse_benchmark_output(stdout: &[u8]) -> Option<Vec<BenchmarkOutput>> {
    match serde_json::from_slice::<Vec<BenchmarkOutput>>(stdout) {
        Ok(outputs) if !outputs.is_empty() => Some(outputs),
        Ok(_) => {
            tracing::debug!("benchmark returned empty device list");
            None
        }
        Err(err) => {
            let error_message = serde_json::from_slice::<serde_json::Value>(stdout)
                .ok()
                .and_then(|val| {
                    val.get("error")
                        .and_then(|v| v.as_str())
                        .map(ToOwned::to_owned)
                });
            if let Some(msg) = error_message {
                tracing::warn!("benchmark reported error: {msg}");
                return None;
            }
            tracing::warn!("failed to parse benchmark output: {err}");
            None
        }
    }
}

pub fn run_benchmark(runner: BenchmarkRunner, _timeout: Duration) -> Result<Vec<BenchmarkOutput>> {
    let mut outputs = match runner.backend {
        BenchmarkBackend::Metal => run_metal_benchmark(),
        BenchmarkBackend::Cuda => run_cuda_benchmark(),
        BenchmarkBackend::Hip => run_hip_benchmark(),
        BenchmarkBackend::Intel => run_intel_benchmark(),
    }?;
    attach_ggml_decode_probes(runner.backend, &mut outputs);
    attach_sampler_probe(&mut outputs);
    Ok(outputs)
}

fn attach_ggml_decode_probes(backend: BenchmarkBackend, outputs: &mut [BenchmarkOutput]) {
    let probes = run_ggml_decode_probes(backend);
    if probes.is_empty() {
        return;
    }
    for output in outputs {
        output.decode_kernel_probes.extend(probes.clone());
    }
}

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
fn run_ggml_decode_probes(backend: BenchmarkBackend) -> Vec<crate::DecodeKernelProbe> {
    match crate::ggml_probe::run(backend) {
        Ok(probes) => probes,
        Err(error) => {
            tracing::warn!("GGML decode kernel probe failed: {error:#}");
            Vec::new()
        }
    }
}

#[cfg(not(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe)))]
fn run_ggml_decode_probes(_backend: BenchmarkBackend) -> Vec<crate::DecodeKernelProbe> {
    Vec::new()
}

fn attach_sampler_probe(outputs: &mut [BenchmarkOutput]) {
    let probe = measure_sampler_probe();
    for output in outputs {
        output.sampler_history_us_per_token = Some(probe.history_us_per_token);
        output.sampler_vocab_us_per_token = Some(probe.vocab_us_per_token);
    }
}

fn measure_sampler_probe() -> SamplerProbe {
    let mut history_samples = Vec::with_capacity(SAMPLER_PROBE_RUNS);
    let mut vocab_samples = Vec::with_capacity(SAMPLER_PROBE_RUNS);
    for _ in 0..SAMPLER_PROBE_RUNS {
        history_samples.push(measure_sampler_history_us_per_token());
        vocab_samples.push(measure_sampler_vocab_us_per_token());
    }
    history_samples.sort_by(|left, right| left.total_cmp(right));
    vocab_samples.sort_by(|left, right| left.total_cmp(right));
    SamplerProbe {
        history_us_per_token: history_samples[SAMPLER_PROBE_RUNS / 2],
        vocab_us_per_token: vocab_samples[SAMPLER_PROBE_RUNS / 2],
    }
}

fn measure_sampler_history_us_per_token() -> f64 {
    let tokens = (0..SAMPLER_PROBE_PROMPT_TOKENS)
        .map(|index| ((index * 1_103 + 17) % SAMPLER_PROBE_VOCAB_TOKENS) as u32)
        .collect::<Vec<_>>();
    let mut recent_counts = vec![0u16; 65_536];
    let started = std::time::Instant::now();
    let mut state = 0u64;
    for token in &tokens {
        let slot = (*token as usize) & (recent_counts.len() - 1);
        recent_counts[slot] = recent_counts[slot].wrapping_add(1);
        state = state
            .wrapping_mul(1_099_511_628_211)
            .wrapping_add(u64::from(*token))
            .wrapping_add(u64::from(recent_counts[slot]));
        black_box(state);
    }
    started.elapsed().as_secs_f64() * 1_000_000.0 / tokens.len() as f64
}

fn measure_sampler_vocab_us_per_token() -> f64 {
    #[derive(Clone, Copy)]
    struct Candidate {
        id: u32,
        logit: f32,
        p: f32,
    }

    let started = std::time::Instant::now();
    let mut candidates = Vec::with_capacity(SAMPLER_PROBE_VOCAB_TOKENS);
    let mut max_logit = f32::NEG_INFINITY;
    let mut max_id = 0u32;
    for id in 0..SAMPLER_PROBE_VOCAB_TOKENS as u32 {
        let logit =
            ((id.wrapping_mul(1_664_525).wrapping_add(1_013_904_223) & 0xffff) as f32) / 65_536.0;
        if logit > max_logit {
            max_logit = logit;
            max_id = id;
        }
        candidates.push(Candidate { id, logit, p: 0.0 });
    }
    let selected = candidates
        .get(max_id as usize % candidates.len())
        .copied()
        .map(|candidate| (candidate.id, candidate.logit, candidate.p));
    black_box((selected, candidates.len()));
    started.elapsed().as_secs_f64() * 1_000_000.0 / SAMPLER_PROBE_VOCAB_TOKENS as f64
}

#[cfg(target_os = "macos")]
fn run_metal_benchmark() -> Result<Vec<BenchmarkOutput>> {
    crate::metal::run()
}

#[cfg(not(target_os = "macos"))]
fn run_metal_benchmark() -> Result<Vec<BenchmarkOutput>> {
    Err(anyhow!(
        "Metal benchmark backend was not compiled into this mesh-llm binary"
    ))
}

#[cfg(feature = "cuda")]
fn run_cuda_benchmark() -> Result<Vec<BenchmarkOutput>> {
    crate::cuda::run()
}

#[cfg(not(feature = "cuda"))]
fn run_cuda_benchmark() -> Result<Vec<BenchmarkOutput>> {
    Err(anyhow!(
        "CUDA benchmark backend was not compiled into this mesh-llm binary"
    ))
}

#[cfg(feature = "hip")]
fn run_hip_benchmark() -> Result<Vec<BenchmarkOutput>> {
    crate::hip::run()
}

#[cfg(not(feature = "hip"))]
fn run_hip_benchmark() -> Result<Vec<BenchmarkOutput>> {
    Err(anyhow!(
        "HIP benchmark backend was not compiled into this mesh-llm binary"
    ))
}

#[cfg(feature = "intel")]
fn run_intel_benchmark() -> Result<Vec<BenchmarkOutput>> {
    crate::intel::run()
}

#[cfg(not(feature = "intel"))]
fn run_intel_benchmark() -> Result<Vec<BenchmarkOutput>> {
    Err(anyhow!(
        "Intel benchmark backend was not compiled into this mesh-llm binary"
    ))
}
