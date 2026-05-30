use crate::{
    AcceleratorKind, AcceleratorProfile, BackendKind, CpuProfile, HardwareProfile,
    MeasurementSource, MemoryProfile,
};
use anyhow::{Result, bail};
pub use mesh_llm_gpu_bench::BenchmarkOutput as GpuBenchmarkOutput;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GpuBenchmarkHardwareInput {
    pub memory: MemoryProfile,
    pub cpu: CpuProfile,
    pub default_backend: BackendKind,
    pub accelerators: Vec<GpuBenchmarkAcceleratorFacts>,
    pub benchmark_outputs: Vec<GpuBenchmarkOutput>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GpuBenchmarkAcceleratorFacts {
    pub name: Option<String>,
    pub kind: AcceleratorKind,
    pub backend: Option<BackendKind>,
    pub total_memory_bytes: Option<u64>,
    pub available_memory_bytes: Option<u64>,
    pub unified_memory: bool,
}

pub fn hardware_profile_from_gpu_benchmark(
    input: GpuBenchmarkHardwareInput,
) -> Result<HardwareProfile> {
    if input.benchmark_outputs.is_empty() {
        bail!("GPU benchmark output must contain at least one device");
    }
    if input
        .benchmark_outputs
        .iter()
        .any(|output| output.p90_gbps <= 0.0)
    {
        bail!("GPU benchmark output must include positive p90_gbps for every device");
    }

    let accelerators = input
        .benchmark_outputs
        .into_iter()
        .enumerate()
        .map(|(index, output)| {
            let facts = input.accelerators.get(index).cloned().unwrap_or_default();
            accelerator_from_benchmark(output, facts, input.default_backend)
        })
        .collect();

    Ok(HardwareProfile {
        memory: input.memory,
        accelerators,
        cpu: input.cpu,
    })
}

fn accelerator_from_benchmark(
    output: GpuBenchmarkOutput,
    facts: GpuBenchmarkAcceleratorFacts,
    default_backend: BackendKind,
) -> AcceleratorProfile {
    AcceleratorProfile {
        name: facts.name.or(Some(output.device)),
        kind: accelerator_kind(facts.kind, facts.unified_memory),
        backend: facts.backend.unwrap_or(default_backend),
        total_memory_bytes: facts.total_memory_bytes,
        available_memory_bytes: facts.available_memory_bytes.or(facts.total_memory_bytes),
        memory_bandwidth_bytes_per_sec: Some((output.p90_gbps * 1_000_000_000.0).round() as u64),
        bandwidth_source: MeasurementSource::Measured,
        unified_memory: facts.unified_memory,
    }
}

fn accelerator_kind(kind: AcceleratorKind, unified_memory: bool) -> AcceleratorKind {
    if kind != AcceleratorKind::Unknown {
        kind
    } else if unified_memory {
        AcceleratorKind::IntegratedGpu
    } else {
        AcceleratorKind::DiscreteGpu
    }
}
