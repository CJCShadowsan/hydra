use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use skippy_runtime::package::{PackagePart, PackageStageRequest, select_layer_package_parts};
use skippy_runtime::{
    ActivationDesc, ActivationFrame, FlashAttentionType, RuntimeActivationDType,
    RuntimeActivationLayout, RuntimeConfig, RuntimeLoadMode, StageModel, parse_cache_type,
};

use crate::cli::GlmDsaLayerMicrobenchArgs;

pub fn glm_dsa_layer_microbench(args: GlmDsaLayerMicrobenchArgs) -> Result<()> {
    validate_args(&args)?;
    configure_env_flags(&args);

    let selected = select_layer_package_parts(&package_request(&args))
        .context("select GLM-DSA layer package parts")?;
    let runtime_config = runtime_config(&args)?;
    let model = StageModel::open_from_parts(&selected.absolute_paths, &runtime_config)
        .context("open GLM-DSA layer microbench model")?;
    let input = synthetic_activation_frame(&args)?;
    let token_ids = vec![1_i32; args.tokens];
    let positions = positions(args.tokens)?;

    let mut timings = Vec::with_capacity(args.iterations);
    let total_runs = args.warmup + args.iterations;
    for run_index in 0..total_runs {
        let mut session = model.create_session().context("create stage session")?;
        let started = Instant::now();
        let output = session
            .prefill_chunk_frame_with_positions(&token_ids, &positions, Some(&input), 0)
            .with_context(|| format!("run microbench iteration {run_index}"))?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        if run_index >= args.warmup {
            timings.push(IterationTiming {
                iteration: run_index - args.warmup,
                elapsed_ms,
                output_payload_bytes: output.payload.len(),
                output_flags: output.desc.flags,
            });
        }
    }

    let report = MicrobenchReport {
        command: "glm-dsa-layer-microbench",
        model_id: args.model_id,
        stage_model: args.stage_model,
        layer_start: args.layer_start,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        activation_width: args.activation_width,
        tokens: args.tokens,
        warmup: args.warmup,
        iterations: args.iterations,
        n_gpu_layers: args.n_gpu_layers,
        n_batch: runtime_config.n_batch,
        n_ubatch: runtime_config.n_ubatch,
        flags: MicrobenchFlags {
            direct_sparse_attn: args.direct_sparse_attn,
            direct_sparse_prefill: args.direct_sparse_prefill,
            fused_sparse_mask: args.fused_sparse_mask,
            parallel_lightning_indexer: args.parallel_lightning_indexer,
            op_timing: args.op_timing,
        },
        selected_parts: selected
            .selected_parts
            .iter()
            .map(package_part_summary)
            .collect(),
        input_payload_bytes: input.payload.len(),
        timings,
    };

    write_report(args.output.as_deref(), &report)
}

fn validate_args(args: &GlmDsaLayerMicrobenchArgs) -> Result<()> {
    if args.layer_start >= args.layer_end {
        bail!("layer_start must be less than layer_end");
    }
    if args.layer_start == 0 {
        bail!(
            "glm-dsa-layer-microbench expects a nonzero layer_start and synthetic activation input"
        );
    }
    if args.tokens == 0 {
        bail!("tokens must be greater than zero");
    }
    if args.iterations == 0 {
        bail!("iterations must be greater than zero");
    }
    if args.activation_width == 0 {
        bail!("activation_width must be greater than zero");
    }
    Ok(())
}

fn configure_env_flags(args: &GlmDsaLayerMicrobenchArgs) {
    set_env_flag(
        "SKIPPY_GLM_DSA_ENABLE_DIRECT_SPARSE_ATTN",
        args.direct_sparse_attn,
    );
    set_env_flag(
        "SKIPPY_GLM_DSA_ENABLE_DIRECT_SPARSE_PREFILL",
        args.direct_sparse_prefill,
    );
    set_env_flag(
        "SKIPPY_GLM_DSA_ENABLE_FUSED_SPARSE_MASK",
        args.fused_sparse_mask,
    );
    set_env_flag(
        "LLAMA_GLM_DSA_PARALLEL_LIGHTNING_INDEXER",
        args.parallel_lightning_indexer,
    );
    set_env_flag("SKIPPY_GLM_DSA_OP_TIMING", args.op_timing);
}

fn set_env_flag(name: &str, enabled: bool) {
    // This command is single-threaded and sets native runtime flags before opening llama.cpp.
    unsafe {
        std::env::set_var(name, if enabled { "1" } else { "0" });
    }
}

fn package_request(args: &GlmDsaLayerMicrobenchArgs) -> PackageStageRequest {
    PackageStageRequest {
        model_id: args.model_id.clone(),
        topology_id: "glm-dsa-layer-microbench".to_string(),
        package_ref: args.stage_model.to_string_lossy().to_string(),
        stage_id: format!("layers-{}-{}", args.layer_start, args.layer_end),
        layer_start: args.layer_start,
        layer_end: args.layer_end,
        include_embeddings: false,
        include_output: false,
    }
}

fn runtime_config(args: &GlmDsaLayerMicrobenchArgs) -> Result<RuntimeConfig> {
    Ok(RuntimeConfig {
        stage_index: 0,
        layer_start: args.layer_start,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        lane_count: 1,
        n_batch: Some(args.n_batch.unwrap_or_else(|| bounded_u32(args.tokens))),
        n_ubatch: Some(args.n_ubatch.unwrap_or_else(|| bounded_u32(args.tokens))),
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: args.n_gpu_layers,
        selected_backend_device: None,
        cache_type_k: parse_cache_type(&args.cache_type_k).context("parse cache_type_k")?,
        cache_type_v: parse_cache_type(&args.cache_type_v).context("parse cache_type_v")?,
        flash_attn_type: FlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::LayerPackage,
        projector_path: None,
        use_mmap: false,
        use_mmap_prefetch: false,
        use_mmap_buffer: false,
        include_embeddings: false,
        include_output: false,
        filter_tensors_on_load: true,
    })
}

fn bounded_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX).max(1)
}

fn synthetic_activation_frame(args: &GlmDsaLayerMicrobenchArgs) -> Result<ActivationFrame> {
    let width = usize::try_from(args.activation_width).context("activation_width exceeds usize")?;
    let value_count = args
        .tokens
        .checked_mul(width)
        .context("synthetic activation value count overflow")?;
    let payload_bytes = value_count
        .checked_mul(std::mem::size_of::<f32>())
        .context("synthetic activation payload size overflow")?;
    let mut payload = Vec::with_capacity(payload_bytes);
    for token in 0..args.tokens {
        for dim in 0..width {
            let value = synthetic_activation_value(token, dim);
            payload.extend_from_slice(&value.to_ne_bytes());
        }
    }
    Ok(ActivationFrame {
        desc: ActivationDesc {
            version: 1,
            dtype: RuntimeActivationDType::F32,
            layout: RuntimeActivationLayout::TokenMajor,
            producer_stage_index: -1,
            layer_start: i32::try_from(args.layer_start.saturating_sub(1))
                .context("layer_start exceeds i32")?,
            layer_end: i32::try_from(args.layer_start).context("layer_start exceeds i32")?,
            token_count: u32::try_from(args.tokens).context("tokens exceeds u32")?,
            sequence_count: 1,
            payload_bytes: u64::try_from(payload.len()).context("payload length exceeds u64")?,
            flags: 0,
        },
        payload,
    })
}

fn synthetic_activation_value(token: usize, dim: usize) -> f32 {
    let residue = ((token.wrapping_mul(31) + dim.wrapping_mul(17)) % 97) as f32;
    (residue / 97.0 - 0.5) * 0.02
}

fn positions(tokens: usize) -> Result<Vec<i32>> {
    (0..tokens)
        .map(|position| i32::try_from(position).context("position exceeds i32"))
        .collect()
}

fn package_part_summary(part: &PackagePart) -> PackagePartSummary {
    PackagePartSummary {
        role: part.role.clone(),
        layer_index: part.layer_index,
        path: part.path.clone(),
        artifact_bytes: part.artifact_bytes,
    }
}

fn write_report(output: Option<&Path>, report: &MicrobenchReport) -> Result<()> {
    let encoded = format!("{}\n", serde_json::to_string_pretty(report)?);
    if let Some(output) = output {
        fs::write(output, encoded).with_context(|| format!("write {}", output.display()))?;
    } else {
        print!("{encoded}");
    }
    Ok(())
}

#[derive(Serialize)]
struct MicrobenchReport {
    command: &'static str,
    model_id: String,
    stage_model: PathBuf,
    layer_start: u32,
    layer_end: u32,
    ctx_size: u32,
    activation_width: u32,
    tokens: usize,
    warmup: usize,
    iterations: usize,
    n_gpu_layers: i32,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
    flags: MicrobenchFlags,
    selected_parts: Vec<PackagePartSummary>,
    input_payload_bytes: usize,
    timings: Vec<IterationTiming>,
}

#[derive(Serialize)]
struct MicrobenchFlags {
    direct_sparse_attn: bool,
    direct_sparse_prefill: bool,
    fused_sparse_mask: bool,
    parallel_lightning_indexer: bool,
    op_timing: bool,
}

#[derive(Serialize)]
struct PackagePartSummary {
    role: String,
    layer_index: Option<u32>,
    path: PathBuf,
    artifact_bytes: u64,
}

#[derive(Serialize)]
struct IterationTiming {
    iteration: usize,
    elapsed_ms: f64,
    output_payload_bytes: usize,
    output_flags: u64,
}
