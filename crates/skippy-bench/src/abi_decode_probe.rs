use anyhow::{Context, Result, bail};
use serde_json::json;
use skippy_runtime::{RuntimeConfig, RuntimeLoadMode, StageModel};

use crate::cli::AbiDecodeProbeArgs;

pub fn abi_decode_probe(args: AbiDecodeProbeArgs) -> Result<()> {
    if args.layer_start >= args.layer_end {
        bail!("layer_start must be less than layer_end");
    }
    if args.measured_tokens == 0 {
        bail!("measured_tokens must be greater than zero");
    }

    let model = StageModel::open(
        &args.model_path,
        &RuntimeConfig {
            stage_index: 0,
            layer_start: args.layer_start,
            layer_end: args.layer_end,
            ctx_size: args.ctx_size,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_threads: None,
            n_threads_batch: None,
            n_gpu_layers: args.n_gpu_layers,
            selected_backend_device: None,
            cache_type_k: skippy_runtime::GGML_TYPE_F16,
            cache_type_v: skippy_runtime::GGML_TYPE_F16,
            flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
            load_mode: RuntimeLoadMode::RuntimeSlice,
            projector_path: None,
            include_embeddings: true,
            include_output: true,
            filter_tensors_on_load: false,
        },
    )
    .with_context(|| format!("open model {}", args.model_path.display()))?;

    let prompt_tokens = model
        .tokenize(&args.prompt, true)
        .context("tokenize probe prompt")?;
    let seed_token = *prompt_tokens
        .last()
        .context("probe prompt produced no tokens")?;
    let mut session = model.create_session().context("create probe session")?;
    if prompt_tokens.len() > 1 {
        session
            .prefill_chunked(&prompt_tokens[..prompt_tokens.len() - 1])
            .context("prefill probe prompt")?;
    }
    let result = session
        .benchmark_decode(seed_token, args.warmup_tokens, args.measured_tokens)
        .context("run native decode benchmark")?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "mode": "abi-decode-probe",
            "model_path": args.model_path,
            "ctx_size": args.ctx_size,
            "n_gpu_layers": args.n_gpu_layers,
            "layer_start": args.layer_start,
            "layer_end": args.layer_end,
            "prompt_token_count": prompt_tokens.len(),
            "warmup_tokens": result.warmup_tokens,
            "measured_tokens": result.measured_tokens,
            "elapsed_ms": result.elapsed_ms,
            "tokens_per_second": result.tokens_per_second,
            "final_token": result.final_token,
        }))?
    );
    Ok(())
}
