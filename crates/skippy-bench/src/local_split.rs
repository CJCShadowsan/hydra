use std::{
    fs,
    net::SocketAddr,
    path::Path,
    process::{Command, Stdio},
};

use anyhow::{Context, Result, bail};
use model_artifact::ModelIdentity;
use serde_json::json;
use skippy_protocol::binary::{
    StageStateHeader, StageWireMessage, WireMessageKind, WireReplyKind, recv_reply,
    write_stage_message,
};
use skippy_runtime::{RuntimeConfig, RuntimeLoadMode, StageModel};
use skippy_topology::{
    BoundaryDecision, NodeSpec, PlannerPolicy, TopologyPlanRequest, WireValidation,
    dense_attention_layers, infer_family_capability, plan_contiguous_with_splits,
};

use crate::{
    cli::{
        LocalSplitBinaryArgs, LocalSplitChainBinaryArgs, LocalSplitCompareArgs,
        LocalSplitInprocessArgs,
    },
    model_identity::model_identity_for_path,
    support::{
        ChildGuard, activation_width, connect_ready, generate_run_id, parse_wire_dtype,
        temp_config_path_for,
    },
};

const LARGE_LOCAL_CHAIN_MODEL_BYTES: u64 = 32 * 1024 * 1024 * 1024;

struct BinarySplitResult {
    model_identity: ModelIdentity,
    token_id: i32,
    predicted_token: i32,
    activation_width: i32,
    wire_dtype: String,
    boundary_producer_stage_index: i32,
    boundary_layer_start: i32,
    boundary_layer_end: i32,
    boundary_token_count: u32,
    boundary_payload_bytes: u64,
    boundary_wire_payload_bytes: usize,
}

struct BinaryChainResult {
    model_identity: ModelIdentity,
    token_id: i32,
    predicted_token: i32,
    activation_width: i32,
    wire_dtype: String,
    splits: Vec<u32>,
    layer_end: u32,
    stages: Vec<serde_json::Value>,
    boundary_transfers: Vec<serde_json::Value>,
}

pub fn local_split_binary(args: LocalSplitBinaryArgs) -> Result<()> {
    let output = args.output.clone();
    let result = run_binary_split(BinarySplitConfig {
        stage_server_bin: args.stage_server_bin,
        model_path: args.model_path,
        model_id: args.model_id,
        split_layer: args.split_layer,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        prompt: args.prompt,
        stage1_bind_addr: args.stage1_bind_addr,
        activation_wire_dtype: args.activation_wire_dtype,
        child_logs: args.child_logs,
        startup_timeout_secs: args.startup_timeout_secs,
    })?;

    write_or_print_report(
        &json!({
            "mode": "local-split-binary",
            "model_identity": result.model_identity,
            "token_id": result.token_id,
            "predicted_token": result.predicted_token,
            "activation_width": result.activation_width,
            "wire_dtype": result.wire_dtype,
            "boundary": {
                "producer_stage_index": result.boundary_producer_stage_index,
                "layer_start": result.boundary_layer_start,
                "layer_end": result.boundary_layer_end,
                "token_count": result.boundary_token_count,
                "payload_bytes": result.boundary_payload_bytes,
                "wire_payload_bytes": result.boundary_wire_payload_bytes,
            }
        }),
        output.as_deref(),
    )?;

    Ok(())
}

pub fn local_split_compare(args: LocalSplitCompareArgs) -> Result<()> {
    let output_path = args.output.clone();
    let baseline = run_full_model_decode(
        &args.model_path,
        args.layer_end,
        args.ctx_size,
        args.n_gpu_layers,
        &args.prompt,
    )?;
    let split = run_binary_split(BinarySplitConfig {
        stage_server_bin: args.stage_server_bin,
        model_path: args.model_path,
        model_id: args.model_id,
        split_layer: args.split_layer,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        prompt: args.prompt,
        stage1_bind_addr: args.stage1_bind_addr,
        activation_wire_dtype: args.activation_wire_dtype,
        child_logs: args.child_logs,
        startup_timeout_secs: args.startup_timeout_secs,
    })?;

    let matches = baseline.predicted_token == split.predicted_token;
    let output = json!({
        "mode": "local-split-compare",
        "model_identity": split.model_identity,
        "matches": matches,
        "baseline": {
            "token_id": baseline.token_id,
            "predicted_token": baseline.predicted_token,
        },
        "split": {
            "token_id": split.token_id,
            "predicted_token": split.predicted_token,
            "activation_width": split.activation_width,
            "wire_dtype": split.wire_dtype,
            "boundary": {
                "producer_stage_index": split.boundary_producer_stage_index,
                "layer_start": split.boundary_layer_start,
                "layer_end": split.boundary_layer_end,
                "token_count": split.boundary_token_count,
                "payload_bytes": split.boundary_payload_bytes,
                "wire_payload_bytes": split.boundary_wire_payload_bytes,
            }
        }
    });
    write_or_print_report(&output, output_path.as_deref())?;

    if !matches && !args.allow_mismatch {
        bail!(
            "split predicted token {} did not match full-model predicted token {}",
            split.predicted_token,
            baseline.predicted_token
        );
    }

    Ok(())
}

pub fn local_split_chain_binary(args: LocalSplitChainBinaryArgs) -> Result<()> {
    let output = args.output.clone();
    let result = run_binary_chain(args)?;
    write_or_print_report(
        &json!({
            "mode": "local-split-chain-binary",
            "model_identity": result.model_identity,
            "token_id": result.token_id,
            "predicted_token": result.predicted_token,
            "activation_width": result.activation_width,
            "wire_dtype": result.wire_dtype,
            "splits": result.splits,
            "layer_end": result.layer_end,
            "stages": result.stages,
            "boundary_transfers": result.boundary_transfers,
        }),
        output.as_deref(),
    )?;
    Ok(())
}

fn write_or_print_report(report: &serde_json::Value, output: Option<&Path>) -> Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    if let Some(output) = output {
        if let Some(parent) = output.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("create report dir {}", parent.display()))?;
        }
        fs::write(output, format!("{json}\n"))
            .with_context(|| format!("write local split report {}", output.display()))?;
    }
    println!("{json}");
    Ok(())
}

struct FullModelResult {
    token_id: i32,
    predicted_token: i32,
}

fn run_full_model_decode(
    model_path: &std::path::Path,
    layer_end: u32,
    ctx_size: u32,
    n_gpu_layers: i32,
    prompt: &str,
) -> Result<FullModelResult> {
    let config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end,
        ctx_size,
        lane_count: 1,
        n_batch: None,
        n_ubatch: None,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers,
        selected_backend_device: None,
        cache_type_k: skippy_runtime::GGML_TYPE_F16,
        cache_type_v: skippy_runtime::GGML_TYPE_F16,
        flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::RuntimeSlice,
        projector_path: None,
        include_embeddings: true,
        include_output: true,
        filter_tensors_on_load: false,
    };
    let model = StageModel::open(model_path, &config).context("failed to open full model")?;
    let tokens = model
        .tokenize(prompt, true)
        .context("failed to tokenize prompt with full model")?;
    let token_id = *tokens.first().context("prompt produced no tokens")?;
    let mut session = model
        .create_session()
        .context("failed to create full-model session")?;
    let predicted_token = session
        .decode_step_frame(token_id, None, 0)
        .context("full model failed to decode")?
        .0;
    Ok(FullModelResult {
        token_id,
        predicted_token,
    })
}

struct BinarySplitConfig {
    stage_server_bin: std::path::PathBuf,
    model_path: std::path::PathBuf,
    model_id: String,
    split_layer: u32,
    layer_end: u32,
    ctx_size: u32,
    n_gpu_layers: i32,
    prompt: String,
    stage1_bind_addr: std::net::SocketAddr,
    activation_wire_dtype: String,
    child_logs: bool,
    startup_timeout_secs: u64,
}

fn run_binary_split(args: BinarySplitConfig) -> Result<BinarySplitResult> {
    if args.split_layer == 0 || args.split_layer >= args.layer_end {
        bail!("split_layer must be greater than zero and less than layer_end");
    }
    validate_local_topology_plan(
        &args.model_path,
        args.layer_end,
        &[args.split_layer],
        2,
        &args.activation_wire_dtype,
    )?;
    let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
    let model_identity = model_identity_for_path(&args.model_id, Some(&args.model_path))?;
    let stage0_config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.split_layer,
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
        include_output: false,
        filter_tensors_on_load: true,
    };
    let stage0 =
        StageModel::open(&args.model_path, &stage0_config).context("failed to open stage 0")?;
    let tokens = stage0
        .tokenize(&args.prompt, true)
        .context("failed to tokenize prompt")?;
    let token_id = *tokens.first().context("prompt produced no tokens")?;
    let mut session0 = stage0
        .create_session()
        .context("failed to create stage 0 session")?;
    let (_boundary_prediction, boundary) = session0
        .decode_step_frame(token_id, None, 0)
        .context("stage 0 failed to produce activation frame")?;
    if boundary.payload.is_empty() {
        bail!("stage 0 produced an empty activation frame");
    }
    let activation_width = activation_width(&boundary)?;

    let run_id = generate_run_id();
    let config_path = temp_config_path_for(&run_id, "stage-1");
    let config = json!({
        "run_id": run_id,
        "topology_id": "local-split-binary",
        "model_id": model_identity.model_id,
        "model_path": args.model_path,
        "stage_id": "stage-1",
        "stage_index": 1,
        "layer_start": args.split_layer,
        "layer_end": args.layer_end,
        "ctx_size": args.ctx_size,
        "n_gpu_layers": args.n_gpu_layers,
        "filter_tensors_on_load": true,
        "load_mode": "runtime-slice",
        "bind_addr": args.stage1_bind_addr,
        "upstream": {
            "stage_id": "stage-0",
            "stage_index": 0,
            "endpoint": "driver"
        },
        "downstream": null
    });
    fs::write(&config_path, serde_json::to_vec_pretty(&config)?)
        .with_context(|| format!("failed to write {}", config_path.display()))?;

    let mut stage_command = Command::new(&args.stage_server_bin);
    stage_command.args([
        "serve-binary",
        "--config",
        config_path
            .to_str()
            .context("stage config path is not valid UTF-8")?,
        "--activation-width",
        &activation_width.to_string(),
        "--activation-wire-dtype",
        &args.activation_wire_dtype,
    ]);
    if args.child_logs {
        stage_command
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
    } else {
        stage_command.stdout(Stdio::null()).stderr(Stdio::null());
    }
    let _stage1 = ChildGuard::spawn(stage_command)?;

    let mut stream = connect_ready(args.stage1_bind_addr, args.startup_timeout_secs)
        .context("stage 1 binary server did not become ready")?;
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, wire_dtype);
    state.prompt_token_count = 0;
    state.decode_step = 0;
    state.current_token = token_id;
    state.source_stage_index = 0;
    state.flags |=
        skippy_protocol::binary::activation_state_flags_from_frame_flags(boundary.desc.flags);
    let activation = skippy_protocol::binary::encode_f32_activation_payload_with_state_flags(
        wire_dtype,
        1,
        activation_width,
        &boundary.payload,
        state.flags,
    )
    .context("failed to encode boundary activation for wire")?;
    let message = StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: 0,
        token_count: 1,
        state,
        request_id: 1,
        session_id: 1,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: vec![token_id],
        positions: vec![0],
        activation,
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut stream, &message, wire_dtype).context("send binary decode")?;
    let reply = recv_reply(&mut stream).context("receive binary reply")?;
    if reply.kind != WireReplyKind::PredictedToken {
        bail!("expected predicted-token reply, got {:?}", reply.kind);
    }
    write_stage_message(&mut stream, &StageWireMessage::stop(wire_dtype), wire_dtype)
        .context("send binary stop")?;

    Ok(BinarySplitResult {
        model_identity,
        token_id,
        predicted_token: reply.predicted,
        activation_width,
        wire_dtype: args.activation_wire_dtype,
        boundary_producer_stage_index: boundary.desc.producer_stage_index,
        boundary_layer_start: boundary.desc.layer_start,
        boundary_layer_end: boundary.desc.layer_end,
        boundary_token_count: boundary.desc.token_count,
        boundary_payload_bytes: boundary.desc.payload_bytes,
        boundary_wire_payload_bytes: message.activation.len(),
    })
}

fn run_binary_chain(args: LocalSplitChainBinaryArgs) -> Result<BinaryChainResult> {
    let splits = chain_splits(&args)?;
    validate_chain_splits(&splits, args.layer_end)?;
    let stage_count = splits.len() + 1;
    ensure_local_chain_memory_guard(&args, stage_count)?;
    let bind_addrs = chain_bind_addrs(&args, stage_count - 1)?;
    validate_local_topology_plan(
        &args.model_path,
        args.layer_end,
        &splits,
        stage_count,
        &args.activation_wire_dtype,
    )?;
    let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
    let model_identity = model_identity_for_path(&args.model_id, Some(&args.model_path))?;
    let stage0_config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: splits[0],
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
        include_output: false,
        filter_tensors_on_load: true,
    };
    let stage0 =
        StageModel::open(&args.model_path, &stage0_config).context("failed to open stage 0")?;
    let tokens = stage0
        .tokenize(&args.prompt, true)
        .context("failed to tokenize prompt")?;
    let token_id = *tokens.first().context("prompt produced no tokens")?;
    let mut session0 = stage0
        .create_session()
        .context("failed to create stage 0 session")?;
    let (_boundary_prediction, boundary) = session0
        .decode_step_frame(token_id, None, 0)
        .context("stage 0 failed to produce activation frame")?;
    if boundary.payload.is_empty() {
        bail!("stage 0 produced an empty activation frame");
    }
    let activation_width = activation_width(&boundary)?;

    let run_id = generate_run_id();
    let mut stage_guards = Vec::with_capacity(stage_count - 1);
    for stage_index in (1..stage_count).rev() {
        let config_path = temp_config_path_for(&run_id, &format!("stage-{stage_index}"));
        let config = chain_stage_config(
            &args,
            &model_identity,
            &run_id,
            &splits,
            &bind_addrs,
            stage_index,
        );
        fs::write(&config_path, serde_json::to_vec_pretty(&config)?)
            .with_context(|| format!("failed to write {}", config_path.display()))?;

        let mut command = Command::new(&args.stage_server_bin);
        command.args([
            "serve-binary",
            "--config",
            config_path
                .to_str()
                .with_context(|| format!("stage {stage_index} config path is not valid UTF-8"))?,
            "--activation-width",
            &activation_width.to_string(),
            "--activation-wire-dtype",
            &args.activation_wire_dtype,
        ]);
        configure_child_logs(&mut command, args.child_logs);
        stage_guards.push(ChildGuard::spawn(command)?);
    }

    let mut stream = connect_ready(bind_addrs[0], args.startup_timeout_secs)
        .context("stage 1 binary server did not become ready")?;
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, wire_dtype);
    state.prompt_token_count = 0;
    state.decode_step = 0;
    state.current_token = token_id;
    state.source_stage_index = 0;
    state.flags |=
        skippy_protocol::binary::activation_state_flags_from_frame_flags(boundary.desc.flags);
    let activation = skippy_protocol::binary::encode_f32_activation_payload_with_state_flags(
        wire_dtype,
        1,
        activation_width,
        &boundary.payload,
        state.flags,
    )
    .context("failed to encode boundary activation for wire")?;
    let message = StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: 0,
        token_count: 1,
        state,
        request_id: 2,
        session_id: 2,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: vec![token_id],
        positions: vec![0],
        activation,
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut stream, &message, wire_dtype).context("send binary chain decode")?;
    let reply = recv_reply(&mut stream).context("receive binary chain reply")?;
    if reply.kind != WireReplyKind::PredictedToken {
        bail!("expected predicted-token reply, got {:?}", reply.kind);
    }
    write_stage_message(&mut stream, &StageWireMessage::stop(wire_dtype), wire_dtype)
        .context("send binary chain stop")?;

    Ok(BinaryChainResult {
        model_identity,
        token_id,
        predicted_token: reply.predicted,
        activation_width,
        wire_dtype: args.activation_wire_dtype,
        stages: chain_stage_reports(
            &splits,
            args.layer_end,
            boundary.desc.payload_bytes,
            message.activation.len(),
        ),
        boundary_transfers: chain_boundary_transfers(
            &splits,
            boundary.desc.payload_bytes,
            message.activation.len(),
        ),
        splits,
        layer_end: args.layer_end,
    })
}

fn ensure_local_chain_memory_guard(
    args: &LocalSplitChainBinaryArgs,
    stage_count: usize,
) -> Result<()> {
    if args.allow_high_memory_local_chain || stage_count <= 2 {
        return Ok(());
    }
    let model_bytes = fs::metadata(&args.model_path)
        .with_context(|| format!("stat model {}", args.model_path.display()))?
        .len();
    if model_bytes < LARGE_LOCAL_CHAIN_MODEL_BYTES {
        return Ok(());
    }
    bail!(
        "refusing high-memory local split-chain: {} is {:.1} GiB and this command would load {} local stages concurrently; use a smaller model, use package/lab evidence, or pass --allow-high-memory-local-chain to override",
        args.model_path.display(),
        model_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
        stage_count
    );
}

fn chain_splits(args: &LocalSplitChainBinaryArgs) -> Result<Vec<u32>> {
    if let Some(splits) = args.splits.as_deref() {
        return parse_split_list(splits);
    }
    Ok(vec![args.split_layer_1, args.split_layer_2])
}

fn parse_split_list(splits: &str) -> Result<Vec<u32>> {
    splits
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<u32>()
                .with_context(|| format!("parse split boundary {part:?}"))
        })
        .collect()
}

fn validate_chain_splits(splits: &[u32], layer_end: u32) -> Result<()> {
    if splits.len() < 2 {
        bail!("local split chain requires at least two split boundaries");
    }
    let mut previous = 0;
    for &split in splits {
        if split == 0 || split >= layer_end {
            bail!("split boundaries must be within 1..layer_end");
        }
        if split <= previous {
            bail!("split boundaries must partition 0..layer_end in ascending order");
        }
        previous = split;
    }
    Ok(())
}

fn chain_bind_addrs(
    args: &LocalSplitChainBinaryArgs,
    spawned_stages: usize,
) -> Result<Vec<SocketAddr>> {
    let mut addrs = Vec::with_capacity(spawned_stages);
    if spawned_stages == 0 {
        return Ok(addrs);
    }
    addrs.push(args.stage1_bind_addr);
    if spawned_stages == 1 {
        return Ok(addrs);
    }
    addrs.push(args.stage2_bind_addr);
    let ip = args.stage2_bind_addr.ip();
    let base_port = args.stage2_bind_addr.port();
    for offset in 1..spawned_stages.saturating_sub(1) {
        let port = base_port
            .checked_add(u16::try_from(offset).context("too many local split stages")?)
            .context("local split stage port overflow")?;
        addrs.push(SocketAddr::new(ip, port));
    }
    Ok(addrs)
}

fn chain_stage_config(
    args: &LocalSplitChainBinaryArgs,
    model_identity: &ModelIdentity,
    run_id: &str,
    splits: &[u32],
    bind_addrs: &[SocketAddr],
    stage_index: usize,
) -> serde_json::Value {
    let stage_count = splits.len() + 1;
    let layer_start = splits[stage_index - 1];
    let layer_end = if stage_index < splits.len() {
        splits[stage_index]
    } else {
        args.layer_end
    };
    let upstream_endpoint = if stage_index == 1 {
        "driver".to_string()
    } else {
        format!("tcp://{}", bind_addrs[stage_index - 2])
    };
    let downstream = (stage_index + 1 < stage_count).then(|| {
        json!({
            "stage_id": format!("stage-{}", stage_index + 1),
            "stage_index": stage_index + 1,
            "endpoint": format!("tcp://{}", bind_addrs[stage_index])
        })
    });
    json!({
        "run_id": run_id,
        "topology_id": "local-split-chain-binary",
        "model_id": model_identity.model_id,
        "model_path": args.model_path,
        "stage_id": format!("stage-{stage_index}"),
        "stage_index": stage_index,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "ctx_size": args.ctx_size,
        "n_gpu_layers": args.n_gpu_layers,
        "filter_tensors_on_load": true,
        "load_mode": "runtime-slice",
        "bind_addr": bind_addrs[stage_index - 1],
        "upstream": {
            "stage_id": format!("stage-{}", stage_index - 1),
            "stage_index": stage_index - 1,
            "endpoint": upstream_endpoint
        },
        "downstream": downstream
    })
}

fn chain_stage_reports(
    splits: &[u32],
    layer_end: u32,
    observed_payload_bytes: u64,
    observed_wire_payload_bytes: usize,
) -> Vec<serde_json::Value> {
    let stage_count = splits.len() + 1;
    (0..stage_count)
        .map(|stage_index| {
            let layer_start = if stage_index == 0 {
                0
            } else {
                splits[stage_index - 1]
            };
            let stage_layer_end = if stage_index < splits.len() {
                splits[stage_index]
            } else {
                layer_end
            };
            let mut stage = json!({
                "stage_index": stage_index,
                "layer_start": layer_start,
                "layer_end": stage_layer_end,
            });
            if stage_index + 1 < stage_count {
                stage["payload_bytes"] = json!(observed_payload_bytes);
                stage["wire_payload_bytes"] = json!(observed_wire_payload_bytes);
                stage["transfer_bytes_source"] = if stage_index == 0 {
                    json!("observed_driver_boundary")
                } else {
                    json!("estimated_from_first_boundary_shape")
                };
            } else {
                stage["returned_predicted_token"] = json!(true);
            }
            if stage_index > 0 {
                stage["forwarded_over_binary"] = json!(true);
            }
            stage
        })
        .collect()
}

fn chain_boundary_transfers(
    splits: &[u32],
    observed_payload_bytes: u64,
    observed_wire_payload_bytes: usize,
) -> Vec<serde_json::Value> {
    splits
        .iter()
        .enumerate()
        .map(|(index, boundary)| {
            json!({
                "from_stage": index,
                "to_stage": index + 1,
                "layer_boundary": boundary,
                "payload_bytes": observed_payload_bytes,
                "wire_payload_bytes": observed_wire_payload_bytes,
                "source": if index == 0 {
                    "observed_driver_boundary"
                } else {
                    "estimated_from_first_boundary_shape"
                }
            })
        })
        .collect()
}

fn validate_local_topology_plan(
    model_path: &std::path::Path,
    layer_end: u32,
    splits: &[u32],
    stage_count: usize,
    activation_wire_dtype: &str,
) -> Result<()> {
    let identity = model_path.display().to_string();
    let family = infer_family_capability(&identity, layer_end, 0);
    let request = TopologyPlanRequest {
        topology_id: "local-split-binary".to_string(),
        model_id: identity,
        layers: dense_attention_layers(layer_end, 0),
        nodes: (0..stage_count)
            .map(|index| NodeSpec {
                node_id: format!("local-stage-{index}"),
                cached_slice_bytes: 0,
                vram_bytes: 0,
            })
            .collect(),
        family: family.clone(),
        policy: PlannerPolicy::default(),
    };
    let plan = plan_contiguous_with_splits(&request, splits).context("topology planner failed")?;

    if activation_wire_dtype.eq_ignore_ascii_case("q8") {
        match family.as_ref().map(|family| family.q8_wire_validation) {
            Some(WireValidation::Validated) => {}
            Some(WireValidation::Rejected) => {
                bail!(
                    "topology planner rejected q8 activation wire dtype for {}; use f16 or add a passing q8 correctness record",
                    model_path.display()
                );
            }
            Some(WireValidation::Untested) => {
                bail!(
                    "topology planner has no q8 validation for {}; use f16 until this family/split passes correctness",
                    model_path.display()
                );
            }
            None => {}
        }
    }

    let rejected = plan
        .boundaries
        .iter()
        .filter(|boundary| boundary.decision == BoundaryDecision::Rejected)
        .collect::<Vec<_>>();
    if !rejected.is_empty() {
        let reasons = rejected
            .iter()
            .map(|boundary| {
                format!(
                    "layer {}: {:?}: {}",
                    boundary.layer_boundary,
                    boundary.reason_codes,
                    boundary.messages.join("; ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        bail!("topology planner rejected split plan:\n{reasons}");
    }

    Ok(())
}

fn configure_child_logs(command: &mut Command, child_logs: bool) {
    if child_logs {
        command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    } else {
        command.stdout(Stdio::null()).stderr(Stdio::null());
    }
}

pub fn local_split_inprocess(args: LocalSplitInprocessArgs) -> Result<()> {
    if args.split_layer == 0 || args.split_layer >= args.layer_end {
        bail!("split_layer must be greater than zero and less than layer_end");
    }

    let stage0_config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.split_layer,
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
        include_output: false,
        filter_tensors_on_load: true,
    };
    let stage1_config = RuntimeConfig {
        stage_index: 1,
        layer_start: args.split_layer,
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
        include_embeddings: false,
        include_output: true,
        filter_tensors_on_load: true,
    };

    let stage0 =
        StageModel::open(&args.model_path, &stage0_config).context("failed to open stage 0")?;
    let stage1 =
        StageModel::open(&args.model_path, &stage1_config).context("failed to open stage 1")?;
    let tokens = stage0
        .tokenize(&args.prompt, true)
        .context("failed to tokenize prompt")?;
    let token_id = *tokens.first().context("prompt produced no tokens")?;

    let mut session0 = stage0
        .create_session()
        .context("failed to create stage 0 session")?;
    let mut session1 = stage1
        .create_session()
        .context("failed to create stage 1 session")?;

    let (_boundary_prediction, boundary) = session0
        .decode_step_frame(token_id, None, 0)
        .context("stage 0 failed to produce activation frame")?;
    if boundary.payload.is_empty() {
        bail!("stage 0 produced an empty activation frame");
    }

    let (predicted_token, final_frame) = session1
        .decode_step_frame(token_id, Some(&boundary), 0)
        .context("stage 1 failed to consume activation frame")?;
    if !final_frame.payload.is_empty() {
        bail!("final stage unexpectedly produced an activation payload");
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "mode": "local-split-inprocess",
            "token_id": token_id,
            "predicted_token": predicted_token,
            "boundary": {
                "producer_stage_index": boundary.desc.producer_stage_index,
                "layer_start": boundary.desc.layer_start,
                "layer_end": boundary.desc.layer_end,
                "token_count": boundary.desc.token_count,
                "sequence_count": boundary.desc.sequence_count,
                "payload_bytes": boundary.desc.payload_bytes,
                "actual_payload_bytes": boundary.payload.len(),
            },
            "final": {
                "producer_stage_index": final_frame.desc.producer_stage_index,
                "layer_start": final_frame.desc.layer_start,
                "layer_end": final_frame.desc.layer_end,
                "payload_bytes": final_frame.desc.payload_bytes,
            }
        }))?
    );

    Ok(())
}
