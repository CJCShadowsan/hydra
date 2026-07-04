use std::{
    fs,
    net::TcpStream,
    process::{Command, Stdio},
};

use anyhow::{Context, Result, bail};
use model_artifact::ModelIdentity;
use serde_json::json;
use skippy_protocol::LoadMode as ProtocolLoadMode;
use skippy_protocol::binary::{
    StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind, WireReplyKind,
    activation_state_flags_from_frame_flags, activation_wire_bytes_with_state_flags, recv_reply,
    state_flags, write_stage_message,
};
use skippy_runtime::{
    ActivationFrame, RuntimeActivationDType, RuntimeActivationLayout, RuntimeConfig,
    RuntimeLoadMode, StageModel,
    package::{PackageStageRequest, inspect_layer_package, select_layer_package_parts},
};
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
        ChildGuard, activation_width, connect_ready, connect_ready_while_child_running,
        generate_run_id, parse_wire_dtype, temp_config_path_for,
    },
};

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
    boundary_wire_sideband_bytes: usize,
    boundary_wire_sideband_i32_count: usize,
    boundary_wire_message_bytes: usize,
}

struct BinaryChainResult {
    model_identity: ModelIdentity,
    token_id: i32,
    predicted_token: i32,
    activation_width: i32,
    wire_dtype: String,
    stage0_wire_payload_bytes: usize,
    stage0_wire_sideband_bytes: usize,
    stage0_wire_sideband_i32_count: usize,
    stage0_wire_message_bytes: usize,
    stage0_payload_bytes: u64,
    split_layer_1: u32,
    split_layer_2: u32,
    layer_end: u32,
}

fn ensure_reply_kind(
    reply: &skippy_protocol::binary::StageReply,
    expected: WireReplyKind,
) -> Result<()> {
    if reply.kind != expected {
        bail!("expected {expected:?} reply, got {:?}", reply.kind);
    }
    Ok(())
}

pub fn local_split_binary(args: LocalSplitBinaryArgs) -> Result<()> {
    let result = run_binary_split(BinarySplitConfig {
        stage_server_bin: args.stage_server_bin,
        model_path: args.model_path,
        stage_load_mode: args.stage_load_mode,
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

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
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
                "wire_sideband_bytes": result.boundary_wire_sideband_bytes,
                "wire_sideband_i32_count": result.boundary_wire_sideband_i32_count,
                "wire_message_bytes": result.boundary_wire_message_bytes,
            }
        }))?
    );

    Ok(())
}

pub fn local_split_compare(args: LocalSplitCompareArgs) -> Result<()> {
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
        stage_load_mode: "runtime-slice".to_string(),
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
                "wire_sideband_bytes": split.boundary_wire_sideband_bytes,
                "wire_sideband_i32_count": split.boundary_wire_sideband_i32_count,
                "wire_message_bytes": split.boundary_wire_message_bytes,
            }
        }
    });
    println!("{}", serde_json::to_string_pretty(&output)?);

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
    let result = run_binary_chain(args)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "mode": "local-split-chain-binary",
            "model_identity": result.model_identity,
            "token_id": result.token_id,
            "predicted_token": result.predicted_token,
            "activation_width": result.activation_width,
            "wire_dtype": result.wire_dtype,
            "stages": [
                {
                    "stage_index": 0,
                    "layer_start": 0,
                    "layer_end": result.split_layer_1,
                    "payload_bytes": result.stage0_payload_bytes,
                    "wire_payload_bytes": result.stage0_wire_payload_bytes,
                    "wire_sideband_bytes": result.stage0_wire_sideband_bytes,
                    "wire_sideband_i32_count": result.stage0_wire_sideband_i32_count,
                    "wire_message_bytes": result.stage0_wire_message_bytes,
                },
                {
                    "stage_index": 1,
                    "layer_start": result.split_layer_1,
                    "layer_end": result.split_layer_2,
                    "forwarded_over_binary": true,
                },
                {
                    "stage_index": 2,
                    "layer_start": result.split_layer_2,
                    "layer_end": result.layer_end,
                    "returned_predicted_token": true,
                }
            ]
        }))?
    );
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
        use_mmap: true,
        use_mmap_prefetch: true,
        use_mmap_buffer: true,
        include_embeddings: true,
        include_output: true,
        filter_tensors_on_load: false,
        glm_dsa_policy: None,
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
    stage_load_mode: String,
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
    let stage_load_mode = parse_stage_load_mode(&args.stage_load_mode)?;
    if stage_load_mode.protocol == ProtocolLoadMode::RuntimeSlice {
        validate_local_topology_plan(
            &args.model_path,
            args.layer_end,
            &[args.split_layer],
            2,
            &args.activation_wire_dtype,
        )?;
    }
    let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
    let model_identity = model_identity_for_path(&args.model_id, Some(&args.model_path))?;
    let package_activation_width =
        package_activation_width_for_binary_split(&args.model_path, &stage_load_mode)
            .context("read layer package activation width")?;
    let prestarted_stage1 = package_activation_width
        .map(|activation_width| {
            start_stage1_binary(&args, &model_identity, &stage_load_mode, activation_width)
        })
        .transpose()?;
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
        load_mode: stage_load_mode.runtime,
        projector_path: None,
        use_mmap: stage_load_mode.use_mmap,
        use_mmap_prefetch: stage_load_mode.use_mmap,
        use_mmap_buffer: stage_load_mode.use_mmap,
        include_embeddings: true,
        include_output: false,
        filter_tensors_on_load: true,
        glm_dsa_policy: None,
    };
    let stage0 = open_stage_model_for_binary_split(
        &args.model_path,
        &args.model_id,
        "local-split-binary",
        "stage-0",
        &stage_load_mode,
        &stage0_config,
    )
    .context("failed to open stage 0")?;
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
    let boundary_activation_width = boundary_activation_width(&boundary, &[0])?;
    let StartedBinaryStage {
        _child: _stage1,
        mut stream,
        activation_width,
    } = match prestarted_stage1 {
        Some(stage1) => {
            if stage1.activation_width != boundary_activation_width {
                bail!(
                    "package activation_width {} did not match stage 0 boundary activation_width {}",
                    stage1.activation_width,
                    boundary_activation_width
                );
            }
            stage1
        }
        None => start_stage1_binary(
            &args,
            &model_identity,
            &stage_load_mode,
            boundary_activation_width,
        )?,
    };
    let request_id = 1;
    let session_id = 1;
    send_generation_config(&mut stream, wire_dtype, request_id, session_id, 1)
        .context("send binary split generation config")?;
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, wire_dtype);
    state.prompt_token_count = 0;
    state.decode_step = 0;
    state.current_token = token_id;
    state.source_stage_index = 0;
    state.flags |= activation_state_flags_from_frame_flags(boundary.desc.flags);
    let wire_payload =
        encode_boundary_wire_payload(&boundary, wire_dtype, 1, activation_width, state.flags)
            .context("failed to encode boundary activation for wire")?;
    let boundary_wire_sideband_bytes = wire_payload.glm_dsa_top_k_sideband_bytes;
    let boundary_wire_sideband_i32_count = wire_payload.glm_dsa_top_k_sideband_i32_count;
    let message = StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: 0,
        token_count: 1,
        state,
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: vec![token_id],
        positions: vec![0],
        activation: wire_payload.activation,
        raw_bytes: wire_payload.raw_bytes,
    };
    let boundary_wire_message_bytes = message.estimated_wire_bytes();
    write_stage_message(&mut stream, &message, wire_dtype).context("send binary decode")?;
    let reply = recv_reply(&mut stream).context("receive binary split prediction reply")?;
    ensure_reply_kind(&reply, WireReplyKind::PredictedToken)?;
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
        boundary_wire_sideband_bytes,
        boundary_wire_sideband_i32_count,
        boundary_wire_message_bytes,
    })
}

fn run_binary_chain(args: LocalSplitChainBinaryArgs) -> Result<BinaryChainResult> {
    if args.split_layer_1 == 0
        || args.split_layer_1 >= args.split_layer_2
        || args.split_layer_2 >= args.layer_end
    {
        bail!("split_layer_1 and split_layer_2 must partition 0..layer_end in ascending order");
    }
    validate_local_topology_plan(
        &args.model_path,
        args.layer_end,
        &[args.split_layer_1, args.split_layer_2],
        3,
        &args.activation_wire_dtype,
    )?;
    let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
    let model_identity = model_identity_for_path(&args.model_id, Some(&args.model_path))?;
    let stage0_config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.split_layer_1,
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
        use_mmap: true,
        use_mmap_prefetch: true,
        use_mmap_buffer: true,
        include_embeddings: true,
        include_output: false,
        filter_tensors_on_load: true,
        glm_dsa_policy: None,
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
    let activation_width = boundary_activation_width(&boundary, &[0])?;

    let run_id = generate_run_id();
    let stage1_config_path = temp_config_path_for(&run_id, "stage-1");
    let stage2_config_path = temp_config_path_for(&run_id, "stage-2");
    let topology_path = temp_config_path_for(&run_id, "topology");
    let stage2_config = json!({
        "run_id": run_id,
        "topology_id": "local-split-chain-binary",
        "model_id": model_identity.model_id,
        "model_path": args.model_path,
        "stage_id": "stage-2",
        "stage_index": 2,
        "layer_start": args.split_layer_2,
        "layer_end": args.layer_end,
        "ctx_size": args.ctx_size,
        "n_gpu_layers": args.n_gpu_layers,
        "filter_tensors_on_load": true,
        "load_mode": "runtime-slice",
        "bind_addr": args.stage2_bind_addr,
        "upstream": {
            "stage_id": "stage-1",
            "stage_index": 1,
            "endpoint": format!("tcp://{}", args.stage1_bind_addr)
        },
        "downstream": null
    });
    let stage1_config = json!({
        "run_id": run_id,
        "topology_id": "local-split-chain-binary",
        "model_id": model_identity.model_id,
        "model_path": args.model_path,
        "stage_id": "stage-1",
        "stage_index": 1,
        "layer_start": args.split_layer_1,
        "layer_end": args.split_layer_2,
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
        "downstream": {
            "stage_id": "stage-2",
            "stage_index": 2,
            "endpoint": format!("tcp://{}", args.stage2_bind_addr)
        }
    });
    fs::write(
        &stage2_config_path,
        serde_json::to_vec_pretty(&stage2_config)?,
    )
    .with_context(|| format!("failed to write {}", stage2_config_path.display()))?;
    fs::write(
        &stage1_config_path,
        serde_json::to_vec_pretty(&stage1_config)?,
    )
    .with_context(|| format!("failed to write {}", stage1_config_path.display()))?;
    let topology = local_split_topology(
        "local-split-chain-binary",
        &model_identity.model_id,
        "runtime-slice",
        &[
            LocalSplitTopologyStage {
                stage_id: "stage-0",
                stage_index: 0,
                endpoint: "driver".to_string(),
                layer_start: 0,
                layer_end: args.split_layer_1,
            },
            LocalSplitTopologyStage {
                stage_id: "stage-1",
                stage_index: 1,
                endpoint: format!("tcp://{}", args.stage1_bind_addr),
                layer_start: args.split_layer_1,
                layer_end: args.split_layer_2,
            },
            LocalSplitTopologyStage {
                stage_id: "stage-2",
                stage_index: 2,
                endpoint: format!("tcp://{}", args.stage2_bind_addr),
                layer_start: args.split_layer_2,
                layer_end: args.layer_end,
            },
        ],
    );
    fs::write(&topology_path, serde_json::to_vec_pretty(&topology)?)
        .with_context(|| format!("failed to write {}", topology_path.display()))?;

    let mut stage2_command = Command::new(&args.stage_server_bin);
    stage2_command.args([
        "serve-binary",
        "--config",
        stage2_config_path
            .to_str()
            .context("stage 2 config path is not valid UTF-8")?,
        "--topology",
        topology_path
            .to_str()
            .context("stage topology path is not valid UTF-8")?,
        "--activation-width",
        &activation_width.to_string(),
        "--activation-wire-dtype",
        &args.activation_wire_dtype,
    ]);
    configure_child_logs(&mut stage2_command, args.child_logs);
    let _stage2 = ChildGuard::spawn(stage2_command)?;

    let mut stage1_command = Command::new(&args.stage_server_bin);
    stage1_command.args([
        "serve-binary",
        "--config",
        stage1_config_path
            .to_str()
            .context("stage 1 config path is not valid UTF-8")?,
        "--topology",
        topology_path
            .to_str()
            .context("stage topology path is not valid UTF-8")?,
        "--activation-width",
        &activation_width.to_string(),
        "--activation-wire-dtype",
        &args.activation_wire_dtype,
    ]);
    configure_child_logs(&mut stage1_command, args.child_logs);
    let _stage1 = ChildGuard::spawn(stage1_command)?;

    let mut stream = connect_ready(args.stage1_bind_addr, args.startup_timeout_secs)
        .context("stage 1 binary server did not become ready")?;
    let request_id = 2;
    let session_id = 2;
    send_generation_config(&mut stream, wire_dtype, request_id, session_id, 1)
        .context("send binary chain generation config")?;
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, wire_dtype);
    state.prompt_token_count = 0;
    state.decode_step = 0;
    state.current_token = token_id;
    state.source_stage_index = 0;
    state.flags |= activation_state_flags_from_frame_flags(boundary.desc.flags);
    let wire_payload =
        encode_boundary_wire_payload(&boundary, wire_dtype, 1, activation_width, state.flags)
            .context("failed to encode boundary activation for wire")?;
    let stage0_wire_sideband_bytes = wire_payload.glm_dsa_top_k_sideband_bytes;
    let stage0_wire_sideband_i32_count = wire_payload.glm_dsa_top_k_sideband_i32_count;
    let message = StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: 0,
        token_count: 1,
        state,
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: vec![token_id],
        positions: vec![0],
        activation: wire_payload.activation,
        raw_bytes: wire_payload.raw_bytes,
    };
    let stage0_wire_message_bytes = message.estimated_wire_bytes();
    write_stage_message(&mut stream, &message, wire_dtype).context("send binary chain decode")?;
    let reply = recv_reply(&mut stream).context("receive binary chain prediction reply")?;
    ensure_reply_kind(&reply, WireReplyKind::PredictedToken)?;
    write_stage_message(&mut stream, &StageWireMessage::stop(wire_dtype), wire_dtype)
        .context("send binary chain stop")?;

    Ok(BinaryChainResult {
        model_identity,
        token_id,
        predicted_token: reply.predicted,
        activation_width,
        wire_dtype: args.activation_wire_dtype,
        stage0_wire_payload_bytes: message.activation.len(),
        stage0_wire_sideband_bytes,
        stage0_wire_sideband_i32_count,
        stage0_wire_message_bytes,
        stage0_payload_bytes: boundary.desc.payload_bytes,
        split_layer_1: args.split_layer_1,
        split_layer_2: args.split_layer_2,
        layer_end: args.layer_end,
    })
}

struct LocalSplitTopologyStage<'a> {
    stage_id: &'a str,
    stage_index: u32,
    endpoint: String,
    layer_start: u32,
    layer_end: u32,
}

fn local_split_topology(
    topology_id: &str,
    model_id: &str,
    load_mode: &str,
    stages: &[LocalSplitTopologyStage<'_>],
) -> serde_json::Value {
    json!({
        "topology_id": topology_id,
        "model_id": model_id,
        "stages": stages.iter().map(|stage| {
            json!({
                "stage_id": stage.stage_id,
                "stage_index": stage.stage_index,
                "host": "localhost",
                "endpoint": stage.endpoint,
                "layer_start": stage.layer_start,
                "layer_end": stage.layer_end,
                "load_mode": load_mode,
            })
        }).collect::<Vec<_>>(),
    })
}

struct ParsedStageLoadMode {
    runtime: RuntimeLoadMode,
    protocol: ProtocolLoadMode,
    protocol_str: &'static str,
    use_mmap: bool,
}

fn parse_stage_load_mode(load_mode: &str) -> Result<ParsedStageLoadMode> {
    match load_mode {
        "artifact-slice" => Ok(ParsedStageLoadMode {
            runtime: RuntimeLoadMode::ArtifactSlice,
            protocol: ProtocolLoadMode::ArtifactSlice,
            protocol_str: "artifact-slice",
            use_mmap: true,
        }),
        "layer-package" => Ok(ParsedStageLoadMode {
            runtime: RuntimeLoadMode::LayerPackage,
            protocol: ProtocolLoadMode::LayerPackage,
            protocol_str: "layer-package",
            use_mmap: false,
        }),
        "runtime-slice" => Ok(ParsedStageLoadMode {
            runtime: RuntimeLoadMode::RuntimeSlice,
            protocol: ProtocolLoadMode::RuntimeSlice,
            protocol_str: "runtime-slice",
            use_mmap: true,
        }),
        other => bail!(
            "unsupported --stage-load-mode {other}; expected runtime-slice, artifact-slice, or layer-package"
        ),
    }
}

struct StartedBinaryStage {
    _child: ChildGuard,
    stream: TcpStream,
    activation_width: i32,
}

fn package_activation_width_for_binary_split(
    model_path: &std::path::Path,
    stage_load_mode: &ParsedStageLoadMode,
) -> Result<Option<i32>> {
    if stage_load_mode.protocol != ProtocolLoadMode::LayerPackage {
        return Ok(None);
    }
    let info = inspect_layer_package(&model_path.display().to_string())?;
    let width = info.activation_width.with_context(|| {
        format!(
            "layer package {} does not declare activation_width",
            model_path.display()
        )
    })?;
    let width = i32::try_from(width).context("layer package activation_width exceeds i32")?;
    Ok(Some(width))
}

fn start_stage1_binary(
    args: &BinarySplitConfig,
    model_identity: &ModelIdentity,
    stage_load_mode: &ParsedStageLoadMode,
    activation_width: i32,
) -> Result<StartedBinaryStage> {
    let run_id = generate_run_id();
    let config_path = temp_config_path_for(&run_id, "stage-1");
    let topology_path = temp_config_path_for(&run_id, "topology");
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
        "load_mode": stage_load_mode.protocol_str,
        "bind_addr": args.stage1_bind_addr,
        "upstream": {
            "stage_id": "stage-0",
            "stage_index": 0,
            "endpoint": "driver"
        },
        "downstream": null
    });
    let topology = local_split_topology(
        "local-split-binary",
        &model_identity.model_id,
        stage_load_mode.protocol_str,
        &[
            LocalSplitTopologyStage {
                stage_id: "stage-0",
                stage_index: 0,
                endpoint: "driver".to_string(),
                layer_start: 0,
                layer_end: args.split_layer,
            },
            LocalSplitTopologyStage {
                stage_id: "stage-1",
                stage_index: 1,
                endpoint: format!("tcp://{}", args.stage1_bind_addr),
                layer_start: args.split_layer,
                layer_end: args.layer_end,
            },
        ],
    );
    fs::write(&config_path, serde_json::to_vec_pretty(&config)?)
        .with_context(|| format!("failed to write {}", config_path.display()))?;
    fs::write(&topology_path, serde_json::to_vec_pretty(&topology)?)
        .with_context(|| format!("failed to write {}", topology_path.display()))?;

    let mut stage_command = Command::new(&args.stage_server_bin);
    stage_command.args([
        "serve-binary",
        "--config",
        config_path
            .to_str()
            .context("stage config path is not valid UTF-8")?,
        "--topology",
        topology_path
            .to_str()
            .context("stage topology path is not valid UTF-8")?,
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
    let mut child = ChildGuard::spawn(stage_command)?;
    let stream = connect_ready_while_child_running(
        args.stage1_bind_addr,
        args.startup_timeout_secs,
        &mut child,
    )
    .context("stage 1 binary server did not become ready")?;
    Ok(StartedBinaryStage {
        _child: child,
        stream,
        activation_width,
    })
}

fn open_stage_model_for_binary_split(
    model_path: &std::path::Path,
    model_id: &str,
    topology_id: &str,
    stage_id: &str,
    stage_load_mode: &ParsedStageLoadMode,
    config: &RuntimeConfig,
) -> Result<StageModel> {
    if stage_load_mode.protocol != ProtocolLoadMode::LayerPackage {
        return StageModel::open(model_path, config)
            .with_context(|| format!("open stage model {}", model_path.display()));
    }

    let selected = select_layer_package_parts(&PackageStageRequest {
        model_id: model_id.to_string(),
        topology_id: topology_id.to_string(),
        package_ref: model_path.display().to_string(),
        stage_id: stage_id.to_string(),
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        include_embeddings: config.include_embeddings,
        include_output: config.include_output,
    })
    .with_context(|| {
        format!(
            "select layer package parts for {stage_id} layers {}..{}",
            config.layer_start, config.layer_end
        )
    })?;

    StageModel::open_from_parts(&selected.absolute_paths, config).with_context(|| {
        let paths = selected
            .absolute_paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!("open layer package stage {stage_id} from parts [{paths}]")
    })
}

fn send_generation_config(
    stream: &mut std::net::TcpStream,
    wire_dtype: skippy_protocol::binary::WireActivationDType,
    request_id: u64,
    session_id: u64,
    prompt_token_count: usize,
) -> Result<()> {
    let message = StageWireMessage::configure_generation(
        wire_dtype,
        request_id,
        session_id,
        i32::try_from(prompt_token_count).context("prompt token count exceeds i32")?,
        None,
        None,
    );
    write_stage_message(&mut *stream, &message, wire_dtype).context("send configure-generation")?;
    let reply = recv_reply(&mut *stream).context("receive configure-generation ACK")?;
    if reply.kind != WireReplyKind::Ack {
        bail!("expected configure-generation ACK, got {:?}", reply.kind);
    }
    Ok(())
}

struct BoundaryWirePayload {
    activation: Vec<u8>,
    raw_bytes: Vec<u8>,
    glm_dsa_top_k_sideband_bytes: usize,
    glm_dsa_top_k_sideband_i32_count: usize,
}

fn boundary_activation_width(frame: &ActivationFrame, positions: &[i32]) -> Result<i32> {
    let state_flags = activation_state_flags_from_frame_flags(frame.desc.flags);
    if (state_flags & state_flags::GLM_DSA_TOP_K_SIDEBAND) == 0 {
        return activation_width(frame);
    }
    if frame.desc.dtype != RuntimeActivationDType::F32
        || frame.desc.layout != RuntimeActivationLayout::TokenMajor
    {
        bail!(
            "GLM-DSA sideband boundary width inference requires F32 token-major activations, got {:?}/{:?}",
            frame.desc.dtype,
            frame.desc.layout
        );
    }
    let sideband_i32 = glm_dsa_decode_sideband_i32_count(positions)?;
    let sideband_bytes = sideband_i32
        .checked_mul(std::mem::size_of::<i32>())
        .context("GLM-DSA sideband byte count overflow")?;
    let hidden_bytes = frame
        .payload
        .len()
        .checked_sub(sideband_bytes)
        .context("GLM-DSA sideband estimate exceeds activation payload")?;
    activation_width_from_hidden_bytes(hidden_bytes, frame.desc.token_count)
}

fn encode_boundary_wire_payload(
    frame: &ActivationFrame,
    wire_dtype: WireActivationDType,
    token_count: i32,
    activation_width: i32,
    state_flags: i32,
) -> Result<BoundaryWirePayload> {
    let hidden_bytes = activation_wire_bytes_with_state_flags(
        WireActivationDType::F32,
        token_count,
        activation_width,
        state_flags & !state_flags::GLM_DSA_TOP_K_SIDEBAND,
    )
    .context("calculate hidden activation payload bytes")?;
    if frame.payload.len() < hidden_bytes {
        bail!(
            "boundary activation payload has {} bytes, expected at least {hidden_bytes}",
            frame.payload.len()
        );
    }
    let has_glm_dsa_sideband = (state_flags & state_flags::GLM_DSA_TOP_K_SIDEBAND) != 0;
    if !has_glm_dsa_sideband && frame.payload.len() != hidden_bytes {
        bail!(
            "boundary activation payload has {} bytes, expected {hidden_bytes}",
            frame.payload.len()
        );
    }
    let sideband_bytes = if has_glm_dsa_sideband {
        frame.payload.len() - hidden_bytes
    } else {
        0
    };
    if !sideband_bytes.is_multiple_of(std::mem::size_of::<i32>()) {
        bail!("GLM-DSA top-k sideband payload is not i32-aligned");
    }
    let activation = skippy_protocol::binary::encode_f32_activation_payload_with_state_flags(
        wire_dtype,
        token_count,
        activation_width,
        &frame.payload[..hidden_bytes],
        state_flags & !state_flags::GLM_DSA_TOP_K_SIDEBAND,
    )
    .context("encode hidden activation payload")?;
    Ok(BoundaryWirePayload {
        activation,
        raw_bytes: frame.payload[hidden_bytes..].to_vec(),
        glm_dsa_top_k_sideband_bytes: sideband_bytes,
        glm_dsa_top_k_sideband_i32_count: sideband_bytes / std::mem::size_of::<i32>(),
    })
}

fn activation_width_from_hidden_bytes(hidden_bytes: usize, token_count: u32) -> Result<i32> {
    if token_count == 0 {
        bail!("activation frame token_count is zero");
    }
    let token_count = usize::try_from(token_count).context("token_count exceeds usize")?;
    if !hidden_bytes.is_multiple_of(token_count) {
        bail!(
            "hidden activation bytes {hidden_bytes} are not divisible by token_count {token_count}"
        );
    }
    let hidden_bytes_per_token = hidden_bytes / token_count;
    if !hidden_bytes_per_token.is_multiple_of(std::mem::size_of::<f32>()) {
        bail!("hidden activation bytes are not F32 aligned");
    }
    i32::try_from(hidden_bytes_per_token / std::mem::size_of::<f32>())
        .context("activation width exceeds i32")
}

fn glm_dsa_decode_sideband_i32_count(positions: &[i32]) -> Result<usize> {
    positions.iter().try_fold(0_usize, |sum, position| {
        let visible = if *position < 0 {
            0
        } else {
            usize::try_from(*position)
                .context("position exceeds usize")?
                .saturating_add(1)
        };
        sum.checked_add(visible)
            .context("GLM-DSA sideband i32 count overflow")
    })
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
        use_mmap: true,
        use_mmap_prefetch: true,
        use_mmap_buffer: true,
        include_embeddings: true,
        include_output: false,
        filter_tensors_on_load: true,
        glm_dsa_policy: None,
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
        use_mmap: true,
        use_mmap_prefetch: true,
        use_mmap_buffer: true,
        include_embeddings: false,
        include_output: true,
        filter_tensors_on_load: true,
        glm_dsa_policy: None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::binary::ACTIVATION_FLAG_GLM_DSA_TOP_K;
    use skippy_runtime::ActivationDesc;

    #[test]
    fn parses_layer_package_stage_load_mode() {
        let parsed = parse_stage_load_mode("layer-package").unwrap();

        assert!(matches!(parsed.runtime, RuntimeLoadMode::LayerPackage));
        assert_eq!(parsed.protocol, ProtocolLoadMode::LayerPackage);
        assert_eq!(parsed.protocol_str, "layer-package");
        assert!(!parsed.use_mmap);
    }

    #[test]
    fn local_split_topology_uses_selected_load_mode() {
        let topology = local_split_topology(
            "test-topology",
            "meshllm/GLM-5.2-Q2_K-MTP-Q8-layers",
            "layer-package",
            &[
                LocalSplitTopologyStage {
                    stage_id: "stage-0",
                    stage_index: 0,
                    endpoint: "driver".to_string(),
                    layer_start: 0,
                    layer_end: 31,
                },
                LocalSplitTopologyStage {
                    stage_id: "stage-1",
                    stage_index: 1,
                    endpoint: "tcp://127.0.0.1:18181".to_string(),
                    layer_start: 31,
                    layer_end: 32,
                },
            ],
        );

        let stages = topology["stages"].as_array().expect("topology stages");
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0]["load_mode"], "layer-package");
        assert_eq!(stages[1]["load_mode"], "layer-package");
    }

    #[test]
    fn boundary_width_uses_payload_width_without_glm_sideband() {
        let frame = activation_frame(vec![0_u8; 16], 0, 1);

        let width = boundary_activation_width(&frame, &[0]).unwrap();

        assert_eq!(width, 4);
    }

    #[test]
    fn boundary_width_subtracts_glm_dsa_visible_sideband() {
        let frame = activation_frame(vec![0_u8; 32], ACTIVATION_FLAG_GLM_DSA_TOP_K, 1);

        let width = boundary_activation_width(&frame, &[3]).unwrap();

        assert_eq!(width, 4);
    }

    #[test]
    fn boundary_wire_payload_splits_glm_dsa_sideband() {
        let mut payload = vec![0_u8; 16];
        for value in [0_i32, 1, 2, 3] {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let frame = activation_frame(payload, ACTIVATION_FLAG_GLM_DSA_TOP_K, 1);
        let state_flags = activation_state_flags_from_frame_flags(frame.desc.flags);

        let wire =
            encode_boundary_wire_payload(&frame, WireActivationDType::F32, 1, 4, state_flags)
                .unwrap();

        assert_eq!(wire.activation.len(), 16);
        assert_eq!(wire.raw_bytes.len(), 16);
        assert_eq!(wire.glm_dsa_top_k_sideband_bytes, 16);
        assert_eq!(wire.glm_dsa_top_k_sideband_i32_count, 4);
    }

    #[test]
    fn glm_dsa_decode_sideband_count_follows_visible_positions() {
        let count = glm_dsa_decode_sideband_i32_count(&[0, 3, 7]).unwrap();

        assert_eq!(count, 13);
    }

    fn activation_frame(payload: Vec<u8>, flags: u64, token_count: u32) -> ActivationFrame {
        ActivationFrame {
            desc: ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F32,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: 0,
                layer_start: 0,
                layer_end: 1,
                token_count,
                sequence_count: 1,
                payload_bytes: payload.len() as u64,
                flags,
            },
            payload,
        }
    }
}
