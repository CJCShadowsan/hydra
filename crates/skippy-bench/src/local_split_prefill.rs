use std::{
    fs, io,
    process::Command,
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;
use serde_json::{Value, json};
use skippy_protocol::binary::{
    StageStateHeader, StageWireMessage, WireMessageKind, WireReplyKind, recv_reply, state_flags,
    stripe_activation_payload, write_activation_stripe_chunk, write_stage_message,
};
use skippy_runtime::{RuntimeConfig, RuntimeLoadMode, StageModel};

use crate::{
    cli::LocalSplitPrefillBinaryArgs,
    direct_return::BenchDirectReturnServer,
    local_split::{
        LocalSplitTopologyStage, configure_child_logs, local_split_topology,
        send_generation_config, validate_local_topology_plan,
    },
    model_identity::model_identity_for_path,
    support::{
        ChildGuard, activation_width, connect_ready, generate_run_id, parse_wire_dtype,
        temp_config_path_for,
    },
};

#[derive(Default)]
struct PrefillTotals {
    stage0_compute: Duration,
    activation_encode: Duration,
    write: Duration,
    ack_wait: Duration,
    wire_payload_bytes: usize,
    runtime_payload_bytes: u64,
    max_wire_payload_bytes: usize,
    max_runtime_payload_bytes: u64,
    stripe_chunk_count: usize,
}

#[derive(Serialize)]
struct PrefillRunReport {
    repetition: usize,
    token_count: usize,
    chunk_count: usize,
    effective_prefill_chunk_size: usize,
    stage0_compute_ms: f64,
    activation_encode_ms: f64,
    downstream_write_ms: f64,
    downstream_ack_wait_ms: f64,
    elapsed_ms: f64,
    wire_payload_bytes: usize,
    runtime_payload_bytes: u64,
    max_chunk_wire_payload_bytes: usize,
    max_chunk_runtime_payload_bytes: u64,
    transport_mode: String,
    activation_stripes: usize,
    stripe_chunk_count: usize,
}

#[derive(Serialize)]
struct PrefillSummary {
    token_count: usize,
    repetitions: usize,
    mean_elapsed_ms: f64,
    mean_stage0_compute_ms: f64,
    mean_activation_encode_ms: f64,
    mean_downstream_write_ms: f64,
    mean_downstream_ack_wait_ms: f64,
    mean_wire_payload_bytes: f64,
    max_chunk_wire_payload_bytes: usize,
    max_chunk_runtime_payload_bytes: u64,
    transport_mode: String,
    activation_stripes: usize,
    stripe_chunk_count: usize,
}

pub fn local_split_prefill_binary(args: LocalSplitPrefillBinaryArgs) -> Result<()> {
    validate_args(&args)?;
    validate_local_topology_plan(
        &args.model_path,
        args.layer_end,
        &[args.split_layer],
        2,
        &args.activation_wire_dtype,
    )?;
    let token_counts = parse_token_counts(&args.token_counts)?;
    let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
    let model_identity = model_identity_for_path(&args.model_id, Some(&args.model_path))?;
    let stage0 = open_stage0(&args)?;
    let activation_width = determine_activation_width(&stage0, &args.prompt)?;
    let direct_returns = BenchDirectReturnServer::start("127.0.0.1:0")?;
    let topology = write_stage1_configs(&args, &model_identity.model_id, &direct_returns)?;
    let _stage1 = spawn_stage1(
        &args,
        activation_width,
        &topology.config_path,
        &topology.path,
    )?;

    let mut runs = Vec::new();
    for token_count in token_counts {
        let token_ids = tokens_for_count(&stage0, &args.prompt, token_count)?;
        for repetition in 0..args.repetitions {
            runs.push(run_prefill_case(
                &args,
                &stage0,
                &direct_returns,
                &token_ids,
                activation_width,
                repetition,
                wire_dtype,
            )?);
        }
    }

    let summaries = summarize_runs(&runs);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "mode": "local-split-prefill-binary",
            "model_identity": model_identity,
            "split": {
                "stage0_layers": [0, args.split_layer],
                "stage1_layers": [args.split_layer, args.layer_end],
            },
            "activation_width": activation_width,
            "wire_dtype": args.activation_wire_dtype,
            "prefill_chunk_size": args.prefill_chunk_size,
            "activation_stripes": args.activation_stripes,
            "activation_stripe_chunk_mib": args.activation_stripe_chunk_mib,
            "repetitions": args.repetitions,
            "results": runs,
            "summary": summaries,
        }))?
    );

    Ok(())
}

struct StageTopologyFiles {
    path: std::path::PathBuf,
    config_path: std::path::PathBuf,
}

fn validate_args(args: &LocalSplitPrefillBinaryArgs) -> Result<()> {
    if args.split_layer == 0 || args.split_layer >= args.layer_end {
        bail!("split_layer must be greater than zero and less than layer_end");
    }
    if args.prefill_chunk_size == 0 {
        bail!("prefill_chunk_size must be greater than zero");
    }
    if args.repetitions == 0 {
        bail!("repetitions must be greater than zero");
    }
    if args.activation_stripes == 0 {
        bail!("activation_stripes must be greater than zero");
    }
    if args.activation_stripe_chunk_mib == 0 {
        bail!("activation_stripe_chunk_mib must be greater than zero");
    }
    Ok(())
}

fn open_stage0(args: &LocalSplitPrefillBinaryArgs) -> Result<StageModel> {
    StageModel::open(
        &args.model_path,
        &RuntimeConfig {
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
        },
    )
    .context("failed to open stage 0")
}

fn determine_activation_width(stage0: &StageModel, prompt: &str) -> Result<i32> {
    let tokens = stage0
        .tokenize(prompt, true)
        .context("failed to tokenize prompt")?;
    let token_id = *tokens.first().context("prompt produced no tokens")?;
    let mut session = stage0
        .create_session()
        .context("failed to create stage 0 session")?;
    let (_predicted, boundary) = session
        .decode_step_frame(token_id, None, 0)
        .context("stage 0 failed to produce activation frame")?;
    activation_width(&boundary)
}

fn write_stage1_configs(
    args: &LocalSplitPrefillBinaryArgs,
    model_id: &str,
    direct_returns: &BenchDirectReturnServer,
) -> Result<StageTopologyFiles> {
    let run_id = generate_run_id();
    let config_path = temp_config_path_for(&run_id, "stage-1");
    let topology_path = temp_config_path_for(&run_id, "topology");
    let config = json!({
        "run_id": run_id,
        "topology_id": "local-split-prefill-binary",
        "model_id": model_id,
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
    let topology = local_split_topology(
        "local-split-prefill-binary",
        model_id,
        &[
            LocalSplitTopologyStage {
                stage_id: "stage-0",
                stage_index: 0,
                endpoint: format!("tcp://{}", direct_returns.endpoint()),
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
    Ok(StageTopologyFiles {
        path: topology_path,
        config_path,
    })
}

fn spawn_stage1(
    args: &LocalSplitPrefillBinaryArgs,
    activation_width: i32,
    config_path: &std::path::Path,
    topology_path: &std::path::Path,
) -> Result<ChildGuard> {
    let mut command = Command::new(&args.stage_server_bin);
    command.args([
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
    configure_child_logs(&mut command, args.child_logs);
    ChildGuard::spawn(command)
}

#[allow(clippy::too_many_arguments)]
fn run_prefill_case(
    args: &LocalSplitPrefillBinaryArgs,
    stage0: &StageModel,
    direct_returns: &BenchDirectReturnServer,
    token_ids: &[i32],
    activation_width: i32,
    repetition: usize,
    wire_dtype: skippy_protocol::binary::WireActivationDType,
) -> Result<PrefillRunReport> {
    let mut stream = connect_ready(args.stage1_bind_addr, args.startup_timeout_secs)
        .context("stage 1 binary server did not become ready")?;
    let token_count_u64 =
        u64::try_from(token_ids.len()).context("token count exceeds u64 range")?;
    let request_id = token_count_u64
        .checked_mul(1_000)
        .and_then(|base| base.checked_add(repetition as u64))
        .context("request id overflow")?;
    let session_id = request_id + 500_000;
    let _direct_return = direct_returns.register(request_id, session_id)?;
    send_generation_config(
        &mut stream,
        wire_dtype,
        request_id,
        session_id,
        token_ids.len(),
    )
    .context("send prefill generation config")?;
    let mut stripe_streams = if args.activation_stripes > 1 {
        open_activation_stripe_streams(
            args.stage1_bind_addr,
            args.startup_timeout_secs,
            wire_dtype,
            args.activation_stripes,
        )?
    } else {
        Vec::new()
    };

    let started = Instant::now();
    let mut totals = PrefillTotals::default();
    let mut session0 = stage0
        .create_session()
        .context("failed to create stage 0 session")?;
    for (chunk_index, chunk) in token_ids.chunks(args.prefill_chunk_size).enumerate() {
        let pos_start = chunk_index
            .checked_mul(args.prefill_chunk_size)
            .context("prefill chunk position overflow")?;
        let positions = positions_for_chunk(pos_start, chunk.len())?;
        let compute_started = Instant::now();
        let boundary = session0
            .prefill_chunk_frame_with_positions(chunk, &positions, None, 0)
            .context("stage 0 failed to produce prefill activation frame")?;
        totals.stage0_compute += compute_started.elapsed();
        let encode_started = Instant::now();
        let message = prefill_boundary_message(PrefillBoundaryMessageArgs {
            request_id,
            session_id,
            pos_start,
            prefill_token_count: token_ids.len(),
            tokens: chunk,
            positions: &positions,
            activation_width,
            wire_dtype,
            boundary: &boundary,
        })?;
        totals.activation_encode += encode_started.elapsed();
        totals.wire_payload_bytes += message.activation.len();
        totals.runtime_payload_bytes = totals
            .runtime_payload_bytes
            .saturating_add(boundary.desc.payload_bytes);
        totals.max_wire_payload_bytes = totals.max_wire_payload_bytes.max(message.activation.len());
        totals.max_runtime_payload_bytes = totals
            .max_runtime_payload_bytes
            .max(boundary.desc.payload_bytes);

        let frame_id = u64::try_from(chunk_index + 1).context("stripe frame id overflow")?;
        let write_started = Instant::now();
        if stripe_streams.is_empty() {
            write_stage_message(&mut stream, &message, wire_dtype)
                .context("send real split prefill activation")?;
        } else {
            totals.stripe_chunk_count += write_striped_stage_message(
                &mut stream,
                &mut stripe_streams,
                &message,
                wire_dtype,
                frame_id,
                args.activation_stripe_chunk_mib * 1024 * 1024,
            )
            .context("send real split striped prefill activation")?;
        }
        totals.write += write_started.elapsed();
        let ack_started = Instant::now();
        let reply = recv_reply(&mut stream).context("receive real split prefill ACK")?;
        totals.ack_wait += ack_started.elapsed();
        if reply.kind != WireReplyKind::Ack {
            bail!("expected prefill ACK, got {:?}", reply.kind);
        }
    }
    write_stage_message(
        &mut stream,
        &StageWireMessage::stop_with_identity(wire_dtype, request_id, session_id),
        wire_dtype,
    )
    .context("send real split prefill stop")?;

    Ok(PrefillRunReport {
        repetition,
        token_count: token_ids.len(),
        chunk_count: token_ids.len().div_ceil(args.prefill_chunk_size),
        effective_prefill_chunk_size: args.prefill_chunk_size,
        stage0_compute_ms: duration_ms(totals.stage0_compute),
        activation_encode_ms: duration_ms(totals.activation_encode),
        downstream_write_ms: duration_ms(totals.write),
        downstream_ack_wait_ms: duration_ms(totals.ack_wait),
        elapsed_ms: duration_ms(started.elapsed()),
        wire_payload_bytes: totals.wire_payload_bytes,
        runtime_payload_bytes: totals.runtime_payload_bytes,
        max_chunk_wire_payload_bytes: totals.max_wire_payload_bytes,
        max_chunk_runtime_payload_bytes: totals.max_runtime_payload_bytes,
        transport_mode: if stripe_streams.is_empty() {
            "inline".to_string()
        } else {
            "striped".to_string()
        },
        activation_stripes: args.activation_stripes,
        stripe_chunk_count: totals.stripe_chunk_count,
    })
}

fn open_activation_stripe_streams(
    addr: std::net::SocketAddr,
    timeout_secs: u64,
    wire_dtype: skippy_protocol::binary::WireActivationDType,
    count: usize,
) -> Result<Vec<std::net::TcpStream>> {
    let mut streams = Vec::with_capacity(count);
    for _ in 0..count {
        let mut stream = connect_ready(addr, timeout_secs).context("connect activation stripe")?;
        let open = StageWireMessage {
            kind: WireMessageKind::ActivationStripeOpen,
            pos_start: 0,
            token_count: 0,
            state: StageStateHeader::new(WireMessageKind::ActivationStripeOpen, wire_dtype),
            request_id: 0,
            session_id: 0,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        write_stage_message(&mut stream, &open, wire_dtype).context("open activation stripe")?;
        streams.push(stream);
    }
    Ok(streams)
}

fn write_striped_stage_message(
    control_stream: &mut std::net::TcpStream,
    stripe_streams: &mut [std::net::TcpStream],
    message: &StageWireMessage,
    wire_dtype: skippy_protocol::binary::WireActivationDType,
    frame_id: u64,
    max_chunk_bytes: usize,
) -> Result<usize> {
    let chunks = stripe_activation_payload(
        message.request_id,
        message.session_id,
        frame_id,
        &message.activation,
        max_chunk_bytes,
    )?;
    let mut control = message.clone();
    control.activation.clear();
    control.state.flags |= state_flags::STRIPED_ACTIVATION;
    control.state.checkpoint_generation =
        i32::try_from(frame_id).context("stripe frame id exceeds i32")?;
    write_stage_message(control_stream, &control, wire_dtype).context("send stripe control")?;

    let mut groups = vec![Vec::new(); stripe_streams.len()];
    for (index, chunk) in chunks.iter().cloned().enumerate() {
        let stream_index = index % stripe_streams.len();
        groups[stream_index].push(chunk);
    }
    thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::with_capacity(stripe_streams.len());
        for (stream, group) in stripe_streams.iter_mut().zip(groups) {
            handles.push(scope.spawn(move || -> io::Result<()> {
                for chunk in &group {
                    write_activation_stripe_chunk(&mut *stream, chunk)?;
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow!("activation stripe writer panicked"))?
                .context("write activation stripe chunks")?;
        }
        Ok(())
    })?;
    Ok(chunks.len())
}

struct PrefillBoundaryMessageArgs<'a> {
    request_id: u64,
    session_id: u64,
    pos_start: usize,
    prefill_token_count: usize,
    tokens: &'a [i32],
    positions: &'a [i32],
    activation_width: i32,
    wire_dtype: skippy_protocol::binary::WireActivationDType,
    boundary: &'a skippy_runtime::ActivationFrame,
}

fn prefill_boundary_message(args: PrefillBoundaryMessageArgs<'_>) -> Result<StageWireMessage> {
    let token_count =
        i32::try_from(args.tokens.len()).context("prefill token count exceeds i32")?;
    let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd, args.wire_dtype);
    state.prompt_token_count =
        i32::try_from(args.prefill_token_count).context("prompt token count exceeds i32")?;
    state.current_token = *args.tokens.last().context("prefill chunk is empty")?;
    state.source_stage_index = 0;
    state.flags |=
        skippy_protocol::binary::activation_state_flags_from_frame_flags(args.boundary.desc.flags);
    let activation = skippy_protocol::binary::encode_f32_activation_payload_with_state_flags(
        args.wire_dtype,
        token_count,
        args.activation_width,
        &args.boundary.payload,
        state.flags,
    )
    .context("failed to encode prefill activation for wire")?;
    Ok(StageWireMessage {
        kind: WireMessageKind::PrefillEmbd,
        pos_start: i32::try_from(args.pos_start).context("prefill position exceeds i32")?,
        token_count,
        state,
        request_id: args.request_id,
        session_id: args.session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: args.tokens.to_vec(),
        positions: args.positions.to_vec(),
        activation,
        raw_bytes: Vec::new(),
    })
}

fn tokens_for_count(stage0: &StageModel, prompt: &str, token_count: usize) -> Result<Vec<i32>> {
    if token_count == 0 {
        bail!("token counts must be greater than zero");
    }
    let seed = stage0
        .tokenize(prompt, true)
        .context("failed to tokenize prompt")?;
    if seed.is_empty() {
        bail!("prompt produced no tokens");
    }
    let mut tokens = Vec::with_capacity(token_count);
    while tokens.len() < token_count {
        tokens.extend_from_slice(&seed);
    }
    tokens.truncate(token_count);
    Ok(tokens)
}

fn positions_for_chunk(pos_start: usize, len: usize) -> Result<Vec<i32>> {
    let pos_start = i32::try_from(pos_start).context("prefill position exceeds i32")?;
    let len = i32::try_from(len).context("prefill chunk length exceeds i32")?;
    Ok((pos_start..pos_start + len).collect())
}

fn parse_token_counts(raw: &str) -> Result<Vec<usize>> {
    let mut counts = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let value = part
            .parse::<usize>()
            .with_context(|| format!("invalid token count '{part}'"))?;
        if value == 0 {
            bail!("token counts must be greater than zero");
        }
        counts.push(value);
    }
    if counts.is_empty() {
        bail!("at least one token count is required");
    }
    counts.sort_unstable();
    counts.dedup();
    Ok(counts)
}

fn summarize_runs(runs: &[PrefillRunReport]) -> Vec<Value> {
    let mut token_counts = runs.iter().map(|run| run.token_count).collect::<Vec<_>>();
    token_counts.sort_unstable();
    token_counts.dedup();
    token_counts
        .into_iter()
        .filter_map(|token_count| summarize_token_count(runs, token_count))
        .map(|summary| json!(summary))
        .collect()
}

fn summarize_token_count(runs: &[PrefillRunReport], token_count: usize) -> Option<PrefillSummary> {
    let matches = runs
        .iter()
        .filter(|run| run.token_count == token_count)
        .collect::<Vec<_>>();
    let repetitions = matches.len();
    if repetitions == 0 {
        return None;
    }
    Some(PrefillSummary {
        token_count,
        repetitions,
        mean_elapsed_ms: mean(&matches, |run| run.elapsed_ms),
        mean_stage0_compute_ms: mean(&matches, |run| run.stage0_compute_ms),
        mean_activation_encode_ms: mean(&matches, |run| run.activation_encode_ms),
        mean_downstream_write_ms: mean(&matches, |run| run.downstream_write_ms),
        mean_downstream_ack_wait_ms: mean(&matches, |run| run.downstream_ack_wait_ms),
        mean_wire_payload_bytes: mean(&matches, |run| run.wire_payload_bytes as f64),
        max_chunk_wire_payload_bytes: matches
            .iter()
            .map(|run| run.max_chunk_wire_payload_bytes)
            .max()
            .unwrap_or_default(),
        max_chunk_runtime_payload_bytes: matches
            .iter()
            .map(|run| run.max_chunk_runtime_payload_bytes)
            .max()
            .unwrap_or_default(),
        transport_mode: matches
            .first()
            .map(|run| run.transport_mode.clone())
            .unwrap_or_default(),
        activation_stripes: matches
            .first()
            .map(|run| run.activation_stripes)
            .unwrap_or_default(),
        stripe_chunk_count: matches.iter().map(|run| run.stripe_chunk_count).sum(),
    })
}

fn mean(runs: &[&PrefillRunReport], value: impl Fn(&PrefillRunReport) -> f64) -> f64 {
    runs.iter().map(|run| value(run)).sum::<f64>() / runs.len() as f64
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
