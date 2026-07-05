use std::{
    fs,
    io::{Read, Write},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use skippy_protocol::binary::{
    StageReplyStats, StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind,
    WireReplyKind, recv_reply, state_flags, write_stage_message,
};

use crate::{
    cli::VerifySpanBinaryArgs,
    support::{connect_ready, parse_wire_dtype},
};

#[derive(Serialize)]
struct VerifySpanBinaryReport {
    first_stage_addr: String,
    activation_wire_dtype: String,
    prompt_token_count: usize,
    prefill_token_count: usize,
    prefill_chunk_size: usize,
    verify_token_count: usize,
    checkpoint: bool,
    request_id: u64,
    session_id: u64,
    predicted_tokens: Vec<i32>,
    elapsed_ms: f64,
    configure_ms: f64,
    prefill_ms: f64,
    verify_ms: f64,
    stop_ms: f64,
    stats: VerifySpanStatsReport,
}

#[derive(Serialize)]
struct VerifySpanStatsReport {
    verify_span_request_count: i64,
    verify_span_token_count: i64,
    verify_span_max_tokens: i64,
    verify_span_stage_count: i64,
    verify_span_compute_us: i64,
    verify_span_forward_write_us: i64,
    verify_span_downstream_wait_us: i64,
    verify_span_total_us: i64,
    verify_span_checkpointed_requests: i64,
    verify_span_skip_checkpoint_requests: i64,
    checkpoint_total_us: i64,
}

#[derive(Clone, Copy)]
struct WireProbeContext {
    wire_dtype: WireActivationDType,
    request_id: u64,
    session_id: u64,
}

struct VerifySpanRequest<'a> {
    prompt_token_count: usize,
    pos_start: usize,
    tokens: &'a [i32],
    checkpoint: bool,
}

pub fn verify_span_binary(args: VerifySpanBinaryArgs) -> Result<()> {
    let started = Instant::now();
    let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
    let prompt_tokens = parse_token_ids(&args.prompt_token_ids, "--prompt-token-ids")?;
    if prompt_tokens.len() < 2 {
        bail!("--prompt-token-ids must contain at least two tokens");
    }
    if args.prefill_chunk_size == 0 {
        bail!("--prefill-chunk-size must be greater than zero");
    }
    let current = *prompt_tokens.last().expect("checked non-empty");
    let verify_tokens = match args.verify_token_ids.as_deref() {
        Some(source) => parse_token_ids(source, "--verify-token-ids")?,
        None => vec![current, current.saturating_add(1)],
    };
    if verify_tokens.is_empty() {
        bail!("VerifySpan requires at least one token");
    }

    let request_id = stable_probe_id(b"verify-span-binary-request");
    let session_id = stable_probe_id(b"verify-span-binary-session");
    let mut stream = connect_ready(args.first_stage_addr, args.startup_timeout_secs)
        .with_context(|| format!("connect first stage {}", args.first_stage_addr))?;
    let timeout = Duration::from_secs(args.io_timeout_secs.max(1));
    stream.set_read_timeout(Some(timeout)).ok();
    stream.set_write_timeout(Some(timeout)).ok();

    let context = WireProbeContext {
        wire_dtype,
        request_id,
        session_id,
    };
    let configure_ms = send_configure(&mut stream, context, prompt_tokens.len())?;
    let prefill_ms = send_prefill(
        &mut stream,
        context,
        &prompt_tokens[..prompt_tokens.len() - 1],
        prompt_tokens.len(),
        args.prefill_chunk_size,
    )?;
    let (predicted_tokens, verify_ms, stats) = send_verify_span(
        &mut stream,
        context,
        VerifySpanRequest {
            prompt_token_count: prompt_tokens.len(),
            pos_start: prompt_tokens.len() - 1,
            tokens: &verify_tokens,
            checkpoint: args.checkpoint,
        },
    )?;
    let stop_ms = send_stop(&mut stream, context)?;

    let report = VerifySpanBinaryReport {
        first_stage_addr: args.first_stage_addr.to_string(),
        activation_wire_dtype: args.activation_wire_dtype,
        prompt_token_count: prompt_tokens.len(),
        prefill_token_count: prompt_tokens.len() - 1,
        prefill_chunk_size: args.prefill_chunk_size,
        verify_token_count: verify_tokens.len(),
        checkpoint: args.checkpoint,
        request_id,
        session_id,
        predicted_tokens,
        elapsed_ms: elapsed_ms(started),
        configure_ms,
        prefill_ms,
        verify_ms,
        stop_ms,
        stats: VerifySpanStatsReport::from_stats(stats),
    };
    let json = serde_json::to_string_pretty(&report)?;
    match args.output {
        Some(path) => fs::write(&path, format!("{json}\n"))
            .with_context(|| format!("write {}", path.display()))?,
        None => println!("{json}"),
    }
    Ok(())
}

fn send_configure<T: Read + Write>(
    stream: &mut T,
    context: WireProbeContext,
    prompt_token_count: usize,
) -> Result<f64> {
    let started = Instant::now();
    let message = StageWireMessage::configure_generation(
        context.wire_dtype,
        context.request_id,
        context.session_id,
        i32::try_from(prompt_token_count).context("prompt token count exceeds i32")?,
        None,
        None,
    );
    write_stage_message(&mut *stream, &message, context.wire_dtype)
        .context("send ConfigureGeneration")?;
    expect_reply(stream, WireReplyKind::Ack).context("receive ConfigureGeneration ACK")?;
    Ok(elapsed_ms(started))
}

fn send_prefill<T: Read + Write>(
    stream: &mut T,
    context: WireProbeContext,
    prefill_tokens: &[i32],
    prompt_token_count: usize,
    chunk_size: usize,
) -> Result<f64> {
    let started = Instant::now();
    for (chunk_index, chunk) in prefill_tokens.chunks(chunk_size).enumerate() {
        let pos_start = chunk_index * chunk_size;
        let final_chunk = pos_start + chunk.len() >= prefill_tokens.len();
        let kind = if final_chunk {
            WireMessageKind::PrefillFinalEmbd
        } else {
            WireMessageKind::PrefillEmbd
        };
        let mut state = StageStateHeader::new(kind, context.wire_dtype);
        state.seq_id = 0;
        state.prompt_token_count =
            i32::try_from(prompt_token_count).context("prompt token count exceeds i32")?;
        state.current_token = *chunk.last().unwrap_or(&0);
        state.source_stage_index = -1;
        let message = StageWireMessage {
            kind,
            pos_start: i32::try_from(pos_start).context("prefill position exceeds i32")?,
            token_count: i32::try_from(chunk.len()).context("prefill chunk exceeds i32")?,
            state,
            request_id: context.request_id,
            session_id: context.session_id,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: chunk.to_vec(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        write_stage_message(&mut *stream, &message, context.wire_dtype)
            .with_context(|| format!("send prefill chunk {chunk_index}"))?;
        let expected = if final_chunk {
            WireReplyKind::PredictedToken
        } else {
            WireReplyKind::Ack
        };
        expect_reply(stream, expected)
            .with_context(|| format!("receive prefill chunk {chunk_index} reply"))?;
    }
    Ok(elapsed_ms(started))
}

fn send_verify_span<T: Read + Write>(
    stream: &mut T,
    context: WireProbeContext,
    request: VerifySpanRequest<'_>,
) -> Result<(Vec<i32>, f64, StageReplyStats)> {
    let started = Instant::now();
    let mut state = StageStateHeader::new(WireMessageKind::VerifySpan, context.wire_dtype);
    state.seq_id = 0;
    state.prompt_token_count =
        i32::try_from(request.prompt_token_count).context("prompt token count exceeds i32")?;
    state.decode_step = 0;
    state.current_token = request.tokens[0];
    state.source_stage_index = -1;
    if !request.checkpoint {
        state.flags |= state_flags::SKIP_VERIFY_CHECKPOINT;
    }
    let message = StageWireMessage {
        kind: WireMessageKind::VerifySpan,
        pos_start: i32::try_from(request.pos_start).context("verify position exceeds i32")?,
        token_count: i32::try_from(request.tokens.len())
            .context("verify token count exceeds i32")?,
        state,
        request_id: context.request_id,
        session_id: context.session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: request.tokens.to_vec(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *stream, &message, context.wire_dtype).context("send VerifySpan")?;
    let reply = expect_reply(stream, WireReplyKind::PredictedTokens)
        .context("receive VerifySpan predicted tokens")?;
    Ok((reply.predicted_tokens, elapsed_ms(started), reply.stats))
}

fn send_stop<T: Write>(stream: &mut T, context: WireProbeContext) -> Result<f64> {
    let started = Instant::now();
    let message = StageWireMessage::stop_with_identity(
        context.wire_dtype,
        context.request_id,
        context.session_id,
    );
    write_stage_message(&mut *stream, &message, context.wire_dtype).context("send Stop")?;
    Ok(elapsed_ms(started))
}

fn expect_reply(
    stream: &mut (impl Read + ?Sized),
    expected: WireReplyKind,
) -> Result<skippy_protocol::binary::StageReply> {
    let reply = recv_reply(stream)?;
    if reply.kind != expected {
        bail!("expected {expected:?}, got {:?}", reply.kind);
    }
    Ok(reply)
}

fn parse_token_ids(source: &str, arg_name: &str) -> Result<Vec<i32>> {
    source
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<i32>()
                .with_context(|| format!("parse {arg_name} token id {part:?}"))
        })
        .collect()
}

fn stable_probe_id(label: &[u8]) -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .to_le_bytes();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in label.iter().chain(now.iter()) {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

impl VerifySpanStatsReport {
    fn from_stats(stats: StageReplyStats) -> Self {
        Self {
            verify_span_request_count: stats.verify_span_request_count,
            verify_span_token_count: stats.verify_span_token_count,
            verify_span_max_tokens: stats.verify_span_max_tokens,
            verify_span_stage_count: stats.verify_span_stage_count,
            verify_span_compute_us: stats.verify_span_compute_us,
            verify_span_forward_write_us: stats.verify_span_forward_write_us,
            verify_span_downstream_wait_us: stats.verify_span_downstream_wait_us,
            verify_span_total_us: stats.verify_span_total_us,
            verify_span_checkpointed_requests: stats.verify_span_checkpointed_requests,
            verify_span_skip_checkpoint_requests: stats.verify_span_skip_checkpoint_requests,
            checkpoint_total_us: stats.checkpoint_total_us,
        }
    }
}
