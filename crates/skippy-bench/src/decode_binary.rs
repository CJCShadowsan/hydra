use std::{
    fs,
    io::{Read, Write},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use skippy_protocol::binary::{
    StageReplyStats, StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind,
    WireReplyKind, recv_reply, write_stage_message,
};

use crate::{
    cli::DecodeBinaryArgs,
    support::{connect_ready, parse_wire_dtype},
};

#[derive(Serialize)]
struct DecodeBinaryReport {
    first_stage_addr: String,
    activation_wire_dtype: String,
    prompt_token_count: usize,
    prefill_token_count: usize,
    prefill_chunk_size: usize,
    max_new_tokens: usize,
    request_id: u64,
    session_id: u64,
    predicted_tokens: Vec<i32>,
    elapsed_ms: f64,
    configure_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    ttft_ms: f64,
    stop_ms: f64,
    stats: DecodeStatsReport,
}

#[derive(Default, Serialize)]
struct DecodeStatsReport {
    prefill_edge_observation_count: i64,
    prefill_edge_activation_bytes_max: i64,
    prefill_edge_write_us_max: i64,
    prefill_edge_wait_us_max: i64,
    prefill_edge_total_us_max: i64,
    prefill_edge_stage_index: i64,
}

#[derive(Clone, Copy)]
struct WireProbeContext {
    wire_dtype: WireActivationDType,
    request_id: u64,
    session_id: u64,
}

struct PrefillChunk<'a> {
    prompt_token_count: usize,
    pos_start: usize,
    tokens: &'a [i32],
}

pub fn decode_binary(args: DecodeBinaryArgs) -> Result<()> {
    let started = Instant::now();
    let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
    let prompt_tokens = parse_token_ids(&args.prompt_token_ids, "--prompt-token-ids")?;
    if prompt_tokens.len() < 2 {
        bail!("--prompt-token-ids must contain at least two tokens");
    }
    if args.prefill_chunk_size == 0 {
        bail!("--prefill-chunk-size must be greater than zero");
    }
    if args.max_new_tokens == 0 {
        bail!("--max-new-tokens must be greater than zero");
    }

    let request_id = stable_probe_id(b"decode-binary-request");
    let session_id = stable_probe_id(b"decode-binary-session");
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
    let (predicted_tokens, decode_ms, ttft_ms, stats) =
        send_decode(&mut stream, context, &prompt_tokens, args.max_new_tokens)?;
    let stop_ms = send_stop(&mut stream, context)?;

    let report = DecodeBinaryReport {
        first_stage_addr: args.first_stage_addr.to_string(),
        activation_wire_dtype: args.activation_wire_dtype,
        prompt_token_count: prompt_tokens.len(),
        prefill_token_count: prompt_tokens.len() - 1,
        prefill_chunk_size: args.prefill_chunk_size,
        max_new_tokens: args.max_new_tokens,
        request_id,
        session_id,
        predicted_tokens,
        elapsed_ms: elapsed_ms(started),
        configure_ms,
        prefill_ms,
        decode_ms,
        ttft_ms,
        stop_ms,
        stats: DecodeStatsReport::from_stats(stats),
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
        let pos_start = chunk_index
            .checked_mul(chunk_size)
            .context("prefill chunk position overflow")?;
        send_prefill_chunk(
            stream,
            context,
            PrefillChunk {
                prompt_token_count,
                pos_start,
                tokens: chunk,
            },
        )
        .with_context(|| format!("prefill chunk {chunk_index}"))?;
    }
    Ok(elapsed_ms(started))
}

fn send_prefill_chunk<T: Read + Write>(
    stream: &mut T,
    context: WireProbeContext,
    chunk: PrefillChunk<'_>,
) -> Result<()> {
    let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd, context.wire_dtype);
    state.seq_id = 0;
    state.prompt_token_count =
        i32::try_from(chunk.prompt_token_count).context("prompt token count exceeds i32")?;
    state.current_token = *chunk.tokens.last().context("prefill chunk is empty")?;
    state.source_stage_index = -1;
    let pos_start = i32::try_from(chunk.pos_start).context("prefill position exceeds i32")?;
    let token_count =
        i32::try_from(chunk.tokens.len()).context("prefill token count exceeds i32")?;
    let positions: Vec<i32> = (pos_start..pos_start + token_count).collect();
    let message = StageWireMessage {
        kind: WireMessageKind::PrefillEmbd,
        pos_start,
        token_count,
        state,
        request_id: context.request_id,
        session_id: context.session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: chunk.tokens.to_vec(),
        positions,
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *stream, &message, context.wire_dtype).context("send PrefillEmbd")?;
    expect_reply(stream, WireReplyKind::Ack).context("receive PrefillEmbd ACK")?;
    Ok(())
}

fn send_decode<T: Read + Write>(
    stream: &mut T,
    context: WireProbeContext,
    prompt_tokens: &[i32],
    max_new_tokens: usize,
) -> Result<(Vec<i32>, f64, f64, StageReplyStats)> {
    let started = Instant::now();
    let mut predicted_tokens = Vec::with_capacity(max_new_tokens);
    let mut current = *prompt_tokens.last().expect("checked non-empty");
    let mut merged_stats = StageReplyStats::default();
    let mut ttft_ms = 0.0;
    let prefill_token_count = prompt_tokens.len() - 1;

    for decode_step in 0..max_new_tokens {
        let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, context.wire_dtype);
        state.seq_id = 0;
        state.prompt_token_count =
            i32::try_from(prompt_tokens.len()).context("prompt token count exceeds i32")?;
        state.decode_step = i32::try_from(decode_step).context("decode step exceeds i32")?;
        state.current_token = current;
        state.source_stage_index = -1;
        let decode_pos = i32::try_from(prefill_token_count + decode_step)
            .context("decode position exceeds i32")?;
        let message = StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: decode_pos,
            token_count: 1,
            state,
            request_id: context.request_id,
            session_id: context.session_id,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![current],
            positions: vec![decode_pos],
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        write_stage_message(&mut *stream, &message, context.wire_dtype)
            .with_context(|| format!("send DecodeEmbd step {decode_step}"))?;
        let reply = expect_reply(stream, WireReplyKind::PredictedToken)
            .with_context(|| format!("receive DecodeEmbd step {decode_step} PredictedToken"))?;
        if decode_step == 0 {
            ttft_ms = elapsed_ms(started);
        }
        merged_stats.merge(reply.stats);
        current = reply.predicted;
        predicted_tokens.push(reply.predicted);
    }

    Ok((predicted_tokens, elapsed_ms(started), ttft_ms, merged_stats))
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

impl DecodeStatsReport {
    fn from_stats(stats: StageReplyStats) -> Self {
        Self {
            prefill_edge_observation_count: stats.prefill_edge_observation_count,
            prefill_edge_activation_bytes_max: stats.prefill_edge_activation_bytes_max,
            prefill_edge_write_us_max: stats.prefill_edge_write_us_max,
            prefill_edge_wait_us_max: stats.prefill_edge_wait_us_max,
            prefill_edge_total_us_max: stats.prefill_edge_total_us_max,
            prefill_edge_stage_index: stats.prefill_edge_stage_index,
        }
    }
}
