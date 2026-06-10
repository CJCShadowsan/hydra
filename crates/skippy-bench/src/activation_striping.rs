use std::time::Instant;

use anyhow::{Context, Result, bail};
use model_artifact::gguf::scan_gguf_compact_meta;
use serde::Serialize;
use skippy_protocol::binary::{
    StageActivationStripeReassembler, StageStateHeader, StageWireMessage, WireActivationDType,
    WireMessageKind, read_activation_stripe_chunk, read_stage_message, state_flags,
    stripe_activation_payload, write_activation_stripe_chunk, write_stage_message,
};

use crate::cli::ActivationStripingArgs;

#[derive(Debug, Clone)]
struct ActivationShape {
    name: String,
    tokens: usize,
    hidden_size: usize,
    bytes_per_element: usize,
}

impl ActivationShape {
    fn payload_bytes(&self) -> Result<usize> {
        self.tokens
            .checked_mul(self.hidden_size)
            .and_then(|value| value.checked_mul(self.bytes_per_element))
            .context("activation payload byte count overflow")
    }
}

#[derive(Debug, Clone, Serialize)]
struct BenchRow {
    shape: String,
    tokens: usize,
    hidden_size: usize,
    bytes_per_element: usize,
    payload_bytes: usize,
    stripes: usize,
    chunk_bytes: usize,
    chunks: usize,
    synthetic_per_stream_mbps: f64,
    wall_ms: f64,
    transfer_floor_ms: f64,
    effective_mbps: f64,
    speedup_vs_single_stream: f64,
}

pub fn activation_striping(args: ActivationStripingArgs) -> Result<()> {
    let shapes = activation_shapes(&args)?;
    let stripe_counts = parse_usize_list(&args.stripes, "stripes")?;
    if args.per_stream_mbps <= 0.0 {
        bail!("--per-stream-mbps must be greater than zero");
    }
    if args.chunk_mib == 0 {
        bail!("--chunk-mib must be greater than zero");
    }
    if args.repetitions == 0 {
        bail!("--repetitions must be greater than zero");
    }
    let chunk_bytes = args
        .chunk_mib
        .checked_mul(1024 * 1024)
        .context("--chunk-mib overflow")?;

    let mut rows = Vec::new();
    for shape in shapes {
        let payload = deterministic_payload(shape.payload_bytes()?);
        let baseline_floor_ms = synthetic_transfer_ms(payload.len(), 1, args.per_stream_mbps);
        for stripes in &stripe_counts {
            let row = bench_shape(
                &shape,
                &payload,
                *stripes,
                chunk_bytes,
                args.per_stream_mbps,
                baseline_floor_ms,
                args.repetitions,
            )?;
            rows.push(row);
        }
    }

    print_markdown_table(&rows);
    println!();
    println!("```json");
    println!("{}", serde_json::to_string_pretty(&rows)?);
    println!("```");
    Ok(())
}

fn activation_shapes(args: &ActivationStripingArgs) -> Result<Vec<ActivationShape>> {
    let mut shapes = parse_shapes(&args.activation_shapes)?;
    if let Some(model_path) = &args.model_path {
        shapes.extend(model_activation_shapes(args, model_path)?);
    }
    if shapes.is_empty() {
        bail!("no activation shapes were provided");
    }
    Ok(shapes)
}

fn model_activation_shapes(
    args: &ActivationStripingArgs,
    model_path: &std::path::Path,
) -> Result<Vec<ActivationShape>> {
    if args.model_bytes_per_element == 0 {
        bail!("--model-bytes-per-element must be greater than zero");
    }
    let meta = scan_gguf_compact_meta(model_path)
        .with_context(|| format!("failed to read GGUF metadata from {}", model_path.display()))?;
    let hidden_size =
        usize::try_from(meta.embedding_size).context("model embedding_length exceeds usize")?;
    if hidden_size == 0 {
        bail!(
            "GGUF metadata for {} does not contain embedding_length",
            model_path.display()
        );
    }
    let token_counts = parse_usize_list(&args.model_token_counts, "model token count")?;
    if token_counts.is_empty() {
        bail!("--model-token-counts must contain at least one token count");
    }
    let model_name = args.model_name.clone().unwrap_or_else(|| {
        model_path
            .file_stem()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or("model")
            .to_string()
    });
    Ok(token_counts
        .into_iter()
        .map(|tokens| ActivationShape {
            name: format!(
                "{model_name}-{tokens}tok-{}bpe",
                args.model_bytes_per_element
            ),
            tokens,
            hidden_size,
            bytes_per_element: args.model_bytes_per_element,
        })
        .collect())
}

fn bench_shape(
    shape: &ActivationShape,
    payload: &[u8],
    stripes: usize,
    chunk_bytes: usize,
    per_stream_mbps: f64,
    baseline_floor_ms: f64,
    repetitions: usize,
) -> Result<BenchRow> {
    let stripe_count = stripes.max(1);
    let started = Instant::now();
    let mut chunks = 0;
    for _ in 0..repetitions {
        if stripe_count == 1 {
            bench_single_stream(shape, payload)?;
            chunks = 1;
        } else {
            chunks = bench_striped(shape, payload, stripe_count, chunk_bytes)?;
        }
    }
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0 / repetitions as f64;
    let active_streams = stripe_count.min(chunks).max(1);
    let transfer_floor_ms = synthetic_transfer_ms(payload.len(), active_streams, per_stream_mbps);
    let effective_mbps = effective_mbps(payload.len(), transfer_floor_ms);
    Ok(BenchRow {
        shape: shape.name.clone(),
        tokens: shape.tokens,
        hidden_size: shape.hidden_size,
        bytes_per_element: shape.bytes_per_element,
        payload_bytes: payload.len(),
        stripes: stripe_count,
        chunk_bytes,
        chunks,
        synthetic_per_stream_mbps: per_stream_mbps,
        wall_ms,
        transfer_floor_ms,
        effective_mbps,
        speedup_vs_single_stream: baseline_floor_ms / transfer_floor_ms,
    })
}

fn bench_single_stream(shape: &ActivationShape, payload: &[u8]) -> Result<()> {
    let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd, WireActivationDType::F16);
    state.source_stage_index = 0;
    let message = StageWireMessage {
        kind: WireMessageKind::PrefillEmbd,
        pos_start: 0,
        token_count: i32::try_from(shape.tokens).context("token count exceeds i32")?,
        state,
        request_id: 7,
        session_id: 11,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        positions: Vec::new(),
        activation: payload.to_vec(),
        raw_bytes: Vec::new(),
    };
    let mut encoded = Vec::new();
    write_stage_message(&mut encoded, &message, WireActivationDType::F16)?;
    let decoded = read_stage_message(
        encoded.as_slice(),
        i32::try_from(shape.hidden_size).context("hidden size exceeds i32")?,
    )?;
    if decoded.activation != payload {
        bail!("baseline activation payload mismatch");
    }
    Ok(())
}

fn bench_striped(
    shape: &ActivationShape,
    payload: &[u8],
    stripes: usize,
    chunk_bytes: usize,
) -> Result<usize> {
    let mut control_state =
        StageStateHeader::new(WireMessageKind::PrefillEmbd, WireActivationDType::F16);
    control_state.source_stage_index = 0;
    control_state.flags |= state_flags::STRIPED_ACTIVATION;
    let control = StageWireMessage {
        kind: WireMessageKind::PrefillEmbd,
        pos_start: 0,
        token_count: i32::try_from(shape.tokens).context("token count exceeds i32")?,
        state: control_state,
        request_id: 7,
        session_id: 11,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    };
    let mut control_encoded = Vec::new();
    write_stage_message(&mut control_encoded, &control, WireActivationDType::F16)?;
    let decoded_control = read_stage_message(
        control_encoded.as_slice(),
        i32::try_from(shape.hidden_size).context("hidden size exceeds i32")?,
    )?;
    if !decoded_control.state.uses_striped_activation() {
        bail!("striped control flag did not round-trip");
    }

    let chunks = stripe_activation_payload(7, 11, 13, payload, chunk_bytes)?;
    let mut streams = (0..stripes).map(|_| Vec::new()).collect::<Vec<_>>();
    for (index, chunk) in chunks.iter().enumerate() {
        let stream_index = index % streams.len();
        write_activation_stripe_chunk(&mut streams[stream_index], chunk)?;
    }

    let mut decoded_chunks = Vec::with_capacity(chunks.len());
    for stream in streams {
        let mut cursor = std::io::Cursor::new(stream);
        while usize::try_from(cursor.position()).unwrap_or(usize::MAX) < cursor.get_ref().len() {
            decoded_chunks.push(read_activation_stripe_chunk(&mut cursor)?);
        }
    }
    decoded_chunks.sort_by_key(|chunk| std::cmp::Reverse(chunk.chunk_index));
    let mut iter = decoded_chunks.into_iter();
    let first = iter.next().context("missing first stripe chunk")?;
    let mut reassembler = StageActivationStripeReassembler::new(first)?;
    for chunk in iter {
        reassembler.push(chunk)?;
    }
    let reassembled = reassembler.finish()?;
    if reassembled != payload {
        bail!("striped activation payload mismatch");
    }
    Ok(chunks.len())
}

fn synthetic_transfer_ms(payload_bytes: usize, stripes: usize, per_stream_mbps: f64) -> f64 {
    let effective_mbps = per_stream_mbps * stripes.max(1) as f64;
    payload_bytes as f64 / (effective_mbps * 125_000.0) * 1000.0
}

fn effective_mbps(payload_bytes: usize, transfer_ms: f64) -> f64 {
    payload_bytes as f64 / (transfer_ms / 1000.0) / 125_000.0
}

fn deterministic_payload(size: usize) -> Vec<u8> {
    (0..size).map(|index| (index % 251) as u8).collect()
}

fn parse_shapes(value: &str) -> Result<Vec<ActivationShape>> {
    value
        .split(',')
        .filter(|entry| !entry.trim().is_empty())
        .map(parse_shape)
        .collect()
}

fn parse_shape(value: &str) -> Result<ActivationShape> {
    let parts = value.split(':').collect::<Vec<_>>();
    if parts.len() != 4 {
        bail!("activation shape must be NAME:TOKENS:HIDDEN_SIZE:BYTES_PER_ELEMENT");
    }
    Ok(ActivationShape {
        name: parts[0].to_string(),
        tokens: parts[1]
            .parse()
            .with_context(|| format!("invalid token count in shape {value}"))?,
        hidden_size: parts[2]
            .parse()
            .with_context(|| format!("invalid hidden size in shape {value}"))?,
        bytes_per_element: parts[3]
            .parse()
            .with_context(|| format!("invalid bytes per element in shape {value}"))?,
    })
}

fn parse_usize_list(value: &str, label: &str) -> Result<Vec<usize>> {
    value
        .split(',')
        .filter(|entry| !entry.trim().is_empty())
        .map(|entry| {
            entry
                .trim()
                .parse()
                .with_context(|| format!("invalid {label} value {entry}"))
        })
        .collect()
}

fn print_markdown_table(rows: &[BenchRow]) {
    println!(
        "| Shape | Payload | Stripes | Chunks | Encode + reassemble wall ms | Synthetic transfer floor ms | Synthetic speedup |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|");
    for row in rows {
        println!(
            "| {} | {} | {} | {} | {:.3} | {:.3} | {:.2}x |",
            row.shape,
            human_bytes(row.payload_bytes),
            row.stripes,
            row.chunks,
            row.wall_ms,
            row.transfer_floor_ms,
            row.speedup_vs_single_stream
        );
    }
}

fn human_bytes(bytes: usize) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    format!("{:.1} MiB", bytes as f64 / MIB)
}
