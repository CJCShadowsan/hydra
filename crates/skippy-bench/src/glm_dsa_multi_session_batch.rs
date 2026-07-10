use std::{ffi::OsString, fs, path::PathBuf, time::Instant};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use skippy_protocol::binary::ACTIVATION_FLAG_GLM_DSA_TOP_K;
use skippy_runtime::package::select_layer_package_parts;
use skippy_runtime::{
    ActivationFrame, DecodeFrameBatchRequest, RuntimeConfig, StageModel, StageSession,
};

use crate::{
    cli::GlmDsaLayerMicrobenchArgs,
    glm_dsa_layer_microbench::{package_request_for_range, runtime_config_for_range},
    glm_dsa_microbench_summary::{TimingDistributionSummary, summarize_elapsed_ms},
};

#[derive(Debug, Serialize)]
struct MultiSessionBatchReport {
    command: &'static str,
    model_id: String,
    layer_start: u32,
    layer_end: u32,
    session_count: usize,
    iterations: usize,
    warmup: usize,
    serial_timing: TimingDistributionSummary,
    generic_serial_timing: TimingDistributionSummary,
    batch_timing: TimingDistributionSummary,
    aggregate_speedup: f64,
    generic_batch_speedup: f64,
    serial_rows_per_second: f64,
    batch_rows_per_second: f64,
    hidden_max_abs_diff: f32,
    hidden_relative_rmse: f64,
    hidden_cosine_similarity: f64,
    optimized_reference_hidden_max_abs_diff: f32,
    optimized_reference_relative_rmse: f64,
    optimized_reference_cosine_similarity: f64,
    hidden_parity: bool,
    native_batch_sideband_free: bool,
    split_chain: Option<SplitChainReport>,
    prefill_split_chain: Option<PrefillSplitChainReport>,
}

#[derive(Debug, Serialize)]
struct SplitChainReport {
    producer_layer_start: u32,
    producer_layer_end: u32,
    consumer_layer_start: u32,
    consumer_layer_end: u32,
    contiguous_batch_timing: TimingDistributionSummary,
    split_batch_timing: TimingDistributionSummary,
    contiguous_over_split_speed_ratio: f64,
    hidden_max_abs_diff: f32,
    hidden_relative_rmse: f64,
    hidden_cosine_similarity: f64,
    output_sideband_exact: bool,
    producer_sideband_rows: usize,
    expected_producer_sideband_rows: usize,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct PrefillSplitChainReport {
    chunk_token_counts: Vec<usize>,
    producer_top_k_widths: Vec<usize>,
    hidden_max_abs_diff: f32,
    hidden_relative_rmse: f64,
    hidden_cosine_similarity: f64,
    output_sideband_exact: bool,
    passed: bool,
}

pub(crate) fn run_glm_dsa_multi_session_batch_parity(
    args: &GlmDsaLayerMicrobenchArgs,
    selected_paths: &[PathBuf],
    runtime_config: &RuntimeConfig,
    input: &ActivationFrame,
) -> Result<()> {
    let session_count = args.tokens;
    let inputs = split_session_inputs(input, args.activation_width, session_count)?;
    let config = multi_session_config(runtime_config, session_count)?;
    let token_ids = (0..session_count)
        .map(|lane| i32::try_from(lane + 1).context("session token id exceeds i32"))
        .collect::<Result<Vec<_>>>()?;

    let reference = run_reference_phase(
        selected_paths,
        &config,
        session_count,
        &inputs,
        &token_ids,
        args.warmup,
        args.iterations,
        args.activation_width,
    )?;
    let performance = run_performance_phase(
        selected_paths,
        &config,
        session_count,
        &inputs,
        &token_ids,
        args.warmup,
        args.iterations,
        args.activation_width,
    )?;
    let split_chain = run_split_chain_gate(
        args,
        selected_paths,
        runtime_config,
        session_count,
        &inputs,
        &token_ids,
    )?;
    let prefill_split_chain =
        run_prefill_split_chain_gate(args, selected_paths, runtime_config, &inputs, &token_ids)?;
    let serial_timing = summarize_elapsed_ms(performance.serial_ms.iter().copied());
    let generic_serial_timing = summarize_elapsed_ms(reference.generic_serial_ms.iter().copied());
    let batch_timing = summarize_elapsed_ms(performance.batch_ms.iter().copied());
    let serial_total_ms = performance.serial_ms.iter().sum::<f64>();
    let generic_serial_total_ms = reference.generic_serial_ms.iter().sum::<f64>();
    let generic_batch_total_ms = reference.batch_ms.iter().sum::<f64>();
    let batch_total_ms = performance.batch_ms.iter().sum::<f64>();
    let aggregate_speedup = serial_total_ms / batch_total_ms.max(f64::EPSILON);
    let generic_batch_speedup = generic_serial_total_ms / generic_batch_total_ms.max(f64::EPSILON);
    let measured_rows = (session_count * args.iterations) as f64;
    let hidden_parity = reference.hidden_max_abs_diff <= args.parity_atol;

    let report = MultiSessionBatchReport {
        command: "glm-dsa-layer-microbench --multi-session-batch-parity",
        model_id: args.model_id.clone(),
        layer_start: args.layer_start,
        layer_end: args.layer_end,
        session_count,
        iterations: args.iterations,
        warmup: args.warmup,
        serial_timing,
        generic_serial_timing,
        batch_timing,
        aggregate_speedup,
        generic_batch_speedup,
        serial_rows_per_second: measured_rows * 1000.0 / serial_total_ms.max(f64::EPSILON),
        batch_rows_per_second: measured_rows * 1000.0 / batch_total_ms.max(f64::EPSILON),
        hidden_max_abs_diff: reference.hidden_max_abs_diff,
        hidden_relative_rmse: reference.hidden_relative_rmse,
        hidden_cosine_similarity: reference.hidden_cosine_similarity,
        optimized_reference_hidden_max_abs_diff: performance
            .optimized_reference_hidden_max_abs_diff,
        optimized_reference_relative_rmse: performance.optimized_reference_relative_rmse,
        optimized_reference_cosine_similarity: performance.optimized_reference_cosine_similarity,
        hidden_parity,
        native_batch_sideband_free: input.desc.flags == 0
            && (reference.output_flags | performance.output_flags) == 0,
        split_chain,
        prefill_split_chain,
    };
    let json = serde_json::to_string_pretty(&report)
        .context("serialize GLM multi-session batch report")?;
    if let Some(path) = args.output.as_deref() {
        fs::write(path, format!("{json}\n"))
            .with_context(|| format!("write {}", path.display()))?;
    }
    println!("{json}");
    if !hidden_parity {
        bail!(
            "native GLM multi-session batch differs from generic serial execution: max_abs={} atol={}",
            reference.hidden_max_abs_diff,
            args.parity_atol
        );
    }
    if report
        .split_chain
        .as_ref()
        .is_some_and(|split_chain| !split_chain.passed)
    {
        bail!("native GLM multi-session split-chain parity failed");
    }
    if report
        .prefill_split_chain
        .as_ref()
        .is_some_and(|split_chain| !split_chain.passed)
    {
        bail!("native GLM multi-session prefill split-chain parity failed");
    }
    Ok(())
}

struct BatchSeries {
    elapsed_ms: Vec<f64>,
    outputs: Vec<Vec<ActivationFrame>>,
}

#[allow(clippy::too_many_arguments)]
fn run_prefill_split_chain_gate(
    args: &GlmDsaLayerMicrobenchArgs,
    contiguous_paths: &[PathBuf],
    runtime_config: &RuntimeConfig,
    inputs: &[ActivationFrame],
    token_ids: &[i32],
) -> Result<Option<PrefillSplitChainReport>> {
    let producer_end = args
        .layer_start
        .checked_add(1)
        .context("multi-session prefill producer layer overflow")?;
    if producer_end >= args.layer_end || inputs.len() < 2 {
        return Ok(None);
    }

    let split_at = (inputs.len() / 2).max(1);
    let input_chunks = vec![
        combine_session_inputs(&inputs[..split_at], args.activation_width)?,
        combine_session_inputs(&inputs[split_at..], args.activation_width)?,
    ];
    let token_chunks = vec![&token_ids[..split_at], &token_ids[split_at..]];
    let chunk_token_counts = token_chunks
        .iter()
        .map(|tokens| tokens.len())
        .collect::<Vec<_>>();
    let config = multi_session_config(runtime_config, input_chunks.len())?;
    let contiguous_outputs =
        run_contiguous_prefill_series(contiguous_paths, &config, &input_chunks, &token_chunks)?;

    let producer_selection = select_layer_package_parts(&package_request_for_range(
        args,
        args.layer_start,
        producer_end,
    ))
    .context("select multi-session prefill GLM Full producer")?;
    let consumer_selection = select_layer_package_parts(&package_request_for_range(
        args,
        producer_end,
        args.layer_end,
    ))
    .context("select multi-session prefill GLM Shared consumers")?;
    let producer_config = multi_session_config(
        &runtime_config_for_range(args, args.layer_start, producer_end)?,
        input_chunks.len(),
    )?;
    let consumer_config = multi_session_config(
        &runtime_config_for_range(args, producer_end, args.layer_end)?,
        input_chunks.len(),
    )?;
    let producer_model =
        StageModel::open_from_parts(&producer_selection.absolute_paths, &producer_config)
            .context("open multi-session prefill GLM Full producer")?;
    let consumer_model =
        StageModel::open_from_parts(&consumer_selection.absolute_paths, &consumer_config)
            .context("open multi-session prefill GLM Shared consumers")?;
    let mut producer_sessions =
        create_sessions(&producer_model, input_chunks.len(), "prefill producer")?;
    let mut consumer_sessions =
        create_sessions(&consumer_model, input_chunks.len(), "prefill consumer")?;
    let mut split_outputs = Vec::with_capacity(input_chunks.len());
    let mut producer_top_k_widths = Vec::with_capacity(input_chunks.len());
    for (((producer, consumer), input), tokens) in producer_sessions
        .iter_mut()
        .zip(&mut consumer_sessions)
        .zip(&input_chunks)
        .zip(&token_chunks)
    {
        let produced = producer.prefill_chunk_frame(tokens, Some(input), 0)?;
        producer_top_k_widths.push(frame_top_k_width(&produced, args.activation_width)?);
        split_outputs.push(consumer.prefill_chunk_frame(tokens, Some(&produced), 0)?);
    }

    let diff = compare_outputs(&contiguous_outputs, &split_outputs, args.activation_width)?;
    let output_sideband_exact =
        output_sidebands_equal(&contiguous_outputs, &split_outputs, args.activation_width)?;
    let passed = diff.max_abs <= args.parity_atol
        && output_sideband_exact
        && producer_top_k_widths == chunk_token_counts;
    Ok(Some(PrefillSplitChainReport {
        chunk_token_counts,
        producer_top_k_widths,
        hidden_max_abs_diff: diff.max_abs,
        hidden_relative_rmse: diff.relative_rmse,
        hidden_cosine_similarity: diff.cosine_similarity,
        output_sideband_exact,
        passed,
    }))
}

fn run_contiguous_prefill_series(
    selected_paths: &[PathBuf],
    config: &RuntimeConfig,
    inputs: &[ActivationFrame],
    token_chunks: &[&[i32]],
) -> Result<Vec<ActivationFrame>> {
    let model = StageModel::open_from_parts(selected_paths, config)
        .context("open contiguous multi-session prefill reference")?;
    let mut sessions = create_sessions(&model, inputs.len(), "contiguous prefill")?;
    sessions
        .iter_mut()
        .zip(inputs)
        .zip(token_chunks)
        .map(|((session, input), tokens)| session.prefill_chunk_frame(tokens, Some(input), 0))
        .collect()
}

fn combine_session_inputs(
    inputs: &[ActivationFrame],
    activation_width: u32,
) -> Result<ActivationFrame> {
    let first = inputs
        .first()
        .context("multi-session prefill chunk cannot be empty")?;
    let hidden_row_bytes = usize::try_from(activation_width)
        .context("activation width exceeds usize")?
        .checked_mul(std::mem::size_of::<f32>())
        .context("activation row byte count overflow")?;
    let mut payload = Vec::with_capacity(hidden_row_bytes * inputs.len());
    for input in inputs {
        if input.desc.flags != 0
            || input.desc.token_count != 1
            || input.payload.len() != hidden_row_bytes
        {
            bail!("multi-session prefill producer inputs must be sideband-free one-row frames");
        }
        payload.extend_from_slice(&input.payload);
    }
    let mut desc = first.desc;
    desc.token_count = u32::try_from(inputs.len()).context("prefill token count exceeds u32")?;
    desc.sequence_count = 1;
    desc.payload_bytes = u64::try_from(payload.len()).context("prefill payload exceeds u64")?;
    Ok(ActivationFrame { desc, payload })
}

fn frame_top_k_width(frame: &ActivationFrame, activation_width: u32) -> Result<usize> {
    if (frame.desc.flags & ACTIVATION_FLAG_GLM_DSA_TOP_K) == 0 {
        bail!("GLM Full producer did not emit a top-k sideband");
    }
    let tokens =
        usize::try_from(frame.desc.token_count).context("activation token count exceeds usize")?;
    let bytes = frame_sideband(frame, activation_width)?.len();
    let row_bytes = bytes
        .checked_div(tokens)
        .filter(|_| bytes % tokens == 0)
        .context("GLM top-k sideband is not token-major")?;
    if !row_bytes.is_multiple_of(std::mem::size_of::<i32>()) {
        bail!("GLM top-k sideband row is not i32-aligned");
    }
    Ok(row_bytes / std::mem::size_of::<i32>())
}

fn output_sidebands_equal(
    reference: &[ActivationFrame],
    candidate: &[ActivationFrame],
    activation_width: u32,
) -> Result<bool> {
    if reference.len() != candidate.len() {
        return Ok(false);
    }
    for (reference, candidate) in reference.iter().zip(candidate) {
        if reference.desc.flags != candidate.desc.flags
            || frame_sideband(reference, activation_width)?
                != frame_sideband(candidate, activation_width)?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn run_split_chain_gate(
    args: &GlmDsaLayerMicrobenchArgs,
    contiguous_paths: &[PathBuf],
    runtime_config: &RuntimeConfig,
    session_count: usize,
    inputs: &[ActivationFrame],
    token_ids: &[i32],
) -> Result<Option<SplitChainReport>> {
    let producer_end = args
        .layer_start
        .checked_add(1)
        .context("multi-session split-chain producer layer overflow")?;
    if producer_end >= args.layer_end {
        return Ok(None);
    }

    let contiguous_config = multi_session_config(runtime_config, session_count)?;
    let contiguous = run_contiguous_batch_series(
        contiguous_paths,
        &contiguous_config,
        session_count,
        inputs,
        token_ids,
        args.warmup,
        args.iterations,
    )?;

    let producer_selection = select_layer_package_parts(&package_request_for_range(
        args,
        args.layer_start,
        producer_end,
    ))
    .context("select batched GLM Full producer layer")?;
    let consumer_selection = select_layer_package_parts(&package_request_for_range(
        args,
        producer_end,
        args.layer_end,
    ))
    .context("select batched GLM Shared consumer layers")?;
    let producer_config = multi_session_config(
        &runtime_config_for_range(args, args.layer_start, producer_end)?,
        session_count,
    )?;
    let consumer_config = multi_session_config(
        &runtime_config_for_range(args, producer_end, args.layer_end)?,
        session_count,
    )?;
    let (split, producer_sideband_rows) = run_split_batch_series(
        &producer_selection.absolute_paths,
        &producer_config,
        &consumer_selection.absolute_paths,
        &consumer_config,
        session_count,
        inputs,
        token_ids,
        args.warmup,
        args.iterations,
        args.activation_width,
    )?;
    let diff = compare_output_series(&contiguous.outputs, &split.outputs, args.activation_width)?;
    let output_sideband_exact =
        output_series_sidebands_equal(&contiguous.outputs, &split.outputs, args.activation_width)?;
    let expected_producer_sideband_rows = session_count
        .checked_mul(args.iterations)
        .context("expected GLM producer sideband row count overflow")?;
    let contiguous_total_ms = contiguous.elapsed_ms.iter().sum::<f64>();
    let split_total_ms = split.elapsed_ms.iter().sum::<f64>();
    let passed = diff.max_abs <= args.parity_atol
        && output_sideband_exact
        && producer_sideband_rows == expected_producer_sideband_rows;

    Ok(Some(SplitChainReport {
        producer_layer_start: args.layer_start,
        producer_layer_end: producer_end,
        consumer_layer_start: producer_end,
        consumer_layer_end: args.layer_end,
        contiguous_batch_timing: summarize_elapsed_ms(contiguous.elapsed_ms.iter().copied()),
        split_batch_timing: summarize_elapsed_ms(split.elapsed_ms.iter().copied()),
        contiguous_over_split_speed_ratio: contiguous_total_ms / split_total_ms.max(f64::EPSILON),
        hidden_max_abs_diff: diff.max_abs,
        hidden_relative_rmse: diff.relative_rmse,
        hidden_cosine_similarity: diff.cosine_similarity,
        output_sideband_exact,
        producer_sideband_rows,
        expected_producer_sideband_rows,
        passed,
    }))
}

#[allow(clippy::too_many_arguments)]
fn run_contiguous_batch_series(
    selected_paths: &[PathBuf],
    config: &RuntimeConfig,
    session_count: usize,
    inputs: &[ActivationFrame],
    token_ids: &[i32],
    warmup: usize,
    iterations: usize,
) -> Result<BatchSeries> {
    let model = StageModel::open_from_parts(selected_paths, config)
        .context("open contiguous GLM multi-session split-chain reference")?;
    let mut sessions = create_sessions(&model, session_count, "contiguous split-chain")?;
    for _ in 0..warmup {
        timed_batch(&mut sessions, inputs, token_ids)?;
    }
    let mut elapsed_ms = Vec::with_capacity(iterations);
    let mut outputs = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let (elapsed, output) = timed_batch(&mut sessions, inputs, token_ids)?;
        elapsed_ms.push(elapsed);
        outputs.push(output);
    }
    Ok(BatchSeries {
        elapsed_ms,
        outputs,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_split_batch_series(
    producer_paths: &[PathBuf],
    producer_config: &RuntimeConfig,
    consumer_paths: &[PathBuf],
    consumer_config: &RuntimeConfig,
    session_count: usize,
    inputs: &[ActivationFrame],
    token_ids: &[i32],
    warmup: usize,
    iterations: usize,
    activation_width: u32,
) -> Result<(BatchSeries, usize)> {
    let producer_model = StageModel::open_from_parts(producer_paths, producer_config)
        .context("open batched GLM Full producer")?;
    let consumer_model = StageModel::open_from_parts(consumer_paths, consumer_config)
        .context("open batched GLM Shared consumers")?;
    let mut producer_sessions =
        create_sessions(&producer_model, session_count, "split-chain producer")?;
    let mut consumer_sessions =
        create_sessions(&consumer_model, session_count, "split-chain consumer")?;
    for _ in 0..warmup {
        let (_, produced) = timed_batch(&mut producer_sessions, inputs, token_ids)?;
        timed_batch(&mut consumer_sessions, &produced, token_ids)?;
    }
    let mut elapsed_ms = Vec::with_capacity(iterations);
    let mut outputs = Vec::with_capacity(iterations);
    let mut producer_sideband_rows = 0_usize;
    for _ in 0..iterations {
        let started = Instant::now();
        let (_, produced) = timed_batch(&mut producer_sessions, inputs, token_ids)?;
        producer_sideband_rows += count_glm_dsa_sideband_rows(&produced, activation_width)?;
        let (_, output) = timed_batch(&mut consumer_sessions, &produced, token_ids)?;
        elapsed_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        outputs.push(output);
    }
    Ok((
        BatchSeries {
            elapsed_ms,
            outputs,
        },
        producer_sideband_rows,
    ))
}

fn compare_output_series(
    reference: &[Vec<ActivationFrame>],
    candidate: &[Vec<ActivationFrame>],
    activation_width: u32,
) -> Result<OutputDiff> {
    if reference.len() != candidate.len() {
        bail!("contiguous and split-chain iteration counts differ");
    }
    let mut max_abs = 0.0_f32;
    let mut relative_rmse = 0.0_f64;
    let mut cosine_similarity = 1.0_f64;
    for (reference, candidate) in reference.iter().zip(candidate) {
        let diff = compare_outputs(reference, candidate, activation_width)?;
        max_abs = max_abs.max(diff.max_abs);
        relative_rmse = relative_rmse.max(diff.relative_rmse);
        cosine_similarity = cosine_similarity.min(diff.cosine_similarity);
    }
    Ok(OutputDiff {
        max_abs,
        relative_rmse,
        cosine_similarity,
    })
}

fn output_series_sidebands_equal(
    reference: &[Vec<ActivationFrame>],
    candidate: &[Vec<ActivationFrame>],
    activation_width: u32,
) -> Result<bool> {
    if reference.len() != candidate.len() {
        return Ok(false);
    }
    for (reference, candidate) in reference.iter().zip(candidate) {
        if reference.len() != candidate.len() {
            return Ok(false);
        }
        for (reference, candidate) in reference.iter().zip(candidate) {
            if reference.desc.flags != candidate.desc.flags
                || frame_sideband(reference, activation_width)?
                    != frame_sideband(candidate, activation_width)?
            {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn count_glm_dsa_sideband_rows(frames: &[ActivationFrame], activation_width: u32) -> Result<usize> {
    let mut rows = 0_usize;
    for frame in frames {
        if (frame.desc.flags & ACTIVATION_FLAG_GLM_DSA_TOP_K) != 0
            && !frame_sideband(frame, activation_width)?.is_empty()
        {
            rows += usize::try_from(frame.desc.token_count)
                .context("GLM producer token count exceeds usize")?;
        }
    }
    Ok(rows)
}

fn frame_sideband(frame: &ActivationFrame, activation_width: u32) -> Result<&[u8]> {
    let tokens =
        usize::try_from(frame.desc.token_count).context("activation token count exceeds usize")?;
    let hidden_bytes = usize::try_from(activation_width)
        .context("activation width exceeds usize")?
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|row_bytes| row_bytes.checked_mul(tokens))
        .context("activation hidden byte count overflow")?;
    frame
        .payload
        .get(hidden_bytes..)
        .context("activation payload is smaller than its hidden rows")
}

struct ReferenceMeasurements {
    generic_serial_ms: Vec<f64>,
    batch_ms: Vec<f64>,
    hidden_max_abs_diff: f32,
    hidden_relative_rmse: f64,
    hidden_cosine_similarity: f64,
    output_flags: u64,
}

struct PerformanceMeasurements {
    serial_ms: Vec<f64>,
    batch_ms: Vec<f64>,
    optimized_reference_hidden_max_abs_diff: f32,
    optimized_reference_relative_rmse: f64,
    optimized_reference_cosine_similarity: f64,
    output_flags: u64,
}

#[allow(clippy::too_many_arguments)]
fn run_reference_phase(
    selected_paths: &[PathBuf],
    config: &RuntimeConfig,
    session_count: usize,
    inputs: &[ActivationFrame],
    token_ids: &[i32],
    warmup: usize,
    iterations: usize,
    activation_width: u32,
) -> Result<ReferenceMeasurements> {
    let generic_model = StageModel::open_from_parts(selected_paths, config)
        .context("open generic serial GLM multi-session reference")?;
    let batch_model = StageModel::open_from_parts(selected_paths, config)
        .context("open GLM multi-session correctness candidate")?;
    let mut generic_sessions = create_sessions(&generic_model, session_count, "generic serial")?;
    let mut batch_sessions = create_sessions(&batch_model, session_count, "correctness batch")?;
    warm_reference_sessions(
        &mut generic_sessions,
        &mut batch_sessions,
        inputs,
        token_ids,
        warmup,
    )?;
    measure_reference_sessions(
        &mut generic_sessions,
        &mut batch_sessions,
        inputs,
        token_ids,
        iterations,
        activation_width,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_performance_phase(
    selected_paths: &[PathBuf],
    config: &RuntimeConfig,
    session_count: usize,
    inputs: &[ActivationFrame],
    token_ids: &[i32],
    warmup: usize,
    iterations: usize,
    activation_width: u32,
) -> Result<PerformanceMeasurements> {
    let serial_model = StageModel::open_from_parts(selected_paths, config)
        .context("open optimized serial GLM multi-session baseline")?;
    let batch_model = StageModel::open_from_parts(selected_paths, config)
        .context("open GLM multi-session performance candidate")?;
    let mut serial_sessions = create_sessions(&serial_model, session_count, "optimized serial")?;
    let mut batch_sessions = create_sessions(&batch_model, session_count, "performance batch")?;
    warm_performance_sessions(
        &mut serial_sessions,
        &mut batch_sessions,
        inputs,
        token_ids,
        warmup,
    )?;
    measure_performance_sessions(
        &mut serial_sessions,
        &mut batch_sessions,
        inputs,
        token_ids,
        iterations,
        activation_width,
    )
}

fn measure_reference_sessions(
    generic_serial_sessions: &mut [StageSession],
    batch_sessions: &mut [StageSession],
    inputs: &[ActivationFrame],
    token_ids: &[i32],
    iterations: usize,
    activation_width: u32,
) -> Result<ReferenceMeasurements> {
    let mut generic_serial_ms = Vec::with_capacity(iterations);
    let mut batch_ms = Vec::with_capacity(iterations);
    let mut hidden_max_abs_diff = 0.0_f32;
    let mut hidden_relative_rmse = 0.0_f64;
    let mut hidden_cosine_similarity = 1.0_f64;
    let mut output_flags = 0_u64;
    for iteration in 0..iterations {
        let (generic_serial, batch) = if iteration % 2 == 0 {
            let generic_serial = timed_generic_serial(generic_serial_sessions, inputs, token_ids)?;
            let batch = timed_batch(batch_sessions, inputs, token_ids)?;
            (generic_serial, batch)
        } else {
            let batch = timed_batch(batch_sessions, inputs, token_ids)?;
            let generic_serial = timed_generic_serial(generic_serial_sessions, inputs, token_ids)?;
            (generic_serial, batch)
        };
        let diff = compare_outputs(&generic_serial.1, &batch.1, activation_width)?;
        hidden_max_abs_diff = hidden_max_abs_diff.max(diff.max_abs);
        hidden_relative_rmse = hidden_relative_rmse.max(diff.relative_rmse);
        hidden_cosine_similarity = hidden_cosine_similarity.min(diff.cosine_similarity);
        output_flags |= generic_serial
            .1
            .iter()
            .chain(batch.1.iter())
            .fold(0_u64, |flags, output| flags | output.desc.flags);
        generic_serial_ms.push(generic_serial.0);
        batch_ms.push(batch.0);
    }
    Ok(ReferenceMeasurements {
        generic_serial_ms,
        batch_ms,
        hidden_max_abs_diff,
        hidden_relative_rmse,
        hidden_cosine_similarity,
        output_flags,
    })
}

fn measure_performance_sessions(
    serial_sessions: &mut [StageSession],
    batch_sessions: &mut [StageSession],
    inputs: &[ActivationFrame],
    token_ids: &[i32],
    iterations: usize,
    activation_width: u32,
) -> Result<PerformanceMeasurements> {
    let mut serial_ms = Vec::with_capacity(iterations);
    let mut batch_ms = Vec::with_capacity(iterations);
    let mut optimized_reference_hidden_max_abs_diff = 0.0_f32;
    let mut optimized_reference_relative_rmse = 0.0_f64;
    let mut optimized_reference_cosine_similarity = 1.0_f64;
    let mut output_flags = 0_u64;
    for iteration in 0..iterations {
        let (serial, batch) = if iteration % 2 == 0 {
            let serial = timed_serial(serial_sessions, inputs, token_ids)?;
            let batch = timed_batch(batch_sessions, inputs, token_ids)?;
            (serial, batch)
        } else {
            let batch = timed_batch(batch_sessions, inputs, token_ids)?;
            let serial = timed_serial(serial_sessions, inputs, token_ids)?;
            (serial, batch)
        };
        let diff = compare_outputs(&serial.1, &batch.1, activation_width)?;
        optimized_reference_hidden_max_abs_diff =
            optimized_reference_hidden_max_abs_diff.max(diff.max_abs);
        optimized_reference_relative_rmse =
            optimized_reference_relative_rmse.max(diff.relative_rmse);
        optimized_reference_cosine_similarity =
            optimized_reference_cosine_similarity.min(diff.cosine_similarity);
        output_flags |= serial
            .1
            .iter()
            .chain(batch.1.iter())
            .fold(0_u64, |flags, output| flags | output.desc.flags);
        serial_ms.push(serial.0);
        batch_ms.push(batch.0);
    }
    Ok(PerformanceMeasurements {
        serial_ms,
        batch_ms,
        optimized_reference_hidden_max_abs_diff,
        optimized_reference_relative_rmse,
        optimized_reference_cosine_similarity,
        output_flags,
    })
}

fn warm_reference_sessions(
    generic_serial_sessions: &mut [StageSession],
    batch_sessions: &mut [StageSession],
    inputs: &[ActivationFrame],
    token_ids: &[i32],
    warmup: usize,
) -> Result<()> {
    for iteration in 0..warmup {
        if iteration % 2 == 0 {
            timed_generic_serial(generic_serial_sessions, inputs, token_ids)?;
            timed_batch(batch_sessions, inputs, token_ids)?;
        } else {
            timed_batch(batch_sessions, inputs, token_ids)?;
            timed_generic_serial(generic_serial_sessions, inputs, token_ids)?;
        }
    }
    Ok(())
}

fn warm_performance_sessions(
    serial_sessions: &mut [StageSession],
    batch_sessions: &mut [StageSession],
    inputs: &[ActivationFrame],
    token_ids: &[i32],
    warmup: usize,
) -> Result<()> {
    for iteration in 0..warmup {
        if iteration % 2 == 0 {
            timed_serial(serial_sessions, inputs, token_ids)?;
            timed_batch(batch_sessions, inputs, token_ids)?;
        } else {
            timed_batch(batch_sessions, inputs, token_ids)?;
            timed_serial(serial_sessions, inputs, token_ids)?;
        }
    }
    Ok(())
}

fn timed_serial(
    sessions: &mut [StageSession],
    inputs: &[ActivationFrame],
    token_ids: &[i32],
) -> Result<(f64, Vec<ActivationFrame>)> {
    let started = Instant::now();
    let outputs = sessions
        .iter_mut()
        .zip(inputs)
        .zip(token_ids)
        .map(|((session, input), token_id)| {
            session
                .decode_step_frame(*token_id, Some(input), 0)
                .map(|(_, output)| output)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((started.elapsed().as_secs_f64() * 1000.0, outputs))
}

fn timed_generic_serial(
    sessions: &mut [StageSession],
    inputs: &[ActivationFrame],
    token_ids: &[i32],
) -> Result<(f64, Vec<ActivationFrame>)> {
    let _w0 = ScopedEnvValue::set(
        "GGML_METAL_DISABLE_Q3_DOWN_SLOT_PARALLEL_REDUCE_R8_NB8_W0",
        "1",
    );
    let _w1 = ScopedEnvValue::set(
        "GGML_METAL_DISABLE_Q3_DOWN_SLOT_PARALLEL_REDUCE_R8_NB8_W1",
        "1",
    );
    timed_serial(sessions, inputs, token_ids)
}

struct ScopedEnvValue {
    name: &'static str,
    previous: Option<OsString>,
}

impl ScopedEnvValue {
    fn set(name: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(name);
        // This benchmark is single-threaded while native graph selection reads these values.
        unsafe { std::env::set_var(name, value) };
        Self { name, previous }
    }
}

impl Drop for ScopedEnvValue {
    fn drop(&mut self) {
        // Restore the process environment before another benchmark case is selected.
        unsafe {
            if let Some(previous) = &self.previous {
                std::env::set_var(self.name, previous);
            } else {
                std::env::remove_var(self.name);
            }
        }
    }
}

fn timed_batch(
    sessions: &mut [StageSession],
    inputs: &[ActivationFrame],
    token_ids: &[i32],
) -> Result<(f64, Vec<ActivationFrame>)> {
    let mut requests = sessions
        .iter_mut()
        .zip(inputs)
        .zip(token_ids)
        .map(|((session, input), token_id)| DecodeFrameBatchRequest {
            session,
            token_id: *token_id,
            sampling: None,
            input: Some(input),
        })
        .collect::<Vec<_>>();
    let started = Instant::now();
    let outputs = StageSession::decode_step_frame_batch_sampled(&mut requests)?;
    Ok((
        started.elapsed().as_secs_f64() * 1000.0,
        outputs.into_iter().map(|output| output.output).collect(),
    ))
}

fn create_sessions(
    model: &StageModel,
    session_count: usize,
    label: &str,
) -> Result<Vec<StageSession>> {
    (0..session_count)
        .map(|lane| {
            model
                .create_session()
                .with_context(|| format!("create {label} GLM session {lane}"))
        })
        .collect()
}

fn multi_session_config(config: &RuntimeConfig, session_count: usize) -> Result<RuntimeConfig> {
    let lane_count = u32::try_from(session_count).context("session count exceeds u32")?;
    let mut config = config.clone();
    config.lane_count = lane_count;
    config.n_batch = Some(config.n_batch.unwrap_or(lane_count).max(lane_count));
    config.n_ubatch = Some(config.n_ubatch.unwrap_or(lane_count).max(lane_count));
    Ok(config)
}

fn split_session_inputs(
    frame: &ActivationFrame,
    activation_width: u32,
    session_count: usize,
) -> Result<Vec<ActivationFrame>> {
    if (frame.desc.flags & !ACTIVATION_FLAG_GLM_DSA_TOP_K) != 0 {
        bail!("multi-session batch gate only supports GLM-DSA activation sidebands");
    }
    let source_tokens =
        usize::try_from(frame.desc.token_count).context("activation token count exceeds usize")?;
    if source_tokens != session_count {
        bail!(
            "multi-session input rows ({source_tokens}) must equal session count ({session_count})"
        );
    }
    let row_bytes = usize::try_from(activation_width)
        .context("activation width exceeds usize")?
        .checked_mul(std::mem::size_of::<f32>())
        .context("activation row byte count overflow")?;
    let hidden_bytes = row_bytes
        .checked_mul(session_count)
        .context("multi-session hidden byte count overflow")?;
    if frame.payload.len() < hidden_bytes {
        bail!("multi-session input is smaller than its token-major hidden rows");
    }
    let sideband_bytes = frame.payload.len() - hidden_bytes;
    let sideband_row_bytes = if (frame.desc.flags & ACTIVATION_FLAG_GLM_DSA_TOP_K) != 0 {
        if sideband_bytes == 0
            || !sideband_bytes.is_multiple_of(session_count)
            || !(sideband_bytes / session_count).is_multiple_of(std::mem::size_of::<i32>())
        {
            bail!(
                "multi-session GLM-DSA sideband must contain one request-major i32 row per session"
            );
        }
        sideband_bytes / session_count
    } else {
        if sideband_bytes != 0 {
            bail!("sideband-free multi-session input has trailing payload bytes");
        }
        0
    };
    (0..session_count)
        .map(|lane| {
            let start = lane * row_bytes;
            let mut payload = frame.payload[start..start + row_bytes].to_vec();
            if sideband_row_bytes > 0 {
                let sideband_start = hidden_bytes + lane * sideband_row_bytes;
                payload.extend_from_slice(
                    &frame.payload[sideband_start..sideband_start + sideband_row_bytes],
                );
            }
            let mut desc = frame.desc;
            desc.token_count = 1;
            desc.sequence_count = 1;
            desc.payload_bytes =
                u64::try_from(payload.len()).context("session payload length exceeds u64")?;
            Ok(ActivationFrame { desc, payload })
        })
        .collect()
}

struct OutputDiff {
    max_abs: f32,
    relative_rmse: f64,
    cosine_similarity: f64,
}

fn compare_outputs(
    serial: &[ActivationFrame],
    batched: &[ActivationFrame],
    activation_width: u32,
) -> Result<OutputDiff> {
    if serial.len() != batched.len() {
        bail!("serial and batched output counts differ");
    }
    let hidden_bytes = usize::try_from(activation_width)
        .context("activation width exceeds usize")?
        .checked_mul(std::mem::size_of::<f32>())
        .context("activation hidden byte count overflow")?;
    let mut max_abs = 0.0_f32;
    let mut squared_error = 0.0_f64;
    let mut reference_squared = 0.0_f64;
    let mut candidate_squared = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut count = 0_usize;
    for (serial, batched) in serial.iter().zip(batched) {
        if serial.payload.len() < hidden_bytes || batched.payload.len() < hidden_bytes {
            bail!("serial or batched output is smaller than one hidden row");
        }
        for (left, right) in serial.payload[..hidden_bytes]
            .chunks_exact(4)
            .zip(batched.payload[..hidden_bytes].chunks_exact(4))
        {
            let left = f32::from_ne_bytes(left.try_into().expect("four-byte f32"));
            let right = f32::from_ne_bytes(right.try_into().expect("four-byte f32"));
            let error = f64::from(left) - f64::from(right);
            max_abs = max_abs.max((left - right).abs());
            squared_error += error * error;
            reference_squared += f64::from(left) * f64::from(left);
            candidate_squared += f64::from(right) * f64::from(right);
            dot += f64::from(left) * f64::from(right);
            count += 1;
        }
    }
    let count = count.max(1) as f64;
    let rmse = (squared_error / count).sqrt();
    let reference_rms = (reference_squared / count).sqrt();
    let norm_product = (reference_squared * candidate_squared).sqrt();
    Ok(OutputDiff {
        max_abs,
        relative_rmse: rmse / reference_rms.max(f64::EPSILON),
        cosine_similarity: if norm_product > 0.0 {
            dot / norm_product
        } else if reference_squared == 0.0 && candidate_squared == 0.0 {
            1.0
        } else {
            0.0
        },
    })
}
