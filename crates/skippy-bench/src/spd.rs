use std::{collections::BTreeMap, fs};

use anyhow::{Context, Result, bail};
use serde_json::json;
use skippy_runtime::{
    ActivationFrame, GGML_TYPE_F16, RuntimeConfig, RuntimeLoadMode, StageModel,
    spd::{
        SpdHeadManifest, SpdQwen3ForwardInput, SpdSafetensorsFile, SpdStageLayerRange,
        plan_hidden_state_taps, project_spd_tap_input_row, run_qwen3_fixture_parity,
        run_qwen3_forward_from_inputs, run_spd_tap_input_fixture_parity,
    },
};

use crate::cli::{SpdFixtureParityArgs, SpdLiveTapParityArgs};

pub fn spd_fixture_parity(args: SpdFixtureParityArgs) -> Result<()> {
    let tap_input = run_spd_tap_input_fixture_parity(&args.manifest, &args.fixture)
        .context("failed to reconstruct SPD fixture cur_in from tap inputs")?;
    let forward = run_qwen3_fixture_parity(&args.manifest, &args.fixture, args.top_k)
        .context("failed to run Qwen3 SPD fixture forward parity")?;
    let report = json!({
        "mode": "spd-fixture-parity",
        "manifest": args.manifest,
        "fixture": args.fixture,
        "tap_input": {
            "max_abs_diff": tap_input.max_abs_diff,
            "rows": tap_input.rows.iter().map(|row| {
                json!({
                    "row_index": row.row_index,
                    "position_id": row.position_id,
                    "stage_id": row.stage_id,
                    "projection_name": row.projection_name,
                    "hf_indices": row.hf_indices,
                    "max_abs_diff": row.max_abs_diff,
                })
            }).collect::<Vec<_>>(),
        },
        "forward": {
            "rust": {
                "draft_indices": forward.rust.draft_indices,
                "token_ids": forward.rust.token_ids,
                "logits": forward.rust.logits,
            },
            "python": {
                "draft_indices": forward.python.draft_indices,
                "token_ids": forward.python.token_ids,
                "logits": forward.python.logits,
            },
            "diagnostics": {
                "layer_input_max_abs_diff": forward.diagnostics.layer_input_max_abs_diff,
                "layer_query_max_abs_diff": forward.diagnostics.layer_query_max_abs_diff,
                "spec_query_max_abs_diff": forward.diagnostics.spec_query_max_abs_diff,
                "final_hidden_max_abs_diff": forward.diagnostics.final_hidden_max_abs_diff,
                "python_top_logit_values_at_rust_indices": forward.diagnostics.python_top_logit_values_at_rust_indices,
            }
        }
    });
    let json = serde_json::to_vec_pretty(&report)?;
    if let Some(output) = args.output {
        fs::write(&output, &json)
            .with_context(|| format!("failed to write {}", output.display()))?;
    }
    println!("{}", String::from_utf8(json)?);
    Ok(())
}

pub fn spd_live_tap_parity(args: SpdLiveTapParityArgs) -> Result<()> {
    let manifest = SpdHeadManifest::from_path(&args.manifest)?;
    manifest.ensure_serving_checkpoint_for_runtime(&args.manifest)?;
    let fixture = SpdSafetensorsFile::open(&args.fixture)?;
    let serving = SpdSafetensorsFile::open(manifest.serving_checkpoint_path(&args.manifest)?)?;
    let prompt_tokens = fixture_prompt_tokens(&fixture)?;
    let row_positions = fixture.read_tensor_i64("row_positions")?;
    let row_i_stages = fixture.read_tensor_i64("row_i_stages")?;
    let position_ids = fixture.read_tensor_i64("position_ids")?;
    let final_norm_weight = fixture.read_tensor_f32("final_norm_weight")?;
    let fixture_cur_in = fixture.read_tensor_f32("cur_in")?;
    let row_count = fixture_cur_in_row_count(&fixture, manifest.topology.hidden_size as usize)?;
    validate_fixture_row_inputs(row_count, &row_positions, &row_i_stages, &position_ids)?;

    let ranges = live_stage_ranges(&args)?;
    let tap_plan = plan_hidden_state_taps(&manifest.topology, &ranges)?;
    if tap_plan.requires_internal_taps() {
        bail!(
            "live SPD tap parity requires boundary-aligned splits; missing hidden states {:?}",
            tap_plan.boundary_only_missing_hf_indices
        );
    }

    let hidden_size =
        usize::try_from(manifest.topology.hidden_size).context("SPD hidden_size exceeds usize")?;
    let taps = collect_live_taps(&args, &prompt_tokens, &ranges)?;
    let live_rows = assemble_live_cur_in(
        &manifest,
        &serving,
        &fixture,
        &taps,
        LiveRowInputs {
            row_count,
            row_positions: &row_positions,
            row_i_stages: &row_i_stages,
            fixture_cur_in: &fixture_cur_in,
            hidden_size,
        },
    )?;
    let live_topk = run_qwen3_forward_from_inputs(
        &args.manifest,
        SpdQwen3ForwardInput {
            cur_in: live_rows.cur_in.clone(),
            seq_len: row_count,
            position_ids,
            final_norm_weight,
        },
        args.top_k,
    )?;
    let fixture_forward = run_qwen3_fixture_parity(&args.manifest, &args.fixture, args.top_k)?;
    let report = json!({
        "mode": "spd-live-tap-parity",
        "manifest": args.manifest,
        "fixture": args.fixture,
        "model_path": args.model_path,
        "splits": args.splits,
        "layer_end": args.layer_end,
        "prompt_token_count": prompt_tokens.len(),
        "tap_plan": {
            "required_hf_indices": tap_plan.required_hf_indices,
            "stage_boundary_hf_indices": tap_plan.stage_boundary_hf_indices,
            "boundary_only": tap_plan.can_use_stage_boundaries_only(),
        },
        "live_taps": live_tap_report(&taps),
        "cur_in": {
            "max_abs_diff_vs_fixture": live_rows.max_abs_diff,
            "rows": live_rows.rows,
        },
        "forward": {
            "live_skippy": {
                "draft_indices": live_topk.draft_indices,
                "token_ids": live_topk.token_ids,
                "logits": live_topk.logits,
            },
            "fixture_rust": {
                "draft_indices": fixture_forward.rust.draft_indices,
                "token_ids": fixture_forward.rust.token_ids,
                "logits": fixture_forward.rust.logits,
            },
            "fixture_python": {
                "draft_indices": fixture_forward.python.draft_indices,
                "token_ids": fixture_forward.python.token_ids,
                "logits": fixture_forward.python.logits,
            }
        }
    });
    let json = serde_json::to_vec_pretty(&report)?;
    if let Some(output) = args.output {
        fs::write(&output, &json)
            .with_context(|| format!("failed to write {}", output.display()))?;
    }
    println!("{}", String::from_utf8(json)?);
    Ok(())
}

struct LiveRows {
    cur_in: Vec<f32>,
    rows: Vec<serde_json::Value>,
    max_abs_diff: f32,
}

struct LiveRowInputs<'a> {
    row_count: usize,
    row_positions: &'a [i64],
    row_i_stages: &'a [i64],
    fixture_cur_in: &'a [f32],
    hidden_size: usize,
}

fn fixture_prompt_tokens(fixture: &SpdSafetensorsFile) -> Result<Vec<i32>> {
    let shape = &fixture.index.tensor("prompt_input_ids")?.shape;
    if shape.len() != 2 || shape[0] != 1 {
        bail!(
            "SPD fixture prompt_input_ids shape {:?} is not [1, seq]",
            shape
        );
    }
    fixture
        .read_tensor_i64("prompt_input_ids")?
        .into_iter()
        .map(|token| i32::try_from(token).context("prompt token id exceeds i32"))
        .collect()
}

fn fixture_cur_in_row_count(fixture: &SpdSafetensorsFile, hidden_size: usize) -> Result<usize> {
    let shape = &fixture.index.tensor("cur_in")?.shape;
    if shape.len() != 3 || shape[0] != 1 || shape[2] != hidden_size as u64 {
        bail!(
            "SPD fixture cur_in shape {:?} is not [1, rows, hidden]",
            shape
        );
    }
    usize::try_from(shape[1]).context("SPD fixture row count exceeds usize")
}

fn validate_fixture_row_inputs(
    row_count: usize,
    row_positions: &[i64],
    row_i_stages: &[i64],
    position_ids: &[i64],
) -> Result<()> {
    if row_positions.len() != row_count
        || row_i_stages.len() != row_count
        || position_ids.len() != row_count
    {
        bail!(
            "SPD fixture row metadata does not match row count {row_count}: positions {}, stages {}, position_ids {}",
            row_positions.len(),
            row_i_stages.len(),
            position_ids.len()
        );
    }
    Ok(())
}

fn live_stage_ranges(args: &SpdLiveTapParityArgs) -> Result<Vec<SpdStageLayerRange>> {
    let mut bounds = Vec::with_capacity(args.splits.len() + 2);
    bounds.push(0);
    bounds.extend(args.splits.iter().copied());
    bounds.push(args.layer_end);
    for pair in bounds.windows(2) {
        if pair[0] >= pair[1] {
            bail!("--splits must partition 0..layer-end in strictly ascending order");
        }
    }
    Ok(bounds
        .windows(2)
        .enumerate()
        .map(|(stage_index, pair)| SpdStageLayerRange::new(stage_index as u32, pair[0], pair[1]))
        .collect())
}

fn collect_live_taps(
    args: &SpdLiveTapParityArgs,
    prompt_tokens: &[i32],
    ranges: &[SpdStageLayerRange],
) -> Result<BTreeMap<u32, ActivationFrame>> {
    let mut taps = BTreeMap::new();
    let h0 = run_live_stage(args, 0, 0, 0, prompt_tokens, None, false, true)
        .context("run embedding-only h0 tap")?;
    taps.insert(0, h0);

    let mut input = None;
    for range in ranges {
        let include_output = range.layer_end == args.layer_end;
        let output = run_live_stage(
            args,
            range.stage_index,
            range.layer_start,
            range.layer_end,
            prompt_tokens,
            input.as_ref(),
            include_output,
            false,
        )
        .with_context(|| {
            format!(
                "run live Skippy stage {} {}..{}",
                range.stage_index, range.layer_start, range.layer_end
            )
        })?;
        if !include_output {
            taps.insert(range.layer_end, output.clone());
            input = Some(output);
        }
    }
    Ok(taps)
}

#[allow(clippy::too_many_arguments)]
fn run_live_stage(
    args: &SpdLiveTapParityArgs,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    prompt_tokens: &[i32],
    input: Option<&ActivationFrame>,
    include_output: bool,
    embedding_only: bool,
) -> Result<ActivationFrame> {
    let config = RuntimeConfig {
        stage_index,
        layer_start,
        layer_end,
        ctx_size: args.ctx_size,
        lane_count: 1,
        n_batch: None,
        n_ubatch: None,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: args.n_gpu_layers,
        selected_backend_device: args.selected_backend_device.clone(),
        cache_type_k: GGML_TYPE_F16,
        cache_type_v: GGML_TYPE_F16,
        flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::RuntimeSlice,
        projector_path: None,
        include_embeddings: layer_start == 0 || embedding_only,
        include_output,
        filter_tensors_on_load: true,
    };
    let model = StageModel::open(&args.model_path, &config).with_context(|| {
        format!("open live Skippy stage {stage_index} {layer_start}..{layer_end}")
    })?;
    let mut session = model
        .create_session()
        .with_context(|| format!("create live Skippy stage {stage_index} session"))?;
    let positions = sequential_positions(prompt_tokens.len())?;
    session.prefill_chunk_frame_with_positions(prompt_tokens, &positions, input, 0)
}

fn sequential_positions(token_count: usize) -> Result<Vec<i32>> {
    (0..token_count)
        .map(|position| i32::try_from(position).context("prompt position exceeds i32"))
        .collect()
}

fn assemble_live_cur_in(
    manifest: &SpdHeadManifest,
    serving: &SpdSafetensorsFile,
    fixture: &SpdSafetensorsFile,
    taps: &BTreeMap<u32, ActivationFrame>,
    inputs: LiveRowInputs<'_>,
) -> Result<LiveRows> {
    let mut cur_in = Vec::with_capacity(inputs.row_count * inputs.hidden_size);
    let mut rows = Vec::with_capacity(inputs.row_count);
    let mut max_diff = 0.0_f32;
    for row_index in 0..inputs.row_count {
        let position = inputs.row_positions[row_index];
        let stage_id = u32::try_from(inputs.row_i_stages[row_index])
            .with_context(|| format!("SPD fixture row {row_index} has negative stage id"))?;
        let hf_indices = fixture_hf_indices(fixture, row_index)?;
        let concat_hidden = concat_live_hidden(taps, &hf_indices, position, inputs.hidden_size)?;
        let projection = project_spd_tap_input_row(
            &manifest.topology,
            serving,
            stage_id,
            &hf_indices,
            &concat_hidden,
        )?;
        let expected = row(inputs.fixture_cur_in, row_index, inputs.hidden_size);
        let row_diff = max_abs_diff(&projection.projected, expected)?;
        max_diff = max_diff.max(row_diff);
        cur_in.extend_from_slice(&projection.projected);
        rows.push(json!({
            "row_index": row_index,
            "position_id": position,
            "stage_id": stage_id,
            "projection_name": projection.projection_name,
            "hf_indices": projection.hf_indices,
            "max_abs_diff_vs_fixture": row_diff,
        }));
    }
    Ok(LiveRows {
        cur_in,
        rows,
        max_abs_diff: max_diff,
    })
}

fn fixture_hf_indices(fixture: &SpdSafetensorsFile, row_index: usize) -> Result<Vec<u32>> {
    fixture
        .read_tensor_i64(&format!("tap_row_{row_index}_hf_indices"))?
        .into_iter()
        .map(|value| {
            u32::try_from(value)
                .with_context(|| format!("SPD fixture row {row_index} has negative hf index"))
        })
        .collect()
}

fn concat_live_hidden(
    taps: &BTreeMap<u32, ActivationFrame>,
    hf_indices: &[u32],
    position: i64,
    hidden_size: usize,
) -> Result<Vec<f32>> {
    let mut concat = Vec::with_capacity(hf_indices.len() * hidden_size);
    for hf_index in hf_indices {
        let frame = taps
            .get(hf_index)
            .with_context(|| format!("missing live Skippy tap for HF hidden-state {hf_index}"))?;
        concat.extend_from_slice(&live_hidden_row(frame, position, hidden_size)?);
    }
    Ok(concat)
}

fn live_hidden_row(frame: &ActivationFrame, position: i64, hidden_size: usize) -> Result<Vec<f32>> {
    let position = usize::try_from(position).context("negative live tap position")?;
    let token_count =
        usize::try_from(frame.desc.token_count).context("token count exceeds usize")?;
    if position >= token_count {
        bail!("live tap position {position} is outside token_count {token_count}");
    }
    let payload_f32 = activation_payload_f32(frame)?;
    Ok(row(&payload_f32, position, hidden_size).to_vec())
}

fn activation_payload_f32(frame: &ActivationFrame) -> Result<Vec<f32>> {
    if !frame.payload.len().is_multiple_of(4) {
        bail!(
            "live activation payload for {}..{} has non-f32 byte length {}",
            frame.desc.layer_start,
            frame.desc.layer_end,
            frame.payload.len()
        );
    }
    Ok(frame
        .payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn row(values: &[f32], row_idx: usize, width: usize) -> &[f32] {
    &values[row_idx * width..(row_idx + 1) * width]
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        bail!("vector length mismatch: {} vs {}", left.len(), right.len());
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max))
}

fn live_tap_report(taps: &BTreeMap<u32, ActivationFrame>) -> Vec<serde_json::Value> {
    taps.iter()
        .map(|(hf_index, frame)| {
            json!({
                "hf_index": hf_index,
                "producer_stage_index": frame.desc.producer_stage_index,
                "layer_start": frame.desc.layer_start,
                "layer_end": frame.desc.layer_end,
                "token_count": frame.desc.token_count,
                "payload_bytes": frame.desc.payload_bytes,
                "actual_payload_bytes": frame.payload.len(),
            })
        })
        .collect()
}
