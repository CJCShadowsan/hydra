use std::{collections::BTreeMap, fs, time::Instant};

use anyhow::{Context, Result, bail};
use serde_json::json;
use skippy_runtime::{
    ActivationFrame, GGML_TYPE_F16, RuntimeConfig, RuntimeLoadMode, StageModel,
    spd::{
        SpdHeadManifest, SpdQwen3ForwardInput, SpdQwen3ForwardTiming, SpdQwen3Head,
        SpdSafetensorsFile, SpdStageLayerRange, SpdTapInputProjector, plan_hidden_state_taps,
        run_qwen3_cached_fixture_parity, run_qwen3_fixture_parity,
        run_spd_tap_input_fixture_parity, spd_fixture_row_hf_indices,
    },
};

use crate::cli::{SpdFixtureParityArgs, SpdLiveTapParityArgs};

pub fn spd_fixture_parity(args: SpdFixtureParityArgs) -> Result<()> {
    let tap_input = run_spd_tap_input_fixture_parity(&args.manifest, &args.fixture)
        .context("failed to reconstruct SPD fixture cur_in from tap inputs")?;
    let forward = run_qwen3_fixture_parity(&args.manifest, &args.fixture, args.top_k)
        .context("failed to run Qwen3 SPD fixture forward parity")?;
    let cached_forward = run_qwen3_cached_fixture_parity(&args.manifest, &args.fixture, args.top_k)
        .context("failed to run cached Qwen3 SPD fixture forward parity")?;
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
        },
        "cached_forward": cached_forward.map(|cached| {
            json!({
                "rust": {
                    "draft_indices": cached.rust.draft_indices,
                    "token_ids": cached.rust.token_ids,
                    "logits": cached.rust.logits,
                },
                "python": {
                    "draft_indices": cached.python.draft_indices,
                    "token_ids": cached.python.token_ids,
                    "logits": cached.python.logits,
                },
                "diagnostics": {
                    "cache_prefix_len": cached.diagnostics.cache_prefix_len,
                    "spec_query_max_abs_diff": cached.diagnostics.spec_query_max_abs_diff,
                    "final_hidden_max_abs_diff": cached.diagnostics.final_hidden_max_abs_diff,
                    "logits_max_abs_diff": cached.diagnostics.logits_max_abs_diff,
                    "python_top_logit_values_at_rust_indices": cached.diagnostics.python_top_logit_values_at_rust_indices,
                }
            })
        })
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
    let row_hf_indices = spd_fixture_row_hf_indices(&fixture, row_count)?;
    let tap_projector = SpdTapInputProjector::from_rows(
        &manifest.topology,
        &serving,
        &row_i_stages,
        &row_hf_indices,
    )
    .context("load SPD tap projection weights")?;

    let ranges = live_stage_ranges(&args)?;
    let tap_plan = plan_hidden_state_taps(&manifest.topology, &ranges)?;
    if tap_plan.requires_internal_taps() {
        bail!(
            "live SPD tap parity requires boundary-aligned splits; missing hidden states {:?}",
            tap_plan.boundary_only_missing_hf_indices
        );
    }

    if args.verify_steps == 0 {
        bail!("--verify-steps must be greater than zero");
    }

    let hidden_size =
        usize::try_from(manifest.topology.hidden_size).context("SPD hidden_size exceeds usize")?;
    let spd_head = SpdQwen3Head::open(&args.manifest).context("open Qwen SPD head")?;
    let live_runner = LiveTapRunner::open(&args, &ranges)?;
    let taps = live_runner.collect_taps(&prompt_tokens)?;
    let live_rows = assemble_live_cur_in(
        &tap_projector,
        &taps,
        LiveRowInputs {
            row_count,
            row_positions: &row_positions,
            row_i_stages: &row_i_stages,
            row_hf_indices: &row_hf_indices,
            fixture_cur_in: &fixture_cur_in,
            hidden_size,
        },
    )?;
    let live_topk = spd_head.forward(
        SpdQwen3ForwardInput {
            cur_in: live_rows.cur_in.clone(),
            seq_len: row_count,
            position_ids,
            fixed_stage_ids: None,
            final_norm_weight: final_norm_weight.clone(),
        },
        args.top_k,
    )?;
    let fixture_forward = run_qwen3_fixture_parity(&args.manifest, &args.fixture, args.top_k)?;
    let verified_generation = run_verified_generation(
        &args,
        &live_runner,
        &spd_head,
        &tap_projector,
        VerifiedGenerationInputs {
            prompt_tokens: &prompt_tokens,
            row_i_stages: &row_i_stages,
            row_hf_indices: &row_hf_indices,
            final_norm_weight: &final_norm_weight,
            row_count,
            hidden_size,
        },
    )
    .context("run repeated live SPD target verification")?;
    let target_verification = verified_generation
        .first_step
        .clone()
        .context("verified SPD generation produced no steps")?;
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
        },
        "target_verification": target_verification,
        "verified_generation": verified_generation.report,
    });
    let json = serde_json::to_vec_pretty(&report)?;
    if let Some(output) = args.output {
        fs::write(&output, &json)
            .with_context(|| format!("failed to write {}", output.display()))?;
    }
    println!("{}", String::from_utf8(json)?);
    Ok(())
}

struct VerifiedGenerationReport {
    first_step: Option<serde_json::Value>,
    report: serde_json::Value,
}

struct VerifiedGenerationInputs<'a> {
    prompt_tokens: &'a [i32],
    row_i_stages: &'a [i64],
    row_hf_indices: &'a [Vec<u32>],
    final_norm_weight: &'a [f32],
    row_count: usize,
    hidden_size: usize,
}

fn run_verified_generation(
    args: &SpdLiveTapParityArgs,
    live_runner: &LiveTapRunner,
    spd_head: &SpdQwen3Head,
    tap_projector: &SpdTapInputProjector,
    inputs: VerifiedGenerationInputs<'_>,
) -> Result<VerifiedGenerationReport> {
    if inputs.prompt_tokens.len() < inputs.row_count {
        bail!(
            "SPD verified generation prompt length {} is shorter than row count {}",
            inputs.prompt_tokens.len(),
            inputs.row_count
        );
    }
    let prefix = inputs
        .prompt_tokens
        .get(..inputs.prompt_tokens.len().saturating_sub(1))
        .context("failed to split prompt prefix")?;
    let target = open_full_target_model(args)?;
    let mut target_session = target
        .create_session()
        .context("create target verification session")?;
    prefill_target_prefix(&mut target_session, prefix)?;

    let total_timer = Instant::now();
    let mut context_tokens = inputs.prompt_tokens.to_vec();
    let mut steps = Vec::with_capacity(args.verify_steps);
    let mut accepted_count = 0usize;
    let mut rejected_count = 0usize;
    for step_index in 0..args.verify_steps {
        let step_timer = Instant::now();
        let current = *context_tokens
            .last()
            .context("verified SPD generation context is empty")?;
        let row_positions = sliding_row_positions(context_tokens.len(), inputs.row_count)?;

        let tap_timer = Instant::now();
        let taps = live_runner.collect_taps(&context_tokens)?;
        let tap_ms = elapsed_ms(tap_timer);

        let assemble_timer = Instant::now();
        let live_rows = assemble_live_cur_in_for_positions(
            tap_projector,
            &taps,
            DynamicLiveRowInputs {
                row_positions: &row_positions,
                row_i_stages: inputs.row_i_stages,
                row_hf_indices: inputs.row_hf_indices,
                hidden_size: inputs.hidden_size,
            },
        )?;
        let assemble_ms = elapsed_ms(assemble_timer);

        let head_timer = Instant::now();
        let live_forward = spd_head.forward_timed(
            SpdQwen3ForwardInput {
                cur_in: live_rows.cur_in,
                seq_len: inputs.row_count,
                position_ids: row_positions.clone(),
                fixed_stage_ids: None,
                final_norm_weight: inputs.final_norm_weight.to_vec(),
            },
            args.top_k,
        )?;
        let head_ms = elapsed_ms(head_timer);
        let live_topk = live_forward.topk;
        let proposal_token = live_topk
            .token_ids
            .first()
            .copied()
            .context("live SPD head returned no proposal tokens")
            .and_then(|token| {
                i32::try_from(token).context("live SPD proposal token exceeds i32")
            })?;

        let verify_inputs = vec![current];
        let verifier_token_count_before = target_session.token_count();
        let verify_timer = Instant::now();
        let predicted_tokens = target_session
            .verify_tokens_rewound(&verify_inputs)
            .context("target verifier rejected SPD proposal window")?;
        let verify_ms = elapsed_ms(verify_timer);
        let verifier_token_count_after_rewind = target_session.token_count();
        let target_token = *predicted_tokens
            .first()
            .context("target verifier returned no predicted token")?;
        let accepted = target_token == proposal_token;
        accepted_count += usize::from(accepted);
        rejected_count += usize::from(!accepted);
        let committed_token = if accepted {
            proposal_token
        } else {
            target_token
        };

        let decode_timer = Instant::now();
        let baseline_token = target_session
            .decode_step(current)
            .context("target greedy baseline decode failed")?;
        let decode_ms = elapsed_ms(decode_timer);
        let baseline_token_count_after = target_session.token_count();
        context_tokens.push(committed_token);

        steps.push(json!({
            "step_index": step_index,
            "context_token_count_before": context_tokens.len() - 1,
            "current_token": current,
            "row_positions": row_positions,
            "row_stage_ids": inputs.row_i_stages,
            "proposal_source": "live_skippy_top1",
            "proposal_tokens": [proposal_token],
            "proposal_top_k": {
                "draft_indices": live_topk.draft_indices,
                "token_ids": live_topk.token_ids,
                "logits": live_topk.logits,
            },
            "verify_inputs": verify_inputs,
            "target_predicted_tokens": predicted_tokens,
            "accepted": accepted,
            "accepted_count": usize::from(accepted),
            "rejected_count": usize::from(!accepted),
            "committed_tokens": [committed_token],
            "baseline_greedy_token": baseline_token,
            "baseline_matches_verifier": baseline_token == target_token,
            "greedy_output_matches_non_spd": committed_token == baseline_token,
            "verifier_rewound": verifier_token_count_after_rewind == verifier_token_count_before,
            "verifier_token_count_before": verifier_token_count_before,
            "verifier_token_count_after_rewind": verifier_token_count_after_rewind,
            "baseline_token_count_after": baseline_token_count_after,
            "timing_ms": {
                "tap_replay": tap_ms,
                "assemble_cur_in": assemble_ms,
                "spd_head": head_ms,
                "spd_head_detail": qwen_forward_timing_report(&live_forward.timing),
                "target_verify_rewound": verify_ms,
                "target_greedy_decode": decode_ms,
                "total": elapsed_ms(step_timer),
            },
        }));
    }

    let generated_tokens = context_tokens[inputs.prompt_tokens.len()..].to_vec();
    let all_match = steps
        .iter()
        .all(|step| step["greedy_output_matches_non_spd"].as_bool() == Some(true));
    let all_rewound = steps
        .iter()
        .all(|step| step["verifier_rewound"].as_bool() == Some(true));
    let report = json!({
        "steps_requested": args.verify_steps,
        "steps_completed": steps.len(),
        "generated_tokens": generated_tokens,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "acceptance_rate": if steps.is_empty() {
            0.0
        } else {
            accepted_count as f64 / steps.len() as f64
        },
        "top1_acceptance_rate": if steps.is_empty() {
            0.0
        } else {
            accepted_count as f64 / steps.len() as f64
        },
        "greedy_output_matches_non_spd": all_match,
        "all_verifier_windows_rewound": all_rewound,
        "total_elapsed_ms": elapsed_ms(total_timer),
        "steps": steps,
    });
    let first_step = report["steps"]
        .as_array()
        .and_then(|steps| steps.first())
        .cloned();
    Ok(VerifiedGenerationReport { first_step, report })
}

fn qwen_forward_timing_report(timing: &SpdQwen3ForwardTiming) -> serde_json::Value {
    json!({
        "fixed_stage_projection": timing.fixed_stage_projection_ms,
        "decoder_layers": timing.decoder_layer_ms,
        "final_norm": timing.final_norm_ms,
        "lm_head_topk": timing.lm_head_topk_ms,
        "total": timing.total_ms,
    })
}

fn open_full_target_model(args: &SpdLiveTapParityArgs) -> Result<StageModel> {
    let config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.layer_end,
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
        include_embeddings: true,
        include_output: true,
        filter_tensors_on_load: false,
    };
    StageModel::open(&args.model_path, &config).with_context(|| {
        format!(
            "open full target model {} for SPD verification",
            args.model_path.display()
        )
    })
}

fn prefill_target_prefix(
    session: &mut skippy_runtime::StageSession,
    prefix_tokens: &[i32],
) -> Result<()> {
    if !prefix_tokens.is_empty() {
        session
            .prefill_chunked(prefix_tokens)
            .context("prefill target verifier prefix")?;
    }
    Ok(())
}

struct LiveRows {
    cur_in: Vec<f32>,
    rows: Vec<serde_json::Value>,
    max_abs_diff: f32,
}

struct DynamicLiveRows {
    cur_in: Vec<f32>,
}

struct LiveRowInputs<'a> {
    row_count: usize,
    row_positions: &'a [i64],
    row_i_stages: &'a [i64],
    row_hf_indices: &'a [Vec<u32>],
    fixture_cur_in: &'a [f32],
    hidden_size: usize,
}

struct DynamicLiveRowInputs<'a> {
    row_positions: &'a [i64],
    row_i_stages: &'a [i64],
    row_hf_indices: &'a [Vec<u32>],
    hidden_size: usize,
}

struct LiveTapRunner {
    h0: StageModel,
    stages: Vec<LiveStage>,
}

struct LiveStage {
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    include_output: bool,
    model: StageModel,
}

impl LiveTapRunner {
    fn open(args: &SpdLiveTapParityArgs, ranges: &[SpdStageLayerRange]) -> Result<Self> {
        let h0 = open_live_stage_model(args, 0, 0, 0, false, true)
            .context("open embedding-only h0 tap stage")?;
        let stages = ranges
            .iter()
            .map(|range| {
                // Live-tap parity needs the final boundary hidden state too.
                // Target logits are verified with the separate full target model.
                let include_output = false;
                let model = open_live_stage_model(
                    args,
                    range.stage_index,
                    range.layer_start,
                    range.layer_end,
                    include_output,
                    false,
                )?;
                Ok(LiveStage {
                    stage_index: range.stage_index,
                    layer_start: range.layer_start,
                    layer_end: range.layer_end,
                    include_output,
                    model,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { h0, stages })
    }

    fn collect_taps(&self, prompt_tokens: &[i32]) -> Result<BTreeMap<u32, ActivationFrame>> {
        let mut taps = BTreeMap::new();
        let h0 = run_live_stage_model(&self.h0, 0, 0, 0, prompt_tokens, None)
            .context("run embedding-only h0 tap")?;
        taps.insert(0, h0);

        let mut input = None;
        for stage in &self.stages {
            let output = run_live_stage_model(
                &stage.model,
                stage.stage_index,
                stage.layer_start,
                stage.layer_end,
                prompt_tokens,
                input.as_ref(),
            )
            .with_context(|| {
                format!(
                    "run live Skippy stage {} {}..{}",
                    stage.stage_index, stage.layer_start, stage.layer_end
                )
            })?;
            if !stage.include_output {
                taps.insert(stage.layer_end, output.clone());
                input = Some(output);
            }
        }
        Ok(taps)
    }
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

fn open_live_stage_model(
    args: &SpdLiveTapParityArgs,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    include_output: bool,
    embedding_only: bool,
) -> Result<StageModel> {
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
    StageModel::open(&args.model_path, &config)
        .with_context(|| format!("open live Skippy stage {stage_index} {layer_start}..{layer_end}"))
}

fn run_live_stage_model(
    model: &StageModel,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    prompt_tokens: &[i32],
    input: Option<&ActivationFrame>,
) -> Result<ActivationFrame> {
    let mut session = model.create_session().with_context(|| {
        format!("create live Skippy stage {stage_index} {layer_start}..{layer_end} session")
    })?;
    let positions = sequential_positions(prompt_tokens.len())?;
    session.prefill_chunk_frame_with_positions(prompt_tokens, &positions, input, 0)
}

fn sequential_positions(token_count: usize) -> Result<Vec<i32>> {
    (0..token_count)
        .map(|position| i32::try_from(position).context("prompt position exceeds i32"))
        .collect()
}

fn assemble_live_cur_in(
    tap_projector: &SpdTapInputProjector,
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
        let hf_indices = &inputs.row_hf_indices[row_index];
        let concat_hidden = concat_live_hidden(taps, hf_indices, position, inputs.hidden_size)?;
        let projection = tap_projector.project(stage_id, hf_indices, &concat_hidden)?;
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

fn assemble_live_cur_in_for_positions(
    tap_projector: &SpdTapInputProjector,
    taps: &BTreeMap<u32, ActivationFrame>,
    inputs: DynamicLiveRowInputs<'_>,
) -> Result<DynamicLiveRows> {
    if inputs.row_positions.len() != inputs.row_i_stages.len()
        || inputs.row_positions.len() != inputs.row_hf_indices.len()
    {
        bail!(
            "dynamic SPD row metadata length mismatch: positions {}, stages {}, hf rows {}",
            inputs.row_positions.len(),
            inputs.row_i_stages.len(),
            inputs.row_hf_indices.len()
        );
    }
    let mut cur_in = Vec::with_capacity(inputs.row_positions.len() * inputs.hidden_size);
    for row_index in 0..inputs.row_positions.len() {
        let position = inputs.row_positions[row_index];
        let stage_id = u32::try_from(inputs.row_i_stages[row_index])
            .with_context(|| format!("SPD dynamic row {row_index} has negative stage id"))?;
        let hf_indices = &inputs.row_hf_indices[row_index];
        let concat_hidden = concat_live_hidden(taps, hf_indices, position, inputs.hidden_size)?;
        let projection = tap_projector.project(stage_id, hf_indices, &concat_hidden)?;
        cur_in.extend_from_slice(&projection.projected);
    }
    Ok(DynamicLiveRows { cur_in })
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
    let row_bytes = hidden_size
        .checked_mul(std::mem::size_of::<f32>())
        .context("live activation row byte width overflow")?;
    let expected_payload_bytes = token_count
        .checked_mul(row_bytes)
        .context("live activation payload byte count overflow")?;
    if frame.payload.len() != expected_payload_bytes {
        bail!(
            "live activation payload for {}..{} has {} bytes, expected {} for {} tokens x hidden {}",
            frame.desc.layer_start,
            frame.desc.layer_end,
            frame.payload.len(),
            expected_payload_bytes,
            token_count,
            hidden_size
        );
    }
    let offset = position
        .checked_mul(row_bytes)
        .context("live activation row offset overflow")?;
    Ok(frame.payload[offset..offset + row_bytes]
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

fn sliding_row_positions(context_len: usize, row_count: usize) -> Result<Vec<i64>> {
    if context_len < row_count {
        bail!("context length {context_len} is shorter than SPD row count {row_count}");
    }
    let start = context_len - row_count;
    (start..context_len)
        .map(|position| i64::try_from(position).context("SPD row position exceeds i64"))
        .collect()
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}
