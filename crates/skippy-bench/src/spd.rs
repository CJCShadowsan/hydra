use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use serde_json::json;
use skippy_runtime::{
    ActivationFrame, GGML_TYPE_F16, RuntimeConfig, RuntimeLoadMode, StageModel,
    package::{PackageStageRequest, inspect_layer_package, select_layer_package_parts},
    spd::{
        SpdHeadManifest, SpdLiveTapModelSource, SpdLiveTapRunner, SpdLiveTapRunnerConfig,
        SpdQwen3FixtureTopK, SpdQwen3ForwardInput, SpdQwen3ForwardTiming, SpdQwen3Head,
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
    let fixture_concat_rows = read_fixture_concat_rows(&fixture, row_count)?;
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
    let live_runner = open_live_tap_runner(&args, &manifest, &ranges)?;
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
            fixture_concat_rows: &fixture_concat_rows,
            final_norm_weight: &final_norm_weight,
            layer_end: args.layer_end,
            hidden_size,
        },
    )?;
    let live_topk = spd_head.forward(
        SpdQwen3ForwardInput {
            cur_in: live_rows.cur_in.clone(),
            seq_len: row_count,
            position_ids: position_ids.clone(),
            fixed_stage_ids: None,
            final_norm_weight: final_norm_weight.clone(),
        },
        args.top_k,
    )?;
    let live_terminal_normed_topk = spd_head.forward(
        SpdQwen3ForwardInput {
            cur_in: live_rows.terminal_normed_cur_in.clone(),
            seq_len: row_count,
            position_ids: position_ids.clone(),
            fixed_stage_ids: None,
            final_norm_weight: final_norm_weight.clone(),
        },
        args.top_k,
    )?;
    let fixture_forward = run_qwen3_fixture_parity(&args.manifest, &args.fixture, args.top_k)?;
    let parity_gate = evaluate_live_tap_parity_gate(
        &args,
        live_rows.max_abs_diff,
        &live_topk,
        &fixture_forward.rust,
    )?;
    let terminal_normed_parity_gate = evaluate_live_tap_parity_gate(
        &args,
        live_rows.terminal_normed_max_abs_diff,
        &live_terminal_normed_topk,
        &fixture_forward.rust,
    )?;
    let mut product_corpus = match &args.product_corpus_dir {
        Some(dir) => Some(ProductActivationCorpusWriter::create(
            ProductActivationCorpusConfig {
                dir: dir.clone(),
                args: &args,
                manifest: &manifest,
                prompt_tokens: &prompt_tokens,
                row_count,
                row_i_stages: &row_i_stages,
                row_hf_indices: &row_hf_indices,
                hidden_size,
                final_norm_weight: &final_norm_weight,
            },
        )?),
        None => None,
    };
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
        product_corpus.as_mut(),
    )
    .context("run repeated live SPD target verification")?;
    let product_corpus_report = product_corpus
        .as_mut()
        .map(ProductActivationCorpusWriter::finish)
        .transpose()?;
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
        "gated_path": "terminal_final_normed_cur_in",
        "parity_gate": parity_gate.report,
        "terminal_final_normed_parity_gate": terminal_normed_parity_gate.report,
        "live_taps": live_tap_report(&taps),
        "cur_in": {
            "max_abs_diff_vs_fixture": live_rows.max_abs_diff,
            "rows": live_rows.rows,
        },
        "terminal_final_normed_cur_in": {
            "max_abs_diff_vs_fixture": live_rows.terminal_normed_max_abs_diff,
            "rows": live_rows.terminal_normed_rows,
        },
        "forward": {
            "live_skippy": {
                "draft_indices": live_topk.draft_indices,
                "token_ids": live_topk.token_ids,
                "logits": live_topk.logits,
            },
            "live_skippy_terminal_final_normed": {
                "draft_indices": live_terminal_normed_topk.draft_indices,
                "token_ids": live_terminal_normed_topk.token_ids,
                "logits": live_terminal_normed_topk.logits,
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
        "product_activation_corpus": product_corpus_report,
    });
    let json = serde_json::to_vec_pretty(&report)?;
    if let Some(output) = args.output {
        fs::write(&output, &json)
            .with_context(|| format!("failed to write {}", output.display()))?;
    }
    println!("{}", String::from_utf8(json)?);
    terminal_normed_parity_gate.ensure_passed()
}

struct VerifiedGenerationReport {
    first_step: Option<serde_json::Value>,
    report: serde_json::Value,
}

struct LiveTapParityGate {
    report: serde_json::Value,
    failures: Vec<String>,
}

impl LiveTapParityGate {
    fn ensure_passed(&self) -> Result<()> {
        if self.failures.is_empty() {
            return Ok(());
        }
        bail!("SPD live tap parity failed: {}", self.failures.join("; "))
    }
}

fn evaluate_live_tap_parity_gate(
    args: &SpdLiveTapParityArgs,
    cur_in_max_abs_diff: f32,
    live_topk: &SpdQwen3FixtureTopK,
    fixture_topk: &SpdQwen3FixtureTopK,
) -> Result<LiveTapParityGate> {
    let rank_logit_max_abs_diff = max_abs_diff(&live_topk.logits, &fixture_topk.logits)
        .context("compare live SPD logits with fixture logits")?;
    let cur_in_within_tol = cur_in_max_abs_diff <= args.cur_in_tol;
    let logits_within_tol = rank_logit_max_abs_diff <= args.logits_tol;
    let draft_indices_match = live_topk.draft_indices == fixture_topk.draft_indices;
    let token_ids_match = live_topk.token_ids == fixture_topk.token_ids;
    let mut failures = Vec::new();
    if !cur_in_within_tol {
        failures.push(format!(
            "cur_in max_abs_diff {cur_in_max_abs_diff} exceeds --cur-in-tol {}",
            args.cur_in_tol
        ));
    }
    if !logits_within_tol {
        failures.push(format!(
            "rank-paired logit max_abs_diff {rank_logit_max_abs_diff} exceeds --logits-tol {}",
            args.logits_tol
        ));
    }
    if !draft_indices_match {
        failures.push("live draft_indices differ from fixture Rust draft_indices".to_string());
    }
    if !token_ids_match {
        failures.push("live token_ids differ from fixture Rust token_ids".to_string());
    }
    let report = json!({
        "cur_in_tol": args.cur_in_tol,
        "logits_tol": args.logits_tol,
        "cur_in_max_abs_diff": cur_in_max_abs_diff,
        "rank_logit_max_abs_diff_vs_fixture_rust": rank_logit_max_abs_diff,
        "cur_in_within_tol": cur_in_within_tol,
        "rank_logits_within_tol": logits_within_tol,
        "topk_draft_indices_match_fixture_rust": draft_indices_match,
        "topk_token_ids_match_fixture_rust": token_ids_match,
        "passed": failures.is_empty(),
        "failures": &failures,
    });
    Ok(LiveTapParityGate { report, failures })
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
    live_runner: &SpdLiveTapRunner,
    spd_head: &SpdQwen3Head,
    tap_projector: &SpdTapInputProjector,
    inputs: VerifiedGenerationInputs<'_>,
    mut product_corpus: Option<&mut ProductActivationCorpusWriter>,
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
                final_norm_weight: inputs.final_norm_weight,
                layer_end: args.layer_end,
                hidden_size: inputs.hidden_size,
            },
        )?;
        let assemble_ms = elapsed_ms(assemble_timer);

        let head_timer = Instant::now();
        let cur_in = live_rows.cur_in;
        let live_forward = spd_head.forward_timed(
            SpdQwen3ForwardInput {
                cur_in: cur_in.clone(),
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
        if let Some(writer) = product_corpus.as_deref_mut() {
            writer.write_step(ProductActivationSample {
                step_index,
                context_tokens: &context_tokens,
                row_positions: &row_positions,
                row_stage_ids: inputs.row_i_stages,
                row_hf_indices: inputs.row_hf_indices,
                query_row_index: inputs.row_count.saturating_sub(1),
                target_position: context_tokens.len(),
                cur_in: &cur_in,
                current_token: current,
                proposal_topk: &live_topk,
                target_token,
                accepted,
                committed_token,
                baseline_greedy_token: baseline_token,
            })?;
        }
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
        load_mode: live_tap_load_mode(&args.model_path),
        projector_path: None,
        include_embeddings: true,
        include_output: true,
        filter_tensors_on_load: false,
    };
    if live_tap_model_path_is_layer_package(&args.model_path) {
        let package_ref = args.model_path.to_string_lossy().to_string();
        let package_info = inspect_layer_package(&package_ref).with_context(|| {
            format!("inspect SPD live tap package {}", args.model_path.display())
        })?;
        let parts = select_layer_package_parts(&PackageStageRequest {
            model_id: package_info.model_id,
            topology_id: "spd-live-tap-parity".to_string(),
            package_ref,
            stage_id: "spd-live-tap-target".to_string(),
            layer_start: 0,
            layer_end: args.layer_end,
            include_embeddings: true,
            include_output: true,
        })
        .context("select SPD live tap full-target package parts")?;
        return StageModel::open_from_parts(&parts.absolute_paths, &config).with_context(|| {
            format!(
                "open full target layer package {} for SPD verification",
                args.model_path.display()
            )
        });
    }
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
    terminal_normed_cur_in: Vec<f32>,
    terminal_normed_rows: Vec<serde_json::Value>,
    terminal_normed_max_abs_diff: f32,
}

struct DynamicLiveRows {
    cur_in: Vec<f32>,
}

struct ProductActivationCorpusConfig<'a> {
    dir: PathBuf,
    args: &'a SpdLiveTapParityArgs,
    manifest: &'a SpdHeadManifest,
    prompt_tokens: &'a [i32],
    row_count: usize,
    row_i_stages: &'a [i64],
    row_hf_indices: &'a [Vec<u32>],
    hidden_size: usize,
    final_norm_weight: &'a [f32],
}

struct ProductActivationCorpusWriter {
    dir: PathBuf,
    rows_f32: BufWriter<File>,
    rows_jsonl: BufWriter<File>,
    row_count: usize,
    hidden_size: usize,
    sample_count: usize,
}

struct ProductActivationSample<'a> {
    step_index: usize,
    context_tokens: &'a [i32],
    row_positions: &'a [i64],
    row_stage_ids: &'a [i64],
    row_hf_indices: &'a [Vec<u32>],
    query_row_index: usize,
    target_position: usize,
    cur_in: &'a [f32],
    current_token: i32,
    proposal_topk: &'a SpdQwen3FixtureTopK,
    target_token: i32,
    accepted: bool,
    committed_token: i32,
    baseline_greedy_token: i32,
}

struct LiveRowInputs<'a> {
    row_count: usize,
    row_positions: &'a [i64],
    row_i_stages: &'a [i64],
    row_hf_indices: &'a [Vec<u32>],
    fixture_cur_in: &'a [f32],
    fixture_concat_rows: &'a [Vec<f32>],
    final_norm_weight: &'a [f32],
    layer_end: u32,
    hidden_size: usize,
}

struct DynamicLiveRowInputs<'a> {
    row_positions: &'a [i64],
    row_i_stages: &'a [i64],
    row_hf_indices: &'a [Vec<u32>],
    final_norm_weight: &'a [f32],
    layer_end: u32,
    hidden_size: usize,
}

impl ProductActivationCorpusWriter {
    fn create(config: ProductActivationCorpusConfig<'_>) -> Result<Self> {
        fs::create_dir_all(&config.dir)
            .with_context(|| format!("create SPD product corpus dir {}", config.dir.display()))?;
        write_f32_file(
            &config.dir.join("final_norm_weight.f32"),
            config.final_norm_weight,
        )?;
        let manifest = json!({
            "schema": "skippy-spd-product-activation-corpus/v1",
            "producer": "skippy-bench spd-live-tap-parity",
            "manifest_path": config.args.manifest.display().to_string(),
            "fixture_path": config.args.fixture.display().to_string(),
            "model_path": config.args.model_path.display().to_string(),
            "splits": &config.args.splits,
            "layer_end": config.args.layer_end,
            "ctx_size": config.args.ctx_size,
            "n_gpu_layers": config.args.n_gpu_layers,
            "selected_backend_device": &config.args.selected_backend_device,
            "top_k": config.args.top_k,
            "verify_steps": config.args.verify_steps,
            "prompt_tokens": config.prompt_tokens,
            "topology": &config.manifest.topology,
            "row_count": config.row_count,
            "hidden_size": config.hidden_size,
            "row_stage_ids": config.row_i_stages,
            "row_hf_indices": config.row_hf_indices,
            "query_row_index": config.row_count.saturating_sub(1),
            "cur_in_convention": "terminal_final_normed_cur_in",
            "row_tensor": {
                "path": "rows.f32",
                "dtype": "f32_le",
                "shape": ["sample_count", config.row_count, config.hidden_size],
            },
            "metadata_rows": {
                "path": "rows.jsonl",
                "schema": "skippy-spd-product-activation-row/v1",
            },
            "final_norm_weight": {
                "path": "final_norm_weight.f32",
                "dtype": "f32_le",
                "shape": [config.hidden_size],
            },
            "label_kind": "target_greedy_top1",
            "target_logits_available": false,
            "paper_kl_training_ready": false,
            "notes": [
                "Rows are captured from live Skippy/product activations after terminal final-norm alignment.",
                "Labels are greedy target tokens from the product verifier; full teacher logits are not exposed by the current native runtime ABI.",
            ],
        });
        fs::write(
            config.dir.join("manifest.json"),
            format!("{}\n", serde_json::to_string_pretty(&manifest)?),
        )
        .with_context(|| format!("write {}", config.dir.join("manifest.json").display()))?;
        let rows_f32 = BufWriter::new(
            File::create(config.dir.join("rows.f32"))
                .with_context(|| format!("create {}", config.dir.join("rows.f32").display()))?,
        );
        let rows_jsonl = BufWriter::new(
            File::create(config.dir.join("rows.jsonl"))
                .with_context(|| format!("create {}", config.dir.join("rows.jsonl").display()))?,
        );
        Ok(Self {
            dir: config.dir,
            rows_f32,
            rows_jsonl,
            row_count: config.row_count,
            hidden_size: config.hidden_size,
            sample_count: 0,
        })
    }

    fn write_step(&mut self, sample: ProductActivationSample<'_>) -> Result<()> {
        let expected_len = self
            .row_count
            .checked_mul(self.hidden_size)
            .context("SPD product corpus row length overflow")?;
        if sample.cur_in.len() != expected_len {
            bail!(
                "SPD product corpus cur_in length {} does not match row_count {} * hidden_size {}",
                sample.cur_in.len(),
                self.row_count,
                self.hidden_size
            );
        }
        write_f32_slice(&mut self.rows_f32, sample.cur_in)?;
        let row = json!({
            "schema": "skippy-spd-product-activation-row/v1",
            "sample_index": self.sample_count,
            "row_f32_offset": self.sample_count * expected_len,
            "row_f32_count": expected_len,
            "step_index": sample.step_index,
            "context_token_count_before": sample.context_tokens.len(),
            "context_tokens": sample.context_tokens,
            "row_positions": sample.row_positions,
            "position_ids": sample.row_positions,
            "row_stage_ids": sample.row_stage_ids,
            "row_hf_indices": sample.row_hf_indices,
            "query_row_index": sample.query_row_index,
            "query_position": sample.row_positions.get(sample.query_row_index).copied(),
            "target_position": sample.target_position,
            "current_token": sample.current_token,
            "proposal_top_k": {
                "draft_indices": &sample.proposal_topk.draft_indices,
                "token_ids": &sample.proposal_topk.token_ids,
                "logits": &sample.proposal_topk.logits,
            },
            "target_token": sample.target_token,
            "accepted": sample.accepted,
            "committed_token": sample.committed_token,
            "baseline_greedy_token": sample.baseline_greedy_token,
            "greedy_output_matches_non_spd": sample.committed_token == sample.baseline_greedy_token,
        });
        serde_json::to_writer(&mut self.rows_jsonl, &row)
            .context("write SPD product corpus JSONL row")?;
        self.rows_jsonl
            .write_all(b"\n")
            .context("terminate SPD product corpus JSONL row")?;
        self.sample_count += 1;
        Ok(())
    }

    fn finish(&mut self) -> Result<serde_json::Value> {
        self.rows_f32
            .flush()
            .context("flush SPD product corpus rows.f32")?;
        self.rows_jsonl
            .flush()
            .context("flush SPD product corpus rows.jsonl")?;
        let rows_bytes = fs::metadata(self.dir.join("rows.f32"))
            .with_context(|| format!("stat {}", self.dir.join("rows.f32").display()))?
            .len();
        let summary = json!({
            "schema": "skippy-spd-product-activation-corpus-summary/v1",
            "dir": self.dir.display().to_string(),
            "sample_count": self.sample_count,
            "row_count": self.row_count,
            "hidden_size": self.hidden_size,
            "rows_f32_bytes": rows_bytes,
            "rows_f32_expected_bytes": self.sample_count * self.row_count * self.hidden_size * std::mem::size_of::<f32>(),
            "rows_path": "rows.f32",
            "metadata_path": "rows.jsonl",
            "manifest_path": "manifest.json",
        });
        fs::write(
            self.dir.join("summary.json"),
            format!("{}\n", serde_json::to_string_pretty(&summary)?),
        )
        .with_context(|| format!("write {}", self.dir.join("summary.json").display()))?;
        Ok(summary)
    }
}

fn write_f32_file(path: &Path, values: &[f32]) -> Result<()> {
    let mut writer =
        BufWriter::new(File::create(path).with_context(|| format!("create {}", path.display()))?);
    write_f32_slice(&mut writer, values)?;
    writer
        .flush()
        .with_context(|| format!("flush {}", path.display()))
}

fn write_f32_slice(writer: &mut impl Write, values: &[f32]) -> Result<()> {
    for value in values {
        writer
            .write_all(&value.to_le_bytes())
            .context("write f32 value")?;
    }
    Ok(())
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

fn read_fixture_concat_rows(
    fixture: &SpdSafetensorsFile,
    row_count: usize,
) -> Result<Vec<Vec<f32>>> {
    (0..row_count)
        .map(|row_index| fixture.read_tensor_f32(&format!("tap_row_{row_index}_concat")))
        .collect()
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

fn open_live_tap_runner(
    args: &SpdLiveTapParityArgs,
    manifest: &SpdHeadManifest,
    ranges: &[SpdStageLayerRange],
) -> Result<SpdLiveTapRunner> {
    if live_tap_model_path_is_layer_package(&args.model_path) {
        let package_ref = args.model_path.to_string_lossy().to_string();
        let package_info = inspect_layer_package(&package_ref).with_context(|| {
            format!("inspect SPD live tap package {}", args.model_path.display())
        })?;
        return open_live_tap_runner_with_source(
            args,
            manifest,
            ranges,
            SpdLiveTapModelSource::LayerPackage {
                package_ref: &package_ref,
                model_id: &package_info.model_id,
                topology_id: "spd-live-tap-parity",
            },
        );
    }
    open_live_tap_runner_with_source(
        args,
        manifest,
        ranges,
        SpdLiveTapModelSource::Gguf(&args.model_path),
    )
}

fn open_live_tap_runner_with_source(
    args: &SpdLiveTapParityArgs,
    manifest: &SpdHeadManifest,
    ranges: &[SpdStageLayerRange],
    model_source: SpdLiveTapModelSource<'_>,
) -> Result<SpdLiveTapRunner> {
    SpdLiveTapRunner::open(SpdLiveTapRunnerConfig {
        model_source,
        stage_ranges: ranges,
        layer_end: args.layer_end,
        hidden_size: usize::try_from(manifest.topology.hidden_size)
            .context("SPD hidden_size exceeds usize")?,
        vocab_size: usize::try_from(manifest.topology.vocab_size)
            .context("SPD vocab_size exceeds usize")?,
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        selected_backend_device: args.selected_backend_device.clone(),
    })
    .context("open live SPD tap runner")
}

fn live_tap_model_path_is_layer_package(path: &Path) -> bool {
    path.is_dir() && path.join("model-package.json").is_file()
}

fn live_tap_load_mode(path: &Path) -> RuntimeLoadMode {
    if live_tap_model_path_is_layer_package(path) {
        RuntimeLoadMode::LayerPackage
    } else {
        RuntimeLoadMode::RuntimeSlice
    }
}

fn assemble_live_cur_in(
    tap_projector: &SpdTapInputProjector,
    taps: &BTreeMap<u32, ActivationFrame>,
    inputs: LiveRowInputs<'_>,
) -> Result<LiveRows> {
    let mut cur_in = Vec::with_capacity(inputs.row_count * inputs.hidden_size);
    let mut terminal_normed_cur_in = Vec::with_capacity(inputs.row_count * inputs.hidden_size);
    let mut rows = Vec::with_capacity(inputs.row_count);
    let mut terminal_normed_rows = Vec::with_capacity(inputs.row_count);
    let mut max_diff = 0.0_f32;
    let mut terminal_normed_max_diff = 0.0_f32;
    for row_index in 0..inputs.row_count {
        let position = inputs.row_positions[row_index];
        let stage_id = u32::try_from(inputs.row_i_stages[row_index])
            .with_context(|| format!("SPD fixture row {row_index} has negative stage id"))?;
        let hf_indices = &inputs.row_hf_indices[row_index];
        let concat_hidden = concat_live_hidden(taps, hf_indices, position, inputs.hidden_size)?;
        let tap_components = tap_component_diffs(
            &concat_hidden,
            &inputs.fixture_concat_rows[row_index],
            hf_indices,
            inputs.hidden_size,
            inputs.layer_end,
            inputs.final_norm_weight,
        )?;
        let projection = tap_projector.project(stage_id, hf_indices, &concat_hidden)?;
        let terminal_normed_concat = terminal_final_normed_concat(
            &concat_hidden,
            hf_indices,
            inputs.hidden_size,
            inputs.layer_end,
            inputs.final_norm_weight,
        )?;
        let terminal_normed_projection =
            tap_projector.project(stage_id, hf_indices, &terminal_normed_concat)?;
        let expected = row(inputs.fixture_cur_in, row_index, inputs.hidden_size);
        let row_diff = max_abs_diff(&projection.projected, expected)?;
        let terminal_normed_row_diff =
            max_abs_diff(&terminal_normed_projection.projected, expected)?;
        max_diff = max_diff.max(row_diff);
        terminal_normed_max_diff = terminal_normed_max_diff.max(terminal_normed_row_diff);
        cur_in.extend_from_slice(&projection.projected);
        terminal_normed_cur_in.extend_from_slice(&terminal_normed_projection.projected);
        rows.push(json!({
            "row_index": row_index,
            "position_id": position,
            "stage_id": stage_id,
            "projection_name": projection.projection_name,
            "hf_indices": projection.hf_indices,
            "tap_components": tap_components,
            "max_abs_diff_vs_fixture": row_diff,
        }));
        terminal_normed_rows.push(json!({
            "row_index": row_index,
            "position_id": position,
            "stage_id": stage_id,
            "projection_name": terminal_normed_projection.projection_name,
            "hf_indices": terminal_normed_projection.hf_indices,
            "max_abs_diff_vs_fixture": terminal_normed_row_diff,
        }));
    }
    Ok(LiveRows {
        cur_in,
        rows,
        max_abs_diff: max_diff,
        terminal_normed_cur_in,
        terminal_normed_rows,
        terminal_normed_max_abs_diff: terminal_normed_max_diff,
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
        let normed_concat = terminal_final_normed_concat(
            &concat_hidden,
            hf_indices,
            inputs.hidden_size,
            inputs.layer_end,
            inputs.final_norm_weight,
        )?;
        let projection = tap_projector.project(stage_id, hf_indices, &normed_concat)?;
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

fn tap_component_diffs(
    live_concat: &[f32],
    fixture_concat: &[f32],
    hf_indices: &[u32],
    hidden_size: usize,
    layer_end: u32,
    final_norm_weight: &[f32],
) -> Result<Vec<serde_json::Value>> {
    let expected_width = hf_indices
        .len()
        .checked_mul(hidden_size)
        .context("SPD tap component width overflow")?;
    if live_concat.len() != expected_width || fixture_concat.len() != expected_width {
        bail!(
            "SPD tap component width mismatch: live {}, fixture {}, expected {}",
            live_concat.len(),
            fixture_concat.len(),
            expected_width
        );
    }
    hf_indices
        .iter()
        .enumerate()
        .map(|(component_index, hf_index)| {
            let start = component_index * hidden_size;
            let end = start + hidden_size;
            let live = &live_concat[start..end];
            let fixture = &fixture_concat[start..end];
            let terminal_final_norm = if *hf_index == layer_end {
                let normed = qwen_final_rms_norm(live, final_norm_weight)?;
                Some(json!({
                    "max_abs_diff_vs_fixture": max_abs_diff(&normed, fixture)?,
                    "mean_abs_diff_vs_fixture": mean_abs_diff(&normed, fixture)?,
                    "cosine_similarity_vs_fixture": cosine_similarity(&normed, fixture)?,
                    "inferred_output_norm_weight_vs_gguf": inferred_output_norm_weight_report(
                        live,
                        fixture,
                        final_norm_weight,
                    )?,
                }))
            } else {
                None
            };
            Ok(json!({
                "component_index": component_index,
                "hf_index": hf_index,
                "max_abs_diff_vs_fixture": max_abs_diff(live, fixture)?,
                "mean_abs_diff_vs_fixture": mean_abs_diff(live, fixture)?,
                "cosine_similarity_vs_fixture": cosine_similarity(live, fixture)?,
                "terminal_final_norm_vs_fixture": terminal_final_norm,
            }))
        })
        .collect()
}

fn terminal_final_normed_concat(
    concat: &[f32],
    hf_indices: &[u32],
    hidden_size: usize,
    layer_end: u32,
    final_norm_weight: &[f32],
) -> Result<Vec<f32>> {
    let mut normed = concat.to_vec();
    for (component_index, hf_index) in hf_indices.iter().enumerate() {
        if *hf_index != layer_end {
            continue;
        }
        let start = component_index * hidden_size;
        let end = start + hidden_size;
        let normalized = qwen_final_rms_norm(&normed[start..end], final_norm_weight)?;
        normed[start..end].copy_from_slice(&normalized);
    }
    Ok(normed)
}

fn qwen_final_rms_norm(values: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
    if values.len() != weight.len() {
        bail!(
            "final RMSNorm weight length {} does not match values {}",
            weight.len(),
            values.len()
        );
    }
    if values.is_empty() {
        return Ok(Vec::new());
    }
    let scale = qwen_rms_norm_scale(values);
    Ok(values
        .iter()
        .zip(weight)
        .map(|(value, weight)| value * scale * weight)
        .collect())
}

fn inferred_output_norm_weight_report(
    live_pre_norm: &[f32],
    fixture_post_norm: &[f32],
    output_norm_weight: &[f32],
) -> Result<serde_json::Value> {
    if live_pre_norm.len() != fixture_post_norm.len()
        || live_pre_norm.len() != output_norm_weight.len()
    {
        bail!(
            "terminal gamma diagnostic length mismatch: live {}, fixture {}, weight {}",
            live_pre_norm.len(),
            fixture_post_norm.len(),
            output_norm_weight.len()
        );
    }
    let scale = qwen_rms_norm_scale(live_pre_norm);
    let inferred = inferred_output_norm_weights(live_pre_norm, fixture_post_norm, scale);
    let inferred_values: Vec<f32> = inferred.iter().filter_map(|value| *value).collect();
    let expected_values: Vec<f32> = inferred
        .iter()
        .zip(output_norm_weight)
        .filter_map(|(inferred, expected)| inferred.map(|_| *expected))
        .collect();
    let skipped = inferred.len() - inferred_values.len();
    Ok(json!({
        "rms_scale": scale,
        "min_abs_denominator": MIN_INFERRED_GAMMA_DENOMINATOR,
        "sample_count": inferred_values.len(),
        "skipped_near_zero_denominator": skipped,
        "max_abs_diff_vs_gguf_weight": max_abs_diff(&inferred_values, &expected_values)?,
        "mean_abs_diff_vs_gguf_weight": mean_abs_diff(&inferred_values, &expected_values)?,
        "cosine_similarity_vs_gguf_weight": cosine_similarity(&inferred_values, &expected_values)?,
    }))
}

fn qwen_rms_norm_scale(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = values.iter().map(|value| value * value).sum();
    (sum_sq / values.len() as f32 + 1.0e-6).sqrt().recip()
}

fn inferred_output_norm_weights(
    live_pre_norm: &[f32],
    fixture_post_norm: &[f32],
    scale: f32,
) -> Vec<Option<f32>> {
    live_pre_norm
        .iter()
        .zip(fixture_post_norm)
        .map(|(live_value, fixture_value)| {
            let normalized_without_weight = live_value * scale;
            if normalized_without_weight.abs() < MIN_INFERRED_GAMMA_DENOMINATOR {
                None
            } else {
                Some(fixture_value / normalized_without_weight)
            }
        })
        .collect()
}

const MIN_INFERRED_GAMMA_DENOMINATOR: f32 = 5.0e-2;

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

fn mean_abs_diff(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        bail!("vector length mismatch: {} vs {}", left.len(), right.len());
    }
    if left.is_empty() {
        return Ok(0.0);
    }
    let sum = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>();
    Ok(sum / left.len() as f32)
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<Option<f32>> {
    if left.len() != right.len() {
        bail!("vector length mismatch: {} vs {}", left.len(), right.len());
    }
    let (dot, left_norm_sq, right_norm_sq) = left.iter().zip(right).fold(
        (0.0_f32, 0.0_f32, 0.0_f32),
        |(dot, left_norm_sq, right_norm_sq), (left, right)| {
            (
                dot + left * right,
                left_norm_sq + left * left,
                right_norm_sq + right * right,
            )
        },
    );
    if left_norm_sq == 0.0 || right_norm_sq == 0.0 {
        return Ok(None);
    }
    Ok(Some(dot / (left_norm_sq.sqrt() * right_norm_sq.sqrt())))
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
