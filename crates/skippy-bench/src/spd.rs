use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use skippy_runtime::{
    ActivationFrame, GGML_TYPE_F16, RuntimeConfig, RuntimeLoadMode, StageModel,
    package::{PackageStageRequest, inspect_layer_package, select_layer_package_parts},
    spd::{
        SpdHeadManifest, SpdHeadTopology, SpdLiveTapModelSource, SpdLiveTapRunner,
        SpdLiveTapRunnerConfig, SpdQwen3FixtureTopK, SpdQwen3ForwardInput, SpdQwen3ForwardTiming,
        SpdQwen3Head, SpdSafetensorsFile, SpdStageLayerRange, SpdTapInputProjector,
        plan_hidden_state_taps, read_gguf_output_norm_weight, run_qwen3_cached_fixture_parity,
        run_qwen3_fixture_parity, run_spd_tap_input_fixture_parity, spd_fixture_row_hf_indices,
    },
};

use crate::{
    cli::{SpdFixtureParityArgs, SpdLiveTapParityArgs, SpdProductCorpusCaptureArgs},
    spd_native_teacher::{
        NativeTeacherLogitsConfig, NativeTeacherLogitsWriter, NativeTeacherSample,
    },
};

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
    let verified_prompt_tokens = verified_prompt_token_sets(&args, &prompt_tokens)?;
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
    if args.product_native_teacher_logits && args.product_corpus_dir.is_none() {
        bail!("--product-native-teacher-logits requires --product-corpus-dir");
    }
    if args.skip_target_verification && args.product_corpus_dir.is_some() {
        bail!("--skip-target-verification cannot write --product-corpus-dir rows");
    }
    if args.skip_target_verification && args.product_native_teacher_logits {
        bail!("--skip-target-verification cannot write native teacher logits");
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
                producer: "skippy-bench spd-live-tap-parity",
                manifest_path: Some(&args.manifest),
                fixture_path: Some(&args.fixture),
                model_path: &args.model_path,
                splits: &args.splits,
                layer_end: args.layer_end,
                ctx_size: args.ctx_size,
                n_gpu_layers: args.n_gpu_layers,
                selected_backend_device: &args.selected_backend_device,
                stage_backend_devices: &args.stage_backend_devices,
                stream_live_tap_stages: args.stream_live_tap_stages,
                top_k: args.top_k,
                verify_steps: args.verify_steps,
                prompt_token_file: args.prompt_token_file.as_deref(),
                topology: serde_json::to_value(&manifest.topology)
                    .context("serialize SPD topology for product corpus")?,
                prompt_token_sets: &verified_prompt_tokens,
                row_count,
                row_i_stages: &row_i_stages,
                row_hf_indices: &row_hf_indices,
                hidden_size,
                final_norm_weight: &final_norm_weight,
                native_teacher_logits: args.product_native_teacher_logits,
                projected_cur_in_available: true,
            },
        )?),
        None => None,
    };
    let mut native_teacher = match (&args.product_corpus_dir, args.product_native_teacher_logits) {
        (Some(dir), true) => Some(NativeTeacherLogitsWriter::create(
            NativeTeacherLogitsConfig {
                dir: dir.clone(),
                draft_token_ids: draft_token_ids_from_manifest(&manifest)?,
                top_k: args.top_k,
                producer: "skippy-bench spd-live-tap-parity",
            },
        )?),
        _ => None,
    };
    let (
        target_verification,
        first_verified_generation_report,
        verified_generation_reports,
        product_corpus_report,
        native_teacher_report,
    ) = if args.skip_target_verification {
        (
            Value::Null,
            Value::Null,
            Vec::new(),
            Option::<Value>::None,
            Option::<Value>::None,
        )
    } else {
        let target_model = open_full_target_model(&args)?;
        let mut verified_generations = Vec::with_capacity(verified_prompt_tokens.len());
        for (prompt_index, prompt_tokens) in verified_prompt_tokens.iter().enumerate() {
            verified_generations.push(
                run_verified_generation(
                    &args,
                    &live_runner,
                    &spd_head,
                    &tap_projector,
                    VerifiedGenerationInputs {
                        prompt_index,
                        prompt_tokens,
                        target_model: &target_model,
                        row_i_stages: &row_i_stages,
                        row_hf_indices: &row_hf_indices,
                        final_norm_weight: &final_norm_weight,
                        row_count,
                        hidden_size,
                    },
                    product_corpus.as_mut(),
                    native_teacher.as_mut(),
                )
                .with_context(|| {
                    format!("run live SPD target verification for prompt_index={prompt_index}")
                })?,
            );
        }
        let product_corpus_report = product_corpus
            .as_mut()
            .map(ProductActivationCorpusWriter::finish)
            .transpose()?;
        let native_teacher_report = native_teacher
            .as_mut()
            .map(NativeTeacherLogitsWriter::finish)
            .transpose()?;
        let first_verified_generation = verified_generations
            .first()
            .context("no verified-generation prompt sets were provided")?;
        let target_verification = first_verified_generation
            .first_step
            .clone()
            .context("verified SPD generation produced no steps")?;
        let first_verified_generation_report = first_verified_generation.report.clone();
        let verified_generation_reports = verified_generations
            .iter()
            .map(|generation| generation.report.clone())
            .collect::<Vec<_>>();
        (
            target_verification,
            first_verified_generation_report,
            verified_generation_reports,
            product_corpus_report,
            native_teacher_report,
        )
    };
    let report = json!({
        "mode": "spd-live-tap-parity",
        "manifest": args.manifest,
        "fixture": args.fixture,
        "model_path": args.model_path,
        "splits": args.splits,
        "layer_end": args.layer_end,
        "stage_backend_devices": args.stage_backend_devices,
        "stream_live_tap_stages": args.stream_live_tap_stages,
        "skip_target_verification": args.skip_target_verification,
        "prompt_token_count": prompt_tokens.len(),
        "verified_prompt_count": verified_prompt_tokens.len(),
        "prompt_token_file": &args.prompt_token_file,
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
        "verified_generation": first_verified_generation_report,
        "verified_generations": verified_generation_reports,
        "product_activation_corpus": product_corpus_report,
        "native_teacher_logits": native_teacher_report,
    });
    let json = serde_json::to_vec_pretty(&report)?;
    if let Some(output) = args.output {
        fs::write(&output, &json)
            .with_context(|| format!("failed to write {}", output.display()))?;
    }
    println!("{}", String::from_utf8(json)?);
    terminal_normed_parity_gate.ensure_passed()
}

pub fn spd_product_corpus_capture(args: SpdProductCorpusCaptureArgs) -> Result<()> {
    if args.verify_steps == 0 {
        bail!("--verify-steps must be greater than zero");
    }
    if args.hidden_size == 0 {
        bail!("--hidden-size must be greater than zero");
    }
    if args.vocab_size == 0 {
        bail!("--vocab-size must be greater than zero");
    }
    if args.draft_vocab_size == 0 || args.draft_vocab_size > args.vocab_size {
        bail!(
            "--draft-vocab-size {} must be in 1..={}",
            args.draft_vocab_size,
            args.vocab_size
        );
    }
    let boundaries = capture_stage_boundaries(&args)?;
    let row_hf_indices = derive_spd_topology_row_hf_indices(&boundaries);
    let row_i_stages = derive_spd_topology_row_stage_ids(row_hf_indices.len())?;
    let topology = capture_topology(&args, &boundaries, &row_hf_indices)?;
    let ranges = capture_stage_ranges(&args)?;
    let tap_plan = plan_hidden_state_taps(&topology, &ranges)?;
    if tap_plan.requires_internal_taps() {
        bail!(
            "topology-only SPD capture requires boundary-aligned splits; missing hidden states {:?}",
            tap_plan.boundary_only_missing_hf_indices
        );
    }
    let prompt_token_sets = prompt_token_sets_from_file(&args.prompt_token_file)?;
    let final_norm_weight = read_capture_final_norm_weight(&args)?;
    let draft_token_ids = capture_draft_token_ids(&args)?;
    let mut product_corpus =
        ProductActivationCorpusWriter::create(ProductActivationCorpusConfig {
            dir: args.product_corpus_dir.clone(),
            producer: "skippy-bench spd-product-corpus-capture",
            manifest_path: None,
            fixture_path: None,
            model_path: &args.model_path,
            splits: &args.splits,
            layer_end: args.layer_end,
            ctx_size: args.ctx_size,
            n_gpu_layers: args.n_gpu_layers,
            selected_backend_device: &args.selected_backend_device,
            stage_backend_devices: &args.stage_backend_devices,
            stream_live_tap_stages: args.stream_live_tap_stages,
            top_k: args.top_k,
            verify_steps: args.verify_steps,
            prompt_token_file: Some(&args.prompt_token_file),
            topology: serde_json::to_value(&topology)
                .context("serialize topology-only SPD capture topology")?,
            prompt_token_sets: &prompt_token_sets,
            row_count: row_hf_indices.len(),
            row_i_stages: &row_i_stages,
            row_hf_indices: &row_hf_indices,
            hidden_size: args.hidden_size,
            final_norm_weight: &final_norm_weight,
            native_teacher_logits: args.product_native_teacher_logits,
            projected_cur_in_available: false,
        })?;
    let mut native_teacher = args
        .product_native_teacher_logits
        .then(|| {
            NativeTeacherLogitsWriter::create(NativeTeacherLogitsConfig {
                dir: args.product_corpus_dir.clone(),
                draft_token_ids: draft_token_ids.clone(),
                top_k: args.top_k,
                producer: "skippy-bench spd-product-corpus-capture",
            })
        })
        .transpose()?;
    let target_traces = {
        let target_model = open_capture_full_target_model(&args)?;
        let mut traces = Vec::with_capacity(prompt_token_sets.len());
        for (prompt_index, prompt_tokens) in prompt_token_sets.iter().enumerate() {
            traces.push(capture_product_target_generation(
                &args,
                ProductTargetCaptureInputs {
                    prompt_index,
                    prompt_tokens,
                    target_model: &target_model,
                    row_count: row_hf_indices.len(),
                },
                native_teacher.as_mut(),
            )?);
        }
        traces
    };
    let live_runner = open_capture_live_tap_runner(&args, &ranges)?;
    let mut reports = Vec::with_capacity(prompt_token_sets.len());
    for trace in &target_traces {
        reports.push(capture_product_tap_generation(
            &args,
            &live_runner,
            trace,
            ProductTapCaptureInputs {
                row_i_stages: &row_i_stages,
                row_hf_indices: &row_hf_indices,
                final_norm_weight: &final_norm_weight,
                row_count: row_hf_indices.len(),
                hidden_size: args.hidden_size,
            },
            &mut product_corpus,
        )?);
    }
    let product_corpus_report = product_corpus.finish()?;
    let native_teacher_report = native_teacher
        .as_mut()
        .map(NativeTeacherLogitsWriter::finish)
        .transpose()?;
    let report = json!({
        "mode": "spd-product-corpus-capture",
        "model_path": args.model_path,
        "splits": args.splits,
        "layer_end": args.layer_end,
        "hidden_size": args.hidden_size,
        "vocab_size": args.vocab_size,
        "draft_vocab_size": args.draft_vocab_size,
        "prompt_token_file": args.prompt_token_file,
        "prompt_count": prompt_token_sets.len(),
        "topology": topology,
        "tap_plan": {
            "required_hf_indices": tap_plan.required_hf_indices,
            "stage_boundary_hf_indices": tap_plan.stage_boundary_hf_indices,
            "boundary_only": tap_plan.can_use_stage_boundaries_only(),
        },
        "base_model_load": "skipped",
        "reference_train": "skipped",
        "product_activation_corpus": product_corpus_report,
        "native_teacher_logits": native_teacher_report,
        "captures": reports,
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

struct ProductTargetCaptureInputs<'a> {
    prompt_index: usize,
    prompt_tokens: &'a [i32],
    target_model: &'a StageModel,
    row_count: usize,
}

struct ProductTapCaptureInputs<'a> {
    row_i_stages: &'a [i64],
    row_hf_indices: &'a [Vec<u32>],
    final_norm_weight: &'a [f32],
    row_count: usize,
    hidden_size: usize,
}

struct ProductTargetTrace {
    prompt_index: usize,
    prompt_token_count: usize,
    generated_tokens: Vec<i32>,
    target_elapsed_ms: f64,
    steps: Vec<ProductTargetStep>,
}

struct ProductTargetStep {
    step_index: usize,
    context_tokens: Vec<i32>,
    row_positions: Vec<i64>,
    current_token: i32,
    target_token: i32,
    decode_ms: f64,
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
    prompt_index: usize,
    prompt_tokens: &'a [i32],
    target_model: &'a StageModel,
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
    mut native_teacher: Option<&mut NativeTeacherLogitsWriter>,
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
    let mut target_session = inputs
        .target_model
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
        let native_teacher_logits = match native_teacher.as_deref() {
            Some(writer) => Some(
                target_session
                    .copy_current_logits_for_tokens(writer.draft_token_ids())
                    .context("copy native target logits for SPD draft vocabulary")?,
            ),
            None => None,
        };
        if let Some(writer) = product_corpus.as_deref_mut() {
            writer.write_step(ProductActivationSample {
                prompt_index: inputs.prompt_index,
                step_index,
                context_tokens: &context_tokens,
                row_positions: &row_positions,
                row_stage_ids: inputs.row_i_stages,
                row_hf_indices: inputs.row_hf_indices,
                query_row_index: inputs.row_count.saturating_sub(1),
                target_position: context_tokens.len(),
                cur_in: &cur_in,
                raw_tap_concat: &live_rows.raw_tap_concat,
                current_token: current,
                proposal_topk: &live_topk,
                target_token,
                accepted,
                committed_token,
                baseline_greedy_token: baseline_token,
            })?;
        }
        if let (Some(writer), Some(logits)) = (
            native_teacher.as_deref_mut(),
            native_teacher_logits.as_deref(),
        ) {
            writer.write_step(NativeTeacherSample {
                prompt_index: inputs.prompt_index,
                step_index,
                target_position: context_tokens.len(),
                query_row_index: inputs.row_count.saturating_sub(1),
                query_position: row_positions[inputs.row_count.saturating_sub(1)],
                target_token,
                logits,
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
        "prompt_index": inputs.prompt_index,
        "prompt_token_count": inputs.prompt_tokens.len(),
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

fn capture_product_target_generation(
    args: &SpdProductCorpusCaptureArgs,
    inputs: ProductTargetCaptureInputs<'_>,
    mut native_teacher: Option<&mut NativeTeacherLogitsWriter>,
) -> Result<ProductTargetTrace> {
    if inputs.prompt_tokens.len() < inputs.row_count {
        bail!(
            "SPD product capture prompt length {} is shorter than row count {}",
            inputs.prompt_tokens.len(),
            inputs.row_count
        );
    }
    let prefix = inputs
        .prompt_tokens
        .get(..inputs.prompt_tokens.len().saturating_sub(1))
        .context("failed to split product-capture prompt prefix")?;
    let mut target_session = inputs
        .target_model
        .create_session()
        .context("create product capture target session")?;
    prefill_target_prefix(&mut target_session, prefix)?;
    let mut context_tokens = inputs.prompt_tokens.to_vec();
    let total_timer = Instant::now();
    let mut steps = Vec::with_capacity(args.verify_steps);
    for step_index in 0..args.verify_steps {
        let current = *context_tokens
            .last()
            .context("product capture context is empty")?;
        let row_positions = sliding_row_positions(context_tokens.len(), inputs.row_count)?;
        let step_context_tokens = context_tokens.clone();
        let decode_timer = Instant::now();
        let target_token = target_session
            .decode_step(current)
            .context("target greedy decode failed during product capture")?;
        let decode_ms = elapsed_ms(decode_timer);
        let native_teacher_logits = match native_teacher.as_deref() {
            Some(writer) => Some(
                target_session
                    .copy_current_logits_for_tokens(writer.draft_token_ids())
                    .context("copy native target logits for topology-only SPD draft vocabulary")?,
            ),
            None => None,
        };
        if let (Some(writer), Some(logits)) = (
            native_teacher.as_deref_mut(),
            native_teacher_logits.as_deref(),
        ) {
            writer.write_step(NativeTeacherSample {
                prompt_index: inputs.prompt_index,
                step_index,
                target_position: context_tokens.len(),
                query_row_index: inputs.row_count.saturating_sub(1),
                query_position: row_positions[inputs.row_count.saturating_sub(1)],
                target_token,
                logits,
            })?;
        }
        context_tokens.push(target_token);
        steps.push(ProductTargetStep {
            step_index,
            context_tokens: step_context_tokens,
            row_positions,
            current_token: current,
            target_token,
            decode_ms,
        });
    }
    Ok(ProductTargetTrace {
        prompt_index: inputs.prompt_index,
        prompt_token_count: inputs.prompt_tokens.len(),
        generated_tokens: context_tokens[inputs.prompt_tokens.len()..].to_vec(),
        target_elapsed_ms: elapsed_ms(total_timer),
        steps,
    })
}

fn capture_product_tap_generation(
    args: &SpdProductCorpusCaptureArgs,
    live_runner: &SpdLiveTapRunner,
    trace: &ProductTargetTrace,
    inputs: ProductTapCaptureInputs<'_>,
    product_corpus: &mut ProductActivationCorpusWriter,
) -> Result<serde_json::Value> {
    let zero_projected_cur_in = vec![0.0_f32; inputs.row_count * inputs.hidden_size];
    let empty_topk = SpdQwen3FixtureTopK {
        draft_indices: Vec::new(),
        token_ids: Vec::new(),
        logits: Vec::new(),
    };
    let total_timer = Instant::now();
    let mut steps = Vec::with_capacity(trace.steps.len());
    for step in &trace.steps {
        let step_timer = Instant::now();
        let tap_timer = Instant::now();
        let taps = live_runner.collect_taps(&step.context_tokens)?;
        let tap_ms = elapsed_ms(tap_timer);
        let assemble_timer = Instant::now();
        let raw_tap_concat = assemble_raw_live_tap_concat_for_positions(
            &taps,
            DynamicLiveRowInputs {
                row_positions: &step.row_positions,
                row_i_stages: inputs.row_i_stages,
                row_hf_indices: inputs.row_hf_indices,
                final_norm_weight: inputs.final_norm_weight,
                layer_end: args.layer_end,
                hidden_size: inputs.hidden_size,
            },
        )?;
        let assemble_ms = elapsed_ms(assemble_timer);
        product_corpus.write_step(ProductActivationSample {
            prompt_index: trace.prompt_index,
            step_index: step.step_index,
            context_tokens: &step.context_tokens,
            row_positions: &step.row_positions,
            row_stage_ids: inputs.row_i_stages,
            row_hf_indices: inputs.row_hf_indices,
            query_row_index: inputs.row_count.saturating_sub(1),
            target_position: step.context_tokens.len(),
            cur_in: &zero_projected_cur_in,
            raw_tap_concat: &raw_tap_concat,
            current_token: step.current_token,
            proposal_topk: &empty_topk,
            target_token: step.target_token,
            accepted: false,
            committed_token: step.target_token,
            baseline_greedy_token: step.target_token,
        })?;
        steps.push(json!({
            "step_index": step.step_index,
            "context_token_count_before": step.context_tokens.len(),
            "current_token": step.current_token,
            "target_token": step.target_token,
            "committed_tokens": [step.target_token],
            "row_positions": &step.row_positions,
            "row_stage_ids": inputs.row_i_stages,
            "proposal_source": "none_topology_only_capture",
            "timing_ms": {
                "tap_replay": tap_ms,
                "assemble_raw_tap_concat": assemble_ms,
                "target_greedy_decode": step.decode_ms,
                "tap_phase_total": elapsed_ms(step_timer),
                "phase_sum": tap_ms + assemble_ms + step.decode_ms,
            },
        }));
    }
    Ok(json!({
        "prompt_index": trace.prompt_index,
        "prompt_token_count": trace.prompt_token_count,
        "steps_requested": args.verify_steps,
        "steps_completed": steps.len(),
        "generated_tokens": &trace.generated_tokens,
        "proposal_source": "none_topology_only_capture",
        "target_phase_elapsed_ms": trace.target_elapsed_ms,
        "tap_phase_elapsed_ms": elapsed_ms(total_timer),
        "steps": steps,
    }))
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

fn open_capture_full_target_model(args: &SpdProductCorpusCaptureArgs) -> Result<StageModel> {
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
            format!(
                "inspect SPD product capture package {}",
                args.model_path.display()
            )
        })?;
        let parts = select_layer_package_parts(&PackageStageRequest {
            model_id: package_info.model_id,
            topology_id: "spd-product-corpus-capture".to_string(),
            package_ref,
            stage_id: "spd-product-capture-target".to_string(),
            layer_start: 0,
            layer_end: args.layer_end,
            include_embeddings: true,
            include_output: true,
        })
        .context("select SPD product capture full-target package parts")?;
        return StageModel::open_from_parts(&parts.absolute_paths, &config).with_context(|| {
            format!(
                "open full target layer package {} for SPD product capture",
                args.model_path.display()
            )
        });
    }
    StageModel::open(&args.model_path, &config).with_context(|| {
        format!(
            "open full target model {} for SPD product capture",
            args.model_path.display()
        )
    })
}

fn open_capture_live_tap_runner(
    args: &SpdProductCorpusCaptureArgs,
    ranges: &[SpdStageLayerRange],
) -> Result<SpdLiveTapRunner> {
    if live_tap_model_path_is_layer_package(&args.model_path) {
        let package_ref = args.model_path.to_string_lossy().to_string();
        let package_info = inspect_layer_package(&package_ref).with_context(|| {
            format!(
                "inspect SPD product capture package {}",
                args.model_path.display()
            )
        })?;
        return SpdLiveTapRunner::open(SpdLiveTapRunnerConfig {
            model_source: SpdLiveTapModelSource::LayerPackage {
                package_ref: &package_ref,
                model_id: &package_info.model_id,
                topology_id: "spd-product-corpus-capture",
            },
            stage_ranges: ranges,
            layer_end: args.layer_end,
            hidden_size: args.hidden_size,
            vocab_size: args.vocab_size,
            ctx_size: args.ctx_size,
            n_gpu_layers: args.n_gpu_layers,
            selected_backend_device: args.selected_backend_device.clone(),
            stage_backend_devices: args.stage_backend_devices.clone(),
            stream_stages: args.stream_live_tap_stages,
        })
        .context("open topology-only SPD product capture live tap runner");
    }
    SpdLiveTapRunner::open(SpdLiveTapRunnerConfig {
        model_source: SpdLiveTapModelSource::Gguf(&args.model_path),
        stage_ranges: ranges,
        layer_end: args.layer_end,
        hidden_size: args.hidden_size,
        vocab_size: args.vocab_size,
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        selected_backend_device: args.selected_backend_device.clone(),
        stage_backend_devices: args.stage_backend_devices.clone(),
        stream_stages: args.stream_live_tap_stages,
    })
    .context("open topology-only SPD product capture live tap runner")
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

fn verified_prompt_token_sets(
    args: &SpdLiveTapParityArgs,
    fixture_prompt_tokens: &[i32],
) -> Result<Vec<Vec<i32>>> {
    let Some(path) = &args.prompt_token_file else {
        return Ok(vec![fixture_prompt_tokens.to_vec()]);
    };
    prompt_token_sets_from_file(path)
}

fn prompt_token_sets_from_file(path: &Path) -> Result<Vec<Vec<i32>>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("read SPD prompt token file {}", path.display()))?;
    let mut prompts = Vec::new();
    for (line_index, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(trimmed).with_context(|| {
            format!(
                "parse SPD prompt token JSON at {}:{}",
                path.display(),
                line_index + 1
            )
        })?;
        prompts.push(prompt_tokens_from_json_value(&value, line_index + 1)?);
    }
    if prompts.is_empty() {
        bail!(
            "SPD prompt token file {} contained no prompts",
            path.display()
        );
    }
    Ok(prompts)
}

fn capture_stage_boundaries(args: &SpdProductCorpusCaptureArgs) -> Result<Vec<u32>> {
    if args.layer_end == 0 {
        bail!("--layer-end must be greater than zero");
    }
    let mut boundaries = args.splits.clone();
    boundaries.push(args.layer_end);
    let mut previous = 0;
    for boundary in &boundaries {
        if *boundary <= previous {
            bail!("capture stage boundaries must be strictly increasing: {boundaries:?}");
        }
        previous = *boundary;
    }
    Ok(boundaries)
}

fn capture_stage_ranges(args: &SpdProductCorpusCaptureArgs) -> Result<Vec<SpdStageLayerRange>> {
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

fn derive_spd_topology_row_hf_indices(boundaries: &[u32]) -> Vec<Vec<u32>> {
    (1..=boundaries.len())
        .rev()
        .map(|depth| {
            let mut row = Vec::with_capacity(depth + 1);
            row.push(0);
            row.extend_from_slice(&boundaries[..depth]);
            row
        })
        .collect()
}

fn derive_spd_topology_row_stage_ids(row_count: usize) -> Result<Vec<i64>> {
    (1..=row_count)
        .rev()
        .map(|stage_id| i64::try_from(stage_id).context("SPD row stage id exceeds i64"))
        .collect()
}

fn capture_topology(
    args: &SpdProductCorpusCaptureArgs,
    boundaries: &[u32],
    row_hf_indices: &[Vec<u32>],
) -> Result<SpdHeadTopology> {
    let rope_theta = (args.rope_theta > 0).then_some(args.rope_theta);
    let rotary_dim = (args.rotary_dim > 0).then_some(args.rotary_dim);
    if rope_theta.is_some() != rotary_dim.is_some() {
        bail!("--rope-theta and --rotary-dim must be provided together");
    }
    Ok(SpdHeadTopology {
        hidden_size: u32::try_from(args.hidden_size).context("hidden_size exceeds u32")?,
        vocab_size: u32::try_from(args.vocab_size).context("vocab_size exceeds u32")?,
        draft_vocab_size: u32::try_from(args.draft_vocab_size)
            .context("draft_vocab_size exceeds u32")?,
        num_stages: u32::try_from(boundaries.len()).context("num_stages exceeds u32")?,
        stage_layer_boundaries: Some(boundaries.to_vec()),
        num_spec_layers: args.num_spec_layers,
        trained_with_use_deepest: false,
        shallow_hidden_layer_indices: row_hf_indices.to_vec(),
        spec_init_from_base_layers: None,
        draft_token_ids: Some(
            capture_draft_token_ids(args)?
                .into_iter()
                .map(|token| u32::try_from(token).context("draft token id is negative"))
                .collect::<Result<Vec<_>>>()?,
        ),
        rope_theta,
        rotary_dim,
    })
}

fn capture_draft_token_ids(args: &SpdProductCorpusCaptureArgs) -> Result<Vec<i32>> {
    if let Some(path) = &args.draft_token_ids_file {
        let value: serde_json::Value = serde_json::from_slice(
            &fs::read(path).with_context(|| format!("read {}", path.display()))?,
        )
        .with_context(|| format!("parse draft token ids JSON {}", path.display()))?;
        let array = value
            .as_array()
            .context("draft token ids JSON must be an array")?;
        let mut out = Vec::with_capacity(array.len());
        for (index, token) in array.iter().enumerate() {
            let token = token
                .as_i64()
                .with_context(|| format!("draft token id {index} is not an integer"))?;
            out.push(i32::try_from(token).context("draft token id exceeds i32")?);
        }
        if out.is_empty() {
            bail!("draft token id file {} is empty", path.display());
        }
        return Ok(out);
    }
    (0..args.draft_vocab_size)
        .map(|token| i32::try_from(token).context("draft token id exceeds i32"))
        .collect()
}

fn draft_token_ids_from_manifest(manifest: &SpdHeadManifest) -> Result<Vec<i32>> {
    manifest
        .topology
        .draft_token_ids
        .as_ref()
        .context("native teacher logits require SPD manifest draft_token_ids")?
        .iter()
        .map(|token| i32::try_from(*token).context("draft token id exceeds i32"))
        .collect()
}

fn read_capture_final_norm_weight(args: &SpdProductCorpusCaptureArgs) -> Result<Vec<f32>> {
    if let Some(path) = &args.final_norm_weight_path {
        return read_f32_le_file(path, args.hidden_size);
    }
    let mut failures = Vec::new();
    for candidate in final_norm_gguf_candidates(&args.model_path) {
        if !candidate.is_file() {
            continue;
        }
        match read_gguf_output_norm_weight(&candidate, args.hidden_size) {
            Ok(weight) => return Ok(weight),
            Err(error) => failures.push(format!("{}: {error:#}", candidate.display())),
        }
    }
    bail!(
        "failed to read output_norm.weight for SPD capture from {}; candidates failed: {}",
        args.model_path.display(),
        failures.join("; ")
    )
}

fn final_norm_gguf_candidates(model_path: &Path) -> Vec<PathBuf> {
    if model_path.is_dir() {
        vec![
            model_path.join("shared/output.gguf"),
            model_path.join("output.gguf"),
            model_path.join("shared/metadata.gguf"),
        ]
    } else {
        vec![model_path.to_path_buf()]
    }
}

fn read_f32_le_file(path: &Path, expected_len: usize) -> Result<Vec<f32>> {
    let raw = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let expected_bytes = expected_len
        .checked_mul(std::mem::size_of::<f32>())
        .context("f32 file expected byte count overflow")?;
    if raw.len() != expected_bytes {
        bail!(
            "{} has {} bytes, expected {} for {} f32 values",
            path.display(),
            raw.len(),
            expected_bytes,
            expected_len
        );
    }
    Ok(raw
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk size checked")))
        .collect())
}

fn prompt_tokens_from_json_value(
    value: &serde_json::Value,
    line_number: usize,
) -> Result<Vec<i32>> {
    let tokens = if let Some(array) = value.as_array() {
        array
    } else {
        value
            .get("prompt_token_ids")
            .or_else(|| value.get("tokens"))
            .and_then(serde_json::Value::as_array)
            .with_context(|| {
                format!(
                    "SPD prompt token line {line_number} must be an array or object with prompt_token_ids/tokens"
                )
            })?
    };
    let mut out = Vec::with_capacity(tokens.len());
    for (token_index, token) in tokens.iter().enumerate() {
        let value = token.as_i64().with_context(|| {
            format!("SPD prompt token line {line_number} token {token_index} is not an integer")
        })?;
        out.push(i32::try_from(value).with_context(|| {
            format!("SPD prompt token line {line_number} token {token_index} exceeds i32")
        })?);
    }
    if out.is_empty() {
        bail!("SPD prompt token line {line_number} is empty");
    }
    Ok(out)
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
    raw_tap_concat: Vec<f32>,
}

struct ProductActivationCorpusConfig<'a> {
    dir: PathBuf,
    producer: &'a str,
    manifest_path: Option<&'a Path>,
    fixture_path: Option<&'a Path>,
    model_path: &'a Path,
    splits: &'a [u32],
    layer_end: u32,
    ctx_size: u32,
    n_gpu_layers: i32,
    selected_backend_device: &'a Option<String>,
    stage_backend_devices: &'a [String],
    stream_live_tap_stages: bool,
    top_k: usize,
    verify_steps: usize,
    prompt_token_file: Option<&'a Path>,
    topology: serde_json::Value,
    prompt_token_sets: &'a [Vec<i32>],
    row_count: usize,
    row_i_stages: &'a [i64],
    row_hf_indices: &'a [Vec<u32>],
    hidden_size: usize,
    final_norm_weight: &'a [f32],
    native_teacher_logits: bool,
    projected_cur_in_available: bool,
}

struct ProductActivationCorpusWriter {
    dir: PathBuf,
    rows_f32: BufWriter<File>,
    raw_rows_f32: BufWriter<File>,
    rows_jsonl: BufWriter<File>,
    row_count: usize,
    hidden_size: usize,
    raw_row_widths: Vec<usize>,
    raw_row_offsets: Vec<usize>,
    raw_width: usize,
    sample_count: usize,
}

struct ProductActivationSample<'a> {
    prompt_index: usize,
    step_index: usize,
    context_tokens: &'a [i32],
    row_positions: &'a [i64],
    row_stage_ids: &'a [i64],
    row_hf_indices: &'a [Vec<u32>],
    query_row_index: usize,
    target_position: usize,
    cur_in: &'a [f32],
    raw_tap_concat: &'a [f32],
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
        let raw_row_widths = raw_row_widths(config.row_hf_indices, config.hidden_size)?;
        let raw_row_offsets = cumulative_offsets(&raw_row_widths)?;
        let raw_width = *raw_row_offsets
            .last()
            .context("SPD raw row offsets should include final width")?;
        let single_prompt_tokens = (config.prompt_token_sets.len() == 1)
            .then(|| config.prompt_token_sets.first())
            .flatten();
        let native_teacher_logits = config.native_teacher_logits.then(|| {
            json!({
                "manifest_path": "native_teacher_manifest.json",
                "logits_path": "native_teacher_logits.f32",
                "metadata_path": "native_teacher_rows.jsonl",
                "summary_path": "native_teacher_summary.json",
                "teacher_source": "native_skippy_product_verifier_current_logits",
                "logit_scope": "draft",
            })
        });
        let manifest = json!({
            "schema": "skippy-spd-product-activation-corpus/v1",
            "producer": config.producer,
            "manifest_path": config.manifest_path.map(|path| path.display().to_string()),
            "fixture_path": config.fixture_path.map(|path| path.display().to_string()),
            "model_path": config.model_path.display().to_string(),
            "splits": config.splits,
            "layer_end": config.layer_end,
            "ctx_size": config.ctx_size,
            "n_gpu_layers": config.n_gpu_layers,
            "selected_backend_device": config.selected_backend_device,
            "stage_backend_devices": config.stage_backend_devices,
            "stream_live_tap_stages": config.stream_live_tap_stages,
            "top_k": config.top_k,
            "verify_steps": config.verify_steps,
            "prompt_token_file": config.prompt_token_file.map(|path| path.display().to_string()),
            "prompt_count": config.prompt_token_sets.len(),
            "prompt_tokens": single_prompt_tokens,
            "prompt_token_sets": config.prompt_token_sets,
            "topology": &config.topology,
            "row_count": config.row_count,
            "hidden_size": config.hidden_size,
            "row_stage_ids": config.row_i_stages,
            "row_hf_indices": config.row_hf_indices,
            "query_row_index": config.row_count.saturating_sub(1),
            "cur_in_convention": if config.projected_cur_in_available {
                "terminal_final_normed_cur_in"
            } else {
                "not_available_zero_placeholder"
            },
            "projected_cur_in_available": config.projected_cur_in_available,
            "raw_tap_convention": "terminal_final_normed_tap_concat",
            "row_tensor": {
                "path": "rows.f32",
                "dtype": "f32_le",
                "shape": ["sample_count", config.row_count, config.hidden_size],
            },
            "raw_row_tensor": {
                "path": "raw_rows.f32",
                "dtype": "f32_le",
                "shape": ["sample_count", raw_width],
                "row_widths": &raw_row_widths,
                "row_offsets": &raw_row_offsets,
                "notes": "Packed terminal-final-normed tap concatenations before stage_projs projection.",
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
            "target_logits_available": config.native_teacher_logits,
            "paper_kl_training_ready": config.native_teacher_logits,
            "native_teacher_logits": native_teacher_logits,
            "notes": [
                if config.projected_cur_in_available {
                    "rows.f32 stores live Skippy/product activations after terminal final-norm alignment and sidecar input projection."
                } else {
                    "rows.f32 is a zero placeholder; use raw_rows.f32 for fresh topology-only training."
                },
                "raw_rows.f32 stores terminal-final-normed tap concatenations before sidecar input projection.",
                "Labels are greedy target tokens from the product verifier.",
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
        let raw_rows_f32 = BufWriter::new(
            File::create(config.dir.join("raw_rows.f32"))
                .with_context(|| format!("create {}", config.dir.join("raw_rows.f32").display()))?,
        );
        let rows_jsonl = BufWriter::new(
            File::create(config.dir.join("rows.jsonl"))
                .with_context(|| format!("create {}", config.dir.join("rows.jsonl").display()))?,
        );
        Ok(Self {
            dir: config.dir,
            rows_f32,
            raw_rows_f32,
            rows_jsonl,
            row_count: config.row_count,
            hidden_size: config.hidden_size,
            raw_row_widths,
            raw_row_offsets,
            raw_width,
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
        if sample.raw_tap_concat.len() != self.raw_width {
            bail!(
                "SPD product corpus raw_tap_concat length {} does not match raw_width {}",
                sample.raw_tap_concat.len(),
                self.raw_width
            );
        }
        write_f32_slice(&mut self.rows_f32, sample.cur_in)?;
        write_f32_slice(&mut self.raw_rows_f32, sample.raw_tap_concat)?;
        let row = json!({
            "schema": "skippy-spd-product-activation-row/v1",
            "sample_index": self.sample_count,
            "prompt_index": sample.prompt_index,
            "row_f32_offset": self.sample_count * expected_len,
            "row_f32_count": expected_len,
            "raw_row_f32_offset": self.sample_count * self.raw_width,
            "raw_row_f32_count": self.raw_width,
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
        self.raw_rows_f32
            .flush()
            .context("flush SPD product corpus raw_rows.f32")?;
        self.rows_jsonl
            .flush()
            .context("flush SPD product corpus rows.jsonl")?;
        let rows_bytes = fs::metadata(self.dir.join("rows.f32"))
            .with_context(|| format!("stat {}", self.dir.join("rows.f32").display()))?
            .len();
        let raw_rows_bytes = fs::metadata(self.dir.join("raw_rows.f32"))
            .with_context(|| format!("stat {}", self.dir.join("raw_rows.f32").display()))?
            .len();
        let summary = json!({
            "schema": "skippy-spd-product-activation-corpus-summary/v1",
            "dir": self.dir.display().to_string(),
            "sample_count": self.sample_count,
            "row_count": self.row_count,
            "hidden_size": self.hidden_size,
            "rows_f32_bytes": rows_bytes,
            "rows_f32_expected_bytes": self.sample_count * self.row_count * self.hidden_size * std::mem::size_of::<f32>(),
            "raw_rows_f32_bytes": raw_rows_bytes,
            "raw_rows_f32_expected_bytes": self.sample_count * self.raw_width * std::mem::size_of::<f32>(),
            "rows_path": "rows.f32",
            "raw_rows_path": "raw_rows.f32",
            "raw_row_width": self.raw_width,
            "raw_row_widths": &self.raw_row_widths,
            "raw_row_offsets": &self.raw_row_offsets,
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

fn raw_row_widths(row_hf_indices: &[Vec<u32>], hidden_size: usize) -> Result<Vec<usize>> {
    row_hf_indices
        .iter()
        .map(|indices| {
            indices
                .len()
                .checked_mul(hidden_size)
                .context("SPD raw row width overflow")
        })
        .collect()
}

fn cumulative_offsets(widths: &[usize]) -> Result<Vec<usize>> {
    let mut offsets = Vec::with_capacity(widths.len() + 1);
    let mut next = 0usize;
    offsets.push(next);
    for width in widths {
        next = next
            .checked_add(*width)
            .context("SPD raw row offset overflow")?;
        offsets.push(next);
    }
    Ok(offsets)
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
        stage_backend_devices: args.stage_backend_devices.clone(),
        stream_stages: args.stream_live_tap_stages,
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
    let raw_width = inputs
        .row_hf_indices
        .iter()
        .map(|indices| indices.len() * inputs.hidden_size)
        .sum::<usize>();
    let mut raw_tap_concat = Vec::with_capacity(raw_width);
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
        raw_tap_concat.extend_from_slice(&normed_concat);
        let projection = tap_projector.project(stage_id, hf_indices, &normed_concat)?;
        cur_in.extend_from_slice(&projection.projected);
    }
    Ok(DynamicLiveRows {
        cur_in,
        raw_tap_concat,
    })
}

fn assemble_raw_live_tap_concat_for_positions(
    taps: &BTreeMap<u32, ActivationFrame>,
    inputs: DynamicLiveRowInputs<'_>,
) -> Result<Vec<f32>> {
    if inputs.row_positions.len() != inputs.row_i_stages.len()
        || inputs.row_positions.len() != inputs.row_hf_indices.len()
    {
        bail!(
            "dynamic SPD raw row metadata length mismatch: positions {}, stages {}, hf rows {}",
            inputs.row_positions.len(),
            inputs.row_i_stages.len(),
            inputs.row_hf_indices.len()
        );
    }
    let raw_width = inputs
        .row_hf_indices
        .iter()
        .map(|indices| indices.len() * inputs.hidden_size)
        .sum::<usize>();
    let mut raw_tap_concat = Vec::with_capacity(raw_width);
    for row_index in 0..inputs.row_positions.len() {
        let position = inputs.row_positions[row_index];
        let hf_indices = &inputs.row_hf_indices[row_index];
        let concat_hidden = concat_live_hidden(taps, hf_indices, position, inputs.hidden_size)?;
        let normed_concat = terminal_final_normed_concat(
            &concat_hidden,
            hf_indices,
            inputs.hidden_size,
            inputs.layer_end,
            inputs.final_norm_weight,
        )?;
        raw_tap_concat.extend_from_slice(&normed_concat);
    }
    Ok(raw_tap_concat)
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
