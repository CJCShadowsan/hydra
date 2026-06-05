use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, clap::Args)]
pub(super) struct QuantPackRankArgs {
    pub(super) runs: Vec<PathBuf>,
    #[arg(long, default_value_t = RankRuntimeShape::DEFAULT_CTX_SIZE)]
    pub(super) ctx_size: u32,
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    pub(super) n_gpu_layers: i32,
    #[arg(long, default_value = KvCacheEstimate::DEFAULT_CACHE_TYPE)]
    pub(super) cache_type_k: String,
    #[arg(long, default_value = KvCacheEstimate::DEFAULT_CACHE_TYPE)]
    pub(super) cache_type_v: String,
    #[arg(long, default_value = ActivationTransferEstimate::DEFAULT_WIRE_DTYPE)]
    pub(super) activation_wire_dtype: String,
    #[arg(long)]
    pub(super) out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct QuantPackRankReport {
    schema_version: u32,
    kind: String,
    score_direction: String,
    score_formula: String,
    ranked_at_unix_secs: u64,
    candidate_count: usize,
    candidates: Vec<RankedCandidate>,
}

#[derive(Debug, Serialize)]
struct RankedCandidate {
    rank: usize,
    candidate: String,
    pack_id: Option<String>,
    run_dir: String,
    valid: bool,
    measured: bool,
    certification_status: Option<RankCertificationStatus>,
    certification_report_status: Option<RankCertificationStatus>,
    certification_subject_status: Option<RankCertificationSubjectStatus>,
    certification_path: Option<String>,
    certification_gate_failures: usize,
    certification_gate_warnings: usize,
    skippy_bench_evidence_count: usize,
    quality_evidence_count: usize,
    score: f64,
    decode_mean_ms: Option<f64>,
    decode_p95_ms: Option<f64>,
    estimated_tokens_per_second: Option<f64>,
    focused_runtime_generated_tokens_per_second: Option<f64>,
    focused_runtime_decode_elapsed_ms_p50: Option<f64>,
    ctx_size: u32,
    n_gpu_layers: i32,
    cache_type_k: String,
    cache_type_v: String,
    activation_width: Option<u32>,
    activation_wire_dtype: String,
    estimated_total_kv_cache_bytes: Option<u64>,
    estimated_largest_stage_kv_cache_bytes: Option<u64>,
    estimated_largest_stage_model_plus_kv_bytes: Option<u64>,
    estimated_boundary_activation_bytes: Option<u64>,
    estimated_decode_activation_transfer_bytes_per_token: Option<u64>,
    activation_transfer_source: Option<ActivationTransferSource>,
    package_artifact_bytes: u64,
    slowest_stage_artifact_bytes: u64,
    stage_imbalance_ratio: Option<f64>,
    layout_hash: Option<String>,
    strategy: Option<String>,
    default_quant: Option<String>,
    group_count: usize,
    source_sha256: Option<String>,
    notes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct BuildManifestInput {
    candidate: String,
    agent_pack: String,
    preflight: String,
    #[serde(default)]
    package: Option<String>,
    #[serde(default)]
    quantized_model: Option<String>,
    #[serde(default)]
    quantize_run: Option<String>,
    #[serde(default)]
    decode_profile: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AgentPackInput {
    pack_id: String,
    #[serde(default)]
    source: Option<AgentPackSourceInput>,
    quant_layout: AgentPackQuantLayoutInput,
}

#[derive(Debug, Deserialize)]
struct AgentPackSourceInput {
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct AgentPackQuantLayoutInput {
    strategy: String,
    #[serde(rename = "default")]
    default_quant: String,
    #[serde(default)]
    layout_hash: Option<String>,
    #[serde(default)]
    groups: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct PreflightInput {
    valid: bool,
    activation_width: Option<u32>,
    #[serde(default)]
    stages: Vec<PreflightStageInput>,
}

#[derive(Debug, Deserialize)]
struct PreflightStageInput {
    artifact_bytes: u64,
    #[serde(default)]
    layer_start: Option<u32>,
    #[serde(default)]
    layer_end: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ProfileInput {
    measurement_status: ProfileMeasurementStatusInput,
    summary: ProfileSummaryInput,
    #[serde(default)]
    stages: Vec<ProfileStageInput>,
}

#[derive(Debug, Deserialize)]
struct ProfileMeasurementStatusInput {
    status: String,
}

#[derive(Debug, Deserialize)]
struct ProfileSummaryInput {
    estimated_tokens_per_second: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ProfileStageInput {
    timing: ProfileTimingInput,
}

#[derive(Debug, Deserialize)]
struct ProfileTimingInput {
    mean_ms: Option<f64>,
    p95_ms: Option<f64>,
}

#[derive(Debug, Clone, Copy, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum RankCertificationStatus {
    Failed,
    MeasurementOnlyCandidate,
    AgentQualityCandidate,
}

#[derive(Debug, Deserialize)]
struct CertificationInput {
    status: RankCertificationStatus,
    #[serde(default)]
    runtime_shape: Option<CertificationRuntimeShapeInput>,
    #[serde(default)]
    subject: Option<CertificationSubjectInput>,
    #[serde(default)]
    gates: Vec<CertificationGateInput>,
    #[serde(default)]
    skippy_bench_reports: Vec<serde_json::Value>,
    #[serde(default)]
    quality_evidence: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct CertificationRuntimeShapeInput {
    ctx_size: Option<u32>,
    n_gpu_layers: Option<i32>,
    cache_type_k: Option<String>,
    cache_type_v: Option<String>,
    activation_wire_dtype: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CertificationSubjectInput {
    #[serde(default)]
    expected_quantized_model: Option<HashedArtifactInput>,
    #[serde(default)]
    package_manifest: Option<HashedArtifactInput>,
    #[serde(default)]
    agent_pack: Option<HashedArtifactInput>,
    #[serde(default)]
    preflight: Option<HashedArtifactInput>,
    #[serde(default)]
    build_manifest: Option<HashedArtifactInput>,
    #[serde(default)]
    quantize_run: Option<HashedArtifactInput>,
}

#[derive(Debug, Deserialize)]
struct HashedArtifactInput {
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct CertificationGateInput {
    status: CertificationGateStatusInput,
}

#[derive(Debug, Clone, Copy, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum CertificationGateStatusInput {
    Pass,
    Warn,
    Fail,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum RankCertificationSubjectStatus {
    Verified,
    Stale,
    NotVerifiable,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ActivationTransferSource {
    CertifiedLocalSplitChain,
    DirectLocalSplitChain,
    PreflightEstimate,
}

pub(super) fn run_quant_pack_rank(args: QuantPackRankArgs) -> Result<()> {
    if args.runs.is_empty() {
        bail!(
            "quant-pack rank requires at least one build directory or quant-pack-build.json path"
        );
    }

    let mut candidates = args
        .runs
        .iter()
        .map(|run| load_ranked_candidate(run.as_path(), RankRuntimeShape::from_args(&args)))
        .collect::<Result<Vec<_>>>()?;
    candidates.sort_by(compare_candidates);
    for (rank, candidate) in candidates.iter_mut().enumerate() {
        candidate.rank = rank + 1;
    }

    let report = QuantPackRankReport {
        schema_version: 1,
        kind: "skippy_quant_pack_rank".to_string(),
        score_direction: "lower_is_better".to_string(),
        score_formula:
            "invalid_penalty + certification_or_stale_subject_penalty + unmeasured_penalty + decode_mean_ms + focused_runtime_generated_tps_penalty + largest_stage_model_plus_kv_gib + stage_imbalance_ratio + activation_transfer_mib_per_token"
                .to_string(),
        ranked_at_unix_secs: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .context("system clock before Unix epoch")?
            .as_secs(),
        candidate_count: candidates.len(),
        candidates,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(out) = args.out {
        fs::write(&out, format!("{json}\n"))
            .with_context(|| format!("write quant-pack rank report {}", out.display()))?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn load_ranked_candidate(
    run: &Path,
    runtime_shape: RankRuntimeShape<'_>,
) -> Result<RankedCandidate> {
    let manifest_path = build_manifest_path(run);
    let manifest = read_json::<BuildManifestInput>(&manifest_path)?;
    let run_dir = manifest_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let agent_pack =
        read_json::<AgentPackInput>(&resolve_manifest_path(&run_dir, &manifest.agent_pack))?;
    let preflight =
        read_json::<PreflightInput>(&resolve_manifest_path(&run_dir, &manifest.preflight))?;
    let profile = manifest
        .decode_profile
        .as_deref()
        .map(|path| read_json::<ProfileInput>(&resolve_manifest_path(&run_dir, path)))
        .transpose()?;
    let certification = read_certification(&run_dir)?;
    let direct_skippy_bench_summary = DirectSkippyBenchSummary::from_run_dir(&run_dir);

    let package_artifact_bytes = preflight
        .stages
        .iter()
        .map(|stage| stage.artifact_bytes)
        .sum::<u64>();
    let slowest_stage_artifact_bytes = preflight
        .stages
        .iter()
        .map(|stage| stage.artifact_bytes)
        .max()
        .unwrap_or_default();
    let stage_imbalance_ratio = stage_imbalance_ratio(&preflight.stages);
    let transfer_estimate =
        ActivationTransferEstimate::from_preflight(&preflight, runtime_shape.activation_wire_dtype);
    let kv_estimate = KvCacheEstimate::from_preflight(
        &preflight,
        runtime_shape.ctx_size,
        runtime_shape.cache_type_k,
        runtime_shape.cache_type_v,
    );
    let measured = profile
        .as_ref()
        .is_some_and(|profile| profile.measurement_status.status == "measured");
    let decode_mean_ms = profile
        .as_ref()
        .and_then(|profile| profile.stages.first())
        .and_then(|stage| stage.timing.mean_ms);
    let decode_p95_ms = profile
        .as_ref()
        .and_then(|profile| profile.stages.first())
        .and_then(|stage| stage.timing.p95_ms);
    let estimated_tokens_per_second = profile
        .as_ref()
        .and_then(|profile| profile.summary.estimated_tokens_per_second);
    let certification_summary = certification
        .as_ref()
        .map(|(_, certification)| CertificationRankSummary::from_certification(certification));
    let certification_report_status = certification_summary.map(|summary| summary.status);
    let subject_check = certification
        .as_ref()
        .map(|(_, certification)| {
            certification_subject_check(
                &run_dir,
                &manifest_path,
                &manifest,
                certification,
                runtime_shape,
            )
        })
        .unwrap_or_else(CertificationSubjectCheck::missing);
    let trusted_certification_summary = certification_summary
        .filter(|_| subject_check.status == RankCertificationSubjectStatus::Verified);
    let (effective_decode_transfer_bytes_per_token, activation_transfer_source) =
        activation_transfer_score_input(
            trusted_certification_summary
                .and_then(|summary| summary.local_split_decode_transfer_bytes_per_token),
            direct_skippy_bench_summary.local_split_decode_transfer_bytes_per_token,
            transfer_estimate.decode_transfer_bytes_per_token,
        );
    let focused_runtime_generated_tokens_per_second = trusted_certification_summary
        .and_then(|summary| summary.focused_runtime_generated_tps)
        .or(direct_skippy_bench_summary.focused_runtime_generated_tps);
    let focused_runtime_decode_elapsed_ms_p50 = trusted_certification_summary
        .and_then(|summary| summary.focused_runtime_decode_elapsed_ms_p50)
        .or(direct_skippy_bench_summary.focused_runtime_decode_elapsed_ms_p50);
    let certification_status =
        effective_certification_status(certification_report_status, subject_check.status);
    let notes = rank_notes(RankNoteInputs {
        valid: preflight.valid,
        measured,
        decode_mean_ms,
        focused_runtime_generated_tokens_per_second,
        direct_skippy_bench_evidence_count: direct_skippy_bench_summary.evidence_count,
        direct_skippy_bench_evidence_labels: &direct_skippy_bench_summary.evidence_labels,
        direct_skippy_bench_ignored_evidence_notes: &direct_skippy_bench_summary
            .ignored_evidence_notes,
        certification_status,
        estimated_total_kv_cache_bytes: kv_estimate.total_bytes,
        activation_transfer_source,
        certification_subject_notes: &subject_check.notes,
    });
    let score = rank_score(RankScoreInputs {
        valid: preflight.valid,
        measured,
        certification_status,
        decode_mean_ms,
        focused_runtime_generated_tokens_per_second,
        slowest_stage_artifact_bytes,
        largest_stage_model_plus_kv_bytes: kv_estimate.largest_stage_model_plus_kv_bytes,
        stage_imbalance_ratio,
        decode_transfer_bytes_per_token: effective_decode_transfer_bytes_per_token,
    });

    Ok(RankedCandidate {
        rank: 0,
        candidate: manifest.candidate,
        pack_id: Some(agent_pack.pack_id),
        run_dir: run_dir.display().to_string(),
        valid: preflight.valid,
        measured,
        certification_status,
        certification_report_status,
        certification_subject_status: certification.as_ref().map(|_| subject_check.status),
        certification_path: certification
            .as_ref()
            .map(|(path, _)| path.display().to_string()),
        certification_gate_failures: certification_summary
            .map(|summary| summary.failed_gates)
            .unwrap_or_default(),
        certification_gate_warnings: certification_summary
            .map(|summary| summary.warned_gates)
            .unwrap_or_default(),
        skippy_bench_evidence_count: trusted_certification_summary
            .map(|summary| summary.skippy_bench_evidence_count)
            .unwrap_or(direct_skippy_bench_summary.evidence_count),
        quality_evidence_count: trusted_certification_summary
            .map(|summary| summary.quality_evidence_count)
            .unwrap_or_default(),
        score,
        decode_mean_ms,
        decode_p95_ms,
        estimated_tokens_per_second,
        focused_runtime_generated_tokens_per_second,
        focused_runtime_decode_elapsed_ms_p50,
        ctx_size: runtime_shape.ctx_size,
        n_gpu_layers: runtime_shape.n_gpu_layers,
        cache_type_k: kv_estimate.cache_type_k.to_string(),
        cache_type_v: kv_estimate.cache_type_v.to_string(),
        activation_width: preflight.activation_width,
        activation_wire_dtype: transfer_estimate.wire_dtype.to_string(),
        estimated_total_kv_cache_bytes: kv_estimate.total_bytes,
        estimated_largest_stage_kv_cache_bytes: kv_estimate.largest_stage_bytes,
        estimated_largest_stage_model_plus_kv_bytes: kv_estimate.largest_stage_model_plus_kv_bytes,
        estimated_boundary_activation_bytes: transfer_estimate.boundary_bytes,
        estimated_decode_activation_transfer_bytes_per_token:
            effective_decode_transfer_bytes_per_token,
        activation_transfer_source,
        package_artifact_bytes,
        slowest_stage_artifact_bytes,
        stage_imbalance_ratio,
        layout_hash: agent_pack.quant_layout.layout_hash,
        strategy: Some(agent_pack.quant_layout.strategy),
        default_quant: Some(agent_pack.quant_layout.default_quant),
        group_count: agent_pack.quant_layout.groups.len(),
        source_sha256: agent_pack.source.map(|source| source.sha256),
        notes,
    })
}

#[derive(Clone, Copy)]
struct CertificationRankSummary {
    status: RankCertificationStatus,
    failed_gates: usize,
    warned_gates: usize,
    skippy_bench_evidence_count: usize,
    quality_evidence_count: usize,
    focused_runtime_generated_tps: Option<f64>,
    focused_runtime_decode_elapsed_ms_p50: Option<f64>,
    local_split_decode_transfer_bytes_per_token: Option<u64>,
}

impl CertificationRankSummary {
    fn from_certification(certification: &CertificationInput) -> Self {
        Self {
            status: certification.status,
            failed_gates: certification
                .gates
                .iter()
                .filter(|gate| gate.status == CertificationGateStatusInput::Fail)
                .count(),
            warned_gates: certification
                .gates
                .iter()
                .filter(|gate| gate.status == CertificationGateStatusInput::Warn)
                .count(),
            skippy_bench_evidence_count: certification
                .skippy_bench_reports
                .iter()
                .filter(|report| evidence_report_status_is_pass(report))
                .count(),
            quality_evidence_count: certification
                .quality_evidence
                .iter()
                .filter(|report| evidence_report_status_is_pass(report))
                .count(),
            focused_runtime_generated_tps: focused_runtime_generated_tps(
                &certification.skippy_bench_reports,
            ),
            focused_runtime_decode_elapsed_ms_p50: focused_runtime_decode_elapsed_ms_p50(
                &certification.skippy_bench_reports,
            ),
            local_split_decode_transfer_bytes_per_token:
                local_split_decode_transfer_bytes_per_token(&certification.skippy_bench_reports),
        }
    }
}

fn focused_runtime_generated_tps(reports: &[serde_json::Value]) -> Option<f64> {
    focused_runtime_summary_value(reports)
        .and_then(|summary| summary.get("throughput_tokens_per_second"))
        .and_then(|throughput| throughput.get("generated"))
        .and_then(serde_json::Value::as_f64)
}

fn focused_runtime_decode_elapsed_ms_p50(reports: &[serde_json::Value]) -> Option<f64> {
    focused_runtime_summary_value(reports)
        .and_then(|summary| summary.get("latency_ms"))
        .and_then(|latency| latency.get("decode_elapsed_ms_p50"))
        .and_then(value_as_f64)
}

fn focused_runtime_summary_value(reports: &[serde_json::Value]) -> Option<&serde_json::Value> {
    reports
        .iter()
        .find(|report| {
            report
                .get("evidence_type")
                .and_then(serde_json::Value::as_str)
                == Some("skippy-bench-focused-runtime")
                && evidence_report_status_is_pass(report)
        })
        .and_then(|report| report.get("summary"))
}

fn local_split_decode_transfer_bytes_per_token(reports: &[serde_json::Value]) -> Option<u64> {
    reports
        .iter()
        .find(|report| {
            report
                .get("evidence_type")
                .and_then(serde_json::Value::as_str)
                == Some("skippy-bench-local-split-chain")
                && evidence_report_status_is_pass(report)
        })
        .and_then(|report| report.get("summary"))
        .and_then(local_split_summary_transfer_bytes)
}

fn local_split_summary_transfer_bytes(summary: &serde_json::Value) -> Option<u64> {
    summary
        .get("boundary_transfers")
        .and_then(serde_json::Value::as_array)
        .and_then(|transfers| sum_boundary_transfer_wire_bytes(transfers))
        .or_else(|| {
            summary
                .get("stages")
                .and_then(serde_json::Value::as_array)
                .and_then(|stages| sum_stage_transfer_wire_bytes(stages))
        })
}

fn evidence_report_status_is_pass(report: &serde_json::Value) -> bool {
    report.get("status").and_then(serde_json::Value::as_str) == Some("pass")
}

#[derive(Default)]
struct DirectSkippyBenchSummary {
    evidence_count: usize,
    evidence_labels: Vec<&'static str>,
    ignored_evidence_notes: Vec<String>,
    focused_runtime_generated_tps: Option<f64>,
    focused_runtime_decode_elapsed_ms_p50: Option<f64>,
    local_split_decode_transfer_bytes_per_token: Option<u64>,
}

impl DirectSkippyBenchSummary {
    fn from_run_dir(run_dir: &Path) -> Self {
        let evidence_dir = run_dir.join("evidence");
        let mut ignored_evidence_notes = Vec::new();
        let focused_runtime = read_direct_json(
            &evidence_dir.join("focused-runtime-report.json"),
            "focused-runtime",
            &mut ignored_evidence_notes,
        )
        .and_then(|report| {
            let measurement = raw_focused_runtime_measurement(&report);
            if measurement.is_none() {
                ignored_evidence_notes.push(
                    "ignored direct focused-runtime evidence: expected mode=executed with positive generated throughput and decode p50 latency"
                        .to_string(),
                );
            }
            measurement
        });
        let chat_corpus_passes = read_direct_json(
            &evidence_dir.join("chat-corpus.json"),
            "chat-corpus",
            &mut ignored_evidence_notes,
        )
        .is_some_and(|report| {
            let passes = raw_chat_corpus_passes(&report);
            if !passes {
                ignored_evidence_notes.push(
                    "ignored direct chat-corpus evidence: summary.errors must be 0".to_string(),
                );
            }
            passes
        });
        let token_lengths_passes = read_direct_json(
            &evidence_dir.join("prompt-lengths-summary.json"),
            "token-lengths",
            &mut ignored_evidence_notes,
        )
        .is_some_and(|report| {
            let passes = raw_token_lengths_passes(&report);
            if !passes {
                ignored_evidence_notes.push(
                    "ignored direct token-length evidence: exceeds_context must be 0".to_string(),
                );
            }
            passes
        });
        let local_split_transfer = read_direct_json(
            &evidence_dir.join("local-split-chain.json"),
            "local-split-chain",
            &mut ignored_evidence_notes,
        )
        .and_then(|report| {
            let transfer = raw_local_split_decode_transfer_bytes_per_token(&report);
            if transfer.is_none() {
                ignored_evidence_notes.push(
                    "ignored direct local-split-chain evidence: expected mode=local-split-chain-binary with predicted_token and positive boundary transfer bytes"
                        .to_string(),
                );
            }
            transfer
        });
        let mut evidence_labels = Vec::new();
        if focused_runtime.is_some() {
            evidence_labels.push("focused-runtime");
        }
        if chat_corpus_passes {
            evidence_labels.push("chat-corpus");
        }
        if token_lengths_passes {
            evidence_labels.push("token-lengths");
        }
        if local_split_transfer.is_some() {
            evidence_labels.push("local-split-chain");
        }
        let evidence_count = usize::from(focused_runtime.is_some())
            + usize::from(chat_corpus_passes)
            + usize::from(token_lengths_passes)
            + usize::from(local_split_transfer.is_some());
        Self {
            evidence_count,
            evidence_labels,
            ignored_evidence_notes,
            focused_runtime_generated_tps: focused_runtime
                .map(|measurement| measurement.generated_tps),
            focused_runtime_decode_elapsed_ms_p50: focused_runtime
                .map(|measurement| measurement.decode_elapsed_ms_p50),
            local_split_decode_transfer_bytes_per_token: local_split_transfer,
        }
    }
}

fn read_direct_json(
    path: &Path,
    evidence_type: &str,
    ignored_evidence_notes: &mut Vec<String>,
) -> Option<serde_json::Value> {
    if !path.is_file() {
        return None;
    }
    match read_json::<serde_json::Value>(path) {
        Ok(value) => Some(value),
        Err(error) => {
            ignored_evidence_notes.push(format!(
                "ignored direct {evidence_type} evidence: cannot parse {}: {error}",
                path.display()
            ));
            None
        }
    }
}

#[derive(Clone, Copy)]
struct RawFocusedRuntimeMeasurement {
    generated_tps: f64,
    decode_elapsed_ms_p50: f64,
}

fn raw_focused_runtime_measurement(
    report: &serde_json::Value,
) -> Option<RawFocusedRuntimeMeasurement> {
    if report.get("mode").and_then(serde_json::Value::as_str) != Some("executed") {
        return None;
    }
    let generated_tps = report
        .get("throughput_tokens_per_second")
        .and_then(|throughput| throughput.get("generated"))
        .and_then(value_as_f64)
        .filter(|value| *value > 0.0)?;
    let decode_elapsed_ms_p50 = report
        .get("latency_ms")
        .and_then(|latency| latency.get("decode_elapsed_ms_p50"))
        .and_then(value_as_f64)?;
    Some(RawFocusedRuntimeMeasurement {
        generated_tps,
        decode_elapsed_ms_p50,
    })
}

fn raw_chat_corpus_passes(report: &serde_json::Value) -> bool {
    report
        .get("summary")
        .and_then(|summary| summary.get("errors"))
        .and_then(serde_json::Value::as_u64)
        == Some(0)
}

fn raw_token_lengths_passes(report: &serde_json::Value) -> bool {
    report
        .get("exceeds_context")
        .and_then(serde_json::Value::as_u64)
        == Some(0)
}

fn raw_local_split_decode_transfer_bytes_per_token(report: &serde_json::Value) -> Option<u64> {
    if report.get("mode").and_then(serde_json::Value::as_str) != Some("local-split-chain-binary") {
        return None;
    }
    report.get("predicted_token")?;
    report
        .get("boundary_transfers")
        .and_then(serde_json::Value::as_array)
        .and_then(|transfers| sum_boundary_transfer_wire_bytes(transfers))
        .or_else(|| {
            report
                .get("stages")
                .and_then(serde_json::Value::as_array)
                .and_then(|stages| sum_stage_transfer_wire_bytes(stages))
        })
}

fn sum_boundary_transfer_wire_bytes(transfers: &[serde_json::Value]) -> Option<u64> {
    if transfers.is_empty() {
        return None;
    }
    transfers.iter().try_fold(0u64, |total, transfer| {
        let wire_bytes = transfer
            .get("wire_payload_bytes")
            .and_then(serde_json::Value::as_u64)
            .filter(|bytes| *bytes > 0)?;
        total.checked_add(wire_bytes)
    })
}

fn sum_stage_transfer_wire_bytes(stages: &[serde_json::Value]) -> Option<u64> {
    let mut total = 0u64;
    let mut observed = false;
    for stage in stages {
        let Some(wire_bytes) = stage
            .get("wire_payload_bytes")
            .and_then(serde_json::Value::as_u64)
            .filter(|bytes| *bytes > 0)
        else {
            continue;
        };
        total = total.checked_add(wire_bytes)?;
        observed = true;
    }
    observed.then_some(total)
}

fn value_as_f64(value: &serde_json::Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_u64().map(|value| value as f64))
        .or_else(|| value.as_i64().map(|value| value as f64))
}

#[derive(Clone)]
struct CertificationSubjectCheck {
    status: RankCertificationSubjectStatus,
    notes: Vec<String>,
}

impl CertificationSubjectCheck {
    fn missing() -> Self {
        Self {
            status: RankCertificationSubjectStatus::NotVerifiable,
            notes: Vec::new(),
        }
    }
}

fn certification_subject_check(
    run_dir: &Path,
    manifest_path: &Path,
    manifest: &BuildManifestInput,
    certification: &CertificationInput,
    runtime_shape: RankRuntimeShape<'_>,
) -> CertificationSubjectCheck {
    let Some(subject) = certification.subject.as_ref() else {
        return CertificationSubjectCheck {
            status: RankCertificationSubjectStatus::NotVerifiable,
            notes: vec![
                "certification has no subject hashes; ranking cannot verify artifact freshness"
                    .to_string(),
            ],
        };
    };

    let mut missing = Vec::new();
    let mut mismatches = Vec::new();
    compare_required_subject_hash(
        &mut missing,
        &mut mismatches,
        "build_manifest",
        Some(manifest_path.to_path_buf()),
        subject.build_manifest.as_ref(),
    );
    compare_required_subject_hash(
        &mut missing,
        &mut mismatches,
        "agent_pack",
        Some(resolve_manifest_path(run_dir, &manifest.agent_pack)),
        subject.agent_pack.as_ref(),
    );
    compare_required_subject_hash(
        &mut missing,
        &mut mismatches,
        "preflight",
        Some(resolve_manifest_path(run_dir, &manifest.preflight)),
        subject.preflight.as_ref(),
    );
    compare_required_subject_hash(
        &mut missing,
        &mut mismatches,
        "quantized_model",
        manifest
            .quantized_model
            .as_deref()
            .map(|path| resolve_manifest_path(run_dir, path)),
        subject.expected_quantized_model.as_ref(),
    );
    compare_required_subject_hash(
        &mut missing,
        &mut mismatches,
        "package_manifest",
        manifest
            .package
            .as_deref()
            .map(|path| resolve_manifest_path(run_dir, path).join("model-package.json")),
        subject.package_manifest.as_ref(),
    );
    compare_optional_subject_hash(
        &mut mismatches,
        "quantize_run",
        manifest
            .quantize_run
            .as_deref()
            .map(|path| resolve_manifest_path(run_dir, path)),
        subject.quantize_run.as_ref(),
    );
    compare_evidence_report_hashes(
        run_dir,
        &mut missing,
        &mut mismatches,
        "skippy_bench_reports",
        &certification.skippy_bench_reports,
    );
    compare_evidence_report_hashes(
        run_dir,
        &mut missing,
        &mut mismatches,
        "quality_evidence",
        &certification.quality_evidence,
    );
    compare_certification_runtime_shape(
        &mut missing,
        &mut mismatches,
        certification.runtime_shape.as_ref(),
        runtime_shape,
    );

    subject_check_result(missing, mismatches)
}

fn compare_certification_runtime_shape(
    missing: &mut Vec<String>,
    mismatches: &mut Vec<String>,
    certified: Option<&CertificationRuntimeShapeInput>,
    expected: RankRuntimeShape<'_>,
) {
    let Some(certified) = certified else {
        missing.push("runtime_shape: missing from certification report".to_string());
        return;
    };
    compare_u32_shape_field(
        missing,
        mismatches,
        "ctx_size",
        certified.ctx_size,
        expected.ctx_size,
    );
    compare_i32_shape_field(
        missing,
        mismatches,
        "n_gpu_layers",
        certified.n_gpu_layers,
        expected.n_gpu_layers,
    );
    compare_cache_shape_field(
        missing,
        mismatches,
        "cache_type_k",
        certified.cache_type_k.as_deref(),
        expected.cache_type_k,
    );
    compare_cache_shape_field(
        missing,
        mismatches,
        "cache_type_v",
        certified.cache_type_v.as_deref(),
        expected.cache_type_v,
    );
    compare_activation_shape_field(
        missing,
        mismatches,
        certified.activation_wire_dtype.as_deref(),
        expected.activation_wire_dtype,
    );
}

fn compare_u32_shape_field(
    missing: &mut Vec<String>,
    mismatches: &mut Vec<String>,
    field: &str,
    certified: Option<u32>,
    expected: u32,
) {
    match certified {
        Some(actual) if actual == expected => {}
        Some(actual) => mismatches.push(format!("runtime_shape.{field} {actual} != {expected}")),
        None => missing.push(format!(
            "runtime_shape.{field}: missing from certification report"
        )),
    }
}

fn compare_i32_shape_field(
    missing: &mut Vec<String>,
    mismatches: &mut Vec<String>,
    field: &str,
    certified: Option<i32>,
    expected: i32,
) {
    match certified {
        Some(actual) if actual == expected => {}
        Some(actual) => mismatches.push(format!("runtime_shape.{field} {actual} != {expected}")),
        None => missing.push(format!(
            "runtime_shape.{field}: missing from certification report"
        )),
    }
}

fn compare_cache_shape_field(
    missing: &mut Vec<String>,
    mismatches: &mut Vec<String>,
    field: &str,
    certified: Option<&str>,
    expected: &str,
) {
    match certified {
        Some(actual) if canonical_cache_type(actual) == canonical_cache_type(expected) => {}
        Some(actual) => mismatches.push(format!("runtime_shape.{field} {actual} != {expected}")),
        None => missing.push(format!(
            "runtime_shape.{field}: missing from certification report"
        )),
    }
}

fn compare_activation_shape_field(
    missing: &mut Vec<String>,
    mismatches: &mut Vec<String>,
    certified: Option<&str>,
    expected: &str,
) {
    match certified {
        Some(actual)
            if canonical_activation_wire_dtype(actual)
                == canonical_activation_wire_dtype(expected) => {}
        Some(actual) => mismatches.push(format!(
            "runtime_shape.activation_wire_dtype {actual} != {expected}"
        )),
        None => missing.push(
            "runtime_shape.activation_wire_dtype: missing from certification report".to_string(),
        ),
    }
}

fn compare_required_subject_hash(
    missing: &mut Vec<String>,
    mismatches: &mut Vec<String>,
    label: &str,
    current_path: Option<PathBuf>,
    certified: Option<&HashedArtifactInput>,
) {
    let Some(current_path) = current_path else {
        missing.push(format!("{label}: current path missing from build manifest"));
        return;
    };
    let Some(certified) = certified else {
        missing.push(format!("{label}: hash missing from certification subject"));
        return;
    };
    compare_subject_hash(mismatches, label, &current_path, certified);
}

fn compare_optional_subject_hash(
    mismatches: &mut Vec<String>,
    label: &str,
    current_path: Option<PathBuf>,
    certified: Option<&HashedArtifactInput>,
) {
    if let (Some(current_path), Some(certified)) = (current_path, certified) {
        compare_subject_hash(mismatches, label, &current_path, certified);
    }
}

fn compare_subject_hash(
    mismatches: &mut Vec<String>,
    label: &str,
    current_path: &Path,
    certified: &HashedArtifactInput,
) {
    match file_sha256(current_path) {
        Ok(current) if current == certified.sha256 => {}
        Ok(current) => mismatches.push(format!(
            "{label}: current sha256 {current} != certified sha256 {}",
            certified.sha256
        )),
        Err(error) => mismatches.push(format!("{label}: cannot hash current artifact: {error}")),
    }
}

fn compare_evidence_report_hashes(
    run_dir: &Path,
    missing: &mut Vec<String>,
    mismatches: &mut Vec<String>,
    label: &str,
    reports: &[serde_json::Value],
) {
    for (index, report) in reports.iter().enumerate() {
        let item_label = format!("{label}[{index}]");
        let Some(path) = report.get("path").and_then(serde_json::Value::as_str) else {
            missing.push(format!(
                "{item_label}: evidence path missing from certification report"
            ));
            continue;
        };
        let Some(sha256) = report.get("sha256").and_then(serde_json::Value::as_str) else {
            missing.push(format!(
                "{item_label}: evidence sha256 missing from certification report"
            ));
            continue;
        };
        compare_evidence_hash(
            run_dir,
            mismatches,
            &item_label,
            path,
            HashedArtifactInput {
                sha256: sha256.to_string(),
            },
        );
    }
}

fn compare_evidence_hash(
    run_dir: &Path,
    mismatches: &mut Vec<String>,
    label: &str,
    path: &str,
    certified: HashedArtifactInput,
) {
    compare_subject_hash(
        mismatches,
        label,
        &resolve_manifest_path(run_dir, path),
        &certified,
    );
}

fn subject_check_result(
    missing: Vec<String>,
    mismatches: Vec<String>,
) -> CertificationSubjectCheck {
    if !mismatches.is_empty() {
        return CertificationSubjectCheck {
            status: RankCertificationSubjectStatus::Stale,
            notes: mismatches
                .into_iter()
                .map(|detail| format!("certification subject is stale: {detail}"))
                .collect(),
        };
    }
    if !missing.is_empty() {
        return CertificationSubjectCheck {
            status: RankCertificationSubjectStatus::NotVerifiable,
            notes: missing
                .into_iter()
                .map(|detail| format!("certification subject is not verifiable: {detail}"))
                .collect(),
        };
    }
    CertificationSubjectCheck {
        status: RankCertificationSubjectStatus::Verified,
        notes: vec!["certification subject hashes match current artifacts".to_string()],
    }
}

fn effective_certification_status(
    report_status: Option<RankCertificationStatus>,
    subject_status: RankCertificationSubjectStatus,
) -> Option<RankCertificationStatus> {
    match (report_status, subject_status) {
        (
            Some(_),
            RankCertificationSubjectStatus::Stale | RankCertificationSubjectStatus::NotVerifiable,
        ) => Some(RankCertificationStatus::Failed),
        (status, _) => status,
    }
}

fn compare_candidates(left: &RankedCandidate, right: &RankedCandidate) -> std::cmp::Ordering {
    left.score
        .partial_cmp(&right.score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.candidate.cmp(&right.candidate))
}

#[derive(Clone, Copy)]
struct RankScoreInputs {
    valid: bool,
    measured: bool,
    certification_status: Option<RankCertificationStatus>,
    decode_mean_ms: Option<f64>,
    focused_runtime_generated_tokens_per_second: Option<f64>,
    slowest_stage_artifact_bytes: u64,
    largest_stage_model_plus_kv_bytes: Option<u64>,
    stage_imbalance_ratio: Option<f64>,
    decode_transfer_bytes_per_token: Option<u64>,
}

fn rank_score(inputs: RankScoreInputs) -> f64 {
    let validity_penalty = if inputs.valid { 0.0 } else { 1_000_000.0 };
    let certification_penalty = certification_penalty(inputs.certification_status);
    let measurement_penalty = if inputs.measured { 0.0 } else { 10_000.0 };
    let latency_penalty = inputs.decode_mean_ms.unwrap_or(1_000.0);
    let focused_runtime_penalty =
        focused_runtime_tps_penalty(inputs.focused_runtime_generated_tokens_per_second);
    let memory_bytes = inputs
        .largest_stage_model_plus_kv_bytes
        .unwrap_or(inputs.slowest_stage_artifact_bytes);
    let memory_penalty = memory_bytes as f64 / 1_073_741_824.0;
    let imbalance_penalty = inputs.stage_imbalance_ratio.unwrap_or(1_000.0);
    let transfer_penalty = inputs
        .decode_transfer_bytes_per_token
        .map_or(1_000.0, |bytes| bytes as f64 / 1_048_576.0);
    validity_penalty
        + certification_penalty
        + measurement_penalty
        + latency_penalty
        + focused_runtime_penalty
        + memory_penalty
        + imbalance_penalty
        + transfer_penalty
}

fn focused_runtime_tps_penalty(generated_tps: Option<f64>) -> f64 {
    generated_tps.map_or(1_000.0, |tps| {
        if tps.is_finite() && tps > 0.0 {
            1_000.0 / tps
        } else {
            1_000.0
        }
    })
}

#[derive(Clone, Copy)]
struct RankRuntimeShape<'a> {
    ctx_size: u32,
    n_gpu_layers: i32,
    cache_type_k: &'a str,
    cache_type_v: &'a str,
    activation_wire_dtype: &'a str,
}

impl RankRuntimeShape<'_> {
    const DEFAULT_CTX_SIZE: u32 = 8192;

    fn from_args(args: &QuantPackRankArgs) -> RankRuntimeShape<'_> {
        RankRuntimeShape {
            ctx_size: args.ctx_size,
            n_gpu_layers: args.n_gpu_layers,
            cache_type_k: &args.cache_type_k,
            cache_type_v: &args.cache_type_v,
            activation_wire_dtype: &args.activation_wire_dtype,
        }
    }
}

#[derive(Clone, Copy)]
struct KvCacheEstimate {
    cache_type_k: &'static str,
    cache_type_v: &'static str,
    total_bytes: Option<u64>,
    largest_stage_bytes: Option<u64>,
    largest_stage_model_plus_kv_bytes: Option<u64>,
}

impl KvCacheEstimate {
    const DEFAULT_CACHE_TYPE: &'static str = "f16";

    fn from_preflight(
        preflight: &PreflightInput,
        ctx_size: u32,
        cache_type_k: &str,
        cache_type_v: &str,
    ) -> Self {
        let cache_type_k = canonical_cache_type(cache_type_k);
        let cache_type_v = canonical_cache_type(cache_type_v);
        let estimate = kv_cache_bytes_by_stage(preflight, ctx_size, cache_type_k, cache_type_v);
        Self {
            cache_type_k,
            cache_type_v,
            total_bytes: estimate.as_ref().map(|bytes| bytes.iter().sum()),
            largest_stage_bytes: estimate
                .as_ref()
                .and_then(|bytes| bytes.iter().copied().max()),
            largest_stage_model_plus_kv_bytes: estimate.as_ref().and_then(|bytes| {
                preflight
                    .stages
                    .iter()
                    .zip(bytes.iter())
                    .map(|(stage, kv_bytes)| stage.artifact_bytes.saturating_add(*kv_bytes))
                    .max()
            }),
        }
    }
}

fn kv_cache_bytes_by_stage(
    preflight: &PreflightInput,
    ctx_size: u32,
    cache_type_k: &str,
    cache_type_v: &str,
) -> Option<Vec<u64>> {
    let width = u64::from(preflight.activation_width?);
    let bytes_per_token_per_layer =
        width * (cache_dtype_bytes(cache_type_k) + cache_dtype_bytes(cache_type_v));
    preflight
        .stages
        .iter()
        .map(|stage| {
            let layer_count = stage_layer_count(stage)?;
            Some(u64::from(ctx_size) * layer_count * bytes_per_token_per_layer)
        })
        .collect()
}

fn stage_layer_count(stage: &PreflightStageInput) -> Option<u64> {
    let layer_start = stage.layer_start?;
    let layer_end = stage.layer_end?;
    Some(u64::from(layer_end.saturating_sub(layer_start)))
}

fn canonical_cache_type(dtype: &str) -> &'static str {
    match dtype.to_ascii_lowercase().as_str() {
        "f32" | "fp32" => "f32",
        "bf16" => "bf16",
        "f16" | "fp16" => "f16",
        "q8" | "q8_0" | "int8" | "i8" => "q8_0",
        "q4" | "q4_0" | "q4_k" => "q4_0",
        _ => KvCacheEstimate::DEFAULT_CACHE_TYPE,
    }
}

fn cache_dtype_bytes(dtype: &str) -> u64 {
    match dtype {
        "f32" | "fp32" => 4,
        "f16" | "fp16" | "bf16" => 2,
        "q8" | "q8_0" | "int8" | "i8" => 1,
        "q4" | "q4_0" | "q4_k" => 1,
        _ => 2,
    }
}

#[derive(Clone, Copy)]
struct ActivationTransferEstimate {
    wire_dtype: &'static str,
    boundary_bytes: Option<u64>,
    decode_transfer_bytes_per_token: Option<u64>,
}

impl ActivationTransferEstimate {
    const DEFAULT_WIRE_DTYPE: &'static str = "f16";

    fn from_preflight(preflight: &PreflightInput, wire_dtype: &str) -> Self {
        let wire_dtype = canonical_activation_wire_dtype(wire_dtype);
        let boundary_bytes = preflight
            .activation_width
            .map(|width| u64::from(width) * activation_dtype_bytes(wire_dtype));
        let boundary_count = preflight.stages.len().saturating_sub(1) as u64;
        Self {
            wire_dtype,
            boundary_bytes,
            decode_transfer_bytes_per_token: boundary_bytes.map(|bytes| bytes * boundary_count),
        }
    }
}

fn activation_transfer_score_input(
    certified_local_split: Option<u64>,
    direct_local_split: Option<u64>,
    preflight_estimate: Option<u64>,
) -> (Option<u64>, Option<ActivationTransferSource>) {
    if let Some(bytes) = certified_local_split {
        return (
            Some(bytes),
            Some(ActivationTransferSource::CertifiedLocalSplitChain),
        );
    }
    if let Some(bytes) = direct_local_split {
        return (
            Some(bytes),
            Some(ActivationTransferSource::DirectLocalSplitChain),
        );
    }
    if let Some(bytes) = preflight_estimate {
        return (
            Some(bytes),
            Some(ActivationTransferSource::PreflightEstimate),
        );
    }
    (None, None)
}

fn canonical_activation_wire_dtype(dtype: &str) -> &'static str {
    match dtype {
        "f32" | "fp32" => "f32",
        "f16" | "fp16" => "f16",
        "q8" | "int8" | "i8" => "q8",
        _ => ActivationTransferEstimate::DEFAULT_WIRE_DTYPE,
    }
}

fn activation_dtype_bytes(dtype: &str) -> u64 {
    match dtype {
        "f32" | "fp32" => 4,
        "f16" | "fp16" => 2,
        "q8" | "int8" | "i8" => 1,
        _ => 2,
    }
}

fn certification_penalty(status: Option<RankCertificationStatus>) -> f64 {
    match status {
        Some(RankCertificationStatus::AgentQualityCandidate) => 0.0,
        Some(RankCertificationStatus::MeasurementOnlyCandidate) => 250.0,
        Some(RankCertificationStatus::Failed) => 100_000.0,
        None => 500.0,
    }
}

struct RankNoteInputs<'a> {
    valid: bool,
    measured: bool,
    decode_mean_ms: Option<f64>,
    focused_runtime_generated_tokens_per_second: Option<f64>,
    direct_skippy_bench_evidence_count: usize,
    direct_skippy_bench_evidence_labels: &'a [&'static str],
    direct_skippy_bench_ignored_evidence_notes: &'a [String],
    certification_status: Option<RankCertificationStatus>,
    estimated_total_kv_cache_bytes: Option<u64>,
    activation_transfer_source: Option<ActivationTransferSource>,
    certification_subject_notes: &'a [String],
}

fn rank_notes(inputs: RankNoteInputs<'_>) -> Vec<String> {
    let mut notes = Vec::new();
    notes.extend(inputs.certification_subject_notes.iter().cloned());
    if !inputs.valid {
        notes.push("package preflight failed; candidate should not be selected".to_string());
    }
    if !inputs.measured {
        notes.push(
            "no measured decode profile attached; ranking falls back to package size".to_string(),
        );
    }
    if inputs.measured && inputs.decode_mean_ms.is_none() {
        notes.push("decode profile is measured but does not contain a stage mean_ms".to_string());
    }
    if let Some(tps) = inputs.focused_runtime_generated_tokens_per_second {
        notes.push(format!(
            "focused-runtime generated throughput is {:.3} tokens/sec",
            tps
        ));
    }
    match inputs.activation_transfer_source {
        Some(ActivationTransferSource::CertifiedLocalSplitChain) => {
            notes.push(
                "activation transfer scoring uses certified local split-chain evidence".to_string(),
            );
        }
        Some(ActivationTransferSource::DirectLocalSplitChain) => {
            notes.push("activation transfer scoring uses direct local split-chain evidence; run quant-pack certify to bind it to artifact hashes".to_string());
        }
        Some(ActivationTransferSource::PreflightEstimate) => {
            notes.push(
                "activation transfer scoring uses preflight activation-width estimate".to_string(),
            );
        }
        None => {
            notes.push("activation transfer estimate unavailable".to_string());
        }
    }
    if inputs.certification_status.is_none() && inputs.direct_skippy_bench_evidence_count > 0 {
        notes.push(format!(
            "{} usable skippy-bench evidence lane(s) found in evidence/: {}; run quant-pack certify after focused-runtime, chat-corpus, and token-lengths evidence are complete to bind them to artifact hashes",
            inputs.direct_skippy_bench_evidence_count,
            inputs.direct_skippy_bench_evidence_labels.join(", ")
        ));
    }
    notes.extend(
        inputs
            .direct_skippy_bench_ignored_evidence_notes
            .iter()
            .cloned(),
    );
    if inputs.estimated_total_kv_cache_bytes.is_none() {
        notes.push(
            "KV cache estimate unavailable; preflight must include activation_width and stage layer ranges"
                .to_string(),
        );
    }
    match inputs.certification_status {
        Some(RankCertificationStatus::AgentQualityCandidate) => {
            notes.push("agent-quality certification evidence is attached".to_string());
        }
        Some(RankCertificationStatus::MeasurementOnlyCandidate) => {
            notes.push(
                "certification is measurement-only; agent quality evidence is incomplete"
                    .to_string(),
            );
        }
        Some(RankCertificationStatus::Failed) => {
            notes.push("certification failed; candidate should not be selected".to_string());
        }
        None => {
            notes.push(
                "no certification.json attached; ranking treats quality as unproven".to_string(),
            );
        }
    }
    notes
}

fn stage_imbalance_ratio(stages: &[PreflightStageInput]) -> Option<f64> {
    let min = stages
        .iter()
        .map(|stage| stage.artifact_bytes)
        .filter(|bytes| *bytes > 0)
        .min()?;
    let max = stages.iter().map(|stage| stage.artifact_bytes).max()?;
    Some(max as f64 / min as f64)
}

fn build_manifest_path(run: &Path) -> PathBuf {
    if run.is_dir() {
        run.join("quant-pack-build.json")
    } else {
        run.to_path_buf()
    }
}

fn read_certification(run_dir: &Path) -> Result<Option<(PathBuf, CertificationInput)>> {
    for path in [
        run_dir.join("evidence/certification.json"),
        run_dir.join("certification.json"),
    ] {
        if path.is_file() {
            let certification = read_json::<CertificationInput>(&path)?;
            return Ok(Some((path, certification)));
        }
    }
    Ok(None)
}

fn resolve_manifest_path(run_dir: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        run_dir.join(path)
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn file_sha256(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
#[path = "rank_tests.rs"]
mod tests;
