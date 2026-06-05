use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Debug, clap::Args)]
pub(crate) struct QuantPackCertifyArgs {
    pub(crate) run: PathBuf,
    #[arg(long)]
    pub(crate) skippy_bench_report: Vec<PathBuf>,
    #[arg(long)]
    pub(crate) quality_evidence: Vec<PathBuf>,
    #[arg(long)]
    pub(crate) require_skippy_bench: bool,
    #[arg(long)]
    pub(crate) require_quality_evidence: bool,
    #[arg(long, default_value_t = 8192)]
    pub(crate) ctx_size: u32,
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    pub(crate) n_gpu_layers: i32,
    #[arg(long, default_value = "f16")]
    pub(crate) cache_type_k: String,
    #[arg(long, default_value = "f16")]
    pub(crate) cache_type_v: String,
    #[arg(long, default_value = "f16")]
    pub(crate) activation_wire_dtype: String,
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct CertificationReport {
    schema_version: u32,
    kind: String,
    created_at_unix_secs: u64,
    status: CertificationStatus,
    run_dir: String,
    candidate: String,
    pack_id: Option<String>,
    expected_model_id: String,
    expected_quantized_model: String,
    subject: CertificationSubject,
    layout_hash: Option<String>,
    gates: Vec<CertificationGate>,
    skippy_bench_reports: Vec<EvidenceReport>,
    quality_evidence: Vec<EvidenceReport>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CertificationStatus {
    Failed,
    MeasurementOnlyCandidate,
    AgentQualityCandidate,
}

#[derive(Debug, Serialize)]
struct CertificationGate {
    name: String,
    status: GateStatus,
    detail: String,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum GateStatus {
    Pass,
    Warn,
    Fail,
}

#[derive(Debug, Serialize)]
struct EvidenceReport {
    path: String,
    sha256: String,
    evidence_type: String,
    status: GateStatus,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    model_ids: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    model_paths: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime: Option<Value>,
    summary: Value,
}

#[derive(Debug, Serialize)]
struct CertificationSubject {
    expected_model_id: String,
    expected_quantized_model: HashedArtifactRef,
    package_dir: String,
    package_manifest: HashedArtifactRef,
    agent_pack: HashedArtifactRef,
    preflight: HashedArtifactRef,
    build_manifest: HashedArtifactRef,
    #[serde(skip_serializing_if = "Option::is_none")]
    quantize_run: Option<HashedArtifactRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_identity: Option<Value>,
}

#[derive(Debug, Serialize)]
struct HashedArtifactRef {
    path: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct BuildManifestInput {
    candidate: String,
    stages: usize,
    agent_pack: String,
    preflight: String,
    package: String,
    quantized_model: String,
    #[serde(default)]
    plan: Option<String>,
    #[serde(default)]
    quantize_run: Option<String>,
    #[serde(default)]
    source_identity: Option<Value>,
    #[serde(default)]
    decode_profile: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AgentPackInput {
    pack_id: String,
    quant_layout: AgentPackQuantLayoutInput,
}

#[derive(Debug, Deserialize)]
struct AgentPackQuantLayoutInput {
    #[serde(default)]
    layout_hash: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PackageManifestInput {
    model_id: String,
    #[serde(default)]
    layer_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct QuantPlanInput {
    candidates: Vec<QuantPlanCandidateInput>,
}

#[derive(Debug, Deserialize)]
struct QuantPlanCandidateInput {
    id: String,
    #[serde(default)]
    stage_hints: Vec<QuantPlanStageHintInput>,
}

#[derive(Debug, Clone, Deserialize)]
struct QuantPlanStageHintInput {
    stage_index: usize,
    layer_start: u32,
    layer_end: u32,
}

#[derive(Debug, Deserialize)]
struct PreflightInput {
    valid: bool,
    issue_count: usize,
}

#[derive(Debug, Deserialize)]
struct ProfileInput {
    measurement_status: ProfileMeasurementStatusInput,
    #[serde(default)]
    stages: Vec<ProfileStageInput>,
}

#[derive(Debug, Deserialize)]
struct ProfileMeasurementStatusInput {
    status: String,
}

#[derive(Debug, Deserialize)]
struct ProfileStageInput {
    timing: ProfileTimingInput,
}

#[derive(Debug, Deserialize)]
struct ProfileTimingInput {
    mean_ms: Option<f64>,
}

pub(crate) fn run_quant_pack_certify(args: QuantPackCertifyArgs) -> Result<()> {
    let manifest_path = build_manifest_path(&args.run);
    let manifest = read_json::<BuildManifestInput>(&manifest_path)?;
    let run_dir = manifest_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let agent_pack_path = resolve_manifest_path(&run_dir, &manifest.agent_pack);
    let agent_pack = read_json::<AgentPackInput>(&agent_pack_path)?;
    let package_dir = resolve_manifest_path(&run_dir, &manifest.package);
    let package_manifest_path = package_dir.join("model-package.json");
    let package_manifest = read_json::<PackageManifestInput>(&package_manifest_path)?;
    let quantized_model = resolve_manifest_path(&run_dir, &manifest.quantized_model);
    let preflight_path = resolve_manifest_path(&run_dir, &manifest.preflight);
    let preflight = read_json::<PreflightInput>(&preflight_path)?;
    let decode_profile = manifest
        .decode_profile
        .as_deref()
        .map(|path| read_json::<ProfileInput>(&resolve_manifest_path(&run_dir, path)))
        .transpose()?;
    let topology_expectation =
        topology_expectation(&run_dir, &manifest, package_manifest.layer_count)?;

    let skippy_bench_reports = args
        .skippy_bench_report
        .iter()
        .map(|path| summarize_evidence_for_run(&run_dir, path, EvidenceLane::SkippyBench))
        .collect::<Result<Vec<_>>>()?;
    let quality_evidence = args
        .quality_evidence
        .iter()
        .map(|path| summarize_evidence_for_run(&run_dir, path, EvidenceLane::Quality))
        .collect::<Result<Vec<_>>>()?;

    let gates = certification_gates(
        &preflight,
        decode_profile.as_ref(),
        CertificationEvidenceContext {
            skippy_bench_reports: &skippy_bench_reports,
            quality_evidence: &quality_evidence,
            expected_model_id: &package_manifest.model_id,
            expected_quantized_model: &quantized_model,
            expected_runtime: RuntimeShapeExpectation {
                ctx_size: args.ctx_size,
                n_gpu_layers: args.n_gpu_layers,
                cache_type_k: &args.cache_type_k,
                cache_type_v: &args.cache_type_v,
                activation_wire_dtype: &args.activation_wire_dtype,
            },
            expected_topology: topology_expectation.as_ref(),
            require_skippy_bench: args.require_skippy_bench,
            require_quality_evidence: args.require_quality_evidence,
        },
    );
    let status = certification_status(&gates, &quality_evidence);
    let expected_model_id = package_manifest.model_id.clone();
    let expected_quantized_model = quantized_model.display().to_string();
    let subject = certification_subject(CertificationSubjectInput {
        run_dir: &run_dir,
        manifest_path: &manifest_path,
        manifest: &manifest,
        package_dir: &package_dir,
        package_manifest_path: &package_manifest_path,
        agent_pack_path: &agent_pack_path,
        preflight_path: &preflight_path,
        quantized_model: &quantized_model,
        expected_model_id: &expected_model_id,
    })?;
    let report = CertificationReport {
        schema_version: 1,
        kind: "skippy_quant_pack_certification".to_string(),
        created_at_unix_secs: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock before Unix epoch")?
            .as_secs(),
        status,
        run_dir: run_dir.display().to_string(),
        candidate: manifest.candidate,
        pack_id: Some(agent_pack.pack_id),
        expected_model_id,
        expected_quantized_model,
        subject,
        layout_hash: agent_pack.quant_layout.layout_hash,
        gates,
        skippy_bench_reports,
        quality_evidence,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(out) = args.out {
        fs::write(&out, format!("{json}\n"))
            .with_context(|| format!("write quant-pack certification report {}", out.display()))?;
    } else {
        println!("{json}");
    }
    Ok(())
}

struct CertificationSubjectInput<'a> {
    run_dir: &'a Path,
    manifest_path: &'a Path,
    manifest: &'a BuildManifestInput,
    package_dir: &'a Path,
    package_manifest_path: &'a Path,
    agent_pack_path: &'a Path,
    preflight_path: &'a Path,
    quantized_model: &'a Path,
    expected_model_id: &'a str,
}

fn certification_subject(input: CertificationSubjectInput<'_>) -> Result<CertificationSubject> {
    Ok(CertificationSubject {
        expected_model_id: input.expected_model_id.to_string(),
        expected_quantized_model: hashed_artifact(input.quantized_model)?,
        package_dir: input.package_dir.display().to_string(),
        package_manifest: hashed_artifact(input.package_manifest_path)?,
        agent_pack: hashed_artifact(input.agent_pack_path)?,
        preflight: hashed_artifact(input.preflight_path)?,
        build_manifest: hashed_artifact(input.manifest_path)?,
        quantize_run: input
            .manifest
            .quantize_run
            .as_deref()
            .map(|path| hashed_artifact(&resolve_manifest_path(input.run_dir, path)))
            .transpose()?,
        source_identity: input.manifest.source_identity.clone(),
    })
}

fn hashed_artifact(path: &Path) -> Result<HashedArtifactRef> {
    Ok(HashedArtifactRef {
        path: path.display().to_string(),
        sha256: file_sha256(path)?,
    })
}

struct CertificationEvidenceContext<'a> {
    skippy_bench_reports: &'a [EvidenceReport],
    quality_evidence: &'a [EvidenceReport],
    expected_model_id: &'a str,
    expected_quantized_model: &'a Path,
    expected_runtime: RuntimeShapeExpectation<'a>,
    expected_topology: Option<&'a TopologyExpectation>,
    require_skippy_bench: bool,
    require_quality_evidence: bool,
}

#[derive(Clone, Copy)]
struct RuntimeShapeExpectation<'a> {
    ctx_size: u32,
    n_gpu_layers: i32,
    cache_type_k: &'a str,
    cache_type_v: &'a str,
    activation_wire_dtype: &'a str,
}

struct TopologyExpectation {
    splits: String,
    layer_end: u32,
    stage_count: usize,
}

fn certification_gates(
    preflight: &PreflightInput,
    decode_profile: Option<&ProfileInput>,
    context: CertificationEvidenceContext<'_>,
) -> Vec<CertificationGate> {
    vec![
        preflight_gate(preflight),
        decode_profile_gate(decode_profile),
        skippy_bench_gate(context.skippy_bench_reports, context.require_skippy_bench),
        evidence_subject_gate(
            context.expected_model_id,
            context.expected_quantized_model,
            context.skippy_bench_reports,
            context.quality_evidence,
            context.require_skippy_bench,
        ),
        runtime_shape_gate(context.expected_runtime, context.skippy_bench_reports),
        topology_gate(context.expected_topology, context.skippy_bench_reports),
        quality_coverage_gate(context.quality_evidence, context.require_quality_evidence),
        evidence_gate(
            "quality_evidence",
            context.quality_evidence,
            context.require_quality_evidence,
        ),
    ]
}

fn topology_expectation(
    run_dir: &Path,
    manifest: &BuildManifestInput,
    package_layer_count: Option<u32>,
) -> Result<Option<TopologyExpectation>> {
    let Some(plan) = manifest
        .plan
        .as_deref()
        .map(str::trim)
        .filter(|plan| !plan.is_empty())
    else {
        return Ok(None);
    };
    let Some(layer_count) = package_layer_count else {
        return Ok(None);
    };
    let plan_path = resolve_manifest_path(run_dir, plan);
    let quant_plan = read_json::<QuantPlanInput>(&plan_path)?;
    let candidate = quant_plan
        .candidates
        .iter()
        .find(|candidate| candidate.id == manifest.candidate)
        .with_context(|| {
            format!(
                "quant plan {} does not contain candidate {:?}",
                plan_path.display(),
                manifest.candidate
            )
        })?;
    let splits = splits_from_stage_hints(&candidate.stage_hints, layer_count, manifest.stages)
        .with_context(|| {
            format!(
                "candidate {:?} in {} has invalid stage_hints",
                candidate.id,
                plan_path.display()
            )
        })?;
    Ok(Some(TopologyExpectation {
        splits,
        layer_end: layer_count,
        stage_count: manifest.stages,
    }))
}

fn splits_from_stage_hints(
    stage_hints: &[QuantPlanStageHintInput],
    layer_count: u32,
    stages: usize,
) -> Result<String> {
    if stage_hints.len() != stages {
        bail!(
            "stage_hints must contain exactly {stages} entries, got {}",
            stage_hints.len()
        );
    }
    let mut ordered = stage_hints.to_vec();
    ordered.sort_by_key(|hint| hint.stage_index);
    let mut expected_start = 0;
    for (expected_index, hint) in ordered.iter().enumerate() {
        if hint.stage_index != expected_index {
            bail!("stage_hints indexes must be contiguous from 0");
        }
        if hint.layer_start != expected_start {
            bail!(
                "stage {} starts at layer {}, expected {}",
                hint.stage_index,
                hint.layer_start,
                expected_start
            );
        }
        if hint.layer_end <= hint.layer_start || hint.layer_end > layer_count {
            bail!(
                "stage {} range {}..{} is outside 0..{}",
                hint.stage_index,
                hint.layer_start,
                hint.layer_end,
                layer_count
            );
        }
        expected_start = hint.layer_end;
    }
    if expected_start != layer_count {
        bail!("stage_hints end at layer {expected_start}, expected {layer_count}");
    }
    Ok(ordered
        .iter()
        .take(stages.saturating_sub(1))
        .map(|hint| hint.layer_end.to_string())
        .collect::<Vec<_>>()
        .join(","))
}

fn topology_gate(
    expected: Option<&TopologyExpectation>,
    evidence: &[EvidenceReport],
) -> CertificationGate {
    let focused = evidence
        .iter()
        .filter(|report| report.evidence_type == "skippy-bench-focused-runtime")
        .collect::<Vec<_>>();
    let Some(expected) = expected else {
        return CertificationGate {
            name: "runtime_topology".to_string(),
            status: GateStatus::Warn,
            detail: "no quant-plan stage_hints available for split topology verification"
                .to_string(),
        };
    };
    if focused.is_empty() {
        return CertificationGate {
            name: "runtime_topology".to_string(),
            status: GateStatus::Warn,
            detail: format!(
                "expected splits={}, layer_end={}, stage_count={}, no focused-runtime report supplied",
                expected.splits, expected.layer_end, expected.stage_count
            ),
        };
    }

    let mut missing = 0usize;
    let mut mismatches = Vec::new();
    for report in focused {
        let Some(topology) = report.summary.get("topology") else {
            missing += 1;
            continue;
        };
        collect_topology_mismatches(&report.evidence_type, topology, expected, &mut mismatches);
    }

    CertificationGate {
        name: "runtime_topology".to_string(),
        status: if mismatches.is_empty() {
            if missing == 0 {
                GateStatus::Pass
            } else {
                GateStatus::Fail
            }
        } else {
            GateStatus::Fail
        },
        detail: if mismatches.is_empty() && missing == 0 {
            format!(
                "expected splits={}, layer_end={}, stage_count={} verified",
                expected.splits, expected.layer_end, expected.stage_count
            )
        } else if mismatches.is_empty() {
            format!(
                "expected splits={}, layer_end={}, stage_count={}, focused reports missing topology blocks={missing}",
                expected.splits, expected.layer_end, expected.stage_count
            )
        } else {
            format!(
                "expected splits={}, layer_end={}, stage_count={}, mismatched evidence: {}",
                expected.splits,
                expected.layer_end,
                expected.stage_count,
                mismatches.join("; ")
            )
        },
    }
}

fn collect_topology_mismatches(
    evidence_type: &str,
    topology: &Value,
    expected: &TopologyExpectation,
    mismatches: &mut Vec<String>,
) {
    if let Some(splits) = topology.get("splits").and_then(Value::as_str)
        && splits != expected.splits
    {
        mismatches.push(format!(
            "{evidence_type} splits {splits} != {}",
            expected.splits
        ));
    }
    if let Some(layer_end) = topology.get("layer_end").and_then(Value::as_u64)
        && layer_end != u64::from(expected.layer_end)
    {
        mismatches.push(format!(
            "{evidence_type} layer_end {layer_end} != {}",
            expected.layer_end
        ));
    }
    if let Some(stage_count) = topology.get("stage_count").and_then(Value::as_u64)
        && stage_count != expected.stage_count as u64
    {
        mismatches.push(format!(
            "{evidence_type} stage_count {stage_count} != {}",
            expected.stage_count
        ));
    }
}

fn runtime_shape_gate(
    expected: RuntimeShapeExpectation<'_>,
    evidence: &[EvidenceReport],
) -> CertificationGate {
    let focused = evidence
        .iter()
        .filter(|report| report.evidence_type == "skippy-bench-focused-runtime")
        .collect::<Vec<_>>();
    if focused.is_empty() {
        return CertificationGate {
            name: "runtime_shape".to_string(),
            status: GateStatus::Warn,
            detail: "no focused-runtime report supplied for runtime-shape verification".to_string(),
        };
    }

    let mut missing = 0usize;
    let mut mismatches = Vec::new();
    for report in focused {
        let Some(runtime) = report.runtime.as_ref() else {
            missing += 1;
            continue;
        };
        collect_runtime_mismatches(&report.evidence_type, runtime, expected, &mut mismatches);
    }

    CertificationGate {
        name: "runtime_shape".to_string(),
        status: if mismatches.is_empty() {
            if missing == 0 {
                GateStatus::Pass
            } else {
                GateStatus::Warn
            }
        } else {
            GateStatus::Fail
        },
        detail: if mismatches.is_empty() {
            format!(
                "expected ctx_size={}, n_gpu_layers={}, cache_type_k={}, cache_type_v={}, activation_wire_dtype={}, focused reports missing runtime blocks={missing}",
                expected.ctx_size,
                expected.n_gpu_layers,
                expected.cache_type_k,
                expected.cache_type_v,
                expected.activation_wire_dtype
            )
        } else {
            format!(
                "expected ctx_size={}, n_gpu_layers={}, cache_type_k={}, cache_type_v={}, activation_wire_dtype={}, mismatched evidence: {}",
                expected.ctx_size,
                expected.n_gpu_layers,
                expected.cache_type_k,
                expected.cache_type_v,
                expected.activation_wire_dtype,
                mismatches.join("; ")
            )
        },
    }
}

fn collect_runtime_mismatches(
    evidence_type: &str,
    runtime: &Value,
    expected: RuntimeShapeExpectation<'_>,
    mismatches: &mut Vec<String>,
) {
    push_u64_runtime_mismatch(
        evidence_type,
        runtime,
        "ctx_size",
        u64::from(expected.ctx_size),
        mismatches,
    );
    push_i64_runtime_mismatch(
        evidence_type,
        runtime,
        "n_gpu_layers",
        i64::from(expected.n_gpu_layers),
        mismatches,
    );
    push_cache_runtime_mismatch(
        evidence_type,
        runtime,
        "cache_type_k",
        expected.cache_type_k,
        mismatches,
    );
    push_cache_runtime_mismatch(
        evidence_type,
        runtime,
        "cache_type_v",
        expected.cache_type_v,
        mismatches,
    );
    if let Some(dtype) = runtime.get("activation_wire_dtype").and_then(Value::as_str)
        && !activation_wire_dtype_matches(expected.activation_wire_dtype, dtype)
    {
        mismatches.push(format!(
            "{evidence_type} activation_wire_dtype {dtype} != {}",
            expected.activation_wire_dtype
        ));
    }
}

fn push_u64_runtime_mismatch(
    evidence_type: &str,
    runtime: &Value,
    field: &str,
    expected: u64,
    mismatches: &mut Vec<String>,
) {
    if let Some(actual) = runtime.get(field).and_then(Value::as_u64)
        && actual != expected
    {
        mismatches.push(format!("{evidence_type} {field} {actual} != {expected}"));
    }
}

fn push_i64_runtime_mismatch(
    evidence_type: &str,
    runtime: &Value,
    field: &str,
    expected: i64,
    mismatches: &mut Vec<String>,
) {
    if let Some(actual) = runtime.get(field).and_then(Value::as_i64)
        && actual != expected
    {
        mismatches.push(format!("{evidence_type} {field} {actual} != {expected}"));
    }
}

fn push_cache_runtime_mismatch(
    evidence_type: &str,
    runtime: &Value,
    field: &str,
    expected: &str,
    mismatches: &mut Vec<String>,
) {
    if let Some(actual) = runtime.get(field).and_then(Value::as_str)
        && !cache_type_matches(expected, actual)
    {
        mismatches.push(format!("{evidence_type} {field} {actual} != {expected}"));
    }
}

fn cache_type_matches(expected: &str, actual: &str) -> bool {
    expected.eq_ignore_ascii_case(actual)
}

fn activation_wire_dtype_matches(expected: &str, actual: &str) -> bool {
    canonical_activation_wire_dtype(expected) == canonical_activation_wire_dtype(actual)
}

fn canonical_activation_wire_dtype(dtype: &str) -> &str {
    match dtype {
        "f32" | "fp32" => "f32",
        "f16" | "fp16" => "f16",
        "q8" | "int8" | "i8" => "q8",
        other => other,
    }
}

fn evidence_subject_gate(
    expected_model_id: &str,
    expected_quantized_model: &Path,
    skippy_bench_reports: &[EvidenceReport],
    quality_evidence: &[EvidenceReport],
    require_skippy_bench: bool,
) -> CertificationGate {
    let mut mismatches = Vec::new();
    let mut missing_skippy_subjects = Vec::new();

    for report in skippy_bench_reports {
        collect_subject_mismatches(
            report,
            expected_model_id,
            expected_quantized_model,
            &mut mismatches,
        );
        if report.model_ids.is_empty() && report.model_paths.is_empty() {
            missing_skippy_subjects.push(report.evidence_type.clone());
        }
    }
    for report in quality_evidence {
        collect_subject_mismatches(
            report,
            expected_model_id,
            expected_quantized_model,
            &mut mismatches,
        );
    }

    CertificationGate {
        name: "evidence_subject".to_string(),
        status: if mismatches.is_empty() {
            missing_subject_status(&missing_skippy_subjects, require_skippy_bench)
        } else {
            GateStatus::Fail
        },
        detail: if mismatches.is_empty() {
            format!(
                "expected model_id={}, expected quantized_model={}, missing skippy subjects: {}",
                expected_model_id,
                expected_quantized_model.display(),
                if missing_skippy_subjects.is_empty() {
                    "none".to_string()
                } else {
                    missing_skippy_subjects.join(", ")
                }
            )
        } else {
            format!(
                "expected model_id={}, expected quantized_model={}, mismatched evidence: {}",
                expected_model_id,
                expected_quantized_model.display(),
                mismatches.join("; ")
            )
        },
    }
}

fn missing_subject_status(
    missing_skippy_subjects: &[String],
    require_skippy_bench: bool,
) -> GateStatus {
    if missing_skippy_subjects.is_empty() {
        GateStatus::Pass
    } else if require_skippy_bench {
        GateStatus::Fail
    } else {
        GateStatus::Warn
    }
}

fn collect_subject_mismatches(
    report: &EvidenceReport,
    expected_model_id: &str,
    expected_quantized_model: &Path,
    mismatches: &mut Vec<String>,
) {
    for model_id in &report.model_ids {
        if model_id != expected_model_id {
            mismatches.push(format!(
                "{} model_id {} != {}",
                report.evidence_type, model_id, expected_model_id
            ));
        }
    }
    for model_path in &report.model_paths {
        if !paths_match(expected_quantized_model, Path::new(model_path)) {
            mismatches.push(format!(
                "{} model_path {} != {}",
                report.evidence_type,
                model_path,
                expected_quantized_model.display()
            ));
        }
    }
}

fn preflight_gate(preflight: &PreflightInput) -> CertificationGate {
    CertificationGate {
        name: "package_preflight".to_string(),
        status: if preflight.valid {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        detail: format!(
            "preflight valid={}, issue_count={}",
            preflight.valid, preflight.issue_count
        ),
    }
}

fn decode_profile_gate(profile: Option<&ProfileInput>) -> CertificationGate {
    let Some(profile) = profile else {
        return CertificationGate {
            name: "decode_profile".to_string(),
            status: GateStatus::Warn,
            detail: "no decode-profile.json attached to quant-pack build".to_string(),
        };
    };
    let mean_ms = profile
        .stages
        .first()
        .and_then(|stage| stage.timing.mean_ms);
    let measured = profile.measurement_status.status == "measured" && mean_ms.is_some();
    CertificationGate {
        name: "decode_profile".to_string(),
        status: if measured {
            GateStatus::Pass
        } else {
            GateStatus::Fail
        },
        detail: format!(
            "measurement_status={}, stage0_mean_ms={:?}",
            profile.measurement_status.status, mean_ms
        ),
    }
}

fn skippy_bench_gate(evidence: &[EvidenceReport], required: bool) -> CertificationGate {
    const REQUIRED_TYPES: &[&str] = &[
        "skippy-bench-focused-runtime",
        "skippy-bench-chat-corpus",
        "skippy-bench-long-context-chat-corpus",
        "skippy-bench-token-lengths",
    ];

    if evidence.is_empty() {
        return CertificationGate {
            name: "skippy_bench".to_string(),
            status: if required {
                GateStatus::Fail
            } else {
                GateStatus::Warn
            },
            detail: "no evidence reports supplied".to_string(),
        };
    }

    let failed = evidence
        .iter()
        .filter(|report| report.status == GateStatus::Fail)
        .count();
    let unknown = evidence
        .iter()
        .filter(|report| report.status == GateStatus::Warn)
        .count();
    let missing = REQUIRED_TYPES
        .iter()
        .copied()
        .filter(|kind| !has_passing_evidence_type(evidence, kind))
        .collect::<Vec<_>>();
    CertificationGate {
        name: "skippy_bench".to_string(),
        status: if failed > 0 || (required && !missing.is_empty()) {
            GateStatus::Fail
        } else if !missing.is_empty() || unknown > 0 {
            GateStatus::Warn
        } else {
            GateStatus::Pass
        },
        detail: format!(
            "{} reports, {} failed, {} informational, missing required evidence: {}",
            evidence.len(),
            failed,
            unknown,
            if missing.is_empty() {
                "none".to_string()
            } else {
                missing.join(", ")
            }
        ),
    }
}

fn has_passing_evidence_type(evidence: &[EvidenceReport], evidence_type: &str) -> bool {
    evidence
        .iter()
        .any(|report| report.evidence_type == evidence_type && report.status == GateStatus::Pass)
}

fn evidence_gate(name: &str, evidence: &[EvidenceReport], required: bool) -> CertificationGate {
    if evidence.is_empty() {
        return CertificationGate {
            name: name.to_string(),
            status: if required {
                GateStatus::Fail
            } else {
                GateStatus::Warn
            },
            detail: "no evidence reports supplied".to_string(),
        };
    }
    let failed = evidence
        .iter()
        .filter(|report| report.status == GateStatus::Fail)
        .count();
    let unknown = evidence
        .iter()
        .filter(|report| report.status == GateStatus::Warn)
        .count();
    CertificationGate {
        name: name.to_string(),
        status: if failed > 0 {
            GateStatus::Fail
        } else if unknown > 0 {
            GateStatus::Warn
        } else {
            GateStatus::Pass
        },
        detail: format!(
            "{} reports, {} failed, {} informational",
            evidence.len(),
            failed,
            unknown
        ),
    }
}

fn quality_coverage_gate(evidence: &[EvidenceReport], required: bool) -> CertificationGate {
    const REQUIRED_TYPES: &[&str] = &["quality-agent-tool-call", "quality-kv-tool-loop"];

    if evidence.is_empty() {
        return CertificationGate {
            name: "quality_coverage".to_string(),
            status: if required {
                GateStatus::Fail
            } else {
                GateStatus::Warn
            },
            detail: "no quality evidence reports supplied".to_string(),
        };
    }

    let missing = REQUIRED_TYPES
        .iter()
        .copied()
        .filter(|kind| !has_passing_evidence_type(evidence, kind))
        .collect::<Vec<_>>();
    CertificationGate {
        name: "quality_coverage".to_string(),
        status: if required && !missing.is_empty() {
            GateStatus::Fail
        } else if missing.is_empty() {
            GateStatus::Pass
        } else {
            GateStatus::Warn
        },
        detail: format!(
            "quality evidence types: {}; missing required quality evidence: {}",
            evidence
                .iter()
                .map(|report| report.evidence_type.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            if missing.is_empty() {
                "none".to_string()
            } else {
                missing.join(", ")
            }
        ),
    }
}

fn certification_status(
    gates: &[CertificationGate],
    quality_evidence: &[EvidenceReport],
) -> CertificationStatus {
    if gates.iter().any(|gate| gate.status == GateStatus::Fail) {
        CertificationStatus::Failed
    } else if !quality_evidence.is_empty()
        && gates.iter().all(|gate| gate.status == GateStatus::Pass)
    {
        CertificationStatus::AgentQualityCandidate
    } else {
        CertificationStatus::MeasurementOnlyCandidate
    }
}

#[derive(Clone, Copy)]
enum EvidenceLane {
    SkippyBench,
    Quality,
}

struct EvidenceSummary {
    evidence_type: String,
    status: GateStatus,
    summary: Value,
    model_ids: Vec<String>,
    model_paths: Vec<String>,
    runtime: Option<Value>,
}

fn summarize_evidence_for_run(
    run_dir: &Path,
    path: &Path,
    lane: EvidenceLane,
) -> Result<EvidenceReport> {
    summarize_evidence_with_base(Some(run_dir), path, lane)
}

fn summarize_evidence_with_base(
    run_dir: Option<&Path>,
    path: &Path,
    lane: EvidenceLane,
) -> Result<EvidenceReport> {
    if path.is_dir() {
        return summarize_evidence_dir(run_dir, path, lane);
    }
    let value = read_json_value(path).ok();
    let mut summary = match (lane, value.as_ref()) {
        (EvidenceLane::SkippyBench, Some(value)) => summarize_skippy_bench_json(value),
        (EvidenceLane::Quality, Some(value)) => summarize_quality_json(value),
        (_, None) if path.extension().and_then(|ext| ext.to_str()) == Some("jsonl") => {
            summarize_jsonl(path)?
        }
        (_, None) => EvidenceSummary::new(
            "unknown-file",
            GateStatus::Warn,
            serde_json::json!({"file_name": path.file_name().and_then(|name| name.to_str())}),
        ),
    };
    if let EvidenceLane::Quality = lane {
        summary = classify_quality_evidence(path, summary);
    }
    Ok(EvidenceReport {
        path: evidence_report_path(run_dir, path)?,
        sha256: file_sha256(path)?,
        evidence_type: summary.evidence_type,
        status: summary.status,
        model_ids: summary.model_ids,
        model_paths: summary.model_paths,
        runtime: summary.runtime,
        summary: summary.summary,
    })
}

fn summarize_evidence_dir(
    run_dir: Option<&Path>,
    path: &Path,
    lane: EvidenceLane,
) -> Result<EvidenceReport> {
    let summary = path.join("summary.json");
    if summary.is_file() {
        return summarize_evidence_with_base(run_dir, &summary, lane);
    }
    let focused = path.join("focused-runtime-report.json");
    if focused.is_file() {
        return summarize_evidence_with_base(run_dir, &focused, lane);
    }
    let results = path.join("results.jsonl");
    if results.is_file() {
        return summarize_evidence_with_base(run_dir, &results, lane);
    }
    bail!(
        "evidence directory {} does not contain summary.json, focused-runtime-report.json, or results.jsonl",
        path.display()
    )
}

fn evidence_report_path(run_dir: Option<&Path>, path: &Path) -> Result<String> {
    let Some(run_dir) = run_dir else {
        return Ok(path.display().to_string());
    };
    let absolute_path =
        fs::canonicalize(path).with_context(|| format!("canonicalize {}", path.display()))?;
    let absolute_run_dir =
        fs::canonicalize(run_dir).with_context(|| format!("canonicalize {}", run_dir.display()))?;
    if let Ok(relative) = absolute_path.strip_prefix(&absolute_run_dir) {
        return Ok(relative.display().to_string());
    }
    Ok(absolute_path.display().to_string())
}

impl EvidenceSummary {
    fn new(evidence_type: &str, status: GateStatus, summary: Value) -> Self {
        Self {
            evidence_type: evidence_type.to_string(),
            status,
            summary,
            model_ids: Vec::new(),
            model_paths: Vec::new(),
            runtime: None,
        }
    }

    fn with_model_ids(mut self, model_ids: Vec<String>) -> Self {
        self.model_ids = model_ids;
        self
    }

    fn with_model_paths(mut self, model_paths: Vec<String>) -> Self {
        self.model_paths = model_paths;
        self
    }

    fn with_runtime(mut self, runtime: Option<Value>) -> Self {
        self.runtime = runtime;
        self
    }
}

fn summarize_skippy_bench_json(value: &Value) -> EvidenceSummary {
    if value.get("mode").and_then(Value::as_str) == Some("local-split-chain-binary") {
        return EvidenceSummary::new(
            "skippy-bench-local-split-chain",
            local_split_chain_gate_status(value),
            serde_json::json!({
                "mode": value.get("mode"),
                "activation_width": value.get("activation_width"),
                "wire_dtype": value.get("wire_dtype"),
                "splits": value.get("splits"),
                "layer_end": value.get("layer_end"),
                "stages": value.get("stages"),
                "boundary_transfers": value.get("boundary_transfers"),
                "predicted_token": value.get("predicted_token"),
            }),
        )
        .with_model_ids(model_ids_from_value(value));
    }
    if value.get("scenario").is_some() && value.get("latency_ms").is_some() {
        return EvidenceSummary::new(
            "skippy-bench-focused-runtime",
            focused_runtime_gate_status(value),
            serde_json::json!({
                "scenario": value.get("scenario"),
                "mode": value.get("mode"),
                "stage_count": value.get("stage_count"),
                "topology": value.get("topology"),
                "latency_ms": value.get("latency_ms"),
                "throughput_tokens_per_second": value.get("throughput_tokens_per_second"),
                "token_counts": value.get("token_counts"),
            }),
        )
        .with_model_ids(model_ids_from_value(value))
        .with_runtime(value.get("runtime").cloned());
    }
    if let Some(summary) = value.get("summary") {
        let errors = summary
            .get("errors")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let evidence_type = if is_long_context_chat_report(value) {
            "skippy-bench-long-context-chat-corpus"
        } else {
            "skippy-bench-chat-corpus"
        };
        return EvidenceSummary::new(
            evidence_type,
            if errors == 0 {
                GateStatus::Pass
            } else {
                GateStatus::Fail
            },
            summary.clone(),
        )
        .with_model_ids(model_ids_from_value(value));
    }
    if value.get("exceeds_context").is_some() && value.get("row_count").is_some() {
        let exceeds = value
            .get("exceeds_context")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        return EvidenceSummary::new(
            "skippy-bench-token-lengths",
            if exceeds == 0 {
                GateStatus::Pass
            } else {
                GateStatus::Fail
            },
            value.clone(),
        )
        .with_model_paths(model_paths_from_value(value));
    }
    EvidenceSummary::new("skippy-bench-unknown-json", GateStatus::Warn, value.clone())
}

fn is_long_context_chat_report(value: &Value) -> bool {
    value
        .get("prompt_corpus")
        .and_then(Value::as_str)
        .is_some_and(|path| path.contains("long-context"))
}

fn local_split_chain_gate_status(value: &Value) -> GateStatus {
    if value.get("predicted_token").is_none() {
        return GateStatus::Fail;
    }
    let has_payload = value
        .get("stages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|stage| {
            stage
                .get("payload_bytes")
                .and_then(Value::as_u64)
                .is_some_and(|bytes| bytes > 0)
                && stage
                    .get("wire_payload_bytes")
                    .and_then(Value::as_u64)
                    .is_some_and(|bytes| bytes > 0)
        });
    if has_payload {
        GateStatus::Pass
    } else {
        GateStatus::Fail
    }
}

fn focused_runtime_gate_status(value: &Value) -> GateStatus {
    if value.get("mode").and_then(Value::as_str) != Some("executed") {
        return GateStatus::Fail;
    }
    if focused_runtime_generated_tps(value).is_none() {
        return GateStatus::Fail;
    }
    if focused_runtime_decode_p50_ms(value).is_none() {
        return GateStatus::Fail;
    }
    GateStatus::Pass
}

fn focused_runtime_generated_tps(value: &Value) -> Option<f64> {
    value
        .get("throughput_tokens_per_second")
        .and_then(|throughput| throughput.get("generated"))
        .and_then(value_as_f64)
        .filter(|value| *value > 0.0)
}

fn focused_runtime_decode_p50_ms(value: &Value) -> Option<f64> {
    value
        .get("latency_ms")
        .and_then(|latency| latency.get("decode_elapsed_ms_p50"))
        .and_then(value_as_f64)
}

fn value_as_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_u64().map(|value| value as f64))
        .or_else(|| value.as_i64().map(|value| value as f64))
}

fn summarize_quality_json(value: &Value) -> EvidenceSummary {
    if let Some(ok) = value.get("ok").and_then(Value::as_bool) {
        let failed = value
            .get("failed")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        return EvidenceSummary::new(
            "quality-summary",
            if ok && failed == 0 {
                GateStatus::Pass
            } else {
                GateStatus::Fail
            },
            value.clone(),
        )
        .with_model_ids(model_ids_from_value(value))
        .with_model_paths(model_paths_from_value(value));
    }
    summarize_skippy_bench_json(value)
}

fn classify_quality_evidence(path: &Path, mut summary: EvidenceSummary) -> EvidenceSummary {
    if quality_evidence_matches(path, &summary.summary, QualityEvidenceKind::AgentToolCall) {
        summary.evidence_type = "quality-agent-tool-call".to_string();
    } else if quality_evidence_matches(path, &summary.summary, QualityEvidenceKind::KvToolLoop) {
        summary.evidence_type = "quality-kv-tool-loop".to_string();
    }
    summary
}

#[derive(Clone, Copy)]
enum QualityEvidenceKind {
    AgentToolCall,
    KvToolLoop,
}

fn quality_evidence_matches(path: &Path, summary: &Value, kind: QualityEvidenceKind) -> bool {
    let path_text = path.display().to_string().to_ascii_lowercase();
    match kind {
        QualityEvidenceKind::AgentToolCall => {
            path_text.contains("agent-tool-call")
                || path_text.contains("tool-call-reliability")
                || phases_contain(summary, &["tool_call", "stream_tool_call"])
        }
        QualityEvidenceKind::KvToolLoop => {
            path_text.contains("kv-tool-loop")
                || path_text.contains("kv_tool_loop")
                || phases_contain(
                    summary,
                    &[
                        "tool_loop",
                        "overlap_tool_loop",
                        "same_prefix_cache",
                        "exact_prefix_cache",
                        "native_log_scan",
                    ],
                )
        }
    }
}

fn phases_contain(summary: &Value, needles: &[&str]) -> bool {
    summary
        .get("phases")
        .and_then(Value::as_object)
        .is_some_and(|phases| needles.iter().any(|needle| phases.contains_key(*needle)))
}

fn summarize_jsonl(path: &Path) -> Result<EvidenceSummary> {
    let contents = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut total = 0usize;
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut phases = BTreeMap::<String, PhaseCounts>::new();
    let mut model_ids = BTreeSet::new();
    let mut model_paths = BTreeSet::new();
    for (index, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value = serde_json::from_str::<Value>(line)
            .with_context(|| format!("parse JSONL line {} in {}", index + 1, path.display()))?;
        model_ids.extend(model_ids_from_value(&value));
        model_paths.extend(model_paths_from_value(&value));
        if let Some(phase) = value.get("phase").and_then(Value::as_str) {
            phases.entry(phase.to_string()).or_default().total += 1;
        }
        let Some(ok) = value.get("ok").and_then(Value::as_bool) else {
            continue;
        };
        total += 1;
        if ok {
            passed += 1;
            if let Some(phase) = value.get("phase").and_then(Value::as_str) {
                phases.entry(phase.to_string()).or_default().passed += 1;
            }
        } else {
            failed += 1;
            if let Some(phase) = value.get("phase").and_then(Value::as_str) {
                phases.entry(phase.to_string()).or_default().failed += 1;
            }
        }
    }
    let status = if total == 0 {
        GateStatus::Warn
    } else if failed == 0 {
        GateStatus::Pass
    } else {
        GateStatus::Fail
    };
    Ok(EvidenceSummary::new(
        "jsonl-ok-results",
        status,
        serde_json::json!({
            "total": total,
            "passed": passed,
            "failed": failed,
            "phases": phases,
        }),
    )
    .with_model_ids(model_ids.into_iter().collect())
    .with_model_paths(model_paths.into_iter().collect()))
}

#[derive(Default, Serialize)]
struct PhaseCounts {
    total: usize,
    passed: usize,
    failed: usize,
}

fn model_ids_from_value(value: &Value) -> Vec<String> {
    let mut ids = BTreeSet::new();
    insert_string_field(&mut ids, value.get("model_id"));
    insert_string_field(&mut ids, value.get("model"));
    insert_string_field(&mut ids, value.pointer("/model/model_id"));
    insert_string_field(&mut ids, value.pointer("/model_identity/model_id"));
    insert_string_field(&mut ids, value.get("models"));
    ids.into_iter().collect()
}

fn model_paths_from_value(value: &Value) -> Vec<String> {
    let mut paths = BTreeSet::new();
    insert_string_field(&mut paths, value.get("model_path"));
    insert_string_field(&mut paths, value.pointer("/model/path"));
    paths.into_iter().collect()
}

fn insert_string_field(values: &mut BTreeSet<String>, value: Option<&Value>) {
    match value {
        Some(Value::String(text)) if !text.is_empty() => {
            values.insert(text.clone());
        }
        Some(Value::Array(items)) => {
            for item in items {
                insert_string_field(values, Some(item));
            }
        }
        _ => {}
    }
}

fn paths_match(expected: &Path, observed: &Path) -> bool {
    if expected == observed {
        return true;
    }
    match (expected.canonicalize(), observed.canonicalize()) {
        (Ok(expected), Ok(observed)) => expected == observed,
        _ => false,
    }
}

fn build_manifest_path(run: &Path) -> PathBuf {
    if run.is_dir() {
        run.join("quant-pack-build.json")
    } else {
        run.to_path_buf()
    }
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

fn read_json_value(path: &Path) -> Result<Value> {
    read_json(path)
}

fn file_sha256(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
#[path = "quant_certify_tests.rs"]
mod tests;
