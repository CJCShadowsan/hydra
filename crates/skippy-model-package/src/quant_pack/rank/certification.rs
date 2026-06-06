use std::path::{Path, PathBuf};

use super::{
    BuildManifestInput, CertificationInput, CertificationRuntimeShapeInput,
    CertificationTopologyInput, HashedArtifactInput, PreflightInput,
    RankCertificationSubjectStatus, RankRuntimeShape, canonical_activation_wire_dtype,
    canonical_cache_type, file_sha256, resolve_manifest_path,
};

#[derive(Clone)]
pub(super) struct CertificationSubjectCheck {
    pub(super) status: RankCertificationSubjectStatus,
    pub(super) notes: Vec<String>,
}

impl CertificationSubjectCheck {
    pub(super) fn missing() -> Self {
        Self {
            status: RankCertificationSubjectStatus::NotVerifiable,
            notes: Vec::new(),
        }
    }
}

pub(super) fn certification_subject_check(
    run_dir: &Path,
    manifest_path: &Path,
    manifest: &BuildManifestInput,
    certification: &CertificationInput,
    preflight: &PreflightInput,
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
    compare_certification_topology(
        &mut missing,
        &mut mismatches,
        certification.expected_topology.as_ref(),
        topology_from_preflight(preflight),
    );

    subject_check_result(missing, mismatches)
}

fn topology_from_preflight(preflight: &PreflightInput) -> Option<CertificationTopologyInput> {
    if preflight.stages.is_empty() {
        return None;
    }
    let ranges = preflight
        .stages
        .iter()
        .map(|stage| Some((stage.layer_start?, stage.layer_end?)))
        .collect::<Option<Vec<_>>>()?;
    let layer_end = ranges.last().map(|(_, layer_end)| *layer_end)?;
    let splits = ranges
        .iter()
        .take(ranges.len().saturating_sub(1))
        .map(|(_, layer_end)| layer_end.to_string())
        .collect::<Vec<_>>()
        .join(",");
    Some(CertificationTopologyInput {
        splits: Some(splits),
        layer_end: Some(layer_end),
        stage_count: Some(ranges.len()),
    })
}

fn compare_certification_topology(
    missing: &mut Vec<String>,
    mismatches: &mut Vec<String>,
    certified: Option<&CertificationTopologyInput>,
    current: Option<CertificationTopologyInput>,
) {
    let Some(certified) = certified else {
        return;
    };
    let Some(current) = current else {
        missing.push("expected_topology: current preflight stage ranges missing".to_string());
        return;
    };
    compare_topology_string_field(
        missing,
        mismatches,
        "splits",
        certified.splits.as_deref(),
        current.splits.as_deref().unwrap_or_default(),
    );
    compare_topology_u32_field(
        missing,
        mismatches,
        "layer_end",
        certified.layer_end,
        current.layer_end.unwrap_or_default(),
    );
    compare_topology_usize_field(
        missing,
        mismatches,
        "stage_count",
        certified.stage_count,
        current.stage_count.unwrap_or_default(),
    );
}

fn compare_topology_string_field(
    missing: &mut Vec<String>,
    mismatches: &mut Vec<String>,
    field: &str,
    certified: Option<&str>,
    expected: &str,
) {
    match certified {
        Some(actual) if actual == expected => {}
        Some(actual) => {
            mismatches.push(format!("expected_topology.{field} {actual} != {expected}"))
        }
        None => missing.push(format!(
            "expected_topology.{field}: missing from certification report"
        )),
    }
}

fn compare_topology_u32_field(
    missing: &mut Vec<String>,
    mismatches: &mut Vec<String>,
    field: &str,
    certified: Option<u32>,
    expected: u32,
) {
    match certified {
        Some(actual) if actual == expected => {}
        Some(actual) => {
            mismatches.push(format!("expected_topology.{field} {actual} != {expected}"))
        }
        None => missing.push(format!(
            "expected_topology.{field}: missing from certification report"
        )),
    }
}

fn compare_topology_usize_field(
    missing: &mut Vec<String>,
    mismatches: &mut Vec<String>,
    field: &str,
    certified: Option<usize>,
    expected: usize,
) {
    match certified {
        Some(actual) if actual == expected => {}
        Some(actual) => {
            mismatches.push(format!("expected_topology.{field} {actual} != {expected}"))
        }
        None => missing.push(format!(
            "expected_topology.{field}: missing from certification report"
        )),
    }
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
