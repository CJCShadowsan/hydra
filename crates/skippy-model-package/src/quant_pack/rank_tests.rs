use super::*;

#[test]
fn stage_imbalance_uses_slowest_over_smallest_stage() {
    let stages = vec![
        PreflightStageInput {
            artifact_bytes: 100,
            layer_start: Some(0),
            layer_end: Some(1),
        },
        PreflightStageInput {
            artifact_bytes: 250,
            layer_start: Some(1),
            layer_end: Some(2),
        },
    ];

    assert_eq!(stage_imbalance_ratio(&stages), Some(2.5));
}

#[test]
fn activation_transfer_estimate_uses_activation_width_and_stage_boundaries() {
    let preflight = PreflightInput {
        valid: true,
        activation_width: Some(4096),
        stages: vec![
            PreflightStageInput {
                artifact_bytes: 100,
                layer_start: Some(0),
                layer_end: Some(1),
            },
            PreflightStageInput {
                artifact_bytes: 100,
                layer_start: Some(1),
                layer_end: Some(2),
            },
            PreflightStageInput {
                artifact_bytes: 100,
                layer_start: Some(2),
                layer_end: Some(3),
            },
        ],
    };
    let estimate = ActivationTransferEstimate::from_preflight(&preflight, "f16");
    let q8_estimate = ActivationTransferEstimate::from_preflight(&preflight, "q8");

    assert_eq!(estimate.wire_dtype, "f16");
    assert_eq!(estimate.boundary_bytes, Some(8192));
    assert_eq!(estimate.decode_transfer_bytes_per_token, Some(16_384));
    assert_eq!(q8_estimate.wire_dtype, "q8");
    assert_eq!(q8_estimate.boundary_bytes, Some(4096));
    assert_eq!(q8_estimate.decode_transfer_bytes_per_token, Some(8192));
}

#[test]
fn kv_cache_estimate_uses_context_cache_dtype_width_and_stage_layers() {
    let preflight = PreflightInput {
        valid: true,
        activation_width: Some(4096),
        stages: vec![
            PreflightStageInput {
                artifact_bytes: 10_000,
                layer_start: Some(0),
                layer_end: Some(2),
            },
            PreflightStageInput {
                artifact_bytes: 25_000,
                layer_start: Some(2),
                layer_end: Some(5),
            },
        ],
    };

    let estimate = KvCacheEstimate::from_preflight(&preflight, 1024, "f16", "q8_0");

    assert_eq!(estimate.cache_type_k, "f16");
    assert_eq!(estimate.cache_type_v, "q8_0");
    assert_eq!(estimate.total_bytes, Some(62_914_560));
    assert_eq!(estimate.largest_stage_bytes, Some(37_748_736));
    assert_eq!(estimate.largest_stage_model_plus_kv_bytes, Some(37_773_736));
}

#[test]
fn ranking_prefers_valid_measured_lower_decode_candidate() {
    let mut candidates = [
        ranked_candidate("unmeasured", true, false, None, None, 100, Some(1.0)),
        ranked_candidate("fast", true, true, None, Some(12.0), 200, Some(1.2)),
        ranked_candidate("slow", true, true, None, Some(20.0), 100, Some(1.0)),
        ranked_candidate("invalid", false, true, None, Some(1.0), 10, Some(1.0)),
    ];

    candidates.sort_by(compare_candidates);

    assert_eq!(candidates[0].candidate, "fast");
    assert_eq!(candidates[1].candidate, "slow");
    assert_eq!(candidates[2].candidate, "unmeasured");
    assert_eq!(candidates[3].candidate, "invalid");
}

#[test]
fn ranking_prefers_agent_quality_certified_candidate_over_uncertified_speed() {
    let mut candidates = [
        ranked_candidate(
            "uncertified-fast",
            true,
            true,
            None,
            Some(12.0),
            100,
            Some(1.0),
        ),
        ranked_candidate(
            "certified-slower",
            true,
            true,
            Some(RankCertificationStatus::AgentQualityCandidate),
            Some(20.0),
            100,
            Some(1.0),
        ),
        ranked_candidate(
            "failed-cert",
            true,
            true,
            Some(RankCertificationStatus::Failed),
            Some(1.0),
            100,
            Some(1.0),
        ),
    ];

    candidates.sort_by(compare_candidates);

    assert_eq!(candidates[0].candidate, "certified-slower");
    assert_eq!(candidates[2].candidate, "failed-cert");
}

#[test]
fn certification_summary_counts_failed_and_warned_gates() {
    let certification = CertificationInput {
        status: RankCertificationStatus::MeasurementOnlyCandidate,
        runtime_shape: None,
        expected_topology: None,
        subject: None,
        gates: vec![
            CertificationGateInput {
                status: CertificationGateStatusInput::Pass,
            },
            CertificationGateInput {
                status: CertificationGateStatusInput::Warn,
            },
            CertificationGateInput {
                status: CertificationGateStatusInput::Fail,
            },
        ],
        skippy_bench_reports: vec![serde_json::json!({
            "evidence_type": "focused-runtime",
            "status": "pass"
        })],
        quality_evidence: vec![serde_json::json!({
            "evidence_type": "jsonl-ok-results",
            "status": "pass"
        })],
    };

    let summary = CertificationRankSummary::from_certification(&certification);

    assert_eq!(
        summary.status,
        RankCertificationStatus::MeasurementOnlyCandidate
    );
    assert_eq!(summary.failed_gates, 1);
    assert_eq!(summary.warned_gates, 1);
    assert_eq!(summary.skippy_bench_evidence_count, 1);
    assert_eq!(summary.quality_evidence_count, 1);
}

#[test]
fn certification_summary_extracts_focused_runtime_measurements_for_ranking() {
    let certification = CertificationInput {
        status: RankCertificationStatus::MeasurementOnlyCandidate,
        runtime_shape: None,
        expected_topology: None,
        subject: None,
        gates: Vec::new(),
        skippy_bench_reports: vec![
            serde_json::json!({
                "evidence_type": "skippy-bench-focused-runtime",
                "status": "pass",
                "summary": {
                    "throughput_tokens_per_second": {"generated": 42.5},
                    "latency_ms": {"decode_elapsed_ms_p50": 120}
                }
            }),
            serde_json::json!({
                "evidence_type": "skippy-bench-local-split-chain",
                "status": "pass",
                "summary": {
                    "mode": "local-split-chain-binary",
                    "predicted_token": 42,
                    "boundary_transfers": [
                        {"wire_payload_bytes": 1024},
                        {"wire_payload_bytes": 2048}
                    ]
                }
            }),
        ],
        quality_evidence: Vec::new(),
    };

    let summary = CertificationRankSummary::from_certification(&certification);

    assert_eq!(summary.focused_runtime_generated_tps, Some(42.5));
    assert_eq!(summary.focused_runtime_decode_elapsed_ms_p50, Some(120.0));
    assert_eq!(
        summary.local_split_decode_transfer_bytes_per_token,
        Some(3072)
    );
}

#[test]
fn certification_summary_ignores_failed_focused_runtime_measurements() {
    let certification = CertificationInput {
        status: RankCertificationStatus::Failed,
        runtime_shape: None,
        expected_topology: None,
        subject: None,
        gates: Vec::new(),
        skippy_bench_reports: vec![serde_json::json!({
            "evidence_type": "skippy-bench-focused-runtime",
            "status": "fail",
            "summary": {
                "throughput_tokens_per_second": {"generated": 999.0},
                "latency_ms": {"decode_elapsed_ms_p50": 1}
            }
        })],
        quality_evidence: vec![serde_json::json!({
            "evidence_type": "quality-agent-tool-call",
            "status": "fail"
        })],
    };

    let summary = CertificationRankSummary::from_certification(&certification);

    assert_eq!(summary.skippy_bench_evidence_count, 0);
    assert_eq!(summary.quality_evidence_count, 0);
    assert_eq!(summary.focused_runtime_generated_tps, None);
    assert_eq!(summary.focused_runtime_decode_elapsed_ms_p50, None);
}

#[test]
fn direct_skippy_bench_summary_reads_generated_evidence_dir() {
    let dir = unique_test_dir("direct-skippy-bench-evidence");
    let evidence_dir = dir.join("evidence");
    fs::create_dir_all(&evidence_dir).expect("create evidence dir");
    fs::write(
        evidence_dir.join("focused-runtime-report.json"),
        r#"{
  "scenario": "steady-decode",
  "mode": "executed",
  "latency_ms": {"decode_elapsed_ms_p50": 88},
  "throughput_tokens_per_second": {"generated": 31.25}
}"#,
    )
    .expect("write focused runtime report");
    fs::write(
        evidence_dir.join("chat-corpus.json"),
        r#"{"summary":{"errors":0}}"#,
    )
    .expect("write chat corpus report");
    fs::write(
        evidence_dir.join("prompt-lengths-summary.json"),
        r#"{"row_count":2,"exceeds_context":0}"#,
    )
    .expect("write token lengths report");
    fs::write(
        evidence_dir.join("local-split-chain.json"),
        r#"{
  "mode": "local-split-chain-binary",
  "predicted_token": 42,
  "boundary_transfers": [
    {"wire_payload_bytes": 1024},
    {"wire_payload_bytes": 2048}
  ]
}"#,
    )
    .expect("write local split report");

    let summary = DirectSkippyBenchSummary::from_run_dir(&dir);

    assert_eq!(summary.evidence_count, 4);
    assert_eq!(
        summary.evidence_labels,
        [
            "focused-runtime",
            "chat-corpus",
            "token-lengths",
            "local-split-chain"
        ]
    );
    assert_eq!(summary.focused_runtime_generated_tps, Some(31.25));
    assert_eq!(summary.focused_runtime_decode_elapsed_ms_p50, Some(88.0));
    assert_eq!(
        summary.local_split_decode_transfer_bytes_per_token,
        Some(3072)
    );
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn direct_skippy_bench_summary_rejects_failed_or_smoke_evidence() {
    let dir = unique_test_dir("direct-skippy-bench-failed-evidence");
    let evidence_dir = dir.join("evidence");
    fs::create_dir_all(&evidence_dir).expect("create evidence dir");
    fs::write(
        evidence_dir.join("focused-runtime-report.json"),
        r#"{
  "scenario": "steady-decode",
  "mode": "schema-smoke",
  "latency_ms": {"decode_elapsed_ms_p50": 5},
  "throughput_tokens_per_second": {"generated": 100.0}
}"#,
    )
    .expect("write focused runtime report");
    fs::write(
        evidence_dir.join("chat-corpus.json"),
        r#"{"summary":{"errors":1}}"#,
    )
    .expect("write chat corpus report");
    fs::write(
        evidence_dir.join("prompt-lengths-summary.json"),
        r#"{"row_count":2,"exceeds_context":1}"#,
    )
    .expect("write token lengths report");
    fs::write(
        evidence_dir.join("local-split-chain.json"),
        r#"{"mode":"local-split-chain-binary","stages":[]}"#,
    )
    .expect("write local split report");

    let summary = DirectSkippyBenchSummary::from_run_dir(&dir);

    assert_eq!(summary.evidence_count, 0);
    assert_eq!(summary.ignored_evidence_notes.len(), 4);
    assert!(
        summary
            .ignored_evidence_notes
            .iter()
            .any(|note| note.contains("ignored direct focused-runtime evidence"))
    );
    assert!(
        summary
            .ignored_evidence_notes
            .iter()
            .any(|note| note.contains("ignored direct chat-corpus evidence"))
    );
    assert!(
        summary
            .ignored_evidence_notes
            .iter()
            .any(|note| note.contains("ignored direct token-length evidence"))
    );
    assert!(
        summary
            .ignored_evidence_notes
            .iter()
            .any(|note| note.contains("ignored direct local-split-chain evidence"))
    );
    assert_eq!(summary.focused_runtime_generated_tps, None);
    assert_eq!(summary.focused_runtime_decode_elapsed_ms_p50, None);
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn stale_certification_measurements_do_not_feed_rank_output() {
    let dir = unique_test_dir("stale-certification-measurements");
    fs::write(
        dir.join("quant-pack-build.json"),
        r#"{
  "candidate": "middle-compressed",
  "agent_pack": "agent-pack.json",
  "preflight": "preflight.json"
}"#,
    )
    .expect("write build manifest");
    fs::write(
        dir.join("agent-pack.json"),
        r#"{
  "pack_id": "middle-compressed",
  "quant_layout": {
    "strategy": "stage-aware-middle-compressed",
    "default": "Q4_K_M",
    "layout_hash": "layout-hash",
    "groups": []
  }
}"#,
    )
    .expect("write agent pack");
    fs::write(
        dir.join("preflight.json"),
        r#"{
  "valid": true,
  "activation_width": 4096,
  "stages": [{"artifact_bytes": 10000, "layer_start": 0, "layer_end": 1}]
}"#,
    )
    .expect("write preflight");
    fs::write(
        dir.join("certification.json"),
        format!(
            r#"{{
  "status": "agent_quality_candidate",
  "subject": {{
    "build_manifest": {{"sha256": "{}"}},
    "agent_pack": {{"sha256": "{}"}},
    "preflight": {{"sha256": "{}"}}
  }},
  "skippy_bench_reports": [
    {{
      "evidence_type": "skippy-bench-focused-runtime",
      "status": "pass",
      "summary": {{
        "throughput_tokens_per_second": {{"generated": 999.0}},
        "latency_ms": {{"decode_elapsed_ms_p50": 1}}
      }}
    }}
  ],
  "quality_evidence": [
    {{"evidence_type": "quality-agent-tool-call", "status": "pass"}}
  ]
}}"#,
            hash_ref(
                r#"{
  "candidate": "middle-compressed",
  "agent_pack": "agent-pack.json",
  "preflight": "preflight.json"
}"#
            )
            .sha256,
            hash_ref(
                r#"{
  "pack_id": "middle-compressed",
  "quant_layout": {
    "strategy": "stage-aware-middle-compressed",
    "default": "Q4_K_M",
    "layout_hash": "layout-hash",
    "groups": []
  }
}"#
            )
            .sha256,
            hash_ref(
                r#"{
  "valid": true,
  "activation_width": 4096,
  "stages": [{"artifact_bytes": 10000, "layer_start": 0, "layer_end": 1}]
}"#
            )
            .sha256,
        ),
    )
    .expect("write certification");

    let candidate = load_ranked_candidate(
        &dir,
        RankRuntimeShape {
            ctx_size: 8192,
            n_gpu_layers: -1,
            cache_type_k: "f16",
            cache_type_v: "f16",
            activation_wire_dtype: "f16",
        },
    )
    .expect("load ranked candidate");

    assert_eq!(
        candidate.certification_status,
        Some(RankCertificationStatus::Failed)
    );
    assert_eq!(
        candidate.certification_subject_status,
        Some(RankCertificationSubjectStatus::NotVerifiable)
    );
    assert_eq!(candidate.skippy_bench_evidence_count, 0);
    assert_eq!(candidate.quality_evidence_count, 0);
    assert_eq!(candidate.focused_runtime_generated_tokens_per_second, None);
    assert_eq!(candidate.focused_runtime_decode_elapsed_ms_p50, None);
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn generated_evidence_certification_takes_precedence_over_root_certification() {
    let dir = unique_test_dir("certification-precedence");
    let evidence_dir = dir.join("evidence");
    fs::create_dir_all(&evidence_dir).expect("create evidence dir");
    fs::write(dir.join("certification.json"), r#"{"status":"failed"}"#)
        .expect("write stale root certification");
    fs::write(
        evidence_dir.join("certification.json"),
        r#"{"status":"agent_quality_candidate"}"#,
    )
    .expect("write generated certification");

    let (path, certification) = read_certification(&dir)
        .expect("read certification")
        .expect("certification exists");

    assert_eq!(path, evidence_dir.join("certification.json"));
    assert_eq!(
        certification.status,
        RankCertificationStatus::AgentQualityCandidate
    );
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn ranking_prefers_higher_focused_runtime_throughput_when_certified() {
    let mut candidates = [
        ranked_candidate_with_runtime_tps(
            "runtime-slower",
            Some(RankCertificationStatus::AgentQualityCandidate),
            25.0,
        ),
        ranked_candidate_with_runtime_tps(
            "runtime-faster",
            Some(RankCertificationStatus::AgentQualityCandidate),
            50.0,
        ),
    ];

    candidates.sort_by(compare_candidates);

    assert_eq!(candidates[0].candidate, "runtime-faster");
}

#[test]
fn stale_certification_subject_is_treated_as_failed_for_ranking() {
    let dir = unique_test_dir("stale-certification");
    fs::write(dir.join("quant-pack-build.json"), b"current-build").expect("write manifest");
    fs::write(dir.join("agent-pack.json"), b"agent-pack").expect("write agent pack");
    fs::write(dir.join("preflight.json"), b"preflight").expect("write preflight");
    fs::write(dir.join("model.gguf"), b"model").expect("write model");
    fs::create_dir_all(dir.join("package")).expect("create package");
    fs::write(dir.join("package/model-package.json"), b"package").expect("write package");
    let manifest = BuildManifestInput {
        candidate: "middle-compressed".to_string(),
        agent_pack: "agent-pack.json".to_string(),
        preflight: "preflight.json".to_string(),
        package: Some("package".to_string()),
        quantized_model: Some("model.gguf".to_string()),
        quantize_run: None,
        decode_profile: None,
    };
    let certification = CertificationInput {
        status: RankCertificationStatus::AgentQualityCandidate,
        runtime_shape: Some(matching_certification_runtime_shape()),
        expected_topology: Some(matching_certification_topology()),
        subject: Some(CertificationSubjectInput {
            build_manifest: Some(hash_ref("old-build")),
            agent_pack: Some(hash_ref("agent-pack")),
            preflight: Some(hash_ref("preflight")),
            expected_quantized_model: Some(hash_ref("model")),
            package_manifest: Some(hash_ref("package")),
            quantize_run: None,
        }),
        gates: Vec::new(),
        skippy_bench_reports: Vec::new(),
        quality_evidence: Vec::new(),
    };

    let check = certification_subject_check(
        &dir,
        &dir.join("quant-pack-build.json"),
        &manifest,
        &certification,
        &matching_preflight(),
        matching_rank_runtime_shape(),
    );
    let effective = effective_certification_status(Some(certification.status), check.status);

    assert_eq!(check.status, RankCertificationSubjectStatus::Stale);
    assert_eq!(effective, Some(RankCertificationStatus::Failed));
    assert!(
        check
            .notes
            .iter()
            .any(|note| note.contains("build_manifest"))
    );
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn unverifiable_certification_subject_is_treated_as_failed_for_ranking() {
    let dir = unique_test_dir("unverifiable-certification");
    fs::write(dir.join("quant-pack-build.json"), b"build").expect("write manifest");
    let manifest = BuildManifestInput {
        candidate: "middle-compressed".to_string(),
        agent_pack: "agent-pack.json".to_string(),
        preflight: "preflight.json".to_string(),
        package: Some("package".to_string()),
        quantized_model: Some("model.gguf".to_string()),
        quantize_run: None,
        decode_profile: None,
    };
    let certification = CertificationInput {
        status: RankCertificationStatus::AgentQualityCandidate,
        runtime_shape: None,
        expected_topology: None,
        subject: None,
        gates: Vec::new(),
        skippy_bench_reports: Vec::new(),
        quality_evidence: Vec::new(),
    };

    let check = certification_subject_check(
        &dir,
        &dir.join("quant-pack-build.json"),
        &manifest,
        &certification,
        &matching_preflight(),
        matching_rank_runtime_shape(),
    );
    let effective = effective_certification_status(Some(certification.status), check.status);

    assert_eq!(check.status, RankCertificationSubjectStatus::NotVerifiable);
    assert_eq!(effective, Some(RankCertificationStatus::Failed));
    assert!(
        check
            .notes
            .iter()
            .any(|note| note.contains("no subject hashes"))
    );
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn certification_runtime_shape_mismatch_is_treated_as_stale_for_ranking() {
    let dir = unique_test_dir("stale-runtime-shape-certification");
    fs::write(dir.join("quant-pack-build.json"), b"build").expect("write manifest");
    fs::write(dir.join("agent-pack.json"), b"agent-pack").expect("write agent pack");
    fs::write(dir.join("preflight.json"), b"preflight").expect("write preflight");
    fs::write(dir.join("model.gguf"), b"model").expect("write model");
    fs::create_dir_all(dir.join("package")).expect("create package");
    fs::write(dir.join("package/model-package.json"), b"package").expect("write package");
    let manifest = BuildManifestInput {
        candidate: "middle-compressed".to_string(),
        agent_pack: "agent-pack.json".to_string(),
        preflight: "preflight.json".to_string(),
        package: Some("package".to_string()),
        quantized_model: Some("model.gguf".to_string()),
        quantize_run: None,
        decode_profile: None,
    };
    let certification = CertificationInput {
        status: RankCertificationStatus::AgentQualityCandidate,
        runtime_shape: Some(CertificationRuntimeShapeInput {
            ctx_size: Some(4096),
            n_gpu_layers: Some(-1),
            cache_type_k: Some("q8_0".to_string()),
            cache_type_v: Some("f16".to_string()),
            activation_wire_dtype: Some("q8".to_string()),
        }),
        expected_topology: Some(matching_certification_topology()),
        subject: Some(CertificationSubjectInput {
            build_manifest: Some(hash_ref("build")),
            agent_pack: Some(hash_ref("agent-pack")),
            preflight: Some(hash_ref("preflight")),
            expected_quantized_model: Some(hash_ref("model")),
            package_manifest: Some(hash_ref("package")),
            quantize_run: None,
        }),
        gates: Vec::new(),
        skippy_bench_reports: Vec::new(),
        quality_evidence: Vec::new(),
    };

    let check = certification_subject_check(
        &dir,
        &dir.join("quant-pack-build.json"),
        &manifest,
        &certification,
        &matching_preflight(),
        matching_rank_runtime_shape(),
    );
    let effective = effective_certification_status(Some(certification.status), check.status);

    assert_eq!(check.status, RankCertificationSubjectStatus::Stale);
    assert_eq!(effective, Some(RankCertificationStatus::Failed));
    assert!(
        check
            .notes
            .iter()
            .any(|note| note.contains("runtime_shape.ctx_size 4096 != 8192"))
    );
    assert!(
        check
            .notes
            .iter()
            .any(|note| note.contains("runtime_shape.cache_type_k q8_0 != f16"))
    );
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn certification_topology_mismatch_is_treated_as_stale_for_ranking() {
    let dir = unique_test_dir("stale-topology-certification");
    fs::write(dir.join("quant-pack-build.json"), b"build").expect("write manifest");
    fs::write(dir.join("agent-pack.json"), b"agent-pack").expect("write agent pack");
    fs::write(dir.join("preflight.json"), b"preflight").expect("write preflight");
    fs::write(dir.join("model.gguf"), b"model").expect("write model");
    fs::create_dir_all(dir.join("package")).expect("create package");
    fs::write(dir.join("package/model-package.json"), b"package").expect("write package");
    let manifest = BuildManifestInput {
        candidate: "middle-compressed".to_string(),
        agent_pack: "agent-pack.json".to_string(),
        preflight: "preflight.json".to_string(),
        package: Some("package".to_string()),
        quantized_model: Some("model.gguf".to_string()),
        quantize_run: None,
        decode_profile: None,
    };
    let certification = CertificationInput {
        status: RankCertificationStatus::AgentQualityCandidate,
        runtime_shape: Some(matching_certification_runtime_shape()),
        expected_topology: Some(CertificationTopologyInput {
            splits: Some("12".to_string()),
            layer_end: Some(40),
            stage_count: Some(2),
        }),
        subject: Some(CertificationSubjectInput {
            build_manifest: Some(hash_ref("build")),
            agent_pack: Some(hash_ref("agent-pack")),
            preflight: Some(hash_ref("preflight")),
            expected_quantized_model: Some(hash_ref("model")),
            package_manifest: Some(hash_ref("package")),
            quantize_run: None,
        }),
        gates: Vec::new(),
        skippy_bench_reports: Vec::new(),
        quality_evidence: Vec::new(),
    };

    let check = certification_subject_check(
        &dir,
        &dir.join("quant-pack-build.json"),
        &manifest,
        &certification,
        &matching_preflight(),
        matching_rank_runtime_shape(),
    );
    let effective = effective_certification_status(Some(certification.status), check.status);

    assert_eq!(check.status, RankCertificationSubjectStatus::Stale);
    assert_eq!(effective, Some(RankCertificationStatus::Failed));
    assert!(
        check
            .notes
            .iter()
            .any(|note| note.contains("expected_topology.splits 12 != 20"))
    );
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn stale_certification_evidence_report_is_treated_as_failed_for_ranking() {
    let dir = unique_test_dir("stale-evidence-certification");
    fs::write(dir.join("quant-pack-build.json"), b"build").expect("write manifest");
    fs::write(dir.join("agent-pack.json"), b"agent-pack").expect("write agent pack");
    fs::write(dir.join("preflight.json"), b"preflight").expect("write preflight");
    fs::write(dir.join("model.gguf"), b"model").expect("write model");
    fs::create_dir_all(dir.join("package")).expect("create package");
    fs::write(dir.join("package/model-package.json"), b"package").expect("write package");
    fs::create_dir_all(dir.join("evidence")).expect("create evidence dir");
    fs::write(
        dir.join("evidence/focused-runtime-report.json"),
        b"current-focused-runtime",
    )
    .expect("write focused runtime evidence");
    let manifest = BuildManifestInput {
        candidate: "middle-compressed".to_string(),
        agent_pack: "agent-pack.json".to_string(),
        preflight: "preflight.json".to_string(),
        package: Some("package".to_string()),
        quantized_model: Some("model.gguf".to_string()),
        quantize_run: None,
        decode_profile: None,
    };
    let certification = CertificationInput {
        status: RankCertificationStatus::AgentQualityCandidate,
        runtime_shape: Some(matching_certification_runtime_shape()),
        expected_topology: Some(matching_certification_topology()),
        subject: Some(CertificationSubjectInput {
            build_manifest: Some(hash_ref("build")),
            agent_pack: Some(hash_ref("agent-pack")),
            preflight: Some(hash_ref("preflight")),
            expected_quantized_model: Some(hash_ref("model")),
            package_manifest: Some(hash_ref("package")),
            quantize_run: None,
        }),
        gates: Vec::new(),
        skippy_bench_reports: vec![serde_json::json!({
            "evidence_type": "skippy-bench-focused-runtime",
            "path": "evidence/focused-runtime-report.json",
            "sha256": hash_ref("old-focused-runtime").sha256,
            "summary": {}
        })],
        quality_evidence: Vec::new(),
    };

    let check = certification_subject_check(
        &dir,
        &dir.join("quant-pack-build.json"),
        &manifest,
        &certification,
        &matching_preflight(),
        matching_rank_runtime_shape(),
    );
    let effective = effective_certification_status(Some(certification.status), check.status);

    assert_eq!(check.status, RankCertificationSubjectStatus::Stale);
    assert_eq!(effective, Some(RankCertificationStatus::Failed));
    assert!(
        check
            .notes
            .iter()
            .any(|note| note.contains("skippy_bench_reports[0]"))
    );
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn matching_certification_subject_is_verified_for_ranking() {
    let dir = unique_test_dir("verified-certification");
    fs::write(dir.join("quant-pack-build.json"), b"build").expect("write manifest");
    fs::write(dir.join("agent-pack.json"), b"agent-pack").expect("write agent pack");
    fs::write(dir.join("preflight.json"), b"preflight").expect("write preflight");
    fs::write(dir.join("model.gguf"), b"model").expect("write model");
    fs::create_dir_all(dir.join("package")).expect("create package");
    fs::write(dir.join("package/model-package.json"), b"package").expect("write package");
    fs::create_dir_all(dir.join("evidence")).expect("create evidence dir");
    fs::write(
        dir.join("evidence/focused-runtime-report.json"),
        b"focused-runtime",
    )
    .expect("write focused runtime evidence");
    fs::write(dir.join("evidence/tool-calls.jsonl"), b"tool-calls")
        .expect("write quality evidence");
    let manifest = BuildManifestInput {
        candidate: "middle-compressed".to_string(),
        agent_pack: "agent-pack.json".to_string(),
        preflight: "preflight.json".to_string(),
        package: Some("package".to_string()),
        quantized_model: Some("model.gguf".to_string()),
        quantize_run: None,
        decode_profile: None,
    };
    let certification = CertificationInput {
        status: RankCertificationStatus::AgentQualityCandidate,
        runtime_shape: Some(matching_certification_runtime_shape()),
        expected_topology: Some(matching_certification_topology()),
        subject: Some(CertificationSubjectInput {
            build_manifest: Some(hash_ref("build")),
            agent_pack: Some(hash_ref("agent-pack")),
            preflight: Some(hash_ref("preflight")),
            expected_quantized_model: Some(hash_ref("model")),
            package_manifest: Some(hash_ref("package")),
            quantize_run: None,
        }),
        gates: Vec::new(),
        skippy_bench_reports: vec![serde_json::json!({
            "evidence_type": "skippy-bench-focused-runtime",
            "path": "evidence/focused-runtime-report.json",
            "sha256": hash_ref("focused-runtime").sha256,
            "summary": {}
        })],
        quality_evidence: vec![serde_json::json!({
            "evidence_type": "agent-tool-call-results",
            "path": "evidence/tool-calls.jsonl",
            "sha256": hash_ref("tool-calls").sha256,
            "summary": {}
        })],
    };

    let check = certification_subject_check(
        &dir,
        &dir.join("quant-pack-build.json"),
        &manifest,
        &certification,
        &matching_preflight(),
        matching_rank_runtime_shape(),
    );

    assert_eq!(check.status, RankCertificationSubjectStatus::Verified);
    assert_eq!(
        effective_certification_status(Some(certification.status), check.status),
        Some(RankCertificationStatus::AgentQualityCandidate)
    );
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn certification_without_topology_is_verified_when_subject_and_shape_match() {
    let dir = unique_test_dir("verified-certification-without-topology");
    fs::write(dir.join("quant-pack-build.json"), b"build").expect("write manifest");
    fs::write(dir.join("agent-pack.json"), b"agent-pack").expect("write agent pack");
    fs::write(dir.join("preflight.json"), b"preflight").expect("write preflight");
    fs::write(dir.join("model.gguf"), b"model").expect("write model");
    fs::create_dir_all(dir.join("package")).expect("create package");
    fs::write(dir.join("package/model-package.json"), b"package").expect("write package");
    let manifest = BuildManifestInput {
        candidate: "middle-compressed".to_string(),
        agent_pack: "agent-pack.json".to_string(),
        preflight: "preflight.json".to_string(),
        package: Some("package".to_string()),
        quantized_model: Some("model.gguf".to_string()),
        quantize_run: None,
        decode_profile: None,
    };
    let certification = CertificationInput {
        status: RankCertificationStatus::MeasurementOnlyCandidate,
        runtime_shape: Some(matching_certification_runtime_shape()),
        expected_topology: None,
        subject: Some(CertificationSubjectInput {
            build_manifest: Some(hash_ref("build")),
            agent_pack: Some(hash_ref("agent-pack")),
            preflight: Some(hash_ref("preflight")),
            expected_quantized_model: Some(hash_ref("model")),
            package_manifest: Some(hash_ref("package")),
            quantize_run: None,
        }),
        gates: Vec::new(),
        skippy_bench_reports: Vec::new(),
        quality_evidence: Vec::new(),
    };

    let check = certification_subject_check(
        &dir,
        &dir.join("quant-pack-build.json"),
        &manifest,
        &certification,
        &matching_preflight(),
        matching_rank_runtime_shape(),
    );

    assert_eq!(check.status, RankCertificationSubjectStatus::Verified);
    fs::remove_dir_all(dir).expect("remove fixture");
}

fn ranked_candidate(
    candidate: &str,
    valid: bool,
    measured: bool,
    certification_status: Option<RankCertificationStatus>,
    decode_mean_ms: Option<f64>,
    slowest_stage_artifact_bytes: u64,
    stage_imbalance_ratio: Option<f64>,
) -> RankedCandidate {
    RankedCandidate {
        rank: 0,
        candidate: candidate.to_string(),
        pack_id: None,
        run_dir: "/tmp/run".to_string(),
        valid,
        measured,
        certification_status,
        certification_report_status: certification_status,
        certification_subject_status: None,
        certification_path: None,
        certification_gate_failures: 0,
        certification_gate_warnings: 0,
        skippy_bench_evidence_count: 0,
        quality_evidence_count: 0,
        score: rank_score(RankScoreInputs {
            valid,
            measured,
            certification_status,
            decode_mean_ms,
            focused_runtime_generated_tokens_per_second: None,
            slowest_stage_artifact_bytes,
            largest_stage_model_plus_kv_bytes: Some(slowest_stage_artifact_bytes),
            stage_imbalance_ratio,
            decode_transfer_bytes_per_token: Some(0),
        }),
        decode_mean_ms,
        decode_p95_ms: None,
        estimated_tokens_per_second: None,
        focused_runtime_generated_tokens_per_second: None,
        focused_runtime_decode_elapsed_ms_p50: None,
        ctx_size: 8192,
        n_gpu_layers: -1,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        activation_width: Some(4096),
        activation_wire_dtype: "f16".to_string(),
        estimated_total_kv_cache_bytes: Some(0),
        estimated_largest_stage_kv_cache_bytes: Some(0),
        estimated_largest_stage_model_plus_kv_bytes: Some(slowest_stage_artifact_bytes),
        estimated_boundary_activation_bytes: Some(8192),
        estimated_decode_activation_transfer_bytes_per_token: Some(8192),
        activation_transfer_source: Some(ActivationTransferSource::PreflightEstimate),
        package_artifact_bytes: slowest_stage_artifact_bytes,
        slowest_stage_artifact_bytes,
        stage_imbalance_ratio,
        layout_hash: None,
        strategy: None,
        default_quant: None,
        group_count: 0,
        source_sha256: None,
        notes: Vec::new(),
    }
}

fn matching_rank_runtime_shape() -> RankRuntimeShape<'static> {
    RankRuntimeShape {
        ctx_size: 8192,
        n_gpu_layers: -1,
        cache_type_k: "f16",
        cache_type_v: "f16",
        activation_wire_dtype: "f16",
    }
}

fn matching_certification_runtime_shape() -> CertificationRuntimeShapeInput {
    CertificationRuntimeShapeInput {
        ctx_size: Some(8192),
        n_gpu_layers: Some(-1),
        cache_type_k: Some("f16".to_string()),
        cache_type_v: Some("f16".to_string()),
        activation_wire_dtype: Some("f16".to_string()),
    }
}

fn matching_certification_topology() -> CertificationTopologyInput {
    CertificationTopologyInput {
        splits: Some("20".to_string()),
        layer_end: Some(40),
        stage_count: Some(2),
    }
}

fn matching_preflight() -> PreflightInput {
    PreflightInput {
        valid: true,
        activation_width: Some(4096),
        stages: vec![
            PreflightStageInput {
                artifact_bytes: 10_000,
                layer_start: Some(0),
                layer_end: Some(20),
            },
            PreflightStageInput {
                artifact_bytes: 12_000,
                layer_start: Some(20),
                layer_end: Some(40),
            },
        ],
    }
}

fn ranked_candidate_with_runtime_tps(
    candidate: &str,
    certification_status: Option<RankCertificationStatus>,
    generated_tps: f64,
) -> RankedCandidate {
    let mut candidate = ranked_candidate(
        candidate,
        true,
        true,
        certification_status,
        Some(20.0),
        100,
        Some(1.0),
    );
    candidate.focused_runtime_generated_tokens_per_second = Some(generated_tps);
    candidate.score = rank_score(RankScoreInputs {
        valid: candidate.valid,
        measured: candidate.measured,
        certification_status: candidate.certification_status,
        decode_mean_ms: candidate.decode_mean_ms,
        focused_runtime_generated_tokens_per_second: Some(generated_tps),
        slowest_stage_artifact_bytes: candidate.slowest_stage_artifact_bytes,
        largest_stage_model_plus_kv_bytes: candidate.estimated_largest_stage_model_plus_kv_bytes,
        stage_imbalance_ratio: candidate.stage_imbalance_ratio,
        decode_transfer_bytes_per_token: candidate
            .estimated_decode_activation_transfer_bytes_per_token,
    });
    candidate
}

fn unique_test_dir(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("skippy-rank-{name}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&dir).expect("create fixture dir");
    dir
}

fn hash_ref(contents: &str) -> HashedArtifactInput {
    let mut hasher = Sha256::new();
    hasher.update(contents.as_bytes());
    HashedArtifactInput {
        sha256: format!("{:x}", hasher.finalize()),
    }
}
