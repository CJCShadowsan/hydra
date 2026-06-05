use super::*;

#[test]
fn skippy_bench_token_lengths_fail_when_rows_exceed_context() {
    let value = serde_json::json!({
        "model_path": "/tmp/model.gguf",
        "row_count": 2,
        "fits_context": 1,
        "exceeds_context": 1
    });

    let summary = summarize_skippy_bench_json(&value);

    assert_eq!(summary.evidence_type, "skippy-bench-token-lengths");
    assert_eq!(summary.status, GateStatus::Fail);
    assert_eq!(summary.model_paths, ["/tmp/model.gguf"]);
}

#[test]
fn skippy_bench_subjects_are_extracted_from_report_shapes() {
    let focused = summarize_skippy_bench_json(&serde_json::json!({
        "scenario": "steady-decode",
        "mode": "executed",
        "model_id": "org/repo:middle-compressed",
        "latency_ms": {"decode_elapsed_ms_p50": 20},
        "throughput_tokens_per_second": {"generated": 31.25},
        "runtime": {"activation_wire_dtype": "q8"},
    }));
    let chat = summarize_skippy_bench_json(&serde_json::json!({
        "model": "org/repo:middle-compressed",
        "prompt_corpus": "target/bench-corpora/coding-loop/corpus.jsonl",
        "summary": {"errors": 0},
    }));
    let long_context_chat = summarize_skippy_bench_json(&serde_json::json!({
        "model": "org/repo:middle-compressed",
        "prompt_corpus": "target/bench-corpora/long-context/corpus.jsonl",
        "summary": {"errors": 0},
    }));
    let token_lengths = summarize_skippy_bench_json(&serde_json::json!({
        "model_path": "/tmp/middle-compressed.gguf",
        "row_count": 1,
        "exceeds_context": 0,
    }));
    let local_split = summarize_skippy_bench_json(&serde_json::json!({
        "mode": "local-split-chain-binary",
        "model_identity": {"model_id": "org/repo:middle-compressed"},
        "predicted_token": "}",
        "activation_width": 8192,
        "wire_dtype": "q8",
        "stages": [
            {"index": 0, "payload_bytes": 1048576, "wire_payload_bytes": 524288}
        ],
    }));

    assert_eq!(focused.model_ids, ["org/repo:middle-compressed"]);
    assert_eq!(
        focused
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.get("activation_wire_dtype"))
            .and_then(Value::as_str),
        Some("q8")
    );
    assert_eq!(chat.model_ids, ["org/repo:middle-compressed"]);
    assert_eq!(
        long_context_chat.evidence_type,
        "skippy-bench-long-context-chat-corpus"
    );
    assert_eq!(long_context_chat.model_ids, ["org/repo:middle-compressed"]);
    assert_eq!(token_lengths.model_paths, ["/tmp/middle-compressed.gguf"]);
    assert_eq!(local_split.evidence_type, "skippy-bench-local-split-chain");
    assert_eq!(local_split.status, GateStatus::Pass);
    assert_eq!(local_split.model_ids, ["org/repo:middle-compressed"]);
    assert_eq!(local_split.summary["wire_dtype"], "q8");
}

#[test]
fn local_split_chain_requires_predicted_token_and_payload_bytes() {
    let missing_token = summarize_skippy_bench_json(&serde_json::json!({
        "mode": "local-split-chain-binary",
        "stages": [
            {"index": 0, "payload_bytes": 1048576, "wire_payload_bytes": 524288}
        ],
    }));
    let missing_payload = summarize_skippy_bench_json(&serde_json::json!({
        "mode": "local-split-chain-binary",
        "predicted_token": "}",
        "stages": [
            {"index": 0, "payload_bytes": 0, "wire_payload_bytes": 0}
        ],
    }));

    assert_eq!(
        missing_token.evidence_type,
        "skippy-bench-local-split-chain"
    );
    assert_eq!(missing_token.status, GateStatus::Fail);
    assert_eq!(missing_payload.status, GateStatus::Fail);
}

#[test]
fn focused_runtime_schema_smoke_does_not_certify_as_runtime_evidence() {
    let summary = summarize_skippy_bench_json(&serde_json::json!({
        "scenario": "steady-decode",
        "mode": "schema-smoke",
        "latency_ms": {"decode_elapsed_ms_p50": 5},
        "throughput_tokens_per_second": {"generated": 100.0},
    }));

    assert_eq!(summary.evidence_type, "skippy-bench-focused-runtime");
    assert_eq!(summary.status, GateStatus::Fail);
}

#[test]
fn focused_runtime_requires_generated_throughput_for_certification() {
    let summary = summarize_skippy_bench_json(&serde_json::json!({
        "scenario": "steady-decode",
        "mode": "executed",
        "latency_ms": {"decode_elapsed_ms_p50": 20},
        "throughput_tokens_per_second": {"generated": 0.0},
    }));

    assert_eq!(summary.evidence_type, "skippy-bench-focused-runtime");
    assert_eq!(summary.status, GateStatus::Fail);
}

#[test]
fn evidence_subject_gate_fails_on_mismatched_model_id() {
    let reports = vec![evidence_with_model_id(
        "skippy-bench-focused-runtime",
        GateStatus::Pass,
        "org/repo:wrong",
    )];

    let gate = evidence_subject_gate(
        "org/repo:middle-compressed",
        Path::new("/tmp/middle-compressed.gguf"),
        &reports,
        &[],
        true,
    );

    assert_eq!(gate.status, GateStatus::Fail);
    assert!(gate.detail.contains("org/repo:wrong"));
}

#[test]
fn evidence_subject_gate_fails_on_mismatched_tokenizer_model_path() {
    let reports = vec![evidence_with_model_path(
        "skippy-bench-token-lengths",
        GateStatus::Pass,
        "/tmp/wrong.gguf",
    )];

    let gate = evidence_subject_gate(
        "org/repo:middle-compressed",
        Path::new("/tmp/middle-compressed.gguf"),
        &reports,
        &[],
        true,
    );

    assert_eq!(gate.status, GateStatus::Fail);
    assert!(gate.detail.contains("/tmp/wrong.gguf"));
}

#[test]
fn required_skippy_bench_subjects_must_identify_the_candidate() {
    let reports = vec![evidence("skippy-bench-chat-corpus", GateStatus::Pass)];

    let exploratory = evidence_subject_gate(
        "org/repo:middle-compressed",
        Path::new("/tmp/middle-compressed.gguf"),
        &reports,
        &[],
        false,
    );
    let required = evidence_subject_gate(
        "org/repo:middle-compressed",
        Path::new("/tmp/middle-compressed.gguf"),
        &reports,
        &[],
        true,
    );

    assert_eq!(exploratory.status, GateStatus::Warn);
    assert_eq!(required.status, GateStatus::Fail);
    assert!(
        required
            .detail
            .contains("missing skippy subjects: skippy-bench-chat-corpus")
    );
}

#[test]
fn runtime_shape_gate_checks_focused_runtime_activation_wire_dtype() {
    let pass = runtime_shape_gate(
        expected_runtime_shape("q8"),
        &[evidence_with_runtime(
            "skippy-bench-focused-runtime",
            GateStatus::Pass,
            runtime_shape_json("q8", 8192, -1, "f16", "f16"),
        )],
    );
    let fail = runtime_shape_gate(
        expected_runtime_shape("q8"),
        &[evidence_with_runtime(
            "skippy-bench-focused-runtime",
            GateStatus::Pass,
            runtime_shape_json("f16", 8192, -1, "f16", "f16"),
        )],
    );

    assert_eq!(pass.status, GateStatus::Pass);
    assert_eq!(fail.status, GateStatus::Fail);
    assert!(fail.detail.contains("f16"));
}

#[test]
fn runtime_shape_gate_checks_context_gpu_and_cache_shape() {
    let fail = runtime_shape_gate(
        expected_runtime_shape("f16"),
        &[evidence_with_runtime(
            "skippy-bench-focused-runtime",
            GateStatus::Pass,
            runtime_shape_json("f16", 4096, 0, "q8_0", "f16"),
        )],
    );

    assert_eq!(fail.status, GateStatus::Fail);
    assert!(fail.detail.contains("ctx_size 4096 != 8192"));
    assert!(fail.detail.contains("n_gpu_layers 0 != -1"));
    assert!(fail.detail.contains("cache_type_k q8_0 != f16"));
}

#[test]
fn certification_runtime_shape_serializes_gate_inputs() {
    let shape = CertificationRuntimeShape::new(
        32768,
        -1,
        "q8_0".to_string(),
        "f16".to_string(),
        "q8".to_string(),
    );
    let value = serde_json::to_value(&shape).expect("serialize runtime shape");

    assert_eq!(value["ctx_size"], 32768);
    assert_eq!(value["n_gpu_layers"], -1);
    assert_eq!(value["cache_type_k"], "q8_0");
    assert_eq!(value["cache_type_v"], "f16");
    assert_eq!(value["activation_wire_dtype"], "q8");

    let gate = runtime_shape_gate(
        shape.as_gate_expectation(),
        &[evidence_with_runtime(
            "skippy-bench-focused-runtime",
            GateStatus::Pass,
            runtime_shape_json("q8", 32768, -1, "q8_0", "f16"),
        )],
    );

    assert_eq!(gate.status, GateStatus::Pass);
}

#[test]
fn topology_expectation_serializes_certified_split_shape() {
    let expected = TopologyExpectation {
        splits: "16,32,47".to_string(),
        layer_end: 62,
        stage_count: 4,
    };
    let value = serde_json::to_value(&expected).expect("serialize topology expectation");

    assert_eq!(value["splits"], "16,32,47");
    assert_eq!(value["layer_end"], 62);
    assert_eq!(value["stage_count"], 4);
}

#[test]
fn required_skippy_bench_gate_fails_when_report_type_is_missing() {
    let reports = vec![evidence("skippy-bench-focused-runtime", GateStatus::Pass)];

    let gate = skippy_bench_gate(&reports, true);

    assert_eq!(gate.status, GateStatus::Fail);
    assert!(gate.detail.contains("skippy-bench-chat-corpus"));
    assert!(
        gate.detail
            .contains("skippy-bench-long-context-chat-corpus")
    );
    assert!(gate.detail.contains("skippy-bench-token-lengths"));
}

#[test]
fn required_skippy_bench_gate_passes_with_runtime_chat_and_token_reports() {
    let reports = vec![
        evidence("skippy-bench-focused-runtime", GateStatus::Pass),
        evidence("skippy-bench-chat-corpus", GateStatus::Pass),
        evidence("skippy-bench-long-context-chat-corpus", GateStatus::Pass),
        evidence("skippy-bench-token-lengths", GateStatus::Pass),
    ];

    let gate = skippy_bench_gate(&reports, true);

    assert_eq!(gate.status, GateStatus::Pass);
    assert!(gate.detail.contains("missing required evidence: none"));
}

#[test]
fn quality_jsonl_summarizes_ok_rows() {
    let dir = std::env::temp_dir().join(format!("skippy-cert-jsonl-test-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("results.jsonl");
    fs::write(
        &path,
        "{\"ok\":true,\"phase\":\"tool_call\"}\n{\"ok\":false,\"phase\":\"tool_result\"}\n",
    )
    .expect("write jsonl");

    let evidence = summarize_jsonl(&path).expect("summarize jsonl");

    assert_eq!(evidence.status, GateStatus::Fail);
    assert_eq!(evidence.summary["total"], 2);
    assert_eq!(evidence.summary["failed"], 1);
    assert!(evidence.summary["phases"].get("tool_call").is_some());
    let _ = fs::remove_file(path);
    let _ = fs::remove_dir(dir);
}

#[test]
fn quality_evidence_is_classified_by_known_agent_and_kv_paths() {
    let dir = std::env::temp_dir().join(format!(
        "skippy-cert-quality-classify-test-{}",
        std::process::id()
    ));
    let agent_dir = dir.join("agent-tool-call-reliability");
    let kv_dir = dir.join("kv-tool-loop-stability");
    fs::create_dir_all(&agent_dir).expect("create agent dir");
    fs::create_dir_all(&kv_dir).expect("create kv dir");
    let agent = agent_dir.join("results.jsonl");
    let kv = kv_dir.join("summary.json");
    fs::write(
        &agent,
        "{\"ok\":true,\"phase\":\"tool_call\",\"model\":\"org/repo:pack\"}\n",
    )
    .expect("write agent evidence");
    fs::write(
        &kv,
        serde_json::to_string(&serde_json::json!({
            "ok": true,
            "total": 1,
            "passed": 1,
            "failed": 0,
            "phases": {"same_prefix_cache": {"passed": 1, "failed": 0, "total": 1}},
        }))
        .expect("serialize kv evidence"),
    )
    .expect("write kv evidence");

    let agent_summary =
        summarize_evidence_with_base(None, &agent, EvidenceLane::Quality).expect("summarize agent");
    let kv_summary =
        summarize_evidence_with_base(None, &kv, EvidenceLane::Quality).expect("summarize kv");

    assert_eq!(agent_summary.evidence_type, "quality-agent-tool-call");
    assert_eq!(agent_summary.status, GateStatus::Pass);
    assert_eq!(kv_summary.evidence_type, "quality-kv-tool-loop");
    assert_eq!(kv_summary.status, GateStatus::Pass);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn certification_evidence_paths_inside_run_are_run_dir_relative() {
    let dir = unique_test_dir("relative-evidence-path");
    let evidence_dir = dir.join("evidence");
    fs::create_dir_all(&evidence_dir).expect("create evidence dir");
    let focused = evidence_dir.join("focused-runtime-report.json");
    fs::write(
        &focused,
        serde_json::to_string(&serde_json::json!({
            "scenario": "steady-decode",
            "latency_ms": {"decode_elapsed_ms_p50": 88},
            "throughput_tokens_per_second": {"generated": 31.25},
        }))
        .expect("serialize focused report"),
    )
    .expect("write focused report");

    let report = summarize_evidence_for_run(&dir, &focused, EvidenceLane::SkippyBench)
        .expect("summarize evidence");

    assert_eq!(report.path, "evidence/focused-runtime-report.json");
    assert_eq!(
        report.sha256,
        file_sha256(&focused).expect("hash focused report")
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn certification_evidence_paths_outside_run_are_absolute() {
    let dir = unique_test_dir("absolute-evidence-run");
    let external = unique_test_dir("absolute-evidence-external");
    fs::create_dir_all(&dir).expect("create run dir");
    fs::create_dir_all(&external).expect("create external dir");
    let report_path = external.join("summary.json");
    fs::write(
        &report_path,
        serde_json::to_string(&serde_json::json!({
            "ok": true,
            "failed": 0,
            "total": 1,
        }))
        .expect("serialize quality report"),
    )
    .expect("write quality report");

    let report = summarize_evidence_for_run(&dir, &report_path, EvidenceLane::Quality)
        .expect("summarize evidence");
    let expected = fs::canonicalize(&report_path).expect("canonicalize report");

    assert_eq!(report.path, expected.display().to_string());
    let _ = fs::remove_dir_all(dir);
    let _ = fs::remove_dir_all(external);
}

#[test]
fn certification_subject_hashes_candidate_artifacts() {
    let dir = unique_test_dir("subject-hashes");
    let package_dir = dir.join("package");
    fs::create_dir_all(&package_dir).expect("create package dir");
    let manifest_path = dir.join("quant-pack-build.json");
    let package_manifest = package_dir.join("model-package.json");
    let agent_pack = dir.join("agent-pack.json");
    let preflight = dir.join("preflight.json");
    let quantized_model = dir.join("model.gguf");
    let quantize_run = dir.join("quantize-run.json");
    fs::write(&manifest_path, b"build-manifest").expect("write build manifest");
    fs::write(&package_manifest, b"package-manifest").expect("write package manifest");
    fs::write(&agent_pack, b"agent-pack").expect("write agent pack");
    fs::write(&preflight, b"preflight").expect("write preflight");
    fs::write(&quantized_model, b"gguf").expect("write quantized model");
    fs::write(&quantize_run, b"quantize-run").expect("write quantize run");
    let manifest = BuildManifestInput {
        candidate: "middle-compressed".to_string(),
        stages: 2,
        agent_pack: "agent-pack.json".to_string(),
        preflight: "preflight.json".to_string(),
        package: "package".to_string(),
        quantized_model: "model.gguf".to_string(),
        plan: None,
        quantize_run: Some("quantize-run.json".to_string()),
        source_identity: Some(serde_json::json!({"revision": "abc123"})),
        decode_profile: None,
    };

    let subject = certification_subject(CertificationSubjectInput {
        run_dir: &dir,
        manifest_path: &manifest_path,
        manifest: &manifest,
        package_dir: &package_dir,
        package_manifest_path: &package_manifest,
        agent_pack_path: &agent_pack,
        preflight_path: &preflight,
        quantized_model: &quantized_model,
        expected_model_id: "org/repo:middle-compressed",
    })
    .expect("build certification subject");

    assert_eq!(subject.expected_model_id, "org/repo:middle-compressed");
    assert_eq!(subject.expected_quantized_model.sha256, sha256_hex(b"gguf"));
    assert_eq!(
        subject.package_manifest.sha256,
        sha256_hex(b"package-manifest")
    );
    assert_eq!(subject.agent_pack.sha256, sha256_hex(b"agent-pack"));
    assert_eq!(subject.preflight.sha256, sha256_hex(b"preflight"));
    assert_eq!(subject.build_manifest.sha256, sha256_hex(b"build-manifest"));
    assert_eq!(
        subject
            .quantize_run
            .as_ref()
            .expect("quantize run hash")
            .sha256,
        sha256_hex(b"quantize-run")
    );
    assert_eq!(
        subject
            .source_identity
            .as_ref()
            .and_then(|value| value.get("revision"))
            .and_then(Value::as_str),
        Some("abc123")
    );
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn required_quality_coverage_needs_agent_and_kv_lanes() {
    let agent_only = vec![evidence("quality-agent-tool-call", GateStatus::Pass)];
    let complete = vec![
        evidence("quality-agent-tool-call", GateStatus::Pass),
        evidence("quality-kv-tool-loop", GateStatus::Pass),
    ];

    let missing = quality_coverage_gate(&agent_only, true);
    let pass = quality_coverage_gate(&complete, true);

    assert_eq!(missing.status, GateStatus::Fail);
    assert!(missing.detail.contains("quality-kv-tool-loop"));
    assert_eq!(pass.status, GateStatus::Pass);
    assert!(
        pass.detail
            .contains("missing required quality evidence: none")
    );
}

#[test]
fn agent_quality_status_requires_complete_quality_coverage() {
    let agent_only = vec![evidence("quality-agent-tool-call", GateStatus::Pass)];
    let partial_gates = vec![
        gate("preflight", GateStatus::Pass),
        gate("skippy_bench", GateStatus::Pass),
        quality_coverage_gate(&agent_only, false),
    ];
    let complete_quality = vec![
        evidence("quality-agent-tool-call", GateStatus::Pass),
        evidence("quality-kv-tool-loop", GateStatus::Pass),
    ];
    let complete_gates = vec![
        gate("preflight", GateStatus::Pass),
        gate("skippy_bench", GateStatus::Pass),
        quality_coverage_gate(&complete_quality, false),
    ];

    assert_eq!(
        certification_status(&partial_gates, &agent_only),
        CertificationStatus::MeasurementOnlyCandidate
    );
    assert_eq!(
        certification_status(&complete_gates, &complete_quality),
        CertificationStatus::AgentQualityCandidate
    );
}

#[test]
fn runtime_topology_gate_checks_focused_runtime_splits() {
    let expected = TopologyExpectation {
        splits: "16,32,47".to_string(),
        layer_end: 62,
        stage_count: 4,
    };
    let pass = topology_gate(
        Some(&expected),
        &[evidence_with_topology(
            "skippy-bench-focused-runtime",
            "16,32,47",
            62,
            4,
        )],
    );
    let fail = topology_gate(
        Some(&expected),
        &[evidence_with_topology(
            "skippy-bench-focused-runtime",
            "15,31,46",
            62,
            4,
        )],
    );

    assert_eq!(pass.status, GateStatus::Pass);
    assert_eq!(fail.status, GateStatus::Fail);
    assert!(fail.detail.contains("15,31,46"));
}

#[test]
fn runtime_topology_gate_fails_when_expected_topology_is_unverifiable() {
    let expected = TopologyExpectation {
        splits: "16,32,47".to_string(),
        layer_end: 62,
        stage_count: 4,
    };
    let missing = topology_gate(
        Some(&expected),
        &[evidence("skippy-bench-focused-runtime", GateStatus::Pass)],
    );

    assert_eq!(missing.status, GateStatus::Fail);
    assert!(missing.detail.contains("missing topology blocks=1"));
}

#[test]
fn status_is_measurement_only_without_quality_evidence() {
    let gates = vec![
        gate("preflight", GateStatus::Pass),
        gate("decode", GateStatus::Pass),
        gate("skippy_bench", GateStatus::Pass),
        gate("quality", GateStatus::Warn),
    ];

    assert_eq!(
        certification_status(&gates, &[]),
        CertificationStatus::MeasurementOnlyCandidate
    );
}

fn gate(name: &str, status: GateStatus) -> CertificationGate {
    CertificationGate {
        name: name.to_string(),
        status,
        detail: String::new(),
    }
}

fn evidence(evidence_type: &str, status: GateStatus) -> EvidenceReport {
    EvidenceReport {
        path: "/tmp/evidence.json".to_string(),
        sha256: "sha".to_string(),
        evidence_type: evidence_type.to_string(),
        status,
        model_ids: Vec::new(),
        model_paths: Vec::new(),
        runtime: None,
        summary: serde_json::json!({}),
    }
}

fn evidence_with_model_id(
    evidence_type: &str,
    status: GateStatus,
    model_id: &str,
) -> EvidenceReport {
    EvidenceReport {
        model_ids: vec![model_id.to_string()],
        ..evidence(evidence_type, status)
    }
}

fn evidence_with_topology(
    evidence_type: &str,
    splits: &str,
    layer_end: u32,
    stage_count: usize,
) -> EvidenceReport {
    EvidenceReport {
        summary: serde_json::json!({
            "topology": {
                "splits": splits,
                "layer_end": layer_end,
                "stage_count": stage_count
            }
        }),
        ..evidence(evidence_type, GateStatus::Pass)
    }
}

fn evidence_with_model_path(
    evidence_type: &str,
    status: GateStatus,
    model_path: &str,
) -> EvidenceReport {
    EvidenceReport {
        model_paths: vec![model_path.to_string()],
        ..evidence(evidence_type, status)
    }
}

fn evidence_with_runtime(
    evidence_type: &str,
    status: GateStatus,
    runtime: Value,
) -> EvidenceReport {
    EvidenceReport {
        runtime: Some(runtime),
        ..evidence(evidence_type, status)
    }
}

fn expected_runtime_shape(activation_wire_dtype: &str) -> RuntimeShapeExpectation<'_> {
    RuntimeShapeExpectation {
        ctx_size: 8192,
        n_gpu_layers: -1,
        cache_type_k: "f16",
        cache_type_v: "f16",
        activation_wire_dtype,
    }
}

fn runtime_shape_json(
    activation_wire_dtype: &str,
    ctx_size: u32,
    n_gpu_layers: i32,
    cache_type_k: &str,
    cache_type_v: &str,
) -> Value {
    serde_json::json!({
        "ctx_size": ctx_size,
        "n_gpu_layers": n_gpu_layers,
        "cache_type_k": cache_type_k,
        "cache_type_v": cache_type_v,
        "activation_wire_dtype": activation_wire_dtype,
    })
}

fn unique_test_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("skippy-cert-{name}-{}-{nanos}", std::process::id()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
