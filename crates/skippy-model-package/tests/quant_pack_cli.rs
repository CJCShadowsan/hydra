use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

#[test]
fn quant_pack_rank_cli_applies_runtime_shape_to_kv_estimate() {
    let run_dir = temp_dir("quant-pack-rank-cli");
    write_rank_fixture(&run_dir);

    let output = Command::new(env!("CARGO_BIN_EXE_skippy-model-package"))
        .arg("quant-pack")
        .arg("rank")
        .arg(&run_dir)
        .arg("--ctx-size")
        .arg("1024")
        .arg("--n-gpu-layers")
        .arg("0")
        .arg("--cache-type-k")
        .arg("f16")
        .arg("--cache-type-v")
        .arg("q8_0")
        .arg("--activation-wire-dtype")
        .arg("q8")
        .output()
        .expect("run skippy-model-package quant-pack rank");

    assert!(
        output.status.success(),
        "quant-pack rank failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let json: Value = serde_json::from_slice(&output.stdout).expect("parse rank json");
    let candidate = &json["candidates"][0];
    assert_eq!(json["kind"], "skippy_quant_pack_rank");
    assert_eq!(candidate["ctx_size"], 1024);
    assert_eq!(candidate["n_gpu_layers"], 0);
    assert_eq!(candidate["cache_type_k"], "f16");
    assert_eq!(candidate["cache_type_v"], "q8_0");
    assert_eq!(candidate["activation_wire_dtype"], "q8");
    assert_eq!(candidate["estimated_total_kv_cache_bytes"], 62_914_560);
    assert_eq!(
        candidate["estimated_largest_stage_model_plus_kv_bytes"],
        37_773_736
    );

    fs::remove_dir_all(run_dir).ok();
}

#[test]
fn quant_pack_rank_cli_reads_generated_skippy_bench_evidence() {
    let run_dir = temp_dir("quant-pack-rank-cli-evidence");
    write_rank_fixture(&run_dir);
    let evidence_dir = run_dir.join("evidence");
    fs::create_dir_all(&evidence_dir).expect("create evidence dir");
    fs::write(
        evidence_dir.join("focused-runtime-report.json"),
        r#"{
  "scenario": "steady-decode",
  "mode": "executed",
  "latency_ms": {"decode_elapsed_ms_p50": 77},
  "throughput_tokens_per_second": {"generated": 22.5}
}"#,
    )
    .expect("write focused-runtime evidence");
    fs::write(
        evidence_dir.join("chat-corpus.json"),
        r#"{"summary":{"errors":0}}"#,
    )
    .expect("write chat evidence");
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
    .expect("write local split evidence");

    let output = Command::new(env!("CARGO_BIN_EXE_skippy-model-package"))
        .arg("quant-pack")
        .arg("rank")
        .arg(&run_dir)
        .output()
        .expect("run skippy-model-package quant-pack rank");

    assert!(
        output.status.success(),
        "quant-pack rank failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let json: Value = serde_json::from_slice(&output.stdout).expect("parse rank json");
    let candidate = &json["candidates"][0];
    assert_eq!(candidate["skippy_bench_evidence_count"], 3);
    assert_eq!(
        candidate["focused_runtime_generated_tokens_per_second"],
        22.5
    );
    assert_eq!(candidate["focused_runtime_decode_elapsed_ms_p50"], 77.0);
    assert_eq!(
        candidate["estimated_decode_activation_transfer_bytes_per_token"],
        3072
    );
    assert_eq!(
        candidate["activation_transfer_source"],
        "direct_local_split_chain"
    );
    assert!(
        candidate["notes"]
            .as_array()
            .expect("notes array")
            .iter()
            .any(|note| note
                .as_str()
                .is_some_and(|text| text.contains("run quant-pack certify")))
    );
    assert!(
        candidate["notes"]
            .as_array()
            .expect("notes array")
            .iter()
            .any(|note| note
                .as_str()
                .is_some_and(|text| text.contains("direct local split-chain evidence")))
    );

    fs::remove_dir_all(run_dir).ok();
}

#[test]
fn quant_pack_rank_cli_fails_unverifiable_certification() {
    let run_dir = temp_dir("quant-pack-rank-cli-unverifiable-cert");
    write_rank_fixture(&run_dir);
    fs::write(
        run_dir.join("certification.json"),
        r#"{
  "status": "agent_quality_candidate",
  "gates": [],
  "skippy_bench_reports": [],
  "quality_evidence": []
}"#,
    )
    .expect("write unverifiable certification");

    let output = Command::new(env!("CARGO_BIN_EXE_skippy-model-package"))
        .arg("quant-pack")
        .arg("rank")
        .arg(&run_dir)
        .output()
        .expect("run skippy-model-package quant-pack rank");

    assert!(
        output.status.success(),
        "quant-pack rank failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let json: Value = serde_json::from_slice(&output.stdout).expect("parse rank json");
    let candidate = &json["candidates"][0];
    assert_eq!(
        candidate["certification_report_status"],
        "agent_quality_candidate"
    );
    assert_eq!(candidate["certification_subject_status"], "not_verifiable");
    assert_eq!(candidate["certification_status"], "failed");
    assert!(
        candidate["notes"]
            .as_array()
            .expect("notes array")
            .iter()
            .any(|note| note
                .as_str()
                .is_some_and(|text| text.contains("no subject hashes")))
    );

    fs::remove_dir_all(run_dir).ok();
}

#[test]
fn quant_pack_evidence_plan_all_cli_selects_ranked_candidates_and_writes_script() {
    let sweep_dir = temp_dir("quant-pack-evidence-plan-all-cli");
    write_evidence_plan_all_fixture(&sweep_dir);
    let report_path = sweep_dir.join("evidence-plan-all.json");
    let script_path = sweep_dir.join("run-evidence.sh");

    let output = Command::new(env!("CARGO_BIN_EXE_skippy-model-package"))
        .arg("quant-pack")
        .arg("evidence-plan-all")
        .arg(&sweep_dir)
        .arg("--hosts")
        .arg("host-a,host-b")
        .arg("--splits")
        .arg("20")
        .arg("--top-ranked")
        .arg("1")
        .arg("--out")
        .arg(&report_path)
        .arg("--script-out")
        .arg(&script_path)
        .output()
        .expect("run skippy-model-package quant-pack evidence-plan-all");

    assert!(
        output.status.success(),
        "quant-pack evidence-plan-all failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let report: Value =
        serde_json::from_slice(&fs::read(&report_path).expect("read evidence plan"))
            .expect("parse evidence plan");
    assert_eq!(report["kind"], "skippy_quant_pack_evidence_plan_all");
    assert_eq!(report["selection_source"], "top_ranked:1");
    let candidates = report["candidates"].as_array().expect("candidate array");
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0]["candidate"], "middle-compressed");
    assert_eq!(candidates[0]["ctx_size"], 32768);
    assert_eq!(report["final_rank"]["id"], "rank-after-evidence-all");
    assert!(
        report["final_rank"]["outputs"][0]
            .as_str()
            .expect("final rank output")
            .ends_with("evidence/rank-after-evidence.json")
    );

    let script = fs::read_to_string(&script_path).expect("read evidence script");
    assert!(script.contains("== shared setup =="));
    assert!(script.contains("just bench-corpus long"));
    assert!(script.contains("== skippy quant-pack evidence: middle-compressed =="));
    assert!(script.contains("skippy-model-package quant-pack certify"));
    assert!(script.contains("skippy-model-package quant-pack rank"));
    assert!(script.contains("== skippy quant-pack sweep rank =="));
    assert!(script.contains("rank-after-evidence-all"));
    assert!(script.contains("rank-after-evidence.json"));
    assert!(script.contains("missing expected evidence output"));

    fs::remove_dir_all(sweep_dir).ok();
}

#[test]
fn quant_pack_source_plan_cli_writes_hf_download_plan_and_script() {
    let dir = temp_dir("quant-pack-source-plan-cli");
    let report_path = dir.join("qwen-source-plan.json");
    let script_path = dir.join("fetch-qwen-source.sh");
    let workload_path = dir.join("qwen-hf-job-workload.sh");
    let submit_path = dir.join("qwen-hf-job-submit.json");
    let validate_path = dir.join("qwen-hf-job-validate.json");
    let local_dir = dir.join("source");
    let sweep_dir = dir.join("sweep");

    let output = Command::new(env!("CARGO_BIN_EXE_skippy-model-package"))
        .arg("quant-pack")
        .arg("source-plan")
        .arg("unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF")
        .arg("--revision")
        .arg("main")
        .arg("--local-dir")
        .arg(&local_dir)
        .arg("--allow-pattern")
        .arg("UD-Q4_K_XL/*.gguf")
        .arg("--source-file")
        .arg("qwen-00001-of-00012.gguf")
        .arg("--llama-quantize")
        .arg("/opt/llama-quantize")
        .arg("--quant-pack-out-dir")
        .arg(&sweep_dir)
        .arg("--candidate")
        .arg("middle-compressed")
        .arg("--expected-download-bytes")
        .arg("275600000000")
        .arg("--min-free-bytes")
        .arg("330000000000")
        .arg("--hf-jobs-workload-out")
        .arg(&workload_path)
        .arg("--hf-jobs-submit-json-out")
        .arg(&submit_path)
        .arg("--hf-jobs-image")
        .arg("ghcr.io/example/skippy-quant-pack:latest")
        .arg("--hf-jobs-timeout")
        .arg("36h")
        .arg("--hf-jobs-upload-repo")
        .arg("example/qwen480-skippy-pack")
        .arg("--out")
        .arg(&report_path)
        .arg("--script-out")
        .arg(&script_path)
        .output()
        .expect("run skippy-model-package quant-pack source-plan");

    assert!(
        output.status.success(),
        "quant-pack source-plan failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let report: Value = serde_json::from_slice(&fs::read(&report_path).expect("read source plan"))
        .expect("parse source plan");
    assert_eq!(report["kind"], "skippy_quant_pack_source_plan");
    assert_eq!(
        report["repo"],
        "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF"
    );
    assert_eq!(report["allow_patterns"][0], "UD-Q4_K_XL/*.gguf");
    assert_eq!(
        report["selected_source"],
        local_dir
            .join("qwen-00001-of-00012.gguf")
            .display()
            .to_string()
    );
    assert_eq!(report["commands"][0]["id"], "download-source");
    assert_eq!(report["commands"][0]["runnable"], true);
    assert_eq!(report["commands"][1]["id"], "quant-pack-build-all");
    assert_eq!(report["commands"][1]["runnable"], false);
    assert_eq!(report["expected_download_bytes"], 275_600_000_000_u64);
    assert_eq!(report["min_free_bytes"], 330_000_000_000_u64);
    assert_eq!(
        report["hf_jobs_workload"]["workload_script"],
        workload_path.display().to_string()
    );
    assert_eq!(
        report["hf_jobs_submit"]["submit_json"],
        submit_path.display().to_string()
    );
    assert_eq!(
        report["hf_jobs_submit"]["image"],
        "ghcr.io/example/skippy-quant-pack:latest"
    );

    let script = fs::read_to_string(&script_path).expect("read source script");
    assert!(script.contains("source-space-check"));
    assert!(script.contains("REQUIRED_BYTES=330000000000"));
    assert!(script.contains("hf download"));
    assert!(script.contains("'UD-Q4_K_XL/*.gguf'"));
    assert!(script.contains("Next command template:"));
    assert!(script.contains("skippy-model-package quant-pack build-all"));
    assert!(script.contains("qwen-00001-of-00012.gguf"));

    let workload = fs::read_to_string(&workload_path).expect("read workload script");
    assert!(workload.contains("HF_TOKEN is required"));
    assert!(workload.contains("LLAMA_QUANTIZE"));
    assert!(workload.contains("quant-pack build-all"));
    assert!(workload.contains("HF_UPLOAD_REPO"));
    assert!(
        workload.contains("hf repos create \"${HF_UPLOAD_REPO}\" --repo-type model --exist-ok")
    );

    let submit: Value = serde_json::from_slice(&fs::read(&submit_path).expect("read submit json"))
        .expect("parse submit json");
    assert_eq!(submit["operation"], "run");
    assert_eq!(
        submit["args"]["image"],
        "ghcr.io/example/skippy-quant-pack:latest"
    );
    assert_eq!(submit["args"]["timeout"], "36h");
    assert_eq!(submit["args"]["secrets"]["HF_TOKEN"], "$HF_TOKEN");
    assert_eq!(
        submit["args"]["env"]["HF_UPLOAD_REPO"],
        "example/qwen480-skippy-pack"
    );

    let validate = Command::new(env!("CARGO_BIN_EXE_skippy-model-package"))
        .arg("quant-pack")
        .arg("hf-jobs-validate")
        .arg(&submit_path)
        .arg("--expected-image")
        .arg("ghcr.io/example/skippy-quant-pack:latest")
        .arg("--expected-upload-repo")
        .arg("example/qwen480-skippy-pack")
        .arg("--out")
        .arg(&validate_path)
        .output()
        .expect("run skippy-model-package quant-pack hf-jobs-validate");

    assert!(
        validate.status.success(),
        "quant-pack hf-jobs-validate failed: {}",
        String::from_utf8_lossy(&validate.stderr)
    );
    let validate_report: Value =
        serde_json::from_slice(&fs::read(&validate_path).expect("read validate report"))
            .expect("parse validate report");
    assert_eq!(
        validate_report["kind"],
        "skippy_quant_pack_hf_jobs_validate"
    );
    assert_eq!(validate_report["status"], "valid");
    let hf_jobs_argv = validate_report["hf_jobs_cli"]["argv"]
        .as_array()
        .expect("hf jobs argv");
    assert_eq!(hf_jobs_argv[0], "hf");
    assert_eq!(hf_jobs_argv[1], "jobs");
    assert_eq!(hf_jobs_argv[2], "run");
    assert!(hf_jobs_argv.iter().any(|arg| arg == "--detach"));
    assert!(hf_jobs_argv.iter().any(|arg| arg == "--flavor"));
    assert!(hf_jobs_argv.iter().any(|arg| arg == "cpu-xl"));
    assert!(
        validate_report["hf_jobs_cli"]["shell"]
            .as_str()
            .expect("hf jobs shell command")
            .contains("--secrets HF_TOKEN")
    );
    assert!(
        validate_report["hf_jobs_cli"]["shell"]
            .as_str()
            .expect("hf jobs shell command")
            .contains("--env example/qwen480-skippy-pack")
            || validate_report["hf_jobs_cli"]["shell"]
                .as_str()
                .expect("hf jobs shell command")
                .contains("HF_UPLOAD_REPO=example/qwen480-skippy-pack")
    );
    assert!(
        validate_report["hf_jobs_cli"]["shell"]
            .as_str()
            .expect("hf jobs shell command")
            .contains("ghcr.io/example/skippy-quant-pack:latest")
    );

    fs::remove_dir_all(dir).ok();
}

#[test]
fn quant_pack_hf_jobs_validate_cli_rejects_bad_payload() {
    let dir = temp_dir("quant-pack-hf-jobs-validate-cli");
    let submit_path = dir.join("bad-submit.json");
    fs::write(
        &submit_path,
        r#"{"operation":"uv","args":{"image":"","command":[],"flavor":"bogus"}}"#,
    )
    .expect("write bad submit json");

    let output = Command::new(env!("CARGO_BIN_EXE_skippy-model-package"))
        .arg("quant-pack")
        .arg("hf-jobs-validate")
        .arg(&submit_path)
        .output()
        .expect("run skippy-model-package quant-pack hf-jobs-validate");

    assert!(
        !output.status.success(),
        "bad submit payload unexpectedly passed"
    );
    let report: Value =
        serde_json::from_slice(&output.stdout).expect("parse invalid validate report");
    assert_eq!(report["status"], "invalid");
    assert!(
        report["checks"]
            .as_array()
            .expect("checks array")
            .iter()
            .any(|check| check["id"] == "operation" && check["status"] == "invalid")
    );

    fs::remove_dir_all(dir).ok();
}

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("skippy-model-package-{name}-{nanos}"));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn write_evidence_plan_all_fixture(sweep_dir: &Path) {
    let invalid = sweep_dir.join("invalid-fast");
    let selected = sweep_dir.join("middle-compressed");
    write_evidence_candidate_fixture(&invalid, "invalid-fast", "org/repo:invalid-fast");
    write_evidence_candidate_fixture(&selected, "middle-compressed", "org/repo:middle-compressed");
    fs::write(
        sweep_dir.join("quant-pack-build-all.json"),
        format!(
            r#"{{
  "schema_version": 1,
  "kind": "skippy_quant_pack_build_all",
  "ctx_size": 32768,
  "n_gpu_layers": -1,
  "cache_type_k": "q8_0",
  "cache_type_v": "f16",
  "activation_wire_dtype": "q8",
  "candidates": [
    {{"candidate": "invalid-fast", "run_dir": "{}", "manifest": "{}/quant-pack-build.json"}},
    {{"candidate": "middle-compressed", "run_dir": "{}", "manifest": "{}/quant-pack-build.json"}}
  ],
  "rank": "{}/quant-pack-rank.json"
}}"#,
            invalid.display(),
            invalid.display(),
            selected.display(),
            selected.display(),
            sweep_dir.display()
        ),
    )
    .expect("write build-all manifest");
    fs::write(
        sweep_dir.join("quant-pack-rank.json"),
        r#"{
  "schema_version": 1,
  "kind": "skippy_quant_pack_rank",
  "candidates": [
    {"rank": 1, "candidate": "invalid-fast", "valid": false},
    {"rank": 2, "candidate": "middle-compressed", "valid": true}
  ]
}"#,
    )
    .expect("write rank report");
}

fn write_evidence_candidate_fixture(run_dir: &Path, candidate: &str, model_id: &str) {
    fs::create_dir_all(run_dir.join("package")).expect("create evidence candidate package");
    fs::write(
        run_dir.join("quant-pack-build.json"),
        format!(
            r#"{{
  "candidate": "{candidate}",
  "stages": 2,
  "package": "package",
  "quantized_model": "model.gguf"
}}"#
        ),
    )
    .expect("write evidence candidate manifest");
    fs::write(
        run_dir.join("package/model-package.json"),
        format!(
            r#"{{
  "model_id": "{model_id}",
  "layer_count": 40
}}"#
        ),
    )
    .expect("write evidence package manifest");
    fs::write(run_dir.join("model.gguf"), b"gguf").expect("write fake model");
}

fn write_rank_fixture(run_dir: &Path) {
    fs::write(
        run_dir.join("quant-pack-build.json"),
        r#"{
  "candidate": "middle-compressed",
  "agent_pack": "agent-pack.json",
  "preflight": "preflight.json"
}"#,
    )
    .expect("write build manifest");
    fs::write(
        run_dir.join("agent-pack.json"),
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
        run_dir.join("preflight.json"),
        r#"{
  "valid": true,
  "activation_width": 4096,
  "stages": [
    {
      "stage_index": 0,
      "layer_start": 0,
      "layer_end": 2,
      "artifact_bytes": 10000
    },
    {
      "stage_index": 1,
      "layer_start": 2,
      "layer_end": 5,
      "artifact_bytes": 25000
    }
  ]
}"#,
    )
    .expect("write preflight");
}
