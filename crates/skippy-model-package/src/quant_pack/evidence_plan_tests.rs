use super::*;

#[test]
fn infer_even_splits_balances_layers_across_stages() {
    assert_eq!(infer_even_splits(40, 3).unwrap(), "13,26");
    assert_eq!(infer_even_splits(40, 2).unwrap(), "20");
    assert_eq!(infer_even_splits(40, 1).unwrap(), "");
}

#[test]
fn split_plan_accepts_cli_override() {
    let manifest = test_build_manifest_input("middle-compressed");
    let plan = split_plan(
        Some("14,27"),
        40,
        3,
        PlanSplitRequest {
            run_dir: Path::new("/tmp/run"),
            manifest: &manifest,
        },
    )
    .expect("valid split override");

    assert_eq!(plan.splits, "14,27");
    assert_eq!(plan.source.as_str(), "cli_override");
}

#[test]
fn evidence_plan_uses_candidate_stage_hints_when_splits_omitted() {
    let dir = unique_test_dir("evidence-plan-stage-hints");
    write_candidate_fixture_with_plan(
        &dir,
        "ffn-compressed-attention-protected",
        "org/repo:ffn-compressed-attention-protected",
        62,
        4,
    );
    let report_path = dir.join("evidence-plan.json");

    run_quant_pack_evidence_plan(QuantPackEvidencePlanArgs {
        run: dir.clone(),
        hosts: "host-a,host-b,host-c,host-d".to_string(),
        splits: None,
        base_url: "http://127.0.0.1:9337/v1".to_string(),
        token_corpus: PathBuf::from("target/bench-corpora/long/corpus.jsonl"),
        chat_corpus: PathBuf::from("target/bench-corpora/coding-loop/corpus.jsonl"),
        long_context_corpus: PathBuf::from("target/bench-corpora/long-context/corpus.jsonl"),
        ctx_size: 8192,
        n_gpu_layers: -1,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        max_tokens: 512,
        include_local_split_evidence: false,
        local_split_prompt: DEFAULT_LOCAL_SPLIT_PROMPT.to_string(),
        skippy_bench_bin: PathBuf::from("skippy-bench"),
        skippy_model_package_bin: PathBuf::from("skippy-model-package"),
        agent_tool_call_script: PathBuf::from(DEFAULT_AGENT_TOOL_CALL_SCRIPT),
        kv_tool_loop_script: PathBuf::from(DEFAULT_KV_TOOL_LOOP_SCRIPT),
        runbook_cwd: Some(dir.clone()),
        activation_wire_dtype: "f16".to_string(),
        attempts: 2,
        focused_runtime: FocusedRuntimeEvidenceArgs::default(),
        evidence_dir: None,
        out: Some(report_path.clone()),
        script_out: None,
    })
    .expect("write evidence plan");

    let report = read_json::<serde_json::Value>(&report_path).expect("read report");
    assert_eq!(report["stage_count"], 4);
    assert_eq!(report["splits"], "16,32,47");
    assert_eq!(report["split_source"], "candidate_stage_hints");
    let commands = report["commands"].as_array().expect("commands array");
    let schema_smoke = commands
        .iter()
        .find(|command| command["id"] == "focused-runtime-schema-smoke")
        .expect("focused-runtime schema-smoke command");
    assert_eq!(
        schema_smoke["evidence_type"],
        "skippy-bench-focused-runtime-schema-smoke"
    );
    let schema_smoke_argv = schema_smoke["argv"].as_array().expect("argv array");
    assert!(schema_smoke_argv.iter().any(|arg| arg == "--schema-smoke"));
    assert!(
        schema_smoke_argv
            .iter()
            .any(|arg| arg == "--allow-uneven-stage-ranges")
    );
    assert!(
        !schema_smoke_argv
            .iter()
            .any(|arg| arg == "--execute-remote")
    );
    let focused_runtime = commands
        .iter()
        .find(|command| command["id"] == "focused-runtime")
        .expect("focused-runtime command");
    assert_eq!(
        focused_runtime["evidence_type"],
        "skippy-bench-focused-runtime"
    );
    let argv = focused_runtime["argv"].as_array().expect("argv array");
    assert!(argv.iter().any(|arg| arg == "16,32,47"));
    assert!(argv.iter().any(|arg| arg == "--allow-uneven-stage-ranges"));
    let token_lengths = commands
        .iter()
        .find(|command| command["id"] == "token-lengths")
        .expect("token-lengths command");
    assert_eq!(token_lengths["evidence_type"], "skippy-bench-token-lengths");
    let chat_corpus = commands
        .iter()
        .find(|command| command["id"] == "chat-corpus")
        .expect("chat-corpus command");
    assert_eq!(chat_corpus["evidence_type"], "skippy-bench-chat-corpus");
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn evidence_plan_cli_splits_override_candidate_stage_hints() {
    let dir = unique_test_dir("evidence-plan-stage-hints-cli");
    write_candidate_fixture_with_plan(
        &dir,
        "ffn-compressed-attention-protected",
        "org/repo:ffn-compressed-attention-protected",
        62,
        4,
    );
    let report_path = dir.join("evidence-plan.json");

    run_quant_pack_evidence_plan(QuantPackEvidencePlanArgs {
        run: dir.clone(),
        hosts: "host-a,host-b,host-c,host-d".to_string(),
        splits: Some("15,31,46".to_string()),
        base_url: "http://127.0.0.1:9337/v1".to_string(),
        token_corpus: PathBuf::from("target/bench-corpora/long/corpus.jsonl"),
        chat_corpus: PathBuf::from("target/bench-corpora/coding-loop/corpus.jsonl"),
        long_context_corpus: PathBuf::from("target/bench-corpora/long-context/corpus.jsonl"),
        ctx_size: 8192,
        n_gpu_layers: -1,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        max_tokens: 512,
        include_local_split_evidence: false,
        local_split_prompt: DEFAULT_LOCAL_SPLIT_PROMPT.to_string(),
        skippy_bench_bin: PathBuf::from("skippy-bench"),
        skippy_model_package_bin: PathBuf::from("skippy-model-package"),
        agent_tool_call_script: PathBuf::from(DEFAULT_AGENT_TOOL_CALL_SCRIPT),
        kv_tool_loop_script: PathBuf::from(DEFAULT_KV_TOOL_LOOP_SCRIPT),
        runbook_cwd: Some(dir.clone()),
        activation_wire_dtype: "f16".to_string(),
        attempts: 2,
        focused_runtime: FocusedRuntimeEvidenceArgs::default(),
        evidence_dir: None,
        out: Some(report_path.clone()),
        script_out: None,
    })
    .expect("write evidence plan");

    let report = read_json::<serde_json::Value>(&report_path).expect("read report");
    assert_eq!(report["splits"], "15,31,46");
    assert_eq!(report["split_source"], "cli_override");
    let commands = report["commands"].as_array().expect("commands array");
    let focused_runtime = commands
        .iter()
        .find(|command| command["id"] == "focused-runtime")
        .expect("focused-runtime command");
    let argv = focused_runtime["argv"].as_array().expect("argv array");
    assert!(!argv.iter().any(|arg| arg == "--allow-uneven-stage-ranges"));
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn invalid_candidate_stage_hints_fail_evidence_plan() {
    let dir = unique_test_dir("evidence-plan-invalid-stage-hints");
    write_candidate_fixture_with_plan(
        &dir,
        "ffn-compressed-attention-protected",
        "org/repo:ffn-compressed-attention-protected",
        62,
        4,
    );
    fs::write(
        dir.join("quant-plan.json"),
        r#"{
  "schema_version": 1,
  "candidates": [
    {
      "id": "ffn-compressed-attention-protected",
      "stage_hints": [
        {"stage_index": 0, "layer_start": 0, "layer_end": 16},
        {"stage_index": 1, "layer_start": 17, "layer_end": 32},
        {"stage_index": 2, "layer_start": 32, "layer_end": 47},
        {"stage_index": 3, "layer_start": 47, "layer_end": 62}
      ]
    }
  ]
}"#,
    )
    .expect("write invalid plan");

    let err = run_quant_pack_evidence_plan(QuantPackEvidencePlanArgs {
        run: dir.clone(),
        hosts: "host-a,host-b,host-c,host-d".to_string(),
        splits: None,
        base_url: "http://127.0.0.1:9337/v1".to_string(),
        token_corpus: PathBuf::from("target/bench-corpora/long/corpus.jsonl"),
        chat_corpus: PathBuf::from("target/bench-corpora/coding-loop/corpus.jsonl"),
        long_context_corpus: PathBuf::from("target/bench-corpora/long-context/corpus.jsonl"),
        ctx_size: 8192,
        n_gpu_layers: -1,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        max_tokens: 512,
        include_local_split_evidence: false,
        local_split_prompt: DEFAULT_LOCAL_SPLIT_PROMPT.to_string(),
        skippy_bench_bin: PathBuf::from("skippy-bench"),
        skippy_model_package_bin: PathBuf::from("skippy-model-package"),
        agent_tool_call_script: PathBuf::from(DEFAULT_AGENT_TOOL_CALL_SCRIPT),
        kv_tool_loop_script: PathBuf::from(DEFAULT_KV_TOOL_LOOP_SCRIPT),
        runbook_cwd: Some(dir.clone()),
        activation_wire_dtype: "f16".to_string(),
        attempts: 2,
        focused_runtime: FocusedRuntimeEvidenceArgs::default(),
        evidence_dir: None,
        out: Some(dir.join("evidence-plan.json")),
        script_out: None,
    })
    .expect_err("invalid stage hints should fail");

    assert!(err.to_string().contains("invalid stage_hints"));
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn split_validation_rejects_wrong_boundary_count() {
    let err = validate_splits("20", 40, 3).expect_err("wrong boundary count should fail");

    assert!(err.to_string().contains("exactly 2 boundaries"));
}

#[test]
fn split_validation_rejects_non_increasing_boundaries() {
    let err = validate_splits("20,20", 40, 3).expect_err("duplicate boundary should fail");

    assert!(err.to_string().contains("strictly increasing"));
}

#[test]
fn evidence_plan_can_include_local_split_chain_evidence() {
    let dir = unique_test_dir("evidence-plan-local-split");
    write_candidate_fixture_with_shape(
        &dir,
        "studio-local",
        "org/qwen-coder:studio-local",
        48,
        3,
        None,
    );
    let report_path = dir.join("evidence-plan.json");
    let focused_runtime = FocusedRuntimeEvidenceArgs {
        startup_timeout_secs: Some(600),
        ..FocusedRuntimeEvidenceArgs::default()
    };

    run_quant_pack_evidence_plan(QuantPackEvidencePlanArgs {
        run: dir.clone(),
        hosts: "studio-stage-0,studio-stage-1,studio-stage-2".to_string(),
        splits: Some("16,32".to_string()),
        base_url: "http://127.0.0.1:9337/v1".to_string(),
        token_corpus: PathBuf::from("target/bench-corpora/long/corpus.jsonl"),
        chat_corpus: PathBuf::from("target/bench-corpora/coding-loop/corpus.jsonl"),
        long_context_corpus: PathBuf::from("target/bench-corpora/long-context/corpus.jsonl"),
        ctx_size: 8192,
        n_gpu_layers: -1,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        max_tokens: 512,
        include_local_split_evidence: true,
        local_split_prompt: "Write a small Rust parser.".to_string(),
        skippy_bench_bin: PathBuf::from("skippy-bench"),
        skippy_model_package_bin: PathBuf::from("skippy-model-package"),
        agent_tool_call_script: PathBuf::from(DEFAULT_AGENT_TOOL_CALL_SCRIPT),
        kv_tool_loop_script: PathBuf::from(DEFAULT_KV_TOOL_LOOP_SCRIPT),
        runbook_cwd: Some(dir.clone()),
        activation_wire_dtype: "q8".to_string(),
        attempts: 2,
        focused_runtime,
        evidence_dir: None,
        out: Some(report_path.clone()),
        script_out: None,
    })
    .expect("write evidence plan");

    let report = read_json::<serde_json::Value>(&report_path).expect("read report");
    assert_eq!(report["include_local_split_evidence"], true);
    assert_eq!(report["local_split_prompt"], "Write a small Rust parser.");
    let commands = report["commands"].as_array().expect("commands array");
    let local_split = commands
        .iter()
        .find(|command| command["id"] == "local-split-chain")
        .expect("local split command");
    assert_eq!(
        local_split["evidence_type"],
        "skippy-bench-local-split-chain"
    );
    let argv = local_split["argv"].as_array().expect("argv array");
    assert!(argv.iter().any(|arg| arg == "local-split-chain-binary"));
    let splits_index = argv
        .iter()
        .position(|arg| arg == "--splits")
        .expect("--splits should be present");
    assert_eq!(argv[splits_index + 1], "16,32");
    let startup_timeout_index = argv
        .iter()
        .position(|arg| arg == "--startup-timeout-secs")
        .expect("--startup-timeout-secs should be present");
    assert_eq!(argv[startup_timeout_index + 1], "600");
    assert!(argv.iter().any(|arg| arg == "--layer-end"));
    assert!(argv.iter().any(|arg| arg == "48"));
    assert!(argv.iter().any(|arg| arg == "--output"));
    assert!(
        argv.iter()
            .any(|arg| arg == &format!("{}/evidence/local-split-chain.json", dir.display()))
    );
    let certify = commands
        .iter()
        .find(|command| command["id"] == "certify")
        .expect("certify command");
    let certify_argv = certify["argv"].as_array().expect("certify argv");
    assert!(
        certify_argv
            .iter()
            .any(|arg| arg == &format!("{}/evidence/local-split-chain.json", dir.display()))
    );
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn local_split_chain_evidence_requires_at_least_three_stages() {
    let dir = unique_test_dir("evidence-plan-local-split-reject");
    write_candidate_fixture(&dir, "two-stage", "org/qwen-coder:two-stage");

    let err = run_quant_pack_evidence_plan(QuantPackEvidencePlanArgs {
        run: dir.clone(),
        hosts: "host-a,host-b".to_string(),
        splits: None,
        base_url: "http://127.0.0.1:9337/v1".to_string(),
        token_corpus: PathBuf::from("target/bench-corpora/long/corpus.jsonl"),
        chat_corpus: PathBuf::from("target/bench-corpora/coding-loop/corpus.jsonl"),
        long_context_corpus: PathBuf::from("target/bench-corpora/long-context/corpus.jsonl"),
        ctx_size: 8192,
        n_gpu_layers: -1,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        max_tokens: 512,
        include_local_split_evidence: true,
        local_split_prompt: DEFAULT_LOCAL_SPLIT_PROMPT.to_string(),
        skippy_bench_bin: PathBuf::from("skippy-bench"),
        skippy_model_package_bin: PathBuf::from("skippy-model-package"),
        agent_tool_call_script: PathBuf::from(DEFAULT_AGENT_TOOL_CALL_SCRIPT),
        kv_tool_loop_script: PathBuf::from(DEFAULT_KV_TOOL_LOOP_SCRIPT),
        runbook_cwd: Some(dir.clone()),
        activation_wire_dtype: "q8".to_string(),
        attempts: 2,
        focused_runtime: FocusedRuntimeEvidenceArgs::default(),
        evidence_dir: None,
        out: Some(dir.join("evidence-plan.json")),
        script_out: None,
    })
    .expect_err("2-stage local split chain should be rejected");

    assert!(
        err.to_string()
            .contains("requires at least two split boundaries")
    );
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn evidence_plan_all_filters_candidates_and_inherits_runtime_shape() {
    let dir = unique_test_dir("evidence-plan-all");
    let run_a = dir.join("baseline-source-quant");
    let run_b = dir.join("middle-compressed");
    write_candidate_fixture(
        &run_a,
        "baseline-source-quant",
        "org/repo:baseline-source-quant",
    );
    write_candidate_fixture(&run_b, "middle-compressed", "org/repo:middle-compressed");
    let build_all = dir.join("quant-pack-build-all.json");
    fs::write(
            &build_all,
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
    {{"candidate": "baseline-source-quant", "run_dir": "{}", "manifest": "{}/quant-pack-build.json"}},
    {{"candidate": "middle-compressed", "run_dir": "{}", "manifest": "{}/quant-pack-build.json"}}
  ],
  "rank": "{}/quant-pack-rank.json"
}}"#,
                run_a.display(),
                run_a.display(),
                run_b.display(),
                run_b.display(),
                dir.display()
            ),
        )
        .expect("write build-all manifest");

    let report_path = dir.join("evidence-plan-all.json");
    run_quant_pack_evidence_plan_all(QuantPackEvidencePlanAllArgs {
        build_all,
        hosts: "host-a,host-b".to_string(),
        splits: Some("20".to_string()),
        candidates: vec!["middle-compressed".to_string()],
        top_ranked: None,
        base_url: "http://127.0.0.1:9337/v1".to_string(),
        token_corpus: PathBuf::from("target/bench-corpora/long/corpus.jsonl"),
        chat_corpus: PathBuf::from("target/bench-corpora/coding-loop/corpus.jsonl"),
        long_context_corpus: PathBuf::from("target/bench-corpora/long-context/corpus.jsonl"),
        ctx_size: None,
        n_gpu_layers: None,
        cache_type_k: None,
        cache_type_v: None,
        max_tokens: 256,
        include_local_split_evidence: false,
        local_split_prompt: DEFAULT_LOCAL_SPLIT_PROMPT.to_string(),
        skippy_bench_bin: PathBuf::from("skippy-bench"),
        skippy_model_package_bin: PathBuf::from("skippy-model-package"),
        agent_tool_call_script: PathBuf::from(DEFAULT_AGENT_TOOL_CALL_SCRIPT),
        kv_tool_loop_script: PathBuf::from(DEFAULT_KV_TOOL_LOOP_SCRIPT),
        runbook_cwd: Some(dir.clone()),
        activation_wire_dtype: None,
        attempts: 3,
        focused_runtime: FocusedRuntimeEvidenceArgs::default(),
        evidence_root: None,
        out: Some(report_path.clone()),
        script_out: None,
    })
    .expect("write evidence-plan-all report");

    let report = read_json::<serde_json::Value>(&report_path).expect("read report");
    let candidates = report["candidates"].as_array().expect("candidate array");
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0]["candidate"], "middle-compressed");
    assert_eq!(candidates[0]["stage_count"], 2);
    assert_eq!(candidates[0]["ctx_size"], 32768);
    assert_eq!(candidates[0]["cache_type_k"], "q8_0");
    assert_eq!(candidates[0]["activation_wire_dtype"], "q8");
    let commands = candidates[0]["commands"]
        .as_array()
        .expect("commands array");
    let schema_smoke = commands
        .iter()
        .find(|command| command["id"] == "focused-runtime-schema-smoke")
        .expect("focused-runtime schema-smoke command");
    let schema_smoke_argv = schema_smoke["argv"].as_array().expect("argv array");
    assert!(schema_smoke_argv.iter().any(|arg| arg == "--schema-smoke"));
    assert!(
        !schema_smoke_argv
            .iter()
            .any(|arg| arg == "--execute-remote")
    );
    let focused_runtime = commands
        .iter()
        .find(|command| command["id"] == "focused-runtime")
        .expect("focused-runtime command");
    let argv = focused_runtime["argv"].as_array().expect("argv array");
    assert!(argv.iter().any(|arg| arg == "--execute-remote"));
    assert!(argv.iter().any(|arg| arg == "32768"));
    assert!(argv.iter().any(|arg| arg == "q8_0"));
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn evidence_plan_all_selects_top_valid_ranked_candidates() {
    let dir = unique_test_dir("evidence-plan-all-ranked");
    let run_a = dir.join("invalid-fast");
    let run_b = dir.join("middle-compressed");
    let run_c = dir.join("ffn-compressed-attention-protected");
    write_candidate_fixture(&run_a, "invalid-fast", "org/repo:invalid-fast");
    write_candidate_fixture(&run_b, "middle-compressed", "org/repo:middle-compressed");
    write_candidate_fixture(
        &run_c,
        "ffn-compressed-attention-protected",
        "org/repo:ffn-compressed-attention-protected",
    );
    write_build_all_fixture(&dir, [&run_a, &run_b, &run_c]);
    fs::write(
        dir.join("quant-pack-rank.json"),
        r#"{
  "schema_version": 1,
  "kind": "skippy_quant_pack_rank",
  "candidates": [
    {"rank": 1, "candidate": "invalid-fast", "valid": false},
    {"rank": 2, "candidate": "middle-compressed", "valid": true},
    {"rank": 3, "candidate": "ffn-compressed-attention-protected", "valid": true}
  ]
}"#,
    )
    .expect("write rank report");
    let report_path = dir.join("evidence-plan-all.json");

    run_quant_pack_evidence_plan_all(QuantPackEvidencePlanAllArgs {
        build_all: dir.clone(),
        hosts: "host-a,host-b".to_string(),
        splits: Some("20".to_string()),
        candidates: Vec::new(),
        top_ranked: Some(1),
        base_url: "http://127.0.0.1:9337/v1".to_string(),
        token_corpus: PathBuf::from("target/bench-corpora/long/corpus.jsonl"),
        chat_corpus: PathBuf::from("target/bench-corpora/coding-loop/corpus.jsonl"),
        long_context_corpus: PathBuf::from("target/bench-corpora/long-context/corpus.jsonl"),
        ctx_size: None,
        n_gpu_layers: None,
        cache_type_k: None,
        cache_type_v: None,
        max_tokens: 256,
        include_local_split_evidence: false,
        local_split_prompt: DEFAULT_LOCAL_SPLIT_PROMPT.to_string(),
        skippy_bench_bin: PathBuf::from("skippy-bench"),
        skippy_model_package_bin: PathBuf::from("skippy-model-package"),
        agent_tool_call_script: PathBuf::from(DEFAULT_AGENT_TOOL_CALL_SCRIPT),
        kv_tool_loop_script: PathBuf::from(DEFAULT_KV_TOOL_LOOP_SCRIPT),
        runbook_cwd: Some(dir.clone()),
        activation_wire_dtype: None,
        attempts: 3,
        focused_runtime: FocusedRuntimeEvidenceArgs::default(),
        evidence_root: None,
        out: Some(report_path.clone()),
        script_out: None,
    })
    .expect("write ranked evidence-plan-all report");

    let report = read_json::<serde_json::Value>(&report_path).expect("read report");
    assert_eq!(report["selection_source"], "top_ranked:1");
    let candidates = report["candidates"].as_array().expect("candidate array");
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0]["candidate"], "middle-compressed");
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn manifest_path_resolution_accepts_cwd_relative_generated_paths() {
    let dir = unique_test_dir("manifest-path-resolution");
    let cwd_relative = dir.join("candidate/package");
    fs::create_dir_all(&cwd_relative).expect("create cwd-relative path");
    let run_dir = dir.join("candidate");

    let resolved = resolve_manifest_path(&run_dir, &cwd_relative.display().to_string());

    assert_eq!(resolved, cwd_relative);
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn evidence_commands_include_required_certification_reports() {
    let commands = test_evidence_commands();

    let certify = commands
        .iter()
        .find(|command| command.id == "certify")
        .expect("certify command");

    assert!(certify.argv.contains(&"--require-skippy-bench".to_string()));
    assert!(
        certify
            .argv
            .contains(&"/tmp/run/evidence/focused-runtime-report.json".to_string())
    );
    assert!(
        certify
            .argv
            .contains(&"/tmp/run/evidence/chat-corpus.json".to_string())
    );
    assert!(
        certify
            .argv
            .contains(&"/tmp/run/evidence/prompt-lengths-summary.json".to_string())
    );
    assert_arg_value(certify, "--activation-wire-dtype", "q8");
    assert_arg_value(certify, "--ctx-size", "8192");
    assert_arg_value(certify, "--n-gpu-layers", "-1");
    assert_arg_value(certify, "--cache-type-k", "f16");
    assert_arg_value(certify, "--cache-type-v", "f16");
}

#[test]
fn evidence_commands_rerank_after_certification() {
    let commands = test_evidence_commands();
    let rank = command_by_id(&commands, "rank-after-evidence");

    assert_eq!(rank.outputs, ["/tmp/run/evidence/rank-after-evidence.json"]);
    assert_eq!(
        rank.shell,
        "skippy-model-package quant-pack rank /tmp/run --activation-wire-dtype q8 --ctx-size 8192 --n-gpu-layers=-1 --cache-type-k f16 --cache-type-v f16 --out /tmp/run/evidence/rank-after-evidence.json"
    );
    assert_arg_value(rank, "--activation-wire-dtype", "q8");
    assert_arg_value(rank, "--ctx-size", "8192");
    assert_arg_value(rank, "--n-gpu-layers", "-1");
    assert_arg_value(rank, "--cache-type-k", "f16");
    assert_arg_value(rank, "--cache-type-v", "f16");
}

#[test]
fn evidence_commands_pin_runtime_shape_inputs() {
    let commands = test_evidence_commands();
    let token_lengths = command_by_id(&commands, "token-lengths");
    let schema_smoke = command_by_id(&commands, "focused-runtime-schema-smoke");
    let focused_runtime = command_by_id(&commands, "focused-runtime");
    let chat_corpus = command_by_id(&commands, "chat-corpus");

    assert_arg_value(token_lengths, "--layer-end", "40");
    assert_arg_value(token_lengths, "--enable-thinking", "false");
    assert!(schema_smoke.argv.contains(&"--schema-smoke".to_string()));
    assert!(!schema_smoke.argv.contains(&"--execute-remote".to_string()));
    assert_arg_value(schema_smoke, "--splits", "20");
    assert_arg_value(schema_smoke, "--layer-end", "40");
    assert_arg_value(schema_smoke, "--ctx-size", "8192");
    assert_arg_value(schema_smoke, "--activation-wire-dtype", "q8");
    assert_arg_value(
        focused_runtime,
        "--prompt-corpus",
        "target/bench-corpora/coding-loop/corpus.jsonl",
    );
    assert_arg_value(focused_runtime, "--max-new-tokens", "512");
    assert_arg_value(focused_runtime, "--n-gpu-layers", "-1");
    assert_arg_value(focused_runtime, "--cache-type-k", "f16");
    assert_arg_value(focused_runtime, "--cache-type-v", "f16");
    assert_arg_value(focused_runtime, "--activation-wire-dtype", "q8");
    assert_arg_value(chat_corpus, "--enable-thinking", "false");
}

#[test]
fn evidence_commands_passthrough_focused_runtime_lab_options() {
    let options = FocusedRuntimeEvidenceArgs {
        metrics_server_bin: Some(PathBuf::from("target/release/metrics-server")),
        stage_server_bin: Some(PathBuf::from("target/release/skippy-server")),
        lab_preflight_script: Some(PathBuf::from("scripts/qwen-lab-preflight.sh")),
        lab_preflight_hosts: Some("192.168.0.2,192.168.0.4".to_string()),
        lab_preflight_min_free_gb: Some(80),
        lab_preflight_ports: Some("9337 14317 19031".to_string()),
        lab_preflight_ssh_opts: Some("-o BatchMode=yes -o ConnectTimeout=7 -p 2222".to_string()),
        work_dir: Some(PathBuf::from("/Volumes/External/qwen-evidence")),
        remote_root: Some("/mnt/skippy-runtime-bench".to_string()),
        remote_root_map: Some("host-b=/data/skippy,host-c=/scratch/skippy".to_string()),
        remote_shared_root_map: Some("host-a=/Volumes/External/qwen-evidence".to_string()),
        endpoint_host_map: Some("host-b=192.168.0.4,host-c=192.168.0.3".to_string()),
        ssh_opts: Some("-p 2222 -o BatchMode=yes".to_string()),
        metrics_otlp_grpc_url: Some("http://studio54.local:14317".to_string()),
        remote_bind_host: Some("0.0.0.0".to_string()),
        first_stage_port: Some(19041),
        startup_timeout_secs: Some(180),
        stage_max_inflight: Some(8),
        stage_reply_credit_limit: Some(16),
        stage_downstream_wire_delay_ms: Some(0.25),
        stage_downstream_wire_mbps: Some(5000.0),
        stage_telemetry_queue_capacity: Some(16384),
        stage_telemetry_level: Some("summary".to_string()),
        rsync_model_artifacts: true,
        keep_remote: true,
        child_logs: true,
        stage_async_prefill_forward: true,
        allow_uneven_stage_ranges: true,
    };
    let commands = evidence_commands(EvidenceCommandInputs {
        focused_runtime: &options,
        ..test_evidence_command_inputs()
    });
    let focused_runtime = command_by_id(&commands, "focused-runtime");
    let schema_smoke = command_by_id(&commands, "focused-runtime-schema-smoke");
    let lab_preflight = command_by_id(&commands, "focused-runtime-lab-preflight");

    assert_eq!(lab_preflight.argv.first().map(String::as_str), Some("bash"));
    assert_arg_value(
        lab_preflight,
        "-lc",
        "scripts/qwen-lab-preflight.sh --hosts 192.168.0.2,192.168.0.4 --min-free-gb 80 --ports '9337 14317 19031' --ssh-opts '-o BatchMode=yes -o ConnectTimeout=7 -p 2222' --out /tmp/run/evidence/focused-runtime-lab-preflight.txt && printf '%s\\n' 'focused-runtime-lab-preflight: ok' > /tmp/run/evidence/focused-runtime-lab-preflight.ok",
    );
    assert_eq!(
        lab_preflight.outputs,
        [
            "/tmp/run/evidence/focused-runtime-lab-preflight.txt",
            "/tmp/run/evidence/focused-runtime-lab-preflight.ok"
        ]
    );

    assert_arg_value(
        focused_runtime,
        "--metrics-server-bin",
        "target/release/metrics-server",
    );
    assert_arg_value(
        focused_runtime,
        "--stage-server-bin",
        "target/release/skippy-server",
    );
    assert_arg_value(
        focused_runtime,
        "--work-dir",
        "/Volumes/External/qwen-evidence",
    );
    assert_arg_value(
        focused_runtime,
        "--remote-root",
        "/mnt/skippy-runtime-bench",
    );
    assert_arg_value(
        focused_runtime,
        "--remote-root-map",
        "host-b=/data/skippy,host-c=/scratch/skippy",
    );
    assert_arg_value(
        focused_runtime,
        "--remote-shared-root-map",
        "host-a=/Volumes/External/qwen-evidence",
    );
    assert_arg_value(
        focused_runtime,
        "--endpoint-host-map",
        "host-b=192.168.0.4,host-c=192.168.0.3",
    );
    assert_arg_value(focused_runtime, "--ssh-opts", "-p 2222 -o BatchMode=yes");
    assert_arg_value(
        focused_runtime,
        "--metrics-otlp-grpc-url",
        "http://studio54.local:14317",
    );
    assert_arg_value(focused_runtime, "--first-stage-port", "19041");
    assert_arg_value(focused_runtime, "--startup-timeout-secs", "180");
    assert_arg_value(focused_runtime, "--stage-max-inflight", "8");
    assert_arg_value(focused_runtime, "--stage-reply-credit-limit", "16");
    assert_arg_value(focused_runtime, "--stage-downstream-wire-delay-ms", "0.25");
    assert_arg_value(focused_runtime, "--stage-downstream-wire-mbps", "5000");
    assert_arg_value(focused_runtime, "--stage-telemetry-queue-capacity", "16384");
    assert_arg_value(focused_runtime, "--stage-telemetry-level", "summary");
    assert!(
        focused_runtime
            .argv
            .contains(&"--rsync-model-artifacts".to_string())
    );
    assert!(focused_runtime.argv.contains(&"--keep-remote".to_string()));
    assert!(focused_runtime.argv.contains(&"--child-logs".to_string()));
    assert!(
        focused_runtime
            .argv
            .contains(&"--stage-async-prefill-forward".to_string())
    );
    assert!(
        focused_runtime
            .argv
            .contains(&"--allow-uneven-stage-ranges".to_string())
    );
    assert!(
        focused_runtime
            .shell
            .contains("--remote-root-map host-b=/data/skippy,host-c=/scratch/skippy")
    );
    assert_arg_value(
        schema_smoke,
        "--remote-root-map",
        "host-b=/data/skippy,host-c=/scratch/skippy",
    );
    assert!(
        !schema_smoke
            .argv
            .contains(&"--rsync-model-artifacts".to_string())
    );
    assert!(!schema_smoke.argv.contains(&"--keep-remote".to_string()));
    assert!(!schema_smoke.argv.contains(&"--child-logs".to_string()));
}

#[test]
fn evidence_commands_prepare_default_bench_corpora() {
    let commands = test_evidence_commands();
    let long = command_by_id(&commands, "prepare-corpus-long");
    let coding_loop = command_by_id(&commands, "prepare-corpus-coding-loop");

    assert_eq!(long.argv, ["just", "bench-corpus", "long"]);
    assert_eq!(coding_loop.argv, ["just", "bench-corpus", "coding-loop"]);
}

#[test]
fn custom_corpora_do_not_get_default_prep_commands() {
    let commands = evidence_commands(EvidenceCommandInputs {
        token_corpus: Path::new("/tmp/custom-token-corpus.jsonl"),
        chat_corpus: Path::new("/tmp/custom-chat-corpus.jsonl"),
        long_context_corpus: Path::new("/tmp/custom-long-context-corpus.jsonl"),
        ..test_evidence_command_inputs()
    });

    assert!(
        commands
            .iter()
            .all(|command| !command.id.starts_with("prepare-corpus-"))
    );
}

#[test]
fn evidence_script_contains_ordered_plan_commands() {
    let dir = unique_test_dir("evidence-script");
    fs::create_dir_all(&dir).expect("create fixture dir");
    let script_path = dir.join("run-evidence.sh");
    let report = EvidencePlanReport {
        schema_version: 1,
        kind: "skippy_quant_pack_evidence_plan".to_string(),
        runbook_cwd: "/tmp/repo".to_string(),
        run_dir: "/tmp/run".to_string(),
        candidate: "middle-compressed".to_string(),
        model_id: "org/repo:middle-compressed".to_string(),
        package: "/tmp/run/package".to_string(),
        quantized_model: "/tmp/run/model.gguf".to_string(),
        evidence_dir: "/tmp/run/evidence".to_string(),
        hosts: vec!["host-a".to_string(), "host-b".to_string()],
        stage_count: 2,
        splits: "20".to_string(),
        split_source: "cli_override".to_string(),
        layer_end: 40,
        ctx_size: 8192,
        n_gpu_layers: -1,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        max_tokens: 512,
        long_context_corpus: DEFAULT_LONG_CONTEXT_CORPUS.to_string(),
        include_local_split_evidence: false,
        local_split_prompt: DEFAULT_LOCAL_SPLIT_PROMPT.to_string(),
        skippy_bench_bin: "/tmp/bin/skippy-bench".to_string(),
        skippy_model_package_bin: "/tmp/bin/skippy-model-package".to_string(),
        agent_tool_call_script: "/tmp/bin/qa-agent-tool-call-reliability.py".to_string(),
        kv_tool_loop_script: "/tmp/bin/qa-kv-tool-loop-stability.py".to_string(),
        activation_wire_dtype: "q8".to_string(),
        focused_runtime: FocusedRuntimeEvidenceArgs::default(),
        warnings: vec!["runtime hosts must be SSH reachable".to_string()],
        commands: evidence_commands(EvidenceCommandInputs {
            skippy_bench_bin: Path::new("/tmp/bin/skippy-bench"),
            skippy_model_package_bin: Path::new("/tmp/bin/skippy-model-package"),
            agent_tool_call_script: Path::new("/tmp/bin/qa-agent-tool-call-reliability.py"),
            kv_tool_loop_script: Path::new("/tmp/bin/qa-kv-tool-loop-stability.py"),
            ..test_evidence_command_inputs()
        }),
    };

    let plan_path = dir.join("evidence-plan.json");
    write_single_evidence_script(
        &script_path,
        Some(&plan_path),
        Path::new("/tmp/bin/skippy-model-package"),
        &report,
    )
    .expect("write script");

    let script = fs::read_to_string(&script_path).expect("read script");
    assert!(script.starts_with("#!/usr/bin/env bash\nset -euo pipefail"));
    assert!(script.contains("\ncd /tmp/repo\n"));
    assert!(script.contains("evidence-status"));
    assert!(script.contains("--fail-on-warning"));
    assert!(script.contains("/tmp/bin/skippy-model-package"));
    assert!(script.contains(&plan_path.display().to_string()));
    assert!(script.contains("warning: runtime hosts must be SSH reachable"));
    assert!(script.contains("== skippy quant-pack evidence: middle-compressed =="));
    assert!(script.contains("# prepare-corpus-long"));
    assert!(script.contains("just bench-corpus long"));
    assert!(
        script.contains("skip evidence command: prepare-corpus-long (outputs already complete)")
    );
    assert!(script.contains("--command-complete prepare-corpus-long"));
    assert!(script.contains("--candidate middle-compressed"));
    assert!(script.contains("test -e target/bench-corpora/long/corpus.jsonl"));
    assert!(script.contains("# focused-runtime-schema-smoke"));
    assert!(
        script.contains("/tmp/bin/skippy-bench focused-runtime --stage-model /tmp/run/package")
    );
    assert!(script.contains("--schema-smoke"));
    assert!(script.contains("test -e /tmp/run/evidence/focused-runtime-schema-smoke.json"));
    assert!(script.contains("--command-complete focused-runtime-schema-smoke"));
    assert!(script.contains(
        "missing expected evidence output: /tmp/run/evidence/focused-runtime-schema-smoke.json"
    ));
    assert!(script.contains("/tmp/bin/skippy-bench focused-runtime"));
    assert!(script.contains(
        "missing expected evidence output: /tmp/run/evidence/focused-runtime-report.json"
    ));
    assert!(script.contains("/tmp/bin/qa-agent-tool-call-reliability.py"));
    assert!(script.contains("/tmp/bin/qa-kv-tool-loop-stability.py"));
    assert!(script.contains("/tmp/bin/skippy-model-package quant-pack certify"));
    assert!(script.contains("/tmp/bin/skippy-model-package quant-pack rank"));
    assert!(
        script.contains(
            "missing expected evidence output: /tmp/run/evidence/rank-after-evidence.json"
        )
    );
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn all_evidence_script_hoists_shared_corpus_prep() {
    let dir = unique_test_dir("all-evidence-script");
    fs::create_dir_all(&dir).expect("create fixture dir");
    let script_path = dir.join("run-evidence.sh");
    let report = EvidencePlanAllReport {
        schema_version: 1,
        kind: "skippy_quant_pack_evidence_plan_all".to_string(),
        runbook_cwd: "/tmp/repo".to_string(),
        build_all: "/tmp/sweep/quant-pack-build-all.json".to_string(),
        evidence_root: "/tmp/sweep/evidence".to_string(),
        selection_source: "top_ranked:2".to_string(),
        final_rank: test_final_rank_command(),
        candidates: vec![
            test_evidence_report("middle-compressed"),
            test_evidence_report("ffn-compressed-attention-protected"),
        ],
    };

    let plan_path = dir.join("evidence-plan-all.json");
    write_all_evidence_script(
        &script_path,
        Some(&plan_path),
        Path::new("skippy-model-package"),
        &report,
    )
    .expect("write all script");

    let script = fs::read_to_string(&script_path).expect("read all script");
    assert!(script.contains("\ncd /tmp/repo\n"));
    assert!(script.contains("evidence-status"));
    assert!(script.contains("--fail-on-warning"));
    assert!(script.contains("== shared setup =="));
    assert_eq!(script.matches("just bench-corpus long\n").count(), 1);
    assert_eq!(script.matches("just bench-corpus coding-loop\n").count(), 1);
    assert_eq!(
        script.matches("just bench-corpus long-context\n").count(),
        1
    );
    assert!(
        script.contains("== skippy quant-pack evidence: ffn-compressed-attention-protected ==")
    );
    assert!(script.contains("== skippy quant-pack sweep rank =="));
    assert!(script.contains("rank-after-evidence-all"));
    assert!(script.contains("skippy-model-package quant-pack rank /tmp/run/middle-compressed /tmp/run/ffn-compressed-attention-protected"));
    assert!(script.contains(
        "missing expected evidence output: /tmp/sweep/evidence/rank-after-evidence.json"
    ));
    assert_eq!(script.matches("# prepare-evidence-dir").count(), 2);
    assert!(script.contains("--command-complete prepare-corpus-long"));
    assert!(script.contains("--candidate middle-compressed"));
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn evidence_script_only_skips_when_all_declared_outputs_exist() {
    let outputs = vec![
        "/tmp/run/evidence/focused-runtime-lab-preflight.txt".to_string(),
        "/tmp/run/evidence/focused-runtime-lab-preflight.ok".to_string(),
    ];

    let test = all_outputs_exist_test(&outputs);

    assert_eq!(
        test,
        "test -e /tmp/run/evidence/focused-runtime-lab-preflight.txt && test -e /tmp/run/evidence/focused-runtime-lab-preflight.ok"
    );
}

#[test]
fn evidence_script_skip_guard_checks_semantic_command_status() {
    let command = EvidenceCommand {
        id: "token-lengths".to_string(),
        description: "token lengths".to_string(),
        evidence_type: Some("skippy-bench-token-lengths".to_string()),
        argv: Vec::new(),
        shell: "token-lengths".to_string(),
        outputs: vec![
            "/tmp/run/evidence/prompt-lengths.tsv".to_string(),
            "/tmp/run/evidence/prompt-lengths-summary.json".to_string(),
        ],
    };
    let check = SemanticSkipCheck {
        plan_path: Path::new("/tmp/run/evidence-plan.json"),
        skippy_model_package_bin: Path::new("/tmp/bin/skippy-model-package"),
        candidate: Some("middle-compressed"),
    };

    let test = command_completion_test(&command, Some(&check));

    assert!(test.contains("test -e /tmp/run/evidence/prompt-lengths.tsv"));
    assert!(test.contains("/tmp/bin/skippy-model-package quant-pack evidence-status"));
    assert!(test.contains("--command-complete token-lengths"));
    assert!(test.contains("--candidate middle-compressed"));
}

#[test]
fn evidence_plan_warns_when_preflight_hosts_differ_from_runtime_hosts() {
    let focused_runtime = FocusedRuntimeEvidenceArgs {
        lab_preflight_hosts: Some("192.168.0.2,192.168.0.4".to_string()),
        ..FocusedRuntimeEvidenceArgs::default()
    };
    let hosts = vec!["lab-stage-0".to_string(), "lab-stage-1".to_string()];

    let warnings = evidence_plan_warnings(&hosts, &focused_runtime);

    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("lab_preflight_hosts"));
    assert!(warnings[0].contains("skippy-bench uses --hosts as SSH targets"));
}

#[test]
fn evidence_plan_warns_when_only_preflight_has_ssh_opts() {
    let focused_runtime = FocusedRuntimeEvidenceArgs {
        lab_preflight_ssh_opts: Some("-p 2222".to_string()),
        ..FocusedRuntimeEvidenceArgs::default()
    };
    let hosts = vec!["host-a".to_string(), "host-b".to_string()];

    let warnings = evidence_plan_warnings(&hosts, &focused_runtime);

    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("focused-runtime ssh_opts are not"));
}

fn test_evidence_report(candidate: &str) -> EvidencePlanReport {
    EvidencePlanReport {
        schema_version: 1,
        kind: "skippy_quant_pack_evidence_plan".to_string(),
        runbook_cwd: "/tmp/repo".to_string(),
        run_dir: format!("/tmp/run/{candidate}"),
        candidate: candidate.to_string(),
        model_id: format!("org/repo:{candidate}"),
        package: format!("/tmp/run/{candidate}/package"),
        quantized_model: format!("/tmp/run/{candidate}/model.gguf"),
        evidence_dir: format!("/tmp/run/{candidate}/evidence"),
        hosts: vec!["host-a".to_string(), "host-b".to_string()],
        stage_count: 2,
        splits: "20".to_string(),
        split_source: "cli_override".to_string(),
        layer_end: 40,
        ctx_size: 8192,
        n_gpu_layers: -1,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        max_tokens: 512,
        long_context_corpus: DEFAULT_LONG_CONTEXT_CORPUS.to_string(),
        include_local_split_evidence: false,
        local_split_prompt: DEFAULT_LOCAL_SPLIT_PROMPT.to_string(),
        skippy_bench_bin: "skippy-bench".to_string(),
        skippy_model_package_bin: "skippy-model-package".to_string(),
        agent_tool_call_script: DEFAULT_AGENT_TOOL_CALL_SCRIPT.to_string(),
        kv_tool_loop_script: DEFAULT_KV_TOOL_LOOP_SCRIPT.to_string(),
        activation_wire_dtype: "q8".to_string(),
        focused_runtime: FocusedRuntimeEvidenceArgs::default(),
        warnings: Vec::new(),
        commands: evidence_commands(EvidenceCommandInputs {
            run: Path::new("/tmp/run"),
            model_id: "org/repo:middle-compressed",
            package: Path::new("/tmp/run/package"),
            quantized_model: Path::new("/tmp/run/model.gguf"),
            evidence_dir: Path::new("/tmp/run/evidence"),
            ..test_evidence_command_inputs()
        }),
    }
}

fn test_final_rank_command() -> EvidenceCommand {
    command(
        "rank-after-evidence-all",
        "Rerank all selected candidates after skippy-bench and certification evidence are present.",
        Some("skippy-quant-pack-rank"),
        vec![
            "skippy-model-package".to_string(),
            "quant-pack".to_string(),
            "rank".to_string(),
            "/tmp/run/middle-compressed".to_string(),
            "/tmp/run/ffn-compressed-attention-protected".to_string(),
            "--activation-wire-dtype".to_string(),
            "q8".to_string(),
            "--ctx-size".to_string(),
            "8192".to_string(),
            "--n-gpu-layers".to_string(),
            "-1".to_string(),
            "--cache-type-k".to_string(),
            "f16".to_string(),
            "--cache-type-v".to_string(),
            "f16".to_string(),
            "--out".to_string(),
            "/tmp/sweep/evidence/rank-after-evidence.json".to_string(),
        ],
        vec!["/tmp/sweep/evidence/rank-after-evidence.json".to_string()],
    )
}

fn test_evidence_commands() -> Vec<EvidenceCommand> {
    evidence_commands(test_evidence_command_inputs())
}

fn test_evidence_command_inputs() -> EvidenceCommandInputs<'static> {
    EvidenceCommandInputs {
        run: Path::new("/tmp/run"),
        model_id: "org/repo:middle-compressed",
        package: Path::new("/tmp/run/package"),
        quantized_model: Path::new("/tmp/run/model.gguf"),
        evidence_dir: Path::new("/tmp/run/evidence"),
        hosts: "host-a,host-b",
        splits: "20",
        layer_end: 40,
        base_url: "http://127.0.0.1:9337/v1",
        token_corpus: Path::new("target/bench-corpora/long/corpus.jsonl"),
        chat_corpus: Path::new("target/bench-corpora/coding-loop/corpus.jsonl"),
        long_context_corpus: Path::new("target/bench-corpora/long-context/corpus.jsonl"),
        ctx_size: 8192,
        n_gpu_layers: -1,
        cache_type_k: "f16",
        cache_type_v: "f16",
        max_tokens: 512,
        include_local_split_evidence: false,
        local_split_prompt: DEFAULT_LOCAL_SPLIT_PROMPT,
        skippy_bench_bin: Path::new("skippy-bench"),
        skippy_model_package_bin: Path::new("skippy-model-package"),
        agent_tool_call_script: Path::new(DEFAULT_AGENT_TOOL_CALL_SCRIPT),
        kv_tool_loop_script: Path::new(DEFAULT_KV_TOOL_LOOP_SCRIPT),
        activation_wire_dtype: "q8",
        attempts: 2,
        focused_runtime: default_focused_runtime_options(),
        allow_uneven_stage_ranges: false,
    }
}

fn default_focused_runtime_options() -> &'static FocusedRuntimeEvidenceArgs {
    Box::leak(Box::new(FocusedRuntimeEvidenceArgs::default()))
}

fn command_by_id<'a>(commands: &'a [EvidenceCommand], id: &str) -> &'a EvidenceCommand {
    commands
        .iter()
        .find(|command| command.id == id)
        .expect("command should exist")
}

fn assert_arg_value(command: &EvidenceCommand, name: &str, expected: &str) {
    if let Some(index) = command.argv.iter().position(|arg| arg == name) {
        assert_eq!(
            command.argv.get(index + 1).map(String::as_str),
            Some(expected)
        );
        return;
    }
    let joined = format!("{name}={expected}");
    assert!(
        command.argv.contains(&joined),
        "{name} missing from {}",
        command.id
    );
}

fn write_candidate_fixture(run_dir: &Path, candidate: &str, model_id: &str) {
    write_candidate_fixture_with_shape(run_dir, candidate, model_id, 40, 2, None);
}

fn write_candidate_fixture_with_plan(
    run_dir: &Path,
    candidate: &str,
    model_id: &str,
    layer_count: u32,
    stages: usize,
) {
    write_candidate_fixture_with_shape(
        run_dir,
        candidate,
        model_id,
        layer_count,
        stages,
        Some("quant-plan.json"),
    );
    fs::write(
        run_dir.join("quant-plan.json"),
        format!(
            r#"{{
  "schema_version": 1,
  "candidates": [
    {{
      "id": "{candidate}",
      "stage_hints": [
        {{"stage_index": 0, "layer_start": 0, "layer_end": 16}},
        {{"stage_index": 1, "layer_start": 16, "layer_end": 32}},
        {{"stage_index": 2, "layer_start": 32, "layer_end": 47}},
        {{"stage_index": 3, "layer_start": 47, "layer_end": {layer_count}}}
      ]
    }}
  ]
}}"#
        ),
    )
    .expect("write quant plan");
}

fn write_candidate_fixture_with_shape(
    run_dir: &Path,
    candidate: &str,
    model_id: &str,
    layer_count: u32,
    stages: usize,
    plan: Option<&str>,
) {
    let package = run_dir.join("package");
    fs::create_dir_all(&package).expect("create package dir");
    let plan_field = plan
        .map(|plan| {
            format!(
                r#",
  "plan": "{plan}""#
            )
        })
        .unwrap_or_default();
    fs::write(
        run_dir.join("quant-pack-build.json"),
        format!(
            r#"{{
  "schema_version": 1,
  "kind": "skippy_quant_pack_build",
  "stages": {stages},
  "candidate": "{candidate}",
  "package": "package",
  "quantized_model": "model.gguf"{plan_field}
}}"#
        ),
    )
    .expect("write candidate manifest");
    fs::write(
        package.join("model-package.json"),
        format!(
            r#"{{
  "schema_version": 1,
  "model_id": "{model_id}",
  "layer_count": {layer_count}
}}"#
        ),
    )
    .expect("write package manifest");
    fs::write(run_dir.join("model.gguf"), b"gguf").expect("write fake model");
}

fn test_build_manifest_input(candidate: &str) -> BuildManifestInput {
    BuildManifestInput {
        candidate: candidate.to_string(),
        stages: 2,
        plan: None,
        package: "package".to_string(),
        quantized_model: "model.gguf".to_string(),
    }
}

fn write_build_all_fixture<'a>(dir: &Path, run_dirs: impl IntoIterator<Item = &'a PathBuf>) {
    let candidates = run_dirs
            .into_iter()
            .map(|run_dir| {
                let candidate = run_dir.file_name().unwrap().to_string_lossy();
                format!(
                    r#"{{"candidate": "{candidate}", "run_dir": "{}", "manifest": "{}/quant-pack-build.json"}}"#,
                    run_dir.display(),
                    run_dir.display()
                )
            })
            .collect::<Vec<_>>()
            .join(",\n    ");
    fs::write(
        dir.join("quant-pack-build-all.json"),
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
    {candidates}
  ],
  "rank": "{}/quant-pack-rank.json"
}}"#,
            dir.display()
        ),
    )
    .expect("write build-all manifest");
}

fn unique_test_dir(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "skippy-evidence-plan-{name}-{}-{nanos}",
        std::process::id()
    ))
}
