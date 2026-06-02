use super::*;
use serde_json::json;

fn node(id: &str, capacity_bytes: u64) -> PlanNode {
    PlanNode {
        id: id.to_string(),
        role: "Worker".to_string(),
        hostname: None,
        raw_capacity_bytes: capacity_bytes,
        reserve_bytes: 0,
        capacity_bytes,
        unified_memory: Some(false),
        bandwidth_gbps: None,
        rtt_ms: None,
    }
}

#[test]
fn parse_size_label_bytes_supports_decimal_and_binary_units() {
    assert_eq!(parse_size_label_bytes("20GB"), Some(20_000_000_000));
    assert_eq!(parse_size_label_bytes("1.5 GB"), Some(1_500_000_000));
    assert_eq!(parse_size_label_bytes("2MiB"), Some(2 * 1024 * 1024));
    assert_eq!(parse_size_label_bytes("bad"), None);
}

#[test]
fn plan_scope_prefers_single_node_fit() {
    let plan = plan_scope(
        "test",
        &[node("big", 40_000_000_000), node("small", 10_000_000_000)],
        20_000_000_000,
        true,
        true,
    );

    assert_eq!(plan.fit, FitVerdict::ComfortableLocal);
    assert_eq!(plan.split_node_count, Some(1));
    assert_eq!(plan.suggested_nodes[0].id, "big");
}

#[test]
fn plan_scope_suggests_minimal_split_set() {
    let plan = plan_scope(
        "test",
        &[
            node("a", 20_000_000_000),
            node("b", 18_000_000_000),
            node("c", 10_000_000_000),
        ],
        35_000_000_000,
        true,
        false,
    );

    assert_eq!(plan.fit, FitVerdict::SplitCandidate);
    assert_eq!(plan.split_node_count, Some(2));
    assert_eq!(
        plan.suggested_nodes
            .iter()
            .map(|node| node.id.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
}

#[test]
fn plan_scope_rejects_aggregate_without_split_package() {
    let plan = plan_scope(
        "test",
        &[node("a", 20_000_000_000), node("b", 20_000_000_000)],
        35_000_000_000,
        false,
        false,
    );

    assert_eq!(plan.fit, FitVerdict::InsufficientCapacity);
    assert_eq!(plan.split_node_count, None);
}

#[test]
fn plan_scope_reports_unknown_when_capacity_is_missing() {
    let plan = plan_scope("test", &[node("unknown", 0)], 35_000_000_000, true, false);

    assert_eq!(plan.fit, FitVerdict::UnknownCapacity);
    assert_eq!(plan.reason, "eligible_nodes_missing_capacity");
}

#[test]
fn nodes_needed_for_capacity_rounds_up() {
    assert_eq!(nodes_needed_for_capacity(41, 20), Some(3));
    assert_eq!(nodes_needed_for_capacity(40, 20), Some(2));
    assert_eq!(nodes_needed_for_capacity(40, 0), None);
}

#[test]
fn recommendation_score_prefers_smaller_models_when_capacity_is_unknown() {
    let small = RecommendationRow {
        display_name: "small".to_string(),
        model_ref: "small".to_string(),
        fit: FitVerdict::UnknownCapacity,
        reason: "eligible_nodes_missing_capacity",
        model_bytes: 5_000_000_000,
        weight_budget_bytes: 5_500_000_000,
        kv_cache_budget_bytes: 1_000_000_000,
        required_bytes: 6_000_000_000,
        context_tokens: 32_768,
        moe: false,
        active_parameter_billions: Some(8.0),
        best_single_node_capacity_bytes: None,
        aggregate_capacity_bytes: 0,
        split_capable: false,
        layer_package_repo: None,
        disk_fits_full_model: Some(true),
        disk_full_model_required_bytes: 5_250_000_000,
        disk_free_bytes: Some(10_000_000_000),
        plan_command: "mesh-llm models plan small".to_string(),
    };
    let large = RecommendationRow {
        model_bytes: 400_000_000_000,
        required_bytes: 440_000_000_000,
        display_name: "large".to_string(),
        model_ref: "large".to_string(),
        plan_command: "mesh-llm models plan large".to_string(),
        ..small.clone()
    };

    assert!(recommendation_score(&small) > recommendation_score(&large));
}

#[test]
fn build_capacity_plan_keeps_size_only_notes_visible() {
    let hint = CatalogModelHint {
        catalog_id: "Example-Q4_K_M".to_string(),
        display_name: "Example".to_string(),
        model_ref: "example/repo:Q4_K_M".to_string(),
        source_repo: "example/repo".to_string(),
        source_file: "model.gguf".to_string(),
        model_bytes: 10_000_000_000,
        split_capable: true,
        layer_package_repo: Some("meshllm/example-layers".to_string()),
        layer_count: Some(32),
        package_total_bytes: Some(12_000_000_000),
        is_moe: false,
        moe_summary: None,
        active_parameter_billions: Some(OrderedF64(8.0)),
    };
    let plan = build_capacity_plan(
        "Example",
        hint,
        vec![node("local", 20_000_000_000)],
        None,
        32_768,
    );

    assert!(plan.mesh.is_none());
    assert!(
        plan.notes
            .iter()
            .any(|note| note.contains("does not download"))
    );
    assert!(plan.model.kv_cache_budget_bytes > 0);
}

#[test]
fn status_capacity_nodes_excludes_clients_later_by_role() {
    let status = StatusPayload {
        node_id: "local".to_string(),
        is_client: true,
        my_hostname: None,
        my_is_soc: Some(true),
        my_vram_gb: 64.0,
        gpus: vec![StatusGpu {
            mem_bandwidth_gbps: Some(800.0),
            unified_memory: Some(true),
        }],
        peers: vec![StatusPeer {
            id: "peer".to_string(),
            role: "Worker".to_string(),
            hostname: Some("box".to_string()),
            is_soc: Some(false),
            vram_gb: 24.0,
            rtt_ms: Some(3),
            gpus: Vec::new(),
        }],
    };

    let nodes = status_capacity_nodes(status);
    assert_eq!(nodes.len(), 2);
    assert_eq!(nodes[0].role, "Client");
    assert_eq!(nodes[1].hostname.as_deref(), Some("box"));
}

#[test]
fn json_shape_includes_model_and_local_scope() {
    let hint = CatalogModelHint {
        catalog_id: "Example-Q4_K_M".to_string(),
        display_name: "Example".to_string(),
        model_ref: "example/repo:Q4_K_M".to_string(),
        source_repo: "example/repo".to_string(),
        source_file: "model.gguf".to_string(),
        model_bytes: 10_000_000_000,
        split_capable: false,
        layer_package_repo: None,
        layer_count: None,
        package_total_bytes: None,
        is_moe: false,
        moe_summary: None,
        active_parameter_billions: Some(OrderedF64(8.0)),
    };
    let plan = build_capacity_plan(
        "Example",
        hint,
        vec![node("local", 20_000_000_000)],
        None,
        32_768,
    );
    let value = json!(plan);

    assert_eq!(value["model"]["requested"], "Example");
    assert_eq!(value["local"]["label"], "local");
    assert_eq!(value["model"]["context_tokens"], 32_768);
    assert_eq!(
        value["disk"]["full_model_required_bytes"],
        10_500_000_000u64
    );
}

#[test]
fn local_memory_reserve_is_larger_for_unified_memory() {
    let raw = 64_000_000_000;

    assert_eq!(local_memory_reserve_bytes(raw, Some(true)), 12_800_000_000);
    assert_eq!(local_memory_reserve_bytes(raw, Some(false)), 6_400_000_000);
}

#[test]
fn moe_uses_active_parameter_hint_for_kv_but_full_weight_bytes_for_fit() {
    let hint = CatalogModelHint {
        catalog_id: "Qwen3-235B-A22B-Q4".to_string(),
        display_name: "Qwen3 235B A22B".to_string(),
        model_ref: "example/repo:Q4".to_string(),
        source_repo: "example/repo".to_string(),
        source_file: "model.gguf".to_string(),
        model_bytes: 134_000_000_000,
        split_capable: true,
        layer_package_repo: Some("meshllm/example-layers".to_string()),
        layer_count: Some(94),
        package_total_bytes: Some(134_000_000_000),
        is_moe: true,
        moe_summary: Some("235B/22B".to_string()),
        active_parameter_billions: Some(OrderedF64(22.0)),
    };
    let plan = build_capacity_plan(
        "Qwen3-235B-A22B-Q4",
        hint,
        vec![node("big", 180_000_000_000)],
        None,
        32_768,
    );

    assert!(plan.model.moe);
    assert_eq!(plan.model.weight_budget_bytes, 147_400_000_000);
    assert_eq!(plan.model.kv_cache_budget_bytes, 8_589_934_592);
    assert_eq!(plan.local.fit, FitVerdict::ComfortableLocal);
}

#[test]
fn parameter_hints_parse_moe_forms() {
    assert_eq!(parameter_after_slash_billions("480B/35B"), Some(35.0));
    assert_eq!(
        parameter_after_a_billions("Qwen3-235B-A22B-GGUF"),
        Some(22.0)
    );
    assert_eq!(total_parameter_billions("Qwen3-8B-Q4_K_M"), Some(8.0));
}
