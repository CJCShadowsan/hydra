use super::test_support::*;
use super::*;
use crate::inference::skippy::SkippyTelemetryOptions;
use skippy_protocol::LoadMode;
use skippy_runtime::package::{
    PackageGenerationInfo, PackageGenerationPolicyInfo, PackageGenerationThresholdsInfo,
    PackageSpeculativeDecodingInfo, PackageSpeculativeStrategyInfo, PackageWindowPolicyInfo,
};
use std::collections::BTreeMap;

fn native_mtp_generation() -> PackageGenerationInfo {
    let mut strategies = BTreeMap::new();
    strategies.insert(
        "native-mtp-n1".to_string(),
        PackageSpeculativeStrategyInfo {
            strategy_type: "native-mtp".to_string(),
            prediction_depth: Some(1),
            layer_indices: vec![46],
            window_policy: Some(PackageWindowPolicyInfo {
                default: "fixed".to_string(),
                initial_window: 1,
                min_window: 1,
                max_window: 1,
            }),
        },
    );

    PackageGenerationInfo {
        policy: None,
        thresholds: None,
        speculative_decoding: Some(PackageSpeculativeDecodingInfo {
            default: "native-mtp-n1".to_string(),
            strategies,
        }),
    }
}

#[test]
fn speculative_strategy_auto_without_package_generation_disables_native_mtp() {
    let mesh_config = parse_config("");
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
    })
    .expect("default speculative strategy should resolve");

    assert_eq!(resolved.speculative.strategy, "auto");
    assert!(!resolved.speculative.native_mtp_enabled);
    let load_options = resolved
        .to_model_load_options(SkippyTelemetryOptions::off())
        .expect("model load options should build");
    assert!(!load_options.native_mtp_enabled);
    let stage = resolved
        .to_stage_config(Some(fake_package_identity(24)), LoadMode::LayerPackage)
        .expect("stage config should build");
    assert_eq!(stage.model_id, "Qwen/Qwen3-0.6B:Q4_K_M");
    let openai = resolved
        .to_embedded_openai_args(4096, true)
        .expect("openai args should build");
    assert!(!openai.native_mtp_enabled);
}

#[test]
fn speculative_strategy_auto_uses_package_native_mtp_default() {
    let mesh_config = parse_config("");
    let model_file = temp_model_file();
    let generation = native_mtp_generation();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/GLM-4.7-Flash-MTP-GGUF",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: Some(&generation),
    })
    .expect("package native MTP default should resolve");

    assert_eq!(resolved.speculative.strategy, "auto");
    assert!(resolved.speculative.native_mtp_enabled);
    let load_options = resolved
        .to_model_load_options(SkippyTelemetryOptions::off())
        .expect("model load options should build");
    assert!(load_options.native_mtp_enabled);
    let stage = resolved
        .to_stage_config(Some(fake_package_identity(24)), LoadMode::LayerPackage)
        .expect("stage config should build");
    assert_eq!(stage.model_id, "meshllm/GLM-4.7-Flash-MTP-GGUF");
    let openai = resolved
        .to_embedded_openai_args(4096, true)
        .expect("openai args should build");
    assert!(openai.native_mtp_enabled);
}

#[test]
fn speculative_strategy_native_mtp_rejects_package_without_native_mtp_metadata() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "native-mtp-n1"
"#,
    );
    let model_file = temp_model_file();
    let generation = PackageGenerationInfo {
        policy: None,
        thresholds: None,
        speculative_decoding: None,
    };

    let error = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/package-without-mtp",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: Some(&generation),
    })
    .unwrap_err()
    .to_string();

    assert!(error.contains("requires package generation metadata advertising native-mtp-n1"));
}

fn glm_dsa_generation() -> PackageGenerationInfo {
    PackageGenerationInfo {
        policy: Some(PackageGenerationPolicyInfo {
            profile: "glm-dsa-v1".to_string(),
            decode: "compact-flash".to_string(),
            short_prefill: "dense".to_string(),
            long_prefill: "sparse-chunked".to_string(),
            verify: "auto".to_string(),
            indexshare: Some("required".to_string()),
            experimental: Some(
                skippy_runtime::package::PackageGenerationExperimentalPolicyInfo {
                    selected_row_flash: Some("off".to_string()),
                },
            ),
        }),
        thresholds: Some(PackageGenerationThresholdsInfo {
            short_prefill_max_tokens: Some(2048),
            compact_flash_min_kv: Some(256),
            dense_mask_max_bytes: Some(268435456),
        }),
        speculative_decoding: None,
    }
}

#[test]
fn resolver_exposes_package_generation_policy() {
    let mesh_config = parse_config("");
    let model_file = temp_model_file();
    let generation = glm_dsa_generation();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "meshllm/GLM-5.2-Q2_K-MTP-Q8-layers",
        model_path: model_file.path(),
        model_bytes: 260 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: Some(&generation),
    })
    .expect("GLM-DSA package policy should resolve");

    let policy = resolved
        .generation_policy
        .as_ref()
        .expect("package policy should be carried into resolved config");
    assert_eq!(policy.profile, "glm-dsa-v1");
    assert_eq!(policy.decode, "compact-flash");
    assert_eq!(policy.short_prefill, "dense");
    assert_eq!(policy.long_prefill, "sparse-chunked");
    assert_eq!(policy.verify, "auto");
    assert_eq!(policy.indexshare.as_deref(), Some("required"));
    assert_eq!(policy.selected_row_flash.as_deref(), Some("off"));
    assert_eq!(policy.thresholds.short_prefill_max_tokens, Some(2048));
    assert_eq!(policy.thresholds.compact_flash_min_kv, Some(256));
    assert_eq!(policy.thresholds.dense_mask_max_bytes, Some(268435456));

    let runtime_options = resolved
        .to_embedded_runtime_options(&SkippyTelemetryOptions::off(), None, LoadMode::RuntimeSlice)
        .expect("GLM-DSA policy should convert into runtime options");
    let runtime_policy = runtime_options
        .glm_dsa_policy
        .expect("GLM-DSA package policy should reach runtime launch options");
    assert_eq!(
        runtime_policy.profile,
        skippy_runtime::GlmDsaPolicyConfig::glm_dsa_v1().profile
    );
    assert!(runtime_policy.direct_sparse_attn);
    assert!(!runtime_policy.direct_sparse_prefill);
    assert_eq!(runtime_policy.short_prefill_max_tokens, Some(2048));
    assert_eq!(runtime_policy.compact_flash_min_kv, Some(256));
    assert_eq!(runtime_policy.dense_sparse_mask_max_bytes, Some(268435456));
}

#[test]
fn speculative_strategy_disabled_reaches_stage_and_openai_args() {
    let mesh_config = parse_config(
        r#"
[defaults.speculative]
strategy = "disabled"
"#,
    );
    let model_file = temp_model_file();

    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "Qwen/Qwen3-0.6B:Q4_K_M",
        model_path: model_file.path(),
        model_bytes: 4 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
    })
    .expect("disabled speculative strategy should resolve");

    assert_eq!(resolved.speculative.strategy, "disabled");
    assert!(!resolved.speculative.native_mtp_enabled);
    let load_options = resolved
        .to_model_load_options(SkippyTelemetryOptions::off())
        .expect("model load options should build");
    assert!(!load_options.native_mtp_enabled);
    let stage = resolved
        .to_stage_config(Some(fake_package_identity(24)), LoadMode::LayerPackage)
        .expect("stage config should build");
    assert_eq!(stage.model_id, "Qwen/Qwen3-0.6B:Q4_K_M");
    let openai = resolved
        .to_embedded_openai_args(4096, true)
        .expect("openai args should build");
    assert!(!openai.native_mtp_enabled);
}
