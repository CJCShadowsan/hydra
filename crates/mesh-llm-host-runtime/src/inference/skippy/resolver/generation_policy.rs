use skippy_runtime::package::PackageGenerationInfo;

use super::types::{ResolvedGenerationPolicyConfig, ResolvedGenerationThresholdsConfig};

pub(super) fn resolve_generation_policy_config(
    package_generation: Option<&PackageGenerationInfo>,
) -> Option<ResolvedGenerationPolicyConfig> {
    let generation = package_generation?;
    let policy = generation.policy.as_ref()?;
    let thresholds = generation
        .thresholds
        .as_ref()
        .map(|thresholds| ResolvedGenerationThresholdsConfig {
            short_prefill_max_tokens: thresholds.short_prefill_max_tokens,
            direct_sparse_decode_max_top_k: thresholds.direct_sparse_decode_max_top_k,
            compact_flash_min_kv: thresholds.compact_flash_min_kv,
            dense_mask_max_bytes: thresholds.dense_mask_max_bytes,
        })
        .unwrap_or_default();

    Some(ResolvedGenerationPolicyConfig {
        profile: policy.profile.clone(),
        decode: policy.decode.clone(),
        short_prefill: policy.short_prefill.clone(),
        long_prefill: policy.long_prefill.clone(),
        verify: policy.verify.clone(),
        indexshare: policy.indexshare.clone(),
        selected_row_flash: policy
            .experimental
            .as_ref()
            .and_then(|experimental| experimental.selected_row_flash.clone()),
        thresholds,
    })
}
