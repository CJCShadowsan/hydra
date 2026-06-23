use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use skippy_runtime::package::{PackageGenerationInfo, PackageSpeculativeDecodingInfo};
use skippy_topology::infer_family_capability;

use super::support::{pick_owned, pick_string, pick_string_owned};
use super::types::ResolvedSpeculativeConfig;
use crate::plugin::{BoolOrAuto, SpeculativeConfig};

const SHARD_PIPELINE_MODE: &str = "shard-pipeline";
const SHARD_PIPELINE_DEFAULT_DEPTH: u32 = 6;

pub(super) fn resolve_speculative_config(
    model_config: Option<&SpeculativeConfig>,
    global_config: Option<&SpeculativeConfig>,
    model_id: &str,
    model_path: &Path,
    package_generation: Option<&PackageGenerationInfo>,
) -> Result<ResolvedSpeculativeConfig> {
    let strategy = pick_string_owned(
        model_config.and_then(|config| config.strategy.as_deref()),
        global_config.and_then(|config| config.strategy.as_deref()),
        Some("auto"),
    );
    let native_mtp_enabled = match strategy.as_str() {
        "auto" => package_generation_supports_default_native_mtp(package_generation),
        "native-mtp-n1" => {
            if package_generation.is_some_and(|generation| {
                !generation
                    .speculative_decoding
                    .as_ref()
                    .is_some_and(speculative_supports_native_mtp_n1)
            }) {
                bail!(
                    "skippy speculative.strategy = \"native-mtp-n1\" requires package generation metadata advertising native-mtp-n1"
                );
            }
            true
        }
        "disabled" => false,
        _ => bail!("skippy speculative.strategy must be auto, disabled, or native-mtp-n1"),
    };
    let mode = pick_string_owned(
        model_config.and_then(|config| config.mode.as_deref()),
        global_config.and_then(|config| config.mode.as_deref()),
        Some("auto"),
    );
    if mode == "ngram" {
        bail!("skippy speculative.mode = \"ngram\" is not supported by the embedded runtime");
    }
    if pick_owned(
        model_config.and_then(|config| config.draft_hf_repo.clone()),
        global_config.and_then(|config| config.draft_hf_repo.clone()),
    )
    .is_some()
        || pick_owned(
            model_config.and_then(|config| config.draft_hf_file.clone()),
            global_config.and_then(|config| config.draft_hf_file.clone()),
        )
        .is_some()
    {
        bail!(
            "skippy speculative Hugging Face draft sources are not supported by the embedded runtime"
        );
    }
    if pick_owned(
        model_config.and_then(|config| config.draft_device.clone()),
        global_config.and_then(|config| config.draft_device.clone()),
    )
    .is_some()
        || pick_owned(
            model_config.and_then(|config| config.draft_threads),
            global_config.and_then(|config| config.draft_threads),
        )
        .is_some()
        || pick_owned(
            model_config.and_then(|config| config.draft_cache_type_k.clone()),
            global_config.and_then(|config| config.draft_cache_type_k.clone()),
        )
        .is_some()
        || pick_owned(
            model_config.and_then(|config| config.draft_cache_type_v.clone()),
            global_config.and_then(|config| config.draft_cache_type_v.clone()),
        )
        .is_some()
    {
        bail!("skippy explicit draft runtime overrides are not supported by the embedded runtime");
    }
    let draft_min_tokens = pick_owned(
        model_config.and_then(|config| config.draft_min_tokens),
        global_config.and_then(|config| config.draft_min_tokens),
    )
    .unwrap_or(0);
    if draft_min_tokens > 0 {
        bail!("skippy speculative.draft_min_tokens is not supported by the embedded runtime");
    }
    let draft_acceptance_threshold = pick_owned(
        model_config.and_then(|config| config.draft_acceptance_threshold),
        global_config.and_then(|config| config.draft_acceptance_threshold),
    )
    .unwrap_or(0.0);
    if draft_acceptance_threshold > 0.0 {
        bail!(
            "skippy speculative.draft_acceptance_threshold is not supported by the embedded runtime"
        );
    }
    let draft_split_probability = pick_owned(
        model_config.and_then(|config| config.draft_split_probability),
        global_config.and_then(|config| config.draft_split_probability),
    )
    .unwrap_or(0.0);
    if draft_split_probability > 0.0 {
        bail!(
            "skippy speculative.draft_split_probability is not supported by the embedded runtime"
        );
    }
    if let Some(BoolOrAuto::Bool(true)) = pick_owned(
        model_config.and_then(|config| config.spec_default.as_ref()),
        global_config.and_then(|config| config.spec_default.as_ref()),
    ) {
        bail!("skippy speculative.spec_default = true is not supported by the embedded runtime");
    }

    let mut mode = mode;
    let mut draft_model_path = pick_owned(
        model_config.and_then(|config| config.draft_model_path.clone()),
        global_config.and_then(|config| config.draft_model_path.clone()),
    )
    .map(PathBuf::from);
    let draft_max_tokens = super::support::pick_value(
        model_config.and_then(|config| config.draft_max_tokens),
        global_config.and_then(|config| config.draft_max_tokens),
        0,
    );
    let pipelined_depth = resolve_pipelined_depth(model_config, global_config, &mode)?;
    if mode == SHARD_PIPELINE_MODE {
        mode = "draft".to_string();
    }
    let draft_n_gpu_layers = pick_owned(
        model_config.and_then(|config| config.draft_gpu_layers),
        global_config.and_then(|config| config.draft_gpu_layers),
    );
    let pairing_fault = normalize_pairing_fault(pick_string(
        model_config.and_then(|config| config.pairing_fault.as_deref()),
        global_config.and_then(|config| config.pairing_fault.as_deref()),
        Some("warn_disable"),
    ));
    let explicit = mode != "auto"
        || draft_model_path.is_some()
        || draft_max_tokens > 0
        || pipelined_depth > 1
        || draft_n_gpu_layers.is_some();
    if mode == "disabled"
        && (draft_model_path.is_some() || draft_max_tokens > 0 || pipelined_depth > 1)
    {
        bail!(
            "skippy speculative draft controls cannot be set when speculative.mode = \"disabled\""
        );
    }
    if uses_draft_model(&mode) || (mode == "auto" && draft_model_path.is_some()) {
        if draft_model_path.is_none() {
            bail!("skippy speculative draft mode requires an explicit draft_model_path");
        }
        if draft_max_tokens == 0 {
            bail!("skippy speculative draft mode requires draft_max_tokens > 0");
        }
        if mode == "auto" {
            mode = "draft".to_string();
        }
        let draft_path = draft_model_path.as_ref().expect("checked above");
        if let Some(reason) = incompatible_draft_pair_reason(model_id, model_path, draft_path) {
            match pairing_fault.as_str() {
                "warn_disable" => {
                    mode = "disabled".to_string();
                    draft_model_path = None;
                }
                "fail_open" => {}
                "fail_closed" => bail!("skippy incompatible speculative draft pairing: {reason}"),
                _ => unreachable!(),
            }
        }
    } else {
        mode = "disabled".to_string();
        draft_model_path = None;
    }
    Ok(ResolvedSpeculativeConfig {
        strategy,
        native_mtp_enabled,
        mode,
        draft_model_path,
        pairing_fault,
        draft_max_tokens,
        pipelined_depth,
        explicit,
        draft_n_gpu_layers,
    })
}

fn resolve_pipelined_depth(
    model_config: Option<&SpeculativeConfig>,
    global_config: Option<&SpeculativeConfig>,
    mode: &str,
) -> Result<u32> {
    let configured = pick_owned(
        model_config.and_then(|config| config.pipelined_depth),
        global_config.and_then(|config| config.pipelined_depth),
    );
    if mode == SHARD_PIPELINE_MODE {
        return match configured {
            Some(1) => bail!(
                "skippy speculative.pipelined_depth must be greater than 1 when speculative.mode = \"shard-pipeline\""
            ),
            Some(depth) => Ok(depth),
            None => Ok(SHARD_PIPELINE_DEFAULT_DEPTH),
        };
    }
    Ok(configured.unwrap_or(1))
}

fn uses_draft_model(mode: &str) -> bool {
    matches!(mode, "draft" | "tree")
}

fn package_generation_supports_default_native_mtp(
    generation: Option<&PackageGenerationInfo>,
) -> bool {
    generation
        .and_then(|generation| generation.speculative_decoding.as_ref())
        .is_some_and(|speculative| {
            speculative
                .strategies
                .get(&speculative.default)
                .is_some_and(|strategy| {
                    strategy.strategy_type == "native-mtp"
                        && strategy.prediction_depth == Some(1)
                        && !strategy.layer_indices.is_empty()
                })
        })
}

fn speculative_supports_native_mtp_n1(speculative: &PackageSpeculativeDecodingInfo) -> bool {
    speculative
        .strategies
        .get("native-mtp-n1")
        .is_some_and(|strategy| {
            strategy.strategy_type == "native-mtp"
                && strategy.prediction_depth == Some(1)
                && !strategy.layer_indices.is_empty()
        })
}

fn normalize_pairing_fault(value: &str) -> String {
    value.replace('-', "_")
}

fn incompatible_draft_pair_reason(
    model_id: &str,
    model_path: &Path,
    draft_model_path: &Path,
) -> Option<String> {
    let target_family = infer_family_capability(model_id, 0, 0)
        .map(|capability| capability.family_id.to_string())
        .or_else(|| infer_family_from_path_string(model_path));
    let draft_family = infer_family_from_path_string(draft_model_path);
    match (target_family, draft_family) {
        (Some(target_family), Some(draft_family)) if target_family != draft_family => Some(
            format!("target family {target_family} does not match draft family {draft_family}"),
        ),
        _ => None,
    }
}

fn infer_family_from_path_string(path: &Path) -> Option<String> {
    family_path_candidates(path)
        .into_iter()
        .find_map(|candidate| {
            infer_family_capability(&candidate, 0, 0)
                .map(|capability| capability.family_id.to_string())
        })
}

fn family_path_candidates(path: &Path) -> Vec<String> {
    let mut candidates = Vec::new();
    if let Some(file_name) = path.file_name().and_then(|name| name.to_str()) {
        candidates.push(file_name.to_string());
    }
    for component in path.components().rev() {
        let Some(raw) = component.as_os_str().to_str() else {
            continue;
        };
        if skip_family_path_component(raw) {
            continue;
        }
        let candidate = raw.replace("--", "/");
        if !candidates.iter().any(|existing| existing == &candidate) {
            candidates.push(candidate);
        }
    }
    candidates
}

fn skip_family_path_component(component: &str) -> bool {
    let lower = component.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "." | ".." | "hub" | "models" | "snapshots" | "blobs" | "refs"
    ) || looks_like_cache_hash(&lower)
}

fn looks_like_cache_hash(component: &str) -> bool {
    component.len() >= 16 && component.bytes().all(|byte| byte.is_ascii_hexdigit())
}
