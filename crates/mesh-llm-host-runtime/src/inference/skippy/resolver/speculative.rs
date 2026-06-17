use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use skippy_topology::infer_family_capability;

use super::support::{pick_owned, pick_string, pick_string_owned};
use super::types::ResolvedSpeculativeConfig;
use crate::plugin::{BoolOrAuto, SpeculativeConfig};

pub(super) fn resolve_speculative_config(
    model_config: Option<&SpeculativeConfig>,
    global_config: Option<&SpeculativeConfig>,
    model_id: &str,
    model_path: &Path,
) -> Result<ResolvedSpeculativeConfig> {
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
    let mut spd = SpdSpeculativeFields::from_configs(model_config, global_config);
    let draft_max_tokens = super::support::pick_value(
        model_config.and_then(|config| config.draft_max_tokens),
        global_config.and_then(|config| config.draft_max_tokens),
        0,
    );
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
        || draft_n_gpu_layers.is_some()
        || spd.is_configured();
    let has_spd_config = spd.is_configured();
    if mode == "disabled" && draft_model_path.is_some() {
        bail!("skippy speculative draft source cannot be set when speculative.mode = \"disabled\"");
    }
    if mode == "disabled" && has_spd_config {
        bail!("skippy SPD source cannot be set when speculative.mode = \"disabled\"");
    }
    if (mode == "draft" && has_spd_config)
        || (draft_model_path.is_some() && (has_spd_config || mode == "spd"))
    {
        bail!("skippy draft-model and SPD speculative sources cannot both be configured");
    }
    if mode == "spd" || (mode == "auto" && has_spd_config) {
        if spd.manifest_path.is_none() || spd.fixture_path.is_none() {
            bail!("skippy SPD mode requires spd_manifest_path and spd_fixture_path");
        }
        if spd.max_tokens == 0 {
            bail!("skippy SPD mode requires spd_max_tokens > 0");
        }
        if spd.top_k == 0 {
            bail!("skippy SPD mode requires spd_top_k > 0");
        }
        if spd.optimistic_min_logit_margin.is_some() && spd.top_k < 2 {
            bail!("skippy SPD optimistic margin gating requires spd_top_k >= 2");
        }
        mode = "spd".to_string();
        draft_model_path = None;
    } else {
        spd.clear_artifacts();
    }
    if mode == "draft" || (mode == "auto" && draft_model_path.is_some()) {
        if draft_model_path.is_none() {
            bail!("skippy speculative draft mode requires an explicit draft_model_path");
        }
        if draft_max_tokens == 0 {
            bail!("skippy speculative draft mode requires draft_max_tokens > 0");
        }
        mode = "draft".to_string();
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
        if mode != "spd" {
            mode = "disabled".to_string();
        }
        draft_model_path = None;
    }
    Ok(ResolvedSpeculativeConfig {
        mode,
        draft_model_path,
        spd_manifest_path: spd.manifest_path,
        spd_fixture_path: spd.fixture_path,
        spd_model_path: spd.model_path,
        pairing_fault,
        draft_max_tokens,
        spd_max_tokens: spd.max_tokens,
        explicit,
        draft_n_gpu_layers,
        spd_n_gpu_layers: spd.n_gpu_layers,
        spd_top_k: spd.top_k,
        spd_replay_fallback: spd.replay_fallback,
        spd_optimistic_decode: spd.optimistic_decode,
        spd_optimistic_min_logit_margin: spd.optimistic_min_logit_margin,
    })
}

struct SpdSpeculativeFields {
    manifest_path: Option<PathBuf>,
    fixture_path: Option<PathBuf>,
    model_path: Option<PathBuf>,
    max_tokens: u32,
    n_gpu_layers: Option<i32>,
    top_k: usize,
    replay_fallback: bool,
    optimistic_decode: bool,
    optimistic_min_logit_margin: Option<f32>,
}

impl SpdSpeculativeFields {
    fn from_configs(
        model_config: Option<&SpeculativeConfig>,
        global_config: Option<&SpeculativeConfig>,
    ) -> Self {
        Self {
            manifest_path: pick_owned(
                model_config.and_then(|config| config.spd_manifest_path.clone()),
                global_config.and_then(|config| config.spd_manifest_path.clone()),
            )
            .map(PathBuf::from),
            fixture_path: pick_owned(
                model_config.and_then(|config| config.spd_fixture_path.clone()),
                global_config.and_then(|config| config.spd_fixture_path.clone()),
            )
            .map(PathBuf::from),
            model_path: pick_owned(
                model_config.and_then(|config| config.spd_model_path.clone()),
                global_config.and_then(|config| config.spd_model_path.clone()),
            )
            .map(PathBuf::from),
            max_tokens: super::support::pick_value(
                model_config.and_then(|config| config.spd_max_tokens),
                global_config.and_then(|config| config.spd_max_tokens),
                0,
            ),
            n_gpu_layers: pick_owned(
                model_config.and_then(|config| config.spd_gpu_layers),
                global_config.and_then(|config| config.spd_gpu_layers),
            ),
            top_k: super::support::pick_value(
                model_config.and_then(|config| config.spd_top_k),
                global_config.and_then(|config| config.spd_top_k),
                1,
            ),
            replay_fallback: super::support::pick_value(
                model_config.and_then(|config| config.spd_replay_fallback),
                global_config.and_then(|config| config.spd_replay_fallback),
                false,
            ),
            optimistic_decode: super::support::pick_value(
                model_config.and_then(|config| config.spd_optimistic_decode),
                global_config.and_then(|config| config.spd_optimistic_decode),
                false,
            ),
            optimistic_min_logit_margin: pick_owned(
                model_config.and_then(|config| config.spd_optimistic_min_logit_margin),
                global_config.and_then(|config| config.spd_optimistic_min_logit_margin),
            )
            .map(|value| value as f32),
        }
    }

    fn is_configured(&self) -> bool {
        self.manifest_path.is_some()
            || self.fixture_path.is_some()
            || self.model_path.is_some()
            || self.max_tokens > 0
            || self.n_gpu_layers.is_some()
            || self.top_k != 1
            || self.replay_fallback
            || self.optimistic_decode
            || self.optimistic_min_logit_margin.is_some()
    }

    fn clear_artifacts(&mut self) {
        self.manifest_path = None;
        self.fixture_path = None;
        self.model_path = None;
    }
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
    infer_family_capability(&path.display().to_string(), 0, 0)
        .map(|capability| capability.family_id.to_string())
}
