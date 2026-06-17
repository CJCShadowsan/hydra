use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};
use skippy_runtime::spd::SpdHeadManifest;
use skippy_topology::infer_family_capability;

use super::support::{pick_owned, pick_string, pick_string_owned};
use super::types::ResolvedSpeculativeConfig;
use crate::plugin::{BoolOrAuto, SpeculativeConfig};

const SPD_BUNDLE_MANIFEST_FILE: &str = "skippy-spd-head.json";
const SPD_BUNDLE_FIXTURE_FILE: &str = "spd-parity-fixture.safetensors";

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
        spd.resolve_bundle_ref()?;
        if spd.manifest_path.is_none() || spd.fixture_path.is_none() {
            bail!(
                "skippy SPD mode requires spd_bundle_ref or both spd_manifest_path and spd_fixture_path"
            );
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
        if spd.rolling_executor && !spd.optimistic_decode {
            bail!("skippy SPD rolling executor requires spd_optimistic_decode = true");
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
        spd_rolling_executor: spd.rolling_executor,
        spd_optimistic_min_logit_margin: spd.optimistic_min_logit_margin,
    })
}

struct SpdSpeculativeFields {
    bundle_ref: Option<String>,
    manifest_path: Option<PathBuf>,
    fixture_path: Option<PathBuf>,
    model_path: Option<PathBuf>,
    max_tokens: u32,
    n_gpu_layers: Option<i32>,
    top_k: usize,
    replay_fallback: bool,
    optimistic_decode: bool,
    rolling_executor: bool,
    optimistic_min_logit_margin: Option<f32>,
}

impl SpdSpeculativeFields {
    fn from_configs(
        model_config: Option<&SpeculativeConfig>,
        global_config: Option<&SpeculativeConfig>,
    ) -> Self {
        Self {
            bundle_ref: pick_owned(
                model_config.and_then(|config| config.spd_bundle_ref.clone()),
                global_config.and_then(|config| config.spd_bundle_ref.clone()),
            ),
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
            rolling_executor: super::support::pick_value(
                model_config.and_then(|config| config.spd_rolling_executor),
                global_config.and_then(|config| config.spd_rolling_executor),
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
        self.bundle_ref.is_some()
            || self.manifest_path.is_some()
            || self.fixture_path.is_some()
            || self.model_path.is_some()
            || self.max_tokens > 0
            || self.n_gpu_layers.is_some()
            || self.top_k != 1
            || self.replay_fallback
            || self.optimistic_decode
            || self.rolling_executor
            || self.optimistic_min_logit_margin.is_some()
    }

    fn clear_artifacts(&mut self) {
        self.bundle_ref = None;
        self.manifest_path = None;
        self.fixture_path = None;
        self.model_path = None;
    }

    fn resolve_bundle_ref(&mut self) -> Result<()> {
        let Some(bundle_ref) = self.bundle_ref.as_deref() else {
            return Ok(());
        };
        let bundle = resolve_spd_bundle_ref(bundle_ref)
            .with_context(|| format!("resolve SPD sidecar bundle {bundle_ref}"))?;
        if self.manifest_path.is_none() {
            self.manifest_path = Some(bundle.manifest_path);
        }
        if self.fixture_path.is_none() {
            self.fixture_path = Some(bundle.fixture_path);
        }
        Ok(())
    }
}

struct ResolvedSpdBundle {
    manifest_path: PathBuf,
    fixture_path: PathBuf,
}

fn resolve_spd_bundle_ref(value: &str) -> Result<ResolvedSpdBundle> {
    if let Some(rest) = value.strip_prefix("hf://") {
        return resolve_hf_spd_bundle_ref(rest);
    }
    let path = PathBuf::from(value);
    let manifest_path = if path.is_dir() {
        path.join(SPD_BUNDLE_MANIFEST_FILE)
    } else {
        path
    };
    let fixture_path = manifest_path
        .parent()
        .context("SPD sidecar bundle manifest has no parent directory")?
        .join(SPD_BUNDLE_FIXTURE_FILE);
    resolve_local_spd_bundle_paths(manifest_path, fixture_path)
}

fn resolve_hf_spd_bundle_ref(value: &str) -> Result<ResolvedSpdBundle> {
    let (repo, revision) = parse_hf_spd_bundle_ref(value)?;
    crate::models::run_hf_sync(move || download_hf_spd_bundle_to_local_sync(&repo, &revision))
}

fn parse_hf_spd_bundle_ref(value: &str) -> Result<(String, String)> {
    let (repo, revision) = if let Some((repo, revision)) = value.split_once('@') {
        (repo, Some(revision))
    } else if let Some(index) = value.rfind(':') {
        (&value[..index], Some(&value[index + 1..]))
    } else {
        (value, None)
    };
    if repo.split('/').count() != 2 || repo.contains(':') || repo.contains('@') {
        bail!("HF SPD sidecar bundle repo id must look like namespace/repo");
    }
    let revision = revision.unwrap_or("main");
    let _ = safe_spd_bundle_file_path(revision)
        .with_context(|| format!("invalid HF SPD sidecar revision: {revision}"))?;
    Ok((repo.to_string(), revision.to_string()))
}

fn download_hf_spd_bundle_to_local_sync(repo: &str, revision: &str) -> Result<ResolvedSpdBundle> {
    let api = crate::models::build_hf_api(false)?;
    let (owner, name) = repo.split_once('/').context("invalid HF repo format")?;
    let model_api = api.model(owner, name);
    let manifest_path = download_hf_spd_bundle_file(&model_api, revision, SPD_BUNDLE_MANIFEST_FILE)
        .context("download SPD sidecar manifest")?;
    let manifest = SpdHeadManifest::from_path(&manifest_path)?;
    let serving_path = manifest.serving_checkpoint_path(&manifest_path)?;
    let serving_file = serving_path
        .strip_prefix(
            manifest_path
                .parent()
                .context("SPD sidecar manifest has no parent directory")?,
        )
        .ok()
        .and_then(|path| path.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| {
            manifest
                .serving_checkpoint
                .as_ref()
                .map(|checkpoint| checkpoint.path.clone())
                .unwrap_or_else(|| "spd-head.safetensors".to_string())
        });
    let serving_file = safe_spd_bundle_file_path(&serving_file)?;
    download_hf_spd_bundle_file(&model_api, revision, &serving_file.to_string_lossy())
        .context("download SPD serving checkpoint")?;
    let fixture_path = download_hf_spd_bundle_file(&model_api, revision, SPD_BUNDLE_FIXTURE_FILE)
        .context("download SPD parity fixture")?;
    resolve_local_spd_bundle_paths(manifest_path, fixture_path)
}

fn download_hf_spd_bundle_file(
    model_api: &hf_hub::HFRepositorySync<hf_hub::RepoTypeModel>,
    revision: &str,
    file_name: &str,
) -> Result<PathBuf> {
    model_api
        .download_file()
        .filename(file_name.to_string())
        .revision(revision.to_string())
        .send()
        .with_context(|| format!("download SPD sidecar bundle file: {file_name}"))
}

fn resolve_local_spd_bundle_paths(
    manifest_path: PathBuf,
    fixture_path: PathBuf,
) -> Result<ResolvedSpdBundle> {
    let manifest = SpdHeadManifest::from_path(&manifest_path)?;
    ensure_existing_file(&manifest_path, "SPD sidecar manifest")?;
    let serving_path = manifest.serving_checkpoint_path(&manifest_path)?;
    ensure_existing_file(&serving_path, "SPD serving checkpoint")?;
    ensure_existing_file(&fixture_path, "SPD parity fixture")?;
    Ok(ResolvedSpdBundle {
        manifest_path,
        fixture_path,
    })
}

fn ensure_existing_file(path: &Path, label: &str) -> Result<()> {
    if !path.is_file() {
        bail!("{label} does not exist: {}", path.display());
    }
    Ok(())
}

fn safe_spd_bundle_file_path(path: &str) -> Result<PathBuf> {
    anyhow::ensure!(!path.is_empty(), "SPD sidecar bundle file path is empty");
    let path = Path::new(path);
    let mut components = path.components();
    let Some(first) = components.next() else {
        bail!("SPD sidecar bundle file path is empty");
    };
    anyhow::ensure!(
        matches!(first, Component::Normal(_))
            && components.all(|component| matches!(component, Component::Normal(_))),
        "SPD sidecar bundle file path must be a safe relative path: {}",
        path.display()
    );
    Ok(path.to_path_buf())
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
