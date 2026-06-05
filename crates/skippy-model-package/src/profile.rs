use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use skippy_ffi::TensorRole;
use skippy_runtime::{
    FlashAttentionType, ModelInfo, RuntimeConfig, RuntimeLoadMode, SamplingConfig, StageModel,
    TensorInfo, parse_cache_type,
};

#[derive(Debug, clap::Args)]
pub(crate) struct ProfileArgs {
    pub(crate) package: PathBuf,
    #[arg(long, default_value_t = 1)]
    pub(crate) stages: usize,
    #[arg(long, value_enum, default_value_t = ProfilePhase::Decode)]
    pub(crate) phase: ProfilePhase,
    #[arg(long, default_value_t = 8192)]
    pub(crate) existing_kv_tokens: u32,
    #[arg(long, default_value_t = 1)]
    pub(crate) generated_tokens: u32,
    #[arg(long, default_value_t = 1)]
    pub(crate) batch_size: u32,
    #[arg(long, default_value = "f16")]
    pub(crate) kv_type: String,
    #[arg(long)]
    pub(crate) backend: Option<String>,
    #[arg(long)]
    pub(crate) device: Option<String>,
    #[arg(long, default_value_t = 20)]
    pub(crate) samples: u32,
    #[arg(long, default_value_t = 3)]
    pub(crate) warmup_samples: u32,
    #[arg(long, value_enum, default_value_t = TimingSourceKind::Static)]
    pub(crate) timing_source: TimingSourceKind,
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProfilePhase {
    Decode,
    Prefill,
    SuffixPrefill,
    CacheReplay,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum TimingSourceKind {
    Static,
    LocalStage,
}

#[derive(Debug, Serialize)]
pub(crate) struct ProfileReport {
    pub(crate) schema_version: u32,
    pub(crate) kind: String,
    pub(crate) input_kind: ProfileInputKind,
    pub(crate) package_path: String,
    pub(crate) model_id: String,
    pub(crate) layer_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) activation_width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) manifest_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) source_sha256: Option<String>,
    pub(crate) runtime: RuntimeProfile,
    pub(crate) request_shape: RequestShape,
    pub(crate) measurement: MeasurementConfig,
    pub(crate) measurement_status: MeasurementStatus,
    pub(crate) summary: ProfileSummary,
    pub(crate) shared: SharedProfile,
    pub(crate) layers: Vec<LayerProfile>,
    pub(crate) stages: Vec<StageProfile>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProfileInputKind {
    LayerPackage,
    DirectGguf,
}

#[derive(Debug, Serialize)]
pub(crate) struct RuntimeProfile {
    pub(crate) skippy_model_package_version: String,
    pub(crate) skippy_abi_version: String,
    pub(crate) package_skippy_abi_version: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestShape {
    pub(crate) phase: ProfilePhase,
    pub(crate) existing_kv_tokens: u32,
    pub(crate) generated_tokens: u32,
    pub(crate) batch_size: u32,
    pub(crate) kv_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) device: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct MeasurementConfig {
    pub(crate) source: TimingSourceKind,
    pub(crate) warmup_samples: u32,
    pub(crate) samples: u32,
}

#[derive(Debug, Serialize)]
pub(crate) struct MeasurementStatus {
    pub(crate) status: String,
    pub(crate) reason: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct ProfileSummary {
    pub(crate) stage_count: usize,
    pub(crate) layer_artifact_bytes: u64,
    pub(crate) shared_artifact_bytes: u64,
    pub(crate) package_artifact_bytes: u64,
    pub(crate) measured_layer_count: usize,
    pub(crate) estimated_tokens_per_second: Option<f64>,
}

#[derive(Debug, Serialize)]
pub(crate) struct SharedProfile {
    pub(crate) metadata: ArtifactProfile,
    pub(crate) embeddings: ArtifactProfile,
    pub(crate) output: ArtifactProfile,
}

#[derive(Debug, Serialize)]
pub(crate) struct ArtifactProfile {
    pub(crate) path: String,
    pub(crate) tensor_count: usize,
    pub(crate) tensor_bytes: u64,
    pub(crate) artifact_bytes: u64,
    pub(crate) sha256: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct LayerProfile {
    pub(crate) layer_index: u32,
    pub(crate) artifact: ArtifactProfile,
    pub(crate) timing: TimingProfile,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TimingProfile {
    pub(crate) status: String,
    pub(crate) mean_ms: Option<f64>,
    pub(crate) p50_ms: Option<f64>,
    pub(crate) p95_ms: Option<f64>,
    pub(crate) samples: u32,
}

#[derive(Debug, Serialize)]
pub(crate) struct StageProfile {
    pub(crate) stage_index: usize,
    pub(crate) layer_start: u32,
    pub(crate) layer_end: u32,
    pub(crate) includes_embeddings: bool,
    pub(crate) includes_output: bool,
    pub(crate) part_count: usize,
    pub(crate) artifact_bytes: u64,
    pub(crate) timing: TimingProfile,
    pub(crate) parts: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct PackageManifest {
    model_id: String,
    layer_count: u32,
    #[serde(default)]
    activation_width: Option<u32>,
    shared: PackageShared,
    layers: Vec<PackageLayer>,
    skippy_abi_version: String,
}

#[derive(Debug, Deserialize)]
struct PackageShared {
    metadata: PackageArtifact,
    embeddings: PackageArtifact,
    output: PackageArtifact,
}

#[derive(Debug, Clone, Deserialize)]
struct PackageArtifact {
    path: String,
    tensor_count: usize,
    tensor_bytes: u64,
    artifact_bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct PackageLayer {
    layer_index: u32,
    path: String,
    tensor_count: usize,
    tensor_bytes: u64,
    artifact_bytes: u64,
    sha256: String,
}

trait ProfileTimingSource {
    fn profile(&self, input: &ProfileTimingInput<'_>) -> Result<ProfileTimingReport>;
}

struct StaticTimingSource;
struct LocalStageTimingSource;

struct ProfileTimingInput<'a> {
    package: &'a Path,
    input_kind: ProfileInputKind,
    stage_count: usize,
    request_shape: &'a RequestShape,
    measurement: &'a MeasurementConfig,
}

#[derive(Debug)]
struct ProfileTimingReport {
    measurement_status: MeasurementStatus,
    layer_timings: BTreeMap<u32, TimingProfile>,
    stage_timings: BTreeMap<usize, TimingProfile>,
    estimated_tokens_per_second: Option<f64>,
}

impl ProfileTimingSource for StaticTimingSource {
    fn profile(&self, input: &ProfileTimingInput<'_>) -> Result<ProfileTimingReport> {
        Ok(ProfileTimingReport {
            measurement_status: MeasurementStatus {
                status: "not_measured".to_string(),
                reason: format!(
                    "timing source {:?} does not execute the package; native hooks will fill this {:?} report shape later for {} warmup samples and {} measured samples from {}",
                    input.measurement.source,
                    input.request_shape.phase,
                    input.measurement.warmup_samples,
                    input.measurement.samples,
                    input.package.display()
                ),
            },
            layer_timings: BTreeMap::new(),
            stage_timings: BTreeMap::new(),
            estimated_tokens_per_second: None,
        })
    }
}

impl ProfileTimingSource for LocalStageTimingSource {
    fn profile(&self, input: &ProfileTimingInput<'_>) -> Result<ProfileTimingReport> {
        validate_local_stage_input(input)?;
        profile_local_stage_decode(input)
    }
}

pub(crate) fn run_profile(args: ProfileArgs) -> Result<()> {
    let report = profile_input(&args)?;
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(out) = &args.out {
        fs::write(out, format!("{json}\n"))
            .with_context(|| format!("write profile report {}", out.display()))?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn profile_input(args: &ProfileArgs) -> Result<ProfileReport> {
    if is_layer_package_dir(&args.package) {
        profile_package(args)
    } else {
        profile_direct_gguf(args)
    }
}

fn profile_package(args: &ProfileArgs) -> Result<ProfileReport> {
    let manifest_path = args.package.join("model-package.json");
    let manifest_contents = fs::read(&manifest_path)
        .with_context(|| format!("read package manifest {}", manifest_path.display()))?;
    let manifest_sha256 = sha256_bytes(&manifest_contents);
    let manifest = serde_json::from_slice::<PackageManifest>(&manifest_contents)
        .with_context(|| format!("parse package manifest {}", manifest_path.display()))?;
    validate_stage_count(args.stages, manifest.layer_count)?;

    let request_shape = request_shape(args);
    let measurement = measurement_config(args);
    let timing_report = timing_source(args.timing_source).profile(&ProfileTimingInput {
        package: &args.package,
        input_kind: ProfileInputKind::LayerPackage,
        stage_count: args.stages,
        request_shape: &request_shape,
        measurement: &measurement,
    })?;
    let layers = layer_profiles(&manifest, &timing_report);
    let shared = shared_profile(&manifest.shared);
    let stages = stage_profiles(&manifest, args.stages, &timing_report);
    let summary = profile_summary(args.stages, &shared, &layers, &timing_report);
    let runtime = runtime_profile(&manifest);

    Ok(ProfileReport {
        schema_version: 1,
        kind: "skippy_agent_quant_profile".to_string(),
        input_kind: ProfileInputKind::LayerPackage,
        package_path: args.package.display().to_string(),
        model_id: manifest.model_id,
        layer_count: manifest.layer_count,
        activation_width: manifest.activation_width,
        manifest_sha256: Some(manifest_sha256),
        source_sha256: None,
        runtime,
        request_shape,
        measurement,
        measurement_status: timing_report.measurement_status,
        summary,
        shared,
        layers,
        stages,
    })
}

fn profile_direct_gguf(args: &ProfileArgs) -> Result<ProfileReport> {
    let metadata = fs::metadata(&args.package)
        .with_context(|| format!("read GGUF metadata {}", args.package.display()))?;
    if !metadata.is_file() {
        bail!(
            "profile input must be a layer package directory or GGUF file: {}",
            args.package.display()
        );
    }
    let model = ModelInfo::open(&args.package)
        .with_context(|| format!("open GGUF model {}", args.package.display()))?;
    let tensors = model
        .tensors()
        .with_context(|| format!("inspect GGUF tensors {}", args.package.display()))?;
    let layer_count = direct_layer_count(&tensors)?;
    validate_stage_count(args.stages, layer_count)?;
    let source_sha256 = file_sha256(&args.package)?;

    let request_shape = request_shape(args);
    let measurement = measurement_config(args);
    let timing_report = timing_source(args.timing_source).profile(&ProfileTimingInput {
        package: &args.package,
        input_kind: ProfileInputKind::DirectGguf,
        stage_count: args.stages,
        request_shape: &request_shape,
        measurement: &measurement,
    })?;
    let layers = direct_layer_profiles(&tensors, &args.package, &source_sha256, &timing_report);
    let shared = direct_shared_profile(&tensors, &args.package, &source_sha256);
    let stages = direct_stage_profiles(&tensors, layer_count, args.stages, &timing_report);
    let summary = profile_summary(args.stages, &shared, &layers, &timing_report);

    Ok(ProfileReport {
        schema_version: 1,
        kind: "skippy_agent_quant_profile".to_string(),
        input_kind: ProfileInputKind::DirectGguf,
        package_path: args.package.display().to_string(),
        model_id: args.package.display().to_string(),
        layer_count,
        activation_width: None,
        manifest_sha256: None,
        source_sha256: Some(source_sha256),
        runtime: RuntimeProfile {
            skippy_model_package_version: env!("CARGO_PKG_VERSION").to_string(),
            skippy_abi_version: format!(
                "{}.{}.{}",
                skippy_ffi::ABI_VERSION_MAJOR,
                skippy_ffi::ABI_VERSION_MINOR,
                skippy_ffi::ABI_VERSION_PATCH
            ),
            package_skippy_abi_version: "synthetic-direct-gguf".to_string(),
        },
        request_shape,
        measurement,
        measurement_status: timing_report.measurement_status,
        summary,
        shared,
        layers,
        stages,
    })
}

fn is_layer_package_dir(path: &Path) -> bool {
    path.is_dir() && path.join("model-package.json").is_file()
}

fn validate_stage_count(stages: usize, layer_count: u32) -> Result<()> {
    if stages == 0 {
        bail!("--stages must be greater than zero");
    }
    if stages as u32 > layer_count {
        bail!("--stages {stages} exceeds package layer_count {layer_count}");
    }
    Ok(())
}

fn direct_layer_count(tensors: &[TensorInfo]) -> Result<u32> {
    tensors
        .iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .map(|max_layer| max_layer + 1)
        .context("GGUF model has no layer tensors")
}

fn layer_profiles(
    manifest: &PackageManifest,
    timing_report: &ProfileTimingReport,
) -> Vec<LayerProfile> {
    let mut layers = manifest.layers.iter().collect::<Vec<_>>();
    layers.sort_by_key(|layer| layer.layer_index);
    layers
        .into_iter()
        .map(|layer| LayerProfile {
            layer_index: layer.layer_index,
            artifact: ArtifactProfile {
                path: layer.path.clone(),
                tensor_count: layer.tensor_count,
                tensor_bytes: layer.tensor_bytes,
                artifact_bytes: layer.artifact_bytes,
                sha256: layer.sha256.clone(),
            },
            timing: timing_report
                .layer_timings
                .get(&layer.layer_index)
                .cloned()
                .unwrap_or_else(unmeasured_timing),
        })
        .collect()
}

fn direct_layer_profiles(
    tensors: &[TensorInfo],
    path: &Path,
    source_sha256: &str,
    timing_report: &ProfileTimingReport,
) -> Vec<LayerProfile> {
    let mut layer_stats = BTreeMap::<u32, TensorStats>::new();
    for tensor in tensors
        .iter()
        .filter(|tensor| tensor.role == TensorRole::Layer)
    {
        let Some(layer_index) = tensor.layer_index else {
            continue;
        };
        layer_stats
            .entry(layer_index)
            .or_default()
            .add_tensor(tensor.byte_size);
    }
    layer_stats
        .into_iter()
        .map(|(layer_index, stats)| LayerProfile {
            layer_index,
            artifact: ArtifactProfile {
                path: format!("{}#layer:{layer_index}", path.display()),
                tensor_count: stats.tensor_count,
                tensor_bytes: stats.tensor_bytes,
                artifact_bytes: stats.tensor_bytes,
                sha256: source_sha256.to_string(),
            },
            timing: timing_report
                .layer_timings
                .get(&layer_index)
                .cloned()
                .unwrap_or_else(unmeasured_timing),
        })
        .collect()
}

fn shared_profile(shared: &PackageShared) -> SharedProfile {
    SharedProfile {
        metadata: artifact_profile(&shared.metadata),
        embeddings: artifact_profile(&shared.embeddings),
        output: artifact_profile(&shared.output),
    }
}

fn direct_shared_profile(
    tensors: &[TensorInfo],
    path: &Path,
    source_sha256: &str,
) -> SharedProfile {
    SharedProfile {
        metadata: direct_artifact_profile(
            path,
            source_sha256,
            "shared:metadata",
            direct_tensor_stats(tensors, |tensor| {
                matches!(
                    tensor.role,
                    TensorRole::Metadata | TensorRole::Tokenizer | TensorRole::Unknown
                )
            }),
        ),
        embeddings: direct_artifact_profile(
            path,
            source_sha256,
            "shared:embeddings",
            direct_tensor_stats(tensors, |tensor| tensor.role == TensorRole::Embedding),
        ),
        output: direct_artifact_profile(
            path,
            source_sha256,
            "shared:output",
            direct_tensor_stats(tensors, |tensor| {
                matches!(tensor.role, TensorRole::FinalNorm | TensorRole::Output)
            }),
        ),
    }
}

fn direct_artifact_profile(
    path: &Path,
    source_sha256: &str,
    fragment: &str,
    stats: TensorStats,
) -> ArtifactProfile {
    ArtifactProfile {
        path: format!("{}#{fragment}", path.display()),
        tensor_count: stats.tensor_count,
        tensor_bytes: stats.tensor_bytes,
        artifact_bytes: stats.tensor_bytes,
        sha256: source_sha256.to_string(),
    }
}

fn artifact_profile(artifact: &PackageArtifact) -> ArtifactProfile {
    ArtifactProfile {
        path: artifact.path.clone(),
        tensor_count: artifact.tensor_count,
        tensor_bytes: artifact.tensor_bytes,
        artifact_bytes: artifact.artifact_bytes,
        sha256: artifact.sha256.clone(),
    }
}

fn stage_profiles(
    manifest: &PackageManifest,
    stage_count: usize,
    timing_report: &ProfileTimingReport,
) -> Vec<StageProfile> {
    let layer_bytes = manifest
        .layers
        .iter()
        .map(|layer| (layer.layer_index, layer.artifact_bytes))
        .collect::<BTreeMap<_, _>>();
    partition_layers(manifest.layer_count, stage_count)
        .into_iter()
        .enumerate()
        .map(|(stage_index, (layer_start, layer_end))| {
            stage_profile(
                manifest,
                &layer_bytes,
                stage_count,
                stage_index,
                layer_start,
                layer_end,
                timing_report,
            )
        })
        .collect()
}

fn direct_stage_profiles(
    tensors: &[TensorInfo],
    layer_count: u32,
    stage_count: usize,
    timing_report: &ProfileTimingReport,
) -> Vec<StageProfile> {
    partition_layers(layer_count, stage_count)
        .into_iter()
        .enumerate()
        .map(|(stage_index, (layer_start, layer_end))| {
            direct_stage_profile(
                tensors,
                stage_count,
                stage_index,
                layer_start,
                layer_end,
                timing_report,
            )
        })
        .collect()
}

fn direct_stage_profile(
    tensors: &[TensorInfo],
    stage_count: usize,
    stage_index: usize,
    layer_start: u32,
    layer_end: u32,
    timing_report: &ProfileTimingReport,
) -> StageProfile {
    let includes_embeddings = stage_index == 0;
    let includes_output = stage_index + 1 == stage_count;
    let selected = tensors
        .iter()
        .filter(|tensor| {
            direct_tensor_in_stage(
                tensor,
                layer_start,
                layer_end,
                includes_embeddings,
                includes_output,
            )
        })
        .collect::<Vec<_>>();
    let mut parts = vec!["metadata".to_string()];
    if includes_embeddings {
        parts.push("embeddings".to_string());
    }
    for layer_index in layer_start..layer_end {
        parts.push(format!("layer:{layer_index}"));
    }
    if includes_output {
        parts.push("output".to_string());
    }
    StageProfile {
        stage_index,
        layer_start,
        layer_end,
        includes_embeddings,
        includes_output,
        part_count: parts.len(),
        artifact_bytes: selected.iter().map(|tensor| tensor.byte_size).sum(),
        timing: timing_report
            .stage_timings
            .get(&stage_index)
            .cloned()
            .unwrap_or_else(unmeasured_timing),
        parts,
    }
}

fn direct_tensor_in_stage(
    tensor: &TensorInfo,
    layer_start: u32,
    layer_end: u32,
    includes_embeddings: bool,
    includes_output: bool,
) -> bool {
    matches!(
        tensor.layer_index,
        Some(layer) if layer >= layer_start && layer < layer_end
    ) || (includes_embeddings && tensor.role == TensorRole::Embedding)
        || (includes_output && matches!(tensor.role, TensorRole::FinalNorm | TensorRole::Output))
        || matches!(
            tensor.role,
            TensorRole::Metadata | TensorRole::Tokenizer | TensorRole::Unknown
        )
}

fn stage_profile(
    manifest: &PackageManifest,
    layer_bytes: &BTreeMap<u32, u64>,
    stage_count: usize,
    stage_index: usize,
    layer_start: u32,
    layer_end: u32,
    timing_report: &ProfileTimingReport,
) -> StageProfile {
    let includes_embeddings = stage_index == 0;
    let includes_output = stage_index + 1 == stage_count;
    let mut parts = vec!["metadata".to_string()];
    let mut artifact_bytes = manifest.shared.metadata.artifact_bytes;
    if includes_embeddings {
        parts.push("embeddings".to_string());
        artifact_bytes += manifest.shared.embeddings.artifact_bytes;
    }
    for layer_index in layer_start..layer_end {
        parts.push(format!("layer:{layer_index}"));
        artifact_bytes += layer_bytes.get(&layer_index).copied().unwrap_or_default();
    }
    if includes_output {
        parts.push("output".to_string());
        artifact_bytes += manifest.shared.output.artifact_bytes;
    }
    StageProfile {
        stage_index,
        layer_start,
        layer_end,
        includes_embeddings,
        includes_output,
        part_count: parts.len(),
        artifact_bytes,
        timing: timing_report
            .stage_timings
            .get(&stage_index)
            .cloned()
            .unwrap_or_else(unmeasured_timing),
        parts,
    }
}

fn profile_summary(
    stage_count: usize,
    shared: &SharedProfile,
    layers: &[LayerProfile],
    timing_report: &ProfileTimingReport,
) -> ProfileSummary {
    let layer_artifact_bytes = layers
        .iter()
        .map(|layer| layer.artifact.artifact_bytes)
        .sum();
    let shared_artifact_bytes = shared.metadata.artifact_bytes
        + shared.embeddings.artifact_bytes
        + shared.output.artifact_bytes;
    ProfileSummary {
        stage_count,
        layer_artifact_bytes,
        shared_artifact_bytes,
        package_artifact_bytes: layer_artifact_bytes + shared_artifact_bytes,
        measured_layer_count: timing_report.layer_timings.len(),
        estimated_tokens_per_second: timing_report.estimated_tokens_per_second,
    }
}

fn runtime_profile(manifest: &PackageManifest) -> RuntimeProfile {
    RuntimeProfile {
        skippy_model_package_version: env!("CARGO_PKG_VERSION").to_string(),
        skippy_abi_version: format!(
            "{}.{}.{}",
            skippy_ffi::ABI_VERSION_MAJOR,
            skippy_ffi::ABI_VERSION_MINOR,
            skippy_ffi::ABI_VERSION_PATCH
        ),
        package_skippy_abi_version: manifest.skippy_abi_version.clone(),
    }
}

fn request_shape(args: &ProfileArgs) -> RequestShape {
    RequestShape {
        phase: args.phase,
        existing_kv_tokens: args.existing_kv_tokens,
        generated_tokens: args.generated_tokens,
        batch_size: args.batch_size,
        kv_type: args.kv_type.clone(),
        backend: args.backend.clone(),
        device: args.device.clone(),
    }
}

fn measurement_config(args: &ProfileArgs) -> MeasurementConfig {
    MeasurementConfig {
        source: args.timing_source,
        warmup_samples: args.warmup_samples,
        samples: args.samples,
    }
}

fn timing_source(kind: TimingSourceKind) -> Box<dyn ProfileTimingSource> {
    match kind {
        TimingSourceKind::Static => Box::new(StaticTimingSource),
        TimingSourceKind::LocalStage => Box::new(LocalStageTimingSource),
    }
}

fn validate_local_stage_input(input: &ProfileTimingInput<'_>) -> Result<()> {
    if input.input_kind != ProfileInputKind::DirectGguf {
        bail!(
            "--timing-source local-stage currently supports direct GGUF inputs only; use --timing-source static for layer packages"
        );
    }
    if input.stage_count != 1 {
        bail!("--timing-source local-stage currently supports --stages 1 only");
    }
    if !matches!(input.request_shape.phase, ProfilePhase::Decode) {
        bail!("--timing-source local-stage currently supports --phase decode only");
    }
    if input.request_shape.generated_tokens != 1 {
        bail!("--timing-source local-stage currently supports --generated-tokens 1 only");
    }
    if input.measurement.samples == 0 {
        bail!("--samples must be greater than zero for --timing-source local-stage");
    }
    Ok(())
}

fn profile_local_stage_decode(input: &ProfileTimingInput<'_>) -> Result<ProfileTimingReport> {
    let model_info = ModelInfo::open(input.package)
        .with_context(|| format!("open {}", input.package.display()))?;
    let tensors = model_info
        .tensors()
        .with_context(|| format!("inspect {}", input.package.display()))?;
    let layer_end = direct_layer_count(&tensors)?;
    let model = StageModel::open(
        input.package,
        &local_stage_runtime_config(input, layer_end)?,
    )
    .with_context(|| format!("load local stage model {}", input.package.display()))?;
    let mut session = model
        .create_session()
        .context("create local stage session")?;
    let prefix_tokens =
        deterministic_prefix_tokens(&model, input.request_shape.existing_kv_tokens)?;
    session
        .prefill_chunked(&prefix_tokens)
        .context("prefill deterministic local-stage prefix")?;
    let sampling = deterministic_sampling();
    let mut token = *prefix_tokens
        .last()
        .context("deterministic local-stage prefix is empty")?;
    for _ in 0..input.measurement.warmup_samples {
        token = session
            .decode_step_sampled(token, Some(&sampling))
            .context("run local-stage warmup decode")?;
    }
    let mut samples = Vec::with_capacity(input.measurement.samples as usize);
    for _ in 0..input.measurement.samples {
        let started = Instant::now();
        token = session
            .decode_step_sampled(token, Some(&sampling))
            .context("run local-stage measured decode")?;
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    let timing = measured_timing(samples)?;
    let estimated_tokens_per_second = timing.mean_ms.map(|mean_ms| 1000.0 / mean_ms);
    Ok(ProfileTimingReport {
        measurement_status: MeasurementStatus {
            status: "measured".to_string(),
            reason: format!(
                "measured local single-stage decode over {} warmup samples and {} measured samples",
                input.measurement.warmup_samples, input.measurement.samples
            ),
        },
        layer_timings: BTreeMap::new(),
        stage_timings: BTreeMap::from([(0, timing)]),
        estimated_tokens_per_second,
    })
}

fn local_stage_runtime_config(
    input: &ProfileTimingInput<'_>,
    layer_end: u32,
) -> Result<RuntimeConfig> {
    let cache_type = parse_cache_type(&input.request_shape.kv_type)?;
    let total_tokens = input
        .request_shape
        .existing_kv_tokens
        .saturating_add(input.measurement.warmup_samples)
        .saturating_add(input.measurement.samples)
        .saturating_add(16);
    Ok(RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end,
        ctx_size: total_tokens.max(128),
        lane_count: 1,
        n_batch: None,
        n_ubatch: None,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: 999,
        selected_backend_device: input.request_shape.device.clone(),
        cache_type_k: cache_type,
        cache_type_v: cache_type,
        flash_attn_type: FlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::RuntimeSlice,
        projector_path: None,
        include_embeddings: true,
        include_output: true,
        filter_tensors_on_load: false,
    })
}

fn deterministic_prefix_tokens(model: &StageModel, target_tokens: u32) -> Result<Vec<i32>> {
    let target = usize::try_from(target_tokens).context("existing_kv_tokens exceeds usize")?;
    if target == 0 {
        bail!("--existing-kv-tokens must be greater than zero for local-stage profiling");
    }
    let scaffold = " mesh-llm profiler decode scaffold.";
    let mut repeat_count = target.div_ceil(8).max(1);
    loop {
        let prompt = scaffold.repeat(repeat_count);
        let mut tokens = model.tokenize(&prompt, true)?;
        if tokens.len() >= target {
            tokens.truncate(target);
            return Ok(tokens);
        }
        repeat_count = repeat_count.saturating_mul(2);
    }
}

fn deterministic_sampling() -> SamplingConfig {
    SamplingConfig {
        enabled: true,
        seed: 1,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        ..SamplingConfig::default()
    }
}

fn measured_timing(mut samples: Vec<f64>) -> Result<TimingProfile> {
    if samples.is_empty() {
        bail!("cannot summarize zero timing samples");
    }
    samples.sort_by(|left, right| left.total_cmp(right));
    let mean_ms = samples.iter().sum::<f64>() / samples.len() as f64;
    Ok(TimingProfile {
        status: "measured".to_string(),
        mean_ms: Some(mean_ms),
        p50_ms: Some(percentile(&samples, 0.50)),
        p95_ms: Some(percentile(&samples, 0.95)),
        samples: samples.len() as u32,
    })
}

fn percentile(sorted_samples: &[f64], percentile: f64) -> f64 {
    let last = sorted_samples.len() - 1;
    let index = (last as f64 * percentile).ceil() as usize;
    sorted_samples[index.min(last)]
}

fn unmeasured_timing() -> TimingProfile {
    TimingProfile {
        status: "not_measured".to_string(),
        mean_ms: None,
        p50_ms: None,
        p95_ms: None,
        samples: 0,
    }
}

fn partition_layers(layer_count: u32, stages: usize) -> Vec<(u32, u32)> {
    let base = layer_count / stages as u32;
    let extra = layer_count % stages as u32;
    let mut start = 0;
    (0..stages)
        .map(|stage_index| {
            let width = base + u32::from((stage_index as u32) < extra);
            let end = start + width;
            let range = (start, end);
            start = end;
            range
        })
        .collect()
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn file_sha256(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(sha256_bytes(&bytes))
}

#[derive(Debug, Default)]
struct TensorStats {
    tensor_count: usize,
    tensor_bytes: u64,
}

impl TensorStats {
    fn add_tensor(&mut self, byte_size: u64) {
        self.tensor_count += 1;
        self.tensor_bytes += byte_size;
    }
}

fn direct_tensor_stats(
    tensors: &[TensorInfo],
    predicate: impl Fn(&TensorInfo) -> bool,
) -> TensorStats {
    let mut stats = TensorStats::default();
    for tensor in tensors.iter().filter(|tensor| predicate(tensor)) {
        stats.add_tensor(tensor.byte_size);
    }
    stats
}

#[cfg(test)]
mod tests;
