use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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

#[derive(Debug, Serialize)]
pub(crate) struct ProfileReport {
    pub(crate) schema_version: u32,
    pub(crate) kind: String,
    pub(crate) package_path: String,
    pub(crate) model_id: String,
    pub(crate) layer_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) activation_width: Option<u32>,
    pub(crate) manifest_sha256: String,
    pub(crate) runtime: RuntimeProfile,
    pub(crate) request_shape: RequestShape,
    pub(crate) measurement_status: MeasurementStatus,
    pub(crate) summary: ProfileSummary,
    pub(crate) shared: SharedProfile,
    pub(crate) layers: Vec<LayerProfile>,
    pub(crate) stages: Vec<StageProfile>,
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

#[derive(Debug, Serialize)]
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

pub(crate) fn run_profile(args: ProfileArgs) -> Result<()> {
    let report = profile_package(&args)?;
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(out) = &args.out {
        fs::write(out, format!("{json}\n"))
            .with_context(|| format!("write profile report {}", out.display()))?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn profile_package(args: &ProfileArgs) -> Result<ProfileReport> {
    let manifest_path = args.package.join("model-package.json");
    let manifest_contents = fs::read(&manifest_path)
        .with_context(|| format!("read package manifest {}", manifest_path.display()))?;
    let manifest_sha256 = sha256_bytes(&manifest_contents);
    let manifest = serde_json::from_slice::<PackageManifest>(&manifest_contents)
        .with_context(|| format!("parse package manifest {}", manifest_path.display()))?;
    validate_stage_count(args.stages, manifest.layer_count)?;

    let layers = layer_profiles(&manifest);
    let shared = shared_profile(&manifest.shared);
    let stages = stage_profiles(&manifest, args.stages);
    let summary = profile_summary(args.stages, &shared, &layers);
    let runtime = runtime_profile(&manifest);

    Ok(ProfileReport {
        schema_version: 1,
        kind: "skippy_agent_quant_profile".to_string(),
        package_path: args.package.display().to_string(),
        model_id: manifest.model_id,
        layer_count: manifest.layer_count,
        activation_width: manifest.activation_width,
        manifest_sha256,
        runtime,
        request_shape: request_shape(args),
        measurement_status: MeasurementStatus {
            status: "not_measured".to_string(),
            reason: "native per-layer timing hooks are not wired yet; this report is a planner-ready static profile envelope".to_string(),
        },
        summary,
        shared,
        layers,
        stages,
    })
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

fn layer_profiles(manifest: &PackageManifest) -> Vec<LayerProfile> {
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
            timing: unmeasured_timing(),
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

fn artifact_profile(artifact: &PackageArtifact) -> ArtifactProfile {
    ArtifactProfile {
        path: artifact.path.clone(),
        tensor_count: artifact.tensor_count,
        tensor_bytes: artifact.tensor_bytes,
        artifact_bytes: artifact.artifact_bytes,
        sha256: artifact.sha256.clone(),
    }
}

fn stage_profiles(manifest: &PackageManifest, stage_count: usize) -> Vec<StageProfile> {
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
            )
        })
        .collect()
}

fn stage_profile(
    manifest: &PackageManifest,
    layer_bytes: &BTreeMap<u32, u64>,
    stage_count: usize,
    stage_index: usize,
    layer_start: u32,
    layer_end: u32,
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
        timing: unmeasured_timing(),
        parts,
    }
}

fn profile_summary(
    stage_count: usize,
    shared: &SharedProfile,
    layers: &[LayerProfile],
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
        measured_layer_count: 0,
        estimated_tokens_per_second: None,
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

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn profile_reports_decode_scaffold_and_stage_bytes() {
        let dir = temp_dir("profile-stage-bytes");
        write_manifest(&dir);
        let args = ProfileArgs {
            package: dir.clone(),
            stages: 2,
            phase: ProfilePhase::Decode,
            existing_kv_tokens: 32_768,
            generated_tokens: 1,
            batch_size: 1,
            kv_type: "f16".to_string(),
            backend: Some("metal".to_string()),
            device: Some("metal:test".to_string()),
            out: None,
        };

        let report = profile_package(&args).expect("profile package");

        assert_eq!(report.kind, "skippy_agent_quant_profile");
        assert_eq!(report.request_shape.phase as u8, ProfilePhase::Decode as u8);
        assert_eq!(report.measurement_status.status, "not_measured");
        assert_eq!(report.summary.stage_count, 2);
        assert_eq!(report.summary.layer_artifact_bytes, 100);
        assert_eq!(report.summary.shared_artifact_bytes, 35);
        assert_eq!(report.layers.len(), 4);
        assert_eq!(report.stages.len(), 2);
        assert_eq!(report.stages[0].layer_start, 0);
        assert_eq!(report.stages[0].layer_end, 2);
        assert_eq!(report.stages[0].artifact_bytes, 65);
        assert_eq!(report.stages[1].layer_start, 2);
        assert_eq!(report.stages[1].layer_end, 4);
        assert_eq!(report.stages[1].artifact_bytes, 80);

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn profile_rejects_too_many_stages() {
        let dir = temp_dir("profile-too-many-stages");
        write_manifest(&dir);
        let args = ProfileArgs {
            package: dir.clone(),
            stages: 5,
            phase: ProfilePhase::Decode,
            existing_kv_tokens: 8192,
            generated_tokens: 1,
            batch_size: 1,
            kv_type: "f16".to_string(),
            backend: None,
            device: None,
            out: None,
        };

        let error = profile_package(&args).expect_err("stage count should fail");

        assert!(
            error
                .to_string()
                .contains("--stages 5 exceeds package layer_count 4")
        );
        fs::remove_dir_all(dir).ok();
    }

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("skippy-profile-{name}-{nanos}"));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn write_manifest(dir: &Path) {
        fs::write(
            dir.join("model-package.json"),
            r#"{
  "model_id": "test/model:Q4_K_M",
  "layer_count": 4,
  "activation_width": 1024,
  "shared": {
    "metadata": {
      "path": "shared/metadata.gguf",
      "tensor_count": 1,
      "tensor_bytes": 10,
      "artifact_bytes": 10,
      "sha256": "metadata"
    },
    "embeddings": {
      "path": "shared/embeddings.gguf",
      "tensor_count": 1,
      "tensor_bytes": 11,
      "artifact_bytes": 20,
      "sha256": "embeddings"
    },
    "output": {
      "path": "shared/output.gguf",
      "tensor_count": 1,
      "tensor_bytes": 12,
      "artifact_bytes": 5,
      "sha256": "output"
    }
  },
  "layers": [
    {
      "layer_index": 0,
      "path": "layers/layer-00000.gguf",
      "tensor_count": 2,
      "tensor_bytes": 21,
      "artifact_bytes": 15,
      "sha256": "layer0"
    },
    {
      "layer_index": 1,
      "path": "layers/layer-00001.gguf",
      "tensor_count": 2,
      "tensor_bytes": 22,
      "artifact_bytes": 20,
      "sha256": "layer1"
    },
    {
      "layer_index": 2,
      "path": "layers/layer-00002.gguf",
      "tensor_count": 2,
      "tensor_bytes": 23,
      "artifact_bytes": 25,
      "sha256": "layer2"
    },
    {
      "layer_index": 3,
      "path": "layers/layer-00003.gguf",
      "tensor_count": 2,
      "tensor_bytes": 24,
      "artifact_bytes": 40,
      "sha256": "layer3"
    }
  ],
  "skippy_abi_version": "0.1.25"
}"#,
        )
        .expect("write manifest");
    }
}
