mod qwen;
mod safetensors;
mod tap_input;
mod tap_plan;

use std::{
    fs,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use qwen::{
    SpdQwen3FixtureDiagnostics, SpdQwen3FixtureParity, SpdQwen3FixtureTopK, SpdQwen3ForwardInput,
    SpdQwen3Head, run_qwen3_fixture_parity, run_qwen3_forward_from_inputs,
};
pub use safetensors::{SpdSafetensorsFile, SpdSafetensorsIndex, SpdSafetensorsTensor};
pub use tap_input::{
    SpdTapInputFixtureParity, SpdTapInputFixtureRowParity, SpdTapInputProjection,
    project_spd_tap_input_row, run_spd_tap_input_fixture_parity,
};
pub use tap_plan::{
    SpdHiddenStateRequirement, SpdHiddenStateSource, SpdHiddenTapPlan, SpdStageLayerRange,
    plan_hidden_state_taps,
};

pub const SPD_HEAD_MANIFEST_SCHEMA: &str = "skippy-spd-head/v1";
pub const TORCH_SPD_HEAD_FORMAT_V10: &str = "torch-speculation-head-v10";
pub const SPD_SERVING_CHECKPOINT_FORMAT_SAFETENSORS_V1: &str = "safetensors-spd-head-v1";
pub const SPD_PARITY_FIXTURE_SCHEMA: &str = "skippy-spd-parity-fixture/v1";

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct SpdHeadManifest {
    pub schema: String,
    pub checkpoint: SpdHeadCheckpoint,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serving_checkpoint: Option<SpdHeadServingCheckpoint>,
    pub source: SpdHeadSource,
    pub topology: SpdHeadTopology,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct SpdHeadCheckpoint {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct SpdHeadServingCheckpoint {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
    pub format: String,
    pub tensor_count: u32,
    pub dtype: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct SpdHeadSource {
    pub format: String,
    pub reference_repo: Option<String>,
    pub base_model_path: String,
    pub model_type: Option<String>,
    pub checkpoint_version: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct SpdHeadTopology {
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub draft_vocab_size: u32,
    pub num_stages: u32,
    pub num_spec_layers: u32,
    pub trained_with_use_deepest: bool,
    pub shallow_hidden_layer_indices: Vec<Vec<u32>>,
    pub spec_init_from_base_layers: Option<Vec<u32>>,
    pub draft_token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpdHeadRuntimeProfile<'a> {
    pub base_model_path: Option<&'a str>,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub num_stages: u32,
}

impl SpdHeadManifest {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let manifest: Self = serde_json::from_slice(
            &fs::read(path)
                .with_context(|| format!("read SPD head manifest {}", path.display()))?,
        )
        .with_context(|| format!("parse SPD head manifest {}", path.display()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != SPD_HEAD_MANIFEST_SCHEMA {
            bail!(
                "unsupported SPD head manifest schema {}; expected {}",
                self.schema,
                SPD_HEAD_MANIFEST_SCHEMA
            );
        }
        if self.source.format != TORCH_SPD_HEAD_FORMAT_V10 || self.source.checkpoint_version != 10 {
            bail!(
                "unsupported SPD head format {} version {}; expected {} version 10",
                self.source.format,
                self.source.checkpoint_version,
                TORCH_SPD_HEAD_FORMAT_V10
            );
        }
        if self.source.base_model_path.trim().is_empty() {
            bail!("SPD head manifest base_model_path must not be empty");
        }
        self.checkpoint.validate()?;
        if let Some(serving_checkpoint) = &self.serving_checkpoint {
            serving_checkpoint.validate()?;
        }
        self.topology.validate()?;
        Ok(())
    }

    pub fn checkpoint_path(&self, manifest_path: impl AsRef<Path>) -> Result<PathBuf> {
        let manifest_path = manifest_path.as_ref();
        let base = manifest_path
            .parent()
            .with_context(|| format!("resolve parent for {}", manifest_path.display()))?;
        Ok(base.join(safe_relative_manifest_path(&self.checkpoint.path)?))
    }

    pub fn serving_checkpoint_path(&self, manifest_path: impl AsRef<Path>) -> Result<PathBuf> {
        let serving_checkpoint = self
            .serving_checkpoint
            .as_ref()
            .context("SPD manifest does not include a serving checkpoint")?;
        let manifest_path = manifest_path.as_ref();
        let base = manifest_path
            .parent()
            .with_context(|| format!("resolve parent for {}", manifest_path.display()))?;
        Ok(base.join(safe_relative_manifest_path(&serving_checkpoint.path)?))
    }

    pub fn verify_checkpoint(&self, manifest_path: impl AsRef<Path>) -> Result<()> {
        let checkpoint_path = self.checkpoint_path(manifest_path)?;
        verify_checkpoint_artifact(
            "SPD checkpoint",
            &checkpoint_path,
            self.checkpoint.bytes,
            &self.checkpoint.sha256,
        )
    }

    pub fn verify_serving_checkpoint(
        &self,
        manifest_path: impl AsRef<Path>,
    ) -> Result<SpdSafetensorsIndex> {
        let serving_checkpoint = self
            .serving_checkpoint
            .as_ref()
            .context("SPD manifest does not include a serving checkpoint")?;
        let path = self.serving_checkpoint_path(manifest_path)?;
        verify_checkpoint_artifact(
            "SPD serving checkpoint",
            &path,
            serving_checkpoint.bytes,
            &serving_checkpoint.sha256,
        )?;
        let index = SpdSafetensorsIndex::from_path(&path)?;
        serving_checkpoint.validate_index(&index)?;
        Ok(index)
    }

    pub fn serving_checkpoint_index(
        &self,
        manifest_path: impl AsRef<Path>,
    ) -> Result<SpdSafetensorsIndex> {
        let serving_checkpoint = self
            .serving_checkpoint
            .as_ref()
            .context("SPD manifest does not include a serving checkpoint")?;
        let index = SpdSafetensorsIndex::from_path(self.serving_checkpoint_path(manifest_path)?)?;
        serving_checkpoint.validate_index(&index)?;
        Ok(index)
    }

    pub fn ensure_serving_checkpoint_for_runtime(
        &self,
        manifest_path: impl AsRef<Path>,
    ) -> Result<SpdSafetensorsIndex> {
        let index = self.verify_serving_checkpoint(manifest_path)?;
        self.ensure_serving_tensor_shapes(&index)?;
        Ok(index)
    }

    fn ensure_serving_tensor_shapes(&self, index: &SpdSafetensorsIndex) -> Result<()> {
        let hidden_size = self.topology.hidden_size as u64;
        for (stage, indices) in self
            .topology
            .shallow_hidden_layer_indices
            .iter()
            .enumerate()
        {
            let expected_width = hidden_size
                .checked_mul(indices.len() as u64)
                .context("SPD stage projection width overflow")?;
            index.ensure_tensor_shape(
                &format!("stage_projs.{stage}.weight"),
                &[hidden_size, expected_width],
            )?;
        }
        index.ensure_tensor_shape("g0_proj.weight", &[hidden_size, hidden_size])?;
        index.ensure_tensor_shape(
            "lm_head.weight",
            &[self.topology.draft_vocab_size as u64, hidden_size],
        )?;
        for layer in 0..self.topology.num_spec_layers {
            index.ensure_tensor_shape(
                &format!("spec_layers.{layer}.input_layernorm.weight"),
                &[hidden_size],
            )?;
            index.ensure_tensor_shape(
                &format!("spec_layers.{layer}.post_attention_layernorm.weight"),
                &[hidden_size],
            )?;
        }
        Ok(())
    }

    pub fn ensure_runtime_compatible(&self, profile: &SpdHeadRuntimeProfile<'_>) -> Result<()> {
        match profile.base_model_path {
            Some(base_model_path) if self.source.base_model_path != base_model_path => {
                bail!(
                    "SPD head was trained for base model {}; runtime model is {}",
                    self.source.base_model_path,
                    base_model_path
                )
            }
            _ => {}
        }
        if self.topology.hidden_size != profile.hidden_size {
            bail!(
                "SPD head hidden_size {} does not match runtime hidden_size {}",
                self.topology.hidden_size,
                profile.hidden_size
            );
        }
        if self.topology.vocab_size != profile.vocab_size {
            bail!(
                "SPD head vocab_size {} does not match runtime vocab_size {}",
                self.topology.vocab_size,
                profile.vocab_size
            );
        }
        if self.topology.num_stages != profile.num_stages {
            bail!(
                "SPD head num_stages {} does not match runtime num_stages {}",
                self.topology.num_stages,
                profile.num_stages
            );
        }
        Ok(())
    }
}

impl SpdHeadCheckpoint {
    fn validate(&self) -> Result<()> {
        let _ = safe_relative_manifest_path(&self.path)?;
        if self.bytes == 0 {
            bail!("SPD checkpoint bytes must be greater than zero");
        }
        validate_sha256_digest("SPD checkpoint sha256", &self.sha256)
    }
}

impl SpdHeadServingCheckpoint {
    fn validate(&self) -> Result<()> {
        let _ = safe_relative_manifest_path(&self.path)?;
        if self.bytes == 0 {
            bail!("SPD serving checkpoint bytes must be greater than zero");
        }
        if self.format != SPD_SERVING_CHECKPOINT_FORMAT_SAFETENSORS_V1 {
            bail!(
                "unsupported SPD serving checkpoint format {}; expected {}",
                self.format,
                SPD_SERVING_CHECKPOINT_FORMAT_SAFETENSORS_V1
            );
        }
        if self.tensor_count == 0 {
            bail!("SPD serving checkpoint tensor_count must be greater than zero");
        }
        if self.dtype.trim().is_empty() {
            bail!("SPD serving checkpoint dtype must not be empty");
        }
        validate_sha256_digest("SPD serving checkpoint sha256", &self.sha256)
    }

    fn validate_index(&self, index: &SpdSafetensorsIndex) -> Result<()> {
        if index.tensors.len() != self.tensor_count as usize {
            bail!(
                "SPD serving checkpoint tensor_count mismatch: expected {}, got {}",
                self.tensor_count,
                index.tensors.len()
            );
        }
        if self.dtype != "mixed"
            && index
                .tensors
                .values()
                .any(|tensor| tensor.dtype != self.dtype)
        {
            bail!(
                "SPD serving checkpoint dtype mismatch: expected all tensors to be {}",
                self.dtype
            );
        }
        match index.metadata.get("format") {
            Some(format) if format != SPD_SERVING_CHECKPOINT_FORMAT_SAFETENSORS_V1 => {
                bail!(
                    "SPD serving checkpoint metadata format {}; expected {}",
                    format,
                    SPD_SERVING_CHECKPOINT_FORMAT_SAFETENSORS_V1
                );
            }
            _ => {}
        }
        Ok(())
    }
}

impl SpdHeadTopology {
    fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            bail!("SPD head hidden_size must be greater than zero");
        }
        if self.vocab_size == 0 {
            bail!("SPD head vocab_size must be greater than zero");
        }
        if self.draft_vocab_size == 0 || self.draft_vocab_size > self.vocab_size {
            bail!(
                "SPD head draft_vocab_size {} must be in 1..={}",
                self.draft_vocab_size,
                self.vocab_size
            );
        }
        if self.num_stages == 0 {
            bail!("SPD head num_stages must be greater than zero");
        }
        if self.num_spec_layers == 0 {
            bail!("SPD head num_spec_layers must be greater than zero");
        }
        if self.shallow_hidden_layer_indices.len() != self.num_stages as usize {
            bail!(
                "SPD head shallow_hidden_layer_indices length {} must match num_stages {}",
                self.shallow_hidden_layer_indices.len(),
                self.num_stages
            );
        }
        for (stage, indices) in self.shallow_hidden_layer_indices.iter().enumerate() {
            validate_sorted_unique_indices(
                &format!("shallow_hidden_layer_indices[{stage}]"),
                indices,
            )?;
        }
        match &self.spec_init_from_base_layers {
            Some(indices) if indices.len() != self.num_spec_layers as usize => {
                bail!(
                    "SPD head spec_init_from_base_layers length {} must match num_spec_layers {}",
                    indices.len(),
                    self.num_spec_layers
                )
            }
            _ => {}
        }
        if let Some(ids) = &self.draft_token_ids {
            if ids.len() != self.draft_vocab_size as usize {
                bail!(
                    "SPD head draft_token_ids length {} must match draft_vocab_size {}",
                    ids.len(),
                    self.draft_vocab_size
                );
            }
            validate_sorted_unique_indices("draft_token_ids", ids)?;
            if ids.iter().any(|id| *id >= self.vocab_size) {
                bail!(
                    "SPD head draft_token_ids must all be less than vocab_size {}",
                    self.vocab_size
                );
            }
        }
        Ok(())
    }
}

fn verify_checkpoint_artifact(
    label: &str,
    path: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
) -> Result<()> {
    let metadata =
        fs::metadata(path).with_context(|| format!("read {label} metadata {}", path.display()))?;
    if metadata.len() != expected_bytes {
        bail!(
            "{label} byte size mismatch for {}: expected {}, got {}",
            path.display(),
            expected_bytes,
            metadata.len()
        );
    }
    let actual = file_sha256(path)?;
    if actual != expected_sha256 {
        bail!(
            "{label} checksum mismatch for {}: expected {}, got {}",
            path.display(),
            expected_sha256,
            actual
        );
    }
    Ok(())
}

fn validate_sorted_unique_indices(label: &str, values: &[u32]) -> Result<()> {
    if values.is_empty() {
        bail!("SPD head {label} must not be empty");
    }
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        bail!("SPD head {label} must be sorted and unique");
    }
    Ok(())
}

fn validate_sha256_digest(label: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{label} must be a 64-character hex digest");
    }
    Ok(())
}

fn safe_relative_manifest_path(path: &str) -> Result<PathBuf> {
    let path = Path::new(path);
    if path.as_os_str().is_empty() || path.is_absolute() {
        bail!("SPD checkpoint path must be a non-empty relative path");
    }
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            _ => bail!(
                "SPD checkpoint path must not contain prefix, root, dot, or parent components"
            ),
        }
    }
    Ok(path.to_path_buf())
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("open SPD checkpoint for hashing {}", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)
        .with_context(|| format!("hash SPD checkpoint {}", path.display()))?;
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_manifest() -> SpdHeadManifest {
        SpdHeadManifest {
            schema: SPD_HEAD_MANIFEST_SCHEMA.to_string(),
            checkpoint: SpdHeadCheckpoint {
                path: "speculation_head_final.pt".to_string(),
                sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_string(),
                bytes: 4,
            },
            serving_checkpoint: None,
            source: SpdHeadSource {
                format: TORCH_SPD_HEAD_FORMAT_V10.to_string(),
                reference_repo: Some("https://example.invalid/spd.git".to_string()),
                base_model_path: "Qwen/Qwen3-0.6B".to_string(),
                model_type: Some("qwen3".to_string()),
                checkpoint_version: 10,
            },
            topology: SpdHeadTopology {
                hidden_size: 1024,
                vocab_size: 10,
                draft_vocab_size: 3,
                num_stages: 2,
                num_spec_layers: 1,
                trained_with_use_deepest: true,
                shallow_hidden_layer_indices: vec![vec![0, 7, 14], vec![0, 14]],
                spec_init_from_base_layers: Some(vec![20]),
                draft_token_ids: Some(vec![1, 3, 5]),
            },
        }
    }

    fn valid_manifest_with_serving_checkpoint() -> SpdHeadManifest {
        let mut manifest = valid_manifest();
        manifest.serving_checkpoint = Some(SpdHeadServingCheckpoint {
            path: "spd-head.safetensors".to_string(),
            sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
            bytes: 0,
            format: SPD_SERVING_CHECKPOINT_FORMAT_SAFETENSORS_V1.to_string(),
            tensor_count: 6,
            dtype: "F32".to_string(),
        });
        manifest
    }

    fn write_test_safetensors(path: &Path, tensors: &[(&str, &str, &[u64])]) {
        let mut header_entries = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape) in tensors {
            let start = data.len() as u64;
            let bytes = test_tensor_byte_len(dtype, shape);
            data.resize(data.len() + bytes as usize, 0);
            let end = data.len() as u64;
            header_entries.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
        }
        header_entries.insert(
            "__metadata__".to_string(),
            serde_json::json!({"format": SPD_SERVING_CHECKPOINT_FORMAT_SAFETENSORS_V1}),
        );
        let header = serde_json::to_vec(&serde_json::Value::Object(header_entries)).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&data);
        fs::write(path, bytes).unwrap();
    }

    fn test_tensor_byte_len(dtype: &str, shape: &[u64]) -> u64 {
        let element_bytes = match dtype {
            "BF16" | "F16" | "I16" | "U16" => 2,
            "F32" | "I32" | "U32" => 4,
            "F64" | "I64" | "U64" => 8,
            _ => 1,
        };
        shape.iter().product::<u64>() * element_bytes
    }

    #[test]
    fn validates_reference_manifest_shape() {
        valid_manifest().validate().unwrap();
    }

    #[test]
    fn rejects_draft_vocab_size_mismatch() {
        let mut manifest = valid_manifest();
        manifest.topology.draft_token_ids = Some(vec![1, 3]);
        let error = manifest.validate().unwrap_err().to_string();
        assert!(error.contains("draft_token_ids length"));
    }

    #[test]
    fn rejects_unsafe_checkpoint_path() {
        let mut manifest = valid_manifest();
        manifest.checkpoint.path = "../speculation_head_final.pt".to_string();
        let error = manifest.validate().unwrap_err().to_string();
        assert!(error.contains("parent components"));
    }

    #[test]
    fn verifies_checkpoint_checksum_relative_to_manifest() {
        let temp = tempfile::tempdir().unwrap();
        let checkpoint = temp.path().join("speculation_head_final.pt");
        fs::write(&checkpoint, b"head").unwrap();
        let sha256 = file_sha256(&checkpoint).unwrap();

        let mut manifest = valid_manifest();
        manifest.checkpoint.sha256 = sha256;
        manifest.checkpoint.bytes = 4;
        let manifest_path = temp.path().join("skippy-spd-head.json");
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let parsed = SpdHeadManifest::from_path(&manifest_path).unwrap();
        parsed.verify_checkpoint(&manifest_path).unwrap();
    }

    #[test]
    fn verifies_serving_checkpoint_checksum_and_shapes() {
        let temp = tempfile::tempdir().unwrap();
        let checkpoint = temp.path().join("spd-head.safetensors");
        write_test_safetensors(
            &checkpoint,
            &[
                ("stage_projs.0.weight", "F32", &[1024, 3072]),
                ("stage_projs.1.weight", "F32", &[1024, 2048]),
                ("g0_proj.weight", "F32", &[1024, 1024]),
                ("lm_head.weight", "F32", &[3, 1024]),
                ("spec_layers.0.input_layernorm.weight", "F32", &[1024]),
                (
                    "spec_layers.0.post_attention_layernorm.weight",
                    "F32",
                    &[1024],
                ),
            ],
        );
        let sha256 = file_sha256(&checkpoint).unwrap();

        let mut manifest = valid_manifest_with_serving_checkpoint();
        let serving_checkpoint = manifest.serving_checkpoint.as_mut().unwrap();
        serving_checkpoint.sha256 = sha256;
        serving_checkpoint.bytes = fs::metadata(&checkpoint).unwrap().len();
        let manifest_path = temp.path().join("skippy-spd-head.json");
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let parsed = SpdHeadManifest::from_path(&manifest_path).unwrap();
        let index = parsed
            .ensure_serving_checkpoint_for_runtime(&manifest_path)
            .unwrap();
        assert_eq!(index.tensors.len(), 6);
        assert_eq!(
            index.metadata.get("format").unwrap(),
            SPD_SERVING_CHECKPOINT_FORMAT_SAFETENSORS_V1
        );

        let file = SpdSafetensorsFile::open(&checkpoint).unwrap();
        let lm_head = file.read_tensor_f32("lm_head.weight").unwrap();
        assert_eq!(lm_head.len(), 3 * 1024);
        assert!(lm_head.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn rejects_serving_checkpoint_shape_mismatch() {
        let temp = tempfile::tempdir().unwrap();
        let checkpoint = temp.path().join("spd-head.safetensors");
        write_test_safetensors(
            &checkpoint,
            &[
                ("stage_projs.0.weight", "F32", &[1024, 2048]),
                ("stage_projs.1.weight", "F32", &[1024, 2048]),
                ("g0_proj.weight", "F32", &[1024, 1024]),
                ("lm_head.weight", "F32", &[3, 1024]),
                ("spec_layers.0.input_layernorm.weight", "F32", &[1024]),
                (
                    "spec_layers.0.post_attention_layernorm.weight",
                    "F32",
                    &[1024],
                ),
            ],
        );
        let sha256 = file_sha256(&checkpoint).unwrap();

        let mut manifest = valid_manifest_with_serving_checkpoint();
        let serving_checkpoint = manifest.serving_checkpoint.as_mut().unwrap();
        serving_checkpoint.sha256 = sha256;
        serving_checkpoint.bytes = fs::metadata(&checkpoint).unwrap().len();
        let manifest_path = temp.path().join("skippy-spd-head.json");
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let parsed = SpdHeadManifest::from_path(&manifest_path).unwrap();
        let error = parsed
            .ensure_serving_checkpoint_for_runtime(&manifest_path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("stage_projs.0.weight shape mismatch"));
    }

    #[test]
    fn rejects_safetensors_byte_length_mismatch() {
        let header = serde_json::json!({
            "bad.weight": {
                "dtype": "F32",
                "shape": [2_u64, 2_u64],
                "data_offsets": [0_u64, 12_u64],
            }
        });
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&[0_u8; 12]);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("bad.safetensors");
        fs::write(&path, bytes).unwrap();

        let error = SpdSafetensorsIndex::from_path(&path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("byte length mismatch"));
    }

    #[test]
    fn validates_external_manifest_when_skippy_spd_manifest_is_set() {
        let Ok(manifest_path) = std::env::var("SKIPPY_SPD_MANIFEST") else {
            return;
        };
        let manifest_path = PathBuf::from(manifest_path);
        let manifest = SpdHeadManifest::from_path(&manifest_path).unwrap();
        let index = manifest
            .ensure_serving_checkpoint_for_runtime(&manifest_path)
            .unwrap();
        assert!(index.tensors.contains_key("lm_head.weight"));
    }

    #[test]
    fn validates_external_parity_fixture_when_skippy_spd_parity_fixture_is_set() {
        let Ok(fixture_path) = std::env::var("SKIPPY_SPD_PARITY_FIXTURE") else {
            return;
        };
        let file = SpdSafetensorsFile::open(PathBuf::from(fixture_path)).unwrap();
        let index = &file.index;
        assert_eq!(
            index.metadata.get("schema").map(String::as_str),
            Some(SPD_PARITY_FIXTURE_SCHEMA)
        );

        let cur_in = index.tensors.get("cur_in").unwrap();
        let position_ids = index.tensors.get("position_ids").unwrap();
        let final_norm_weight = index.tensors.get("final_norm_weight").unwrap();
        let python_logits = index.tensors.get("python_logits").unwrap();
        let topk_token_ids = index.tensors.get("python_topk_token_ids").unwrap();

        assert_eq!(cur_in.shape.len(), 3);
        assert_eq!(position_ids.shape, vec![cur_in.shape[0], cur_in.shape[1]]);
        assert_eq!(final_norm_weight.shape, vec![cur_in.shape[2]]);
        assert_eq!(python_logits.shape.len(), 3);
        assert_eq!(python_logits.shape[0], cur_in.shape[0]);
        assert_eq!(python_logits.shape[1], 1);
        assert_eq!(topk_token_ids.shape.len(), 1);

        let cur_in = file.read_tensor_f32("cur_in").unwrap();
        let token_ids = file.read_tensor_i64("python_topk_token_ids").unwrap();
        assert!(!cur_in.is_empty());
        assert!(!token_ids.is_empty());
    }

    #[test]
    fn qwen3_fixture_forward_matches_python_topk_when_env_is_set() {
        let (Ok(manifest_path), Ok(fixture_path)) = (
            std::env::var("SKIPPY_SPD_MANIFEST"),
            std::env::var("SKIPPY_SPD_PARITY_FIXTURE"),
        ) else {
            return;
        };
        let parity = qwen::run_qwen3_fixture_parity(manifest_path, fixture_path, 8).unwrap();
        assert_eq!(
            parity.rust.draft_indices, parity.python.draft_indices,
            "diagnostics: {:?}",
            parity.diagnostics
        );
        assert_eq!(
            parity.rust.token_ids, parity.python.token_ids,
            "diagnostics: {:?}",
            parity.diagnostics
        );
        assert!(
            parity.diagnostics.spec_query_max_abs_diff <= 0.125,
            "diagnostics: {:?}",
            parity.diagnostics
        );
        assert!(
            parity.diagnostics.final_hidden_max_abs_diff <= 0.125,
            "diagnostics: {:?}",
            parity.diagnostics
        );
    }

    #[test]
    fn tap_input_fixture_reconstructs_cur_in_when_env_is_set() {
        let (Ok(manifest_path), Ok(fixture_path)) = (
            std::env::var("SKIPPY_SPD_MANIFEST"),
            std::env::var("SKIPPY_SPD_PARITY_FIXTURE"),
        ) else {
            return;
        };
        let fixture_file = SpdSafetensorsFile::open(PathBuf::from(&fixture_path)).unwrap();
        if !fixture_file.index.tensors.contains_key("tap_row_0_concat") {
            return;
        }
        let parity =
            tap_input::run_spd_tap_input_fixture_parity(manifest_path, fixture_path).unwrap();
        assert!(
            parity.max_abs_diff <= 0.125,
            "tap input parity: {:?}",
            parity
        );
    }

    #[test]
    fn checks_runtime_compatibility_profile() {
        let manifest = valid_manifest();
        manifest
            .ensure_runtime_compatible(&SpdHeadRuntimeProfile {
                base_model_path: Some("Qwen/Qwen3-0.6B"),
                hidden_size: 1024,
                vocab_size: 10,
                num_stages: 2,
            })
            .unwrap();
    }

    #[test]
    fn rejects_runtime_profile_mismatch() {
        let manifest = valid_manifest();
        let error = manifest
            .ensure_runtime_compatible(&SpdHeadRuntimeProfile {
                base_model_path: Some("Qwen/Qwen3-1.7B"),
                hidden_size: 1024,
                vocab_size: 10,
                num_stages: 2,
            })
            .unwrap_err()
            .to_string();
        assert!(error.contains("trained for base model"));
    }
}
