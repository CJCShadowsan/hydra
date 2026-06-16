use std::{
    fs,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SPD_HEAD_MANIFEST_SCHEMA: &str = "skippy-spd-head/v1";
pub const TORCH_SPD_HEAD_FORMAT_V10: &str = "torch-speculation-head-v10";

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct SpdHeadManifest {
    pub schema: String,
    pub checkpoint: SpdHeadCheckpoint,
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

    pub fn verify_checkpoint(&self, manifest_path: impl AsRef<Path>) -> Result<()> {
        let checkpoint_path = self.checkpoint_path(manifest_path)?;
        let metadata = fs::metadata(&checkpoint_path).with_context(|| {
            format!("read SPD checkpoint metadata {}", checkpoint_path.display())
        })?;
        if metadata.len() != self.checkpoint.bytes {
            bail!(
                "SPD checkpoint byte size mismatch for {}: expected {}, got {}",
                checkpoint_path.display(),
                self.checkpoint.bytes,
                metadata.len()
            );
        }
        let actual = file_sha256(&checkpoint_path)?;
        if actual != self.checkpoint.sha256 {
            bail!(
                "SPD checkpoint checksum mismatch for {}: expected {}, got {}",
                checkpoint_path.display(),
                self.checkpoint.sha256,
                actual
            );
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
