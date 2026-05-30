use crate::NativeRuntimeFlavor;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

pub const NATIVE_RUNTIME_MANIFEST_FILE: &str = "manifest.json";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeRequirement {
    #[serde(default)]
    pub gpu_name_contains: Vec<String>,
    #[serde(default)]
    pub min_cuda_compute_capability: Option<u32>,
    #[serde(default)]
    pub min_driver_version: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeArtifact {
    pub native_runtime_id: String,
    pub mesh_version: String,
    #[serde(default)]
    pub target_triple: Option<String>,
    pub os: String,
    pub arch: String,
    pub flavor: NativeRuntimeFlavor,
    #[serde(default)]
    pub priority: i64,
    #[serde(default)]
    pub skippy_abi_version: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub signature: Option<String>,
    #[serde(default)]
    pub library_paths: Vec<String>,
    #[serde(default)]
    pub requirements: Vec<NativeRuntimeRequirement>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeManifest {
    pub artifact: NativeRuntimeArtifact,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeReleaseManifest {
    pub mesh_version: String,
    #[serde(default)]
    pub artifacts: Vec<NativeRuntimeArtifact>,
}

impl NativeRuntimeManifest {
    pub fn read_from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join(NATIVE_RUNTIME_MANIFEST_FILE);
        let text = fs::read_to_string(&path)
            .with_context(|| format!("read native runtime manifest {}", path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("parse native runtime manifest {}", path.display()))
    }

    pub fn write_to_dir(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)
            .with_context(|| format!("create native runtime dir {}", dir.display()))?;
        let path = dir.join(NATIVE_RUNTIME_MANIFEST_FILE);
        let text = serde_json::to_string_pretty(self)?;
        fs::write(&path, format!("{text}\n"))
            .with_context(|| format!("write native runtime manifest {}", path.display()))
    }

    pub fn validate(&self) -> Result<()> {
        validate_artifact(&self.artifact)
    }
}

impl NativeRuntimeReleaseManifest {
    pub fn read_from_path(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("read native runtime release manifest {}", path.display()))?;
        let manifest: Self = serde_json::from_str(&text)
            .with_context(|| format!("parse native runtime release manifest {}", path.display()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        if self.mesh_version.trim().is_empty() {
            bail!("native runtime release manifest mesh_version is empty");
        }
        for artifact in &self.artifacts {
            validate_artifact(artifact)?;
            if artifact.mesh_version != self.mesh_version {
                bail!(
                    "native runtime artifact {} has mesh_version {}, expected {}",
                    artifact.native_runtime_id,
                    artifact.mesh_version,
                    self.mesh_version
                );
            }
        }
        Ok(())
    }
}

fn validate_artifact(artifact: &NativeRuntimeArtifact) -> Result<()> {
    if artifact.native_runtime_id.trim().is_empty() {
        bail!("native runtime artifact id is empty");
    }
    if artifact.mesh_version.trim().is_empty() {
        bail!(
            "native runtime artifact {} mesh_version is empty",
            artifact.native_runtime_id
        );
    }
    if artifact.os.trim().is_empty() || artifact.arch.trim().is_empty() {
        bail!(
            "native runtime artifact {} must declare os and arch",
            artifact.native_runtime_id
        );
    }
    Ok(())
}
