use std::{fs, path::Path};

use anyhow::{Context, Result, bail};
use semver::Version;
use serde::Deserialize;

pub const CURRENT_MESH_LLM_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct PluginTomlManifest {
    #[serde(default, alias = "minimum_mesh_llm_version", alias = "min_version")]
    pub min_mesh_llm_version: Option<String>,
}

pub fn load_plugin_toml(plugin_dir: &Path) -> Result<PluginTomlManifest> {
    let manifest_path = plugin_dir.join("plugin.toml");
    let contents = fs::read_to_string(&manifest_path)
        .with_context(|| format!("read plugin manifest {}", manifest_path.display()))?;
    toml::from_str(&contents)
        .with_context(|| format!("parse plugin manifest {}", manifest_path.display()))
}

pub fn validate_plugin_compatibility(plugin_dir: &Path, current_version: &str) -> Result<()> {
    let manifest = load_plugin_toml(plugin_dir)?;
    validate_plugin_manifest_compatibility(&manifest, current_version)
}

pub fn validate_plugin_manifest_compatibility(
    manifest: &PluginTomlManifest,
    current_version: &str,
) -> Result<()> {
    let Some(required_version) = manifest.min_mesh_llm_version.as_deref() else {
        return Ok(());
    };

    let required = parse_version(required_version, "min_mesh_llm_version")?;
    let current = parse_version(current_version, "current mesh-llm version")?;
    if current < required {
        bail!(
            "🚫 Plugin requires mesh-llm >= {}, current host is {}",
            required_version,
            current_version
        );
    }
    Ok(())
}

fn parse_version(raw: &str, label: &str) -> Result<Version> {
    let normalized = raw.trim().strip_prefix('v').unwrap_or_else(|| raw.trim());
    Version::parse(normalized).with_context(|| format!("invalid {label}: {raw}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_missing_min_mesh_llm_version() {
        let manifest = PluginTomlManifest::default();

        validate_plugin_manifest_compatibility(&manifest, "0.68.0").unwrap();
    }

    #[test]
    fn accepts_current_host_at_or_above_minimum() {
        let manifest = PluginTomlManifest {
            min_mesh_llm_version: Some("v0.68.0".to_string()),
        };

        validate_plugin_manifest_compatibility(&manifest, "0.68.0").unwrap();
        validate_plugin_manifest_compatibility(&manifest, "0.69.0").unwrap();
    }

    #[test]
    fn rejects_current_host_below_minimum() {
        let manifest = PluginTomlManifest {
            min_mesh_llm_version: Some("0.69.0".to_string()),
        };

        let err = validate_plugin_manifest_compatibility(&manifest, "0.68.0")
            .expect_err("host below minimum should fail");

        assert!(
            err.to_string()
                .contains("🚫 Plugin requires mesh-llm >= 0.69.0")
        );
    }

    #[test]
    fn parses_aliases() {
        let manifest: PluginTomlManifest = toml::from_str(r#"min_version = "0.68.0""#).unwrap();

        assert_eq!(manifest.min_mesh_llm_version.as_deref(), Some("0.68.0"));
    }
}
