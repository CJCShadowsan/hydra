use crate::placement::{
    ArtifactKind, PlacementManifest, PlacementPrefetchRequest, PlacementProviderConfig,
};
use anyhow::{Context, Result, bail};
use chrono::Utc;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Duration;

const VAST_TRIGGER_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VastTriggerMode {
    #[default]
    Disabled,
    Webhook,
    DataEngineWebhook,
}

impl VastTriggerMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Webhook => "webhook",
            Self::DataEngineWebhook => "data_engine_webhook",
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VastTriggerConfig {
    #[serde(default)]
    pub mode: VastTriggerMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorization_header: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataspace: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_namespace: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub destination_namespace: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub target_sites: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
}

impl VastTriggerConfig {
    pub fn data_engine_webhook(endpoint: impl Into<String>) -> Self {
        Self {
            mode: VastTriggerMode::DataEngineWebhook,
            endpoint: Some(endpoint.into()),
            ..Self::default()
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.mode != VastTriggerMode::Disabled || self.endpoint.is_some()
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VastProviderLocation {
    pub provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posix_root: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub s3_endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub s3_bucket: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub s3_prefix: Option<String>,
}

impl VastProviderLocation {
    pub fn from_provider(provider: &PlacementProviderConfig) -> Self {
        match provider {
            PlacementProviderConfig::Posix { root } => Self {
                provider: "posix".to_string(),
                posix_root: Some(root.display().to_string()),
                s3_endpoint: None,
                s3_bucket: None,
                s3_prefix: None,
            },
            PlacementProviderConfig::S3 {
                endpoint,
                bucket,
                prefix,
                authorization_header: _,
            } => Self {
                provider: "s3".to_string(),
                posix_root: None,
                s3_endpoint: Some(endpoint.clone()),
                s3_bucket: Some(bucket.clone()),
                s3_prefix: prefix.clone(),
            },
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VastTriggerRequest {
    pub schema_version: u32,
    pub operation_id: String,
    pub artifact_kind: ArtifactKind,
    pub artifact_id: String,
    pub manifest: PlacementManifest,
    pub provider_location: VastProviderLocation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataspace: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_namespace: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub destination_namespace: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub target_sites: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl VastTriggerRequest {
    pub fn from_prefetch_operation(
        operation_id: &str,
        request: &PlacementPrefetchRequest,
        manifest: &PlacementManifest,
        config: &VastTriggerConfig,
    ) -> Self {
        Self {
            schema_version: VAST_TRIGGER_SCHEMA_VERSION,
            operation_id: operation_id.to_string(),
            artifact_kind: request.kind,
            artifact_id: request.artifact_id.clone(),
            manifest: manifest.clone(),
            provider_location: VastProviderLocation::from_provider(&request.provider),
            source_path: Some(request.source_path.display().to_string()),
            tenant: config.tenant.clone(),
            dataspace: config.dataspace.clone(),
            source_namespace: config.source_namespace.clone(),
            destination_namespace: config.destination_namespace.clone(),
            target_sites: config.target_sites.clone(),
            metadata: request.metadata.clone(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VastTriggerResponse {
    pub triggered: bool,
    pub mode: VastTriggerMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status_code: Option<u16>,
    pub message: String,
    pub triggered_at_unix_ms: u64,
}

pub fn send_vast_trigger(
    config: &VastTriggerConfig,
    request: &VastTriggerRequest,
) -> Result<VastTriggerResponse> {
    if !config.is_enabled() {
        return Ok(VastTriggerResponse {
            triggered: false,
            mode: VastTriggerMode::Disabled,
            endpoint: None,
            status_code: None,
            message: "VAST trigger disabled".to_string(),
            triggered_at_unix_ms: now_ms(),
        });
    }

    let endpoint = config
        .endpoint
        .as_deref()
        .filter(|endpoint| !endpoint.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("VAST trigger endpoint is required when trigger is enabled"))?;
    if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
        bail!("VAST trigger endpoint must be http:// or https://: {endpoint}");
    }
    let mode = match config.mode {
        VastTriggerMode::Disabled => VastTriggerMode::Webhook,
        mode => mode,
    };

    let client = Client::builder()
        .timeout(Duration::from_secs(config.timeout_secs.unwrap_or(30)))
        .build()
        .context("building VAST trigger HTTP client")?;
    let mut builder = client
        .post(endpoint)
        .header("content-type", "application/json")
        .header("x-mesh-llm-vast-trigger-mode", mode.as_str())
        .json(request);
    if let Some(header) = config.authorization_header.as_deref() {
        builder = builder.header("authorization", header);
    }
    for (name, value) in &config.headers {
        if !name.eq_ignore_ascii_case("authorization") {
            builder = builder.header(name.as_str(), value.as_str());
        }
    }

    let response = builder
        .send()
        .with_context(|| format!("sending VAST trigger to {endpoint}"))?;
    let status = response.status();
    let body = response
        .text()
        .context("reading VAST trigger response body")?;
    if !status.is_success() {
        bail!("VAST trigger returned {status}: {body}");
    }

    Ok(VastTriggerResponse {
        triggered: true,
        mode,
        endpoint: Some(endpoint.to_string()),
        status_code: Some(status.as_u16()),
        message: if body.trim().is_empty() {
            "VAST trigger accepted".to_string()
        } else {
            body
        },
        triggered_at_unix_ms: now_ms(),
    })
}

fn now_ms() -> u64 {
    Utc::now().timestamp_millis().max(0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_trigger_is_a_noop() {
        let manifest = PlacementManifest {
            schema_version: 1,
            kind: ArtifactKind::ModelWeights,
            artifact_id: "model".to_string(),
            checksum_blake3: "abc".to_string(),
            byte_size: 1,
            payload_shape: None,
            source_node: None,
            created_at_unix_ms: 1,
            ttl_secs: None,
            compatibility: Default::default(),
            metadata: BTreeMap::new(),
        };
        let request = VastTriggerRequest {
            schema_version: 1,
            operation_id: "op".to_string(),
            artifact_kind: ArtifactKind::ModelWeights,
            artifact_id: "model".to_string(),
            manifest,
            provider_location: VastProviderLocation {
                provider: "posix".to_string(),
                posix_root: Some("/vast/global".to_string()),
                s3_endpoint: None,
                s3_bucket: None,
                s3_prefix: None,
            },
            source_path: None,
            tenant: None,
            dataspace: None,
            source_namespace: None,
            destination_namespace: None,
            target_sites: Vec::new(),
            metadata: BTreeMap::new(),
        };

        let response = send_vast_trigger(&VastTriggerConfig::default(), &request).unwrap();
        assert!(!response.triggered);
        assert_eq!(response.mode, VastTriggerMode::Disabled);
    }
}
