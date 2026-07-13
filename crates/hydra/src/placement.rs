use crate::vast::{
    VastTriggerConfig, VastTriggerRequest, VastTriggerResponse, send_vast_trigger,
};
use anyhow::{Context, Result, bail};
use chrono::Utc;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fs::{self, File};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};

const MANIFEST_FILE: &str = "manifest.json";
const FILE_PAYLOAD: &str = "payload.bin";
const DIRECTORY_PAYLOAD: &str = "payload";
const S3_PAYLOAD_OBJECT: &str = "payload.tar";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    ModelWeights,
    LayerPackage,
    KvState,
    RecurrentState,
    ActivationFrame,
}

impl ArtifactKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ModelWeights => "model_weights",
            Self::LayerPackage => "layer_package",
            Self::KvState => "kv_state",
            Self::RecurrentState => "recurrent_state",
            Self::ActivationFrame => "activation_frame",
        }
    }
}

impl fmt::Display for ArtifactKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ArtifactKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "model_weights" | "model-weights" | "weights" => Ok(Self::ModelWeights),
            "layer_package" | "layer-package" | "layers" => Ok(Self::LayerPackage),
            "kv_state" | "kv-state" | "kv" | "kv_cache" | "kv-cache" => Ok(Self::KvState),
            "recurrent_state" | "recurrent-state" => Ok(Self::RecurrentState),
            "activation_frame" | "activation-frame" | "activation" => Ok(Self::ActivationFrame),
            other => bail!("unknown placement artifact kind: {other}"),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct PlacementCompatibility {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_range: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abi: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype_layout: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_config_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_prefix_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub flags: Vec<String>,
}

impl PlacementCompatibility {
    pub fn exact_cache_matches(&self, expected: &Self) -> bool {
        exact_field_matches(&self.model_id, &expected.model_id)
            && exact_field_matches(&self.tokenizer_hash, &expected.tokenizer_hash)
            && exact_field_matches(&self.template_hash, &expected.template_hash)
            && exact_field_matches(&self.topology, &expected.topology)
            && exact_field_matches(&self.stage_id, &expected.stage_id)
            && exact_field_matches(&self.layer_range, &expected.layer_range)
            && exact_field_matches(&self.abi, &expected.abi)
            && exact_field_matches(&self.dtype_layout, &expected.dtype_layout)
            && exact_field_matches(&self.context_config_hash, &expected.context_config_hash)
            && exact_field_matches(&self.token_prefix_hash, &expected.token_prefix_hash)
            && self.flags == expected.flags
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PlacementManifest {
    pub schema_version: u32,
    pub kind: ArtifactKind,
    pub artifact_id: String,
    pub checksum_blake3: String,
    pub byte_size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_shape: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_node: Option<String>,
    pub created_at_unix_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_secs: Option<u64>,
    #[serde(default)]
    pub compatibility: PlacementCompatibility,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl PlacementManifest {
    pub fn is_exact_cache_compatible(&self, expected: &PlacementCompatibility) -> bool {
        matches!(
            self.kind,
            ArtifactKind::KvState | ArtifactKind::RecurrentState | ArtifactKind::ActivationFrame
        ) && self.compatibility.exact_cache_matches(expected)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PlacementProviderConfig {
    Posix {
        root: PathBuf,
    },
    S3 {
        endpoint: String,
        bucket: String,
        #[serde(default)]
        prefix: Option<String>,
        #[serde(default)]
        authorization_header: Option<String>,
    },
}

impl PlacementProviderConfig {
    pub fn provider(&self) -> Box<dyn ArtifactPlacementProvider> {
        match self {
            Self::Posix { root } => Box::new(PosixNamespaceProvider::new(root.clone())),
            Self::S3 {
                endpoint,
                bucket,
                prefix,
                authorization_header,
            } => Box::new(S3NamespaceProvider::new(
                endpoint.clone(),
                bucket.clone(),
                prefix.clone(),
                authorization_header.clone(),
            )),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PlacementPrefetchRequest {
    pub kind: ArtifactKind,
    pub artifact_id: String,
    pub source_path: PathBuf,
    pub provider: PlacementProviderConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_node: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_shape: Option<String>,
    #[serde(default)]
    pub compatibility: PlacementCompatibility,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_secs: Option<u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vast_trigger: Option<VastTriggerConfig>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PlacementPinRequest {
    pub artifact_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PlacementEvictRequest {
    pub artifact_id: String,
    #[serde(default)]
    pub kind: Option<ArtifactKind>,
    #[serde(default)]
    pub provider: Option<PlacementProviderConfig>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementOperationStatus {
    Running,
    Completed,
    Failed,
    Pinned,
    Evicted,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PlacementOperationSnapshot {
    pub operation_id: String,
    pub artifact_id: String,
    pub kind: ArtifactKind,
    pub status: PlacementOperationStatus,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest: Option<PlacementManifest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vast_trigger: Option<VastTriggerResponse>,
    pub updated_at_unix_ms: u64,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct PlacementCacheSnapshot {
    pub operations: Vec<PlacementOperationSnapshot>,
}

pub trait ArtifactPlacementProvider: Send + Sync {
    fn publish(
        &self,
        source: &Path,
        request: &PlacementPrefetchRequest,
    ) -> Result<PlacementManifest>;
    fn fetch(&self, manifest: &PlacementManifest, destination: &Path) -> Result<()>;
    fn evict(&self, kind: ArtifactKind, artifact_id: &str) -> Result<()>;
    fn manifest(&self, kind: ArtifactKind, artifact_id: &str) -> Result<PlacementManifest>;
}

#[derive(Clone, Debug, Default)]
pub struct PlacementManager {
    operations: Arc<Mutex<HashMap<String, PlacementOperationSnapshot>>>,
}

impl PlacementManager {
    pub fn prefetch(&self, request: PlacementPrefetchRequest) -> Result<PlacementOperationSnapshot> {
        let operation_id = operation_id(&request.artifact_id);
        let running = PlacementOperationSnapshot {
            operation_id: operation_id.clone(),
            artifact_id: request.artifact_id.clone(),
            kind: request.kind,
            status: PlacementOperationStatus::Running,
            message: "prefetch started".to_string(),
            manifest: None,
            vast_trigger: None,
            updated_at_unix_ms: now_ms(),
        };
        self.upsert(running.clone());

        let provider = request.provider.provider();
        match provider.publish(&request.source_path, &request) {
            Ok(manifest) => {
                let vast_trigger = match request.vast_trigger.as_ref() {
                    Some(config) => {
                        let trigger_request = VastTriggerRequest::from_prefetch_operation(
                            &operation_id,
                            &request,
                            &manifest,
                            config,
                        );
                        match send_vast_trigger(config, &trigger_request) {
                            Ok(response) => Some(response),
                            Err(error) => {
                                let failed = PlacementOperationSnapshot {
                                    operation_id,
                                    artifact_id: request.artifact_id,
                                    kind: request.kind,
                                    status: PlacementOperationStatus::Failed,
                                    message: format!(
                                        "artifact published but VAST trigger failed: {error}"
                                    ),
                                    manifest: Some(manifest),
                                    vast_trigger: None,
                                    updated_at_unix_ms: now_ms(),
                                };
                                self.upsert(failed.clone());
                                return Err(error).with_context(|| {
                                    serde_json::to_string(&failed).unwrap_or_default()
                                });
                            }
                        }
                    }
                    None => None,
                };
                let message = if vast_trigger
                    .as_ref()
                    .map(|response| response.triggered)
                    .unwrap_or(false)
                {
                    "artifact published, manifest committed, and VAST trigger accepted"
                        .to_string()
                } else {
                    "artifact published and manifest committed".to_string()
                };
                let completed = PlacementOperationSnapshot {
                    operation_id,
                    artifact_id: request.artifact_id,
                    kind: request.kind,
                    status: PlacementOperationStatus::Completed,
                    message,
                    manifest: Some(manifest),
                    vast_trigger,
                    updated_at_unix_ms: now_ms(),
                };
                self.upsert(completed.clone());
                Ok(completed)
            }
            Err(error) => {
                let failed = PlacementOperationSnapshot {
                    operation_id,
                    artifact_id: request.artifact_id,
                    kind: request.kind,
                    status: PlacementOperationStatus::Failed,
                    message: error.to_string(),
                    manifest: None,
                    vast_trigger: None,
                    updated_at_unix_ms: now_ms(),
                };
                self.upsert(failed.clone());
                Err(error).with_context(|| serde_json::to_string(&failed).unwrap_or_default())
            }
        }
    }

    pub fn status(&self, operation_id: &str) -> Option<PlacementOperationSnapshot> {
        self.operations.lock().unwrap().get(operation_id).cloned()
    }

    pub fn cache_snapshot(&self) -> PlacementCacheSnapshot {
        let mut operations = self
            .operations
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        operations.sort_by(|left, right| {
            right
                .updated_at_unix_ms
                .cmp(&left.updated_at_unix_ms)
                .then_with(|| left.artifact_id.cmp(&right.artifact_id))
        });
        PlacementCacheSnapshot { operations }
    }

    pub fn pin(&self, request: PlacementPinRequest) -> PlacementOperationSnapshot {
        let mut operations = self.operations.lock().unwrap();
        let pinned = operations
            .values_mut()
            .find(|operation| operation.artifact_id == request.artifact_id)
            .map(|operation| {
                operation.status = PlacementOperationStatus::Pinned;
                operation.message = "artifact pinned in placement cache".to_string();
                operation.updated_at_unix_ms = now_ms();
                operation.clone()
            });
        pinned.unwrap_or_else(|| {
            let operation = PlacementOperationSnapshot {
                operation_id: operation_id(&request.artifact_id),
                artifact_id: request.artifact_id,
                kind: ArtifactKind::ModelWeights,
                status: PlacementOperationStatus::Pinned,
                message: "pin recorded; artifact manifest has not been observed locally".to_string(),
                manifest: None,
                vast_trigger: None,
                updated_at_unix_ms: now_ms(),
            };
            operations.insert(operation.operation_id.clone(), operation.clone());
            operation
        })
    }

    pub fn evict(&self, request: PlacementEvictRequest) -> Result<PlacementOperationSnapshot> {
        if let (Some(kind), Some(provider)) = (request.kind, request.provider.as_ref()) {
            provider.provider().evict(kind, &request.artifact_id)?;
        }

        let mut operations = self.operations.lock().unwrap();
        let evicted = operations
            .values_mut()
            .find(|operation| operation.artifact_id == request.artifact_id)
            .map(|operation| {
                operation.status = PlacementOperationStatus::Evicted;
                operation.message = "artifact evicted from placement cache".to_string();
                operation.updated_at_unix_ms = now_ms();
                operation.clone()
            });
        Ok(evicted.unwrap_or_else(|| {
            let operation = PlacementOperationSnapshot {
                operation_id: operation_id(&request.artifact_id),
                artifact_id: request.artifact_id,
                kind: request.kind.unwrap_or(ArtifactKind::ModelWeights),
                status: PlacementOperationStatus::Evicted,
                message: "evict recorded; artifact manifest has not been observed locally"
                    .to_string(),
                manifest: None,
                vast_trigger: None,
                updated_at_unix_ms: now_ms(),
            };
            operations.insert(operation.operation_id.clone(), operation.clone());
            operation
        }))
    }

    fn upsert(&self, operation: PlacementOperationSnapshot) {
        self.operations
            .lock()
            .unwrap()
            .insert(operation.operation_id.clone(), operation);
    }
}

#[derive(Clone, Debug)]
pub struct PosixNamespaceProvider {
    root: PathBuf,
}

impl PosixNamespaceProvider {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    fn artifact_dir(&self, kind: ArtifactKind, artifact_id: &str) -> PathBuf {
        self.root.join(kind.as_str()).join(sanitize(artifact_id))
    }
}

impl ArtifactPlacementProvider for PosixNamespaceProvider {
    fn publish(
        &self,
        source: &Path,
        request: &PlacementPrefetchRequest,
    ) -> Result<PlacementManifest> {
        if !source.exists() {
            bail!("placement source does not exist: {}", source.display());
        }
        fs::create_dir_all(&self.root)
            .with_context(|| format!("creating placement root {}", self.root.display()))?;
        let staging_root = self.root.join(".staging");
        fs::create_dir_all(&staging_root)
            .with_context(|| format!("creating staging root {}", staging_root.display()))?;
        let staging_dir = staging_root.join(operation_id(&request.artifact_id));
        if staging_dir.exists() {
            fs::remove_dir_all(&staging_dir)
                .with_context(|| format!("removing stale staging dir {}", staging_dir.display()))?;
        }
        fs::create_dir_all(&staging_dir)
            .with_context(|| format!("creating staging dir {}", staging_dir.display()))?;

        let payload_path = if source.is_dir() {
            let destination = staging_dir.join(DIRECTORY_PAYLOAD);
            copy_dir(source, &destination)?;
            destination
        } else {
            let destination = staging_dir.join(FILE_PAYLOAD);
            fs::copy(source, &destination).with_context(|| {
                format!(
                    "copying placement payload {} to {}",
                    source.display(),
                    destination.display()
                )
            })?;
            destination
        };
        let digest = digest_path(&payload_path)?;
        let manifest = PlacementManifest {
            schema_version: 1,
            kind: request.kind,
            artifact_id: request.artifact_id.clone(),
            checksum_blake3: digest.checksum_blake3,
            byte_size: digest.byte_size,
            payload_shape: request.payload_shape.clone().or_else(|| {
                if source.is_dir() {
                    Some("directory".to_string())
                } else {
                    Some("file".to_string())
                }
            }),
            source_node: request.source_node.clone(),
            created_at_unix_ms: now_ms(),
            ttl_secs: request.ttl_secs,
            compatibility: request.compatibility.clone(),
            metadata: request.metadata.clone(),
        };
        write_manifest(&staging_dir.join(MANIFEST_FILE), &manifest)?;

        let final_dir = self.artifact_dir(request.kind, &request.artifact_id);
        if let Some(parent) = final_dir.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating placement kind dir {}", parent.display()))?;
        }
        if final_dir.exists() {
            let existing = self.manifest(request.kind, &request.artifact_id).ok();
            if existing
                .as_ref()
                .map(|existing| existing.checksum_blake3 == manifest.checksum_blake3)
                .unwrap_or(false)
            {
                fs::remove_dir_all(&staging_dir).ok();
                return Ok(manifest);
            }
            fs::remove_dir_all(&final_dir).with_context(|| {
                format!("replacing existing placement dir {}", final_dir.display())
            })?;
        }
        fs::rename(&staging_dir, &final_dir).with_context(|| {
            format!(
                "publishing placement artifact {} to {}",
                request.artifact_id,
                final_dir.display()
            )
        })?;
        Ok(manifest)
    }

    fn fetch(&self, manifest: &PlacementManifest, destination: &Path) -> Result<()> {
        let artifact_dir = self.artifact_dir(manifest.kind, &manifest.artifact_id);
        let current = self.manifest(manifest.kind, &manifest.artifact_id)?;
        if current.checksum_blake3 != manifest.checksum_blake3 {
            bail!(
                "manifest checksum mismatch for {}: expected {}, found {}",
                manifest.artifact_id,
                manifest.checksum_blake3,
                current.checksum_blake3
            );
        }
        let file_payload = artifact_dir.join(FILE_PAYLOAD);
        if file_payload.exists() {
            if destination.is_dir() {
                fs::copy(&file_payload, destination.join(FILE_PAYLOAD)).with_context(|| {
                    format!(
                        "copying placement payload {} to {}",
                        file_payload.display(),
                        destination.display()
                    )
                })?;
            } else {
                if let Some(parent) = destination.parent() {
                    fs::create_dir_all(parent).with_context(|| {
                        format!("creating placement fetch dir {}", parent.display())
                    })?;
                }
                fs::copy(&file_payload, destination).with_context(|| {
                    format!(
                        "copying placement payload {} to {}",
                        file_payload.display(),
                        destination.display()
                    )
                })?;
            }
            return Ok(());
        }
        copy_dir(&artifact_dir.join(DIRECTORY_PAYLOAD), destination)
    }

    fn evict(&self, kind: ArtifactKind, artifact_id: &str) -> Result<()> {
        let artifact_dir = self.artifact_dir(kind, artifact_id);
        if artifact_dir.exists() {
            fs::remove_dir_all(&artifact_dir)
                .with_context(|| format!("evicting placement artifact {}", artifact_dir.display()))?;
        }
        Ok(())
    }

    fn manifest(&self, kind: ArtifactKind, artifact_id: &str) -> Result<PlacementManifest> {
        let manifest_path = self.artifact_dir(kind, artifact_id).join(MANIFEST_FILE);
        let file = File::open(&manifest_path)
            .with_context(|| format!("opening placement manifest {}", manifest_path.display()))?;
        serde_json::from_reader(file)
            .with_context(|| format!("parsing placement manifest {}", manifest_path.display()))
    }
}

#[derive(Clone, Debug)]
pub struct S3NamespaceProvider {
    endpoint: String,
    bucket: String,
    prefix: Option<String>,
    authorization_header: Option<String>,
    client: Client,
}

impl S3NamespaceProvider {
    pub fn new(
        endpoint: String,
        bucket: String,
        prefix: Option<String>,
        authorization_header: Option<String>,
    ) -> Self {
        Self {
            endpoint,
            bucket,
            prefix,
            authorization_header,
            client: Client::new(),
        }
    }

    fn object_url(&self, kind: ArtifactKind, artifact_id: &str, object: &str) -> String {
        let mut parts = vec![self.bucket.as_str()];
        if let Some(prefix) = self.prefix.as_deref() {
            parts.extend(prefix.trim_matches('/').split('/').filter(|part| !part.is_empty()));
        }
        let artifact = sanitize(artifact_id);
        parts.push(kind.as_str());
        parts.push(&artifact);
        parts.push(object);
        format!(
            "{}/{}",
            self.endpoint.trim_end_matches('/'),
            parts.join("/")
        )
    }

    fn put_bytes(&self, url: String, bytes: Vec<u8>) -> Result<()> {
        let mut request = self.client.put(url);
        if let Some(header) = self.authorization_header.as_deref() {
            request = request.header("authorization", header);
        }
        request
            .body(bytes)
            .send()
            .context("sending S3-compatible placement PUT")?
            .error_for_status()
            .context("S3-compatible placement PUT failed")?;
        Ok(())
    }

    fn get_bytes(&self, url: String) -> Result<Vec<u8>> {
        let mut request = self.client.get(url);
        if let Some(header) = self.authorization_header.as_deref() {
            request = request.header("authorization", header);
        }
        let bytes = request
            .send()
            .context("sending S3-compatible placement GET")?
            .error_for_status()
            .context("S3-compatible placement GET failed")?
            .bytes()
            .context("reading S3-compatible placement GET body")?;
        Ok(bytes.to_vec())
    }
}

impl ArtifactPlacementProvider for S3NamespaceProvider {
    fn publish(
        &self,
        source: &Path,
        request: &PlacementPrefetchRequest,
    ) -> Result<PlacementManifest> {
        if !source.exists() {
            bail!("placement source does not exist: {}", source.display());
        }
        let bytes = if source.is_dir() {
            tar_directory(source)?
        } else {
            fs::read(source)
                .with_context(|| format!("reading placement source {}", source.display()))?
        };
        let checksum_blake3 = blake3::hash(&bytes).to_hex().to_string();
        let manifest = PlacementManifest {
            schema_version: 1,
            kind: request.kind,
            artifact_id: request.artifact_id.clone(),
            checksum_blake3,
            byte_size: bytes.len() as u64,
            payload_shape: request.payload_shape.clone().or_else(|| {
                if source.is_dir() {
                    Some("tar".to_string())
                } else {
                    Some("file".to_string())
                }
            }),
            source_node: request.source_node.clone(),
            created_at_unix_ms: now_ms(),
            ttl_secs: request.ttl_secs,
            compatibility: request.compatibility.clone(),
            metadata: request.metadata.clone(),
        };
        self.put_bytes(
            self.object_url(request.kind, &request.artifact_id, S3_PAYLOAD_OBJECT),
            bytes,
        )?;
        self.put_bytes(
            self.object_url(request.kind, &request.artifact_id, MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        Ok(manifest)
    }

    fn fetch(&self, manifest: &PlacementManifest, destination: &Path) -> Result<()> {
        let bytes = self.get_bytes(self.object_url(
            manifest.kind,
            &manifest.artifact_id,
            S3_PAYLOAD_OBJECT,
        ))?;
        let checksum = blake3::hash(&bytes).to_hex().to_string();
        if checksum != manifest.checksum_blake3 {
            bail!(
                "S3-compatible placement payload checksum mismatch for {}",
                manifest.artifact_id
            );
        }
        if manifest.payload_shape.as_deref() == Some("tar") {
            fs::create_dir_all(destination)
                .with_context(|| format!("creating fetch dir {}", destination.display()))?;
            tar::Archive::new(Cursor::new(bytes))
                .unpack(destination)
                .with_context(|| format!("unpacking placement tar to {}", destination.display()))?;
        } else {
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("creating fetch dir {}", parent.display()))?;
            }
            fs::write(destination, bytes)
                .with_context(|| format!("writing placement payload {}", destination.display()))?;
        }
        Ok(())
    }

    fn evict(&self, kind: ArtifactKind, artifact_id: &str) -> Result<()> {
        for object in [S3_PAYLOAD_OBJECT, MANIFEST_FILE] {
            let mut request = self.client.delete(self.object_url(kind, artifact_id, object));
            if let Some(header) = self.authorization_header.as_deref() {
                request = request.header("authorization", header);
            }
            request
                .send()
                .context("sending S3-compatible placement DELETE")?
                .error_for_status()
                .context("S3-compatible placement DELETE failed")?;
        }
        Ok(())
    }

    fn manifest(&self, kind: ArtifactKind, artifact_id: &str) -> Result<PlacementManifest> {
        let bytes = self.get_bytes(self.object_url(kind, artifact_id, MANIFEST_FILE))?;
        serde_json::from_slice(&bytes).context("parsing S3-compatible placement manifest")
    }
}

#[derive(Clone, Debug)]
struct PayloadDigest {
    checksum_blake3: String,
    byte_size: u64,
}

fn copy_dir(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)
        .with_context(|| format!("creating directory {}", destination.display()))?;
    let mut entries = fs::read_dir(source)
        .with_context(|| format!("reading directory {}", source.display()))?
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("listing directory {}", source.display()))?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let entry_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if entry_path.is_dir() {
            copy_dir(&entry_path, &destination_path)?;
        } else {
            fs::copy(&entry_path, &destination_path).with_context(|| {
                format!(
                    "copying placement file {} to {}",
                    entry_path.display(),
                    destination_path.display()
                )
            })?;
        }
    }
    Ok(())
}

fn digest_path(path: &Path) -> Result<PayloadDigest> {
    let mut hasher = blake3::Hasher::new();
    let byte_size = if path.is_dir() {
        digest_dir(path, path, &mut hasher)?
    } else {
        digest_file(path, &mut hasher)?
    };
    Ok(PayloadDigest {
        checksum_blake3: hasher.finalize().to_hex().to_string(),
        byte_size,
    })
}

fn digest_dir(root: &Path, path: &Path, hasher: &mut blake3::Hasher) -> Result<u64> {
    let mut total = 0;
    let mut entries = fs::read_dir(path)
        .with_context(|| format!("reading directory {}", path.display()))?
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("listing directory {}", path.display()))?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let entry_path = entry.path();
        if entry_path.is_dir() {
            total += digest_dir(root, &entry_path, hasher)?;
        } else {
            let relative = entry_path.strip_prefix(root).unwrap_or(&entry_path);
            hasher.update(relative.to_string_lossy().as_bytes());
            hasher.update(&[0]);
            total += digest_file(&entry_path, hasher)?;
        }
    }
    Ok(total)
}

fn digest_file(path: &Path, hasher: &mut blake3::Hasher) -> Result<u64> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut buffer = [0_u8; 64 * 1024];
    let mut total = 0;
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("reading {}", path.display()))?;
        if read == 0 {
            break;
        }
        total += read as u64;
        hasher.update(&buffer[..read]);
    }
    Ok(total)
}

fn write_manifest(path: &Path, manifest: &PlacementManifest) -> Result<()> {
    let mut file =
        File::create(path).with_context(|| format!("creating manifest {}", path.display()))?;
    file.write_all(&serde_json::to_vec_pretty(manifest)?)
        .with_context(|| format!("writing manifest {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("syncing manifest {}", path.display()))?;
    Ok(())
}

fn tar_directory(source: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    {
        let mut builder = tar::Builder::new(&mut bytes);
        builder
            .append_dir_all(".", source)
            .with_context(|| format!("tarring placement directory {}", source.display()))?;
        builder.finish().context("finalizing placement tar")?;
    }
    Ok(bytes)
}

fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn exact_field_matches(left: &Option<String>, right: &Option<String>) -> bool {
    matches!((left.as_deref(), right.as_deref()), (Some(left), Some(right)) if left == right)
}

fn operation_id(artifact_id: &str) -> String {
    let hash = blake3::hash(artifact_id.as_bytes()).to_hex().to_string();
    format!("plc-{}-{}", now_ms(), &hash[..12])
}

fn now_ms() -> u64 {
    Utc::now().timestamp_millis().max(0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posix_publish_fetch_and_evict_file() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("weights.bin");
        fs::write(&source, b"hello weights").unwrap();
        let root = temp.path().join("namespace");
        let provider = PosixNamespaceProvider::new(root.clone());
        let request = PlacementPrefetchRequest {
            kind: ArtifactKind::ModelWeights,
            artifact_id: "model/a".to_string(),
            source_path: source.clone(),
            provider: PlacementProviderConfig::Posix { root },
            source_node: Some("node-a".to_string()),
            payload_shape: None,
            compatibility: PlacementCompatibility {
                model_id: Some("model-a".to_string()),
                ..PlacementCompatibility::default()
            },
            ttl_secs: Some(60),
            metadata: BTreeMap::new(),
            vast_trigger: None,
        };

        let manifest = provider.publish(&source, &request).unwrap();
        assert_eq!(manifest.kind, ArtifactKind::ModelWeights);
        assert_eq!(manifest.source_node.as_deref(), Some("node-a"));
        assert_eq!(manifest.byte_size, 13);

        let fetched = temp.path().join("fetched.bin");
        provider.fetch(&manifest, &fetched).unwrap();
        assert_eq!(fs::read(&fetched).unwrap(), b"hello weights");

        provider
            .evict(ArtifactKind::ModelWeights, &request.artifact_id)
            .unwrap();
        assert!(
            provider
                .manifest(ArtifactKind::ModelWeights, &request.artifact_id)
                .is_err()
        );
    }

    #[test]
    fn manager_tracks_failed_and_completed_operations() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("layer.pkg");
        fs::write(&source, b"layer").unwrap();
        let manager = PlacementManager::default();
        let completed = manager
            .prefetch(PlacementPrefetchRequest {
                kind: ArtifactKind::LayerPackage,
                artifact_id: "layer-0".to_string(),
                source_path: source,
                provider: PlacementProviderConfig::Posix {
                    root: temp.path().join("namespace"),
                },
                source_node: None,
                payload_shape: None,
                compatibility: PlacementCompatibility::default(),
                ttl_secs: None,
                metadata: BTreeMap::new(),
                vast_trigger: None,
            })
            .unwrap();
        assert_eq!(completed.status, PlacementOperationStatus::Completed);
        assert_eq!(
            manager
                .status(&completed.operation_id)
                .unwrap()
                .manifest
                .unwrap()
                .artifact_id,
            "layer-0"
        );
        assert_eq!(manager.cache_snapshot().operations.len(), 1);
    }

    #[test]
    fn exact_cache_compatibility_rejects_partial_identity() {
        let expected = PlacementCompatibility {
            model_id: Some("model-a".to_string()),
            tokenizer_hash: Some("tok".to_string()),
            template_hash: Some("tmpl".to_string()),
            topology: Some("topo".to_string()),
            stage_id: Some("stage-0".to_string()),
            layer_range: Some("0..8".to_string()),
            abi: Some("stage-abi-0.1/native-kv-page-v1".to_string()),
            dtype_layout: Some("f16/contiguous".to_string()),
            context_config_hash: Some("ctx".to_string()),
            token_prefix_hash: Some("prefix".to_string()),
            flags: vec!["portable".to_string()],
        };
        let manifest = PlacementManifest {
            schema_version: 1,
            kind: ArtifactKind::KvState,
            artifact_id: "kv".to_string(),
            checksum_blake3: "abc".to_string(),
            byte_size: 1,
            payload_shape: Some("native-kv-page".to_string()),
            source_node: None,
            created_at_unix_ms: 1,
            ttl_secs: None,
            compatibility: expected.clone(),
            metadata: BTreeMap::new(),
        };

        assert!(manifest.is_exact_cache_compatible(&expected));

        let mut mismatch = expected;
        mismatch.token_prefix_hash = Some("other-prefix".to_string());
        assert!(!manifest.is_exact_cache_compatible(&mismatch));
        assert!(!PlacementManifest {
            kind: ArtifactKind::ModelWeights,
            ..manifest
        }
        .is_exact_cache_compatible(&mismatch));
    }
}
