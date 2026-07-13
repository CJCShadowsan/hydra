use anyhow::{Context, Result, bail};
use mesh_llm_cli::PlacementCommand;
use hydra::{
    ArtifactKind, PlacementEvictRequest, PlacementPinRequest, PlacementPrefetchRequest,
    PlacementProviderConfig, VastTriggerConfig, VastTriggerMode,
};
use std::collections::BTreeMap;
use std::str::FromStr;

pub async fn dispatch_placement_command(command: &PlacementCommand) -> Result<()> {
    match command {
        PlacementCommand::Prefetch {
            kind,
            artifact_id,
            source,
            posix_root,
            vast_trigger_endpoint,
            vast_trigger_auth_header,
            vast_tenant,
            vast_dataspace,
            vast_source_namespace,
            vast_destination_namespace,
            vast_target_sites,
            vast_trigger_timeout_secs,
            port,
        } => {
            let vast_trigger = vast_trigger_endpoint.as_ref().map(|endpoint| VastTriggerConfig {
                mode: VastTriggerMode::DataEngineWebhook,
                endpoint: Some(endpoint.clone()),
                authorization_header: vast_trigger_auth_header.clone(),
                tenant: vast_tenant.clone(),
                dataspace: vast_dataspace.clone(),
                source_namespace: vast_source_namespace.clone(),
                destination_namespace: vast_destination_namespace.clone(),
                target_sites: vast_target_sites.clone(),
                timeout_secs: *vast_trigger_timeout_secs,
                headers: BTreeMap::new(),
            });
            let request = PlacementPrefetchRequest {
                kind: ArtifactKind::from_str(kind)?,
                artifact_id: artifact_id.clone(),
                source_path: source.clone(),
                provider: PlacementProviderConfig::Posix {
                    root: posix_root.clone(),
                },
                source_node: None,
                payload_shape: None,
                compatibility: hydra::PlacementCompatibility::default(),
                ttl_secs: None,
                metadata: BTreeMap::new(),
                vast_trigger,
            };
            post_json(*port, "/api/placement/prefetch", &request).await
        }
        PlacementCommand::Status { operation_id, port } => {
            get_json(
                *port,
                &format!(
                    "/api/placement/status/{}",
                    urlencoding::encode(operation_id)
                ),
            )
            .await
        }
        PlacementCommand::Cache { port } => get_json(*port, "/api/placement/cache").await,
        PlacementCommand::Pin { artifact_id, port } => {
            post_json(
                *port,
                "/api/placement/pin",
                &PlacementPinRequest {
                    artifact_id: artifact_id.clone(),
                },
            )
            .await
        }
        PlacementCommand::Evict {
            artifact_id,
            kind,
            posix_root,
            port,
        } => {
            if posix_root.is_some() && kind.is_none() {
                bail!("--kind is required when --posix-root is provided for placement evict");
            }
            let request = PlacementEvictRequest {
                artifact_id: artifact_id.clone(),
                kind: kind.as_deref().map(ArtifactKind::from_str).transpose()?,
                provider: posix_root.as_ref().map(|root| PlacementProviderConfig::Posix {
                    root: root.clone(),
                }),
            };
            post_json(*port, "/api/placement/evict", &request).await
        }
    }
}

async fn get_json(port: u16, path: &str) -> Result<()> {
    let url = local_api_url(port, path);
    let response = reqwest::Client::new()
        .get(&url)
        .send()
        .await
        .with_context(|| format!("calling {url}"))?;
    print_json_response(response).await
}

async fn post_json<T: serde::Serialize>(port: u16, path: &str, body: &T) -> Result<()> {
    let url = local_api_url(port, path);
    let response = reqwest::Client::new()
        .post(&url)
        .json(body)
        .send()
        .await
        .with_context(|| format!("calling {url}"))?;
    print_json_response(response).await
}

async fn print_json_response(response: reqwest::Response) -> Result<()> {
    let status = response.status();
    let text = response.text().await.context("reading response body")?;
    if !status.is_success() {
        bail!("placement API returned {status}: {text}");
    }
    let value: serde_json::Value = serde_json::from_str(&text).context("parsing response JSON")?;
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

fn local_api_url(port: u16, path: &str) -> String {
    format!("http://127.0.0.1:{port}{path}")
}
