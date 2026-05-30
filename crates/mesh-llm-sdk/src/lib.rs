use std::collections::BTreeMap;
use std::net::IpAddr;
use std::path::PathBuf;
#[cfg(any(feature = "client", feature = "serve"))]
use std::time::Duration;

#[cfg(any(feature = "client", feature = "serve"))]
use anyhow::Result;

#[cfg(feature = "console")]
pub mod console {
    pub use mesh_llm_console_server::{
        ConsoleServerHandle, ConsoleServerOptions, start_file_console,
    };
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MeshDiscoveryMode {
    #[default]
    Nostr,
    Mdns,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LogFormat {
    Pretty,
    #[default]
    Json,
}

#[derive(Clone, Debug)]
pub struct HttpConfig {
    pub api_port: u16,
    pub console_port: u16,
    pub console_ui: bool,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            api_port: 9337,
            console_port: 3131,
            console_ui: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NetworkConfig {
    pub join_tokens: Vec<String>,
    pub auto_join: bool,
    pub discovery_mode: MeshDiscoveryMode,
    pub publish: bool,
    pub mesh_name: Option<String>,
    pub region: Option<String>,
    pub node_name: Option<String>,
    pub iroh_relays: Vec<String>,
    pub iroh_relay_auth: BTreeMap<String, String>,
    pub nostr_relays: Vec<String>,
    pub bind_ip: Option<IpAddr>,
    pub bind_port: Option<u16>,
    pub listen_all: bool,
    pub enumerate_host: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            join_tokens: Vec::new(),
            auto_join: false,
            discovery_mode: MeshDiscoveryMode::Nostr,
            publish: false,
            mesh_name: None,
            region: None,
            node_name: None,
            iroh_relays: Vec::new(),
            iroh_relay_auth: BTreeMap::new(),
            nostr_relays: Vec::new(),
            bind_ip: None,
            bind_port: None,
            listen_all: false,
            enumerate_host: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StorageConfig {
    pub config_path: Option<PathBuf>,
    pub isolated_config: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            config_path: None,
            isolated_config: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ServingConfig {
    pub models: Vec<String>,
    pub max_vram_gb: Option<f64>,
}

#[cfg(any(feature = "client", feature = "serve"))]
macro_rules! impl_common_builder_methods {
    () => {
        pub fn api_port(mut self, port: u16) -> Self {
            self.config.http.api_port = port;
            self
        }

        pub fn console_port(mut self, port: u16) -> Self {
            self.config.http.console_port = port;
            self
        }

        pub fn console_ui(mut self, enabled: bool) -> Self {
            self.config.http.console_ui = enabled;
            self
        }

        pub fn join_token(mut self, token: impl Into<String>) -> Self {
            self.config.network.join_tokens.push(token.into());
            self
        }

        pub fn join_tokens<I, S>(mut self, tokens: I) -> Self
        where
            I: IntoIterator<Item = S>,
            S: Into<String>,
        {
            self.config.network.join_tokens = tokens.into_iter().map(Into::into).collect();
            self
        }

        pub fn auto_join(mut self, enabled: bool) -> Self {
            self.config.network.auto_join = enabled;
            self
        }

        pub fn discovery_mode(mut self, mode: MeshDiscoveryMode) -> Self {
            self.config.network.discovery_mode = mode;
            self
        }

        pub fn publish(mut self, enabled: bool) -> Self {
            self.config.network.publish = enabled;
            self
        }

        pub fn mesh_name(mut self, name: impl Into<String>) -> Self {
            self.config.network.mesh_name = Some(name.into());
            self
        }

        pub fn region(mut self, region: impl Into<String>) -> Self {
            self.config.network.region = Some(region.into());
            self
        }

        pub fn node_name(mut self, name: impl Into<String>) -> Self {
            self.config.network.node_name = Some(name.into());
            self
        }

        pub fn iroh_relay(mut self, url: impl Into<String>) -> Self {
            self.config.network.iroh_relays.push(url.into());
            self
        }

        pub fn iroh_relays<I, S>(mut self, urls: I) -> Self
        where
            I: IntoIterator<Item = S>,
            S: Into<String>,
        {
            self.config.network.iroh_relays = urls.into_iter().map(Into::into).collect();
            self
        }

        pub fn iroh_relay_auth(
            mut self,
            relay_url: impl Into<String>,
            bearer_token: impl Into<String>,
        ) -> Self {
            self.config
                .network
                .iroh_relay_auth
                .insert(relay_url.into(), bearer_token.into());
            self
        }

        pub fn nostr_relay(mut self, url: impl Into<String>) -> Self {
            self.config.network.nostr_relays.push(url.into());
            self
        }

        pub fn nostr_relays<I, S>(mut self, urls: I) -> Self
        where
            I: IntoIterator<Item = S>,
            S: Into<String>,
        {
            self.config.network.nostr_relays = urls.into_iter().map(Into::into).collect();
            self
        }

        pub fn bind_ip(mut self, ip: IpAddr) -> Self {
            self.config.network.bind_ip = Some(ip);
            self
        }

        pub fn bind_port(mut self, port: u16) -> Self {
            self.config.network.bind_port = Some(port);
            self
        }

        pub fn listen_all(mut self, enabled: bool) -> Self {
            self.config.network.listen_all = enabled;
            self
        }

        pub fn enumerate_host(mut self, enabled: bool) -> Self {
            self.config.network.enumerate_host = enabled;
            self
        }

        pub fn config_path(mut self, path: impl Into<PathBuf>) -> Self {
            self.config.storage.config_path = Some(path.into());
            self
        }

        pub fn isolated_config(mut self, enabled: bool) -> Self {
            self.config.storage.isolated_config = enabled;
            self
        }

        pub fn log_format(mut self, format: LogFormat) -> Self {
            self.config.log_format = format;
            self
        }

        pub fn startup_timeout(mut self, timeout: Duration) -> Self {
            self.config.startup_timeout = timeout;
            self
        }
    };
}

#[cfg(any(feature = "client", feature = "serve"))]
pub struct EmbeddedNodeHandle {
    inner: mesh_llm_host_runtime::sdk::EmbeddedMeshNodeHandle,
}

#[cfg(any(feature = "client", feature = "serve"))]
impl EmbeddedNodeHandle {
    pub fn api_base_url(&self) -> &str {
        self.inner.api_base_url()
    }

    pub fn console_url(&self) -> &str {
        self.inner.console_url()
    }

    pub fn invite_token(&self) -> Option<&str> {
        self.inner.invite_token()
    }

    pub async fn status(&self) -> Result<EmbeddedNodeStatus> {
        self.inner.status().await.map(EmbeddedNodeStatus::from)
    }

    pub async fn stop(self) -> Result<()> {
        self.inner.stop().await
    }
}

#[cfg(any(feature = "client", feature = "serve"))]
#[derive(Clone, Debug)]
pub struct EmbeddedNodeStatus {
    pub api_base_url: String,
    pub console_url: String,
    pub invite_token: Option<String>,
    pub payload: serde_json::Value,
}

#[cfg(any(feature = "client", feature = "serve"))]
impl From<mesh_llm_host_runtime::sdk::EmbeddedMeshNodeStatus> for EmbeddedNodeStatus {
    fn from(status: mesh_llm_host_runtime::sdk::EmbeddedMeshNodeStatus) -> Self {
        Self {
            api_base_url: status.api_base_url,
            console_url: status.console_url,
            invite_token: status.invite_token,
            payload: status.payload,
        }
    }
}

#[cfg(feature = "client")]
pub mod client {
    use super::*;

    #[derive(Clone, Debug)]
    pub struct EmbeddedClientConfig {
        pub http: HttpConfig,
        pub network: NetworkConfig,
        pub storage: StorageConfig,
        pub log_format: LogFormat,
        pub startup_timeout: Duration,
    }

    impl Default for EmbeddedClientConfig {
        fn default() -> Self {
            Self {
                http: HttpConfig::default(),
                network: NetworkConfig::default(),
                storage: StorageConfig::default(),
                log_format: LogFormat::default(),
                startup_timeout: Duration::from_secs(30),
            }
        }
    }

    impl EmbeddedClientConfig {
        pub fn builder() -> EmbeddedClientConfigBuilder {
            EmbeddedClientConfigBuilder::default()
        }
    }

    #[derive(Clone, Debug, Default)]
    pub struct EmbeddedClientConfigBuilder {
        config: EmbeddedClientConfig,
    }

    impl EmbeddedClientConfigBuilder {
        impl_common_builder_methods!();

        pub fn build(self) -> EmbeddedClientConfig {
            self.config
        }
    }

    pub async fn start(config: EmbeddedClientConfig) -> Result<EmbeddedNodeHandle> {
        start_embedded_node(EmbeddedNodeParts {
            mode: EmbeddedMode::Client,
            http: config.http,
            network: config.network,
            storage: config.storage,
            serving: ServingConfig::default(),
            log_format: config.log_format,
            startup_timeout: config.startup_timeout,
        })
        .await
    }
}

#[cfg(feature = "serve")]
pub mod serve {
    use super::*;

    #[derive(Clone, Debug)]
    pub struct EmbeddedServeConfig {
        pub http: HttpConfig,
        pub network: NetworkConfig,
        pub storage: StorageConfig,
        pub serving: ServingConfig,
        pub log_format: LogFormat,
        pub startup_timeout: Duration,
    }

    impl Default for EmbeddedServeConfig {
        fn default() -> Self {
            Self {
                http: HttpConfig::default(),
                network: NetworkConfig::default(),
                storage: StorageConfig::default(),
                serving: ServingConfig::default(),
                log_format: LogFormat::default(),
                startup_timeout: Duration::from_secs(30),
            }
        }
    }

    impl EmbeddedServeConfig {
        pub fn builder() -> EmbeddedServeConfigBuilder {
            EmbeddedServeConfigBuilder::default()
        }
    }

    #[derive(Clone, Debug, Default)]
    pub struct EmbeddedServeConfigBuilder {
        config: EmbeddedServeConfig,
    }

    impl EmbeddedServeConfigBuilder {
        impl_common_builder_methods!();

        pub fn model(mut self, model_ref: impl Into<String>) -> Self {
            self.config.serving.models.push(model_ref.into());
            self
        }

        pub fn models<I, S>(mut self, model_refs: I) -> Self
        where
            I: IntoIterator<Item = S>,
            S: Into<String>,
        {
            self.config.serving.models = model_refs.into_iter().map(Into::into).collect();
            self
        }

        pub fn max_vram_gb(mut self, max_vram_gb: f64) -> Self {
            self.config.serving.max_vram_gb = Some(max_vram_gb);
            self
        }

        pub fn build(self) -> EmbeddedServeConfig {
            self.config
        }
    }

    pub async fn start(config: EmbeddedServeConfig) -> Result<EmbeddedNodeHandle> {
        start_embedded_node(EmbeddedNodeParts {
            mode: EmbeddedMode::Serve,
            http: config.http,
            network: config.network,
            storage: config.storage,
            serving: config.serving,
            log_format: config.log_format,
            startup_timeout: config.startup_timeout,
        })
        .await
    }
}

#[cfg(any(feature = "client", feature = "serve"))]
#[derive(Clone, Copy, Debug)]
enum EmbeddedMode {
    #[cfg(feature = "serve")]
    Serve,
    #[cfg(feature = "client")]
    Client,
}

#[cfg(any(feature = "client", feature = "serve"))]
struct EmbeddedNodeParts {
    mode: EmbeddedMode,
    http: HttpConfig,
    network: NetworkConfig,
    storage: StorageConfig,
    serving: ServingConfig,
    log_format: LogFormat,
    startup_timeout: Duration,
}

#[cfg(any(feature = "client", feature = "serve"))]
async fn start_embedded_node(parts: EmbeddedNodeParts) -> Result<EmbeddedNodeHandle> {
    let handle = mesh_llm_host_runtime::sdk::start_embedded_node(host_config(parts)).await?;
    Ok(EmbeddedNodeHandle { inner: handle })
}

#[cfg(any(feature = "client", feature = "serve"))]
fn host_config(parts: EmbeddedNodeParts) -> mesh_llm_host_runtime::sdk::EmbeddedMeshNodeConfig {
    mesh_llm_host_runtime::sdk::EmbeddedMeshNodeConfig {
        mode: match parts.mode {
            #[cfg(feature = "serve")]
            EmbeddedMode::Serve => mesh_llm_host_runtime::sdk::EmbeddedMeshNodeMode::Serve,
            #[cfg(feature = "client")]
            EmbeddedMode::Client => mesh_llm_host_runtime::sdk::EmbeddedMeshNodeMode::Client,
        },
        http: mesh_llm_host_runtime::sdk::EmbeddedMeshHttpConfig {
            api_port: parts.http.api_port,
            console_port: parts.http.console_port,
            console_ui: parts.http.console_ui,
        },
        serving: mesh_llm_host_runtime::sdk::EmbeddedMeshServingConfig {
            models: parts.serving.models,
            max_vram_gb: parts.serving.max_vram_gb,
        },
        network: mesh_llm_host_runtime::sdk::EmbeddedMeshNetworkConfig {
            join_tokens: parts.network.join_tokens,
            auto_join: parts.network.auto_join,
            discovery_mode: match parts.network.discovery_mode {
                MeshDiscoveryMode::Nostr => {
                    mesh_llm_host_runtime::sdk::EmbeddedMeshDiscoveryMode::Nostr
                }
                MeshDiscoveryMode::Mdns => {
                    mesh_llm_host_runtime::sdk::EmbeddedMeshDiscoveryMode::Mdns
                }
            },
            publish: parts.network.publish,
            mesh_name: parts.network.mesh_name,
            region: parts.network.region,
            node_name: parts.network.node_name,
            iroh_relays: parts.network.iroh_relays,
            iroh_relay_auth: parts.network.iroh_relay_auth,
            nostr_relays: parts.network.nostr_relays,
            bind_ip: parts.network.bind_ip,
            bind_port: parts.network.bind_port,
            listen_all: parts.network.listen_all,
            enumerate_host: parts.network.enumerate_host,
        },
        storage: mesh_llm_host_runtime::sdk::EmbeddedMeshStorageConfig {
            config_path: parts.storage.config_path,
            isolated_config: parts.storage.isolated_config,
        },
        log_format: match parts.log_format {
            LogFormat::Pretty => mesh_llm_host_runtime::sdk::EmbeddedMeshLogFormat::Pretty,
            LogFormat::Json => mesh_llm_host_runtime::sdk::EmbeddedMeshLogFormat::Json,
        },
        startup_timeout: parts.startup_timeout,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "client")]
    fn client_builder_sets_network_fields() {
        let config = client::EmbeddedClientConfig::builder()
            .auto_join(true)
            .join_token("mesh-token")
            .api_port(19337)
            .console_port(13131)
            .discovery_mode(MeshDiscoveryMode::Mdns)
            .log_format(LogFormat::Pretty)
            .build();

        assert!(config.network.auto_join);
        assert_eq!(config.network.join_tokens, vec!["mesh-token"]);
        assert_eq!(config.http.api_port, 19337);
        assert_eq!(config.http.console_port, 13131);
        assert_eq!(config.network.discovery_mode, MeshDiscoveryMode::Mdns);
        assert_eq!(config.log_format, LogFormat::Pretty);
    }

    #[test]
    #[cfg(feature = "serve")]
    fn serve_builder_sets_model_fields() {
        let config = serve::EmbeddedServeConfig::builder()
            .model("unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
            .max_vram_gb(6.0)
            .mesh_name("sprout")
            .build();

        assert_eq!(
            config.serving.models,
            vec!["unsloth/Qwen3-0.6B-GGUF:Q4_K_M"]
        );
        assert_eq!(config.serving.max_vram_gb, Some(6.0));
        assert_eq!(config.network.mesh_name.as_deref(), Some("sprout"));
    }
}
