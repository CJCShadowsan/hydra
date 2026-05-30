use std::collections::BTreeSet;

use a2a::{
    AgentCapabilities, AgentCard, AgentInterface, AgentSkill, TRANSPORT_PROTOCOL_HTTP_JSON,
    TRANSPORT_PROTOCOL_JSONRPC,
};
use anyhow::{Result, bail};
use mesh_llm_config::{AgentConfigEntry, AgentProtocol, MeshConfig};
use serde::Serialize;
use url::Url;

pub const MESH_AGENT_CARD_PROVIDER: &str = "Mesh LLM";
pub const A2A_JSONRPC_TRANSPORT: &str = TRANSPORT_PROTOCOL_JSONRPC;
pub const A2A_HTTP_JSON_TRANSPORT: &str = TRANSPORT_PROTOCOL_HTTP_JSON;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MeshAgent {
    pub id: String,
    pub name: String,
    pub description: String,
    pub protocol: AgentProtocol,
    pub enabled: bool,
    pub endpoint: Option<String>,
    pub command: Option<String>,
    pub args: Vec<String>,
    pub skills: Vec<String>,
    pub input_modes: Vec<String>,
    pub output_modes: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct AgentDirectory {
    agents: Vec<MeshAgent>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AgentDirectorySummary {
    pub total: usize,
    pub enabled: usize,
    pub a2a: usize,
    pub acp: usize,
    pub agents: Vec<MeshAgentSummary>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MeshAgentSummary {
    pub id: String,
    pub name: String,
    pub protocol: AgentProtocol,
    pub enabled: bool,
    pub endpoint: Option<String>,
    pub command: Option<String>,
    pub skills: Vec<String>,
}

impl AgentDirectory {
    pub fn from_config(config: &MeshConfig) -> Result<Self> {
        let mut seen = BTreeSet::new();
        let mut agents = Vec::with_capacity(config.agents.len());
        for entry in &config.agents {
            let agent = MeshAgent::from_entry(entry)?;
            if !seen.insert(agent.id.clone()) {
                bail!("duplicate agent id `{}`", agent.id);
            }
            agents.push(agent);
        }
        Ok(Self { agents })
    }

    pub fn agents(&self) -> &[MeshAgent] {
        &self.agents
    }

    pub fn enabled_agents(&self) -> impl Iterator<Item = &MeshAgent> {
        self.agents.iter().filter(|agent| agent.enabled)
    }

    pub fn get(&self, id: &str) -> Option<&MeshAgent> {
        self.agents.iter().find(|agent| agent.id == id)
    }

    pub fn summary(&self) -> AgentDirectorySummary {
        AgentDirectorySummary {
            total: self.agents.len(),
            enabled: self.enabled_agents().count(),
            a2a: self
                .agents
                .iter()
                .filter(|agent| agent.protocol == AgentProtocol::A2a)
                .count(),
            acp: self
                .agents
                .iter()
                .filter(|agent| agent.protocol == AgentProtocol::Acp)
                .count(),
            agents: self.agents.iter().map(MeshAgent::summary).collect(),
        }
    }

    pub fn a2a_cards(&self) -> Result<Vec<AgentCard>> {
        self.enabled_agents()
            .filter(|agent| agent.protocol == AgentProtocol::A2a)
            .map(MeshAgent::to_a2a_card)
            .collect()
    }
}

impl MeshAgent {
    fn from_entry(entry: &AgentConfigEntry) -> Result<Self> {
        let id = normalize_agent_id(&entry.id)?;
        let name = normalize_required(&entry.name, "agent name")?.to_string();
        let description = normalize_required(&entry.description, "agent description")?.to_string();
        let skills = normalize_string_list(&entry.skills, "agent skills")?;
        let input_modes = normalize_modes(&entry.input_modes);
        let output_modes = normalize_modes(&entry.output_modes);
        let enabled = entry.enabled.unwrap_or(true);
        validate_protocol_binding(entry)?;

        Ok(Self {
            id,
            name,
            description,
            protocol: entry.protocol,
            enabled,
            endpoint: normalize_optional_string(entry.endpoint.as_deref()),
            command: normalize_optional_string(entry.command.as_deref()),
            args: entry.args.clone(),
            skills,
            input_modes,
            output_modes,
        })
    }

    pub fn summary(&self) -> MeshAgentSummary {
        MeshAgentSummary {
            id: self.id.clone(),
            name: self.name.clone(),
            protocol: self.protocol,
            enabled: self.enabled,
            endpoint: self.endpoint.clone(),
            command: self.command.clone(),
            skills: self.skills.clone(),
        }
    }

    pub fn to_a2a_card(&self) -> Result<AgentCard> {
        if self.protocol != AgentProtocol::A2a {
            bail!("agent `{}` is {:?}, not a2a", self.id, self.protocol);
        }
        let endpoint = self
            .endpoint
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("agent `{}` has no A2A endpoint", self.id))?;
        validate_http_endpoint(endpoint, "agent endpoint")?;
        Ok(AgentCard {
            name: self.name.clone(),
            description: self.description.clone(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            supported_interfaces: vec![AgentInterface::new(endpoint, A2A_JSONRPC_TRANSPORT)],
            capabilities: AgentCapabilities {
                streaming: Some(true),
                push_notifications: Some(false),
                extensions: None,
                extended_agent_card: Some(false),
            },
            default_input_modes: self.input_modes.clone(),
            default_output_modes: self.output_modes.clone(),
            skills: self
                .skills
                .iter()
                .map(|skill| AgentSkill {
                    id: skill.clone(),
                    name: skill.clone(),
                    description: format!("Mesh agent skill `{skill}`"),
                    tags: vec![skill.clone()],
                    examples: None,
                    input_modes: Some(self.input_modes.clone()),
                    output_modes: Some(self.output_modes.clone()),
                    security_requirements: None,
                })
                .collect(),
            provider: Some(a2a::AgentProvider {
                organization: MESH_AGENT_CARD_PROVIDER.to_string(),
                url: "https://github.com/Mesh-LLM/mesh-llm".to_string(),
            }),
            documentation_url: None,
            icon_url: None,
            security_schemes: None,
            security_requirements: None,
            signatures: None,
        })
    }
}

fn normalize_agent_id(value: &str) -> Result<String> {
    let value = normalize_required(value, "agent id")?;
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
    {
        Ok(value.to_string())
    } else {
        bail!("agent id `{value}` must contain only ASCII letters, numbers, '.', '_' or '-'")
    }
}

fn normalize_required<'a>(value: &'a str, label: &str) -> Result<&'a str> {
    let value = value.trim();
    if value.is_empty() {
        bail!("{label} must not be empty");
    }
    Ok(value)
}

fn normalize_optional_string(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn normalize_string_list(values: &[String], label: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for value in values {
        let value = normalize_required(value, label)?;
        if !out.iter().any(|existing| existing == value) {
            out.push(value.to_string());
        }
    }
    Ok(out)
}

fn normalize_modes(values: &[String]) -> Vec<String> {
    let modes: Vec<String> = values
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    if modes.is_empty() {
        vec!["text".to_string()]
    } else {
        modes
    }
}

fn validate_protocol_binding(entry: &AgentConfigEntry) -> Result<()> {
    match entry.protocol {
        AgentProtocol::A2a => {
            let endpoint = entry
                .endpoint
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("agent `{}` requires endpoint for a2a", entry.id))?;
            validate_http_endpoint(endpoint, "agent endpoint")?;
        }
        AgentProtocol::Acp => {
            let command = entry
                .command
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("agent `{}` requires command for acp", entry.id))?;
            normalize_required(command, "agent command")?;
        }
    }
    Ok(())
}

fn validate_http_endpoint(value: &str, label: &str) -> Result<()> {
    let url =
        Url::parse(value).map_err(|err| anyhow::anyhow!("invalid {label} `{value}`: {err}"))?;
    match url.scheme() {
        "http" | "https" => Ok(()),
        scheme => bail!("{label} `{value}` must use http or https, got {scheme}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(raw: &str) -> MeshConfig {
        mesh_llm_config::parse_config_toml(raw).unwrap()
    }

    #[test]
    fn loads_enabled_a2a_agent_card_from_config() {
        let config = config(
            r#"
version = 1

[[agent]]
id = "coder"
name = "Coder"
description = "Writes code through the mesh"
protocol = "a2a"
endpoint = "http://127.0.0.1:3131/agents/coder"
skills = ["coding", "tools"]
"#,
        );

        let directory = AgentDirectory::from_config(&config).unwrap();
        let cards = directory.a2a_cards().unwrap();

        assert_eq!(directory.summary().enabled, 1);
        assert_eq!(cards[0].name, "Coder");
        assert_eq!(
            cards[0].supported_interfaces[0].protocol_binding,
            A2A_JSONRPC_TRANSPORT
        );
        assert_eq!(cards[0].skills.len(), 2);
    }

    #[test]
    fn rejects_duplicate_agent_ids() {
        let config = MeshConfig {
            agents: vec![
                AgentConfigEntry {
                    id: "coder".to_string(),
                    name: "Coder".to_string(),
                    description: "one".to_string(),
                    protocol: AgentProtocol::Acp,
                    command: Some("codex".to_string()),
                    ..agent_defaults()
                },
                AgentConfigEntry {
                    id: "coder".to_string(),
                    name: "Coder 2".to_string(),
                    description: "two".to_string(),
                    protocol: AgentProtocol::Acp,
                    command: Some("goose".to_string()),
                    ..agent_defaults()
                },
            ],
            ..MeshConfig::default()
        };

        let err = AgentDirectory::from_config(&config)
            .unwrap_err()
            .to_string();
        assert!(err.contains("duplicate agent id"));
    }

    fn agent_defaults() -> AgentConfigEntry {
        AgentConfigEntry {
            id: String::new(),
            name: String::new(),
            description: String::new(),
            protocol: AgentProtocol::Acp,
            enabled: None,
            endpoint: None,
            command: None,
            args: Vec::new(),
            skills: Vec::new(),
            input_modes: Vec::new(),
            output_modes: Vec::new(),
        }
    }

    #[test]
    fn validates_protocol_specific_entry_shape() {
        let err = mesh_llm_config::parse_config_toml(
            r#"
version = 1

[[agent]]
id = "remote"
name = "Remote"
description = "Bad endpoint"
protocol = "a2a"
endpoint = "file:///tmp/socket"
"#,
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("http or https"));
    }
}
