use std::{any::type_name, collections::BTreeMap, path::PathBuf};

use anyhow::{Result, bail};
use mesh_llm_a2a::MeshAgent;
use mesh_llm_config::AgentProtocol;
use serde::Serialize;

pub const MESH_ACP_BRIDGE_NAME: &str = "mesh-llm-acp-bridge";

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AcpBridgePlan {
    pub agent_id: String,
    pub command: String,
    pub args: Vec<String>,
    pub working_directory: Option<PathBuf>,
    pub environment: BTreeMap<String, String>,
    pub client_role_type: &'static str,
    pub agent_role_type: &'static str,
}

impl AcpBridgePlan {
    pub fn from_agent(agent: &MeshAgent) -> Result<Self> {
        if agent.protocol != AgentProtocol::Acp {
            bail!("agent `{}` is {:?}, not acp", agent.id, agent.protocol);
        }
        let command = agent
            .command
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("agent `{}` has no ACP command", agent.id))?;
        Ok(Self {
            agent_id: agent.id.clone(),
            command: command.to_string(),
            args: agent.args.clone(),
            working_directory: None,
            environment: bridge_environment(agent),
            client_role_type: type_name::<agent_client_protocol::Client>(),
            agent_role_type: type_name::<agent_client_protocol::Agent>(),
        })
    }
}

fn bridge_environment(agent: &MeshAgent) -> BTreeMap<String, String> {
    let mut environment = BTreeMap::new();
    environment.insert("MESH_LLM_AGENT_ID".to_string(), agent.id.clone());
    environment.insert("MESH_LLM_AGENT_NAME".to_string(), agent.name.clone());
    environment.insert(
        "MESH_LLM_ACP_BRIDGE".to_string(),
        MESH_ACP_BRIDGE_NAME.to_string(),
    );
    environment
}

#[cfg(test)]
mod tests {
    use mesh_llm_a2a::AgentDirectory;

    use super::*;

    #[test]
    fn builds_acp_launch_plan_from_configured_agent() {
        let config = mesh_llm_config::parse_config_toml(
            r#"
version = 1

[[agent]]
id = "codex"
name = "Codex"
description = "Local coding agent"
protocol = "acp"
command = "codex"
args = ["--model", "gpt-5"]
"#,
        )
        .unwrap();
        let directory = AgentDirectory::from_config(&config).unwrap();
        let agent = directory.get("codex").unwrap();

        let plan = AcpBridgePlan::from_agent(agent).unwrap();

        assert_eq!(plan.command, "codex");
        assert_eq!(plan.args, ["--model", "gpt-5"]);
        assert_eq!(plan.environment["MESH_LLM_AGENT_ID"], "codex");
        assert!(plan.client_role_type.contains("Client"));
        assert!(plan.agent_role_type.contains("Agent"));
    }

    #[test]
    fn rejects_non_acp_agent() {
        let config = mesh_llm_config::parse_config_toml(
            r#"
version = 1

[[agent]]
id = "remote"
name = "Remote"
description = "Remote A2A agent"
protocol = "a2a"
endpoint = "https://agents.example.com/remote"
"#,
        )
        .unwrap();
        let directory = AgentDirectory::from_config(&config).unwrap();
        let agent = directory.get("remote").unwrap();

        let err = AcpBridgePlan::from_agent(agent).unwrap_err().to_string();

        assert!(err.contains("not acp"));
    }
}
