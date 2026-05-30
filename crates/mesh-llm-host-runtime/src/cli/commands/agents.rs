use anyhow::{Context, Result, bail};
use mesh_llm_a2a::{AgentDirectory, MeshAgent};
use mesh_llm_acp_bridge::AcpBridgePlan;
use serde::Serialize;

use crate::cli::{AgentCommand, Cli};
use crate::plugin;

pub(crate) fn run_agents_command(command: &AgentCommand, cli: &Cli) -> Result<()> {
    let directory = load_directory(cli)?;
    match command {
        AgentCommand::List { json } => list_agents(&directory, *json),
        AgentCommand::Show { id, json } => show_agent(&directory, id, *json),
        AgentCommand::Validate { json } => validate_agents(&directory, *json),
    }
}

fn load_directory(cli: &Cli) -> Result<AgentDirectory> {
    let config = plugin::load_config(cli.config.as_deref())?;
    AgentDirectory::from_config(&config).context("invalid agent directory")
}

fn list_agents(directory: &AgentDirectory, json: bool) -> Result<()> {
    if json {
        print_json(&directory.summary())
    } else {
        print_agent_table(directory.agents());
        Ok(())
    }
}

fn show_agent(directory: &AgentDirectory, id: &str, json: bool) -> Result<()> {
    let agent = directory
        .get(id)
        .ok_or_else(|| anyhow::anyhow!("agent `{id}` is not configured"))?;
    if json {
        print_json(&AgentDetail::from_agent(agent)?)
    } else {
        print_agent_detail(agent)
    }
}

fn validate_agents(directory: &AgentDirectory, json: bool) -> Result<()> {
    let report = ValidationReport {
        status: "ok",
        total: directory.agents().len(),
        enabled: directory.enabled_agents().count(),
        a2a_cards: directory.a2a_cards()?.len(),
        acp_bridge_plans: directory
            .enabled_agents()
            .filter(|agent| agent.protocol == mesh_llm_config::AgentProtocol::Acp)
            .map(AcpBridgePlan::from_agent)
            .collect::<Result<Vec<_>>>()?
            .len(),
    };

    if json {
        print_json(&report)
    } else {
        println!(
            "Agent directory ok: {} configured, {} enabled, {} A2A cards, {} ACP bridge plans",
            report.total, report.enabled, report.a2a_cards, report.acp_bridge_plans
        );
        Ok(())
    }
}

fn print_agent_table(agents: &[MeshAgent]) {
    if agents.is_empty() {
        println!("No agents configured.");
        return;
    }
    println!("ID\tPROTOCOL\tENABLED\tTARGET\tNAME");
    for agent in agents {
        println!(
            "{}\t{}\t{}\t{}\t{}",
            agent.id,
            protocol_label(agent.protocol),
            agent.enabled,
            agent_target(agent),
            agent.name
        );
    }
}

fn print_agent_detail(agent: &MeshAgent) -> Result<()> {
    println!("id: {}", agent.id);
    println!("name: {}", agent.name);
    println!("description: {}", agent.description);
    println!("protocol: {}", protocol_label(agent.protocol));
    println!("enabled: {}", agent.enabled);
    println!("target: {}", agent_target(agent));
    if !agent.skills.is_empty() {
        println!("skills: {}", agent.skills.join(", "));
    }
    if agent.protocol == mesh_llm_config::AgentProtocol::Acp {
        let plan = AcpBridgePlan::from_agent(agent)?;
        println!("bridge: {}", plan.command);
    }
    Ok(())
}

fn agent_target(agent: &MeshAgent) -> String {
    agent
        .endpoint
        .clone()
        .or_else(|| agent.command.clone())
        .unwrap_or_else(|| "-".to_string())
}

fn protocol_label(protocol: mesh_llm_config::AgentProtocol) -> &'static str {
    match protocol {
        mesh_llm_config::AgentProtocol::A2a => "a2a",
        mesh_llm_config::AgentProtocol::Acp => "acp",
    }
}

fn print_json<T: Serialize>(value: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

#[derive(Serialize)]
struct ValidationReport {
    status: &'static str,
    total: usize,
    enabled: usize,
    a2a_cards: usize,
    acp_bridge_plans: usize,
}

#[derive(Serialize)]
struct AgentDetail<'a> {
    agent: &'a MeshAgent,
    a2a_card: Option<serde_json::Value>,
    acp_bridge_plan: Option<AcpBridgePlan>,
}

impl<'a> AgentDetail<'a> {
    fn from_agent(agent: &'a MeshAgent) -> Result<Self> {
        let a2a_card = match agent.protocol {
            mesh_llm_config::AgentProtocol::A2a => {
                Some(serde_json::to_value(agent.to_a2a_card()?)?)
            }
            mesh_llm_config::AgentProtocol::Acp => None,
        };
        let acp_bridge_plan = match agent.protocol {
            mesh_llm_config::AgentProtocol::A2a => None,
            mesh_llm_config::AgentProtocol::Acp => Some(AcpBridgePlan::from_agent(agent)?),
        };
        if a2a_card.is_none() && acp_bridge_plan.is_none() {
            bail!("agent `{}` has no supported protocol projection", agent.id);
        }
        Ok(Self {
            agent,
            a2a_card,
            acp_bridge_plan,
        })
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    fn cli_with_config(path: &std::path::Path) -> Cli {
        use clap::Parser;

        Cli::parse_from([
            "mesh-llm",
            "--config",
            path.to_str().unwrap(),
            "agents",
            "list",
        ])
    }

    #[test]
    fn loads_agent_directory_from_cli_config() {
        let temp = TempDir::new().unwrap();
        let config_path = temp.path().join("config.toml");
        std::fs::write(
            &config_path,
            r#"
version = 1

[[agent]]
id = "codex"
name = "Codex"
description = "Local coding agent"
protocol = "acp"
command = "codex"
"#,
        )
        .unwrap();

        let cli = cli_with_config(&config_path);
        let directory = load_directory(&cli).unwrap();

        assert_eq!(directory.summary().total, 1);
        assert_eq!(directory.summary().acp, 1);
    }
}
