mod auth;
mod benchmark;
mod discover;
mod download;
mod gpus;
mod model_package;
mod models;
mod plugin;
mod plugin_cli;
mod runtime;
mod update;

use anyhow::Result;

use crate::cli::commands::benchmark::dispatch_benchmark_command;
use crate::cli::commands::discover::{DiscoverOptions, run_discover, run_stop};
use crate::cli::commands::download::dispatch_download_command;
use crate::cli::commands::gpus::dispatch_gpu_command;
use crate::cli::commands::models::dispatch_models_command;
use crate::cli::commands::plugin::run_plugin_command;
use crate::cli::commands::plugin_cli::{run_external_plugin_command, run_named_plugin_command};
use crate::cli::commands::runtime::{dispatch_runtime_command, run_drop, run_load, run_status};
use crate::cli::commands::update::run_update;
use crate::cli::{AuthCommand, Cli, Command};
use crate::network::nostr;

pub(crate) async fn dispatch(cli: &Cli) -> Result<bool> {
    let Some(cmd) = cli.command.as_ref() else {
        return Ok(false);
    };
    dispatch_command(cli, cmd).await?;
    Ok(true)
}

async fn dispatch_command(cli: &Cli, cmd: &Command) -> Result<()> {
    match cmd {
        Command::Auth { command } => dispatch_auth_command(command),
        Command::ModelPrepare { .. } => dispatch_model_prepare(cmd).await,
        Command::Blackboard { .. } => dispatch_blackboard_command(cli, cmd).await,
        _ => dispatch_general_command(cli, cmd).await,
    }
}

async fn dispatch_general_command(cli: &Cli, cmd: &Command) -> Result<()> {
    match cmd {
        Command::Models { command } => {
            dispatch_models_command(command).await?;
            Ok(())
        }
        Command::Download { name, draft } => {
            dispatch_download_command(name.as_deref(), *draft).await
        }
        Command::Update { .. } => run_update(cli).await,
        Command::Gpus { json, command } => {
            dispatch_gpu_command(*json, command.as_ref())?;
            Ok(())
        }
        Command::Runtime { command } => dispatch_runtime_command(command.as_ref()).await,
        Command::Load { name, port } => run_load(name, *port).await,
        Command::Unload { name, port } => run_drop(name, *port).await,
        Command::Status { port } => run_status(*port).await,
        Command::Stop => run_stop(),
        Command::Discover {
            name,
            model,
            min_vram,
            region,
            auto,
            relay,
        } => {
            run_discover(DiscoverOptions {
                name: name.clone(),
                model: model.clone(),
                min_vram_gb: *min_vram,
                region: region.clone(),
                auto_join: *auto,
                relays: relay.clone(),
                discovery_mode: cli.mesh_discovery_mode,
                supplied_join_tokens: cli.join.clone(),
            })
            .await
        }
        Command::RotateKey => nostr::rotate_keys(),
        Command::Goose { model, port } => {
            run_named_plugin_command(cli, "goose", model_port_plugin_args(model, *port)).await
        }
        Command::Claude { model, port } => {
            run_named_plugin_command(cli, "claude", model_port_plugin_args(model, *port)).await
        }
        Command::Pi { model, host, write } => {
            run_named_plugin_command(cli, "pi", host_write_plugin_args(model, host, *write)).await
        }
        Command::Opencode { model, host, write } => {
            run_named_plugin_command(cli, "opencode", host_write_plugin_args(model, host, *write))
                .await
        }
        Command::Blackboard { .. } => dispatch_blackboard_command(cli, cmd).await,
        Command::Plugin { command } => run_plugin_command(command, cli).await,
        Command::Benchmark { command } => dispatch_benchmark_command(command).await,
        Command::ModelPrepare { .. } => dispatch_model_prepare(cmd).await,
        Command::Auth { command } => dispatch_auth_command(command),
        Command::ExternalPlugin(args) => run_external_plugin_command(cli, args).await,
    }
}

fn model_port_plugin_args(model: &Option<String>, port: u16) -> Vec<String> {
    let mut args = Vec::new();
    if let Some(model) = model {
        args.extend(["--model".to_string(), model.clone()]);
    }
    args.extend(["--port".to_string(), port.to_string()]);
    args
}

fn host_write_plugin_args(model: &Option<String>, host: &str, write: bool) -> Vec<String> {
    let mut args = Vec::new();
    if let Some(model) = model {
        args.extend(["--model".to_string(), model.clone()]);
    }
    args.extend(["--host".to_string(), host.to_string()]);
    if write {
        args.push("--write".to_string());
    }
    args
}

async fn dispatch_blackboard_command(cli: &Cli, cmd: &Command) -> Result<()> {
    let Command::Blackboard {
        text,
        search,
        from,
        since,
        limit,
        port,
        mcp,
    } = cmd
    else {
        unreachable!("dispatch_blackboard_command called for non-blackboard command");
    };

    if *mcp {
        return crate::runtime::run_plugin_mcp(cli).await;
    }
    run_named_plugin_command(
        cli,
        crate::plugin::BLACKBOARD_PLUGIN_ID,
        blackboard_plugin_args(text, search, from, *since, *limit, *port),
    )
    .await
}

fn blackboard_plugin_args(
    text: &Option<String>,
    search: &Option<String>,
    from: &Option<String>,
    since: Option<f64>,
    limit: usize,
    port: u16,
) -> Vec<String> {
    let mut args = Vec::new();
    if let Some(text) = text {
        args.push(text.clone());
    }
    if let Some(search) = search {
        args.extend(["--search".to_string(), search.clone()]);
    }
    if let Some(from) = from {
        args.extend(["--from".to_string(), from.clone()]);
    }
    if let Some(since) = since {
        args.extend(["--since".to_string(), since.to_string()]);
    }
    args.extend(["--limit".to_string(), limit.to_string()]);
    args.extend(["--port".to_string(), port.to_string()]);
    args
}

async fn dispatch_model_prepare(cmd: &Command) -> Result<()> {
    let Command::ModelPrepare {
        source_repo,
        quant,
        target,
        model_id,
        flavor,
        timeout,
        mesh_llm_ref,
        dry_run,
        confirm,
        follow,
        json,
        status,
        logs,
        cancel,
        list,
        update_script,
    } = cmd
    else {
        unreachable!("dispatch_model_prepare called for non-model-prepare command");
    };

    model_package::dispatch_model_package(model_package::ModelPrepareArgs {
        source_repo: source_repo.as_deref(),
        quant: quant.as_deref(),
        target: target.as_deref(),
        model_id: model_id.as_deref(),
        flavor,
        timeout,
        mesh_llm_ref,
        dry_run: *dry_run,
        confirm: *confirm,
        follow: *follow,
        json: *json,
        status: status.as_deref(),
        logs: logs.as_deref(),
        cancel: cancel.as_deref(),
        list: *list,
        update_script: *update_script,
    })
    .await
}

fn dispatch_auth_command(command: &AuthCommand) -> Result<()> {
    match command {
        AuthCommand::Init {
            owner_key,
            force,
            no_passphrase,
            keychain,
        } => auth::run_init(owner_key.clone(), *force, *no_passphrase, *keychain),
        AuthCommand::Status {
            owner_key,
            node_key,
            node_ownership,
            trust_store,
        } => auth::run_status(
            owner_key.clone(),
            node_key.clone(),
            node_ownership.clone(),
            trust_store.clone(),
        ),
        AuthCommand::SignNode {
            owner_key,
            node_key,
            out,
            hostname_hint,
            node_label,
            expires_in_hours,
        } => auth::run_sign_node(
            owner_key.clone(),
            node_key.clone(),
            out.clone(),
            node_label.clone(),
            hostname_hint.clone(),
            *expires_in_hours,
        ),
        AuthCommand::RenewNode {
            owner_key,
            node_key,
            out,
            hostname_hint,
            node_label,
            expires_in_hours,
        } => auth::run_renew_node(
            owner_key.clone(),
            node_key.clone(),
            out.clone(),
            node_label.clone(),
            hostname_hint.clone(),
            *expires_in_hours,
        ),
        AuthCommand::VerifyNode {
            file,
            node_id,
            trust_store,
            trust_policy,
        } => auth::run_verify_node(
            file.clone(),
            node_id.clone(),
            trust_store.clone(),
            *trust_policy,
        ),
        AuthCommand::RotateNode {
            owner_key,
            node_key,
            out,
            hostname_hint,
            node_label,
            expires_in_hours,
            revoke_current,
            reason,
            trust_store,
        } => auth::run_rotate_node(
            owner_key.clone(),
            node_key.clone(),
            out.clone(),
            node_label.clone(),
            hostname_hint.clone(),
            *expires_in_hours,
            *revoke_current,
            reason.clone(),
            trust_store.clone(),
        ),
        AuthCommand::RevokeOwner {
            owner_id,
            reason,
            trust_store,
        } => auth::run_revoke_owner(owner_id.clone(), reason.clone(), trust_store.clone()),
        AuthCommand::RevokeNode {
            cert_id,
            node_id,
            reason,
            trust_store,
        } => auth::run_revoke_node(
            cert_id.clone(),
            node_id.clone(),
            reason.clone(),
            trust_store.clone(),
        ),
        AuthCommand::RotateOwner {
            owner_key,
            no_passphrase,
            force,
        } => auth::run_rotate_owner(owner_key.clone(), *no_passphrase, *force),
        AuthCommand::Trust { command } => auth::run_trust_command(command),
    }
}
