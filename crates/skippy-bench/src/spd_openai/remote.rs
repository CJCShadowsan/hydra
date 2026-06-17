use std::{
    collections::BTreeMap,
    fs::{self, File},
    net::{SocketAddr, TcpListener},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use serde_json::json;

use crate::{cli::SpdOpenAiSmokeArgs, support::ChildGuard};

use super::SmokeCase;

const LOCAL_STAGE_HOST: &str = "local";

pub(super) struct CaseDeployment {
    pub(super) stages: Vec<StageDeployment>,
    pub(super) openai_addr: SocketAddr,
    topology_path: PathBuf,
}

pub(super) struct StageDeployment {
    index: usize,
    host: String,
    local: bool,
    config_path: PathBuf,
    pub(super) log_path: PathBuf,
    remote_dir: Option<String>,
    remote_config_path: Option<String>,
    remote_topology_path: Option<String>,
    remote_log_path: Option<String>,
    remote_pid_path: Option<String>,
    remote_exit_path: Option<String>,
    remote_model_path: Option<String>,
}

impl StageDeployment {
    fn stage_id(&self) -> String {
        stage_id(self.index)
    }
}

pub(super) fn prepare_case_deployment(
    args: &SpdOpenAiSmokeArgs,
    case_dir: &Path,
    run_id: &str,
    stage_ranges: &[(u32, u32)],
    tap_allowlist: &[u32],
) -> Result<CaseDeployment> {
    let explicit_stage_hosts = !args.stage_hosts.is_empty();
    let endpoint_hosts = parse_string_map(args.endpoint_host_map.as_deref())?;
    let remote_model_paths = parse_string_map(args.remote_model_path_map.as_deref())?;
    let openai_addr = SocketAddr::from(([127, 0, 0, 1], allocate_ports(1)?[0]));
    let ports = if explicit_stage_hosts {
        explicit_stage_ports(args.stage_port_base, stage_ranges.len())?
    } else {
        allocate_ports(stage_ranges.len())?
    };
    let stage_hosts = if explicit_stage_hosts {
        normalized_stage_hosts(&args.stage_hosts)?
    } else {
        vec![LOCAL_STAGE_HOST.to_string()]
    };
    if explicit_stage_hosts
        && stage_hosts
            .first()
            .is_none_or(|host| !is_local_stage_host(host))
    {
        bail!("--stage-hosts must place stage 0 on 'local' for SPD OpenAI smoke");
    }

    let mut stages = Vec::with_capacity(stage_ranges.len());
    for (index, _) in stage_ranges.iter().enumerate() {
        let host = stage_hosts[index % stage_hosts.len()].clone();
        let local = is_local_stage_host(&host);
        let remote_dir =
            (!local).then(|| format!("{}/{run_id}/{}", args.remote_root, stage_id(index)));
        let remote_config_path = remote_dir.as_ref().map(|dir| format!("{dir}/stage.json"));
        let remote_topology_path = remote_dir
            .as_ref()
            .map(|dir| format!("{dir}/topology.json"));
        let remote_log_path = remote_dir.as_ref().map(|dir| format!("{dir}/stage.log"));
        let remote_pid_path = remote_dir.as_ref().map(|dir| format!("{dir}/stage.pid"));
        let remote_exit_path = remote_dir.as_ref().map(|dir| format!("{dir}/stage.exit"));
        let remote_model_path = if !local && args.rsync_model_artifacts {
            Some(format!("{}/{run_id}/model.gguf", args.remote_root))
        } else if !local {
            remote_model_paths.get(&host).cloned()
        } else {
            None
        };
        stages.push(StageDeployment {
            index,
            host,
            local,
            config_path: case_dir.join(format!("stage{index}.json")),
            log_path: case_dir.join(format!("stage{index}.log")),
            remote_dir,
            remote_config_path,
            remote_topology_path,
            remote_log_path,
            remote_pid_path,
            remote_exit_path,
            remote_model_path,
        });
    }

    validate_remote_stage0_endpoint(explicit_stage_hosts, &stages, &endpoint_hosts)?;
    let topology_path = case_dir.join("topology.json");
    write_topology(
        args,
        &topology_path,
        stage_ranges,
        &stages,
        &ports,
        &endpoint_hosts,
    )?;
    write_stage_configs(StageConfigPlan {
        args,
        run_id,
        stage_ranges,
        tap_allowlist,
        stages: &stages,
        ports: &ports,
        endpoint_hosts: &endpoint_hosts,
        explicit_stage_hosts,
    })?;
    Ok(CaseDeployment {
        stages,
        openai_addr,
        topology_path,
    })
}

fn write_topology(
    args: &SpdOpenAiSmokeArgs,
    topology_path: &Path,
    stage_ranges: &[(u32, u32)],
    stages: &[StageDeployment],
    ports: &[u16],
    endpoint_hosts: &BTreeMap<String, String>,
) -> Result<()> {
    let topology_stages = stage_ranges
        .iter()
        .enumerate()
        .map(|(index, (layer_start, layer_end))| {
            let stage = &stages[index];
            json!({
                "stage_id": stage_id(index),
                "stage_index": index,
                "host": stage.host,
                "endpoint": stage_endpoint(stage, ports[index], endpoint_hosts),
                "layer_start": layer_start,
                "layer_end": layer_end,
                "load_mode": "runtime-slice",
            })
        })
        .collect::<Vec<_>>();
    fs::write(
        topology_path,
        serde_json::to_vec_pretty(&json!({
            "topology_id": "spd-openai-smoke",
            "model_id": args.model_id,
            "stages": topology_stages,
        }))?,
    )
    .with_context(|| format!("failed to write {}", topology_path.display()))
}

struct StageConfigPlan<'a> {
    args: &'a SpdOpenAiSmokeArgs,
    run_id: &'a str,
    stage_ranges: &'a [(u32, u32)],
    tap_allowlist: &'a [u32],
    stages: &'a [StageDeployment],
    ports: &'a [u16],
    endpoint_hosts: &'a BTreeMap<String, String>,
    explicit_stage_hosts: bool,
}

fn write_stage_configs(plan: StageConfigPlan<'_>) -> Result<()> {
    let args = plan.args;
    for (index, (layer_start, layer_end)) in plan.stage_ranges.iter().enumerate() {
        let stage = &plan.stages[index];
        let upstream = (index > 0).then(|| {
            let upstream_stage = &plan.stages[index - 1];
            json!({
                "stage_id": stage_id(index - 1),
                "stage_index": index - 1,
                "endpoint": stage_endpoint(
                    upstream_stage,
                    plan.ports[index - 1],
                    plan.endpoint_hosts,
                ),
            })
        });
        let downstream = (index + 1 < plan.stage_ranges.len()).then(|| {
            let downstream_stage = &plan.stages[index + 1];
            json!({
                "stage_id": stage_id(index + 1),
                "stage_index": index + 1,
                "endpoint": stage_endpoint(
                    downstream_stage,
                    plan.ports[index + 1],
                    plan.endpoint_hosts,
                ),
            })
        });
        let selected_device = args
            .selected_backend_device
            .as_ref()
            .map(|backend_device| json!({ "backend_device": backend_device }));
        let model_path = stage.remote_model_path.as_deref().map_or_else(
            || args.model_path.display().to_string(),
            ToString::to_string,
        );
        fs::write(
            &stage.config_path,
            serde_json::to_vec_pretty(&json!({
                "run_id": plan.run_id,
                "topology_id": "spd-openai-smoke",
                "model_id": args.model_id,
                "source_model_path": model_path,
                "model_path": model_path,
                "stage_id": stage_id(index),
                "stage_index": index,
                "layer_start": layer_start,
                "layer_end": layer_end,
                "spd_tap_return_hf_indices": plan.tap_allowlist,
                "ctx_size": args.ctx_size,
                "n_gpu_layers": args.n_gpu_layers,
                "selected_device": selected_device,
                "filter_tensors_on_load": true,
                "load_mode": "runtime-slice",
                "bind_addr": stage_bind_addr(
                    stage,
                    plan.explicit_stage_hosts,
                    plan.ports[index],
                    args,
                ),
                "upstream": upstream,
                "downstream": downstream,
            }))?,
        )
        .with_context(|| format!("failed to write {}", stage.config_path.display()))?;
    }
    Ok(())
}

fn normalized_stage_hosts(hosts: &[String]) -> Result<Vec<String>> {
    let hosts = hosts
        .iter()
        .map(|host| host.trim())
        .filter(|host| !host.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if hosts.is_empty() {
        bail!("--stage-hosts cannot be empty when provided");
    }
    Ok(hosts)
}

fn explicit_stage_ports(base: u16, count: usize) -> Result<Vec<u16>> {
    (0..count)
        .map(|index| {
            base.checked_add(u16::try_from(index).context("stage index exceeds u16")?)
                .context("stage port overflow")
        })
        .collect()
}

fn parse_string_map(input: Option<&str>) -> Result<BTreeMap<String, String>> {
    let Some(input) = input else {
        return Ok(BTreeMap::new());
    };
    input
        .split(',')
        .filter_map(|entry| {
            let entry = entry.trim();
            (!entry.is_empty()).then_some(entry)
        })
        .map(|entry| {
            let (key, value) = entry
                .split_once('=')
                .with_context(|| format!("map entry must be HOST=VALUE, got {entry:?}"))?;
            let key = key.trim();
            let value = value.trim();
            if key.is_empty() || value.is_empty() {
                bail!("map entry must not have an empty key or value: {entry:?}");
            }
            Ok((key.to_string(), value.to_string()))
        })
        .collect()
}

fn is_local_stage_host(host: &str) -> bool {
    host == LOCAL_STAGE_HOST || host == "localhost" || host == "127.0.0.1"
}

fn stage_endpoint(
    stage: &StageDeployment,
    port: u16,
    endpoint_hosts: &BTreeMap<String, String>,
) -> String {
    let host = if stage.local {
        endpoint_hosts
            .get(LOCAL_STAGE_HOST)
            .or_else(|| endpoint_hosts.get("localhost"))
            .map(String::as_str)
            .unwrap_or("127.0.0.1")
    } else {
        endpoint_hosts
            .get(&stage.host)
            .map(String::as_str)
            .unwrap_or(&stage.host)
    };
    format!("{host}:{port}")
}

fn stage_bind_addr(
    stage: &StageDeployment,
    explicit_stage_hosts: bool,
    port: u16,
    args: &SpdOpenAiSmokeArgs,
) -> String {
    if !explicit_stage_hosts {
        return format!("127.0.0.1:{port}");
    }
    if stage.local {
        format!("0.0.0.0:{port}")
    } else {
        format!("{}:{port}", args.remote_bind_host)
    }
}

fn validate_remote_stage0_endpoint(
    explicit_stage_hosts: bool,
    stages: &[StageDeployment],
    endpoint_hosts: &BTreeMap<String, String>,
) -> Result<()> {
    if !explicit_stage_hosts || stages.iter().all(|stage| stage.local) {
        return Ok(());
    }
    let stage0_endpoint = endpoint_hosts
        .get(LOCAL_STAGE_HOST)
        .or_else(|| endpoint_hosts.get("localhost"))
        .map(String::as_str)
        .unwrap_or("127.0.0.1");
    if matches!(stage0_endpoint, "127.0.0.1" | "localhost" | "::1") {
        bail!(
            "--stage-hosts with remote stages requires --endpoint-host-map local=<reachable-stage0-host>"
        );
    }
    Ok(())
}

pub(super) fn start_case_stages(
    args: &SpdOpenAiSmokeArgs,
    deployment: &CaseDeployment,
    case: SmokeCase,
) -> Result<Vec<ChildGuard>> {
    let mut stage_processes = Vec::with_capacity(deployment.stages.len());
    for stage in deployment.stages.iter().rev() {
        prepare_stage_for_start(args, deployment, stage)?;
        let index = stage.index;
        let mut command = Command::new(&args.stage_server_bin);
        command
            .arg("serve-binary")
            .arg("--config")
            .arg(&stage.config_path)
            .arg("--topology")
            .arg(&deployment.topology_path)
            .arg("--activation-width")
            .arg(args.activation_width.to_string())
            .arg("--activation-wire-dtype")
            .arg(&args.activation_wire_dtype)
            .arg("--telemetry-level")
            .arg("debug")
            .arg("--max-inflight")
            .arg(args.max_inflight.to_string());
        append_wire_condition_args(&mut command, args);

        if index == 0 {
            command
                .arg("--openai-bind-addr")
                .arg(deployment.openai_addr.to_string())
                .arg("--openai-model-id")
                .arg(&args.model_id)
                .arg("--openai-default-max-tokens")
                .arg(args.max_tokens.to_string())
                .arg("--openai-generation-concurrency")
                .arg(args.openai_generation_concurrency.to_string());
            if case.uses_spd() {
                command
                    .arg("--openai-spd-manifest")
                    .arg(&args.manifest)
                    .arg("--openai-spd-fixture")
                    .arg(&args.fixture)
                    .arg("--openai-spd-model-path")
                    .arg(&args.model_path)
                    .arg("--openai-spd-top-k")
                    .arg(args.spd_top_k.to_string())
                    .arg("--openai-spd-n-gpu-layers")
                    .arg(args.spd_n_gpu_layers.to_string())
                    .arg("--openai-speculative-window")
                    .arg(args.speculative_window.to_string());
                if args.optimistic_decode {
                    command.arg("--openai-spd-optimistic-decode");
                }
                if args.spd_replay_fallback {
                    command.arg("--openai-spd-replay-fallback");
                }
                if let Some(min_margin) = args.optimistic_min_logit_margin {
                    command
                        .arg("--openai-spd-optimistic-min-logit-margin")
                        .arg(min_margin.to_string());
                }
            }
        }

        command.env("SKIPPY_TELEMETRY_STDERR", "1");
        if stage.local {
            let log = File::create(&stage.log_path)
                .with_context(|| format!("create stage {index} log"))?;
            command
                .stdout(Stdio::from(
                    log.try_clone().context("clone stdout log file")?,
                ))
                .stderr(Stdio::from(log));
            stage_processes.push(ChildGuard::spawn(command)?);
        } else {
            stage_processes.push(start_remote_stage(args, deployment, stage)?);
        }
        wait_stage_log_ready(stage, args.startup_timeout_secs)
            .with_context(|| format!("wait for {} on {} to start", stage.stage_id(), stage.host))?;
        thread::sleep(Duration::from_millis(200));
    }
    Ok(stage_processes)
}

fn prepare_stage_for_start(
    args: &SpdOpenAiSmokeArgs,
    deployment: &CaseDeployment,
    stage: &StageDeployment,
) -> Result<()> {
    if stage.local {
        return Ok(());
    }
    let remote_dir = stage
        .remote_dir
        .as_ref()
        .context("remote stage has no dir")?;
    run_command(
        Command::new("ssh")
            .arg(&stage.host)
            .arg(format!("mkdir -p {}", shell_quote(remote_dir))),
    )
    .with_context(|| format!("create remote stage dir on {}", stage.host))?;
    run_command(
        Command::new("rsync")
            .arg("-az")
            .arg(&args.stage_server_bin)
            .arg(remote_target(
                &stage.host,
                &format!("{remote_dir}/skippy-server"),
            )),
    )
    .with_context(|| format!("rsync skippy-server to {}", stage.host))?;
    run_command(
        Command::new("rsync")
            .arg("-az")
            .arg(&stage.config_path)
            .arg(remote_target(
                &stage.host,
                stage
                    .remote_config_path
                    .as_ref()
                    .context("remote stage has no config path")?,
            )),
    )
    .with_context(|| format!("rsync stage config to {}", stage.host))?;
    run_command(
        Command::new("rsync")
            .arg("-az")
            .arg(&deployment.topology_path)
            .arg(remote_target(
                &stage.host,
                stage
                    .remote_topology_path
                    .as_ref()
                    .context("remote stage has no topology path")?,
            )),
    )
    .with_context(|| format!("rsync topology to {}", stage.host))?;
    if args.rsync_model_artifacts
        && let Some(remote_model_path) = stage.remote_model_path.as_ref()
    {
        run_command(
            Command::new("rsync")
                .arg("-az")
                .arg(&args.model_path)
                .arg(remote_target(&stage.host, remote_model_path)),
        )
        .with_context(|| format!("rsync model to {}", stage.host))?;
    }
    Ok(())
}

fn start_remote_stage(
    args: &SpdOpenAiSmokeArgs,
    deployment: &CaseDeployment,
    stage: &StageDeployment,
) -> Result<ChildGuard> {
    let remote_dir = stage
        .remote_dir
        .as_ref()
        .context("remote stage has no dir")?;
    let remote_bin = format!("{remote_dir}/skippy-server");
    let stage_command = remote_stage_server_command(args, deployment, stage, &remote_bin)?;
    let exit_path = shell_quote(
        stage
            .remote_exit_path
            .as_ref()
            .context("remote stage has no exit path")?,
    );
    let log_path = shell_quote(
        stage
            .remote_log_path
            .as_ref()
            .context("remote stage has no log path")?,
    );
    let pid_path = shell_quote(
        stage
            .remote_pid_path
            .as_ref()
            .context("remote stage has no pid path")?,
    );
    let wrapper = format!(
        "trap 'kill \"$child\" 2>/dev/null || true; wait \"$child\" 2>/dev/null; status=$?; printf \"%s\\n\" \"$status\" > {exit_path}; exit \"$status\"' TERM INT HUP; {stage_command} > {log_path} 2>&1 & child=$!; printf \"%s\\n\" \"$child\" > {pid_path}; wait \"$child\"; status=$?; printf \"%s\\n\" \"$status\" > {exit_path}; exit \"$status\""
    );
    let remote = format!(
        "chmod +x {} && rm -f {} {} && sh -c {}",
        shell_quote(&remote_bin),
        exit_path,
        pid_path,
        shell_quote(&wrapper),
    );
    let mut command = Command::new("ssh");
    command
        .arg(&stage.host)
        .arg(remote)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    ChildGuard::spawn(command).with_context(|| format!("start remote {}", stage.stage_id()))
}

fn remote_stage_server_command(
    args: &SpdOpenAiSmokeArgs,
    _deployment: &CaseDeployment,
    stage: &StageDeployment,
    remote_bin: &str,
) -> Result<String> {
    let config_path = stage
        .remote_config_path
        .as_ref()
        .context("remote stage has no config path")?;
    let topology_path = stage
        .remote_topology_path
        .as_ref()
        .context("remote stage has no topology path")?;
    Ok(format!(
        "SKIPPY_TELEMETRY_STDERR=1 {} serve-binary --config {} --topology {} --activation-width {} --activation-wire-dtype {} --telemetry-level debug --max-inflight {}{}",
        shell_quote(remote_bin),
        shell_quote(config_path),
        shell_quote(topology_path),
        args.activation_width,
        shell_quote(&args.activation_wire_dtype),
        args.max_inflight,
        remote_wire_condition_args(args),
    ))
}

fn remote_wire_condition_args(args: &SpdOpenAiSmokeArgs) -> String {
    let mut rendered = String::new();
    if args.downstream_wire_delay_ms > 0.0 {
        rendered.push_str(&format!(
            " --downstream-wire-delay-ms {}",
            args.downstream_wire_delay_ms
        ));
    }
    if let Some(mbps) = args.downstream_wire_mbps {
        rendered.push_str(&format!(" --downstream-wire-mbps {mbps}"));
    }
    rendered
}

fn wait_stage_log_ready(stage: &StageDeployment, timeout_secs: u64) -> Result<()> {
    let attempts = timeout_secs.saturating_mul(2).max(1);
    for _ in 0..attempts {
        if stage_log_ready(stage).unwrap_or(false) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(500));
    }
    bail!(
        "stage log did not become ready: {}",
        stage.log_path.display()
    )
}

fn stage_log_ready(stage: &StageDeployment) -> Result<bool> {
    if stage.local {
        return Ok(fs::read_to_string(&stage.log_path)
            .map(|content| content.contains("stage.binary_server_start"))
            .unwrap_or(false));
    }
    let remote_log = stage
        .remote_log_path
        .as_ref()
        .context("remote stage has no log path")?;
    let status = Command::new("ssh")
        .arg(&stage.host)
        .arg(format!(
            "test -s {} && grep -q stage.binary_server_start {}",
            shell_quote(remote_log),
            shell_quote(remote_log)
        ))
        .status()
        .with_context(|| format!("check remote stage log on {}", stage.host))?;
    Ok(status.success())
}

pub(super) fn collect_remote_case_logs(deployment: &CaseDeployment) -> Result<()> {
    for stage in deployment.stages.iter().filter(|stage| !stage.local) {
        let remote_log = stage
            .remote_log_path
            .as_ref()
            .context("remote stage has no log path")?;
        run_command(
            Command::new("rsync")
                .arg("-az")
                .arg(remote_target(&stage.host, remote_log))
                .arg(&stage.log_path),
        )
        .with_context(|| format!("collect remote log from {}", stage.host))?;
    }
    Ok(())
}

fn append_wire_condition_args(command: &mut Command, args: &SpdOpenAiSmokeArgs) {
    if args.downstream_wire_delay_ms > 0.0 {
        command
            .arg("--downstream-wire-delay-ms")
            .arg(args.downstream_wire_delay_ms.to_string());
    }
    if let Some(mbps) = args.downstream_wire_mbps {
        command.arg("--downstream-wire-mbps").arg(mbps.to_string());
    }
}

fn remote_target(host: &str, path: &str) -> String {
    format!("{host}:{path}")
}

fn run_command(command: &mut Command) -> Result<()> {
    let status = command
        .status()
        .with_context(|| format!("failed to spawn {:?}", command))?;
    if !status.success() {
        bail!("command failed with status {status}: {:?}", command);
    }
    Ok(())
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn allocate_ports(count: usize) -> Result<Vec<u16>> {
    (0..count)
        .map(|_| {
            let listener = TcpListener::bind("127.0.0.1:0").context("allocate local port")?;
            Ok(listener.local_addr()?.port())
        })
        .collect()
}

fn stage_id(index: usize) -> String {
    format!("stage-{index}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_stage_validation_requires_reachable_stage0_endpoint() {
        let stages = vec![
            test_stage(0, LOCAL_STAGE_HOST, true),
            test_stage(1, "worker", false),
        ];

        assert!(validate_remote_stage0_endpoint(true, &stages, &BTreeMap::new()).is_err());

        let endpoint_hosts = parse_string_map(Some("local=host-a,worker=host-b")).unwrap();
        validate_remote_stage0_endpoint(true, &stages, &endpoint_hosts).unwrap();
    }

    #[test]
    fn parse_string_map_rejects_missing_separator() {
        assert!(parse_string_map(Some("local=host-a,worker")).is_err());
    }

    fn test_stage(index: usize, host: &str, local: bool) -> StageDeployment {
        StageDeployment {
            index,
            host: host.to_string(),
            local,
            config_path: PathBuf::from(format!("stage{index}.json")),
            log_path: PathBuf::from(format!("stage{index}.log")),
            remote_dir: None,
            remote_config_path: None,
            remote_topology_path: None,
            remote_log_path: None,
            remote_pid_path: None,
            remote_exit_path: None,
            remote_model_path: None,
        }
    }
}
