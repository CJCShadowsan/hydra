use std::{
    fs::{self, File},
    net::SocketAddr,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;
use serde_json::{Value, json};

use crate::{
    cli::{NativeMtpOpenAiAbArgs, StageLoadMode},
    report::{NativeMtpOpenAiAbReport, NativeMtpOpenAiCaseReport, NativeMtpOpenAiMetricsReport},
    support::{ChildGuard, connect_ready, generate_run_id},
};

const BATCHED_VERIFY_ENV: &str = "SKIPPY_NATIVE_MTP_BATCHED_VERIFY";
const NATIVE_MTP_ENABLED_ENV: &str = "SKIPPY_NATIVE_MTP_ENABLED";

struct OpenAiCaseConfig {
    case: &'static str,
    native_mtp_enabled: bool,
    batched_verify_enabled: bool,
    run_id: String,
    root: PathBuf,
    openai_bind_addr: SocketAddr,
    stage0_bind_addr: SocketAddr,
    stage1_bind_addr: SocketAddr,
}

struct OpenAiStageConfig<'a> {
    run_id: &'a str,
    model_id: &'a str,
    model_path: &'a Path,
    stage_id: &'a str,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    bind_addr: SocketAddr,
    upstream: Option<Value>,
    downstream: Option<Value>,
}

pub fn native_mtp_openai_ab(args: NativeMtpOpenAiAbArgs) -> Result<()> {
    if args.runtime.stage_load_mode != StageLoadMode::RuntimeSlice {
        bail!("native-mtp-open-ai-ab currently supports --stage-load-mode runtime-slice only");
    }
    if args.split_layer == 0 || args.split_layer >= args.runtime.layer_end {
        bail!("split_layer must be greater than zero and less than layer_end");
    }

    let model_id = args.runtime.model_id.clone().unwrap_or_else(|| {
        args.runtime
            .model
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("local-model")
            .to_string()
    });
    let root = std::env::temp_dir().join(generate_run_id());
    fs::create_dir_all(&root).with_context(|| format!("failed to create {}", root.display()))?;
    let client = Client::builder()
        .timeout(Duration::from_secs(args.request_timeout_secs.max(1)))
        .build()
        .context("failed to build HTTP client")?;

    let baseline = run_openai_case(
        &args,
        &client,
        &model_id,
        OpenAiCaseConfig {
            case: "baseline",
            native_mtp_enabled: false,
            batched_verify_enabled: false,
            run_id: format!("{}-baseline", generate_run_id()),
            root: root.join("baseline"),
            openai_bind_addr: args.openai_bind_addr,
            stage0_bind_addr: args.stage0_bind_addr,
            stage1_bind_addr: args.stage1_bind_addr,
        },
    )?;
    let n1 = run_openai_case(
        &args,
        &client,
        &model_id,
        OpenAiCaseConfig {
            case: "n1",
            native_mtp_enabled: true,
            batched_verify_enabled: false,
            run_id: format!("{}-n1", generate_run_id()),
            root: root.join("n1"),
            openai_bind_addr: offset_port(args.openai_bind_addr, args.batched_port_offset)?,
            stage0_bind_addr: offset_port(args.stage0_bind_addr, args.batched_port_offset)?,
            stage1_bind_addr: offset_port(args.stage1_bind_addr, args.batched_port_offset)?,
        },
    )?;
    let batched = run_openai_case(
        &args,
        &client,
        &model_id,
        OpenAiCaseConfig {
            case: "batched",
            native_mtp_enabled: true,
            batched_verify_enabled: true,
            run_id: format!("{}-batched", generate_run_id()),
            root: root.join("batched"),
            openai_bind_addr: offset_port(
                args.openai_bind_addr,
                args.batched_port_offset.saturating_mul(2),
            )?,
            stage0_bind_addr: offset_port(
                args.stage0_bind_addr,
                args.batched_port_offset.saturating_mul(2),
            )?,
            stage1_bind_addr: offset_port(
                args.stage1_bind_addr,
                args.batched_port_offset.saturating_mul(2),
            )?,
        },
    )?;

    let exact_content_match = baseline.content == n1.content && baseline.content == batched.content;
    let batched_events_present = batched.metrics.batched_verify_events > 0;
    let require_batched_events = !args.allow_missing_batched_events;
    let matches = baseline.http_status == 200
        && n1.http_status == 200
        && batched.http_status == 200
        && exact_content_match
        && !baseline.metrics.native_mtp_enabled
        && n1.metrics.native_mtp_enabled
        && batched.metrics.native_mtp_enabled
        && baseline.metrics.fatal_error_events == 0
        && n1.metrics.fatal_error_events == 0
        && batched.metrics.fatal_error_events == 0
        && (!require_batched_events || batched_events_present);

    let report = NativeMtpOpenAiAbReport {
        mode: "native-mtp-open-ai-ab",
        status: status(matches),
        model_id,
        model_path: args.runtime.model.display().to_string(),
        prompt: args.runtime.prompt,
        max_tokens: args.max_tokens,
        split_layer: args.split_layer,
        layer_end: args.runtime.layer_end,
        activation_width: args.activation_width,
        activation_wire_dtype: args.activation_wire_dtype,
        exact_content_match,
        batched_events_required: require_batched_events,
        batched_events_present,
        matches,
        baseline,
        n1,
        batched,
    };
    emit_report(&report, args.output.report_out.as_deref())?;
    if !report.matches && !args.allow_mismatch {
        bail!("native MTP OpenAI n=1 and batched verification did not match");
    }
    Ok(())
}

fn run_openai_case(
    args: &NativeMtpOpenAiAbArgs,
    client: &Client,
    model_id: &str,
    case: OpenAiCaseConfig,
) -> Result<NativeMtpOpenAiCaseReport> {
    fs::create_dir_all(&case.root)
        .with_context(|| format!("failed to create {}", case.root.display()))?;
    let stage0_config_path = case.root.join("stage0.json");
    let stage1_config_path = case.root.join("stage1.json");
    let topology_path = case.root.join("topology.json");
    let stage0_log = case.root.join("stage0.log");
    let stage1_log = case.root.join("stage1.log");

    write_stage_config(
        &stage0_config_path,
        &stage_config_json(
            args,
            OpenAiStageConfig {
                run_id: &case.run_id,
                model_id,
                model_path: &args.runtime.model,
                stage_id: "stage-0",
                stage_index: 0,
                layer_start: 0,
                layer_end: args.split_layer,
                bind_addr: case.stage0_bind_addr,
                upstream: None,
                downstream: Some(json!({
                    "stage_id": "stage-1",
                    "stage_index": 1,
                    "endpoint": format!("tcp://{}", case.stage1_bind_addr),
                })),
            },
        ),
    )?;
    write_stage_config(
        &stage1_config_path,
        &stage_config_json(
            args,
            OpenAiStageConfig {
                run_id: &case.run_id,
                model_id,
                model_path: &args.runtime.model,
                stage_id: "stage-1",
                stage_index: 1,
                layer_start: args.split_layer,
                layer_end: args.runtime.layer_end,
                bind_addr: case.stage1_bind_addr,
                upstream: Some(json!({
                    "stage_id": "stage-0",
                    "stage_index": 0,
                    "endpoint": format!("tcp://{}", case.stage0_bind_addr),
                })),
                downstream: None,
            },
        ),
    )?;
    write_stage_config(
        &topology_path,
        &json!({
            "topology_id": "native-mtp-open-ai-ab",
            "model_id": model_id,
            "stages": [
                {
                    "stage_id": "stage-0",
                    "stage_index": 0,
                    "host": "localhost",
                    "endpoint": format!("tcp://{}", case.stage0_bind_addr),
                    "layer_start": 0,
                    "layer_end": args.split_layer,
                    "load_mode": "runtime-slice",
                },
                {
                    "stage_id": "stage-1",
                    "stage_index": 1,
                    "host": "localhost",
                    "endpoint": format!("tcp://{}", case.stage1_bind_addr),
                    "layer_start": args.split_layer,
                    "layer_end": args.runtime.layer_end,
                    "load_mode": "runtime-slice",
                },
            ],
        }),
    )?;

    let _stage1 = spawn_stage(
        args,
        &stage1_config_path,
        None,
        &topology_path,
        &stage1_log,
        case.native_mtp_enabled,
        case.batched_verify_enabled,
    )?;
    drop(
        connect_ready(case.stage1_bind_addr, args.server.startup_timeout_secs)
            .context("stage 1 binary server did not become ready")?,
    );
    let _stage0 = spawn_stage(
        args,
        &stage0_config_path,
        Some(case.openai_bind_addr),
        &topology_path,
        &stage0_log,
        case.native_mtp_enabled,
        case.batched_verify_enabled,
    )?;
    wait_openai_ready(
        client,
        case.openai_bind_addr,
        args.server.startup_timeout_secs,
    )
    .context("stage 0 OpenAI server did not become ready")?;

    let response = client
        .post(format!(
            "http://{}/v1/chat/completions",
            case.openai_bind_addr
        ))
        .json(&json!({
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": args.runtime.prompt,
                },
            ],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }))
        .send()
        .context("failed to send OpenAI chat completion request")?;
    let http_status = response.status().as_u16();
    let body: Value = response.json().context("failed to parse OpenAI response")?;
    let content = body
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let completion_tokens = body
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64);

    drop(_stage0);
    drop(_stage1);

    let metrics = read_metrics(&stage0_log, &stage1_log)?;
    Ok(NativeMtpOpenAiCaseReport {
        case: case.case,
        native_mtp_enabled: case.native_mtp_enabled,
        batched_verify_enabled: case.batched_verify_enabled,
        http_status,
        content,
        completion_tokens,
        openai_bind_addr: case.openai_bind_addr.to_string(),
        stage0_bind_addr: case.stage0_bind_addr.to_string(),
        stage1_bind_addr: case.stage1_bind_addr.to_string(),
        stage0_log: stage0_log.display().to_string(),
        stage1_log: stage1_log.display().to_string(),
        metrics,
    })
}

fn stage_config_json(args: &NativeMtpOpenAiAbArgs, stage: OpenAiStageConfig<'_>) -> Value {
    json!({
        "run_id": stage.run_id,
        "topology_id": "native-mtp-open-ai-ab",
        "model_id": stage.model_id,
        "model_path": stage.model_path,
        "stage_id": stage.stage_id,
        "stage_index": stage.stage_index,
        "layer_start": stage.layer_start,
        "layer_end": stage.layer_end,
        "ctx_size": args.runtime.ctx_size,
        "lane_count": 1,
        "n_batch": args.runtime.n_batch,
        "n_ubatch": args.runtime.n_ubatch,
        "n_gpu_layers": args.runtime.n_gpu_layers,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "flash_attn_type": protocol_flash_attn(args.runtime.flash_attn),
        "filter_tensors_on_load": true,
        "load_mode": "runtime-slice",
        "bind_addr": stage.bind_addr,
        "upstream": stage.upstream,
        "downstream": stage.downstream,
    })
}

fn spawn_stage(
    args: &NativeMtpOpenAiAbArgs,
    config_path: &Path,
    openai_bind_addr: Option<SocketAddr>,
    topology_path: &Path,
    log_path: &Path,
    native_mtp_enabled: bool,
    batched_verify_enabled: bool,
) -> Result<ChildGuard> {
    let log = File::create(log_path)
        .with_context(|| format!("failed to create {}", log_path.display()))?;
    let mut command = Command::new(&args.server.stage_server_bin);
    command.args([
        "serve-binary",
        "--config",
        config_path
            .to_str()
            .context("stage config path is not valid UTF-8")?,
        "--topology",
        topology_path
            .to_str()
            .context("topology path is not valid UTF-8")?,
        "--activation-width",
        &args.activation_width.to_string(),
        "--activation-wire-dtype",
        &args.activation_wire_dtype,
        "--telemetry-level",
        "debug",
    ]);
    if let Some(openai_bind_addr) = openai_bind_addr {
        command.args(["--openai-bind-addr", &openai_bind_addr.to_string()]);
    }
    command.env("SKIPPY_TELEMETRY_STDERR", "1");
    if native_mtp_enabled {
        command.env(NATIVE_MTP_ENABLED_ENV, "1");
    } else {
        command.env(NATIVE_MTP_ENABLED_ENV, "0");
    }
    if batched_verify_enabled {
        command.env_remove(BATCHED_VERIFY_ENV);
    } else {
        command.env(BATCHED_VERIFY_ENV, "0");
    }
    command.stdout(Stdio::from(log.try_clone()?));
    command.stderr(Stdio::from(log));
    ChildGuard::spawn(command)
}

fn wait_openai_ready(client: &Client, addr: SocketAddr, timeout_secs: u64) -> Result<()> {
    let attempts = timeout_secs.saturating_mul(4).max(1);
    let url = format!("http://{addr}/v1/models");
    let mut last_error = None;
    for _ in 0..attempts {
        match client.get(&url).send() {
            Ok(response) if response.status().is_success() => return Ok(()),
            Ok(response) => last_error = Some(format!("HTTP {}", response.status())),
            Err(error) => last_error = Some(error.to_string()),
        }
        thread::sleep(Duration::from_millis(250));
    }
    bail!(
        "timed out waiting for {url}: {}",
        last_error.unwrap_or_else(|| "no attempts made".to_string())
    );
}

fn read_metrics(stage0_log: &Path, stage1_log: &Path) -> Result<NativeMtpOpenAiMetricsReport> {
    let mut metrics = NativeMtpOpenAiMetricsReport::default();
    for path in [stage0_log, stage1_log] {
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        metrics.fatal_error_events += count_fatal_log_lines(&text);
        for line in text.lines() {
            let Ok(event) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            apply_telemetry_event(&mut metrics, &event);
        }
    }
    Ok(metrics)
}

fn apply_telemetry_event(metrics: &mut NativeMtpOpenAiMetricsReport, event: &Value) {
    let Some(name) = event.get("event").and_then(Value::as_str) else {
        return;
    };
    let attrs = event.get("attributes").unwrap_or(&Value::Null);
    match name {
        "stage.openai_decode_token" => {
            metrics.decode_token_events += 1;
            apply_native_mtp_verification(metrics, attrs);
        }
        "stage.openai_native_mtp_verify" => {
            metrics.batched_verify_events += 1;
            match apply_native_mtp_verification(metrics, attrs) {
                Some("accepted") => metrics.batched_accepted_events += 1,
                Some("rejected") => metrics.batched_rejected_events += 1,
                _ => {}
            }
        }
        "stage.openai_decode" | "stage.openai_generation_summary" => {
            apply_generation_summary(metrics, attrs);
        }
        _ => {}
    }
}

fn apply_generation_summary(metrics: &mut NativeMtpOpenAiMetricsReport, attrs: &Value) {
    metrics.native_mtp_enabled |= attr_bool(attrs, "llama_stage.native_mtp.enabled");
    if let Some(drafted) = attr_u64(attrs, "llama_stage.native_mtp.drafted") {
        metrics.drafted_tokens = drafted;
    }
    if let Some(accepted) = attr_u64(attrs, "llama_stage.native_mtp.accepted") {
        metrics.accepted_tokens = accepted;
    }
    if let Some(rejected) = attr_u64(attrs, "llama_stage.native_mtp.rejected") {
        metrics.rejected_tokens = rejected;
    }
    if let Some(pending) = attr_u64(attrs, "llama_stage.native_mtp.pending") {
        metrics.pending_tokens = pending;
    }
    if let Some(verifications) = attr_u64(attrs, "llama_stage.native_mtp.verifications") {
        metrics.verification_count = verifications;
    }
    if let Some(proposal_compute_us) = attr_i64(attrs, "llama_stage.native_mtp.proposal_compute_us")
    {
        metrics.proposal_compute_us = proposal_compute_us;
    }
    if let Some(verification_compute_us) =
        attr_i64(attrs, "llama_stage.native_mtp.verification_compute_us")
    {
        metrics.verification_compute_us = verification_compute_us;
    }
    if let Some(accept_rate) = attrs
        .get("llama_stage.native_mtp.accept_rate")
        .and_then(Value::as_f64)
    {
        metrics.accept_rate = accept_rate;
    }
}

fn apply_native_mtp_verification<'a>(
    metrics: &mut NativeMtpOpenAiMetricsReport,
    attrs: &'a Value,
) -> Option<&'a str> {
    let verification = attrs
        .get("llama_stage.native_mtp.verification")
        .and_then(Value::as_str)?;
    match verification {
        "accepted" => {
            metrics.native_mtp_enabled = true;
            metrics.accepted_tokens += 1;
            metrics.verification_count += 1;
        }
        "rejected" => {
            metrics.native_mtp_enabled = true;
            metrics.rejected_tokens += 1;
            metrics.verification_count += 1;
        }
        _ => {}
    }
    Some(verification)
}

fn count_fatal_log_lines(text: &str) -> u64 {
    text.lines()
        .filter(|line| {
            line.contains("panicked")
                || line.contains("service_unavailable")
                || line.contains("llama_decode failed for MTP sidecar sync")
        })
        .count() as u64
}

fn attr_bool(attrs: &Value, key: &str) -> bool {
    attrs.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn attr_u64(attrs: &Value, key: &str) -> Option<u64> {
    attrs.get(key).and_then(Value::as_u64)
}

fn attr_i64(attrs: &Value, key: &str) -> Option<i64> {
    attrs.get(key).and_then(Value::as_i64)
}

fn offset_port(mut addr: SocketAddr, offset: u16) -> Result<SocketAddr> {
    let port = addr
        .port()
        .checked_add(offset)
        .context("batched port offset exceeds u16")?;
    addr.set_port(port);
    Ok(addr)
}

fn protocol_flash_attn(value: crate::cli::FlashAttentionArg) -> &'static str {
    match value {
        crate::cli::FlashAttentionArg::Auto => "auto",
        crate::cli::FlashAttentionArg::Disabled => "disabled",
        crate::cli::FlashAttentionArg::Enabled => "enabled",
    }
}

fn write_stage_config(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn emit_report<T: serde::Serialize>(report: &T, report_out: Option<&Path>) -> Result<()> {
    let json = serde_json::to_vec_pretty(report)?;
    if let Some(path) = report_out {
        fs::write(path, &json).with_context(|| format!("failed to write {}", path.display()))?;
    }
    println!("{}", String::from_utf8(json)?);
    Ok(())
}

fn status(matches: bool) -> &'static str {
    if matches { "pass" } else { "fail" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_parser_extracts_native_mtp_counts() {
        let mut metrics = NativeMtpOpenAiMetricsReport::default();
        apply_telemetry_event(
            &mut metrics,
            &json!({
                "event": "stage.openai_native_mtp_verify",
                "attributes": {
                    "llama_stage.native_mtp.verification": "accepted",
                }
            }),
        );
        apply_telemetry_event(
            &mut metrics,
            &json!({
                "event": "stage.openai_decode_token",
                "attributes": {}
            }),
        );
        apply_telemetry_event(
            &mut metrics,
            &json!({
                "event": "stage.openai_decode",
                "attributes": {
                    "llama_stage.native_mtp.enabled": true,
                    "llama_stage.native_mtp.drafted": 4,
                    "llama_stage.native_mtp.accepted": 3,
                    "llama_stage.native_mtp.rejected": 1,
                    "llama_stage.native_mtp.pending": 0,
                    "llama_stage.native_mtp.verifications": 4,
                    "llama_stage.native_mtp.accept_rate": 0.75,
                    "llama_stage.native_mtp.proposal_compute_us": 11,
                    "llama_stage.native_mtp.verification_compute_us": 22,
                }
            }),
        );

        assert!(metrics.native_mtp_enabled);
        assert_eq!(metrics.drafted_tokens, 4);
        assert_eq!(metrics.accepted_tokens, 3);
        assert_eq!(metrics.rejected_tokens, 1);
        assert_eq!(metrics.verification_count, 4);
        assert_eq!(metrics.accept_rate, 0.75);
        assert_eq!(metrics.proposal_compute_us, 11);
        assert_eq!(metrics.verification_compute_us, 22);
        assert_eq!(metrics.batched_verify_events, 1);
        assert_eq!(metrics.batched_accepted_events, 1);
        assert_eq!(metrics.decode_token_events, 1);
    }

    #[test]
    fn fatal_counter_ignores_connection_retry_noise() {
        let text = "\
downstream connect retry: error=Connection refused (os error 61)
llama_decode failed for MTP sidecar sync
";
        assert_eq!(count_fatal_log_lines(text), 1);
    }
}
