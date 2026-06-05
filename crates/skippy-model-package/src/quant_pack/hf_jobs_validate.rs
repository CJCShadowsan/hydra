use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;

use crate::write_json_file;

#[derive(Debug, clap::Args)]
pub(super) struct QuantPackHfJobsValidateArgs {
    submit_json: PathBuf,
    #[arg(long)]
    expected_image: Option<String>,
    #[arg(long)]
    expected_upload_repo: Option<String>,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    require_detach: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    require_hf_token_secret: bool,
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct HfJobsValidateReport {
    schema_version: u32,
    kind: String,
    status: HfJobsValidateStatus,
    submit_json: String,
    operation: Option<String>,
    image: Option<String>,
    flavor: Option<String>,
    timeout: Option<String>,
    detach: Option<bool>,
    upload_repo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hf_jobs_cli: Option<HfJobsCliCommand>,
    checks: Vec<HfJobsValidateCheck>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum HfJobsValidateStatus {
    Valid,
    Invalid,
}

#[derive(Debug, Serialize)]
struct HfJobsValidateCheck {
    id: String,
    status: HfJobsValidateStatus,
    message: String,
}

#[derive(Debug, Serialize)]
struct HfJobsCliCommand {
    argv: Vec<String>,
    shell: String,
}

pub(super) fn run_quant_pack_hf_jobs_validate(args: QuantPackHfJobsValidateArgs) -> Result<()> {
    let payload = read_submit_payload(&args.submit_json)?;
    let report = build_validate_report(&args, &payload);
    write_validate_report(args.out.as_deref(), &report)?;
    if matches!(report.status, HfJobsValidateStatus::Invalid) {
        bail!("HF Jobs submit JSON failed validation");
    }
    Ok(())
}

fn read_submit_payload(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read HF Jobs submit JSON {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse HF Jobs submit JSON {}", path.display()))
}

fn build_validate_report(
    args: &QuantPackHfJobsValidateArgs,
    payload: &Value,
) -> HfJobsValidateReport {
    let operation = payload
        .get("operation")
        .and_then(Value::as_str)
        .map(str::to_string);
    let job_args = payload.get("args").unwrap_or(&Value::Null);
    let image = job_args
        .get("image")
        .and_then(Value::as_str)
        .map(str::to_string);
    let flavor = job_args
        .get("flavor")
        .and_then(Value::as_str)
        .map(str::to_string);
    let timeout = job_args
        .get("timeout")
        .and_then(Value::as_str)
        .map(str::to_string);
    let detach = job_args.get("detach").and_then(Value::as_bool);
    let upload_repo = job_args
        .get("env")
        .and_then(|env| env.get("HF_UPLOAD_REPO"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let command_text = command_text(job_args.get("command"));
    let checks = validate_submit_payload(args, payload, job_args, &command_text);
    let hf_jobs_cli = build_hf_jobs_cli_command(job_args);
    let status = if checks
        .iter()
        .all(|check| matches!(check.status, HfJobsValidateStatus::Valid))
    {
        HfJobsValidateStatus::Valid
    } else {
        HfJobsValidateStatus::Invalid
    };

    HfJobsValidateReport {
        schema_version: 1,
        kind: "skippy_quant_pack_hf_jobs_validate".to_string(),
        status,
        submit_json: args.submit_json.display().to_string(),
        operation,
        image,
        flavor,
        timeout,
        detach,
        upload_repo,
        hf_jobs_cli,
        checks,
    }
}

fn build_hf_jobs_cli_command(job_args: &Value) -> Option<HfJobsCliCommand> {
    let image = job_args.get("image")?.as_str()?;
    let command = job_args.get("command")?;
    let command_parts = command_parts(command)?;
    if command_parts.is_empty() {
        return None;
    }

    let mut argv = vec!["hf".to_string(), "jobs".to_string(), "run".to_string()];
    if job_args.get("detach").and_then(Value::as_bool) == Some(true) {
        argv.push("--detach".to_string());
    }
    if let Some(flavor) = job_args.get("flavor").and_then(Value::as_str) {
        argv.extend(["--flavor".to_string(), flavor.to_string()]);
    }
    if let Some(timeout) = job_args.get("timeout").and_then(Value::as_str) {
        argv.extend(["--timeout".to_string(), timeout.to_string()]);
    }
    append_hf_jobs_cli_env(job_args, &mut argv);
    append_hf_jobs_cli_secrets(job_args, &mut argv);
    argv.push(image.to_string());
    argv.extend(command_parts);
    let shell = argv
        .iter()
        .map(|arg| super::shell_quote(arg))
        .collect::<Vec<_>>()
        .join(" ");
    Some(HfJobsCliCommand { argv, shell })
}

fn append_hf_jobs_cli_env(job_args: &Value, argv: &mut Vec<String>) {
    let Some(env) = job_args.get("env").and_then(Value::as_object) else {
        return;
    };
    let mut entries = env
        .iter()
        .filter_map(|(key, value)| value.as_str().map(|value| (key.as_str(), value)))
        .collect::<Vec<_>>();
    entries.sort_by_key(|(key, _)| *key);
    for (key, value) in entries {
        argv.extend(["--env".to_string(), format!("{key}={value}")]);
    }
}

fn append_hf_jobs_cli_secrets(job_args: &Value, argv: &mut Vec<String>) {
    let Some(secrets) = job_args.get("secrets").and_then(Value::as_object) else {
        return;
    };
    let mut entries = secrets
        .iter()
        .filter_map(|(key, value)| value.as_str().map(|value| (key.as_str(), value)))
        .collect::<Vec<_>>();
    entries.sort_by_key(|(key, _)| *key);
    for (key, value) in entries {
        if value == "$HF_TOKEN" && key == "HF_TOKEN" {
            argv.extend(["--secrets".to_string(), key.to_string()]);
        } else {
            argv.extend(["--secrets".to_string(), format!("{key}={value}")]);
        }
    }
}

fn command_parts(command: &Value) -> Option<Vec<String>> {
    match command {
        Value::String(text) => Some(vec![text.clone()]),
        Value::Array(parts) => parts
            .iter()
            .map(|part| part.as_str().map(str::to_string))
            .collect(),
        _ => None,
    }
}

fn validate_submit_payload(
    args: &QuantPackHfJobsValidateArgs,
    payload: &Value,
    job_args: &Value,
    command_text: &str,
) -> Vec<HfJobsValidateCheck> {
    let mut checks = vec![
        check_equal(
            "operation",
            payload.get("operation").and_then(Value::as_str),
            "run",
            "payload operation must be HF Jobs run",
        ),
        check_present(
            "args",
            payload
                .get("args")
                .filter(|value| value.is_object())
                .is_some(),
            "payload must contain an args object",
        ),
        check_allowed_flavor(job_args),
        check_timeout(job_args),
        check_command(job_args, command_text),
        check_required_text(
            "command_downloads_source",
            command_text,
            "hf download ",
            "command must download the source model in the job",
        ),
        check_required_text(
            "command_builds_quant_pack",
            command_text,
            "quant-pack build-all",
            "command must run quant-pack build-all",
        ),
        check_required_text(
            "command_creates_upload_repo",
            command_text,
            "hf repos create \"${HF_UPLOAD_REPO}\" --repo-type model --exist-ok",
            "command must create the upload repo idempotently before upload",
        ),
        check_required_text(
            "command_uploads_outputs",
            command_text,
            "hf upload \"${HF_UPLOAD_REPO}\"",
            "command must upload generated quant-pack outputs",
        ),
    ];
    add_optional_checks(args, job_args, &mut checks);
    checks
}

fn add_optional_checks(
    args: &QuantPackHfJobsValidateArgs,
    job_args: &Value,
    checks: &mut Vec<HfJobsValidateCheck>,
) {
    if args.require_detach {
        checks.push(check_present(
            "detach",
            job_args.get("detach").and_then(Value::as_bool) == Some(true),
            "payload must set detach=true for long-running jobs",
        ));
    }
    if args.require_hf_token_secret {
        checks.push(check_equal(
            "hf_token_secret",
            job_args
                .get("secrets")
                .and_then(|secrets| secrets.get("HF_TOKEN"))
                .and_then(Value::as_str),
            "$HF_TOKEN",
            "payload must pass HF_TOKEN as a secret placeholder",
        ));
    }
    if let Some(expected_image) = args.expected_image.as_deref() {
        checks.push(check_equal(
            "expected_image",
            job_args.get("image").and_then(Value::as_str),
            expected_image,
            "payload image must match expected image",
        ));
    }
    if let Some(expected_upload_repo) = args.expected_upload_repo.as_deref() {
        checks.push(check_equal(
            "expected_upload_repo",
            job_args
                .get("env")
                .and_then(|env| env.get("HF_UPLOAD_REPO"))
                .and_then(Value::as_str),
            expected_upload_repo,
            "payload upload repo must match expected repo",
        ));
    }
}

fn check_command(job_args: &Value, command_text: &str) -> HfJobsValidateCheck {
    let valid = match job_args.get("command") {
        Some(Value::Array(parts)) => {
            parts.len() >= 3
                && parts.iter().all(Value::is_string)
                && !command_text.trim().is_empty()
        }
        Some(Value::String(text)) => !text.trim().is_empty(),
        _ => false,
    };
    check_present(
        "command",
        valid,
        "payload must include a non-empty command string or command array",
    )
}

fn check_allowed_flavor(job_args: &Value) -> HfJobsValidateCheck {
    const ALLOWED: &[&str] = &[
        "cpu-basic",
        "cpu-upgrade",
        "cpu-performance",
        "cpu-xl",
        "sprx8",
        "zero-a10g",
        "t4-small",
        "t4-medium",
        "l4x1",
        "l4x4",
        "l40sx1",
        "l40sx4",
        "l40sx8",
        "a10g-small",
        "a10g-large",
        "a10g-largex2",
        "a10g-largex4",
        "a100-large",
        "a100x4",
        "a100x8",
        "inf2x6",
    ];
    let flavor = job_args.get("flavor").and_then(Value::as_str);
    check_present(
        "flavor",
        flavor.is_some_and(|value| ALLOWED.contains(&value)),
        "payload flavor must be a known HF Jobs flavor",
    )
}

fn check_timeout(job_args: &Value) -> HfJobsValidateCheck {
    let timeout = job_args.get("timeout").and_then(Value::as_str);
    check_present(
        "timeout",
        timeout.is_some_and(|value| !value.trim().is_empty()),
        "payload timeout must be set for long-running quant-pack jobs",
    )
}

fn check_equal(
    id: &str,
    actual: Option<&str>,
    expected: &str,
    valid_message: &str,
) -> HfJobsValidateCheck {
    let valid = actual == Some(expected);
    let message = if valid {
        valid_message.to_string()
    } else {
        format!(
            "{valid_message}; expected {expected:?}, got {:?}",
            actual.unwrap_or("<missing>")
        )
    };
    check(id, valid, message)
}

fn check_required_text(
    id: &str,
    command_text: &str,
    required: &str,
    valid_message: &str,
) -> HfJobsValidateCheck {
    let valid = command_text.contains(required);
    let message = if valid {
        valid_message.to_string()
    } else {
        format!("{valid_message}; missing {required:?}")
    };
    check(id, valid, message)
}

fn check_present(id: &str, valid: bool, message: &str) -> HfJobsValidateCheck {
    check(id, valid, message.to_string())
}

fn check(id: &str, valid: bool, message: String) -> HfJobsValidateCheck {
    HfJobsValidateCheck {
        id: id.to_string(),
        status: if valid {
            HfJobsValidateStatus::Valid
        } else {
            HfJobsValidateStatus::Invalid
        },
        message,
    }
}

fn command_text(command: Option<&Value>) -> String {
    match command {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn write_validate_report(out: Option<&Path>, report: &HfJobsValidateReport) -> Result<()> {
    if let Some(out) = out {
        write_json_file(out, report)
    } else {
        println!("{}", serde_json::to_string_pretty(report)?);
        Ok(())
    }
}
