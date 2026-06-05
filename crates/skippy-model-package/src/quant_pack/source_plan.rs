use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::quant_plan::QuantPlanProfile;

#[derive(Debug, clap::Args)]
pub(super) struct QuantPackSourcePlanArgs {
    repo: String,
    #[arg(long, default_value = "main")]
    revision: String,
    #[arg(long)]
    local_dir: PathBuf,
    #[arg(long = "allow-pattern", default_value = "*.gguf")]
    allow_patterns: Vec<String>,
    #[arg(long)]
    source_file: Option<String>,
    #[arg(long)]
    llama_quantize: Option<PathBuf>,
    #[arg(long)]
    quant_pack_out_dir: Option<PathBuf>,
    #[arg(long)]
    model_id_prefix: Option<String>,
    #[arg(long, value_enum, default_value_t = QuantPlanProfile::CodingAgent)]
    profile: QuantPlanProfile,
    #[arg(long, default_value_t = 4)]
    stages: usize,
    #[arg(long = "candidate")]
    candidates: Vec<String>,
    #[arg(long, default_value_t = 8192)]
    ctx_size: u32,
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    n_gpu_layers: i32,
    #[arg(long, default_value = "f16")]
    cache_type_k: String,
    #[arg(long, default_value = "f16")]
    cache_type_v: String,
    #[arg(long, default_value = "f16")]
    activation_wire_dtype: String,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    keep_split: bool,
    #[arg(long)]
    decode_profile: bool,
    #[arg(long)]
    expected_download_bytes: Option<u64>,
    #[arg(long)]
    min_free_bytes: Option<u64>,
    #[arg(long)]
    hf_jobs_workload_out: Option<PathBuf>,
    #[arg(long, default_value = "/tmp/skippy-quant-pack-job")]
    hf_jobs_work_dir: String,
    #[arg(long)]
    hf_jobs_submit_json_out: Option<PathBuf>,
    #[arg(long)]
    hf_jobs_image: Option<String>,
    #[arg(long, default_value = "cpu-xl")]
    hf_jobs_flavor: String,
    #[arg(long, default_value = "24h")]
    hf_jobs_timeout: String,
    #[arg(long)]
    hf_jobs_upload_repo: Option<String>,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    script_out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct SourcePlanReport {
    schema_version: u32,
    kind: String,
    repo: String,
    revision: String,
    local_dir: String,
    allow_patterns: Vec<String>,
    selected_source: String,
    quant_pack_out_dir: String,
    model_id_prefix: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_download_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_free_bytes: Option<u64>,
    commands: Vec<SourcePlanCommand>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hf_jobs_workload: Option<HfJobsWorkloadPlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hf_jobs_submit: Option<HfJobsSubmitPlan>,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct SourcePlanCommand {
    id: String,
    description: String,
    runnable: bool,
    argv: Vec<String>,
    shell: String,
    outputs: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct HfJobsWorkloadPlan {
    workload_script: String,
    work_dir: String,
    source_dir: String,
    quant_pack_out_dir: String,
    upload_repo_env: String,
}

#[derive(Debug, Clone, Serialize)]
struct HfJobsSubmitPlan {
    submit_json: String,
    operation: String,
    image: String,
    flavor: String,
    timeout: String,
    detach: bool,
    secrets: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    upload_repo: Option<String>,
}

#[derive(Debug, Serialize)]
struct HfJobsSubmitPayload {
    operation: String,
    args: HfJobsSubmitArgs,
}

#[derive(Debug, Serialize)]
struct HfJobsSubmitArgs {
    image: String,
    command: Vec<String>,
    flavor: String,
    timeout: String,
    detach: bool,
    secrets: std::collections::BTreeMap<String, String>,
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    env: std::collections::BTreeMap<String, String>,
}

pub(super) fn run_quant_pack_source_plan(args: QuantPackSourcePlanArgs) -> Result<()> {
    let report = build_source_plan(args)?;
    if let Some(script_out) = report.script_path.as_deref() {
        write_source_script(script_out, &report.report)?;
    }
    if let Some((path, script)) = report.hf_jobs_workload.as_ref() {
        write_hf_jobs_workload(path, script)?;
    }
    if let Some((path, payload)) = report.hf_jobs_submit_json.as_ref() {
        write_hf_jobs_submit_json(path, payload)?;
    }
    write_source_report(report.report_path.as_deref(), &report.report)
}

struct BuiltSourcePlan {
    report: SourcePlanReport,
    report_path: Option<PathBuf>,
    script_path: Option<PathBuf>,
    hf_jobs_workload: Option<(PathBuf, String)>,
    hf_jobs_submit_json: Option<(PathBuf, HfJobsSubmitPayload)>,
}

fn build_source_plan(args: QuantPackSourcePlanArgs) -> Result<BuiltSourcePlan> {
    validate_hf_jobs_args(&args)?;
    let selected_source = selected_source_path(&args.local_dir, args.source_file.as_deref());
    let quant_pack_out_dir = args
        .quant_pack_out_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("target/skippy-quant-packs").join(repo_slug(&args.repo)));
    let model_id_prefix = args
        .model_id_prefix
        .clone()
        .unwrap_or_else(|| args.repo.clone());
    let expected_download_bytes = args.expected_download_bytes;
    let min_free_bytes = min_free_bytes(&args);
    let commands = vec![
        download_command(&args),
        build_all_command(
            &args,
            SourceBuildInputs {
                selected_source: &selected_source,
                quant_pack_out_dir: &quant_pack_out_dir,
                model_id_prefix: &model_id_prefix,
            },
        ),
    ];
    let hf_jobs_workload = args.hf_jobs_workload_out.as_ref().map(|path| {
        let plan = build_hf_jobs_workload_plan(&args, &model_id_prefix);
        let script = build_hf_jobs_workload_script(&args, &plan, &model_id_prefix);
        (path.clone(), plan, script)
    });
    let hf_jobs_submit_json = args.hf_jobs_submit_json_out.as_ref().map(|path| {
        let workload_script = hf_jobs_workload
            .as_ref()
            .map(|(_, _, script)| script.clone())
            .unwrap_or_else(|| {
                let plan = build_hf_jobs_workload_plan(&args, &model_id_prefix);
                build_hf_jobs_workload_script(&args, &plan, &model_id_prefix)
            });
        (
            path.clone(),
            build_hf_jobs_submit_payload(&args, workload_script),
        )
    });
    let hf_jobs_submit = hf_jobs_submit_json
        .as_ref()
        .map(|(path, payload)| build_hf_jobs_submit_plan(&args, path, payload));
    Ok(BuiltSourcePlan {
        report: SourcePlanReport {
            schema_version: 1,
            kind: "skippy_quant_pack_source_plan".to_string(),
            repo: args.repo,
            revision: args.revision,
            local_dir: args.local_dir.display().to_string(),
            allow_patterns: args.allow_patterns,
            selected_source: selected_source.display().to_string(),
            quant_pack_out_dir: quant_pack_out_dir.display().to_string(),
            model_id_prefix,
            expected_download_bytes,
            min_free_bytes,
            commands,
            hf_jobs_workload: hf_jobs_workload.as_ref().map(|(_, plan, _)| plan.clone()),
            hf_jobs_submit,
            notes: source_plan_notes(),
        },
        report_path: args.out,
        script_path: args.script_out,
        hf_jobs_workload: hf_jobs_workload.map(|(path, _, script)| (path, script)),
        hf_jobs_submit_json,
    })
}

fn min_free_bytes(args: &QuantPackSourcePlanArgs) -> Option<u64> {
    args.min_free_bytes.or(args.expected_download_bytes)
}

fn validate_hf_jobs_args(args: &QuantPackSourcePlanArgs) -> Result<()> {
    if args.hf_jobs_submit_json_out.is_some() && args.hf_jobs_image.is_none() {
        bail!("--hf-jobs-submit-json-out requires --hf-jobs-image");
    }
    Ok(())
}

struct SourceBuildInputs<'a> {
    selected_source: &'a Path,
    quant_pack_out_dir: &'a Path,
    model_id_prefix: &'a str,
}

fn download_command(args: &QuantPackSourcePlanArgs) -> SourcePlanCommand {
    let mut argv = vec![
        "hf".to_string(),
        "download".to_string(),
        args.repo.clone(),
        "--revision".to_string(),
        args.revision.clone(),
        "--local-dir".to_string(),
        args.local_dir.display().to_string(),
    ];
    for pattern in &args.allow_patterns {
        argv.push("--include".to_string());
        argv.push(pattern.clone());
    }
    command(
        "download-source",
        "Download the original source GGUF files from Hugging Face.",
        true,
        argv,
        vec![args.local_dir.display().to_string()],
    )
}

fn build_all_command(
    args: &QuantPackSourcePlanArgs,
    inputs: SourceBuildInputs<'_>,
) -> SourcePlanCommand {
    let llama_quantize = args
        .llama_quantize
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "<path-to-llama-quantize>".to_string());
    let source_file = args
        .source_file
        .clone()
        .unwrap_or_else(|| "<source-gguf-or-first-shard.gguf>".to_string());
    let mut argv = vec![
        "skippy-model-package".to_string(),
        "quant-pack".to_string(),
        "build-all".to_string(),
        inputs.selected_source.display().to_string(),
        "--profile".to_string(),
        profile_arg(args.profile).to_string(),
        "--stages".to_string(),
        args.stages.to_string(),
        "--llama-quantize".to_string(),
        llama_quantize,
        "--out-dir".to_string(),
        inputs.quant_pack_out_dir.display().to_string(),
        "--model-id-prefix".to_string(),
        inputs.model_id_prefix.to_string(),
        "--source-repo".to_string(),
        args.repo.clone(),
        "--source-revision".to_string(),
        args.revision.clone(),
        "--source-file".to_string(),
        source_file,
        "--ctx-size".to_string(),
        args.ctx_size.to_string(),
    ];
    push_i32_option(&mut argv, "--n-gpu-layers", args.n_gpu_layers);
    argv.extend([
        "--cache-type-k".to_string(),
        args.cache_type_k.clone(),
        "--cache-type-v".to_string(),
        args.cache_type_v.clone(),
        "--activation-wire-dtype".to_string(),
        args.activation_wire_dtype.clone(),
    ]);
    for candidate in &args.candidates {
        argv.push("--candidate".to_string());
        argv.push(candidate.clone());
    }
    if args.keep_split {
        argv.push("--keep-split".to_string());
    }
    if args.decode_profile {
        argv.push("--decode-profile".to_string());
    }
    command(
        "quant-pack-build-all",
        "Run this after choosing the downloaded source GGUF or first shard.",
        false,
        argv,
        vec![inputs.quant_pack_out_dir.display().to_string()],
    )
}

fn build_hf_jobs_workload_plan(
    args: &QuantPackSourcePlanArgs,
    model_id_prefix: &str,
) -> HfJobsWorkloadPlan {
    let source_dir = format!("{}/source", args.hf_jobs_work_dir.trim_end_matches('/'));
    let quant_pack_out_dir = args
        .quant_pack_out_dir
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| {
            format!(
                "{}/quant-packs/{}",
                args.hf_jobs_work_dir.trim_end_matches('/'),
                repo_slug(model_id_prefix)
            )
        });
    HfJobsWorkloadPlan {
        workload_script: args
            .hf_jobs_workload_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
        work_dir: args.hf_jobs_work_dir.clone(),
        source_dir,
        quant_pack_out_dir,
        upload_repo_env: "HF_UPLOAD_REPO".to_string(),
    }
}

fn build_hf_jobs_workload_script(
    args: &QuantPackSourcePlanArgs,
    plan: &HfJobsWorkloadPlan,
    model_id_prefix: &str,
) -> String {
    let mut script = "#!/usr/bin/env bash\nset -euo pipefail\n\n".to_string();
    script.push_str("# Skippy quant-pack HF Jobs workload.\n");
    script.push_str("# Submit this script as the job payload; do not run it on Studio for 480B-scale models.\n\n");
    script.push_str(": \"${HF_TOKEN:?HF_TOKEN is required for Hugging Face downloads/uploads}\"\n");
    script.push_str(
        "SKIPPY_MODEL_PACKAGE_BIN=\"${SKIPPY_MODEL_PACKAGE_BIN:-skippy-model-package}\"\n",
    );
    if let Some(llama_quantize) = args.llama_quantize.as_ref() {
        writeln!(
            script,
            "LLAMA_QUANTIZE=\"${{LLAMA_QUANTIZE:-{}}}\"",
            super::shell_quote(&llama_quantize.display().to_string())
        )
        .expect("write script");
    } else {
        script.push_str(
            ": \"${LLAMA_QUANTIZE:?set LLAMA_QUANTIZE to the llama-quantize binary inside the job}\"\n",
        );
    }
    writeln!(script, "WORK_DIR={}", super::shell_quote(&plan.work_dir)).expect("write script");
    writeln!(
        script,
        "SOURCE_DIR={}",
        super::shell_quote(&plan.source_dir)
    )
    .expect("write script");
    writeln!(
        script,
        "QUANT_PACK_OUT_DIR={}",
        super::shell_quote(&plan.quant_pack_out_dir)
    )
    .expect("write script");
    script.push_str("mkdir -p \"${WORK_DIR}\" \"${SOURCE_DIR}\" \"${QUANT_PACK_OUT_DIR}\"\n\n");
    append_hf_jobs_download(&mut script, args);
    append_hf_jobs_source_selection(&mut script, args);
    append_hf_jobs_build_all(&mut script, args, model_id_prefix);
    script.push_str(
        "\nif [[ -n \"${HF_UPLOAD_REPO:-}\" ]]; then\n  hf repos create \"${HF_UPLOAD_REPO}\" --repo-type model --exist-ok\n  hf upload \"${HF_UPLOAD_REPO}\" \"${QUANT_PACK_OUT_DIR}\" . --repo-type model\nelse\n  printf '%s\\n' \"HF_UPLOAD_REPO not set; quant-pack outputs remain at ${QUANT_PACK_OUT_DIR} inside the job.\"\nfi\n",
    );
    script
}

fn build_hf_jobs_submit_payload(
    args: &QuantPackSourcePlanArgs,
    workload_script: String,
) -> HfJobsSubmitPayload {
    let mut secrets = std::collections::BTreeMap::new();
    secrets.insert("HF_TOKEN".to_string(), "$HF_TOKEN".to_string());
    let mut env = std::collections::BTreeMap::new();
    if let Some(upload_repo) = args.hf_jobs_upload_repo.as_ref() {
        env.insert("HF_UPLOAD_REPO".to_string(), upload_repo.clone());
    }
    HfJobsSubmitPayload {
        operation: "run".to_string(),
        args: HfJobsSubmitArgs {
            image: args.hf_jobs_image.clone().unwrap_or_default(),
            command: vec!["/bin/bash".to_string(), "-lc".to_string(), workload_script],
            flavor: args.hf_jobs_flavor.clone(),
            timeout: args.hf_jobs_timeout.clone(),
            detach: true,
            secrets,
            env,
        },
    }
}

fn build_hf_jobs_submit_plan(
    args: &QuantPackSourcePlanArgs,
    path: &Path,
    payload: &HfJobsSubmitPayload,
) -> HfJobsSubmitPlan {
    HfJobsSubmitPlan {
        submit_json: path.display().to_string(),
        operation: payload.operation.clone(),
        image: payload.args.image.clone(),
        flavor: payload.args.flavor.clone(),
        timeout: payload.args.timeout.clone(),
        detach: payload.args.detach,
        secrets: payload.args.secrets.keys().cloned().collect(),
        upload_repo: args.hf_jobs_upload_repo.clone(),
    }
}

fn append_hf_jobs_download(script: &mut String, args: &QuantPackSourcePlanArgs) {
    writeln!(
        script,
        "hf download {} --revision {} --local-dir \"${{SOURCE_DIR}}\" \\",
        super::shell_quote(&args.repo),
        super::shell_quote(&args.revision)
    )
    .expect("write script");
    for (index, pattern) in args.allow_patterns.iter().enumerate() {
        let suffix = if index + 1 == args.allow_patterns.len() {
            ""
        } else {
            " \\"
        };
        writeln!(
            script,
            "  --include {}{suffix}",
            super::shell_quote(pattern)
        )
        .expect("write script");
    }
    script.push('\n');
}

fn append_hf_jobs_source_selection(script: &mut String, args: &QuantPackSourcePlanArgs) {
    if let Some(source_file) = args.source_file.as_deref() {
        writeln!(script, "SOURCE_FILE={}", super::shell_quote(source_file)).expect("write script");
        script.push_str("SOURCE_GGUF=\"${SOURCE_DIR}/${SOURCE_FILE}\"\n");
        script.push_str("test -f \"${SOURCE_GGUF}\" || { echo \"missing source GGUF ${SOURCE_GGUF}\" >&2; exit 1; }\n\n");
        return;
    }
    script.push_str("SOURCE_GGUF=''\n");
    script.push_str("while IFS= read -r candidate; do\n  SOURCE_GGUF=\"${candidate}\"\n  break\ndone < <(\n  {\n");
    for pattern in &args.allow_patterns {
        if pattern.ends_with(".gguf") || pattern.contains(".gguf") {
            writeln!(
                script,
                "    find \"${{SOURCE_DIR}}\" -type f -path \"${{SOURCE_DIR}}/{}\"",
                pattern.replace('"', "\\\"")
            )
            .expect("write script");
        }
    }
    script.push_str("  } | sort\n)\n");
    script.push_str("test -n \"${SOURCE_GGUF}\" || { echo \"no downloaded GGUF source found under ${SOURCE_DIR}\" >&2; exit 1; }\n");
    script.push_str("SOURCE_FILE=$(basename \"${SOURCE_GGUF}\")\n\n");
}

fn append_hf_jobs_build_all(
    script: &mut String,
    args: &QuantPackSourcePlanArgs,
    model_id_prefix: &str,
) {
    script.push_str("\"${SKIPPY_MODEL_PACKAGE_BIN}\" quant-pack build-all \"${SOURCE_GGUF}\" \\\n");
    writeln!(
        script,
        "  --profile {} \\",
        super::shell_quote(profile_arg(args.profile))
    )
    .expect("write script");
    writeln!(script, "  --stages {} \\", args.stages).expect("write script");
    script.push_str("  --llama-quantize \"${LLAMA_QUANTIZE}\" \\\n");
    script.push_str("  --out-dir \"${QUANT_PACK_OUT_DIR}\" \\\n");
    writeln!(
        script,
        "  --model-id-prefix {} \\",
        super::shell_quote(model_id_prefix)
    )
    .expect("write script");
    writeln!(
        script,
        "  --source-repo {} \\",
        super::shell_quote(&args.repo)
    )
    .expect("write script");
    writeln!(
        script,
        "  --source-revision {} \\",
        super::shell_quote(&args.revision)
    )
    .expect("write script");
    script.push_str("  --source-file \"${SOURCE_FILE}\" \\\n");
    writeln!(script, "  --ctx-size {} \\", args.ctx_size).expect("write script");
    append_hf_jobs_i32_option(script, "n-gpu-layers", args.n_gpu_layers, true);
    writeln!(
        script,
        "  --cache-type-k {} \\",
        super::shell_quote(&args.cache_type_k)
    )
    .expect("write script");
    writeln!(
        script,
        "  --cache-type-v {} \\",
        super::shell_quote(&args.cache_type_v)
    )
    .expect("write script");
    writeln!(
        script,
        "  --activation-wire-dtype {}{}",
        super::shell_quote(&args.activation_wire_dtype),
        if args.candidates.is_empty() && !args.keep_split && !args.decode_profile {
            ""
        } else {
            " \\"
        }
    )
    .expect("write script");
    append_hf_jobs_candidate_and_flags(script, args);
}

fn append_hf_jobs_i32_option(script: &mut String, name: &str, value: i32, trailing: bool) {
    let suffix = if trailing { " \\" } else { "" };
    if value < 0 {
        writeln!(script, "  --{name}={value}{suffix}").expect("write script");
    } else {
        writeln!(script, "  --{name} {value}{suffix}").expect("write script");
    }
}

fn append_hf_jobs_candidate_and_flags(script: &mut String, args: &QuantPackSourcePlanArgs) {
    let mut extras = Vec::new();
    for candidate in &args.candidates {
        extras.push(format!("--candidate {}", super::shell_quote(candidate)));
    }
    if args.keep_split {
        extras.push("--keep-split".to_string());
    }
    if args.decode_profile {
        extras.push("--decode-profile".to_string());
    }
    for (index, extra) in extras.iter().enumerate() {
        let suffix = if index + 1 == extras.len() { "" } else { " \\" };
        writeln!(script, "  {extra}{suffix}").expect("write script");
    }
}

fn selected_source_path(local_dir: &Path, source_file: Option<&str>) -> PathBuf {
    local_dir.join(source_file.unwrap_or("<source-gguf-or-first-shard.gguf>"))
}

fn profile_arg(profile: QuantPlanProfile) -> &'static str {
    match profile {
        QuantPlanProfile::CodingAgent => "coding-agent",
    }
}

fn repo_slug(repo: &str) -> String {
    repo.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '-'
            }
        })
        .collect()
}

fn source_plan_notes() -> Vec<String> {
    vec![
        "Use the original model GGUF or original split GGUF shard set as source; Skippy materialized stage/tokenizer slices are derived cache and are not valid quantization sources.".to_string(),
        "The generated script downloads source files only; it prints the quant-pack build-all command template instead of running a large quantization job automatically.".to_string(),
        "For split GGUF repos, the generated script can discover the first downloaded shard; pass --source-file when you want to pin a known first shard explicitly.".to_string(),
        "For Qwen-scale models, generate --hf-jobs-workload-out and submit that workload to Hugging Face Jobs or another remote runner instead of running quantize/package/profile work on Studio.".to_string(),
    ]
}

fn command(
    id: &str,
    description: &str,
    runnable: bool,
    argv: Vec<String>,
    outputs: Vec<String>,
) -> SourcePlanCommand {
    let shell = super::shell_command(&argv);
    SourcePlanCommand {
        id: id.to_string(),
        description: description.to_string(),
        runnable,
        argv,
        shell,
        outputs,
    }
}

fn push_i32_option(argv: &mut Vec<String>, name: &str, value: i32) {
    if value < 0 {
        argv.push(format!("{name}={value}"));
    } else {
        argv.extend([name.to_string(), value.to_string()]);
    }
}

fn write_source_report(out: Option<&Path>, report: &SourcePlanReport) -> Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    if let Some(out) = out {
        create_parent_dir(out)?;
        fs::write(out, format!("{json}\n"))
            .with_context(|| format!("write quant-pack source plan {}", out.display()))?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn write_hf_jobs_workload(path: &Path, script: &str) -> Result<()> {
    create_parent_dir(path)?;
    fs::write(path, script)
        .with_context(|| format!("write HF Jobs workload {}", path.display()))?;
    make_executable(path)
}

fn write_hf_jobs_submit_json(path: &Path, payload: &HfJobsSubmitPayload) -> Result<()> {
    create_parent_dir(path)?;
    let json = serde_json::to_string_pretty(payload)?;
    fs::write(path, format!("{json}\n"))
        .with_context(|| format!("write HF Jobs submit JSON {}", path.display()))
}

fn write_source_script(path: &Path, report: &SourcePlanReport) -> Result<()> {
    let mut script = "#!/usr/bin/env bash\nset -euo pipefail\n\n".to_string();
    append_space_check(&mut script, report)?;
    for command in &report.commands {
        if command.runnable {
            append_runnable_command(&mut script, command)?;
        } else {
            append_template_command(&mut script, report, command)?;
        }
    }
    create_parent_dir(path)?;
    fs::write(path, script).with_context(|| format!("write source script {}", path.display()))?;
    make_executable(path)
}

fn append_space_check(script: &mut String, report: &SourcePlanReport) -> Result<()> {
    let Some(required_bytes) = report.min_free_bytes else {
        return Ok(());
    };
    writeln!(script, "# source-space-check")?;
    writeln!(
        script,
        "# Require enough free space for the planned source download."
    )?;
    writeln!(script, "mkdir -p {}", super::shell_quote(&report.local_dir))?;
    writeln!(
        script,
        "AVAILABLE_KIB=$(df -Pk {} | awk 'NR==2 {{print $4}}')",
        super::shell_quote(&report.local_dir)
    )?;
    writeln!(script, "AVAILABLE_BYTES=$((AVAILABLE_KIB * 1024))")?;
    writeln!(script, "REQUIRED_BYTES={required_bytes}")?;
    writeln!(
        script,
        "test \"${{AVAILABLE_BYTES}}\" -ge \"${{REQUIRED_BYTES}}\" || {{ echo {} >&2; exit 1; }}",
        super::shell_quote(&format!(
            "not enough free space under {}; need at least {required_bytes} bytes",
            report.local_dir
        ))
    )?;
    script.push('\n');
    Ok(())
}

fn append_runnable_command(script: &mut String, command: &SourcePlanCommand) -> Result<()> {
    writeln!(script, "# {}", command.id)?;
    writeln!(script, "# {}", command.description)?;
    writeln!(script, "{}", command.shell)?;
    for output in &command.outputs {
        writeln!(
            script,
            "test -e {} || {{ echo {} >&2; exit 1; }}",
            super::shell_quote(output),
            super::shell_quote(&format!("missing expected source-plan output: {output}"))
        )?;
    }
    script.push('\n');
    Ok(())
}

fn append_template_command(
    script: &mut String,
    report: &SourcePlanReport,
    command: &SourcePlanCommand,
) -> Result<()> {
    writeln!(script, "# {}", command.id)?;
    writeln!(script, "# {}", command.description)?;
    if command_uses_source_placeholder(report, command) {
        append_source_discovery(script, report)?;
    }
    writeln!(
        script,
        "printf '%s\\n' {}",
        super::shell_quote("Next command template:")
    )?;
    writeln!(
        script,
        "printf '%s\\n' \"{}\"",
        template_shell(report, command).replace('"', "\\\"")
    )?;
    script.push('\n');
    Ok(())
}

fn append_source_discovery(script: &mut String, report: &SourcePlanReport) -> Result<()> {
    writeln!(script, "SOURCE_GGUF=''")?;
    writeln!(script, "while IFS= read -r candidate; do")?;
    writeln!(script, "  SOURCE_GGUF=\"${{candidate}}\"")?;
    writeln!(script, "  break")?;
    writeln!(script, "done < <(")?;
    writeln!(script, "  {{")?;
    for pattern in source_discovery_patterns(report) {
        writeln!(
            script,
            "    find {} -type f -path {}",
            super::shell_quote(&report.local_dir),
            super::shell_quote(&pattern)
        )?;
    }
    writeln!(script, "  }} | sort")?;
    writeln!(script, ")")?;
    writeln!(
        script,
        "test -n \"${{SOURCE_GGUF}}\" || {{ echo {} >&2; exit 1; }}",
        super::shell_quote(&format!(
            "no downloaded GGUF source found under {}",
            report.local_dir
        ))
    )?;
    writeln!(script, "SOURCE_FILE=$(basename \"${{SOURCE_GGUF}}\")")?;
    Ok(())
}

fn source_discovery_patterns(report: &SourcePlanReport) -> Vec<String> {
    let mut patterns = report
        .allow_patterns
        .iter()
        .filter(|pattern| pattern.ends_with(".gguf") || pattern.contains(".gguf"))
        .map(|pattern| format!("{}/{}", report.local_dir, pattern))
        .collect::<Vec<_>>();
    if patterns.is_empty() {
        patterns.push(format!("{}/{}", report.local_dir, "*.gguf"));
    }
    patterns
}

fn command_uses_source_placeholder(report: &SourcePlanReport, command: &SourcePlanCommand) -> bool {
    if !source_placeholder_is_unresolved(report) {
        return false;
    }
    command
        .argv
        .iter()
        .any(|arg| arg == "<source-gguf-or-first-shard.gguf>" || arg == &report.selected_source)
}

fn template_shell(report: &SourcePlanReport, command: &SourcePlanCommand) -> String {
    let unresolved_source = source_placeholder_is_unresolved(report);
    command
        .argv
        .iter()
        .map(|arg| {
            if unresolved_source && arg == &report.selected_source {
                "${SOURCE_GGUF}".to_string()
            } else if unresolved_source && arg == "<source-gguf-or-first-shard.gguf>" {
                "${SOURCE_FILE}".to_string()
            } else {
                super::shell_quote(arg)
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn source_placeholder_is_unresolved(report: &SourcePlanReport) -> bool {
    report
        .selected_source
        .contains("<source-gguf-or-first-shard.gguf>")
}

fn create_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    Ok(())
}

fn make_executable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mut permissions = fs::metadata(path)
            .with_context(|| format!("stat source script {}", path.display()))?
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)
            .with_context(|| format!("chmod source script {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_plan_defaults_to_hf_gguf_download_and_non_runnable_build_template() {
        let plan = build_source_plan(source_plan_args()).expect("source plan");
        let report = plan.report;

        assert_eq!(report.kind, "skippy_quant_pack_source_plan");
        assert_eq!(report.allow_patterns, ["*.gguf"]);
        assert_eq!(
            report.model_id_prefix,
            "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF"
        );
        assert_eq!(report.commands.len(), 2);
        assert!(report.commands[0].runnable);
        assert!(!report.commands[1].runnable);
        assert!(report.commands[0].shell.contains("hf download"));
        assert!(report.commands[0].shell.contains("'*.gguf'"));
        assert!(
            report.commands[1]
                .shell
                .contains("<source-gguf-or-first-shard.gguf>")
        );
        assert!(report.commands[1].shell.contains("--keep-split"));
    }

    #[test]
    fn source_plan_uses_explicit_source_file_and_quantizer() {
        let mut args = source_plan_args();
        args.source_file = Some("model-00001-of-00012.gguf".to_string());
        args.llama_quantize = Some(PathBuf::from("/opt/llama-quantize"));
        args.quant_pack_out_dir = Some(PathBuf::from("/packs/qwen"));
        args.model_id_prefix = Some("local/qwen-coder".to_string());
        args.candidates = vec!["baseline-source-quant".to_string()];
        let report = build_source_plan(args).expect("source plan").report;
        let build = &report.commands[1];

        assert_eq!(
            report.selected_source,
            "/models/qwen/model-00001-of-00012.gguf"
        );
        assert!(build.shell.contains("/opt/llama-quantize"));
        assert!(
            build
                .shell
                .contains("--source-file model-00001-of-00012.gguf")
        );
        assert!(build.shell.contains("--candidate baseline-source-quant"));
    }

    #[test]
    fn source_script_downloads_but_only_prints_build_template() {
        let dir = unique_test_dir("source-script");
        let script = dir.join("fetch-source.sh");
        let mut args = source_plan_args();
        args.expected_download_bytes = Some(123_456);
        args.min_free_bytes = Some(234_567);
        let mut plan = build_source_plan(args).expect("source plan").report;
        plan.commands[0].outputs = vec![dir.display().to_string()];

        write_source_script(&script, &plan).expect("write source script");

        let script_text = fs::read_to_string(&script).expect("read source script");
        assert!(script_text.contains("source-space-check"));
        assert!(script_text.contains("REQUIRED_BYTES=234567"));
        assert!(script_text.contains("df -Pk /models/qwen"));
        assert!(script_text.contains("hf download"));
        assert!(script_text.contains("missing expected source-plan output"));
        assert!(script_text.contains("find /models/qwen -type f -path '/models/qwen/*.gguf'"));
        assert!(script_text.contains("SOURCE_FILE=$(basename"));
        assert!(script_text.contains("Next command template:"));
        assert!(script_text.contains("${SOURCE_GGUF}"));
        assert!(script_text.contains("${SOURCE_FILE}"));
        assert!(script_text.contains("skippy-model-package quant-pack build-all"));
        assert_eq!(
            script_text
                .matches("skippy-model-package quant-pack build-all")
                .count(),
            1
        );
        fs::remove_dir_all(dir).expect("remove temp dir");
    }

    #[test]
    fn source_plan_can_write_hf_jobs_workload() {
        let dir = unique_test_dir("hf-jobs-workload");
        let workload = dir.join("qwen-hf-job.sh");
        let mut args = source_plan_args();
        args.hf_jobs_workload_out = Some(workload.clone());
        args.hf_jobs_work_dir = "/job/skippy-qwen".to_string();
        args.llama_quantize = Some(PathBuf::from("/job/bin/llama-quantize"));
        args.candidates = vec!["ffn-compressed-attention-protected".to_string()];

        let plan = build_source_plan(args).expect("source plan");
        let report = plan.report;
        let workload_plan = report.hf_jobs_workload.expect("hf jobs plan");

        assert_eq!(
            workload_plan.workload_script,
            workload.display().to_string()
        );
        assert_eq!(workload_plan.work_dir, "/job/skippy-qwen");
        assert_eq!(workload_plan.source_dir, "/job/skippy-qwen/source");
        assert_eq!(
            workload_plan.quant_pack_out_dir,
            "/job/skippy-qwen/quant-packs/unsloth-Qwen3-Coder-480B-A35B-Instruct-GGUF"
        );

        let (_, script) = plan.hf_jobs_workload.expect("hf jobs workload script");
        assert!(script.contains("HF_TOKEN is required"));
        assert!(script.contains("hf download unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF"));
        assert!(script.contains("--local-dir \"${SOURCE_DIR}\""));
        assert!(script.contains("\"${SKIPPY_MODEL_PACKAGE_BIN}\" quant-pack build-all"));
        assert!(script.contains("--n-gpu-layers=-1"));
        assert!(script.contains("--candidate ffn-compressed-attention-protected"));
        assert!(
            script.contains("hf repos create \"${HF_UPLOAD_REPO}\" --repo-type model --exist-ok")
        );
        assert!(script.contains("hf upload \"${HF_UPLOAD_REPO}\""));

        write_hf_jobs_workload(&workload, &script).expect("write hf jobs workload");
        let script_text = fs::read_to_string(&workload).expect("read hf jobs workload");
        assert_eq!(script_text, script);
        fs::remove_dir_all(dir).expect("remove temp dir");
    }

    #[test]
    fn source_plan_can_write_hf_jobs_submit_json() {
        let dir = unique_test_dir("hf-jobs-submit");
        let submit_json = dir.join("submit-qwen-job.json");
        let mut args = source_plan_args();
        args.hf_jobs_submit_json_out = Some(submit_json.clone());
        args.hf_jobs_image = Some("ghcr.io/example/skippy-quant-pack:latest".to_string());
        args.hf_jobs_flavor = "cpu-xl".to_string();
        args.hf_jobs_timeout = "36h".to_string();
        args.hf_jobs_upload_repo = Some("alexz-oai/qwen480-skippy-pack".to_string());

        let plan = build_source_plan(args).expect("source plan");
        let submit_plan = plan.report.hf_jobs_submit.expect("submit plan");
        assert_eq!(submit_plan.submit_json, submit_json.display().to_string());
        assert_eq!(submit_plan.operation, "run");
        assert_eq!(
            submit_plan.image,
            "ghcr.io/example/skippy-quant-pack:latest"
        );
        assert_eq!(submit_plan.timeout, "36h");
        assert_eq!(
            submit_plan.upload_repo.as_deref(),
            Some("alexz-oai/qwen480-skippy-pack")
        );

        let (_, payload) = plan.hf_jobs_submit_json.expect("submit payload");
        assert_eq!(payload.operation, "run");
        assert_eq!(payload.args.flavor, "cpu-xl");
        assert!(payload.args.detach);
        assert_eq!(
            payload.args.secrets.get("HF_TOKEN").map(String::as_str),
            Some("$HF_TOKEN")
        );
        assert_eq!(
            payload.args.env.get("HF_UPLOAD_REPO").map(String::as_str),
            Some("alexz-oai/qwen480-skippy-pack")
        );
        assert!(payload.args.command[2].contains("quant-pack build-all"));

        write_hf_jobs_submit_json(&submit_json, &payload).expect("write submit json");
        let json = fs::read_to_string(&submit_json).expect("read submit json");
        assert!(json.contains("\"operation\": \"run\""));
        assert!(json.contains("\"HF_TOKEN\": \"$HF_TOKEN\""));
        fs::remove_dir_all(dir).expect("remove temp dir");
    }

    #[test]
    fn source_plan_submit_json_requires_image() {
        let mut args = source_plan_args();
        args.hf_jobs_submit_json_out = Some(PathBuf::from("submit.json"));

        let err = build_source_plan(args)
            .err()
            .expect("missing image should fail");

        assert!(err.to_string().contains("--hf-jobs-image"));
    }

    #[test]
    fn source_discovery_patterns_follow_hf_include_globs() {
        let mut args = source_plan_args();
        args.allow_patterns = vec!["UD-Q4_K_XL/*.gguf".to_string()];
        let report = build_source_plan(args).expect("source plan").report;

        assert_eq!(
            source_discovery_patterns(&report),
            ["/models/qwen/UD-Q4_K_XL/*.gguf"]
        );
    }

    fn source_plan_args() -> QuantPackSourcePlanArgs {
        QuantPackSourcePlanArgs {
            repo: "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF".to_string(),
            revision: "main".to_string(),
            local_dir: PathBuf::from("/models/qwen"),
            allow_patterns: vec!["*.gguf".to_string()],
            source_file: None,
            llama_quantize: None,
            quant_pack_out_dir: None,
            model_id_prefix: None,
            profile: QuantPlanProfile::CodingAgent,
            stages: 4,
            candidates: Vec::new(),
            ctx_size: 8192,
            n_gpu_layers: -1,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            activation_wire_dtype: "f16".to_string(),
            keep_split: true,
            decode_profile: false,
            expected_download_bytes: None,
            min_free_bytes: None,
            hf_jobs_workload_out: None,
            hf_jobs_work_dir: "/tmp/skippy-quant-pack-job".to_string(),
            hf_jobs_submit_json_out: None,
            hf_jobs_image: None,
            hf_jobs_flavor: "cpu-xl".to_string(),
            hf_jobs_timeout: "24h".to_string(),
            hf_jobs_upload_repo: None,
            out: None,
            script_out: None,
        }
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "skippy-quant-pack-source-plan-{name}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }
}
