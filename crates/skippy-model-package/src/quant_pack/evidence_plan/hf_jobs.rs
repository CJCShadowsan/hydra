use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use super::{EvidencePlanReport, shell_quote};

#[derive(Debug, Default, clap::Args)]
pub(super) struct HfJobsEvidenceArgs {
    #[arg(long)]
    pub(super) hf_jobs_input_upload_script_out: Option<PathBuf>,
    #[arg(long)]
    pub(super) hf_jobs_workload_out: Option<PathBuf>,
    #[arg(long)]
    pub(super) hf_jobs_submit_json_out: Option<PathBuf>,
    #[arg(long)]
    pub(super) hf_jobs_image: Option<String>,
    #[arg(long, default_value = "cpu-xl")]
    pub(super) hf_jobs_flavor: String,
    #[arg(long, default_value = "24h")]
    pub(super) hf_jobs_timeout: String,
    #[arg(long)]
    pub(super) hf_jobs_input_repo: Option<String>,
    #[arg(long, default_value = "main")]
    pub(super) hf_jobs_input_revision: String,
    #[arg(long = "hf-jobs-input-include")]
    pub(super) hf_jobs_input_includes: Vec<String>,
    #[arg(long)]
    pub(super) hf_jobs_upload_repo: Option<String>,
}

impl HfJobsEvidenceArgs {
    pub(super) fn validate(&self) -> Result<()> {
        if self.has_hf_jobs_request() && self.hf_jobs_input_repo.is_none() {
            bail!("--hf-jobs-input-repo is required when writing evidence HF Jobs artifacts");
        }
        if self.hf_jobs_submit_json_out.is_some() && self.hf_jobs_image.is_none() {
            bail!("--hf-jobs-submit-json-out requires --hf-jobs-image");
        }
        Ok(())
    }

    pub(super) fn has_workload_request(&self) -> bool {
        self.hf_jobs_workload_out.is_some() || self.hf_jobs_submit_json_out.is_some()
    }

    fn has_hf_jobs_request(&self) -> bool {
        self.has_workload_request() || self.hf_jobs_input_upload_script_out.is_some()
    }
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct HfJobsEvidenceInputUploadPlan {
    upload_script: String,
    input_repo: String,
    input_revision: String,
    source_run_dir: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    include_patterns: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    exclude_patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct HfJobsEvidenceWorkloadPlan {
    workload_script: String,
    input_repo: String,
    input_revision: String,
    execution_run_dir: String,
    runbook_cwd: String,
    plan_path: String,
    runbook_path: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    input_includes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    upload_repo: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct HfJobsEvidenceSubmitPlan {
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
    secrets: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    env: BTreeMap<String, String>,
}

pub(super) struct HfJobsEvidenceArtifacts {
    pub(super) input_upload_plan: Option<HfJobsEvidenceInputUploadPlan>,
    pub(super) workload_plan: Option<HfJobsEvidenceWorkloadPlan>,
    pub(super) submit_plan: Option<HfJobsEvidenceSubmitPlan>,
}

pub(super) fn plan_hf_jobs_artifacts(
    args: &HfJobsEvidenceArgs,
    report: &EvidencePlanReport,
    plan_path: &Path,
) -> Result<HfJobsEvidenceArtifacts> {
    if !args.has_hf_jobs_request() {
        return Ok(HfJobsEvidenceArtifacts {
            input_upload_plan: None,
            workload_plan: None,
            submit_plan: None,
        });
    }
    args.validate()?;
    let input_upload_plan = args
        .hf_jobs_input_upload_script_out
        .as_deref()
        .map(|path| build_input_upload_plan(args, report, path));
    let workload_plan = if args.has_workload_request() {
        Some(build_workload_plan(args, report, plan_path))
    } else {
        None
    };
    let submit_plan = args
        .hf_jobs_submit_json_out
        .as_deref()
        .map(|path| build_submit_plan(args, path));
    Ok(HfJobsEvidenceArtifacts {
        input_upload_plan,
        workload_plan,
        submit_plan,
    })
}

pub(super) fn write_hf_jobs_artifacts(
    args: &HfJobsEvidenceArgs,
    report: &EvidencePlanReport,
    runbook_script: &str,
    plan_path: &Path,
) -> Result<()> {
    if !args.has_hf_jobs_request() {
        return Ok(());
    }
    args.validate()?;
    if let Some(path) = args.hf_jobs_input_upload_script_out.as_deref() {
        let input_upload_plan = build_input_upload_plan(args, report, path);
        let upload_script = build_input_upload_script(args, &input_upload_plan)?;
        write_script(path, &upload_script, "HF Jobs evidence input upload")?;
    }
    let workload_script = if args.has_workload_request() {
        let workload_plan = build_workload_plan(args, report, plan_path);
        let workload_script = build_workload_script(args, report, runbook_script, &workload_plan)?;
        if let Some(path) = args.hf_jobs_workload_out.as_deref() {
            write_script(path, &workload_script, "HF Jobs evidence workload")?;
        }
        Some(workload_script)
    } else {
        None
    };
    if let Some(path) = args.hf_jobs_submit_json_out.as_deref() {
        let payload = build_submit_payload(args, workload_script.unwrap_or_default());
        write_submit_json(path, &payload)?;
    }
    Ok(())
}

fn build_input_upload_plan(
    args: &HfJobsEvidenceArgs,
    report: &EvidencePlanReport,
    script_path: &Path,
) -> HfJobsEvidenceInputUploadPlan {
    HfJobsEvidenceInputUploadPlan {
        upload_script: script_path.display().to_string(),
        input_repo: args.hf_jobs_input_repo.clone().unwrap_or_default(),
        input_revision: args.hf_jobs_input_revision.clone(),
        source_run_dir: report
            .source_run_dir
            .clone()
            .unwrap_or_else(|| report.run_dir.clone()),
        include_patterns: input_upload_include_patterns(report),
        exclude_patterns: input_upload_exclude_patterns(),
    }
}

fn input_upload_include_patterns(report: &EvidencePlanReport) -> Vec<String> {
    let quantized_model = Path::new(&report.quantized_model)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "*.gguf".to_string());
    vec![
        "quant-pack-build.json".to_string(),
        "quant-plan.json".to_string(),
        "preflight.json".to_string(),
        "decode-profile.json".to_string(),
        quantized_model,
        "package/**".to_string(),
        "quantize/**".to_string(),
    ]
}

fn input_upload_exclude_patterns() -> Vec<String> {
    vec![
        "evidence/**".to_string(),
        "evidence-plan*.json".to_string(),
        "evidence-hf-job-submit.json".to_string(),
        "run-evidence*.sh".to_string(),
    ]
}

fn build_input_upload_script(
    args: &HfJobsEvidenceArgs,
    plan: &HfJobsEvidenceInputUploadPlan,
) -> Result<String> {
    let mut script = "#!/usr/bin/env bash\nset -euo pipefail\n\n".to_string();
    script.push_str("# Skippy quant-pack evidence input upload.\n");
    script.push_str(
        "# This uploads the candidate bundle consumed by the evidence HF Jobs workload.\n\n",
    );
    script.push_str(": \"${HF_TOKEN:?HF_TOKEN is required for Hugging Face uploads}\"\n");
    writeln!(
        script,
        "HF_INPUT_REPO={}",
        shell_quote(args.hf_jobs_input_repo.as_deref().unwrap_or_default())
    )?;
    writeln!(
        script,
        "HF_INPUT_REVISION={}",
        shell_quote(&args.hf_jobs_input_revision)
    )?;
    writeln!(
        script,
        "SOURCE_RUN_DIR={}",
        shell_quote(&plan.source_run_dir)
    )?;
    script.push_str("test -d \"${SOURCE_RUN_DIR}\" || { echo \"missing source run dir ${SOURCE_RUN_DIR}\" >&2; exit 1; }\n");
    script.push_str("hf repos create \"${HF_INPUT_REPO}\" --repo-type model --exist-ok\n");
    script.push_str("hf upload-large-folder \"${HF_INPUT_REPO}\" \"${SOURCE_RUN_DIR}\" --repo-type model --revision \"${HF_INPUT_REVISION}\"");
    for include in &plan.include_patterns {
        write!(script, " --include {}", shell_quote(include))?;
    }
    for exclude in &plan.exclude_patterns {
        write!(script, " --exclude {}", shell_quote(exclude))?;
    }
    script.push('\n');
    Ok(script)
}

fn build_workload_plan(
    args: &HfJobsEvidenceArgs,
    report: &EvidencePlanReport,
    plan_path: &Path,
) -> HfJobsEvidenceWorkloadPlan {
    let run_dir = Path::new(&report.run_dir);
    HfJobsEvidenceWorkloadPlan {
        workload_script: args
            .hf_jobs_workload_out
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
        input_repo: args.hf_jobs_input_repo.clone().unwrap_or_default(),
        input_revision: args.hf_jobs_input_revision.clone(),
        execution_run_dir: report.run_dir.clone(),
        runbook_cwd: report.runbook_cwd.clone(),
        plan_path: plan_path.display().to_string(),
        runbook_path: run_dir.join("run-evidence.sh").display().to_string(),
        input_includes: args.hf_jobs_input_includes.clone(),
        upload_repo: args.hf_jobs_upload_repo.clone(),
    }
}

fn build_workload_script(
    args: &HfJobsEvidenceArgs,
    report: &EvidencePlanReport,
    runbook_script: &str,
    plan: &HfJobsEvidenceWorkloadPlan,
) -> Result<String> {
    let mut script = "#!/usr/bin/env bash\nset -euo pipefail\n\n".to_string();
    script.push_str("# Skippy quant-pack evidence HF Jobs workload.\n");
    script.push_str("# This script downloads a completed candidate bundle, writes the evidence plan/runbook, runs evidence, and optionally uploads evidence artifacts.\n\n");
    script.push_str(": \"${HF_TOKEN:?HF_TOKEN is required for Hugging Face downloads/uploads}\"\n");
    writeln!(
        script,
        "HF_INPUT_REPO={}",
        shell_quote(args.hf_jobs_input_repo.as_deref().unwrap_or_default())
    )?;
    writeln!(
        script,
        "HF_INPUT_REVISION={}",
        shell_quote(&args.hf_jobs_input_revision)
    )?;
    writeln!(script, "EXECUTION_RUN_DIR={}", shell_quote(&report.run_dir))?;
    writeln!(script, "RUNBOOK_CWD={}", shell_quote(&report.runbook_cwd))?;
    writeln!(script, "PLAN_PATH={}", shell_quote(&plan.plan_path))?;
    writeln!(script, "RUNBOOK_PATH={}", shell_quote(&plan.runbook_path))?;
    if let Some(upload_repo) = args.hf_jobs_upload_repo.as_ref() {
        writeln!(
            script,
            "HF_UPLOAD_REPO=${{HF_UPLOAD_REPO:-{}}}",
            shell_quote(upload_repo)
        )?;
    }
    script.push_str("mkdir -p \"${EXECUTION_RUN_DIR}\"\n");
    append_candidate_download(&mut script, args)?;
    script.push_str("test -d \"${RUNBOOK_CWD}\" || { echo \"missing runbook cwd ${RUNBOOK_CWD}\" >&2; exit 1; }\n");
    append_embedded_file(
        &mut script,
        "PLAN_PATH",
        serde_json::to_string_pretty(report)?,
    )?;
    append_embedded_file(&mut script, "RUNBOOK_PATH", runbook_script.to_string())?;
    script.push_str("chmod +x \"${RUNBOOK_PATH}\"\n");
    script.push_str("\"${RUNBOOK_PATH}\"\n");
    append_evidence_upload(&mut script);
    Ok(script)
}

fn append_candidate_download(script: &mut String, args: &HfJobsEvidenceArgs) -> Result<()> {
    script.push_str("hf download \"${HF_INPUT_REPO}\" --revision \"${HF_INPUT_REVISION}\" --local-dir \"${EXECUTION_RUN_DIR}\"");
    for include in &args.hf_jobs_input_includes {
        write!(script, " --include {}", shell_quote(include))?;
    }
    script.push_str("\n\n");
    Ok(())
}

fn append_embedded_file(script: &mut String, variable: &str, contents: String) -> Result<()> {
    writeln!(script, "mkdir -p \"$(dirname \"${{{variable}}}\")\"")?;
    writeln!(script, "cat > \"${{{variable}}}\" <<'SKIPPY_HF_JOB_FILE'")?;
    script.push_str(&contents);
    if !contents.ends_with('\n') {
        script.push('\n');
    }
    script.push_str("SKIPPY_HF_JOB_FILE\n\n");
    Ok(())
}

fn append_evidence_upload(script: &mut String) {
    script.push_str("if [[ -n \"${HF_UPLOAD_REPO:-}\" ]]; then\n");
    script.push_str("  hf repos create \"${HF_UPLOAD_REPO}\" --repo-type model --exist-ok\n");
    script.push_str("  hf upload \"${HF_UPLOAD_REPO}\" \"${EXECUTION_RUN_DIR}/evidence\" evidence --repo-type model\n");
    script.push_str(
        "  hf upload \"${HF_UPLOAD_REPO}\" \"${PLAN_PATH}\" evidence-plan.json --repo-type model\n",
    );
    script.push_str(
        "  hf upload \"${HF_UPLOAD_REPO}\" \"${RUNBOOK_PATH}\" run-evidence.sh --repo-type model\n",
    );
    script.push_str("else\n");
    script.push_str("  printf '%s\\n' \"HF_UPLOAD_REPO not set; evidence remains at ${EXECUTION_RUN_DIR}/evidence inside the job.\"\n");
    script.push_str("fi\n");
}

fn build_submit_payload(args: &HfJobsEvidenceArgs, workload_script: String) -> HfJobsSubmitPayload {
    let mut secrets = BTreeMap::new();
    secrets.insert("HF_TOKEN".to_string(), "$HF_TOKEN".to_string());
    let mut env = BTreeMap::new();
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

fn build_submit_plan(args: &HfJobsEvidenceArgs, path: &Path) -> HfJobsEvidenceSubmitPlan {
    HfJobsEvidenceSubmitPlan {
        submit_json: path.display().to_string(),
        operation: "run".to_string(),
        image: args.hf_jobs_image.clone().unwrap_or_default(),
        flavor: args.hf_jobs_flavor.clone(),
        timeout: args.hf_jobs_timeout.clone(),
        detach: true,
        secrets: vec!["HF_TOKEN".to_string()],
        upload_repo: args.hf_jobs_upload_repo.clone(),
    }
}

fn write_script(path: &Path, script: &str, artifact: &str) -> Result<()> {
    create_parent_dir(path)?;
    fs::write(path, script).with_context(|| format!("write {artifact} {}", path.display()))?;
    make_executable(path)
}

fn write_submit_json(path: &Path, payload: &HfJobsSubmitPayload) -> Result<()> {
    create_parent_dir(path)?;
    let json = serde_json::to_string_pretty(payload)?;
    fs::write(path, format!("{json}\n"))
        .with_context(|| format!("write HF Jobs evidence submit JSON {}", path.display()))
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
            .with_context(|| format!("stat HF Jobs evidence workload {}", path.display()))?
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)
            .with_context(|| format!("chmod HF Jobs evidence workload {}", path.display()))?;
    }
    Ok(())
}
