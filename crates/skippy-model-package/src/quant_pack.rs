use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Subcommand;
use serde::{Deserialize, Serialize};

use crate::profile::{ProfileArgs, ProfilePhase, TimingSourceKind};
use crate::quant_plan::{QuantPlanArgs, QuantPlanProfile};
use crate::quantize::QuantizeArgs;
use crate::{
    ArtifactHook, ExplicitSourceIdentity, resolve_local_package_input, write_json_file,
    write_package,
};

mod rank;
mod source_plan;

mod evidence_plan;
mod evidence_status;
mod hf_jobs_validate;

use evidence_plan::{QuantPackEvidencePlanAllArgs, QuantPackEvidencePlanArgs};
use evidence_status::QuantPackEvidenceStatusArgs;
use hf_jobs_validate::QuantPackHfJobsValidateArgs;
use rank::QuantPackRankArgs;
use source_plan::QuantPackSourcePlanArgs;

#[derive(Debug, clap::Args)]
pub(crate) struct QuantPackArgs {
    #[command(subcommand)]
    command: QuantPackCommand,
}

#[derive(Debug, Subcommand)]
enum QuantPackCommand {
    Build(Box<QuantPackBuildArgs>),
    BuildAll(Box<QuantPackBuildAllArgs>),
    Certify(crate::quant_certify::QuantPackCertifyArgs),
    EvidencePlan(Box<QuantPackEvidencePlanArgs>),
    EvidencePlanAll(Box<QuantPackEvidencePlanAllArgs>),
    EvidenceStatus(QuantPackEvidenceStatusArgs),
    Finalize(Box<QuantPackFinalizeArgs>),
    HfJobsValidate(QuantPackHfJobsValidateArgs),
    Rank(QuantPackRankArgs),
    SourcePlan(Box<QuantPackSourcePlanArgs>),
}

#[derive(Debug, clap::Args)]
struct QuantPackBuildArgs {
    source: PathBuf,
    #[arg(long, value_enum, default_value_t = QuantPlanProfile::CodingAgent)]
    profile: QuantPlanProfile,
    #[arg(long, default_value_t = 2)]
    stages: usize,
    #[arg(long, default_value = "middle-compressed")]
    candidate: String,
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long)]
    llama_quantize: PathBuf,
    #[arg(long)]
    quantized_model_out: Option<PathBuf>,
    #[arg(long)]
    package_dir: Option<PathBuf>,
    #[arg(long)]
    model_id: String,
    #[arg(long)]
    source_repo: Option<String>,
    #[arg(long)]
    source_revision: Option<String>,
    #[arg(long)]
    source_file: Option<String>,
    #[arg(long = "projector")]
    projectors: Vec<PathBuf>,
    #[arg(long)]
    after_artifact_command: Option<PathBuf>,
    #[arg(long)]
    nthreads: Option<u32>,
    #[arg(long)]
    keep_split: bool,
    #[arg(long)]
    verify_sha256: bool,
    #[arg(long)]
    decode_profile: bool,
    #[arg(long, default_value_t = 8192)]
    profile_existing_kv_tokens: u32,
    #[arg(long, default_value_t = 3)]
    profile_warmup_samples: u32,
    #[arg(long, default_value_t = 20)]
    profile_samples: u32,
}

#[derive(Debug, clap::Args)]
struct QuantPackBuildAllArgs {
    source: PathBuf,
    #[arg(long, value_enum, default_value_t = QuantPlanProfile::CodingAgent)]
    profile: QuantPlanProfile,
    #[arg(long, default_value_t = 2)]
    stages: usize,
    #[arg(long = "candidate")]
    candidates: Vec<String>,
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long)]
    llama_quantize: PathBuf,
    #[arg(long)]
    model_id_prefix: String,
    #[arg(long)]
    source_repo: Option<String>,
    #[arg(long)]
    source_revision: Option<String>,
    #[arg(long)]
    source_file: Option<String>,
    #[arg(long = "projector")]
    projectors: Vec<PathBuf>,
    #[arg(long)]
    after_artifact_command: Option<PathBuf>,
    #[arg(long)]
    nthreads: Option<u32>,
    #[arg(long)]
    keep_split: bool,
    #[arg(long)]
    verify_sha256: bool,
    #[arg(long)]
    decode_profile: bool,
    #[arg(long, default_value_t = 8192)]
    profile_existing_kv_tokens: u32,
    #[arg(long, default_value_t = 3)]
    profile_warmup_samples: u32,
    #[arg(long, default_value_t = 20)]
    profile_samples: u32,
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
}

#[derive(Debug, clap::Args)]
struct QuantPackFinalizeArgs {
    quantize_run: PathBuf,
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long, default_value_t = 2)]
    stages: usize,
    #[arg(long)]
    package_dir: Option<PathBuf>,
    #[arg(long)]
    model_id: String,
    #[arg(long)]
    source_repo: Option<String>,
    #[arg(long)]
    source_revision: Option<String>,
    #[arg(long)]
    source_file: Option<String>,
    #[arg(long = "projector")]
    projectors: Vec<PathBuf>,
    #[arg(long)]
    after_artifact_command: Option<PathBuf>,
    #[arg(long)]
    verify_sha256: bool,
    #[arg(long)]
    decode_profile: bool,
    #[arg(long, default_value_t = 8192)]
    profile_existing_kv_tokens: u32,
    #[arg(long, default_value_t = 3)]
    profile_warmup_samples: u32,
    #[arg(long, default_value_t = 20)]
    profile_samples: u32,
    #[arg(long)]
    plan: Option<PathBuf>,
    #[arg(long)]
    reuse_package_if_present: bool,
}

#[derive(Debug, Serialize)]
struct QuantPackBuildManifest {
    schema_version: u32,
    kind: String,
    profile: QuantPlanProfile,
    stages: usize,
    candidate: String,
    source: String,
    source_identity: QuantPackSourceIdentity,
    quantize: QuantPackQuantizeReproducibility,
    package_build: QuantPackPackageReproducibility,
    profile_request: QuantPackDecodeProfileRequest,
    plan: String,
    tensor_type_file: String,
    agent_pack: String,
    quantize_run: String,
    quantized_model: String,
    package: String,
    preflight: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_profile: Option<String>,
    preflight_verified_sha256: bool,
}

#[derive(Debug, Clone, Serialize)]
struct QuantPackSourceIdentity {
    model_id: String,
    path: String,
    repo: Option<String>,
    revision: Option<String>,
    primary_file: Option<String>,
    canonical_ref: Option<String>,
    distribution_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct QuantPackQuantizeReproducibility {
    llama_quantize: String,
    nthreads: Option<u32>,
    keep_split: bool,
}

#[derive(Debug, Clone, Serialize)]
struct QuantPackPackageReproducibility {
    verify_sha256: bool,
    projectors: Vec<String>,
    after_artifact_command: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct QuantPackDecodeProfileRequest {
    enabled: bool,
    timing_source: String,
    phase: String,
    existing_kv_tokens: u32,
    generated_tokens: u32,
    batch_size: u32,
    kv_type: String,
    warmup_samples: u32,
    samples: u32,
}

#[derive(Debug, Clone, Serialize)]
struct QuantPackRankRuntimeShape {
    ctx_size: u32,
    n_gpu_layers: i32,
    cache_type_k: String,
    cache_type_v: String,
    activation_wire_dtype: String,
}

struct QuantPackBuildPaths {
    plan: PathBuf,
    quantize_dir: PathBuf,
    tensor_type_file: PathBuf,
    agent_pack: PathBuf,
    quantize_run: PathBuf,
    quantized_model: PathBuf,
    package: PathBuf,
    preflight: PathBuf,
    decode_profile: PathBuf,
    manifest: PathBuf,
}

struct QuantPackFinalizePaths {
    quantize_run: PathBuf,
    quantize_dir: PathBuf,
    tensor_type_file: PathBuf,
    agent_pack: PathBuf,
    package: PathBuf,
    preflight: PathBuf,
    decode_profile: PathBuf,
    manifest: PathBuf,
}

#[derive(Debug, Deserialize)]
struct QuantizeRunManifestInput {
    source: QuantizeRunSourceInput,
    candidate: QuantizeRunCandidateInput,
    tensor_type_file: String,
    keep_split: bool,
    quantized_model: Option<String>,
    #[serde(default)]
    command: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct QuantizeRunSourceInput {
    path: String,
}

#[derive(Debug, Deserialize)]
struct QuantizeRunCandidateInput {
    id: String,
}

pub(crate) fn run_quant_pack(args: QuantPackArgs) -> Result<()> {
    match args.command {
        QuantPackCommand::Build(args) => run_quant_pack_build(*args),
        QuantPackCommand::BuildAll(args) => run_quant_pack_build_all(*args),
        QuantPackCommand::Certify(args) => crate::quant_certify::run_quant_pack_certify(args),
        QuantPackCommand::EvidencePlan(args) => evidence_plan::run_quant_pack_evidence_plan(*args),
        QuantPackCommand::EvidencePlanAll(args) => {
            evidence_plan::run_quant_pack_evidence_plan_all(*args)
        }
        QuantPackCommand::EvidenceStatus(args) => {
            evidence_status::run_quant_pack_evidence_status(args)
        }
        QuantPackCommand::Finalize(args) => run_quant_pack_finalize(*args),
        QuantPackCommand::HfJobsValidate(args) => {
            hf_jobs_validate::run_quant_pack_hf_jobs_validate(args)
        }
        QuantPackCommand::Rank(args) => rank::run_quant_pack_rank(args),
        QuantPackCommand::SourcePlan(args) => source_plan::run_quant_pack_source_plan(*args),
    }
}

fn run_quant_pack_build(args: QuantPackBuildArgs) -> Result<()> {
    let paths = quant_pack_build_paths(
        &args.out_dir,
        &args.candidate,
        args.quantized_model_out.as_deref(),
        args.package_dir.as_deref(),
    );
    fs::create_dir_all(&args.out_dir).with_context(|| {
        format!(
            "create quant pack output directory {}",
            args.out_dir.display()
        )
    })?;
    let quantize_repro = QuantPackQuantizeReproducibility {
        llama_quantize: args.llama_quantize.display().to_string(),
        nthreads: args.nthreads,
        keep_split: args.keep_split,
    };
    let package_repro = QuantPackPackageReproducibility {
        verify_sha256: args.verify_sha256,
        projectors: display_paths(&args.projectors),
        after_artifact_command: args
            .after_artifact_command
            .as_ref()
            .map(|path| path.display().to_string()),
    };
    let profile_request = decode_profile_request(&args);

    let package_input = resolve_local_package_input(
        args.source.clone(),
        ExplicitSourceIdentity {
            model_id: Some(args.model_id),
            source_repo: args.source_repo,
            source_revision: args.source_revision,
            source_file: args.source_file,
        },
    )
    .with_context(|| format!("resolve source identity for {}", args.source.display()))?;
    let package_model_id = package_input.model_id;
    let source_identity = package_input.source_identity;
    let manifest_source_identity =
        source_identity_manifest(&package_model_id, &args.source, &source_identity);

    crate::quant_plan::run_quant_plan(QuantPlanArgs {
        source: args.source.clone(),
        profile: args.profile,
        stages: args.stages,
        out: Some(paths.plan.clone()),
    })?;

    let quantize_output = crate::quantize::run_quantize(QuantizeArgs {
        source: args.source.clone(),
        plan: paths.plan.clone(),
        candidate: args.candidate.clone(),
        out_dir: paths.quantize_dir.clone(),
        llama_quantize: Some(args.llama_quantize),
        quantized_model_out: Some(paths.quantized_model.clone()),
        emit_only: false,
        keep_split: args.keep_split,
        nthreads: args.nthreads,
    })?;
    let quantized_model = quantize_output
        .quantized_model
        .unwrap_or_else(|| paths.quantized_model.clone());

    write_package(
        quantized_model.display().to_string(),
        paths.package.clone(),
        args.projectors,
        ArtifactHook {
            command: args.after_artifact_command,
        },
        Some(paths.agent_pack.clone()),
        ExplicitSourceIdentity {
            model_id: Some(package_model_id),
            source_repo: source_identity.repo,
            source_revision: source_identity.revision,
            source_file: source_identity.primary_file,
        },
    )?;

    let preflight = crate::preflight::preflight_package(
        &paths.package,
        &crate::preflight::PackagePreflightOptions {
            stages: Some(args.stages),
            verify_sha256: args.verify_sha256,
        },
    );
    write_json_file(&paths.preflight, &preflight)?;
    println!("{}", serde_json::to_string_pretty(&preflight)?);
    if !preflight.valid {
        bail!("package preflight failed");
    }

    let decode_profile = if args.decode_profile {
        crate::profile::run_profile(ProfileArgs {
            package: quantized_model.clone(),
            stages: 1,
            phase: ProfilePhase::Decode,
            existing_kv_tokens: args.profile_existing_kv_tokens,
            generated_tokens: 1,
            batch_size: 1,
            kv_type: "f16".to_string(),
            backend: None,
            device: None,
            samples: args.profile_samples,
            warmup_samples: args.profile_warmup_samples,
            timing_source: TimingSourceKind::LocalStage,
            out: Some(paths.decode_profile.clone()),
        })?;
        Some(paths.decode_profile.display().to_string())
    } else {
        None
    };

    let manifest = QuantPackBuildManifest {
        schema_version: 1,
        kind: "skippy_quant_pack_build".to_string(),
        profile: args.profile,
        stages: args.stages,
        candidate: args.candidate,
        source: args.source.display().to_string(),
        source_identity: manifest_source_identity,
        quantize: quantize_repro,
        package_build: package_repro,
        profile_request,
        plan: paths.plan.display().to_string(),
        tensor_type_file: paths.tensor_type_file.display().to_string(),
        agent_pack: paths.agent_pack.display().to_string(),
        quantize_run: paths.quantize_run.display().to_string(),
        quantized_model: quantized_model.display().to_string(),
        package: paths.package.display().to_string(),
        preflight: paths.preflight.display().to_string(),
        decode_profile,
        preflight_verified_sha256: args.verify_sha256,
    };
    write_json_file(&paths.manifest, &manifest)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn run_quant_pack_finalize(args: QuantPackFinalizeArgs) -> Result<()> {
    let paths = quant_pack_finalize_paths(
        &args.out_dir,
        &args.quantize_run,
        args.package_dir.as_deref(),
    )?;
    fs::create_dir_all(&args.out_dir).with_context(|| {
        format!(
            "create quant-pack finalize output directory {}",
            args.out_dir.display()
        )
    })?;
    let quantize_run = read_json::<QuantizeRunManifestInput>(&paths.quantize_run)?;
    let quantized_model = quantize_run
        .quantized_model
        .as_deref()
        .map(|path| resolve_manifest_path(&paths.quantize_dir, path))
        .transpose()?
        .with_context(|| {
            format!(
                "{} does not record quantized_model",
                paths.quantize_run.display()
            )
        })?;
    let tensor_type_file = if quantize_run.tensor_type_file.is_empty() {
        paths.tensor_type_file.clone()
    } else {
        resolve_manifest_path(&paths.quantize_dir, &quantize_run.tensor_type_file)?
    };
    let package_repro = QuantPackPackageReproducibility {
        verify_sha256: args.verify_sha256,
        projectors: display_paths(&args.projectors),
        after_artifact_command: args
            .after_artifact_command
            .as_ref()
            .map(|path| path.display().to_string()),
    };
    let manifest_source_identity =
        finalize_source_identity_manifest(&args, &quantize_run.source.path);

    ensure_finalized_package(&args, &paths, &quantized_model)?;
    let preflight = crate::preflight::preflight_package(
        &paths.package,
        &crate::preflight::PackagePreflightOptions {
            stages: Some(args.stages),
            verify_sha256: args.verify_sha256,
        },
    );
    write_json_file(&paths.preflight, &preflight)?;
    println!("{}", serde_json::to_string_pretty(&preflight)?);
    if !preflight.valid {
        bail!("package preflight failed");
    }

    let decode_profile = finalized_decode_profile(&args, &paths, &quantized_model)?;
    let llama_quantize = quantize_run_binary(&quantize_run);
    let manifest = QuantPackBuildManifest {
        schema_version: 1,
        kind: "skippy_quant_pack_build".to_string(),
        profile: QuantPlanProfile::CodingAgent,
        stages: args.stages,
        candidate: quantize_run.candidate.id,
        source: quantize_run.source.path,
        source_identity: manifest_source_identity,
        quantize: QuantPackQuantizeReproducibility {
            llama_quantize,
            nthreads: None,
            keep_split: quantize_run.keep_split,
        },
        package_build: package_repro,
        profile_request: finalize_decode_profile_request(&args),
        plan: args
            .plan
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
        tensor_type_file: tensor_type_file.display().to_string(),
        agent_pack: paths.agent_pack.display().to_string(),
        quantize_run: paths.quantize_run.display().to_string(),
        quantized_model: quantized_model.display().to_string(),
        package: paths.package.display().to_string(),
        preflight: paths.preflight.display().to_string(),
        decode_profile,
        preflight_verified_sha256: args.verify_sha256,
    };
    write_json_file(&paths.manifest, &manifest)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn ensure_finalized_package(
    args: &QuantPackFinalizeArgs,
    paths: &QuantPackFinalizePaths,
    quantized_model: &Path,
) -> Result<()> {
    let manifest = paths.package.join("model-package.json");
    if args.reuse_package_if_present && manifest.exists() {
        return Ok(());
    }
    write_package(
        quantized_model.display().to_string(),
        paths.package.clone(),
        args.projectors.clone(),
        ArtifactHook {
            command: args.after_artifact_command.clone(),
        },
        Some(paths.agent_pack.clone()),
        ExplicitSourceIdentity {
            model_id: Some(args.model_id.clone()),
            source_repo: args.source_repo.clone(),
            source_revision: args.source_revision.clone(),
            source_file: args.source_file.clone(),
        },
    )
}

fn finalized_decode_profile(
    args: &QuantPackFinalizeArgs,
    paths: &QuantPackFinalizePaths,
    quantized_model: &Path,
) -> Result<Option<String>> {
    if !args.decode_profile {
        return Ok(None);
    }
    crate::profile::run_profile(ProfileArgs {
        package: quantized_model.to_path_buf(),
        stages: 1,
        phase: ProfilePhase::Decode,
        existing_kv_tokens: args.profile_existing_kv_tokens,
        generated_tokens: 1,
        batch_size: 1,
        kv_type: "f16".to_string(),
        backend: None,
        device: None,
        samples: args.profile_samples,
        warmup_samples: args.profile_warmup_samples,
        timing_source: TimingSourceKind::LocalStage,
        out: Some(paths.decode_profile.clone()),
    })?;
    Ok(Some(paths.decode_profile.display().to_string()))
}

fn finalize_decode_profile_request(args: &QuantPackFinalizeArgs) -> QuantPackDecodeProfileRequest {
    QuantPackDecodeProfileRequest {
        enabled: args.decode_profile,
        timing_source: "local-stage".to_string(),
        phase: "decode".to_string(),
        existing_kv_tokens: args.profile_existing_kv_tokens,
        generated_tokens: 1,
        batch_size: 1,
        kv_type: "f16".to_string(),
        warmup_samples: args.profile_warmup_samples,
        samples: args.profile_samples,
    }
}

fn finalize_source_identity_manifest(
    args: &QuantPackFinalizeArgs,
    source_path: &str,
) -> QuantPackSourceIdentity {
    QuantPackSourceIdentity {
        model_id: args.model_id.clone(),
        path: source_path.to_string(),
        repo: args.source_repo.clone(),
        revision: args.source_revision.clone(),
        primary_file: args.source_file.clone(),
        canonical_ref: canonical_ref(
            args.source_repo.as_deref(),
            args.source_revision.as_deref(),
            args.source_file.as_deref(),
        ),
        distribution_id: args
            .source_file
            .as_deref()
            .and_then(distribution_id_from_file),
    }
}

fn canonical_ref(
    repo: Option<&str>,
    revision: Option<&str>,
    source_file: Option<&str>,
) -> Option<String> {
    Some(format!("{}@{}/{}", repo?, revision?, source_file?))
}

fn distribution_id_from_file(source_file: &str) -> Option<String> {
    let name = Path::new(source_file).file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    let shard_marker = "-00001-of-";
    let distribution = stem
        .find(shard_marker)
        .map_or(stem, |shard_start| &stem[..shard_start]);
    Some(distribution.to_string())
}

fn quantize_run_binary(run: &QuantizeRunManifestInput) -> String {
    run.command
        .as_ref()
        .and_then(|command| command.first())
        .cloned()
        .unwrap_or_else(|| "unknown".to_string())
}

fn source_identity_manifest(
    model_id: &str,
    source: &Path,
    identity: &crate::PackageSourceIdentity,
) -> QuantPackSourceIdentity {
    QuantPackSourceIdentity {
        model_id: model_id.to_string(),
        path: source.display().to_string(),
        repo: identity.repo.clone(),
        revision: identity.revision.clone(),
        primary_file: identity.primary_file.clone(),
        canonical_ref: identity.canonical_ref.clone(),
        distribution_id: identity.distribution_id.clone(),
    }
}

fn decode_profile_request(args: &QuantPackBuildArgs) -> QuantPackDecodeProfileRequest {
    QuantPackDecodeProfileRequest {
        enabled: args.decode_profile,
        timing_source: "local-stage".to_string(),
        phase: "decode".to_string(),
        existing_kv_tokens: args.profile_existing_kv_tokens,
        generated_tokens: 1,
        batch_size: 1,
        kv_type: "f16".to_string(),
        warmup_samples: args.profile_warmup_samples,
        samples: args.profile_samples,
    }
}

fn display_paths(paths: &[PathBuf]) -> Vec<String> {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect()
}

fn quant_pack_build_paths(
    out_dir: &Path,
    candidate: &str,
    quantized_model_out: Option<&Path>,
    package_dir: Option<&Path>,
) -> QuantPackBuildPaths {
    let quantize_dir = out_dir.join("quantize");
    QuantPackBuildPaths {
        plan: out_dir.join("quant-plan.json"),
        tensor_type_file: quantize_dir.join("tensor-types.txt"),
        agent_pack: quantize_dir.join("agent-pack.json"),
        quantize_run: quantize_dir.join("quantize-run.json"),
        quantized_model: quantized_model_out
            .map(Path::to_path_buf)
            .unwrap_or_else(|| out_dir.join(format!("{candidate}.gguf"))),
        package: package_dir
            .map(Path::to_path_buf)
            .unwrap_or_else(|| out_dir.join("package")),
        preflight: out_dir.join("preflight.json"),
        decode_profile: out_dir.join("decode-profile.json"),
        manifest: out_dir.join("quant-pack-build.json"),
        quantize_dir,
    }
}

fn quant_pack_finalize_paths(
    out_dir: &Path,
    quantize_run: &Path,
    package_dir: Option<&Path>,
) -> Result<QuantPackFinalizePaths> {
    let quantize_run = absolutize_manifest_path(quantize_run)?;
    let quantize_dir = quantize_run
        .parent()
        .map(Path::to_path_buf)
        .context("quantize-run path has no parent directory")?;
    Ok(QuantPackFinalizePaths {
        tensor_type_file: quantize_dir.join("tensor-types.txt"),
        agent_pack: quantize_dir.join("agent-pack.json"),
        package: package_dir
            .map(Path::to_path_buf)
            .unwrap_or_else(|| out_dir.join("package")),
        preflight: out_dir.join("preflight.json"),
        decode_profile: out_dir.join("decode-profile.json"),
        manifest: out_dir.join("quant-pack-build.json"),
        quantize_run,
        quantize_dir,
    })
}

fn absolutize_manifest_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(path))
        .context("resolve current directory")
}

fn resolve_manifest_path(base_dir: &Path, path: &str) -> Result<PathBuf> {
    let parsed = PathBuf::from(path);
    if parsed.is_absolute() {
        Ok(parsed)
    } else {
        Ok(base_dir.join(parsed))
    }
}

#[derive(Debug, Serialize)]
struct QuantPackBuildAllManifest {
    schema_version: u32,
    kind: String,
    profile: QuantPlanProfile,
    stages: usize,
    source: String,
    model_id_prefix: String,
    quantize: QuantPackQuantizeReproducibility,
    package_build: QuantPackPackageReproducibility,
    profile_request: QuantPackDecodeProfileRequest,
    rank_runtime_shape: QuantPackRankRuntimeShape,
    plan: String,
    ctx_size: u32,
    n_gpu_layers: i32,
    cache_type_k: String,
    cache_type_v: String,
    activation_wire_dtype: String,
    candidates: Vec<QuantPackBuildAllCandidate>,
    rank: String,
    next_steps: QuantPackBuildAllNextSteps,
}

#[derive(Debug, Serialize)]
struct QuantPackBuildAllCandidate {
    candidate: String,
    run_dir: String,
    manifest: String,
    artifacts: QuantPackBuildAllCandidateArtifacts,
    readiness: QuantPackBuildAllCandidateReadiness,
}

#[derive(Debug, Serialize)]
struct QuantPackBuildAllCandidateArtifacts {
    quantized_model: String,
    package: String,
    preflight: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_profile: Option<String>,
    evidence_dir: String,
    certification: String,
}

#[derive(Debug, Serialize)]
struct QuantPackBuildAllCandidateReadiness {
    build_artifacts_complete: bool,
    decode_profile_attached: bool,
    certification_present: bool,
    missing_artifacts: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct QuantPackBuildAllNextSteps {
    evidence_plan_all: QuantPackNextCommand,
}

#[derive(Debug, Clone, Serialize)]
struct QuantPackNextCommand {
    id: String,
    description: String,
    requires: Vec<String>,
    argv: Vec<String>,
    shell: String,
    outputs: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct QuantPlanCandidateList {
    candidates: Vec<QuantPlanCandidateId>,
}

#[derive(Debug, Deserialize)]
struct QuantPlanCandidateId {
    id: String,
}

#[derive(Debug, Deserialize)]
struct QuantPackBuildManifestInventory {
    quantized_model: String,
    package: String,
    preflight: String,
    #[serde(default)]
    decode_profile: Option<String>,
}

fn run_quant_pack_build_all(args: QuantPackBuildAllArgs) -> Result<()> {
    fs::create_dir_all(&args.out_dir).with_context(|| {
        format!(
            "create quant-pack build-all output directory {}",
            args.out_dir.display()
        )
    })?;
    let plan_path = args.out_dir.join("quant-plan.json");
    crate::quant_plan::run_quant_plan(QuantPlanArgs {
        source: args.source.clone(),
        profile: args.profile,
        stages: args.stages,
        out: Some(plan_path.clone()),
    })?;
    let candidates = selected_candidate_ids(&plan_path, &args.candidates)?;
    let mut candidate_manifests = Vec::new();
    let mut rank_inputs = Vec::new();
    let quantize_repro = QuantPackQuantizeReproducibility {
        llama_quantize: args.llama_quantize.display().to_string(),
        nthreads: args.nthreads,
        keep_split: args.keep_split,
    };
    let package_repro = QuantPackPackageReproducibility {
        verify_sha256: args.verify_sha256,
        projectors: display_paths(&args.projectors),
        after_artifact_command: args
            .after_artifact_command
            .as_ref()
            .map(|path| path.display().to_string()),
    };
    let profile_request = QuantPackDecodeProfileRequest {
        enabled: args.decode_profile,
        timing_source: "local-stage".to_string(),
        phase: "decode".to_string(),
        existing_kv_tokens: args.profile_existing_kv_tokens,
        generated_tokens: 1,
        batch_size: 1,
        kv_type: "f16".to_string(),
        warmup_samples: args.profile_warmup_samples,
        samples: args.profile_samples,
    };
    let rank_runtime_shape = QuantPackRankRuntimeShape {
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        cache_type_k: args.cache_type_k.clone(),
        cache_type_v: args.cache_type_v.clone(),
        activation_wire_dtype: args.activation_wire_dtype.clone(),
    };

    for candidate in candidates {
        let run_dir = args.out_dir.join(&candidate);
        run_quant_pack_build(QuantPackBuildArgs {
            source: args.source.clone(),
            profile: args.profile,
            stages: args.stages,
            candidate: candidate.clone(),
            out_dir: run_dir.clone(),
            llama_quantize: args.llama_quantize.clone(),
            quantized_model_out: None,
            package_dir: None,
            model_id: format!("{}:{candidate}", args.model_id_prefix),
            source_repo: args.source_repo.clone(),
            source_revision: args.source_revision.clone(),
            source_file: args.source_file.clone(),
            projectors: args.projectors.clone(),
            after_artifact_command: args.after_artifact_command.clone(),
            nthreads: args.nthreads,
            keep_split: args.keep_split,
            verify_sha256: args.verify_sha256,
            decode_profile: args.decode_profile,
            profile_existing_kv_tokens: args.profile_existing_kv_tokens,
            profile_warmup_samples: args.profile_warmup_samples,
            profile_samples: args.profile_samples,
        })?;
        let manifest = run_dir.join("quant-pack-build.json");
        rank_inputs.push(run_dir.clone());
        candidate_manifests.push(build_all_candidate_summary(candidate, &run_dir, &manifest)?);
    }

    let rank_path = args.out_dir.join("quant-pack-rank.json");
    rank::run_quant_pack_rank(QuantPackRankArgs {
        runs: rank_inputs,
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        cache_type_k: args.cache_type_k.clone(),
        cache_type_v: args.cache_type_v.clone(),
        activation_wire_dtype: args.activation_wire_dtype.clone(),
        out: Some(rank_path.clone()),
    })?;

    let manifest_path = args.out_dir.join("quant-pack-build-all.json");
    let next_steps = build_all_next_steps(
        &manifest_path,
        &args.out_dir,
        args.stages,
        candidate_manifests.len(),
    );
    let manifest = QuantPackBuildAllManifest {
        schema_version: 1,
        kind: "skippy_quant_pack_build_all".to_string(),
        profile: args.profile,
        stages: args.stages,
        source: args.source.display().to_string(),
        model_id_prefix: args.model_id_prefix,
        quantize: quantize_repro,
        package_build: package_repro,
        profile_request,
        rank_runtime_shape,
        plan: plan_path.display().to_string(),
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        cache_type_k: args.cache_type_k,
        cache_type_v: args.cache_type_v,
        activation_wire_dtype: args.activation_wire_dtype,
        candidates: candidate_manifests,
        rank: rank_path.display().to_string(),
        next_steps,
    };
    write_json_file(&manifest_path, &manifest)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn build_all_candidate_summary(
    candidate: String,
    run_dir: &Path,
    manifest_path: &Path,
) -> Result<QuantPackBuildAllCandidate> {
    let manifest = read_json::<QuantPackBuildManifestInventory>(manifest_path)?;
    let artifacts = QuantPackBuildAllCandidateArtifacts {
        quantized_model: manifest.quantized_model.clone(),
        package: manifest.package.clone(),
        preflight: manifest.preflight.clone(),
        decode_profile: manifest.decode_profile.clone(),
        evidence_dir: run_dir.join("evidence").display().to_string(),
        certification: run_dir.join("certification.json").display().to_string(),
    };
    let readiness = candidate_readiness(run_dir, manifest_path, &artifacts);
    Ok(QuantPackBuildAllCandidate {
        candidate,
        run_dir: run_dir.display().to_string(),
        manifest: manifest_path.display().to_string(),
        artifacts,
        readiness,
    })
}

fn candidate_readiness(
    run_dir: &Path,
    manifest_path: &Path,
    artifacts: &QuantPackBuildAllCandidateArtifacts,
) -> QuantPackBuildAllCandidateReadiness {
    let mut missing = Vec::new();
    push_missing_artifact(&mut missing, manifest_path, "manifest");
    push_missing_artifact(
        &mut missing,
        &resolve_build_manifest_path(run_dir, &artifacts.quantized_model),
        "quantized_model",
    );
    push_missing_artifact(
        &mut missing,
        &resolve_build_manifest_path(run_dir, &artifacts.package),
        "package",
    );
    push_missing_artifact(
        &mut missing,
        &resolve_build_manifest_path(run_dir, &artifacts.preflight),
        "preflight",
    );
    if let Some(decode_profile) = artifacts.decode_profile.as_deref() {
        push_missing_artifact(
            &mut missing,
            &resolve_build_manifest_path(run_dir, decode_profile),
            "decode_profile",
        );
    }
    QuantPackBuildAllCandidateReadiness {
        build_artifacts_complete: missing.is_empty(),
        decode_profile_attached: artifacts.decode_profile.is_some(),
        certification_present: Path::new(&artifacts.certification).exists(),
        missing_artifacts: missing,
    }
}

fn push_missing_artifact(missing: &mut Vec<String>, path: &Path, label: &str) {
    if !path.exists() {
        missing.push(format!("{label}:{}", path.display()));
    }
}

fn resolve_build_manifest_path(run_dir: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        run_dir.join(path)
    }
}

fn build_all_next_steps(
    manifest_path: &Path,
    out_dir: &Path,
    stages: usize,
    candidate_count: usize,
) -> QuantPackBuildAllNextSteps {
    let top_ranked = recommended_top_ranked_count(candidate_count);
    let evidence_plan = out_dir.join("evidence-plan-all.json");
    let evidence_script = out_dir.join("run-evidence.sh");
    let argv = vec![
        "skippy-model-package".to_string(),
        "quant-pack".to_string(),
        "evidence-plan-all".to_string(),
        manifest_path.display().to_string(),
        "--hosts".to_string(),
        host_placeholder(stages),
        "--top-ranked".to_string(),
        top_ranked.to_string(),
        "--out".to_string(),
        evidence_plan.display().to_string(),
        "--script-out".to_string(),
        evidence_script.display().to_string(),
    ];
    QuantPackBuildAllNextSteps {
        evidence_plan_all: QuantPackNextCommand {
            id: "evidence-plan-all".to_string(),
            description:
                "Generate skippy-bench and certification evidence commands for top-ranked valid candidates."
                    .to_string(),
            requires: vec![
                "replace --hosts placeholder with one reachable host per stage".to_string(),
                "pass --splits if the lab topology should not use inferred even splits".to_string(),
            ],
            shell: shell_command(&argv),
            outputs: vec![
                evidence_plan.display().to_string(),
                evidence_script.display().to_string(),
            ],
            argv,
        },
    }
}

fn recommended_top_ranked_count(candidate_count: usize) -> usize {
    candidate_count.clamp(1, 2)
}

fn host_placeholder(stages: usize) -> String {
    (0..stages.max(1))
        .map(|stage| format!("host-{stage}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn shell_command(argv: &[String]) -> String {
    argv.iter()
        .map(|arg| shell_quote(arg))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_quote(value: &str) -> String {
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '_' | '-' | ':' | ',' | '='))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn selected_candidate_ids(plan_path: &Path, requested: &[String]) -> Result<Vec<String>> {
    let plan = read_json::<QuantPlanCandidateList>(plan_path)?;
    let plan_ids = plan
        .candidates
        .into_iter()
        .map(|candidate| candidate.id)
        .collect::<Vec<_>>();
    if requested.is_empty() {
        return Ok(plan_ids);
    }
    for candidate in requested {
        if !plan_ids.iter().any(|id| id == candidate) {
            bail!("quant plan does not contain requested candidate {candidate:?}");
        }
    }
    Ok(requested.to_vec())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_pack_build_paths_are_stable_and_auditable() {
        let paths = quant_pack_build_paths(Path::new("/tmp/run"), "middle-compressed", None, None);

        assert_eq!(paths.plan, Path::new("/tmp/run/quant-plan.json"));
        assert_eq!(
            paths.tensor_type_file,
            Path::new("/tmp/run/quantize/tensor-types.txt")
        );
        assert_eq!(
            paths.agent_pack,
            Path::new("/tmp/run/quantize/agent-pack.json")
        );
        assert_eq!(
            paths.quantize_run,
            Path::new("/tmp/run/quantize/quantize-run.json")
        );
        assert_eq!(
            paths.quantized_model,
            Path::new("/tmp/run/middle-compressed.gguf")
        );
        assert_eq!(paths.package, Path::new("/tmp/run/package"));
        assert_eq!(paths.preflight, Path::new("/tmp/run/preflight.json"));
        assert_eq!(
            paths.decode_profile,
            Path::new("/tmp/run/decode-profile.json")
        );
        assert_eq!(paths.manifest, Path::new("/tmp/run/quant-pack-build.json"));
    }

    #[test]
    fn quant_pack_finalize_paths_bind_existing_quantize_run_to_package_outputs() {
        let paths = quant_pack_finalize_paths(
            Path::new("/tmp/run"),
            Path::new("/tmp/run/quantize/quantize-run.json"),
            None,
        )
        .expect("finalize paths");

        assert_eq!(
            paths.quantize_run,
            Path::new("/tmp/run/quantize/quantize-run.json")
        );
        assert_eq!(paths.quantize_dir, Path::new("/tmp/run/quantize"));
        assert_eq!(
            paths.tensor_type_file,
            Path::new("/tmp/run/quantize/tensor-types.txt")
        );
        assert_eq!(
            paths.agent_pack,
            Path::new("/tmp/run/quantize/agent-pack.json")
        );
        assert_eq!(paths.package, Path::new("/tmp/run/package"));
        assert_eq!(paths.preflight, Path::new("/tmp/run/preflight.json"));
        assert_eq!(paths.manifest, Path::new("/tmp/run/quant-pack-build.json"));
    }

    #[test]
    fn finalize_source_identity_records_original_source_and_distribution() {
        let args = QuantPackFinalizeArgs {
            quantize_run: PathBuf::from("/tmp/run/quantize/quantize-run.json"),
            out_dir: PathBuf::from("/tmp/run"),
            stages: 4,
            package_dir: None,
            model_id: "org/repo:candidate".to_string(),
            source_repo: Some("org/repo".to_string()),
            source_revision: Some("abc123".to_string()),
            source_file: Some("UD-Q4/model-00001-of-00006.gguf".to_string()),
            projectors: Vec::new(),
            after_artifact_command: None,
            verify_sha256: false,
            decode_profile: false,
            profile_existing_kv_tokens: 8192,
            profile_warmup_samples: 3,
            profile_samples: 20,
            plan: Some(PathBuf::from("/tmp/run/quant-plan.json")),
            reuse_package_if_present: true,
        };

        let identity = finalize_source_identity_manifest(&args, "/models/source.gguf");

        assert_eq!(identity.model_id, "org/repo:candidate");
        assert_eq!(identity.path, "/models/source.gguf");
        assert_eq!(
            identity.canonical_ref.as_deref(),
            Some("org/repo@abc123/UD-Q4/model-00001-of-00006.gguf")
        );
        assert_eq!(identity.distribution_id.as_deref(), Some("model"));
    }

    #[test]
    fn decode_profile_request_records_default_measurement_shape() {
        let args = QuantPackBuildArgs {
            source: PathBuf::from("/models/source.gguf"),
            profile: QuantPlanProfile::CodingAgent,
            stages: 2,
            candidate: "middle-compressed".to_string(),
            out_dir: PathBuf::from("/tmp/run"),
            llama_quantize: PathBuf::from("/opt/llama-quantize"),
            quantized_model_out: None,
            package_dir: None,
            model_id: "org/repo:middle-compressed".to_string(),
            source_repo: Some("org/repo".to_string()),
            source_revision: Some("abc123".to_string()),
            source_file: Some("source.gguf".to_string()),
            projectors: Vec::new(),
            after_artifact_command: None,
            nthreads: Some(12),
            keep_split: true,
            verify_sha256: true,
            decode_profile: true,
            profile_existing_kv_tokens: 32_768,
            profile_warmup_samples: 5,
            profile_samples: 30,
        };

        let request = decode_profile_request(&args);

        assert!(request.enabled);
        assert_eq!(request.timing_source, "local-stage");
        assert_eq!(request.phase, "decode");
        assert_eq!(request.existing_kv_tokens, 32_768);
        assert_eq!(request.generated_tokens, 1);
        assert_eq!(request.batch_size, 1);
        assert_eq!(request.kv_type, "f16");
        assert_eq!(request.warmup_samples, 5);
        assert_eq!(request.samples, 30);
    }

    #[test]
    fn build_manifest_keeps_artifact_paths_and_reproducibility_blocks() {
        let manifest = QuantPackBuildManifest {
            schema_version: 1,
            kind: "skippy_quant_pack_build".to_string(),
            profile: QuantPlanProfile::CodingAgent,
            stages: 2,
            candidate: "middle-compressed".to_string(),
            source: "/models/source.gguf".to_string(),
            source_identity: QuantPackSourceIdentity {
                model_id: "org/repo:middle-compressed".to_string(),
                path: "/models/source.gguf".to_string(),
                repo: Some("org/repo".to_string()),
                revision: Some("abc123".to_string()),
                primary_file: Some("source.gguf".to_string()),
                canonical_ref: Some("org/repo@abc123:source.gguf".to_string()),
                distribution_id: Some("source".to_string()),
            },
            quantize: QuantPackQuantizeReproducibility {
                llama_quantize: "/opt/llama-quantize".to_string(),
                nthreads: Some(12),
                keep_split: true,
            },
            package_build: QuantPackPackageReproducibility {
                verify_sha256: true,
                projectors: vec!["/models/mmproj.gguf".to_string()],
                after_artifact_command: Some("/opt/hook".to_string()),
            },
            profile_request: QuantPackDecodeProfileRequest {
                enabled: true,
                timing_source: "local-stage".to_string(),
                phase: "decode".to_string(),
                existing_kv_tokens: 8192,
                generated_tokens: 1,
                batch_size: 1,
                kv_type: "f16".to_string(),
                warmup_samples: 3,
                samples: 20,
            },
            plan: "quant-plan.json".to_string(),
            tensor_type_file: "quantize/tensor-types.txt".to_string(),
            agent_pack: "quantize/agent-pack.json".to_string(),
            quantize_run: "quantize/quantize-run.json".to_string(),
            quantized_model: "middle-compressed.gguf".to_string(),
            package: "package".to_string(),
            preflight: "preflight.json".to_string(),
            decode_profile: Some("decode-profile.json".to_string()),
            preflight_verified_sha256: true,
        };

        let json = serde_json::to_value(&manifest).expect("serialize manifest");

        assert_eq!(json["package"], "package");
        assert_eq!(json["source_identity"]["revision"], "abc123");
        assert_eq!(json["quantize"]["llama_quantize"], "/opt/llama-quantize");
        assert_eq!(json["quantize"]["keep_split"], true);
        assert_eq!(json["package_build"]["verify_sha256"], true);
        assert_eq!(json["profile_request"]["existing_kv_tokens"], 8192);
    }

    #[test]
    fn build_all_next_steps_point_to_top_ranked_evidence_plan() {
        let next_steps = build_all_next_steps(
            Path::new("/tmp/sweep/quant-pack-build-all.json"),
            Path::new("/tmp/sweep"),
            3,
            5,
        );
        let command = next_steps.evidence_plan_all;

        assert_eq!(command.id, "evidence-plan-all");
        assert_eq!(
            command.argv,
            [
                "skippy-model-package",
                "quant-pack",
                "evidence-plan-all",
                "/tmp/sweep/quant-pack-build-all.json",
                "--hosts",
                "host-0,host-1,host-2",
                "--top-ranked",
                "2",
                "--out",
                "/tmp/sweep/evidence-plan-all.json",
                "--script-out",
                "/tmp/sweep/run-evidence.sh"
            ]
        );
        assert_eq!(command.shell, command.argv.join(" "));
        assert!(command.requires.contains(
            &"replace --hosts placeholder with one reachable host per stage".to_string()
        ));
        assert_eq!(
            command.outputs,
            [
                "/tmp/sweep/evidence-plan-all.json",
                "/tmp/sweep/run-evidence.sh"
            ]
        );
    }

    #[test]
    fn build_all_next_steps_never_request_zero_candidates() {
        assert_eq!(recommended_top_ranked_count(0), 1);
        assert_eq!(recommended_top_ranked_count(1), 1);
        assert_eq!(recommended_top_ranked_count(4), 2);
    }

    #[test]
    fn build_all_candidate_summary_reports_complete_artifacts() {
        let dir = unique_test_dir("candidate-summary-complete");
        let run_dir = dir.join("middle-compressed");
        fs::create_dir_all(run_dir.join("package")).expect("create package dir");
        fs::write(run_dir.join("model.gguf"), b"gguf").expect("write model");
        fs::write(run_dir.join("preflight.json"), b"{}").expect("write preflight");
        fs::write(run_dir.join("decode-profile.json"), b"{}").expect("write profile");
        let manifest = run_dir.join("quant-pack-build.json");
        fs::write(
            &manifest,
            r#"{
  "quantized_model": "model.gguf",
  "package": "package",
  "preflight": "preflight.json",
  "decode_profile": "decode-profile.json"
}"#,
        )
        .expect("write manifest");

        let summary =
            build_all_candidate_summary("middle-compressed".to_string(), &run_dir, &manifest)
                .expect("candidate summary");

        assert_eq!(summary.artifacts.quantized_model, "model.gguf");
        assert_eq!(
            summary.artifacts.decode_profile.as_deref(),
            Some("decode-profile.json")
        );
        assert!(summary.readiness.build_artifacts_complete);
        assert!(summary.readiness.decode_profile_attached);
        assert!(!summary.readiness.certification_present);
        assert!(summary.readiness.missing_artifacts.is_empty());
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn candidate_readiness_reports_missing_decode_profile_artifact() {
        let dir = unique_test_dir("candidate-summary-missing");
        fs::create_dir_all(&dir).expect("create run dir");
        fs::write(dir.join("model.gguf"), b"gguf").expect("write model");
        fs::write(dir.join("preflight.json"), b"{}").expect("write preflight");
        fs::create_dir_all(dir.join("package")).expect("create package dir");
        let manifest = dir.join("quant-pack-build.json");
        fs::write(&manifest, b"{}").expect("write manifest");
        let artifacts = QuantPackBuildAllCandidateArtifacts {
            quantized_model: "model.gguf".to_string(),
            package: "package".to_string(),
            preflight: "preflight.json".to_string(),
            decode_profile: Some("decode-profile.json".to_string()),
            evidence_dir: dir.join("evidence").display().to_string(),
            certification: dir.join("certification.json").display().to_string(),
        };

        let readiness = candidate_readiness(&dir, &manifest, &artifacts);

        assert!(!readiness.build_artifacts_complete);
        assert!(readiness.decode_profile_attached);
        assert_eq!(readiness.missing_artifacts.len(), 1);
        assert!(readiness.missing_artifacts[0].contains("decode_profile:"));
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn selected_candidate_ids_default_to_plan_order() {
        let dir =
            std::env::temp_dir().join(format!("skippy-quant-pack-test-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("create temp dir");
        let plan = dir.join("quant-plan.json");
        fs::write(
            &plan,
            r#"{"candidates":[{"id":"baseline-source-quant"},{"id":"middle-compressed"}]}"#,
        )
        .expect("write plan");

        let selected = selected_candidate_ids(&plan, &[]).expect("select all candidates");

        assert_eq!(selected, ["baseline-source-quant", "middle-compressed"]);
        let _ = fs::remove_file(plan);
        let _ = fs::remove_dir(dir);
    }

    #[test]
    fn selected_candidate_ids_reject_unknown_candidate() {
        let dir = std::env::temp_dir().join(format!(
            "skippy-quant-pack-missing-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        let plan = dir.join("quant-plan.json");
        fs::write(&plan, r#"{"candidates":[{"id":"middle-compressed"}]}"#).expect("write plan");

        let error = selected_candidate_ids(&plan, &["missing".to_string()])
            .expect_err("unknown candidate should fail");

        assert!(error.to_string().contains("missing"));
        let _ = fs::remove_file(plan);
        let _ = fs::remove_dir(dir);
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "skippy-quant-pack-{name}-{}-{nanos}",
            std::process::id()
        ))
    }
}
