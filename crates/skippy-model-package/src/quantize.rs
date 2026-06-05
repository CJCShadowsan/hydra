use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use anyhow::{Context, Result, bail};
use model_ref::split_gguf_shard_info;
use serde::{Deserialize, Serialize};

#[derive(Debug, clap::Args)]
pub(crate) struct QuantizeArgs {
    pub(crate) source: PathBuf,
    #[arg(long)]
    pub(crate) plan: PathBuf,
    #[arg(long)]
    pub(crate) candidate: String,
    #[arg(long)]
    pub(crate) out_dir: PathBuf,
    #[arg(long)]
    pub(crate) llama_quantize: Option<PathBuf>,
    #[arg(long)]
    pub(crate) quantized_model_out: Option<PathBuf>,
    #[arg(long)]
    pub(crate) emit_only: bool,
    #[arg(long)]
    pub(crate) keep_split: bool,
    #[arg(long)]
    pub(crate) nthreads: Option<u32>,
}

pub(crate) struct QuantizeRunOutput {
    pub(crate) quantized_model: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct QuantPlanReport {
    source: QuantPlanSource,
    profile: String,
    candidates: Vec<QuantPlanCandidate>,
}

#[derive(Debug, Deserialize)]
struct QuantPlanSource {
    path: String,
    sha256: String,
    inferred_source_quant: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct QuantPlanCandidate {
    id: String,
    layout_hash: String,
    name: String,
    status: String,
    strategy: String,
    default_quant: String,
    groups: Vec<QuantGroup>,
    stage_hints: Vec<StageHint>,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct QuantGroup {
    name: String,
    quant: String,
    selector: QuantSelector,
    reason: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum QuantSelector {
    Role { roles: Vec<String> },
    LayerRange { start: u32, end: u32 },
    TensorNamePattern { patterns: Vec<String> },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct StageHint {
    stage_index: usize,
    layer_start: u32,
    layer_end: u32,
    tensor_bytes: u64,
    role: String,
}

#[derive(Debug, Serialize)]
struct QuantizeManifest {
    schema_version: u32,
    kind: String,
    source: QuantizeSource,
    candidate: QuantPlanCandidate,
    tensor_type_file: String,
    tensor_type_entry_count: usize,
    tensor_type_entries: Vec<String>,
    keep_split: bool,
    requested_quantized_model: Option<String>,
    quantized_model: Option<String>,
    command: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct QuantizeSource {
    path: String,
    sha256: String,
    inferred_source_quant: String,
}

#[derive(Debug, Serialize)]
struct AgentPackMetadata {
    schema_version: u32,
    profile: String,
    pack_id: String,
    source: QuantizeSource,
    quant_layout: QuantLayoutMetadata,
}

#[derive(Debug, Serialize)]
struct QuantLayoutMetadata {
    strategy: String,
    #[serde(rename = "default")]
    default_quant: String,
    layout_hash: String,
    groups: Vec<QuantGroup>,
}

pub(crate) fn run_quantize(args: QuantizeArgs) -> Result<QuantizeRunOutput> {
    let plan = read_plan(&args.plan)?;
    let candidate = select_candidate(&plan, &args.candidate)?;
    fs::create_dir_all(&args.out_dir).with_context(|| {
        format!(
            "create quantize output directory {}",
            args.out_dir.display()
        )
    })?;

    let tensor_type_path = args.out_dir.join("tensor-types.txt");
    let tensor_type_entries = tensor_type_entries(&candidate)?;
    fs::write(&tensor_type_path, tensor_type_entries.join("\n") + "\n")
        .with_context(|| format!("write tensor type overrides {}", tensor_type_path.display()))?;

    let agent_pack_path = args.out_dir.join("agent-pack.json");
    let agent_pack = agent_pack_metadata(&plan, &candidate);
    write_json_file(&agent_pack_path, &agent_pack)?;

    let quantized_model = args
        .quantized_model_out
        .clone()
        .unwrap_or_else(|| args.out_dir.join(format!("{}.gguf", candidate.id)));
    let command = if args.emit_only {
        None
    } else {
        let llama_quantize = args.llama_quantize.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "--llama-quantize is required unless --emit-only is set; pass a llama-quantize binary to perform real quantization"
            )
        })?;
        let command = quantize_command(
            llama_quantize,
            &args.source,
            &quantized_model,
            &candidate,
            &tensor_type_path,
            args.keep_split,
            args.nthreads,
        );
        run_quantize_command(&command)?;
        Some(command)
    };
    let actual_quantized_model = if args.emit_only {
        None
    } else {
        Some(resolve_quantized_model_output(
            &quantized_model,
            args.keep_split,
        )?)
    };

    let manifest = QuantizeManifest {
        schema_version: 1,
        kind: "skippy_quantize_run".to_string(),
        source: QuantizeSource {
            path: args.source.display().to_string(),
            sha256: plan.source.sha256,
            inferred_source_quant: plan.source.inferred_source_quant,
        },
        candidate,
        tensor_type_file: tensor_type_path.display().to_string(),
        tensor_type_entry_count: tensor_type_entries.len(),
        tensor_type_entries,
        keep_split: args.keep_split,
        requested_quantized_model: (!args.emit_only).then(|| quantized_model.display().to_string()),
        quantized_model: actual_quantized_model
            .as_ref()
            .map(|path| path.display().to_string()),
        command,
    };
    let manifest_path = args.out_dir.join("quantize-run.json");
    write_json_file(&manifest_path, &manifest)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(QuantizeRunOutput {
        quantized_model: actual_quantized_model,
    })
}

fn read_plan(path: &Path) -> Result<QuantPlanReport> {
    let contents = fs::read(path).with_context(|| format!("read quant plan {}", path.display()))?;
    serde_json::from_slice(&contents)
        .with_context(|| format!("parse quant plan {}", path.display()))
}

fn select_candidate(plan: &QuantPlanReport, candidate_id: &str) -> Result<QuantPlanCandidate> {
    plan.candidates
        .iter()
        .find(|candidate| candidate.id == candidate_id)
        .cloned()
        .with_context(|| format!("quant plan does not contain candidate {candidate_id:?}"))
}

fn tensor_type_entries(candidate: &QuantPlanCandidate) -> Result<Vec<String>> {
    let mut entries = Vec::new();
    for group in &candidate.groups {
        let target = tensor_type_quant(&group.quant)?;
        match &group.selector {
            QuantSelector::Role { roles } => {
                for role in roles {
                    entries.extend(role_tensor_patterns(role, target)?);
                }
            }
            QuantSelector::LayerRange { start, end } => {
                if start >= end {
                    bail!("candidate group {} has an empty layer range", group.name);
                }
                for layer in *start..*end {
                    entries.push(format!("blk\\.{layer}\\..*\\.weight={target}"));
                }
            }
            QuantSelector::TensorNamePattern { patterns } => {
                for pattern in patterns {
                    entries.push(format!("{pattern}={target}"));
                }
            }
        }
    }
    dedup_preserving_order(&mut entries);
    Ok(entries)
}

fn dedup_preserving_order(entries: &mut Vec<String>) {
    let mut seen = std::collections::BTreeSet::new();
    entries.retain(|entry| seen.insert(entry.clone()));
}

fn role_tensor_patterns(role: &str, target: &str) -> Result<Vec<String>> {
    match role {
        "embedding" => Ok(vec![format!("token_embd\\.weight={target}")]),
        "output" => Ok(vec![format!("output\\.weight={target}")]),
        "final_norm" => Ok(Vec::new()),
        other => bail!("unsupported role selector {other:?} for tensor type overrides"),
    }
}

fn tensor_type_quant(quant: &str) -> Result<&'static str> {
    match quant {
        "F16" => Ok("f16"),
        "BF16" => Ok("bf16"),
        "Q8_0" => Ok("q8_0"),
        "Q6_K" => Ok("q6_k"),
        "Q5_K" | "Q5_K_M" | "Q5_K_S" => Ok("q5_k"),
        "Q4_K" | "Q4_K_M" | "Q4_K_S" => Ok("q4_k"),
        "Q3_K" | "Q3_K_M" | "Q3_K_S" => Ok("q3_k"),
        "Q2_K" => Ok("q2_k"),
        other => bail!("unsupported quant target {other:?} for llama tensor overrides"),
    }
}

fn agent_pack_metadata(
    plan: &QuantPlanReport,
    candidate: &QuantPlanCandidate,
) -> AgentPackMetadata {
    AgentPackMetadata {
        schema_version: 1,
        profile: plan.profile.clone(),
        pack_id: candidate.id.clone(),
        source: QuantizeSource {
            path: plan.source.path.clone(),
            sha256: plan.source.sha256.clone(),
            inferred_source_quant: plan.source.inferred_source_quant.clone(),
        },
        quant_layout: QuantLayoutMetadata {
            strategy: candidate.strategy.clone(),
            default_quant: candidate.default_quant.clone(),
            layout_hash: candidate.layout_hash.clone(),
            groups: candidate.groups.clone(),
        },
    }
}

fn quantize_command(
    llama_quantize: &Path,
    source: &Path,
    quantized_model: &Path,
    candidate: &QuantPlanCandidate,
    tensor_type_path: &Path,
    keep_split: bool,
    nthreads: Option<u32>,
) -> Vec<String> {
    let mut command = vec![
        llama_quantize.display().to_string(),
        "--allow-requantize".to_string(),
        "--tensor-type-file".to_string(),
        tensor_type_path.display().to_string(),
    ];
    if keep_split {
        command.push("--keep-split".to_string());
    }
    command.extend([
        source.display().to_string(),
        quantized_model.display().to_string(),
        candidate.default_quant.clone(),
    ]);
    if let Some(nthreads) = nthreads {
        command.push(nthreads.to_string());
    }
    command
}

fn run_quantize_command(command: &[String]) -> Result<()> {
    let (program, args) = command
        .split_first()
        .context("quantize command is unexpectedly empty")?;
    let status = ProcessCommand::new(program)
        .args(args)
        .status()
        .with_context(|| format!("run {}", command.join(" ")))?;
    if !status.success() {
        bail!("llama quantize command failed with status {status}");
    }
    Ok(())
}

fn resolve_quantized_model_output(requested: &Path, keep_split: bool) -> Result<PathBuf> {
    if !keep_split || requested.exists() {
        return Ok(requested.to_path_buf());
    }
    let output_prefix = keep_split_output_prefix(requested);
    discover_first_split_output(&output_prefix)
}

fn keep_split_output_prefix(requested: &Path) -> PathBuf {
    if requested
        .extension()
        .is_some_and(|extension| extension == "gguf")
    {
        return requested.with_extension("");
    }
    requested.to_path_buf()
}

fn discover_first_split_output(output_prefix: &Path) -> Result<PathBuf> {
    let parent = output_prefix.parent().unwrap_or_else(|| Path::new("."));
    let prefix = output_prefix
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| {
            format!(
                "quantized split output prefix has no file name: {}",
                output_prefix.display()
            )
        })?;
    let mut matches = fs::read_dir(parent)
        .with_context(|| format!("read quantized split output directory {}", parent.display()))?
        .filter_map(|entry| first_shard_match(entry.ok()?.path(), prefix))
        .collect::<Vec<_>>();
    matches.sort();
    match matches.as_slice() {
        [path] => Ok(path.clone()),
        [] => bail!(
            "llama quantize --keep-split did not produce first shard matching {}-00001-of-*.gguf",
            output_prefix.display()
        ),
        _ => bail!(
            "llama quantize --keep-split produced ambiguous first shards for {}: {}",
            output_prefix.display(),
            matches
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn first_shard_match(path: PathBuf, expected_prefix: &str) -> Option<PathBuf> {
    let file_name = path.file_name()?.to_str()?;
    let shard = split_gguf_shard_info(file_name)?;
    (shard.prefix == expected_prefix && shard.part == "00001").then_some(path)
}

fn write_json_file(path: &Path, value: &impl Serialize) -> Result<()> {
    let json = serde_json::to_vec_pretty(value)?;
    fs::write(path, json).with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_split_output_prefix_strips_gguf_extension() {
        assert_eq!(
            keep_split_output_prefix(Path::new("/tmp/candidate.gguf")),
            Path::new("/tmp/candidate")
        );
        assert_eq!(
            keep_split_output_prefix(Path::new("/tmp/candidate")),
            Path::new("/tmp/candidate")
        );
    }

    #[test]
    fn resolve_quantized_output_uses_requested_file_when_present() {
        let dir = unique_test_dir("present-output");
        fs::create_dir_all(&dir).expect("create test dir");
        let requested = dir.join("candidate.gguf");
        fs::write(&requested, b"gguf").expect("write requested output");

        let resolved =
            resolve_quantized_model_output(&requested, true).expect("resolve requested output");

        assert_eq!(resolved, requested);
        fs::remove_dir_all(dir).expect("remove test dir");
    }

    #[test]
    fn resolve_quantized_output_discovers_keep_split_first_shard() {
        let dir = unique_test_dir("split-output");
        fs::create_dir_all(&dir).expect("create test dir");
        let requested = dir.join("candidate.gguf");
        for part in 1..=3 {
            fs::write(
                dir.join(format!("candidate-{part:05}-of-00003.gguf")),
                b"gguf",
            )
            .expect("write split output");
        }

        let resolved =
            resolve_quantized_model_output(&requested, true).expect("resolve split output");

        assert_eq!(resolved, dir.join("candidate-00001-of-00003.gguf"));
        fs::remove_dir_all(dir).expect("remove test dir");
    }

    #[test]
    fn resolve_quantized_output_rejects_ambiguous_first_shards() {
        let dir = unique_test_dir("ambiguous-split-output");
        fs::create_dir_all(&dir).expect("create test dir");
        let requested = dir.join("candidate.gguf");
        fs::write(dir.join("candidate-00001-of-00002.gguf"), b"gguf").expect("write first shard");
        fs::write(dir.join("candidate-00001-of-00003.gguf"), b"gguf")
            .expect("write ambiguous first shard");

        let error = resolve_quantized_model_output(&requested, true).unwrap_err();

        assert!(error.to_string().contains("ambiguous first shards"));
        fs::remove_dir_all(dir).expect("remove test dir");
    }

    #[test]
    fn tensor_type_entries_cover_roles_and_layer_ranges() {
        let candidate = candidate_with_groups(vec![
            group(
                "embedding-and-output",
                "Q6_K",
                QuantSelector::Role {
                    roles: vec![
                        "embedding".to_string(),
                        "final_norm".to_string(),
                        "output".to_string(),
                    ],
                },
            ),
            group(
                "middle",
                "Q3_K_M",
                QuantSelector::LayerRange { start: 2, end: 4 },
            ),
        ]);

        let entries = tensor_type_entries(&candidate).expect("tensor entries");

        assert_eq!(
            entries,
            [
                "token_embd\\.weight=q6_k",
                "output\\.weight=q6_k",
                "blk\\.2\\..*\\.weight=q3_k",
                "blk\\.3\\..*\\.weight=q3_k",
            ]
        );
    }

    #[test]
    fn tensor_name_patterns_keep_precedence_before_broad_layer_ranges() {
        let candidate = candidate_with_groups(vec![
            group(
                "moe-experts",
                "Q4_K_M",
                QuantSelector::TensorNamePattern {
                    patterns: vec![r"blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight".to_string()],
                },
            ),
            group(
                "middle",
                "Q3_K_M",
                QuantSelector::LayerRange { start: 2, end: 3 },
            ),
        ]);

        let entries = tensor_type_entries(&candidate).expect("tensor entries");

        assert_eq!(
            entries,
            [
                r"blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight=q4_k",
                "blk\\.2\\..*\\.weight=q3_k",
            ]
        );
    }

    #[test]
    fn quantize_command_uses_layout_default_and_override_file() {
        let candidate = candidate_with_groups(Vec::new());

        let command = quantize_command(
            Path::new("/bin/llama-quantize"),
            Path::new("source.gguf"),
            Path::new("out.gguf"),
            &candidate,
            Path::new("tensor-types.txt"),
            true,
            Some(8),
        );

        assert_eq!(
            command,
            [
                "/bin/llama-quantize",
                "--allow-requantize",
                "--tensor-type-file",
                "tensor-types.txt",
                "--keep-split",
                "source.gguf",
                "out.gguf",
                "Q4_K_M",
                "8"
            ]
        );
    }

    fn candidate_with_groups(groups: Vec<QuantGroup>) -> QuantPlanCandidate {
        QuantPlanCandidate {
            id: "middle-compressed".to_string(),
            layout_hash: "hash".to_string(),
            name: "Middle compressed".to_string(),
            status: "experimental".to_string(),
            strategy: "stage-aware-middle-compressed".to_string(),
            default_quant: "Q4_K_M".to_string(),
            groups,
            stage_hints: Vec::new(),
            notes: Vec::new(),
        }
    }

    fn group(name: &str, quant: &str, selector: QuantSelector) -> QuantGroup {
        QuantGroup {
            name: name.to_string(),
            quant: quant.to_string(),
            selector,
            reason: "test".to_string(),
        }
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "skippy-quantize-{name}-{}-{nanos}",
            std::process::id()
        ))
    }
}
