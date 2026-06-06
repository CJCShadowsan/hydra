mod source;

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use serde::Serialize;
use sha2::{Digest, Sha256};
use skippy_ffi::{ABI_VERSION_MAJOR, ABI_VERSION_MINOR, ABI_VERSION_PATCH, TensorRole};
use skippy_runtime::TensorInfo;

use self::source::{SourceShardSummary, inspect_quant_source};

#[derive(Debug, clap::Args)]
pub(crate) struct QuantPlanArgs {
    pub(crate) source: PathBuf,
    #[arg(long, value_enum, default_value_t = QuantPlanProfile::CodingAgent)]
    pub(crate) profile: QuantPlanProfile,
    #[arg(long, default_value_t = 2)]
    pub(crate) stages: usize,
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum QuantPlanProfile {
    CodingAgent,
}

#[derive(Debug, Serialize)]
struct QuantPlanReport {
    schema_version: u32,
    kind: String,
    source: SourceModelSummary,
    toolchain: QuantPlanToolchain,
    profile: QuantPlanProfile,
    stage_count: usize,
    protected_band_width: u32,
    candidates: Vec<QuantLayoutCandidate>,
}

#[derive(Debug, Serialize)]
struct QuantPlanToolchain {
    skippy_model_package_version: String,
    skippy_abi_version: String,
    layout_hash_version: u32,
}

#[derive(Debug, Serialize)]
struct SourceModelSummary {
    path: String,
    sha256: String,
    sha256_kind: String,
    inferred_source_quant: String,
    shard_count: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    shards: Vec<SourceShardSummary>,
    layer_count: u32,
    tensor_count: usize,
    tensor_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct QuantLayoutCandidate {
    id: String,
    layout_hash: String,
    name: String,
    status: String,
    strategy: String,
    default_quant: String,
    groups: Vec<QuantGroup>,
    stage_hints: Vec<StageQuantHint>,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct QuantGroup {
    name: String,
    quant: String,
    selector: QuantSelector,
    reason: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum QuantSelector {
    Role { roles: Vec<String> },
    LayerRange { start: u32, end: u32 },
    TensorNamePattern { patterns: Vec<String> },
}

#[derive(Debug, Clone, Serialize)]
struct StageQuantHint {
    stage_index: usize,
    layer_start: u32,
    layer_end: u32,
    tensor_bytes: u64,
    role: String,
}

struct QuantPlanInput {
    source_path: PathBuf,
    source_sha256: String,
    source_sha256_kind: String,
    source_quant: String,
    default_quant: String,
    source_shards: Vec<SourceShardSummary>,
    profile: QuantPlanProfile,
    stage_count: usize,
    tensors: Vec<TensorInfo>,
}

pub(crate) fn run_quant_plan(args: QuantPlanArgs) -> Result<()> {
    reject_materialized_skippy_slice_source(&args.source)?;
    let source = inspect_quant_source(&args.source)?;
    let source_quant = infer_quant_from_path(&args.source);
    let input = QuantPlanInput {
        source_sha256: source.source_sha256,
        source_sha256_kind: source.source_sha256_kind,
        default_quant: default_quant_for_source(&source_quant).to_string(),
        source_quant,
        source_path: args.source,
        source_shards: source.shards,
        profile: args.profile,
        stage_count: args.stages,
        tensors: source.tensors,
    };
    let report = build_quant_plan(input)?;
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(out) = args.out {
        fs::write(&out, json).with_context(|| format!("write quant plan {}", out.display()))?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn reject_materialized_skippy_slice_source(path: &Path) -> Result<()> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if is_materialized_skippy_slice_name(name) {
        bail!(
            "quant-pack source {} looks like a materialized Skippy stage/tokenizer slice; use the original source GGUF or layer-package source instead of derived cache artifacts",
            path.display()
        );
    }
    Ok(())
}

fn is_materialized_skippy_slice_name(name: &str) -> bool {
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    has_materialized_slice_marker(stem, "-stage-")
        || has_materialized_slice_marker(stem, "-tokenizer-")
}

fn has_materialized_slice_marker(stem: &str, marker: &str) -> bool {
    let Some((_, suffix)) = stem.rsplit_once(marker) else {
        return false;
    };
    let mut parts = suffix.split('-');
    let numeric_parts = parts
        .by_ref()
        .take_while(|part| part.chars().all(|ch| ch.is_ascii_digit()))
        .count();
    numeric_parts >= 2
}

fn build_quant_plan(input: QuantPlanInput) -> Result<QuantPlanReport> {
    let layer_count = layer_count(&input.tensors)?;
    validate_stage_count(input.stage_count, layer_count)?;
    let protected_band_width = protected_band_width(layer_count);
    let stage_hints = stage_hints(&input.tensors, layer_count, input.stage_count);
    let candidates = candidate_layouts(
        &input.tensors,
        &input.source_quant,
        &input.default_quant,
        layer_count,
        protected_band_width,
        &stage_hints,
    );

    Ok(QuantPlanReport {
        schema_version: 1,
        kind: "skippy_quant_plan".to_string(),
        source: SourceModelSummary {
            path: input.source_path.display().to_string(),
            sha256: input.source_sha256,
            sha256_kind: input.source_sha256_kind,
            inferred_source_quant: input.source_quant,
            shard_count: input.source_shards.len().max(1),
            shards: input.source_shards,
            layer_count,
            tensor_count: input.tensors.len(),
            tensor_bytes: input.tensors.iter().map(|tensor| tensor.byte_size).sum(),
        },
        toolchain: QuantPlanToolchain {
            skippy_model_package_version: env!("CARGO_PKG_VERSION").to_string(),
            skippy_abi_version: format!(
                "{}.{}.{}",
                ABI_VERSION_MAJOR, ABI_VERSION_MINOR, ABI_VERSION_PATCH
            ),
            layout_hash_version: 4,
        },
        profile: input.profile,
        stage_count: input.stage_count,
        protected_band_width,
        candidates,
    })
}

fn validate_stage_count(stage_count: usize, layer_count: u32) -> Result<()> {
    if stage_count == 0 {
        bail!("--stages must be greater than zero");
    }
    if stage_count as u32 > layer_count {
        bail!("--stages {stage_count} exceeds model layer_count {layer_count}");
    }
    Ok(())
}

fn candidate_layouts(
    tensors: &[TensorInfo],
    source_quant: &str,
    default_quant: &str,
    layer_count: u32,
    protected_width: u32,
    stage_hints: &[StageQuantHint],
) -> Vec<QuantLayoutCandidate> {
    let boundary_groups =
        boundary_protection_groups(default_quant, source_quant, protected_width, layer_count);
    let has_moe = has_moe_tensor(tensors);
    let moe_groups = moe_sensitive_groups(tensors, source_quant);
    vec![
        baseline_candidate(default_quant, stage_hints),
        boundary_protected_candidate(
            default_quant,
            combine_groups(boundary_groups.clone(), moe_groups.clone()),
            stage_hints,
        ),
        middle_compressed_candidate(
            default_quant,
            source_quant,
            protected_width,
            layer_count,
            stage_hints,
            moe_groups.clone(),
        ),
        ffn_compressed_attention_protected_candidate(
            default_quant,
            source_quant,
            protected_width,
            layer_count,
            stage_hints,
            has_moe,
        ),
        stage_balanced_ffn_candidate(
            default_quant,
            source_quant,
            combine_groups(boundary_groups.clone(), moe_groups.clone()),
            protected_width,
            layer_count,
            stage_hints,
            has_moe,
        ),
        stage_balanced_ffn_part_candidate(
            StageBalancedFfnPartCandidateInput {
                default_quant,
                source_quant,
                groups: combine_groups(boundary_groups.clone(), moe_groups.clone()),
                protected_width,
                layer_count,
                stage_hints,
                has_moe,
            },
            "down",
        ),
        stage_balanced_ffn_part_candidate(
            StageBalancedFfnPartCandidateInput {
                default_quant,
                source_quant,
                groups: combine_groups(boundary_groups.clone(), moe_groups.clone()),
                protected_width,
                layer_count,
                stage_hints,
                has_moe,
            },
            "gate",
        ),
        stage_balanced_ffn_part_candidate(
            StageBalancedFfnPartCandidateInput {
                default_quant,
                source_quant,
                groups: combine_groups(boundary_groups.clone(), moe_groups.clone()),
                protected_width,
                layer_count,
                stage_hints,
                has_moe,
            },
            "up",
        ),
        stage_balanced_candidate(
            default_quant,
            source_quant,
            combine_groups(boundary_groups, moe_groups),
            protected_width,
            layer_count,
            stage_hints,
        ),
    ]
}

fn combine_groups(mut first: Vec<QuantGroup>, second: Vec<QuantGroup>) -> Vec<QuantGroup> {
    first.extend(second);
    first
}

fn baseline_candidate(source_quant: &str, stage_hints: &[StageQuantHint]) -> QuantLayoutCandidate {
    with_layout_hash(QuantLayoutCandidate {
        id: "baseline-source-quant".to_string(),
        layout_hash: String::new(),
        name: "Whole-model source quant baseline".to_string(),
        status: "experimental".to_string(),
        strategy: "baseline".to_string(),
        default_quant: source_quant.to_string(),
        groups: Vec::new(),
        stage_hints: stage_hints.to_vec(),
        notes: vec![
            "Packages the source quant layout unchanged to establish Skippy correctness and latency baselines."
                .to_string(),
        ],
    })
}

fn boundary_protected_candidate(
    default_quant: &str,
    groups: Vec<QuantGroup>,
    stage_hints: &[StageQuantHint],
) -> QuantLayoutCandidate {
    with_layout_hash(QuantLayoutCandidate {
        id: "boundary-protected".to_string(),
        layout_hash: String::new(),
        name: "Higher precision embeddings, output, first band, and last band".to_string(),
        status: "experimental".to_string(),
        strategy: "stage-aware-boundary-protected".to_string(),
        default_quant: default_quant.to_string(),
        groups,
        stage_hints: stage_hints.to_vec(),
        notes: vec![
            "Spends precision where coding-agent behavior is commonly sensitive before lowering middle layers."
                .to_string(),
        ],
    })
}

fn middle_compressed_candidate(
    default_quant: &str,
    source_quant: &str,
    protected_width: u32,
    layer_count: u32,
    stage_hints: &[StageQuantHint],
    moe_groups: Vec<QuantGroup>,
) -> QuantLayoutCandidate {
    let mut groups = combine_groups(
        boundary_protection_groups(default_quant, source_quant, protected_width, layer_count),
        moe_groups,
    );
    if let Some(selector) = middle_selector(protected_width, layer_count) {
        groups.push(QuantGroup {
            name: "middle-latency-band".to_string(),
            quant: compression_quant_for_source(source_quant, "Q3_K_M").to_string(),
            selector,
            reason: "Lower precision for the broad middle band to reduce steady-state Skippy decode cost.".to_string(),
        });
    }
    with_layout_hash(QuantLayoutCandidate {
        id: "middle-compressed".to_string(),
        layout_hash: String::new(),
        name: "Compressed middle band with protected boundaries".to_string(),
        status: "experimental".to_string(),
        strategy: "stage-aware-middle-compressed".to_string(),
        default_quant: default_quant.to_string(),
        groups,
        stage_hints: stage_hints.to_vec(),
        notes: vec![
            "First aggressive latency candidate; certification decides whether the middle-band precision drop holds."
                .to_string(),
        ],
    })
}

fn stage_balanced_candidate(
    default_quant: &str,
    source_quant: &str,
    mut groups: Vec<QuantGroup>,
    protected_width: u32,
    layer_count: u32,
    stage_hints: &[StageQuantHint],
) -> QuantLayoutCandidate {
    if let Some(stage) = largest_unprotected_stage(stage_hints, protected_width, layer_count) {
        groups.push(QuantGroup {
            name: format!("stage-{}-balance-band", stage.stage_index),
            quant: compression_quant_for_source(source_quant, "Q3_K_M").to_string(),
            selector: QuantSelector::LayerRange {
                start: stage.layer_start,
                end: stage.layer_end,
            },
            reason: "Uses tensor bytes as an initial proxy for the slowest stage until measured latency is available."
                .to_string(),
        });
    }
    with_layout_hash(QuantLayoutCandidate {
        id: "stage-balanced-proxy".to_string(),
        layout_hash: String::new(),
        name: "Byte-proxy stage balance candidate".to_string(),
        status: "experimental".to_string(),
        strategy: "stage-balanced-byte-proxy".to_string(),
        default_quant: default_quant.to_string(),
        groups,
        stage_hints: stage_hints.to_vec(),
        notes: vec![
            "Replace byte-proxy lowering with profiler-guided lowering once per-stage decode profiles are attached."
                .to_string(),
        ],
    })
}

fn stage_balanced_ffn_candidate(
    default_quant: &str,
    source_quant: &str,
    mut groups: Vec<QuantGroup>,
    protected_width: u32,
    layer_count: u32,
    stage_hints: &[StageQuantHint],
    has_moe: bool,
) -> QuantLayoutCandidate {
    if let Some(stage) = largest_unprotected_stage(stage_hints, protected_width, layer_count) {
        groups.push(QuantGroup {
            name: format!("stage-{}-ffn-balance-band", stage.stage_index),
            quant: compression_quant_for_source(source_quant, "Q3_K_M").to_string(),
            selector: QuantSelector::TensorNamePattern {
                patterns: stage_ffn_patterns(stage.layer_start, stage.layer_end, has_moe),
            },
            reason: "Compresses FFN tensors in the largest byte stage while keeping attention at the source quant tier."
                .to_string(),
        });
    }
    with_layout_hash(QuantLayoutCandidate {
        id: "stage-balanced-ffn-proxy".to_string(),
        layout_hash: String::new(),
        name: "Byte-proxy stage balance with attention protected".to_string(),
        status: "experimental".to_string(),
        strategy: "stage-balanced-ffn-byte-proxy".to_string(),
        default_quant: default_quant.to_string(),
        groups,
        stage_hints: stage_hints.to_vec(),
        notes: vec![
            "Narrows the stage-balance experiment to FFN tensors after broad largest-stage compression regressed decode latency."
                .to_string(),
        ],
    })
}

struct StageBalancedFfnPartCandidateInput<'a> {
    default_quant: &'a str,
    source_quant: &'a str,
    groups: Vec<QuantGroup>,
    protected_width: u32,
    layer_count: u32,
    stage_hints: &'a [StageQuantHint],
    has_moe: bool,
}

fn stage_balanced_ffn_part_candidate(
    mut input: StageBalancedFfnPartCandidateInput<'_>,
    part: &'static str,
) -> QuantLayoutCandidate {
    if let Some(stage) =
        largest_unprotected_stage(input.stage_hints, input.protected_width, input.layer_count)
    {
        input.groups.push(QuantGroup {
            name: format!("stage-{}-ffn-{part}-balance-band", stage.stage_index),
            quant: compression_quant_for_source(input.source_quant, "Q3_K_M").to_string(),
            selector: QuantSelector::TensorNamePattern {
                patterns: stage_ffn_part_patterns(
                    stage.layer_start,
                    stage.layer_end,
                    part,
                    input.has_moe,
                ),
            },
            reason: format!(
                "Compresses only FFN {part} tensors in the largest byte stage for per-tensor sensitivity evidence."
            ),
        });
    }
    with_layout_hash(QuantLayoutCandidate {
        id: format!("stage-balanced-ffn-{part}-proxy"),
        layout_hash: String::new(),
        name: format!("Byte-proxy stage balance for FFN {part} tensors"),
        status: "experimental".to_string(),
        strategy: format!("stage-balanced-ffn-{part}-byte-proxy"),
        default_quant: input.default_quant.to_string(),
        groups: input.groups,
        stage_hints: input.stage_hints.to_vec(),
        notes: vec![format!(
            "Narrows stage-balance compression to FFN {part} tensors after all-FFN lowering still trailed the baseline."
        )],
    })
}

fn ffn_compressed_attention_protected_candidate(
    default_quant: &str,
    source_quant: &str,
    protected_width: u32,
    layer_count: u32,
    stage_hints: &[StageQuantHint],
    has_moe: bool,
) -> QuantLayoutCandidate {
    let mut groups = combine_groups(
        boundary_protection_groups(default_quant, source_quant, protected_width, layer_count),
        attention_protection_groups(default_quant, source_quant, protected_width, layer_count),
    );
    groups = combine_groups(
        groups,
        ffn_compression_groups(source_quant, protected_width, layer_count, has_moe),
    );
    with_layout_hash(QuantLayoutCandidate {
        id: "ffn-compressed-attention-protected".to_string(),
        layout_hash: String::new(),
        name: "Compressed FFN middle band with attention protected".to_string(),
        status: "experimental".to_string(),
        strategy: "stage-aware-ffn-compressed-attention-protected".to_string(),
        default_quant: default_quant.to_string(),
        groups,
        stage_hints: stage_hints.to_vec(),
        notes: vec![
            "Targets FFN tensors for Skippy latency and memory while keeping middle-band attention at a safer tier for long-context recall."
                .to_string(),
        ],
    })
}

fn with_layout_hash(mut candidate: QuantLayoutCandidate) -> QuantLayoutCandidate {
    candidate.layout_hash = quant_layout_hash(&candidate);
    candidate
}

fn quant_layout_hash(candidate: &QuantLayoutCandidate) -> String {
    let mut hasher = Sha256::new();
    hash_str(&mut hasher, "skippy-quant-layout-v3");
    hash_str(&mut hasher, &candidate.id);
    hash_str(&mut hasher, &candidate.strategy);
    hash_str(&mut hasher, &candidate.default_quant);
    for group in &candidate.groups {
        hash_str(&mut hasher, &group.name);
        hash_str(&mut hasher, &group.quant);
        hash_selector(&mut hasher, &group.selector);
    }
    format!("{:x}", hasher.finalize())
}

fn hash_selector(hasher: &mut Sha256, selector: &QuantSelector) {
    match selector {
        QuantSelector::Role { roles } => {
            hash_str(hasher, "role");
            for role in roles {
                hash_str(hasher, role);
            }
        }
        QuantSelector::LayerRange { start, end } => {
            hash_str(hasher, "layer_range");
            hasher.update(start.to_le_bytes());
            hasher.update(end.to_le_bytes());
        }
        QuantSelector::TensorNamePattern { patterns } => {
            hash_str(hasher, "tensor_name_pattern");
            for pattern in patterns {
                hash_str(hasher, pattern);
            }
        }
    }
}

fn hash_str(hasher: &mut Sha256, value: &str) {
    hasher.update(value.len().to_le_bytes());
    hasher.update(value.as_bytes());
}

fn unprotected_stage_selector(
    stage: &StageQuantHint,
    protected_width: u32,
    layer_count: u32,
) -> Option<StageQuantHint> {
    let protected_tail_start = layer_count.saturating_sub(protected_width);
    let layer_start = stage.layer_start.max(protected_width);
    let layer_end = stage.layer_end.min(protected_tail_start);
    (layer_start < layer_end).then_some(StageQuantHint {
        stage_index: stage.stage_index,
        layer_start,
        layer_end,
        tensor_bytes: stage.tensor_bytes,
        role: stage.role.clone(),
    })
}

fn largest_unprotected_stage(
    stage_hints: &[StageQuantHint],
    protected_width: u32,
    layer_count: u32,
) -> Option<StageQuantHint> {
    stage_hints
        .iter()
        .max_by_key(|stage| stage.tensor_bytes)
        .and_then(|stage| unprotected_stage_selector(stage, protected_width, layer_count))
}

fn boundary_protection_groups(
    default_quant: &str,
    source_quant: &str,
    protected_width: u32,
    layer_count: u32,
) -> Vec<QuantGroup> {
    if default_quant == "COPY" {
        return Vec::new();
    }
    let last_start = layer_count.saturating_sub(protected_width);
    vec![
        QuantGroup {
            name: "embedding-and-output".to_string(),
            quant: cap_quant_at_source(source_quant, "Q6_K").to_string(),
            selector: QuantSelector::Role {
                roles: vec![
                    "embedding".to_string(),
                    "final_norm".to_string(),
                    "output".to_string(),
                ],
            },
            reason: "Protect token identity and final logits for tool calls and exact code edits."
                .to_string(),
        },
        QuantGroup {
            name: "early-layers".to_string(),
            quant: cap_quant_at_source(source_quant, "Q5_K_M").to_string(),
            selector: QuantSelector::LayerRange {
                start: 0,
                end: protected_width,
            },
            reason:
                "Protect the first transformer band while staged quality evidence is still sparse."
                    .to_string(),
        },
        QuantGroup {
            name: "late-layers".to_string(),
            quant: cap_quant_at_source(source_quant, "Q5_K_M").to_string(),
            selector: QuantSelector::LayerRange {
                start: last_start,
                end: layer_count,
            },
            reason: "Protect the final transformer band that feeds logits and structured outputs."
                .to_string(),
        },
    ]
}

fn attention_protection_groups(
    default_quant: &str,
    source_quant: &str,
    protected_width: u32,
    layer_count: u32,
) -> Vec<QuantGroup> {
    if default_quant == "COPY" || protected_width >= layer_count.saturating_sub(protected_width) {
        return Vec::new();
    }
    vec![QuantGroup {
        name: "middle-attention-protected".to_string(),
        quant: cap_quant_at_source(source_quant, "Q5_K_M").to_string(),
        selector: QuantSelector::TensorNamePattern {
            patterns: vec![
                middle_tensor_pattern(protected_width, layer_count, "attn_.*"),
                middle_tensor_pattern(protected_width, layer_count, "attn.*"),
            ],
        },
        reason:
            "Protect middle-band attention tensors that carry long-context retrieval and tool-call grounding."
                .to_string(),
    }]
}

fn ffn_compression_groups(
    source_quant: &str,
    protected_width: u32,
    layer_count: u32,
    has_moe: bool,
) -> Vec<QuantGroup> {
    if protected_width >= layer_count.saturating_sub(protected_width) {
        return Vec::new();
    }
    let patterns = if has_moe {
        vec![
            middle_tensor_pattern(protected_width, layer_count, "ffn_(gate|up|down)_exps"),
            middle_tensor_pattern(protected_width, layer_count, "ffn_gate_up_exps"),
        ]
    } else {
        vec![
            middle_tensor_pattern(protected_width, layer_count, "ffn_.*"),
            middle_tensor_pattern(protected_width, layer_count, "ffn.*"),
        ]
    };
    vec![QuantGroup {
        name: "middle-ffn-compressed".to_string(),
        quant: compression_quant_for_source(source_quant, "Q3_K_M").to_string(),
        selector: QuantSelector::TensorNamePattern { patterns },
        reason:
            "Lower middle-band FFN tensors as a first tensor-aware Skippy latency and memory candidate."
                .to_string(),
    }]
}

fn stage_ffn_patterns(start: u32, end: u32, has_moe: bool) -> Vec<String> {
    if has_moe {
        vec![
            stage_tensor_pattern(start, end, "ffn_(gate|up|down)_exps"),
            stage_tensor_pattern(start, end, "ffn_gate_up_exps"),
        ]
    } else {
        vec![
            stage_tensor_pattern(start, end, "ffn_.*"),
            stage_tensor_pattern(start, end, "ffn.*"),
        ]
    }
}

fn stage_ffn_part_patterns(start: u32, end: u32, part: &str, has_moe: bool) -> Vec<String> {
    if has_moe {
        return stage_moe_ffn_part_patterns(start, end, part);
    }
    vec![stage_tensor_pattern(start, end, &format!("ffn_{part}"))]
}

fn stage_moe_ffn_part_patterns(start: u32, end: u32, part: &str) -> Vec<String> {
    match part {
        "down" => vec![stage_tensor_pattern(start, end, "ffn_down_exps")],
        "gate" => vec![
            stage_tensor_pattern(start, end, "ffn_gate_exps"),
            stage_tensor_pattern(start, end, "ffn_gate_up_exps"),
        ],
        "up" => vec![
            stage_tensor_pattern(start, end, "ffn_up_exps"),
            stage_tensor_pattern(start, end, "ffn_gate_up_exps"),
        ],
        _ => Vec::new(),
    }
}

fn stage_tensor_pattern(start: u32, end: u32, suffix: &str) -> String {
    format!(
        "blk\\.({})\\.{suffix}\\.weight",
        layer_regex_range(start, end)
    )
}

fn middle_tensor_pattern(protected_width: u32, layer_count: u32, suffix: &str) -> String {
    let end = layer_count.saturating_sub(protected_width);
    format!(
        "blk\\.({})\\.{suffix}\\.weight",
        layer_regex_range(protected_width, end)
    )
}

fn layer_regex_range(start: u32, end: u32) -> String {
    (start..end)
        .map(|layer| layer.to_string())
        .collect::<Vec<_>>()
        .join("|")
}

fn moe_sensitive_groups(tensors: &[TensorInfo], source_quant: &str) -> Vec<QuantGroup> {
    if !has_moe_tensor(tensors) {
        return Vec::new();
    }
    let expert_quant = moe_expert_floor_quant(source_quant);
    vec![
        QuantGroup {
            name: "moe-routers".to_string(),
            quant: cap_quant_at_source(source_quant, "Q6_K").to_string(),
            selector: QuantSelector::TensorNamePattern {
                patterns: vec![r"blk\.[0-9]+\.ffn_gate_inp\.weight".to_string()],
            },
            reason:
                "Protect MoE router logits so coding-agent tool and patch routing stays stable."
                    .to_string(),
        },
        QuantGroup {
            name: "moe-experts".to_string(),
            quant: expert_quant.to_string(),
            selector: QuantSelector::TensorNamePattern {
                patterns: vec![
                    r"blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight".to_string(),
                    r"blk\.[0-9]+\.ffn_gate_up_exps\.weight".to_string(),
                ],
            },
            reason:
                "Keep MoE expert tensors at a source-aware floor when broad layer bands are compressed."
                    .to_string(),
        },
    ]
}

fn moe_expert_floor_quant(source_quant: &str) -> &'static str {
    match source_quant {
        "Q8_0" => "Q8_0",
        "Q6_K" => "Q6_K",
        "Q5_K" | "Q5_K_M" | "Q5_K_S" => "Q5_K_M",
        "Q4_K" | "Q4_K_M" | "Q4_K_S" | "Q4_0" | "Q4_1" => "Q4_K_M",
        _ => "Q4_K_M",
    }
}

fn cap_quant_at_source(source_quant: &str, desired_quant: &'static str) -> &'static str {
    match (quant_rank(source_quant), quant_rank(desired_quant)) {
        (Some(source), Some(desired)) if source < desired => canonical_quant_for_rank(source),
        _ => desired_quant,
    }
}

fn compression_quant_for_source(source_quant: &str, desired_quant: &'static str) -> &'static str {
    cap_quant_at_source(source_quant, desired_quant)
}

fn quant_rank(quant: &str) -> Option<u8> {
    match quant {
        "F16" | "BF16" => Some(90),
        "Q8_0" => Some(80),
        "Q6_K" => Some(60),
        "Q5_K" | "Q5_K_M" | "Q5_K_S" => Some(50),
        "Q4_K" | "Q4_K_M" | "Q4_K_S" | "Q4_0" | "Q4_1" | "IQ4_XS" => Some(40),
        "Q3_K" | "Q3_K_M" | "Q3_K_S" | "IQ3_M" => Some(30),
        "Q2_K" | "IQ2_M" => Some(20),
        "IQ1_M" => Some(10),
        _ => None,
    }
}

fn canonical_quant_for_rank(rank: u8) -> &'static str {
    match rank {
        90 => "F16",
        80 => "Q8_0",
        60 => "Q6_K",
        50 => "Q5_K_M",
        40 => "Q4_K_M",
        30 => "Q3_K_M",
        20 => "Q2_K",
        10 => "IQ1_M",
        _ => "Q4_K_M",
    }
}

fn has_moe_tensor(tensors: &[TensorInfo]) -> bool {
    tensors.iter().any(|tensor| {
        let name = tensor.name.as_str();
        name.contains("ffn_gate_inp")
            || name.contains("_exps")
            || name.contains(".experts.")
            || name.contains(".router.")
    })
}

fn middle_selector(protected_width: u32, layer_count: u32) -> Option<QuantSelector> {
    let start = protected_width;
    let end = layer_count.saturating_sub(protected_width);
    (start < end).then_some(QuantSelector::LayerRange { start, end })
}

fn stage_hints(
    tensors: &[TensorInfo],
    layer_count: u32,
    stage_count: usize,
) -> Vec<StageQuantHint> {
    partition_layers(layer_count, stage_count)
        .into_iter()
        .enumerate()
        .map(|(stage_index, (layer_start, layer_end))| StageQuantHint {
            stage_index,
            layer_start,
            layer_end,
            tensor_bytes: tensors
                .iter()
                .filter(|tensor| {
                    tensor_in_stage(tensor, stage_index, stage_count, layer_start, layer_end)
                })
                .map(|tensor| tensor.byte_size)
                .sum(),
            role: if stage_count == 1 {
                "single".to_string()
            } else if stage_index == 0 {
                "first".to_string()
            } else if stage_index + 1 == stage_count {
                "last".to_string()
            } else {
                "middle".to_string()
            },
        })
        .collect()
}

fn tensor_in_stage(
    tensor: &TensorInfo,
    stage_index: usize,
    stage_count: usize,
    layer_start: u32,
    layer_end: u32,
) -> bool {
    matches!(
        tensor.layer_index,
        Some(layer) if layer >= layer_start && layer < layer_end
    ) || (stage_index == 0 && tensor.role == TensorRole::Embedding)
        || (stage_index + 1 == stage_count
            && matches!(tensor.role, TensorRole::FinalNorm | TensorRole::Output))
}

fn partition_layers(layer_count: u32, stage_count: usize) -> Vec<(u32, u32)> {
    let base = layer_count / stage_count as u32;
    let extra = layer_count % stage_count as u32;
    let mut start = 0;
    (0..stage_count)
        .map(|stage_index| {
            let width = base + u32::from((stage_index as u32) < extra);
            let end = start + width;
            let range = (start, end);
            start = end;
            range
        })
        .collect()
}

fn protected_band_width(layer_count: u32) -> u32 {
    layer_count.div_ceil(8).clamp(1, 8)
}

fn layer_count(tensors: &[TensorInfo]) -> Result<u32> {
    tensors
        .iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .map(|layer| layer + 1)
        .context("source model has no layer tensors")
}

fn infer_quant_from_path(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_uppercase();
    if name.contains("UD-Q4") {
        return "Q4_K_M".to_string();
    }
    if name.contains("UD-Q3") {
        return "Q3_K_M".to_string();
    }
    if name.contains("UD-Q2") {
        return "Q2_K".to_string();
    }
    if name.contains("UD-IQ4") {
        return "IQ4_XS".to_string();
    }
    if name.contains("UD-IQ3") {
        return "IQ3_M".to_string();
    }
    if name.contains("UD-IQ2") {
        return "IQ2_M".to_string();
    }
    for quant in known_quant_names() {
        if name.contains(quant) {
            return quant.to_string();
        }
    }
    "source".to_string()
}

fn default_quant_for_source(source_quant: &str) -> &'static str {
    match source_quant {
        "F32" | "F16" | "BF16" => "Q4_K_M",
        quant if quant_rank(quant).is_some() => "COPY",
        _ => "COPY",
    }
}

fn known_quant_names() -> &'static [&'static str] {
    &[
        "F32", "BF16", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_1",
        "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K", "IQ4_XS", "IQ3_M", "IQ2_M", "IQ1_M",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_plan_generates_coding_agent_candidate_matrix() {
        let report = build_quant_plan(test_input(8, 2)).expect("build quant plan");

        assert_eq!(report.kind, "skippy_quant_plan");
        assert_eq!(report.toolchain.layout_hash_version, 4);
        assert_eq!(report.source.layer_count, 8);
        assert_eq!(report.stage_count, 2);
        assert_eq!(report.protected_band_width, 1);
        assert_eq!(report.candidates.len(), 9);
        assert_eq!(report.candidates[0].id, "baseline-source-quant");
        assert_eq!(report.candidates[1].id, "boundary-protected");
        assert_eq!(report.candidates[2].id, "middle-compressed");
        assert_eq!(
            report.candidates[3].id,
            "ffn-compressed-attention-protected"
        );
        assert_eq!(report.candidates[4].id, "stage-balanced-ffn-proxy");
        assert_eq!(report.candidates[5].id, "stage-balanced-ffn-down-proxy");
        assert_eq!(report.candidates[6].id, "stage-balanced-ffn-gate-proxy");
        assert_eq!(report.candidates[7].id, "stage-balanced-ffn-up-proxy");
        assert_eq!(report.candidates[8].id, "stage-balanced-proxy");
        assert!(
            report
                .candidates
                .iter()
                .all(|candidate| candidate.layout_hash.len() == 64)
        );
    }

    #[test]
    fn quant_layout_hashes_are_stable() {
        let first = build_quant_plan(test_input(16, 4)).expect("first quant plan");
        let second = build_quant_plan(test_input(16, 4)).expect("second quant plan");
        let first_hashes = first
            .candidates
            .iter()
            .map(|candidate| candidate.layout_hash.as_str())
            .collect::<Vec<_>>();
        let second_hashes = second
            .candidates
            .iter()
            .map(|candidate| candidate.layout_hash.as_str())
            .collect::<Vec<_>>();

        assert_eq!(first_hashes, second_hashes);
    }

    #[test]
    fn quantized_source_boundary_candidate_copies_without_upquantizing_source() {
        let report = build_quant_plan(test_input(16, 4)).expect("build quant plan");
        let boundary = report
            .candidates
            .iter()
            .find(|candidate| candidate.id == "boundary-protected")
            .expect("boundary candidate");

        assert_eq!(report.protected_band_width, 2);
        assert_eq!(boundary.default_quant, "COPY");
        assert!(boundary.groups.is_empty());
    }

    #[test]
    fn protection_groups_can_use_higher_tiers_when_source_quant_allows() {
        let input = test_input_with_source_quant(16, 4, "F16");
        let report = build_quant_plan(input).expect("build quant plan");
        let boundary = report
            .candidates
            .iter()
            .find(|candidate| candidate.id == "boundary-protected")
            .expect("boundary candidate");

        assert!(
            boundary
                .groups
                .iter()
                .any(|group| { group.name == "embedding-and-output" && group.quant == "Q6_K" })
        );
        assert!(
            boundary
                .groups
                .iter()
                .any(|group| { group.name == "early-layers" && group.quant == "Q5_K_M" })
        );
    }

    #[test]
    fn stage_balanced_candidate_lowers_largest_byte_stage() {
        let report = build_quant_plan(test_input_with_heavy_stage()).expect("build quant plan");
        let balanced = report
            .candidates
            .iter()
            .find(|candidate| candidate.id == "stage-balanced-proxy")
            .expect("stage-balanced candidate");

        assert!(balanced.groups.iter().any(|group| {
            group.name == "stage-1-balance-band"
                && matches!(
                    group.selector,
                    QuantSelector::LayerRange { start: 2, end: 3 }
                )
        }));
    }

    #[test]
    fn stage_balanced_ffn_candidate_lowers_largest_byte_stage_ffn_only() {
        let report = build_quant_plan(test_input_with_heavy_stage()).expect("build quant plan");
        let balanced = report
            .candidates
            .iter()
            .find(|candidate| candidate.id == "stage-balanced-ffn-proxy")
            .expect("stage-balanced ffn candidate");
        let group = balanced
            .groups
            .iter()
            .find(|group| group.name == "stage-1-ffn-balance-band")
            .expect("stage ffn group");

        let QuantSelector::TensorNamePattern { patterns } = &group.selector else {
            panic!("stage ffn group should use tensor patterns");
        };
        assert!(patterns.iter().any(|pattern| {
            pattern == r"blk\.(2)\.ffn_.*\.weight" || pattern == r"blk\.(2)\.ffn.*\.weight"
        }));
        assert!(!patterns.iter().any(|pattern| pattern.contains("attn")));
    }

    #[test]
    fn stage_balanced_ffn_part_candidates_target_single_ffn_parts() {
        let report = build_quant_plan(test_input_with_heavy_stage()).expect("build quant plan");

        assert_ffn_part_candidate(&report, "down", r"blk\.(2)\.ffn_down\.weight");
        assert_ffn_part_candidate(&report, "gate", r"blk\.(2)\.ffn_gate\.weight");
        assert_ffn_part_candidate(&report, "up", r"blk\.(2)\.ffn_up\.weight");
    }

    #[test]
    fn ffn_compressed_candidate_protects_attention_before_lowering_ffn() {
        let report =
            build_quant_plan(test_input_with_source_quant(16, 4, "F16")).expect("build quant plan");
        let candidate = report
            .candidates
            .iter()
            .find(|candidate| candidate.id == "ffn-compressed-attention-protected")
            .expect("ffn compressed candidate");

        let attention_index = candidate
            .groups
            .iter()
            .position(|group| {
                group.name == "middle-attention-protected"
                    && group.quant == "Q5_K_M"
                    && matches!(group.selector, QuantSelector::TensorNamePattern { .. })
            })
            .expect("attention protection group");
        let ffn_index = candidate
            .groups
            .iter()
            .position(|group| {
                group.name == "middle-ffn-compressed"
                    && group.quant == "Q3_K_M"
                    && matches!(group.selector, QuantSelector::TensorNamePattern { .. })
            })
            .expect("ffn compression group");

        assert!(attention_index < ffn_index);
    }

    #[test]
    fn moe_tensors_add_router_and_expert_protection_groups() {
        let report = build_quant_plan(test_input_with_moe_tensors()).expect("build quant plan");
        let middle = report
            .candidates
            .iter()
            .find(|candidate| candidate.id == "middle-compressed")
            .expect("middle candidate");

        assert!(middle.groups.iter().any(|group| {
            group.name == "moe-routers"
                && group.quant == "Q4_K_M"
                && matches!(group.selector, QuantSelector::TensorNamePattern { .. })
        }));
        assert!(middle.groups.iter().any(|group| {
            group.name == "moe-experts"
                && group.quant == "Q4_K_M"
                && matches!(group.selector, QuantSelector::TensorNamePattern { .. })
        }));
        let expert_index = middle
            .groups
            .iter()
            .position(|group| group.name == "moe-experts")
            .expect("moe expert group");
        let middle_index = middle
            .groups
            .iter()
            .position(|group| group.name == "middle-latency-band")
            .expect("middle latency group");
        assert!(
            expert_index < middle_index,
            "MoE expert patterns must precede broad middle-layer lowering"
        );
    }

    #[test]
    fn ffn_compressed_candidate_targets_moe_experts_without_router_pattern() {
        let report = build_quant_plan(test_input_with_moe_tensors()).expect("build quant plan");
        let candidate = report
            .candidates
            .iter()
            .find(|candidate| candidate.id == "ffn-compressed-attention-protected")
            .expect("ffn compressed candidate");

        assert!(
            !candidate
                .groups
                .iter()
                .any(|group| group.name == "moe-experts")
        );
        let ffn_group = candidate
            .groups
            .iter()
            .find(|group| group.name == "middle-ffn-compressed")
            .expect("ffn compression group");

        let QuantSelector::TensorNamePattern { patterns } = &ffn_group.selector else {
            panic!("ffn compression group should use tensor patterns");
        };
        assert!(
            patterns
                .iter()
                .any(|pattern| pattern.contains("ffn_(gate|up|down)_exps"))
        );
        assert!(
            !patterns
                .iter()
                .any(|pattern| pattern.contains("ffn_gate_inp"))
        );
    }

    #[test]
    fn moe_expert_floor_preserves_higher_source_quant_tiers() {
        let mut q6_input = test_input_with_moe_tensors();
        q6_input.source_quant = "Q6_K".to_string();
        let q6_report = build_quant_plan(q6_input).expect("build q6 quant plan");
        assert_eq!(moe_expert_quant(&q6_report), Some("Q6_K"));

        let mut q5_input = test_input_with_moe_tensors();
        q5_input.source_quant = "Q5_K_S".to_string();
        let q5_report = build_quant_plan(q5_input).expect("build q5 quant plan");
        assert_eq!(moe_expert_quant(&q5_report), Some("Q5_K_M"));
    }

    #[test]
    fn compression_targets_do_not_raise_lower_precision_sources() {
        let mut input = test_input(8, 2);
        input.source_quant = "Q2_K".to_string();
        let report = build_quant_plan(input).expect("build q2 quant plan");
        let middle = report
            .candidates
            .iter()
            .find(|candidate| candidate.id == "middle-compressed")
            .expect("middle candidate");

        assert!(
            middle
                .groups
                .iter()
                .any(|group| { group.name == "middle-latency-band" && group.quant == "Q2_K" })
        );
    }

    #[test]
    fn dense_models_do_not_get_moe_specific_groups() {
        let report = build_quant_plan(test_input(8, 2)).expect("build quant plan");

        assert!(report.candidates.iter().all(|candidate| {
            candidate
                .groups
                .iter()
                .all(|group| !group.name.starts_with("moe-"))
        }));
    }

    #[test]
    fn quant_name_is_inferred_from_source_path() {
        let quant = infer_quant_from_path(Path::new("Qwen3-Coder-30B-A3B-Q4_K_M.gguf"));

        assert_eq!(quant, "Q4_K_M");
        assert_eq!(
            infer_quant_from_path(Path::new(
                "UD-Q4_K_XL/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-00001-of-00006.gguf"
            )),
            "Q4_K_M"
        );
        assert_eq!(default_quant_for_source("Q4_K_M"), "COPY");
        assert_eq!(default_quant_for_source("F16"), "Q4_K_M");
    }

    #[test]
    fn materialized_skippy_stage_and_tokenizer_names_are_rejected_as_sources() {
        assert!(is_materialized_skippy_slice_name(
            "unsloth_Qwen3-Coder-480B-A35B-Instruct-GGUF_UD-Q4_K_XL-stage-1-3-4-65998a63231926f31995a5d7.gguf"
        ));
        assert!(is_materialized_skippy_slice_name(
            "unsloth_Qwen3-Coder-480B-A35B-Instruct-GGUF_UD-Q4_K_XL-tokenizer-0-1-08d5e039a6ddc0eafa57fc42.gguf"
        ));
    }

    #[test]
    fn ordinary_source_and_split_gguf_names_are_allowed_as_sources() {
        assert!(!is_materialized_skippy_slice_name(
            "Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-00001-of-00009.gguf"
        ));
        assert!(!is_materialized_skippy_slice_name(
            "Qwen3-Coder-30B-A3B-Q4_K_M.gguf"
        ));
    }

    fn moe_expert_quant(report: &QuantPlanReport) -> Option<&str> {
        report
            .candidates
            .iter()
            .find(|candidate| candidate.id == "middle-compressed")?
            .groups
            .iter()
            .find(|group| group.name == "moe-experts")
            .map(|group| group.quant.as_str())
    }

    fn assert_ffn_part_candidate(report: &QuantPlanReport, part: &str, expected_pattern: &str) {
        let candidate_id = format!("stage-balanced-ffn-{part}-proxy");
        let group_name = format!("stage-1-ffn-{part}-balance-band");
        let candidate = report
            .candidates
            .iter()
            .find(|candidate| candidate.id == candidate_id)
            .expect("stage-balanced ffn part candidate");
        let group = candidate
            .groups
            .iter()
            .find(|group| group.name == group_name)
            .expect("stage ffn part group");

        let QuantSelector::TensorNamePattern { patterns } = &group.selector else {
            panic!("stage ffn part group should use tensor patterns");
        };
        assert_eq!(patterns, &[expected_pattern.to_string()]);
    }

    fn test_input(layer_count: u32, stage_count: usize) -> QuantPlanInput {
        test_input_with_source_quant(layer_count, stage_count, "Q4_K_M")
    }

    fn test_input_with_source_quant(
        layer_count: u32,
        stage_count: usize,
        source_quant: &str,
    ) -> QuantPlanInput {
        QuantPlanInput {
            source_path: PathBuf::from(format!("model-{source_quant}.gguf")),
            source_sha256: "sha".to_string(),
            source_sha256_kind: "file".to_string(),
            source_quant: source_quant.to_string(),
            default_quant: default_quant_for_source(source_quant).to_string(),
            source_shards: Vec::new(),
            profile: QuantPlanProfile::CodingAgent,
            stage_count,
            tensors: test_tensors(layer_count, |_| 10),
        }
    }

    fn test_input_with_heavy_stage() -> QuantPlanInput {
        QuantPlanInput {
            source_path: PathBuf::from("model-Q4_K_M.gguf"),
            source_sha256: "sha".to_string(),
            source_sha256_kind: "file".to_string(),
            source_quant: "Q4_K_M".to_string(),
            default_quant: default_quant_for_source("Q4_K_M").to_string(),
            source_shards: Vec::new(),
            profile: QuantPlanProfile::CodingAgent,
            stage_count: 2,
            tensors: test_tensors(4, |layer| if layer >= 2 { 100 } else { 10 }),
        }
    }

    fn test_input_with_moe_tensors() -> QuantPlanInput {
        let mut input = test_input(8, 2);
        input.tensors.push(tensor(
            "blk.3.ffn_gate_inp.weight",
            Some(3),
            TensorRole::Layer,
            10,
        ));
        input.tensors.push(tensor(
            "blk.3.ffn_down_exps.weight",
            Some(3),
            TensorRole::Layer,
            100,
        ));
        input
    }

    fn test_tensors(layer_count: u32, layer_bytes: impl Fn(u32) -> u64) -> Vec<TensorInfo> {
        let mut tensors = vec![
            tensor("token_embd.weight", None, TensorRole::Embedding, 50),
            tensor("output_norm.weight", None, TensorRole::FinalNorm, 20),
            tensor("output.weight", None, TensorRole::Output, 50),
        ];
        for layer in 0..layer_count {
            tensors.push(tensor(
                &format!("blk.{layer}.attn_q.weight"),
                Some(layer),
                TensorRole::Layer,
                layer_bytes(layer),
            ));
        }
        tensors
    }

    fn tensor(
        name: &str,
        layer_index: Option<u32>,
        role: TensorRole,
        byte_size: u64,
    ) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            layer_index,
            role,
            ggml_type: 0,
            byte_size,
            element_count: byte_size,
        }
    }
}
