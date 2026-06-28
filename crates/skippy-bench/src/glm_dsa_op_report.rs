use std::{collections::BTreeMap, fs, path::PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::cli::{GlmDsaOpCompareArgs, GlmDsaOpReportArgs};

const OP_TIMING_PREFIX: &str = "skippy: glm_dsa_op_timing ";
const GROUP_TIMING_PREFIX: &str = "skippy: glm_dsa_group_timing ";
const SIDEBAND_PREFIX: &str = "skippy: glm_dsa_top_k_sideband_forward ";
const DIRECT_SPARSE_DECISION_PREFIX: &str = "skippy: glm_dsa_direct_sparse_decision ";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct OpBucket {
    nodes: u64,
    elapsed_us: u64,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct PhaseSummary {
    records: usize,
    tokens: u64,
    total_us: u64,
    avg_total_us_per_record: Option<f64>,
    avg_total_us_per_token: Option<f64>,
    indexer_topk: OpBucket,
    #[serde(skip_serializing_if = "Option::is_none")]
    indexer: Option<OpBucket>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<OpBucket>,
    sparse_mask: OpBucket,
    #[serde(skip_serializing_if = "Option::is_none")]
    sparse_mask_fill: Option<OpBucket>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sparse_mask_topk: Option<OpBucket>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sparse_mask_add: Option<OpBucket>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dsa_sparse_attn: Option<OpBucket>,
    mla_attention: OpBucket,
    routed_moe: OpBucket,
    shared_expert: OpBucket,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct LogSummary {
    path: PathBuf,
    records: usize,
    stage_records: BTreeMap<i32, BTreeMap<Phase, PhaseSummary>>,
    group_records: BTreeMap<i32, BTreeMap<String, BTreeMap<Phase, PhaseSummary>>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    hottest_group_records: Vec<HotGroupSummary>,
    sideband_records: BTreeMap<String, BTreeMap<Phase, SidebandSummary>>,
}

#[derive(Debug, Deserialize, Serialize)]
struct GlmDsaOpReport {
    logs: Vec<LogSummary>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct HotGroupSummary {
    stage: i32,
    phase: Phase,
    group: String,
    summary: PhaseSummary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct TimingRecord {
    pub(crate) stage: i32,
    pub(crate) tokens: u64,
    pub(crate) total_us: u64,
    pub(crate) indexer_topk_nodes: u64,
    pub(crate) indexer_topk_us: u64,
    pub(crate) indexer_nodes: Option<u64>,
    pub(crate) indexer_us: Option<u64>,
    pub(crate) top_k_nodes: Option<u64>,
    pub(crate) top_k_us: Option<u64>,
    pub(crate) sparse_mask_nodes: u64,
    pub(crate) sparse_mask_us: u64,
    pub(crate) sparse_mask_fill_nodes: Option<u64>,
    pub(crate) sparse_mask_fill_us: Option<u64>,
    pub(crate) sparse_mask_topk_nodes: Option<u64>,
    pub(crate) sparse_mask_topk_us: Option<u64>,
    pub(crate) sparse_mask_add_nodes: Option<u64>,
    pub(crate) sparse_mask_add_us: Option<u64>,
    pub(crate) dsa_sparse_attn_nodes: Option<u64>,
    pub(crate) dsa_sparse_attn_us: Option<u64>,
    pub(crate) mla_attention_nodes: u64,
    pub(crate) mla_attention_us: u64,
    pub(crate) routed_moe_nodes: u64,
    pub(crate) routed_moe_us: u64,
    pub(crate) shared_expert_nodes: u64,
    pub(crate) shared_expert_us: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct TimingGroupRecord {
    pub(crate) record_index: usize,
    pub(crate) group: String,
    pub(crate) timing: TimingRecord,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct DirectSparseDecisionRecord {
    pub(crate) layer: i32,
    pub(crate) ubatch_tokens: i64,
    pub(crate) sparse_batch: i64,
    pub(crate) sparse_streams: i64,
    pub(crate) prefill_cap: i64,
    pub(crate) direct_enabled: bool,
    pub(crate) prefill_enabled: bool,
    pub(crate) decode_shape: bool,
    pub(crate) prefill_shape: bool,
    pub(crate) token_shape_allowed: bool,
    pub(crate) kq_b_ok: bool,
    pub(crate) sinks_ok: bool,
    pub(crate) alibi_ok: bool,
    pub(crate) soft_cap_ok: bool,
    pub(crate) use_direct: bool,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct SidebandSummary {
    records: usize,
    tokens: u64,
    hidden_bytes: u64,
    sideband_bytes: u64,
    sideband_i32: u64,
    causal_visible_sideband_i32: u64,
    padded_sideband_i32: u64,
    avg_hidden_bytes_per_token: Option<f64>,
    avg_sideband_bytes_per_token: Option<f64>,
    avg_sideband_i32_per_token: Option<f64>,
    avg_causal_visible_sideband_i32_per_token: Option<f64>,
    sideband_padding_ratio: Option<f64>,
    sideband_to_hidden_ratio: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SidebandRecord {
    stage: String,
    kind: String,
    pos_start: u64,
    tokens: u64,
    hidden_bytes: u64,
    sideband_bytes: u64,
    sideband_i32: u64,
}

pub fn glm_dsa_op_report(args: GlmDsaOpReportArgs) -> Result<()> {
    let output = args.output.clone();
    let report = build_report(&args)?;
    let encoded = serde_json::to_vec_pretty(&report)?;
    if let Some(path) = output {
        fs::write(&path, &encoded).with_context(|| format!("write {}", path.display()))?;
    }
    println!("{}", String::from_utf8(encoded)?);
    Ok(())
}

pub fn glm_dsa_op_compare(args: GlmDsaOpCompareArgs) -> Result<()> {
    let output = args.output.clone();
    let report = build_comparison_report(&args)?;
    let encoded = serde_json::to_vec_pretty(&report)?;
    if let Some(path) = output {
        fs::write(&path, &encoded).with_context(|| format!("write {}", path.display()))?;
    }
    println!("{}", String::from_utf8(encoded)?);
    Ok(())
}

fn build_report(args: &GlmDsaOpReportArgs) -> Result<GlmDsaOpReport> {
    let mut logs = Vec::with_capacity(args.log.len());
    for path in &args.log {
        let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let records = parse_timing_records(&text)
            .with_context(|| format!("parse GLM-DSA op timing records in {}", path.display()))?;
        if records.is_empty() {
            bail!("{} contains no GLM-DSA op timing records", path.display());
        }
        let sideband_records = parse_sideband_records(&text).with_context(|| {
            format!("parse GLM-DSA top-k sideband records in {}", path.display())
        })?;
        let group_records = parse_timing_group_records(&text)
            .with_context(|| format!("parse GLM-DSA group timing records in {}", path.display()))?;
        let records = match args.first_records {
            Some(limit) => records.into_iter().take(limit).collect::<Vec<_>>(),
            None => records,
        };
        let group_records = match args.first_records {
            Some(limit) => group_records
                .into_iter()
                .filter(|record| record.record_index < limit)
                .collect::<Vec<_>>(),
            None => group_records,
        };
        let sideband_records = match args.first_records {
            Some(limit) => sideband_records.into_iter().take(limit).collect::<Vec<_>>(),
            None => sideband_records,
        };
        logs.push(summarize_log(
            path.clone(),
            &records,
            &group_records,
            &sideband_records,
        ));
    }
    Ok(GlmDsaOpReport { logs })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct ComparisonKey {
    stage: i32,
    phase: Phase,
}

#[derive(Debug, Serialize)]
struct GlmDsaOpComparisonReport {
    baseline_reports: Vec<PathBuf>,
    candidate_reports: Vec<PathBuf>,
    summary: GlmDsaOpComparisonSummary,
    rows: Vec<GlmDsaOpComparisonRow>,
}

#[derive(Debug, Default, Serialize)]
struct GlmDsaOpComparisonSummary {
    rows: usize,
    candidate_sparse_mask_eliminated_rows: usize,
    candidate_direct_sparse_rows: usize,
    faster_rows: usize,
    slower_rows: usize,
    prefill_rows: usize,
    prefill_slower_rows: usize,
    decode_rows: usize,
    decode_faster_rows: usize,
}

#[derive(Debug, Serialize)]
struct GlmDsaOpComparisonRow {
    stage: i32,
    phase: Phase,
    baseline_tokens: u64,
    candidate_tokens: u64,
    baseline_total_us: u64,
    candidate_total_us: u64,
    total_us_ratio: Option<f64>,
    baseline_avg_total_us_per_token: Option<f64>,
    candidate_avg_total_us_per_token: Option<f64>,
    avg_total_us_per_token_ratio: Option<f64>,
    baseline_sparse_mask_us: u64,
    candidate_sparse_mask_us: u64,
    sparse_mask_us_delta: i128,
    candidate_eliminated_sparse_mask: bool,
    baseline_dsa_sparse_attn_us: u64,
    candidate_dsa_sparse_attn_us: u64,
    dsa_sparse_attn_us_delta: i128,
    candidate_uses_direct_sparse_attn: bool,
    baseline_indexer_topk_us: u64,
    candidate_indexer_topk_us: u64,
    indexer_topk_us_ratio: Option<f64>,
    baseline_shared_expert_us: u64,
    candidate_shared_expert_us: u64,
    shared_expert_us_ratio: Option<f64>,
}

fn build_comparison_report(args: &GlmDsaOpCompareArgs) -> Result<GlmDsaOpComparisonReport> {
    let baseline = load_phase_summaries(&args.baseline_report, "baseline")?;
    let candidate = load_phase_summaries(&args.candidate_report, "candidate")?;
    let mut rows = Vec::with_capacity(baseline.len());
    for (key, baseline_summary) in &baseline {
        let candidate_summary = candidate.get(key).with_context(|| {
            format!(
                "candidate report is missing stage {} {:?}",
                key.stage, key.phase
            )
        })?;
        rows.push(compare_phase(*key, baseline_summary, candidate_summary));
    }
    for key in candidate.keys() {
        if !baseline.contains_key(key) {
            bail!(
                "candidate report has no matching baseline for stage {} {:?}",
                key.stage,
                key.phase
            );
        }
    }
    let summary = summarize_comparison_rows(&rows);
    Ok(GlmDsaOpComparisonReport {
        baseline_reports: args.baseline_report.clone(),
        candidate_reports: args.candidate_report.clone(),
        summary,
        rows,
    })
}

fn load_phase_summaries(
    paths: &[PathBuf],
    label: &str,
) -> Result<BTreeMap<ComparisonKey, PhaseSummary>> {
    let mut summaries = BTreeMap::new();
    for path in paths {
        let text = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let report = serde_json::from_slice::<GlmDsaOpReport>(&text)
            .with_context(|| format!("parse {label} report {}", path.display()))?;
        for log in report.logs {
            for (stage, phases) in log.stage_records {
                for (phase, summary) in phases {
                    let key = ComparisonKey { stage, phase };
                    if summaries.insert(key, summary).is_some() {
                        bail!("{label} report contains duplicate stage {stage} {phase:?}");
                    }
                }
            }
        }
    }
    Ok(summaries)
}

fn compare_phase(
    key: ComparisonKey,
    baseline: &PhaseSummary,
    candidate: &PhaseSummary,
) -> GlmDsaOpComparisonRow {
    let baseline_dsa_sparse_attn_us = optional_elapsed_us(&baseline.dsa_sparse_attn);
    let candidate_dsa_sparse_attn_us = optional_elapsed_us(&candidate.dsa_sparse_attn);
    GlmDsaOpComparisonRow {
        stage: key.stage,
        phase: key.phase,
        baseline_tokens: baseline.tokens,
        candidate_tokens: candidate.tokens,
        baseline_total_us: baseline.total_us,
        candidate_total_us: candidate.total_us,
        total_us_ratio: ratio(candidate.total_us, baseline.total_us),
        baseline_avg_total_us_per_token: baseline.avg_total_us_per_token,
        candidate_avg_total_us_per_token: candidate.avg_total_us_per_token,
        avg_total_us_per_token_ratio: option_ratio(
            candidate.avg_total_us_per_token,
            baseline.avg_total_us_per_token,
        ),
        baseline_sparse_mask_us: baseline.sparse_mask.elapsed_us,
        candidate_sparse_mask_us: candidate.sparse_mask.elapsed_us,
        sparse_mask_us_delta: delta(
            candidate.sparse_mask.elapsed_us,
            baseline.sparse_mask.elapsed_us,
        ),
        candidate_eliminated_sparse_mask: candidate.sparse_mask.elapsed_us == 0
            && baseline.sparse_mask.elapsed_us > 0,
        baseline_dsa_sparse_attn_us,
        candidate_dsa_sparse_attn_us,
        dsa_sparse_attn_us_delta: delta(candidate_dsa_sparse_attn_us, baseline_dsa_sparse_attn_us),
        candidate_uses_direct_sparse_attn: candidate_dsa_sparse_attn_us > 0,
        baseline_indexer_topk_us: baseline.indexer_topk.elapsed_us,
        candidate_indexer_topk_us: candidate.indexer_topk.elapsed_us,
        indexer_topk_us_ratio: ratio(
            candidate.indexer_topk.elapsed_us,
            baseline.indexer_topk.elapsed_us,
        ),
        baseline_shared_expert_us: baseline.shared_expert.elapsed_us,
        candidate_shared_expert_us: candidate.shared_expert.elapsed_us,
        shared_expert_us_ratio: ratio(
            candidate.shared_expert.elapsed_us,
            baseline.shared_expert.elapsed_us,
        ),
    }
}

fn summarize_comparison_rows(rows: &[GlmDsaOpComparisonRow]) -> GlmDsaOpComparisonSummary {
    let mut summary = GlmDsaOpComparisonSummary {
        rows: rows.len(),
        ..Default::default()
    };
    for row in rows {
        if row.candidate_eliminated_sparse_mask {
            summary.candidate_sparse_mask_eliminated_rows += 1;
        }
        if row.candidate_uses_direct_sparse_attn {
            summary.candidate_direct_sparse_rows += 1;
        }
        if matches!(row.avg_total_us_per_token_ratio, Some(ratio) if ratio < 1.0) {
            summary.faster_rows += 1;
        }
        if matches!(row.avg_total_us_per_token_ratio, Some(ratio) if ratio > 1.0) {
            summary.slower_rows += 1;
        }
        match row.phase {
            Phase::Prefill => {
                summary.prefill_rows += 1;
                if matches!(row.avg_total_us_per_token_ratio, Some(ratio) if ratio > 1.0) {
                    summary.prefill_slower_rows += 1;
                }
            }
            Phase::Decode => {
                summary.decode_rows += 1;
                if matches!(row.avg_total_us_per_token_ratio, Some(ratio) if ratio < 1.0) {
                    summary.decode_faster_rows += 1;
                }
            }
        }
    }
    summary
}

fn optional_elapsed_us(bucket: &Option<OpBucket>) -> u64 {
    bucket.as_ref().map_or(0, |bucket| bucket.elapsed_us)
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

fn option_ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    numerator
        .zip(denominator)
        .and_then(|(numerator, denominator)| {
            (denominator != 0.0).then_some(numerator / denominator)
        })
}

fn delta(candidate: u64, baseline: u64) -> i128 {
    i128::from(candidate) - i128::from(baseline)
}

pub(crate) fn parse_timing_records(text: &str) -> Result<Vec<TimingRecord>> {
    text.lines()
        .filter_map(|line| {
            line.find(OP_TIMING_PREFIX)
                .map(|index| &line[index + OP_TIMING_PREFIX.len()..])
        })
        .map(parse_timing_record)
        .collect()
}

fn parse_timing_record(line: &str) -> Result<TimingRecord> {
    let fields = line
        .split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect::<BTreeMap<_, _>>();
    parse_timing_fields(&fields)
}

pub(crate) fn parse_timing_group_records(text: &str) -> Result<Vec<TimingGroupRecord>> {
    let mut records = Vec::new();
    let mut timing_record_count = 0usize;
    for line in text.lines() {
        if line.contains(OP_TIMING_PREFIX) {
            timing_record_count += 1;
            continue;
        }
        let Some(index) = line.find(GROUP_TIMING_PREFIX) else {
            continue;
        };
        let record_index = timing_record_count.saturating_sub(1);
        records.push(parse_timing_group_record(
            record_index,
            &line[index + GROUP_TIMING_PREFIX.len()..],
        )?);
    }
    Ok(records)
}

pub(crate) fn parse_direct_sparse_decision_records(
    text: &str,
) -> Result<Vec<DirectSparseDecisionRecord>> {
    text.lines()
        .filter_map(|line| {
            line.find(DIRECT_SPARSE_DECISION_PREFIX)
                .map(|index| &line[index + DIRECT_SPARSE_DECISION_PREFIX.len()..])
        })
        .map(parse_direct_sparse_decision_record)
        .collect()
}

fn parse_direct_sparse_decision_record(line: &str) -> Result<DirectSparseDecisionRecord> {
    let fields = line
        .split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect::<BTreeMap<_, _>>();
    Ok(DirectSparseDecisionRecord {
        layer: parse_field(&fields, "layer")?,
        ubatch_tokens: parse_field(&fields, "ubatch_tokens")?,
        sparse_batch: parse_field(&fields, "sparse_batch")?,
        sparse_streams: parse_field(&fields, "sparse_streams")?,
        prefill_cap: parse_field(&fields, "prefill_cap")?,
        direct_enabled: parse_bool_int_field(&fields, "direct_enabled")?,
        prefill_enabled: parse_bool_int_field(&fields, "prefill_enabled")?,
        decode_shape: parse_bool_int_field(&fields, "decode_shape")?,
        prefill_shape: parse_bool_int_field(&fields, "prefill_shape")?,
        token_shape_allowed: parse_bool_int_field(&fields, "token_shape_allowed")?,
        kq_b_ok: parse_bool_int_field(&fields, "kq_b_ok")?,
        sinks_ok: parse_bool_int_field(&fields, "sinks_ok")?,
        alibi_ok: parse_bool_int_field(&fields, "alibi_ok")?,
        soft_cap_ok: parse_bool_int_field(&fields, "soft_cap_ok")?,
        use_direct: parse_bool_int_field(&fields, "use_direct")?,
    })
}

fn parse_timing_group_record(record_index: usize, line: &str) -> Result<TimingGroupRecord> {
    let fields = line
        .split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect::<BTreeMap<_, _>>();
    Ok(TimingGroupRecord {
        record_index,
        group: parse_string_field(&fields, "group")?,
        timing: parse_timing_fields(&fields)?,
    })
}

fn parse_timing_fields(fields: &BTreeMap<&str, &str>) -> Result<TimingRecord> {
    let indexer = parse_optional_bucket(fields, "indexer")?;
    let top_k = parse_optional_bucket(fields, "top_k")?;
    let sparse_mask_fill = parse_optional_bucket(fields, "sparse_mask_fill")?;
    let sparse_mask_topk = parse_optional_bucket(fields, "sparse_mask_topk")?;
    let sparse_mask_add = parse_optional_bucket(fields, "sparse_mask_add")?;
    let dsa_sparse_attn = parse_optional_bucket(fields, "dsa_sparse_attn")?;
    Ok(TimingRecord {
        stage: parse_field(fields, "stage")?,
        tokens: parse_field(fields, "tokens")?,
        total_us: parse_field(fields, "total_us")?,
        indexer_topk_nodes: parse_field(fields, "indexer_topk_nodes")?,
        indexer_topk_us: parse_field(fields, "indexer_topk_us")?,
        indexer_nodes: indexer.nodes,
        indexer_us: indexer.elapsed_us,
        top_k_nodes: top_k.nodes,
        top_k_us: top_k.elapsed_us,
        sparse_mask_nodes: parse_field(fields, "sparse_mask_nodes")?,
        sparse_mask_us: parse_field(fields, "sparse_mask_us")?,
        sparse_mask_fill_nodes: sparse_mask_fill.nodes,
        sparse_mask_fill_us: sparse_mask_fill.elapsed_us,
        sparse_mask_topk_nodes: sparse_mask_topk.nodes,
        sparse_mask_topk_us: sparse_mask_topk.elapsed_us,
        sparse_mask_add_nodes: sparse_mask_add.nodes,
        sparse_mask_add_us: sparse_mask_add.elapsed_us,
        dsa_sparse_attn_nodes: dsa_sparse_attn.nodes,
        dsa_sparse_attn_us: dsa_sparse_attn.elapsed_us,
        mla_attention_nodes: parse_field(fields, "mla_attention_nodes")?,
        mla_attention_us: parse_field(fields, "mla_attention_us")?,
        routed_moe_nodes: parse_field(fields, "routed_moe_nodes")?,
        routed_moe_us: parse_field(fields, "routed_moe_us")?,
        shared_expert_nodes: parse_field(fields, "shared_expert_nodes")?,
        shared_expert_us: parse_field(fields, "shared_expert_us")?,
    })
}

#[derive(Debug, Clone, Copy)]
struct OptionalBucketFields {
    nodes: Option<u64>,
    elapsed_us: Option<u64>,
}

fn parse_optional_bucket(
    fields: &BTreeMap<&str, &str>,
    name: &str,
) -> Result<OptionalBucketFields> {
    let nodes = parse_optional_field(fields, &format!("{name}_nodes"))?;
    let elapsed_us = parse_optional_field(fields, &format!("{name}_us"))?;
    if nodes.is_some() != elapsed_us.is_some() {
        bail!("{name} must include both nodes and us fields");
    }
    Ok(OptionalBucketFields { nodes, elapsed_us })
}

fn parse_sideband_records(text: &str) -> Result<Vec<SidebandRecord>> {
    text.lines()
        .filter_map(|line| {
            line.find(SIDEBAND_PREFIX)
                .map(|index| &line[index + SIDEBAND_PREFIX.len()..])
        })
        .map(parse_sideband_record)
        .collect()
}

fn parse_sideband_record(line: &str) -> Result<SidebandRecord> {
    let fields = line
        .split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect::<BTreeMap<_, _>>();
    Ok(SidebandRecord {
        stage: parse_string_field(&fields, "stage")?,
        kind: parse_string_field(&fields, "kind")?,
        pos_start: parse_field(&fields, "pos_start")?,
        tokens: parse_field(&fields, "tokens")?,
        hidden_bytes: parse_field(&fields, "hidden_bytes")?,
        sideband_bytes: parse_field(&fields, "sideband_bytes")?,
        sideband_i32: parse_field(&fields, "sideband_i32")?,
    })
}

fn parse_field<T>(fields: &BTreeMap<&str, &str>, name: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    fields
        .get(name)
        .with_context(|| format!("missing {name}"))?
        .parse::<T>()
        .map_err(|error| anyhow::anyhow!("invalid {name}: {error}"))
}

fn parse_optional_field<T>(fields: &BTreeMap<&str, &str>, name: &str) -> Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    fields
        .get(name)
        .map(|value| {
            value
                .parse::<T>()
                .map_err(|error| anyhow::anyhow!("invalid {name}: {error}"))
        })
        .transpose()
}

fn parse_bool_int_field(fields: &BTreeMap<&str, &str>, name: &str) -> Result<bool> {
    let value: u8 = parse_field(fields, name)?;
    Ok(value != 0)
}

fn parse_string_field(fields: &BTreeMap<&str, &str>, name: &str) -> Result<String> {
    Ok(fields
        .get(name)
        .with_context(|| format!("missing {name}"))?
        .to_string())
}

fn summarize_log(
    path: PathBuf,
    records: &[TimingRecord],
    group_records: &[TimingGroupRecord],
    sideband_records: &[SidebandRecord],
) -> LogSummary {
    let mut stage_records: BTreeMap<i32, BTreeMap<Phase, PhaseSummary>> = BTreeMap::new();
    for record in records {
        let summary = stage_records
            .entry(record.stage)
            .or_default()
            .entry(timing_phase(record))
            .or_default();
        add_timing_record(summary, record);
    }
    finalize_phase_summaries(&mut stage_records);

    let mut grouped_records: BTreeMap<i32, BTreeMap<String, BTreeMap<Phase, PhaseSummary>>> =
        BTreeMap::new();
    for record in group_records {
        let timing = &record.timing;
        let summary = grouped_records
            .entry(timing.stage)
            .or_default()
            .entry(record.group.clone())
            .or_default()
            .entry(timing_phase(timing))
            .or_default();
        add_timing_record(summary, timing);
    }
    for groups in grouped_records.values_mut() {
        for phases in groups.values_mut() {
            finalize_phase_summary_map(phases);
        }
    }
    let hottest_group_records = summarize_hottest_groups(&grouped_records);

    let sideband_records = summarize_sideband_records(sideband_records);
    LogSummary {
        path,
        records: records.len(),
        stage_records,
        group_records: grouped_records,
        hottest_group_records,
        sideband_records,
    }
}

fn summarize_hottest_groups(
    group_records: &BTreeMap<i32, BTreeMap<String, BTreeMap<Phase, PhaseSummary>>>,
) -> Vec<HotGroupSummary> {
    let mut hottest: BTreeMap<(i32, Phase), HotGroupSummary> = BTreeMap::new();
    for (stage, groups) in group_records {
        for (group, phases) in groups {
            for (phase, summary) in phases {
                let key = (*stage, *phase);
                let candidate = HotGroupSummary {
                    stage: *stage,
                    phase: *phase,
                    group: group.clone(),
                    summary: summary.clone(),
                };
                match hottest.get(&key) {
                    Some(existing) if existing.summary.total_us >= summary.total_us => {}
                    _ => {
                        hottest.insert(key, candidate);
                    }
                }
            }
        }
    }
    hottest.into_values().collect()
}

fn timing_phase(record: &TimingRecord) -> Phase {
    if record.tokens == 1 {
        Phase::Decode
    } else {
        Phase::Prefill
    }
}

fn add_timing_record(summary: &mut PhaseSummary, record: &TimingRecord) {
    summary.records += 1;
    summary.tokens += record.tokens;
    summary.total_us += record.total_us;
    add_bucket(
        &mut summary.indexer_topk,
        record.indexer_topk_nodes,
        record.indexer_topk_us,
    );
    add_optional_bucket(
        &mut summary.indexer,
        record.indexer_nodes,
        record.indexer_us,
    );
    add_optional_bucket(&mut summary.top_k, record.top_k_nodes, record.top_k_us);
    add_bucket(
        &mut summary.sparse_mask,
        record.sparse_mask_nodes,
        record.sparse_mask_us,
    );
    add_optional_bucket(
        &mut summary.sparse_mask_fill,
        record.sparse_mask_fill_nodes,
        record.sparse_mask_fill_us,
    );
    add_optional_bucket(
        &mut summary.sparse_mask_topk,
        record.sparse_mask_topk_nodes,
        record.sparse_mask_topk_us,
    );
    add_optional_bucket(
        &mut summary.sparse_mask_add,
        record.sparse_mask_add_nodes,
        record.sparse_mask_add_us,
    );
    add_optional_bucket(
        &mut summary.dsa_sparse_attn,
        record.dsa_sparse_attn_nodes,
        record.dsa_sparse_attn_us,
    );
    add_bucket(
        &mut summary.mla_attention,
        record.mla_attention_nodes,
        record.mla_attention_us,
    );
    add_bucket(
        &mut summary.routed_moe,
        record.routed_moe_nodes,
        record.routed_moe_us,
    );
    add_bucket(
        &mut summary.shared_expert,
        record.shared_expert_nodes,
        record.shared_expert_us,
    );
}

fn finalize_phase_summaries(records: &mut BTreeMap<i32, BTreeMap<Phase, PhaseSummary>>) {
    for phases in records.values_mut() {
        finalize_phase_summary_map(phases);
    }
}

fn finalize_phase_summary_map(phases: &mut BTreeMap<Phase, PhaseSummary>) {
    for summary in phases.values_mut() {
        summary.avg_total_us_per_record = nonzero_div(summary.total_us, summary.records as u64);
        summary.avg_total_us_per_token = nonzero_div(summary.total_us, summary.tokens);
    }
}

fn summarize_sideband_records(
    records: &[SidebandRecord],
) -> BTreeMap<String, BTreeMap<Phase, SidebandSummary>> {
    let mut stages: BTreeMap<String, BTreeMap<Phase, SidebandSummary>> = BTreeMap::new();
    for record in records {
        let phase = sideband_phase(&record.kind, record.tokens);
        let summary = stages
            .entry(record.stage.clone())
            .or_default()
            .entry(phase)
            .or_default();
        summary.records += 1;
        summary.tokens += record.tokens;
        summary.hidden_bytes += record.hidden_bytes;
        summary.sideband_bytes += record.sideband_bytes;
        summary.sideband_i32 += record.sideband_i32;
        let causal_visible_sideband_i32 = causal_visible_sideband_i32(record);
        summary.causal_visible_sideband_i32 += causal_visible_sideband_i32;
        summary.padded_sideband_i32 += record
            .sideband_i32
            .saturating_sub(causal_visible_sideband_i32);
    }
    for phases in stages.values_mut() {
        for summary in phases.values_mut() {
            summary.avg_hidden_bytes_per_token = nonzero_div(summary.hidden_bytes, summary.tokens);
            summary.avg_sideband_bytes_per_token =
                nonzero_div(summary.sideband_bytes, summary.tokens);
            summary.avg_sideband_i32_per_token = nonzero_div(summary.sideband_i32, summary.tokens);
            summary.avg_causal_visible_sideband_i32_per_token =
                nonzero_div(summary.causal_visible_sideband_i32, summary.tokens);
            summary.sideband_padding_ratio =
                nonzero_div(summary.padded_sideband_i32, summary.sideband_i32);
            summary.sideband_to_hidden_ratio =
                nonzero_div(summary.sideband_bytes, summary.hidden_bytes);
        }
    }
    stages
}

fn causal_visible_sideband_i32(record: &SidebandRecord) -> u64 {
    if record.tokens == 0 || record.sideband_i32 == 0 {
        return 0;
    }
    let sideband_width = record.sideband_i32 / record.tokens;
    (0..record.tokens)
        .map(|token_index| {
            let causal_visible_width = record
                .pos_start
                .saturating_add(token_index)
                .saturating_add(1);
            sideband_width.min(causal_visible_width)
        })
        .sum()
}

fn sideband_phase(kind: &str, tokens: u64) -> Phase {
    if kind == "DecodeEmbd" || tokens == 1 {
        Phase::Decode
    } else {
        Phase::Prefill
    }
}

fn add_bucket(bucket: &mut OpBucket, nodes: u64, elapsed_us: u64) {
    bucket.nodes += nodes;
    bucket.elapsed_us += elapsed_us;
}

fn add_optional_bucket(bucket: &mut Option<OpBucket>, nodes: Option<u64>, elapsed_us: Option<u64>) {
    if let (Some(nodes), Some(elapsed_us)) = (nodes, elapsed_us) {
        add_bucket(
            bucket.get_or_insert_with(OpBucket::default),
            nodes,
            elapsed_us,
        );
    }
}

fn nonzero_div(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

#[cfg(test)]
mod tests {
    use super::{
        ComparisonKey, OpBucket, Phase, PhaseSummary, compare_phase,
        parse_direct_sparse_decision_records, parse_sideband_record, parse_sideband_records,
        parse_timing_group_records, parse_timing_record, parse_timing_records,
        summarize_comparison_rows, summarize_log,
    };

    const LINE: &str = "skippy: glm_dsa_op_timing stage=1 tokens=128 total_us=1475800 indexer_topk_nodes=275 indexer_topk_us=129065 sparse_mask_nodes=235 sparse_mask_us=114543 mla_attention_nodes=47 mla_attention_us=35234 routed_moe_nodes=47 routed_moe_us=379574 shared_expert_nodes=47 shared_expert_us=817384";
    const LINE_WITH_INDEXER_BREAKDOWN: &str = "skippy: glm_dsa_op_timing stage=1 tokens=128 total_us=1475800 indexer_topk_nodes=275 indexer_topk_us=129065 indexer_nodes=235 indexer_us=80000 top_k_nodes=40 top_k_us=49065 sparse_mask_nodes=235 sparse_mask_us=114543 mla_attention_nodes=47 mla_attention_us=35234 routed_moe_nodes=47 routed_moe_us=379574 shared_expert_nodes=47 shared_expert_us=817384";
    const LINE_WITH_SPARSE_BREAKDOWN: &str = "skippy: glm_dsa_op_timing stage=1 tokens=128 total_us=1475800 indexer_topk_nodes=275 indexer_topk_us=129065 sparse_mask_nodes=235 sparse_mask_us=114543 sparse_mask_fill_nodes=47 sparse_mask_fill_us=1000 sparse_mask_topk_nodes=47 sparse_mask_topk_us=2000 sparse_mask_add_nodes=47 sparse_mask_add_us=3000 mla_attention_nodes=47 mla_attention_us=35234 routed_moe_nodes=47 routed_moe_us=379574 shared_expert_nodes=47 shared_expert_us=817384";
    const LINE_WITH_DSA_SPARSE_ATTN: &str = "skippy: glm_dsa_op_timing stage=1 tokens=128 total_us=1475800 indexer_topk_nodes=275 indexer_topk_us=129065 sparse_mask_nodes=0 sparse_mask_us=0 dsa_sparse_attn_nodes=47 dsa_sparse_attn_us=114543 mla_attention_nodes=47 mla_attention_us=35234 routed_moe_nodes=47 routed_moe_us=379574 shared_expert_nodes=47 shared_expert_us=817384";
    const GROUP_LINE_LAYER_0: &str = "skippy: glm_dsa_group_timing stage=1 tokens=128 group=layer_0 total_us=600000 indexer_topk_nodes=100 indexer_topk_us=50000 indexer_nodes=80 indexer_us=30000 top_k_nodes=20 top_k_us=20000 sparse_mask_nodes=40 sparse_mask_us=1000 sparse_mask_fill_nodes=0 sparse_mask_fill_us=0 sparse_mask_topk_nodes=40 sparse_mask_topk_us=1000 sparse_mask_add_nodes=0 sparse_mask_add_us=0 dsa_sparse_attn_nodes=0 dsa_sparse_attn_us=0 mla_attention_nodes=1 mla_attention_us=9000 routed_moe_nodes=0 routed_moe_us=0 shared_expert_nodes=0 shared_expert_us=0";
    const GROUP_LINE_LAYER_1: &str = "skippy: glm_dsa_group_timing stage=1 tokens=128 group=layer_1 total_us=875800 indexer_topk_nodes=175 indexer_topk_us=79065 indexer_nodes=155 indexer_us=50000 top_k_nodes=20 top_k_us=29065 sparse_mask_nodes=195 sparse_mask_us=113543 sparse_mask_fill_nodes=47 sparse_mask_fill_us=1000 sparse_mask_topk_nodes=47 sparse_mask_topk_us=2000 sparse_mask_add_nodes=47 sparse_mask_add_us=3000 dsa_sparse_attn_nodes=0 dsa_sparse_attn_us=0 mla_attention_nodes=46 mla_attention_us=26234 routed_moe_nodes=47 routed_moe_us=379574 shared_expert_nodes=47 shared_expert_us=817384";
    const SIDEBAND_LINE: &str = "skippy: glm_dsa_top_k_sideband_forward stage=stage-0 request=1 session=2 kind=DecodeEmbd pos_start=718 tokens=1 hidden_bytes=24576 sideband_bytes=3072 sideband_i32=768";
    const PADDED_PREFILL_SIDEBAND_LINE: &str = "skippy: glm_dsa_top_k_sideband_forward stage=stage-0 request=1 session=2 kind=PrefillEmbd pos_start=512 tokens=128 hidden_bytes=3145728 sideband_bytes=393216 sideband_i32=98304";
    const DIRECT_SPARSE_DECISION_LINE: &str = "skippy: glm_dsa_direct_sparse_decision layer=30 ubatch_tokens=33 sparse_batch=33 sparse_streams=1 prefill_cap=32 direct_enabled=1 prefill_enabled=1 decode_shape=0 prefill_shape=0 token_shape_allowed=0 kq_b_ok=1 sinks_ok=1 alibi_ok=1 soft_cap_ok=1 use_direct=0";

    #[test]
    fn parses_timing_record_with_prefix() {
        let records = parse_timing_records(LINE).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].stage, 1);
        assert_eq!(records[0].tokens, 128);
        assert_eq!(records[0].indexer_topk_us, 129065);
        assert_eq!(records[0].shared_expert_nodes, 47);
    }

    #[test]
    fn parses_optional_indexer_breakdown() {
        let record = parse_timing_record(LINE_WITH_INDEXER_BREAKDOWN).unwrap();
        assert_eq!(record.indexer_nodes, Some(235));
        assert_eq!(record.indexer_us, Some(80_000));
        assert_eq!(record.top_k_nodes, Some(40));
        assert_eq!(record.top_k_us, Some(49_065));

        let summary = summarize_log("stage1.log".into(), &[record], &[], &[]);
        let prefill = summary
            .stage_records
            .get(&1)
            .unwrap()
            .get(&Phase::Prefill)
            .unwrap();
        assert_eq!(prefill.indexer_topk.elapsed_us, 129_065);
        assert_eq!(prefill.indexer.as_ref().unwrap().elapsed_us, 80_000);
        assert_eq!(prefill.top_k.as_ref().unwrap().elapsed_us, 49_065);
    }

    #[test]
    fn parses_optional_sparse_mask_breakdown() {
        let record = parse_timing_record(LINE_WITH_SPARSE_BREAKDOWN).unwrap();
        assert_eq!(record.sparse_mask_fill_us, Some(1000));
        assert_eq!(record.sparse_mask_topk_us, Some(2000));
        assert_eq!(record.sparse_mask_add_us, Some(3000));

        let summary = summarize_log("stage1.log".into(), &[record], &[], &[]);
        let prefill = summary
            .stage_records
            .get(&1)
            .unwrap()
            .get(&Phase::Prefill)
            .unwrap();
        assert_eq!(prefill.sparse_mask.elapsed_us, 114543);
        assert_eq!(prefill.sparse_mask_fill.as_ref().unwrap().elapsed_us, 1000);
        assert_eq!(prefill.sparse_mask_topk.as_ref().unwrap().elapsed_us, 2000);
        assert_eq!(prefill.sparse_mask_add.as_ref().unwrap().elapsed_us, 3000);
    }

    #[test]
    fn parses_optional_dsa_sparse_attention_breakdown() {
        let record = parse_timing_record(LINE_WITH_DSA_SPARSE_ATTN).unwrap();
        assert_eq!(record.dsa_sparse_attn_nodes, Some(47));
        assert_eq!(record.dsa_sparse_attn_us, Some(114543));

        let summary = summarize_log("stage1.log".into(), &[record], &[], &[]);
        let prefill = summary
            .stage_records
            .get(&1)
            .unwrap()
            .get(&Phase::Prefill)
            .unwrap();
        assert_eq!(prefill.dsa_sparse_attn.as_ref().unwrap().elapsed_us, 114543);
    }

    #[test]
    fn parses_direct_sparse_decision_records() {
        let records = parse_direct_sparse_decision_records(DIRECT_SPARSE_DECISION_LINE).unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.layer, 30);
        assert_eq!(record.ubatch_tokens, 33);
        assert_eq!(record.sparse_batch, 33);
        assert_eq!(record.prefill_cap, 32);
        assert!(record.direct_enabled);
        assert!(record.prefill_enabled);
        assert!(!record.decode_shape);
        assert!(!record.prefill_shape);
        assert!(!record.token_shape_allowed);
        assert!(record.kq_b_ok);
        assert!(record.sinks_ok);
        assert!(record.alibi_ok);
        assert!(record.soft_cap_ok);
        assert!(!record.use_direct);
    }

    #[test]
    fn rejects_partial_sparse_mask_breakdown() {
        let error = parse_timing_record(&LINE.replace(
            "sparse_mask_nodes=235",
            "sparse_mask_nodes=235 sparse_mask_fill_nodes=47",
        ))
        .unwrap_err()
        .to_string();
        assert!(error.contains("sparse_mask_fill must include both nodes and us fields"));
    }

    #[test]
    fn rejects_partial_indexer_breakdown() {
        let error = parse_timing_record(&LINE.replace(
            "indexer_topk_nodes=275",
            "indexer_topk_nodes=275 indexer_nodes=235",
        ))
        .unwrap_err()
        .to_string();
        assert!(error.contains("indexer must include both nodes and us fields"));
    }

    #[test]
    fn rejects_missing_fields() {
        let error = parse_timing_record("stage=0 tokens=1")
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing total_us"));
    }

    #[test]
    fn summarizes_prefill_and_decode() {
        let text = format!(
            "{LINE}\n{}",
            LINE.replace("tokens=128", "tokens=1")
                .replace("total_us=1475800", "total_us=200")
        );
        let records = parse_timing_records(&text).unwrap();
        let summary = summarize_log("stage1.log".into(), &records, &[], &[]);
        let stages = summary.stage_records.get(&1).unwrap();
        let prefill = stages.get(&Phase::Prefill).unwrap();
        let decode = stages.get(&Phase::Decode).unwrap();
        assert_eq!(prefill.records, 1);
        assert_eq!(prefill.tokens, 128);
        assert_eq!(decode.records, 1);
        assert_eq!(decode.tokens, 1);
        assert_eq!(decode.total_us, 200);
    }

    #[test]
    fn parses_group_timing_records_with_parent_index() {
        let text = format!(
            "{LINE}\n{GROUP_LINE_LAYER_0}\n{GROUP_LINE_LAYER_1}\n{LINE_WITH_DSA_SPARSE_ATTN}"
        );
        let records = parse_timing_group_records(&text).unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].record_index, 0);
        assert_eq!(records[0].group, "layer_0");
        assert_eq!(records[0].timing.stage, 1);
        assert_eq!(records[0].timing.indexer_topk_us, 50_000);
        assert_eq!(records[1].record_index, 0);
        assert_eq!(records[1].group, "layer_1");
        assert_eq!(records[1].timing.sparse_mask_us, 113_543);
    }

    #[test]
    fn summarizes_group_timing_by_stage_group_and_phase() {
        let timing = parse_timing_records(LINE).unwrap();
        let groups = parse_timing_group_records(&format!(
            "{LINE}\n{GROUP_LINE_LAYER_0}\n{GROUP_LINE_LAYER_1}"
        ))
        .unwrap();
        let summary = summarize_log("stage1.log".into(), &timing, &groups, &[]);
        let stage = summary.group_records.get(&1).unwrap();
        let layer_0 = stage.get("layer_0").unwrap().get(&Phase::Prefill).unwrap();
        let layer_1 = stage.get("layer_1").unwrap().get(&Phase::Prefill).unwrap();

        assert_eq!(layer_0.records, 1);
        assert_eq!(layer_0.tokens, 128);
        assert_eq!(layer_0.total_us, 600_000);
        assert_eq!(layer_0.indexer_topk.elapsed_us, 50_000);
        assert_eq!(layer_0.avg_total_us_per_token, Some(4687.5));
        assert_eq!(layer_1.total_us, 875_800);
        assert_eq!(layer_1.sparse_mask.elapsed_us, 113_543);
        assert_eq!(summary.hottest_group_records.len(), 1);
        assert_eq!(summary.hottest_group_records[0].stage, 1);
        assert_eq!(summary.hottest_group_records[0].phase, Phase::Prefill);
        assert_eq!(summary.hottest_group_records[0].group, "layer_1");
        assert_eq!(summary.hottest_group_records[0].summary.total_us, 875_800);
    }

    #[test]
    fn parses_sideband_record_with_prefix() {
        let records = parse_sideband_records(SIDEBAND_LINE).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].stage, "stage-0");
        assert_eq!(records[0].kind, "DecodeEmbd");
        assert_eq!(records[0].pos_start, 718);
        assert_eq!(records[0].sideband_bytes, 3072);
        assert_eq!(records[0].sideband_i32, 768);
    }

    #[test]
    fn rejects_malformed_sideband_record() {
        let error = parse_sideband_record("stage=stage-0 kind=DecodeEmbd pos_start=0")
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing tokens"));
    }

    #[test]
    fn summarizes_sideband_payload_ratios() {
        let timing = parse_timing_records(LINE).unwrap();
        let sideband = parse_sideband_records(SIDEBAND_LINE).unwrap();
        let summary = summarize_log("stage0.log".into(), &timing, &[], &sideband);
        let stages = summary.sideband_records.get("stage-0").unwrap();
        let decode = stages.get(&Phase::Decode).unwrap();
        assert_eq!(decode.records, 1);
        assert_eq!(decode.tokens, 1);
        assert_eq!(decode.hidden_bytes, 24576);
        assert_eq!(decode.sideband_bytes, 3072);
        assert_eq!(decode.sideband_i32, 768);
        assert_eq!(decode.causal_visible_sideband_i32, 719);
        assert_eq!(decode.padded_sideband_i32, 49);
        assert_eq!(decode.avg_sideband_bytes_per_token, Some(3072.0));
        assert_eq!(decode.avg_sideband_i32_per_token, Some(768.0));
        assert_eq!(
            decode.avg_causal_visible_sideband_i32_per_token,
            Some(719.0)
        );
        assert_eq!(decode.sideband_padding_ratio, Some(49.0 / 768.0));
        assert_eq!(decode.sideband_to_hidden_ratio, Some(0.125));
    }

    #[test]
    fn summarizes_sideband_padding_for_prefill() {
        let timing = parse_timing_records(LINE).unwrap();
        let sideband = parse_sideband_records(PADDED_PREFILL_SIDEBAND_LINE).unwrap();
        let summary = summarize_log("stage0.log".into(), &timing, &[], &sideband);
        let stages = summary.sideband_records.get("stage-0").unwrap();
        let prefill = stages.get(&Phase::Prefill).unwrap();
        assert_eq!(prefill.tokens, 128);
        assert_eq!(prefill.sideband_i32, 98_304);
        assert_eq!(prefill.causal_visible_sideband_i32, 73_792);
        assert_eq!(prefill.padded_sideband_i32, 24_512);
        assert_eq!(
            prefill.avg_causal_visible_sideband_i32_per_token,
            Some(576.5)
        );
        assert_eq!(prefill.sideband_padding_ratio, Some(24_512.0 / 98_304.0));
    }

    #[test]
    fn compares_sparse_mask_elimination_and_direct_sparse_cost() {
        let baseline = phase_summary(128, 12_800, 2_000, 0, 1_000, 2_000);
        let candidate = phase_summary(128, 25_600, 0, 7_500, 1_100, 2_100);
        let row = compare_phase(
            ComparisonKey {
                stage: 0,
                phase: Phase::Prefill,
            },
            &baseline,
            &candidate,
        );

        assert_eq!(row.avg_total_us_per_token_ratio, Some(2.0));
        assert_eq!(row.sparse_mask_us_delta, -2_000);
        assert!(row.candidate_eliminated_sparse_mask);
        assert_eq!(row.dsa_sparse_attn_us_delta, 7_500);
        assert!(row.candidate_uses_direct_sparse_attn);
    }

    #[test]
    fn summarizes_prefill_regression_and_decode_improvement() {
        let baseline_prefill = phase_summary(128, 12_800, 2_000, 0, 1_000, 2_000);
        let candidate_prefill = phase_summary(128, 25_600, 0, 7_500, 1_100, 2_100);
        let baseline_decode = phase_summary(1, 400, 50, 0, 20, 100);
        let candidate_decode = phase_summary(1, 300, 0, 80, 21, 100);
        let rows = vec![
            compare_phase(
                ComparisonKey {
                    stage: 0,
                    phase: Phase::Prefill,
                },
                &baseline_prefill,
                &candidate_prefill,
            ),
            compare_phase(
                ComparisonKey {
                    stage: 0,
                    phase: Phase::Decode,
                },
                &baseline_decode,
                &candidate_decode,
            ),
        ];

        let summary = summarize_comparison_rows(&rows);
        assert_eq!(summary.rows, 2);
        assert_eq!(summary.candidate_sparse_mask_eliminated_rows, 2);
        assert_eq!(summary.candidate_direct_sparse_rows, 2);
        assert_eq!(summary.prefill_slower_rows, 1);
        assert_eq!(summary.decode_faster_rows, 1);
    }

    fn phase_summary(
        tokens: u64,
        total_us: u64,
        sparse_mask_us: u64,
        dsa_sparse_attn_us: u64,
        indexer_topk_us: u64,
        shared_expert_us: u64,
    ) -> PhaseSummary {
        PhaseSummary {
            records: 1,
            tokens,
            total_us,
            avg_total_us_per_record: Some(total_us as f64),
            avg_total_us_per_token: Some(total_us as f64 / tokens as f64),
            indexer_topk: OpBucket {
                nodes: 1,
                elapsed_us: indexer_topk_us,
            },
            sparse_mask: OpBucket {
                nodes: u64::from(sparse_mask_us > 0),
                elapsed_us: sparse_mask_us,
            },
            dsa_sparse_attn: (dsa_sparse_attn_us > 0).then_some(OpBucket {
                nodes: 1,
                elapsed_us: dsa_sparse_attn_us,
            }),
            shared_expert: OpBucket {
                nodes: 1,
                elapsed_us: shared_expert_us,
            },
            ..PhaseSummary::default()
        }
    }
}
