use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::cli::{GlmDsaOpCompareArgs, GlmDsaOpReportArgs};

const OP_TIMING_PREFIX: &str = "skippy: glm_dsa_op_timing ";
const GROUP_TIMING_PREFIX: &str = "skippy: glm_dsa_group_timing ";
const SIDEBAND_FORWARD_PREFIX: &str = "skippy: glm_dsa_top_k_sideband_forward ";
const SIDEBAND_RECEIVE_PREFIX: &str = "skippy: glm_dsa_top_k_sideband_receive ";
const DIRECT_SPARSE_DECISION_PREFIX: &str = "skippy: glm_dsa_direct_sparse_decision ";
const COMPACT_FLASH_POLICY_PREFIX: &str = "skippy: glm_dsa_compact_flash_policy ";
const COMPACT_FLASH_MASK_PREFIX: &str = "skippy: glm_dsa_compact_flash_mask ";
const METAL_DISPATCH_PREFIX: &str = "skippy: glm_dsa_metal_dispatch ";
const HOT_TENSOR_PREFIX: &str = "skippy: glm_dsa_hot_tensor ";
const COMPUTE_BUFFER_PREFIX: &str = "~llama_context:";
const INDEXSHARE_TRACE_MARKER: &str = "GLM_DSA IndexShare ";

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
    #[serde(skip_serializing_if = "Option::is_none")]
    compact_get_rows: Option<OpBucket>,
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
    indexshare_trace: IndexShareTraceSummary,
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
    pub(crate) compact_get_rows_nodes: Option<u64>,
    pub(crate) compact_get_rows_us: Option<u64>,
    pub(crate) mla_attention_nodes: u64,
    pub(crate) mla_attention_us: u64,
    pub(crate) routed_moe_nodes: u64,
    pub(crate) routed_moe_us: u64,
    pub(crate) routed_moe_route_nodes: Option<u64>,
    pub(crate) routed_moe_route_us: Option<u64>,
    pub(crate) routed_moe_gate_up_nodes: Option<u64>,
    pub(crate) routed_moe_gate_up_us: Option<u64>,
    pub(crate) routed_moe_gate_nodes: Option<u64>,
    pub(crate) routed_moe_gate_us: Option<u64>,
    pub(crate) routed_moe_up_nodes: Option<u64>,
    pub(crate) routed_moe_up_us: Option<u64>,
    pub(crate) routed_moe_act_nodes: Option<u64>,
    pub(crate) routed_moe_act_us: Option<u64>,
    pub(crate) routed_moe_down_nodes: Option<u64>,
    pub(crate) routed_moe_down_us: Option<u64>,
    pub(crate) routed_moe_weighted_nodes: Option<u64>,
    pub(crate) routed_moe_weighted_us: Option<u64>,
    pub(crate) routed_moe_aggregate_nodes: Option<u64>,
    pub(crate) routed_moe_aggregate_us: Option<u64>,
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
pub(crate) struct HotTensorRecord {
    pub(crate) record_index: usize,
    pub(crate) stage: i32,
    pub(crate) tokens: u64,
    pub(crate) rank: u64,
    pub(crate) op: String,
    pub(crate) kind: String,
    pub(crate) elapsed_us: u64,
    pub(crate) name: String,
    pub(crate) ne0: i64,
    pub(crate) ne1: i64,
    pub(crate) ne2: i64,
    pub(crate) ne3: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ComputeBufferRecord {
    pub(crate) device: String,
    pub(crate) size_mib: f64,
    pub(crate) expected_mib: f64,
    pub(crate) matches_expectation: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct DirectSparseDecisionRecord {
    pub(crate) layer: i32,
    pub(crate) ubatch_tokens: i64,
    pub(crate) sparse_batch: i64,
    pub(crate) sparse_streams: i64,
    pub(crate) prefill_cap: i64,
    pub(crate) sparse_kv: Option<i64>,
    pub(crate) sparse_top_k: Option<i64>,
    pub(crate) min_kv_topk_ratio: Option<i64>,
    pub(crate) kv_topk_ratio: Option<i64>,
    pub(crate) dense_mask_bytes: Option<u64>,
    pub(crate) dense_mask_limit: Option<u64>,
    pub(crate) phase: Option<String>,
    pub(crate) selector_reason: Option<String>,
    pub(crate) direct_enabled: bool,
    pub(crate) prefill_enabled: bool,
    pub(crate) decode_shape: bool,
    pub(crate) prefill_shape: bool,
    pub(crate) large_prefill_shape: Option<bool>,
    pub(crate) token_shape_allowed: bool,
    pub(crate) kq_b_ok: bool,
    pub(crate) sinks_ok: bool,
    pub(crate) alibi_ok: bool,
    pub(crate) soft_cap_ok: bool,
    pub(crate) use_direct: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct CompactFlashPolicyRecord {
    pub(crate) layer: i32,
    pub(crate) ubatch_tokens: i64,
    pub(crate) visible_kv: i64,
    pub(crate) top_k: i64,
    pub(crate) kv_topk_ratio: i64,
    pub(crate) min_kv_topk_ratio: Option<i64>,
    pub(crate) forced: bool,
    pub(crate) disabled: bool,
    pub(crate) ratio_ok: Option<bool>,
    pub(crate) enabled: bool,
    pub(crate) flash_attn: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) phase: Option<String>,
    pub(crate) decode_shape: bool,
    pub(crate) kq_b_ok: bool,
    pub(crate) sinks_ok: bool,
    pub(crate) alibi_ok: bool,
    pub(crate) soft_cap_ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) no_mask: Option<bool>,
    pub(crate) use_compact: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) selector_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct CompactFlashMaskRecord {
    pub(crate) layer: i32,
    pub(crate) omitted_mla_kq_mask: bool,
    pub(crate) visible_kv: i64,
    pub(crate) ubatch_tokens: i64,
    pub(crate) streams: i64,
    pub(crate) max_top_k: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct MetalDispatchRecord {
    pub(crate) op: String,
    pub(crate) kernel: Option<String>,
    pub(crate) tensor: String,
    pub(crate) next: Option<String>,
    pub(crate) next_op: Option<String>,
    pub(crate) shared_gate: Option<String>,
    pub(crate) shared_up: Option<String>,
    pub(crate) weighted_sum: Option<String>,
    pub(crate) weighted_sum_op: Option<String>,
    pub(crate) reason: Option<String>,
    pub(crate) shared_branch: Option<bool>,
    pub(crate) weighted_sum_uses_down: Option<bool>,
    pub(crate) natural_order: Option<bool>,
    pub(crate) backend_candidate: Option<bool>,
    pub(crate) pair_fusable: Option<bool>,
    pub(crate) subgraph_fusable: Option<bool>,
    pub(crate) motif_nodes: Option<u64>,
    pub(crate) fusion_outputs: Option<u64>,
    pub(crate) filtered_gap: Option<u64>,
    pub(crate) graph_gap: Option<i64>,
    pub(crate) weighted_sum_gap: Option<i64>,
    pub(crate) weighted_sum_graph_gap: Option<i64>,
    pub(crate) parallel: Option<bool>,
    pub(crate) generic: Option<bool>,
    pub(crate) view: Option<bool>,
    pub(crate) get_rows_uses: Option<u64>,
    pub(crate) use_count: Option<u64>,
    pub(crate) consumer_count: Option<u64>,
    pub(crate) consumer_graph_idx: Option<i64>,
    pub(crate) consumer_op: Option<String>,
    pub(crate) consumer_tensor: Option<String>,
    pub(crate) consumer_src_slot: Option<i64>,
    pub(crate) flash_graph_idx: Option<i64>,
    pub(crate) q_type: Option<String>,
    pub(crate) k_type: Option<String>,
    pub(crate) v_type: Option<String>,
    pub(crate) mask_type: Option<String>,
    pub(crate) top_k_type: Option<String>,
    pub(crate) src_type: Option<String>,
    pub(crate) dst_type: Option<String>,
    pub(crate) q_width: Option<u64>,
    pub(crate) v_width: Option<u64>,
    pub(crate) batch: Option<u64>,
    pub(crate) heads: Option<u64>,
    pub(crate) stream: Option<u64>,
    pub(crate) kv: Option<u64>,
    pub(crate) top_k: Option<u64>,
    pub(crate) top_stream: Option<u64>,
    pub(crate) selected_keys: Option<u64>,
    pub(crate) q_read_bytes: Option<u64>,
    pub(crate) k_read_bytes: Option<u64>,
    pub(crate) v_read_bytes: Option<u64>,
    pub(crate) mask_read_bytes: Option<u64>,
    pub(crate) top_k_read_bytes: Option<u64>,
    pub(crate) scratch_per_tg_bytes: Option<u64>,
    pub(crate) score_fma: Option<u64>,
    pub(crate) value_fma: Option<u64>,
    pub(crate) reduction_strategy: Option<String>,
    pub(crate) rows: Option<u64>,
    pub(crate) partial_bytes: Option<u64>,
    pub(crate) softmax_bytes: Option<u64>,
    pub(crate) tmp_bytes: Option<u64>,
    pub(crate) nwg: Option<u64>,
    pub(crate) tmp_f16: Option<bool>,
    pub(crate) dst_partial: Option<bool>,
    pub(crate) grid_x: u64,
    pub(crate) grid_y: u64,
    pub(crate) grid_z: u64,
    pub(crate) threads_x: u64,
    pub(crate) threads_y: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct SidebandSummary {
    records: usize,
    forward_records: usize,
    receive_records: usize,
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
    direction: SidebandDirection,
    stage: String,
    kind: String,
    pos_start: u64,
    tokens: u64,
    hidden_bytes: u64,
    sideband_bytes: u64,
    sideband_i32: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SidebandDirection {
    Forward,
    Receive,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct IndexShareTraceSummary {
    pub(crate) records: usize,
    pub(crate) contract_records: usize,
    pub(crate) contract_sources: Vec<String>,
    pub(crate) contract_full_layers: Option<usize>,
    pub(crate) contract_shared_layers: Option<usize>,
    pub(crate) contract_indexer_tensor_layers: Option<usize>,
    pub(crate) contract_target_indexer_tensor_layers: Option<usize>,
    pub(crate) contract_nextn_layers: Option<usize>,
    pub(crate) exec_records: usize,
    pub(crate) full_exec_records: usize,
    pub(crate) shared_exec_records: usize,
    pub(crate) shared_exec_with_input_top_k: usize,
    pub(crate) shared_exec_missing_input_top_k: usize,
    pub(crate) top_k_records: usize,
    pub(crate) top_k_from_indexer: usize,
    pub(crate) top_k_from_full_visible: usize,
    pub(crate) consume_records: usize,
    pub(crate) min_consume_width: Option<i64>,
    pub(crate) max_consume_width: Option<i64>,
    pub(crate) full_layers: Vec<i32>,
    pub(crate) shared_layers: Vec<i32>,
}

impl IndexShareTraceSummary {
    pub(crate) fn is_empty(summary: &Self) -> bool {
        summary.records == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum IndexShareTraceEvent {
    Exec,
    TopK,
    Consume,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct IndexShareTraceRecord {
    pub(crate) event: IndexShareTraceEvent,
    pub(crate) layer: i32,
    pub(crate) role: Option<String>,
    pub(crate) input_top_k: Option<bool>,
    pub(crate) stage_filtered: Option<bool>,
    pub(crate) layer_start: Option<i32>,
    pub(crate) layer_end: Option<i32>,
    pub(crate) source: Option<String>,
    pub(crate) width: Option<i64>,
    pub(crate) batch: Option<i64>,
    pub(crate) stream: Option<i64>,
    pub(crate) visible_kv: Option<i64>,
    pub(crate) score_width: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct IndexShareContractRecord {
    pub(crate) source: String,
    pub(crate) full_layers: usize,
    pub(crate) shared_layers: usize,
    pub(crate) indexer_tensor_layers: usize,
    pub(crate) target_indexer_tensor_layers: Option<usize>,
    pub(crate) filtered_indexer_groups: Option<usize>,
    pub(crate) out_of_stage_indexer_groups: Option<usize>,
    pub(crate) stage_filtered: bool,
    pub(crate) layer_start: i32,
    pub(crate) layer_end: i32,
    pub(crate) top_k: usize,
    pub(crate) top_k_frequency: Option<usize>,
    pub(crate) skip_top_k_offset: Option<usize>,
    pub(crate) nextn_layers: Option<usize>,
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
        let indexshare_records = parse_indexshare_trace_records(&text)
            .with_context(|| format!("parse GLM-DSA IndexShare trace in {}", path.display()))?;
        let indexshare_contract_records =
            parse_indexshare_contract_records(&text).with_context(|| {
                format!(
                    "parse GLM-DSA IndexShare contract records in {}",
                    path.display()
                )
            })?;
        if records.is_empty()
            && indexshare_records.is_empty()
            && indexshare_contract_records.is_empty()
        {
            bail!(
                "{} contains no GLM-DSA op timing or IndexShare trace records",
                path.display()
            );
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
        let summary = summarize_log(
            path.clone(),
            &records,
            &group_records,
            &sideband_records,
            &indexshare_records,
            &indexshare_contract_records,
        );
        if args.require_indexshare_producer_consumer {
            require_indexshare_producer_consumer_trace(path, &summary)?;
        }
        logs.push(summary);
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

fn require_indexshare_producer_consumer_trace(path: &Path, summary: &LogSummary) -> Result<()> {
    let trace = &summary.indexshare_trace;
    let mut missing = Vec::new();
    if trace.contract_records == 0 {
        missing.push("indexshare_contract");
    }
    if trace.contract_sources.is_empty() {
        missing.push("indexshare_contract_source");
    }
    if trace.contract_full_layers.unwrap_or(0) == 0 {
        missing.push("contract_full_layers");
    }
    if trace.contract_shared_layers.unwrap_or(0) == 0 {
        missing.push("contract_shared_layers");
    }
    if trace.records == 0 {
        missing.push("indexshare_trace_records");
    }
    if trace.full_exec_records == 0 {
        missing.push("full_exec");
    }
    if trace.top_k_from_indexer == 0 {
        missing.push("top_k_from_indexer");
    }
    if trace.shared_exec_records == 0 {
        missing.push("shared_exec");
    }
    if trace.shared_exec_with_input_top_k != trace.shared_exec_records {
        missing.push("shared_exec_with_input_top_k");
    }
    if trace.shared_exec_missing_input_top_k != 0 {
        missing.push("no_shared_exec_missing_input_top_k");
    }
    if trace.consume_records == 0 {
        missing.push("consume");
    }
    if !matches!(trace.min_consume_width, Some(width) if width > 0) {
        missing.push("positive_consume_width");
    }
    if missing.is_empty() {
        return Ok(());
    }

    bail!(
        "{} does not prove native GLM-DSA IndexShare producer/consumer flow; missing {}; trace={:?}",
        path.display(),
        missing.join(","),
        trace
    )
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

pub(crate) fn parse_hot_tensor_records(text: &str) -> Result<Vec<HotTensorRecord>> {
    let mut records = Vec::new();
    let mut timing_record_count = 0usize;
    for line in text.lines() {
        if line.contains(OP_TIMING_PREFIX) {
            timing_record_count += 1;
            continue;
        }
        let Some(index) = line.find(HOT_TENSOR_PREFIX) else {
            continue;
        };
        let record_index = timing_record_count.saturating_sub(1);
        records.push(parse_hot_tensor_record(
            record_index,
            &line[index + HOT_TENSOR_PREFIX.len()..],
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

pub(crate) fn parse_compact_flash_policy_records(
    text: &str,
) -> Result<Vec<CompactFlashPolicyRecord>> {
    text.lines()
        .filter_map(|line| {
            line.find(COMPACT_FLASH_POLICY_PREFIX)
                .map(|index| &line[index + COMPACT_FLASH_POLICY_PREFIX.len()..])
        })
        .map(|line| {
            parse_compact_flash_policy_record(line)
                .with_context(|| format!("parse compact flash policy line: {line}"))
        })
        .collect()
}

pub(crate) fn parse_compact_flash_mask_records(text: &str) -> Result<Vec<CompactFlashMaskRecord>> {
    text.lines()
        .filter_map(|line| {
            line.find(COMPACT_FLASH_MASK_PREFIX)
                .map(|index| &line[index + COMPACT_FLASH_MASK_PREFIX.len()..])
        })
        .map(parse_compact_flash_mask_record)
        .collect()
}

pub(crate) fn parse_metal_dispatch_records(text: &str) -> Result<Vec<MetalDispatchRecord>> {
    text.lines()
        .filter_map(|line| {
            line.find(METAL_DISPATCH_PREFIX)
                .map(|index| &line[index + METAL_DISPATCH_PREFIX.len()..])
        })
        .map(parse_metal_dispatch_record)
        .collect()
}

pub(crate) fn parse_compute_buffer_records(text: &str) -> Result<Vec<ComputeBufferRecord>> {
    text.lines()
        .filter_map(|line| {
            line.find(COMPUTE_BUFFER_PREFIX)
                .map(|index| &line[index + COMPUTE_BUFFER_PREFIX.len()..])
        })
        .filter(|line| line.contains("compute buffer size"))
        .map(parse_compute_buffer_record)
        .collect()
}

pub(crate) fn parse_indexshare_trace_records(text: &str) -> Result<Vec<IndexShareTraceRecord>> {
    text.lines()
        .filter_map(|line| {
            line.find(INDEXSHARE_TRACE_MARKER)
                .map(|index| &line[index + INDEXSHARE_TRACE_MARKER.len()..])
        })
        .filter(|line| {
            line.starts_with("exec ") || line.starts_with("top_k ") || line.starts_with("consume ")
        })
        .map(parse_indexshare_trace_record)
        .collect()
}

pub(crate) fn parse_indexshare_contract_records(
    text: &str,
) -> Result<Vec<IndexShareContractRecord>> {
    text.lines()
        .filter_map(|line| {
            line.find(INDEXSHARE_TRACE_MARKER)
                .map(|index| &line[index + INDEXSHARE_TRACE_MARKER.len()..])
        })
        .filter(|line| line.starts_with("source="))
        .map(parse_indexshare_contract_record)
        .collect()
}

fn parse_indexshare_contract_record(line: &str) -> Result<IndexShareContractRecord> {
    let fields = line
        .split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect::<BTreeMap<_, _>>();
    Ok(IndexShareContractRecord {
        source: parse_string_field(&fields, "source")?,
        full_layers: parse_field(&fields, "full_layers")?,
        shared_layers: parse_field(&fields, "shared_layers")?,
        indexer_tensor_layers: parse_field(&fields, "indexer_tensor_layers")?,
        target_indexer_tensor_layers: parse_optional_field(
            &fields,
            "target_indexer_tensor_layers",
        )?,
        filtered_indexer_groups: parse_optional_field(&fields, "filtered_indexer_groups")?,
        out_of_stage_indexer_groups: parse_optional_field(&fields, "out_of_stage_indexer_groups")?,
        stage_filtered: parse_bool_int_field(&fields, "stage_filtered")?,
        layer_start: parse_field(&fields, "layer_start")?,
        layer_end: parse_field(&fields, "layer_end")?,
        top_k: parse_field(&fields, "top_k")?,
        top_k_frequency: parse_optional_field(&fields, "top_k_frequency")?,
        skip_top_k_offset: parse_optional_field(&fields, "skip_top_k_offset")?,
        nextn_layers: parse_optional_field(&fields, "nextn_layers")?,
    })
}

fn parse_indexshare_trace_record(line: &str) -> Result<IndexShareTraceRecord> {
    let mut parts = line.split_whitespace();
    let event = match parts.next().context("missing IndexShare trace event")? {
        "exec" => IndexShareTraceEvent::Exec,
        "top_k" => IndexShareTraceEvent::TopK,
        "consume" => IndexShareTraceEvent::Consume,
        value => bail!("unknown IndexShare trace event {value}"),
    };
    let fields = parts
        .filter_map(|field| field.split_once('='))
        .collect::<BTreeMap<_, _>>();
    Ok(IndexShareTraceRecord {
        event,
        layer: parse_field(&fields, "layer")?,
        role: parse_optional_string_field(&fields, "role"),
        input_top_k: parse_optional_bool_int_field(&fields, "input_top_k")?,
        stage_filtered: parse_optional_bool_int_field(&fields, "stage_filtered")?,
        layer_start: parse_optional_field(&fields, "layer_start")?,
        layer_end: parse_optional_field(&fields, "layer_end")?,
        source: parse_optional_string_field(&fields, "source"),
        width: parse_optional_field(&fields, "width")?,
        batch: parse_optional_field(&fields, "batch")?,
        stream: parse_optional_field(&fields, "stream")?,
        visible_kv: parse_optional_field(&fields, "visible_kv")?,
        score_width: parse_optional_field(&fields, "score_width")?,
    })
}

fn parse_compute_buffer_record(line: &str) -> Result<ComputeBufferRecord> {
    let line = line.trim();
    let (device, rest) = line
        .split_once(" compute buffer size ")
        .context("compute buffer record missing device or size marker")?;
    let rest = rest
        .strip_prefix("is ")
        .or_else(|| rest.strip_prefix("of "))
        .context("compute buffer record missing size verb")?;
    let (size_text, rest) = rest
        .split_once(" MiB, ")
        .context("compute buffer record missing MiB size")?;
    let (_, expected) = rest
        .split_once("expectation of ")
        .context("compute buffer record missing expected size")?;
    Ok(ComputeBufferRecord {
        device: device.to_string(),
        size_mib: parse_first_f64(size_text)
            .map_err(|error| anyhow::anyhow!("invalid compute buffer size: {error}"))?,
        expected_mib: parse_first_f64(expected)
            .map_err(|error| anyhow::anyhow!("invalid expected compute buffer size: {error}"))?,
        matches_expectation: rest.contains("matches expectation"),
    })
}

fn parse_first_f64(text: &str) -> Result<f64> {
    let token = text
        .split_whitespace()
        .next()
        .context("missing float token")?;
    token.parse().context("parse float token")
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
        sparse_kv: parse_optional_field(&fields, "sparse_kv")?,
        sparse_top_k: parse_optional_field(&fields, "sparse_top_k")?,
        min_kv_topk_ratio: parse_optional_field(&fields, "min_kv_topk_ratio")?,
        kv_topk_ratio: parse_optional_field(&fields, "kv_topk_ratio")?,
        dense_mask_bytes: parse_optional_field(&fields, "dense_mask_bytes")?,
        dense_mask_limit: parse_optional_field(&fields, "dense_mask_limit")?,
        phase: fields.get("phase").map(|value| (*value).to_string()),
        selector_reason: fields
            .get("selector_reason")
            .map(|value| (*value).to_string()),
        direct_enabled: parse_bool_int_field(&fields, "direct_enabled")?,
        prefill_enabled: parse_bool_int_field(&fields, "prefill_enabled")?,
        decode_shape: parse_bool_int_field(&fields, "decode_shape")?,
        prefill_shape: parse_bool_int_field(&fields, "prefill_shape")?,
        large_prefill_shape: parse_optional_bool_int_field(&fields, "large_prefill_shape")?,
        token_shape_allowed: parse_bool_int_field(&fields, "token_shape_allowed")?,
        kq_b_ok: parse_bool_int_field(&fields, "kq_b_ok")?,
        sinks_ok: parse_bool_int_field(&fields, "sinks_ok")?,
        alibi_ok: parse_bool_int_field(&fields, "alibi_ok")?,
        soft_cap_ok: parse_bool_int_field(&fields, "soft_cap_ok")?,
        use_direct: parse_bool_int_field(&fields, "use_direct")?,
    })
}

fn parse_compact_flash_policy_record(line: &str) -> Result<CompactFlashPolicyRecord> {
    let fields = line
        .split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect::<BTreeMap<_, _>>();
    Ok(CompactFlashPolicyRecord {
        layer: parse_field(&fields, "layer")?,
        ubatch_tokens: parse_field(&fields, "ubatch_tokens")?,
        visible_kv: parse_field(&fields, "visible_kv")?,
        top_k: parse_field(&fields, "top_k")?,
        kv_topk_ratio: parse_field(&fields, "kv_topk_ratio")?,
        min_kv_topk_ratio: parse_optional_field(&fields, "min_kv_topk_ratio")?,
        forced: parse_bool_int_field(&fields, "forced")?,
        disabled: parse_bool_int_field(&fields, "disabled")?,
        ratio_ok: parse_optional_bool_int_field(&fields, "ratio_ok")?,
        enabled: parse_bool_int_field(&fields, "enabled")?,
        flash_attn: parse_bool_int_field(&fields, "flash_attn")?,
        phase: fields.get("phase").map(|value| (*value).to_string()),
        decode_shape: parse_bool_int_field(&fields, "decode_shape")?,
        kq_b_ok: parse_bool_int_field(&fields, "kq_b_ok")?,
        sinks_ok: parse_bool_int_field(&fields, "sinks_ok")?,
        alibi_ok: parse_bool_int_field(&fields, "alibi_ok")?,
        soft_cap_ok: parse_bool_int_field(&fields, "soft_cap_ok")?,
        no_mask: parse_optional_bool_int_field(&fields, "no_mask")?,
        use_compact: parse_bool_int_field(&fields, "use_compact")?,
        selector_reason: fields
            .get("selector_reason")
            .map(|value| (*value).to_string()),
    })
}

fn parse_compact_flash_mask_record(line: &str) -> Result<CompactFlashMaskRecord> {
    let fields = line
        .split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect::<BTreeMap<_, _>>();
    Ok(CompactFlashMaskRecord {
        layer: parse_field(&fields, "layer")?,
        omitted_mla_kq_mask: parse_bool_int_field(&fields, "omitted_mla_kq_mask")?,
        visible_kv: parse_field(&fields, "visible_kv")?,
        ubatch_tokens: parse_field(&fields, "ubatch_tokens")?,
        streams: parse_field(&fields, "streams")?,
        max_top_k: parse_field(&fields, "max_top_k")?,
    })
}

fn parse_metal_dispatch_record(line: &str) -> Result<MetalDispatchRecord> {
    let fields = line
        .split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect::<BTreeMap<_, _>>();
    Ok(MetalDispatchRecord {
        op: parse_string_field(&fields, "op")?,
        kernel: parse_optional_string_field(&fields, "kernel"),
        tensor: parse_string_field(&fields, "tensor")?,
        next: parse_optional_string_field(&fields, "next")
            .or_else(|| parse_optional_string_field(&fields, "next_tensor")),
        next_op: parse_optional_string_field(&fields, "next_op"),
        shared_gate: parse_optional_string_field(&fields, "shared_gate"),
        shared_up: parse_optional_string_field(&fields, "shared_up"),
        weighted_sum: parse_optional_string_field(&fields, "weighted_sum"),
        weighted_sum_op: parse_optional_string_field(&fields, "weighted_sum_op"),
        reason: parse_optional_string_field(&fields, "reason"),
        shared_branch: parse_optional_bool_int_field(&fields, "shared_branch")?,
        weighted_sum_uses_down: parse_optional_bool_int_field(&fields, "weighted_sum_uses_down")?,
        natural_order: parse_optional_bool_int_field(&fields, "natural_order")?,
        backend_candidate: parse_optional_bool_int_field(&fields, "backend_candidate")?,
        pair_fusable: parse_optional_bool_int_field(&fields, "pair_fusable")?,
        subgraph_fusable: parse_optional_bool_int_field(&fields, "subgraph_fusable")?,
        motif_nodes: parse_optional_field(&fields, "motif_nodes")?,
        fusion_outputs: parse_optional_field(&fields, "fusion_outputs")?,
        filtered_gap: parse_optional_field(&fields, "filtered_gap")?,
        graph_gap: parse_optional_field(&fields, "graph_gap")?,
        weighted_sum_gap: parse_optional_field(&fields, "weighted_sum_gap")?,
        weighted_sum_graph_gap: parse_optional_field(&fields, "weighted_sum_graph_gap")?,
        parallel: parse_optional_bool_int_field(&fields, "parallel")?,
        generic: parse_optional_bool_int_field(&fields, "generic")?,
        view: parse_optional_bool_int_field(&fields, "view")?,
        get_rows_uses: parse_optional_field(&fields, "get_rows_uses")?,
        use_count: parse_optional_field(&fields, "use_count")?,
        consumer_count: parse_optional_field(&fields, "consumer_count")?,
        consumer_graph_idx: parse_optional_field(&fields, "consumer_graph_idx")?,
        consumer_op: parse_optional_string_field(&fields, "consumer_op"),
        consumer_tensor: parse_optional_string_field(&fields, "consumer_tensor"),
        consumer_src_slot: parse_optional_field(&fields, "consumer_src_slot")?,
        flash_graph_idx: parse_optional_field(&fields, "flash_graph_idx")?,
        q_type: parse_optional_string_field(&fields, "q_type"),
        k_type: parse_optional_string_field(&fields, "k_type"),
        v_type: parse_optional_string_field(&fields, "v_type"),
        mask_type: parse_optional_string_field(&fields, "mask_type"),
        top_k_type: parse_optional_string_field(&fields, "top_k_type"),
        src_type: parse_optional_string_field(&fields, "src_type")
            .or_else(|| parse_optional_string_field(&fields, "src0_type")),
        dst_type: parse_optional_string_field(&fields, "dst_type"),
        q_width: parse_optional_field(&fields, "q_width")?,
        v_width: parse_optional_field(&fields, "v_width")?,
        batch: parse_optional_field(&fields, "batch")?,
        heads: parse_optional_field(&fields, "heads")?,
        stream: parse_optional_field(&fields, "stream")?,
        kv: parse_optional_field(&fields, "kv")?,
        top_k: parse_optional_field(&fields, "top_k")?,
        top_stream: parse_optional_field(&fields, "top_stream")?,
        selected_keys: parse_optional_field(&fields, "selected_keys")?,
        q_read_bytes: parse_optional_field(&fields, "q_read_bytes")?,
        k_read_bytes: parse_optional_field(&fields, "k_read_bytes")?,
        v_read_bytes: parse_optional_field(&fields, "v_read_bytes")?,
        mask_read_bytes: parse_optional_field(&fields, "mask_read_bytes")?,
        top_k_read_bytes: parse_optional_field(&fields, "top_k_read_bytes")?,
        scratch_per_tg_bytes: parse_optional_field(&fields, "scratch_per_tg_bytes")?,
        score_fma: parse_optional_field(&fields, "score_fma")?,
        value_fma: parse_optional_field(&fields, "value_fma")?,
        reduction_strategy: parse_optional_string_field(&fields, "reduction_strategy"),
        rows: parse_optional_field(&fields, "rows")?,
        partial_bytes: parse_optional_field(&fields, "partial_bytes")?,
        softmax_bytes: parse_optional_field(&fields, "softmax_bytes")?,
        tmp_bytes: parse_optional_field(&fields, "tmp_bytes")?,
        nwg: parse_optional_field(&fields, "nwg")?,
        tmp_f16: parse_optional_bool_int_field(&fields, "tmp_f16")?,
        dst_partial: parse_optional_bool_int_field(&fields, "dst_partial")?,
        grid_x: parse_field(&fields, "grid_x")?,
        grid_y: parse_field(&fields, "grid_y")?,
        grid_z: parse_field(&fields, "grid_z")?,
        threads_x: parse_field(&fields, "threads_x")?,
        threads_y: parse_optional_field(&fields, "threads_y")?,
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

fn parse_hot_tensor_record(record_index: usize, line: &str) -> Result<HotTensorRecord> {
    let fields = line
        .split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect::<BTreeMap<_, _>>();
    Ok(HotTensorRecord {
        record_index,
        stage: parse_field(&fields, "stage")?,
        tokens: parse_field(&fields, "tokens")?,
        rank: parse_field(&fields, "rank")?,
        op: parse_string_field(&fields, "op")?,
        kind: parse_string_field(&fields, "kind")?,
        elapsed_us: parse_field(&fields, "elapsed_us")?,
        name: parse_string_field(&fields, "name")?,
        ne0: parse_field(&fields, "ne0")?,
        ne1: parse_field(&fields, "ne1")?,
        ne2: parse_field(&fields, "ne2")?,
        ne3: parse_field(&fields, "ne3")?,
    })
}

fn parse_timing_fields(fields: &BTreeMap<&str, &str>) -> Result<TimingRecord> {
    let indexer = parse_optional_bucket(fields, "indexer")?;
    let top_k = parse_optional_bucket(fields, "top_k")?;
    let sparse_mask_fill = parse_optional_bucket(fields, "sparse_mask_fill")?;
    let sparse_mask_topk = parse_optional_bucket(fields, "sparse_mask_topk")?;
    let sparse_mask_add = parse_optional_bucket(fields, "sparse_mask_add")?;
    let dsa_sparse_attn = parse_optional_bucket(fields, "dsa_sparse_attn")?;
    let compact_get_rows = parse_optional_bucket(fields, "compact_get_rows")?;
    let routed_moe_route = parse_optional_bucket(fields, "routed_moe_route")?;
    let routed_moe_gate_up = parse_optional_bucket(fields, "routed_moe_gate_up")?;
    let routed_moe_gate = parse_optional_bucket(fields, "routed_moe_gate")?;
    let routed_moe_up = parse_optional_bucket(fields, "routed_moe_up")?;
    let routed_moe_act = parse_optional_bucket(fields, "routed_moe_act")?;
    let routed_moe_down = parse_optional_bucket(fields, "routed_moe_down")?;
    let routed_moe_weighted = parse_optional_bucket(fields, "routed_moe_weighted")?;
    let routed_moe_aggregate = parse_optional_bucket(fields, "routed_moe_aggregate")?;
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
        compact_get_rows_nodes: compact_get_rows.nodes,
        compact_get_rows_us: compact_get_rows.elapsed_us,
        mla_attention_nodes: parse_field(fields, "mla_attention_nodes")?,
        mla_attention_us: parse_field(fields, "mla_attention_us")?,
        routed_moe_nodes: parse_field(fields, "routed_moe_nodes")?,
        routed_moe_us: parse_field(fields, "routed_moe_us")?,
        routed_moe_route_nodes: routed_moe_route.nodes,
        routed_moe_route_us: routed_moe_route.elapsed_us,
        routed_moe_gate_up_nodes: routed_moe_gate_up.nodes,
        routed_moe_gate_up_us: routed_moe_gate_up.elapsed_us,
        routed_moe_gate_nodes: routed_moe_gate.nodes,
        routed_moe_gate_us: routed_moe_gate.elapsed_us,
        routed_moe_up_nodes: routed_moe_up.nodes,
        routed_moe_up_us: routed_moe_up.elapsed_us,
        routed_moe_act_nodes: routed_moe_act.nodes,
        routed_moe_act_us: routed_moe_act.elapsed_us,
        routed_moe_down_nodes: routed_moe_down.nodes,
        routed_moe_down_us: routed_moe_down.elapsed_us,
        routed_moe_weighted_nodes: routed_moe_weighted.nodes,
        routed_moe_weighted_us: routed_moe_weighted.elapsed_us,
        routed_moe_aggregate_nodes: routed_moe_aggregate.nodes,
        routed_moe_aggregate_us: routed_moe_aggregate.elapsed_us,
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
        .filter_map(sideband_line)
        .map(|(direction, line)| parse_sideband_record(direction, line))
        .collect()
}

fn sideband_line(line: &str) -> Option<(SidebandDirection, &str)> {
    if let Some(index) = line.find(SIDEBAND_FORWARD_PREFIX) {
        return Some((
            SidebandDirection::Forward,
            &line[index + SIDEBAND_FORWARD_PREFIX.len()..],
        ));
    }
    line.find(SIDEBAND_RECEIVE_PREFIX).map(|index| {
        (
            SidebandDirection::Receive,
            &line[index + SIDEBAND_RECEIVE_PREFIX.len()..],
        )
    })
}

fn parse_sideband_record(direction: SidebandDirection, line: &str) -> Result<SidebandRecord> {
    let fields = line
        .split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect::<BTreeMap<_, _>>();
    Ok(SidebandRecord {
        direction,
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

fn parse_optional_bool_int_field(
    fields: &BTreeMap<&str, &str>,
    name: &str,
) -> Result<Option<bool>> {
    parse_optional_field::<u8>(fields, name).map(|value| value.map(|value| value != 0))
}

fn parse_string_field(fields: &BTreeMap<&str, &str>, name: &str) -> Result<String> {
    Ok(fields
        .get(name)
        .with_context(|| format!("missing {name}"))?
        .to_string())
}

fn parse_optional_string_field(fields: &BTreeMap<&str, &str>, name: &str) -> Option<String> {
    fields.get(name).map(ToString::to_string)
}

fn summarize_log(
    path: PathBuf,
    records: &[TimingRecord],
    group_records: &[TimingGroupRecord],
    sideband_records: &[SidebandRecord],
    indexshare_records: &[IndexShareTraceRecord],
    indexshare_contract_records: &[IndexShareContractRecord],
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
    let indexshare_trace =
        summarize_indexshare_trace_records(indexshare_records, indexshare_contract_records);
    LogSummary {
        path,
        records: records.len(),
        stage_records,
        group_records: grouped_records,
        hottest_group_records,
        sideband_records,
        indexshare_trace,
    }
}

pub(crate) fn summarize_indexshare_trace_records(
    records: &[IndexShareTraceRecord],
    contract_records: &[IndexShareContractRecord],
) -> IndexShareTraceSummary {
    let mut summary = IndexShareTraceSummary {
        records: records.len(),
        contract_records: contract_records.len(),
        ..IndexShareTraceSummary::default()
    };
    for record in contract_records {
        push_unique_string(&mut summary.contract_sources, &record.source);
        summary.contract_full_layers = Some(record.full_layers);
        summary.contract_shared_layers = Some(record.shared_layers);
        summary.contract_indexer_tensor_layers = Some(record.indexer_tensor_layers);
        if let Some(value) = record.target_indexer_tensor_layers {
            summary.contract_target_indexer_tensor_layers = Some(value);
        }
        if let Some(value) = record.nextn_layers {
            summary.contract_nextn_layers = Some(value);
        }
    }
    for record in records {
        match record.event {
            IndexShareTraceEvent::Exec => {
                summary.exec_records += 1;
                match record.role.as_deref() {
                    Some("full") => {
                        summary.full_exec_records += 1;
                        push_unique_sorted(&mut summary.full_layers, record.layer);
                    }
                    Some("shared") => {
                        summary.shared_exec_records += 1;
                        push_unique_sorted(&mut summary.shared_layers, record.layer);
                        if record.input_top_k == Some(true) {
                            summary.shared_exec_with_input_top_k += 1;
                        } else {
                            summary.shared_exec_missing_input_top_k += 1;
                        }
                    }
                    _ => {}
                }
            }
            IndexShareTraceEvent::TopK => {
                summary.top_k_records += 1;
                match record.source.as_deref() {
                    Some("indexer") => summary.top_k_from_indexer += 1,
                    Some("full_visible") => summary.top_k_from_full_visible += 1,
                    _ => {}
                }
            }
            IndexShareTraceEvent::Consume => {
                summary.consume_records += 1;
                if let Some(width) = record.width {
                    summary.min_consume_width = Some(
                        summary
                            .min_consume_width
                            .map_or(width, |current| current.min(width)),
                    );
                    summary.max_consume_width = Some(
                        summary
                            .max_consume_width
                            .map_or(width, |current| current.max(width)),
                    );
                }
            }
        }
    }
    summary
}

fn push_unique_sorted(values: &mut Vec<i32>, value: i32) {
    match values.binary_search(&value) {
        Ok(_) => {}
        Err(index) => values.insert(index, value),
    }
}

fn push_unique_string(values: &mut Vec<String>, value: &str) {
    match values.binary_search_by(|current| current.as_str().cmp(value)) {
        Ok(_) => {}
        Err(index) => values.insert(index, value.to_string()),
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
    add_optional_bucket(
        &mut summary.compact_get_rows,
        record.compact_get_rows_nodes,
        record.compact_get_rows_us,
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
        match record.direction {
            SidebandDirection::Forward => summary.forward_records += 1,
            SidebandDirection::Receive => summary.receive_records += 1,
        }
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
    use std::path::Path;

    use super::{
        ComparisonKey, OpBucket, Phase, PhaseSummary, SidebandDirection, compare_phase,
        parse_compact_flash_mask_records, parse_compact_flash_policy_records,
        parse_compute_buffer_records, parse_direct_sparse_decision_records,
        parse_indexshare_contract_records, parse_indexshare_trace_records,
        parse_metal_dispatch_records, parse_sideband_record, parse_sideband_records,
        parse_timing_group_records, parse_timing_record, parse_timing_records,
        require_indexshare_producer_consumer_trace, summarize_comparison_rows, summarize_log,
    };

    const LINE: &str = "skippy: glm_dsa_op_timing stage=1 tokens=128 total_us=1475800 indexer_topk_nodes=275 indexer_topk_us=129065 sparse_mask_nodes=235 sparse_mask_us=114543 mla_attention_nodes=47 mla_attention_us=35234 routed_moe_nodes=47 routed_moe_us=379574 shared_expert_nodes=47 shared_expert_us=817384";
    const LINE_WITH_INDEXER_BREAKDOWN: &str = "skippy: glm_dsa_op_timing stage=1 tokens=128 total_us=1475800 indexer_topk_nodes=275 indexer_topk_us=129065 indexer_nodes=235 indexer_us=80000 top_k_nodes=40 top_k_us=49065 sparse_mask_nodes=235 sparse_mask_us=114543 mla_attention_nodes=47 mla_attention_us=35234 routed_moe_nodes=47 routed_moe_us=379574 shared_expert_nodes=47 shared_expert_us=817384";
    const LINE_WITH_SPARSE_BREAKDOWN: &str = "skippy: glm_dsa_op_timing stage=1 tokens=128 total_us=1475800 indexer_topk_nodes=275 indexer_topk_us=129065 sparse_mask_nodes=235 sparse_mask_us=114543 sparse_mask_fill_nodes=47 sparse_mask_fill_us=1000 sparse_mask_topk_nodes=47 sparse_mask_topk_us=2000 sparse_mask_add_nodes=47 sparse_mask_add_us=3000 mla_attention_nodes=47 mla_attention_us=35234 routed_moe_nodes=47 routed_moe_us=379574 shared_expert_nodes=47 shared_expert_us=817384";
    const LINE_WITH_DSA_SPARSE_ATTN: &str = "skippy: glm_dsa_op_timing stage=1 tokens=128 total_us=1475800 indexer_topk_nodes=275 indexer_topk_us=129065 sparse_mask_nodes=0 sparse_mask_us=0 dsa_sparse_attn_nodes=47 dsa_sparse_attn_us=114543 mla_attention_nodes=47 mla_attention_us=35234 routed_moe_nodes=47 routed_moe_us=379574 shared_expert_nodes=47 shared_expert_us=817384";
    const LINE_WITH_COMPACT_GET_ROWS: &str = "skippy: glm_dsa_op_timing stage=1 tokens=1 total_us=24000 indexer_topk_nodes=0 indexer_topk_us=0 sparse_mask_nodes=0 sparse_mask_us=0 dsa_sparse_attn_nodes=0 dsa_sparse_attn_us=0 compact_get_rows_nodes=6 compact_get_rows_us=900 mla_attention_nodes=3 mla_attention_us=2100 routed_moe_nodes=54 routed_moe_us=18000 shared_expert_nodes=12 shared_expert_us=3000";
    const GROUP_LINE_LAYER_0: &str = "skippy: glm_dsa_group_timing stage=1 tokens=128 group=layer_0 total_us=600000 indexer_topk_nodes=100 indexer_topk_us=50000 indexer_nodes=80 indexer_us=30000 top_k_nodes=20 top_k_us=20000 sparse_mask_nodes=40 sparse_mask_us=1000 sparse_mask_fill_nodes=0 sparse_mask_fill_us=0 sparse_mask_topk_nodes=40 sparse_mask_topk_us=1000 sparse_mask_add_nodes=0 sparse_mask_add_us=0 dsa_sparse_attn_nodes=0 dsa_sparse_attn_us=0 mla_attention_nodes=1 mla_attention_us=9000 routed_moe_nodes=0 routed_moe_us=0 shared_expert_nodes=0 shared_expert_us=0";
    const GROUP_LINE_LAYER_1: &str = "skippy: glm_dsa_group_timing stage=1 tokens=128 group=layer_1 total_us=875800 indexer_topk_nodes=175 indexer_topk_us=79065 indexer_nodes=155 indexer_us=50000 top_k_nodes=20 top_k_us=29065 sparse_mask_nodes=195 sparse_mask_us=113543 sparse_mask_fill_nodes=47 sparse_mask_fill_us=1000 sparse_mask_topk_nodes=47 sparse_mask_topk_us=2000 sparse_mask_add_nodes=47 sparse_mask_add_us=3000 dsa_sparse_attn_nodes=0 dsa_sparse_attn_us=0 mla_attention_nodes=46 mla_attention_us=26234 routed_moe_nodes=47 routed_moe_us=379574 shared_expert_nodes=47 shared_expert_us=817384";
    const SIDEBAND_LINE: &str = "skippy: glm_dsa_top_k_sideband_forward stage=stage-0 request=1 session=2 kind=DecodeEmbd pos_start=718 tokens=1 hidden_bytes=24576 sideband_bytes=3072 sideband_i32=768";
    const SIDEBAND_RECEIVE_LINE: &str = "skippy: glm_dsa_top_k_sideband_receive stage=stage-1 request=1 session=2 kind=DecodeEmbd pos_start=718 tokens=1 hidden_bytes=24576 sideband_bytes=3072 sideband_i32=768";
    const PADDED_PREFILL_SIDEBAND_LINE: &str = "skippy: glm_dsa_top_k_sideband_forward stage=stage-0 request=1 session=2 kind=PrefillEmbd pos_start=512 tokens=128 hidden_bytes=3145728 sideband_bytes=393216 sideband_i32=98304";
    const DIRECT_SPARSE_DECISION_LINE: &str = "skippy: glm_dsa_direct_sparse_decision layer=30 ubatch_tokens=33 sparse_batch=33 sparse_streams=1 prefill_cap=32 dense_mask_bytes=270336 dense_mask_limit=536870912 direct_enabled=1 prefill_enabled=1 decode_shape=0 prefill_shape=0 large_prefill_shape=0 token_shape_allowed=0 kq_b_ok=1 sinks_ok=1 alibi_ok=1 soft_cap_ok=1 use_direct=0";
    const DIRECT_SPARSE_DECISION_LINE_WITH_REASON: &str = "skippy: glm_dsa_direct_sparse_decision layer=30 ubatch_tokens=1024 sparse_batch=1024 sparse_streams=1 prefill_cap=32 sparse_kv=99328 sparse_top_k=1024 min_kv_topk_ratio=32 kv_topk_ratio=97 dense_mask_bytes=203423744 dense_mask_limit=268435456 phase=prefill selector_reason=dense_mask_guard_large_prefill direct_enabled=1 prefill_enabled=1 decode_shape=0 prefill_shape=0 large_prefill_shape=1 token_shape_allowed=1 kq_b_ok=1 sinks_ok=1 alibi_ok=1 soft_cap_ok=1 use_direct=1";
    const LARGE_PREFILL_DIRECT_SPARSE_DECISION_LINE: &str = "skippy: glm_dsa_direct_sparse_decision layer=30 ubatch_tokens=4096 sparse_batch=4096 sparse_streams=1 prefill_cap=32 dense_mask_bytes=2147483648 dense_mask_limit=536870912 direct_enabled=1 prefill_enabled=1 decode_shape=0 prefill_shape=0 large_prefill_shape=1 token_shape_allowed=1 kq_b_ok=1 sinks_ok=1 alibi_ok=1 soft_cap_ok=1 use_direct=1";
    const COMPACT_FLASH_POLICY_LINE: &str = "skippy: glm_dsa_compact_flash_policy layer=30 ubatch_tokens=1 visible_kv=8192 top_k=2048 kv_topk_ratio=4 min_kv_topk_ratio=2 forced=0 disabled=0 ratio_ok=1 enabled=1 flash_attn=1 phase=decode decode_shape=1 kq_b_ok=1 sinks_ok=1 alibi_ok=1 soft_cap_ok=1 no_mask=1 use_compact=1 selector_reason=decode_compact";
    const COMPACT_FLASH_POLICY_LINE_WITHOUT_MIN_RATIO: &str = "skippy: glm_dsa_compact_flash_policy layer=30 ubatch_tokens=1 visible_kv=8192 top_k=2048 kv_topk_ratio=4 forced=0 disabled=0 ratio_ok=1 enabled=1 flash_attn=1 phase=decode decode_shape=1 kq_b_ok=1 sinks_ok=1 alibi_ok=1 soft_cap_ok=1 no_mask=1 use_compact=1 selector_reason=decode_compact";
    const COMPACT_FLASH_MASK_LINE: &str = "skippy: glm_dsa_compact_flash_mask layer=30 omitted_mla_kq_mask=1 visible_kv=8192 ubatch_tokens=1 streams=1 max_top_k=2048";
    const METAL_DISPATCH_LINE: &str = "skippy: glm_dsa_metal_dispatch op=dsa_sparse_attn kernel=decode_vec tensor=blk.30.dsa_sparse_attn q_type=f32 k_type=f16 v_type=f16 mask_type=f32 top_k_type=i32 dst_type=f32 q_width=576 v_width=512 batch=32 heads=4 stream=1 kv=32 top_k=1024 top_stream=1 selected_keys=1048576 q_read_bytes=2415919104 k_read_bytes=1207959552 v_read_bytes=1073741824 mask_read_bytes=4194304 top_k_read_bytes=4194304 scratch_per_tg_bytes=1024 score_fma=603979776 value_fma=536870912 reduction_strategy=threadgroup_direct grid_x=32 grid_y=4 grid_z=8 threads_x=256 nwg=8 tmp_f16=1 dst_partial=1";
    const METAL_DECODE_VEC_REDUCE_DISPATCH_LINE: &str = "skippy: glm_dsa_metal_dispatch op=dsa_sparse_attn kernel=decode_vec_reduce tensor=blk.30.dsa_sparse_attn q_type=f32 k_type=f16 v_type=f16 mask_type=f32 top_k_type=i32 dst_type=f32 q_width=576 v_width=512 batch=4096 heads=64 stream=1 kv=4096 top_k=2048 top_stream=1 rows=262144 partial_bytes=536870912 softmax_bytes=4194304 tmp_bytes=541065216 grid_x=262144 grid_y=1 grid_z=1 threads_x=32 threads_y=2 nwg=2 tmp_f16=1 dst_partial=1";
    const METAL_MUL_MAT_ID_DISPATCH_LINE: &str = "skippy: glm_dsa_metal_dispatch op=mul_mat_id kernel=mul_mv_id tensor=ffn_moe_down-45 src0_type=q3_K src1_type=f32 ids_type=i32 dst_type=f32 ne00=5120 ne01=6144 experts=256 used_experts=8 tokens=1 min_tokens=128 nr0=2 nr1=1 nsg=1 grid_x=1536 grid_y=1 grid_z=8 threads_x=32 threads_y=2";
    const METAL_ROUTE_ENCODE_CANDIDATE_LINE: &str = "skippy: glm_dsa_metal_dispatch op=topk_moe_route_encode tensor=blk.45.ffn_moe_probs candidate=UNARY/blk.45.ffn_moe_probs,RESHAPE/view reason=fused filtered_nodes=65 graph_nodes=71 graph_idx=30 grid_x=1 grid_y=1 grid_z=1 threads_x=1";
    const METAL_SELECTED_ROW_FLASH_CANDIDATE_LINE: &str = "skippy: glm_dsa_metal_dispatch op=selected_row_flash_candidate tensor=dsa_compact_k_topk_rows-30 reason=accepted_view next_tensor=__fattn__-30 generic=0 view=1 get_rows_uses=2 grid_x=0 grid_y=0 grid_z=0 threads_x=0";
    const METAL_WEIGHTED_DOWN_CANDIDATE_LINE: &str = "skippy: glm_dsa_metal_dispatch op=mul_mat_id_weighted_down_candidate tensor=ffn_moe_down-45 next=ffn_gate-45 next_op=MUL_MAT shared_gate=ffn_gate-45 shared_up=ffn_up-45 weighted_sum=ffn_moe_out-45 weighted_sum_op=MOE_WEIGHTED_SUM reason=full_motif shared_branch=1 weighted_sum_uses_down=1 pair_fusable=0 subgraph_fusable=1 filtered_gap=0 graph_gap=0 weighted_sum_gap=2 weighted_sum_graph_gap=2 src0_type=q3_K src1_type=f32 ids_type=i32 dst_type=f32 experts=256 used_experts=8 tokens=1 grid_x=1 grid_y=1 grid_z=1 threads_x=1";
    const METAL_GLM_DSA_MOE_MOTIF_CANDIDATE_LINE: &str = "skippy: glm_dsa_metal_dispatch op=glm_dsa_moe_motif_candidate tensor=ffn_moe_down-45 shared_gate=ffn_gate-45 shared_up=ffn_up-45 weighted_sum=ffn_moe_out-45 reason=full_motif natural_order=1 backend_candidate=1 subgraph_fusable=1 motif_nodes=4 fusion_outputs=3 weighted_sum_gap=2 weighted_sum_graph_gap=2 src0_type=q3_K src1_type=f32 ids_type=i32 dst_type=f32 experts=256 used_experts=8 tokens=1 grid_x=1 grid_y=1 grid_z=1 threads_x=1";
    const COMPUTE_BUFFER_LINES: &str = "~llama_context:       MTL0 compute buffer size is 2421.0264 MiB, matches expectation of 2421.0264 MiB\n~llama_context:       MTL0 compute buffer size of 667.8496 MiB, does not match expectation of 507.0029 MiB\n~llama_context:        CPU compute buffer size is  24.0059 MiB, matches expectation of  24.0059 MiB\n~llama_context:        CPU compute buffer size is   0.0000 MiB, matches expectation of   0.0000 MiB, trailing native detail";
    const INDEXSHARE_CONTRACT_LINE: &str = "llama_glm_dsa_log_indexshare_contract: GLM_DSA IndexShare source=metadata_types full_layers=21 shared_layers=57 indexer_tensor_layers=1 filtered_indexer_groups=0 out_of_stage_indexer_groups=76 stage_filtered=1 layer_start=30 layer_end=32 top_k=2048 top_k_frequency=0 skip_top_k_offset=0";
    const INDEXSHARE_EXEC_FULL_LINE: &str = "llama_model_glm_dsa::graph::graph: GLM_DSA IndexShare exec layer=30 role=full input_top_k=0 stage_filtered=1 layer_start=30 layer_end=34";
    const INDEXSHARE_TOP_K_LINE: &str = "llama_model_glm_dsa::graph::graph: GLM_DSA IndexShare top_k layer=30 source=indexer width=1024 score_width=4096";
    const INDEXSHARE_EXEC_SHARED_LINE: &str = "llama_model_glm_dsa::graph::graph: GLM_DSA IndexShare exec layer=31 role=shared input_top_k=1 stage_filtered=1 layer_start=30 layer_end=34";
    const INDEXSHARE_CONSUME_LINE: &str = "llama_model_glm_dsa::graph::graph: GLM_DSA IndexShare consume layer=31 source=last_top_k width=1024 batch=1 stream=1";

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

        let summary = summarize_log("stage1.log".into(), &[record], &[], &[], &[], &[]);
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

        let summary = summarize_log("stage1.log".into(), &[record], &[], &[], &[], &[]);
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

        let summary = summarize_log("stage1.log".into(), &[record], &[], &[], &[], &[]);
        let prefill = summary
            .stage_records
            .get(&1)
            .unwrap()
            .get(&Phase::Prefill)
            .unwrap();
        assert_eq!(prefill.dsa_sparse_attn.as_ref().unwrap().elapsed_us, 114543);
    }

    #[test]
    fn parses_optional_compact_get_rows_breakdown() {
        let record = parse_timing_record(LINE_WITH_COMPACT_GET_ROWS).unwrap();
        assert_eq!(record.compact_get_rows_nodes, Some(6));
        assert_eq!(record.compact_get_rows_us, Some(900));

        let summary = summarize_log("stage1.log".into(), &[record], &[], &[], &[], &[]);
        let decode = summary
            .stage_records
            .get(&1)
            .unwrap()
            .get(&Phase::Decode)
            .unwrap();
        assert_eq!(decode.compact_get_rows.as_ref().unwrap().elapsed_us, 900);
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
        assert_eq!(record.sparse_kv, None);
        assert_eq!(record.sparse_top_k, None);
        assert_eq!(record.min_kv_topk_ratio, None);
        assert_eq!(record.kv_topk_ratio, None);
        assert_eq!(record.dense_mask_bytes, Some(270_336));
        assert_eq!(record.dense_mask_limit, Some(536_870_912));
        assert_eq!(record.phase, None);
        assert_eq!(record.selector_reason, None);
        assert!(record.direct_enabled);
        assert!(record.prefill_enabled);
        assert!(!record.decode_shape);
        assert!(!record.prefill_shape);
        assert_eq!(record.large_prefill_shape, Some(false));
        assert!(!record.token_shape_allowed);
        assert!(record.kq_b_ok);
        assert!(record.sinks_ok);
        assert!(record.alibi_ok);
        assert!(record.soft_cap_ok);
        assert!(!record.use_direct);
    }

    #[test]
    fn parses_compact_flash_policy_records() {
        let records = parse_compact_flash_policy_records(COMPACT_FLASH_POLICY_LINE).unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.layer, 30);
        assert_eq!(record.ubatch_tokens, 1);
        assert_eq!(record.visible_kv, 8192);
        assert_eq!(record.top_k, 2048);
        assert_eq!(record.kv_topk_ratio, 4);
        assert_eq!(record.min_kv_topk_ratio, Some(2));
        assert!(!record.forced);
        assert!(!record.disabled);
        assert_eq!(record.ratio_ok, Some(true));
        assert!(record.enabled);
        assert!(record.flash_attn);
        assert_eq!(record.phase.as_deref(), Some("decode"));
        assert!(record.decode_shape);
        assert!(record.kq_b_ok);
        assert!(record.sinks_ok);
        assert!(record.alibi_ok);
        assert!(record.soft_cap_ok);
        assert_eq!(record.no_mask, Some(true));
        assert!(record.use_compact);
        assert_eq!(record.selector_reason.as_deref(), Some("decode_compact"));
    }

    #[test]
    fn parses_compact_flash_policy_without_min_ratio() {
        let records =
            parse_compact_flash_policy_records(COMPACT_FLASH_POLICY_LINE_WITHOUT_MIN_RATIO)
                .unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.min_kv_topk_ratio, None);
        assert_eq!(record.kv_topk_ratio, 4);
        assert!(record.use_compact);
    }

    #[test]
    fn parses_compact_flash_mask_records() {
        let records = parse_compact_flash_mask_records(COMPACT_FLASH_MASK_LINE).unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.layer, 30);
        assert!(record.omitted_mla_kq_mask);
        assert_eq!(record.visible_kv, 8192);
        assert_eq!(record.ubatch_tokens, 1);
        assert_eq!(record.streams, 1);
        assert_eq!(record.max_top_k, 2048);
    }

    #[test]
    fn parses_direct_sparse_decision_selector_reason() {
        let records =
            parse_direct_sparse_decision_records(DIRECT_SPARSE_DECISION_LINE_WITH_REASON).unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.ubatch_tokens, 1024);
        assert_eq!(record.sparse_kv, Some(99_328));
        assert_eq!(record.sparse_top_k, Some(1024));
        assert_eq!(record.min_kv_topk_ratio, Some(32));
        assert_eq!(record.kv_topk_ratio, Some(97));
        assert_eq!(record.dense_mask_bytes, Some(203_423_744));
        assert_eq!(record.dense_mask_limit, Some(268_435_456));
        assert_eq!(record.phase.as_deref(), Some("prefill"));
        assert_eq!(
            record.selector_reason.as_deref(),
            Some("dense_mask_guard_large_prefill")
        );
        assert!(record.use_direct);
    }

    #[test]
    fn parses_large_prefill_direct_sparse_decision_records() {
        let records =
            parse_direct_sparse_decision_records(LARGE_PREFILL_DIRECT_SPARSE_DECISION_LINE)
                .unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.ubatch_tokens, 4096);
        assert_eq!(record.sparse_batch, 4096);
        assert_eq!(record.dense_mask_bytes, Some(2_147_483_648));
        assert_eq!(record.dense_mask_limit, Some(536_870_912));
        assert!(!record.prefill_shape);
        assert_eq!(record.large_prefill_shape, Some(true));
        assert!(record.token_shape_allowed);
        assert!(record.use_direct);
    }

    #[test]
    fn parses_metal_dispatch_records() {
        let records = parse_metal_dispatch_records(METAL_DISPATCH_LINE).unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.op, "dsa_sparse_attn");
        assert_eq!(record.kernel.as_deref(), Some("decode_vec"));
        assert_eq!(record.tensor, "blk.30.dsa_sparse_attn");
        assert_eq!(record.q_width, Some(576));
        assert_eq!(record.v_width, Some(512));
        assert_eq!(record.batch, Some(32));
        assert_eq!(record.heads, Some(4));
        assert_eq!(record.top_k, Some(1024));
        assert_eq!(record.selected_keys, Some(1_048_576));
        assert_eq!(record.q_read_bytes, Some(2_415_919_104));
        assert_eq!(record.k_read_bytes, Some(1_207_959_552));
        assert_eq!(record.v_read_bytes, Some(1_073_741_824));
        assert_eq!(record.mask_read_bytes, Some(4_194_304));
        assert_eq!(record.top_k_read_bytes, Some(4_194_304));
        assert_eq!(record.scratch_per_tg_bytes, Some(1024));
        assert_eq!(record.score_fma, Some(603_979_776));
        assert_eq!(record.generic, None);
        assert_eq!(record.view, None);
        assert_eq!(record.get_rows_uses, None);
        assert_eq!(record.value_fma, Some(536_870_912));
        assert_eq!(
            record.reduction_strategy.as_deref(),
            Some("threadgroup_direct")
        );
        assert_eq!(record.grid_x, 32);
        assert_eq!(record.grid_y, 4);
        assert_eq!(record.grid_z, 8);
        assert_eq!(record.nwg, Some(8));
        assert_eq!(record.tmp_f16, Some(true));
        assert_eq!(record.dst_partial, Some(true));
        assert_eq!(record.threads_x, 256);
        assert_eq!(record.threads_y, None);
    }

    #[test]
    fn parses_selected_row_flash_candidate_record() {
        let records =
            parse_metal_dispatch_records(METAL_SELECTED_ROW_FLASH_CANDIDATE_LINE).unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.op, "selected_row_flash_candidate");
        assert_eq!(record.tensor, "dsa_compact_k_topk_rows-30");
        assert_eq!(record.reason.as_deref(), Some("accepted_view"));
        assert_eq!(record.generic, Some(false));
        assert_eq!(record.view, Some(true));
        assert_eq!(record.get_rows_uses, Some(2));
    }

    #[test]
    fn parses_decode_vec_reduce_dispatch_fields() {
        let records = parse_metal_dispatch_records(METAL_DECODE_VEC_REDUCE_DISPATCH_LINE).unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.op, "dsa_sparse_attn");
        assert_eq!(record.kernel.as_deref(), Some("decode_vec_reduce"));
        assert_eq!(record.rows, Some(262_144));
        assert_eq!(record.partial_bytes, Some(536_870_912));
        assert_eq!(record.softmax_bytes, Some(4_194_304));
        assert_eq!(record.tmp_bytes, Some(541_065_216));
        assert_eq!(record.grid_x, 262_144);
        assert_eq!(record.threads_x, 32);
        assert_eq!(record.threads_y, Some(2));
        assert_eq!(record.nwg, Some(2));
        assert_eq!(record.tmp_f16, Some(true));
        assert_eq!(record.dst_partial, Some(true));
    }

    #[test]
    fn parses_compute_buffer_records() {
        let records = parse_compute_buffer_records(COMPUTE_BUFFER_LINES).unwrap();
        assert_eq!(records.len(), 4);
        assert_eq!(records[0].device, "MTL0");
        assert_eq!(records[0].size_mib, 2421.0264);
        assert_eq!(records[0].expected_mib, 2421.0264);
        assert!(records[0].matches_expectation);
        assert_eq!(records[1].device, "MTL0");
        assert_eq!(records[1].size_mib, 667.8496);
        assert_eq!(records[1].expected_mib, 507.0029);
        assert!(!records[1].matches_expectation);
        assert_eq!(records[2].device, "CPU");
        assert_eq!(records[2].size_mib, 24.0059);
        assert!(records[2].matches_expectation);
        assert_eq!(records[3].device, "CPU");
        assert_eq!(records[3].size_mib, 0.0);
        assert_eq!(records[3].expected_mib, 0.0);
        assert!(records[3].matches_expectation);
    }

    #[test]
    fn parses_indexshare_trace_records() {
        let text = format!(
            "{INDEXSHARE_CONTRACT_LINE}\n{INDEXSHARE_EXEC_FULL_LINE}\n{INDEXSHARE_TOP_K_LINE}\n{INDEXSHARE_EXEC_SHARED_LINE}\n{INDEXSHARE_CONSUME_LINE}"
        );
        let records = parse_indexshare_trace_records(&text).unwrap();
        let contract_records = parse_indexshare_contract_records(&text).unwrap();

        assert_eq!(contract_records.len(), 1);
        assert_eq!(contract_records[0].source, "metadata_types");
        assert_eq!(contract_records[0].full_layers, 21);
        assert_eq!(contract_records[0].shared_layers, 57);
        assert_eq!(contract_records[0].indexer_tensor_layers, 1);
        assert_eq!(records.len(), 4);
        assert_eq!(records[0].event, super::IndexShareTraceEvent::Exec);
        assert_eq!(records[0].layer, 30);
        assert_eq!(records[0].role.as_deref(), Some("full"));
        assert_eq!(records[0].input_top_k, Some(false));
        assert_eq!(records[0].stage_filtered, Some(true));
        assert_eq!(records[0].layer_start, Some(30));
        assert_eq!(records[0].layer_end, Some(34));
        assert_eq!(records[1].event, super::IndexShareTraceEvent::TopK);
        assert_eq!(records[1].source.as_deref(), Some("indexer"));
        assert_eq!(records[1].width, Some(1024));
        assert_eq!(records[1].score_width, Some(4096));
        assert_eq!(records[3].event, super::IndexShareTraceEvent::Consume);
        assert_eq!(records[3].source.as_deref(), Some("last_top_k"));
        assert_eq!(records[3].batch, Some(1));
        assert_eq!(records[3].stream, Some(1));
    }

    #[test]
    fn summarizes_indexshare_trace_records() {
        let timing = parse_timing_records(LINE).unwrap();
        let text = format!(
            "{INDEXSHARE_EXEC_FULL_LINE}\n{INDEXSHARE_TOP_K_LINE}\n{INDEXSHARE_EXEC_SHARED_LINE}\n{INDEXSHARE_CONSUME_LINE}"
        );
        let records = parse_indexshare_trace_records(&text).unwrap();
        let contract_records = parse_indexshare_contract_records(INDEXSHARE_CONTRACT_LINE).unwrap();
        let summary = summarize_log(
            "stage1.log".into(),
            &timing,
            &[],
            &[],
            &records,
            &contract_records,
        );

        assert_eq!(summary.indexshare_trace.contract_records, 1);
        assert_eq!(
            summary.indexshare_trace.contract_sources,
            vec!["metadata_types".to_string()]
        );
        assert_eq!(summary.indexshare_trace.contract_full_layers, Some(21));
        assert_eq!(summary.indexshare_trace.contract_shared_layers, Some(57));
        assert_eq!(summary.indexshare_trace.records, 4);
        assert_eq!(summary.indexshare_trace.exec_records, 2);
        assert_eq!(summary.indexshare_trace.full_exec_records, 1);
        assert_eq!(summary.indexshare_trace.shared_exec_records, 1);
        assert_eq!(summary.indexshare_trace.shared_exec_with_input_top_k, 1);
        assert_eq!(summary.indexshare_trace.shared_exec_missing_input_top_k, 0);
        assert_eq!(summary.indexshare_trace.top_k_from_indexer, 1);
        assert_eq!(summary.indexshare_trace.consume_records, 1);
        assert_eq!(summary.indexshare_trace.min_consume_width, Some(1024));
        assert_eq!(summary.indexshare_trace.max_consume_width, Some(1024));
        assert_eq!(summary.indexshare_trace.full_layers, vec![30]);
        assert_eq!(summary.indexshare_trace.shared_layers, vec![31]);
    }

    #[test]
    fn accepts_indexshare_producer_consumer_guard() {
        let timing = parse_timing_records(LINE).unwrap();
        let text = format!(
            "{INDEXSHARE_EXEC_FULL_LINE}\n{INDEXSHARE_TOP_K_LINE}\n{INDEXSHARE_EXEC_SHARED_LINE}\n{INDEXSHARE_CONSUME_LINE}"
        );
        let records = parse_indexshare_trace_records(&text).unwrap();
        let contract_records = parse_indexshare_contract_records(INDEXSHARE_CONTRACT_LINE).unwrap();
        let summary = summarize_log(
            "stage1.log".into(),
            &timing,
            &[],
            &[],
            &records,
            &contract_records,
        );

        require_indexshare_producer_consumer_trace(Path::new("stage1.log"), &summary).unwrap();
    }

    #[test]
    fn rejects_indexshare_trace_without_shared_consumer() {
        let timing = parse_timing_records(LINE).unwrap();
        let text = format!("{INDEXSHARE_EXEC_FULL_LINE}\n{INDEXSHARE_TOP_K_LINE}");
        let records = parse_indexshare_trace_records(&text).unwrap();
        let contract_records = parse_indexshare_contract_records(INDEXSHARE_CONTRACT_LINE).unwrap();
        let summary = summarize_log(
            "stage1.log".into(),
            &timing,
            &[],
            &[],
            &records,
            &contract_records,
        );

        let error = require_indexshare_producer_consumer_trace(Path::new("stage1.log"), &summary)
            .unwrap_err();

        assert!(
            error.to_string().contains("shared_exec"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn parses_mul_mat_id_src0_type_as_source_type() {
        let records = parse_metal_dispatch_records(METAL_MUL_MAT_ID_DISPATCH_LINE).unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.op, "mul_mat_id");
        assert_eq!(record.kernel.as_deref(), Some("mul_mv_id"));
        assert_eq!(record.tensor, "ffn_moe_down-45");
        assert_eq!(record.src_type.as_deref(), Some("q3_K"));
        assert_eq!(record.dst_type.as_deref(), Some("f32"));
        assert_eq!(record.grid_x, 1536);
        assert_eq!(record.grid_z, 8);
        assert_eq!(record.threads_y, Some(2));
    }

    #[test]
    fn parses_metal_dispatch_route_candidate_reason() {
        let records = parse_metal_dispatch_records(METAL_ROUTE_ENCODE_CANDIDATE_LINE).unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.op, "topk_moe_route_encode");
        assert_eq!(record.tensor, "blk.45.ffn_moe_probs");
        assert_eq!(record.reason.as_deref(), Some("fused"));
        assert_eq!(record.grid_x, 1);
        assert_eq!(record.threads_x, 1);
    }

    #[test]
    fn parses_weighted_down_candidate_fields() {
        let records = parse_metal_dispatch_records(METAL_WEIGHTED_DOWN_CANDIDATE_LINE).unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.op, "mul_mat_id_weighted_down_candidate");
        assert_eq!(record.tensor, "ffn_moe_down-45");
        assert_eq!(record.next.as_deref(), Some("ffn_gate-45"));
        assert_eq!(record.next_op.as_deref(), Some("MUL_MAT"));
        assert_eq!(record.shared_gate.as_deref(), Some("ffn_gate-45"));
        assert_eq!(record.shared_up.as_deref(), Some("ffn_up-45"));
        assert_eq!(record.weighted_sum.as_deref(), Some("ffn_moe_out-45"));
        assert_eq!(record.weighted_sum_op.as_deref(), Some("MOE_WEIGHTED_SUM"));
        assert_eq!(record.reason.as_deref(), Some("full_motif"));
        assert_eq!(record.shared_branch, Some(true));
        assert_eq!(record.weighted_sum_uses_down, Some(true));
        assert_eq!(record.pair_fusable, Some(false));
        assert_eq!(record.subgraph_fusable, Some(true));
        assert_eq!(record.filtered_gap, Some(0));
        assert_eq!(record.graph_gap, Some(0));
        assert_eq!(record.weighted_sum_gap, Some(2));
        assert_eq!(record.weighted_sum_graph_gap, Some(2));
        assert_eq!(record.src_type.as_deref(), Some("q3_K"));
    }

    #[test]
    fn parses_glm_dsa_moe_motif_candidate_fields() {
        let records = parse_metal_dispatch_records(METAL_GLM_DSA_MOE_MOTIF_CANDIDATE_LINE).unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.op, "glm_dsa_moe_motif_candidate");
        assert_eq!(record.tensor, "ffn_moe_down-45");
        assert_eq!(record.shared_gate.as_deref(), Some("ffn_gate-45"));
        assert_eq!(record.shared_up.as_deref(), Some("ffn_up-45"));
        assert_eq!(record.weighted_sum.as_deref(), Some("ffn_moe_out-45"));
        assert_eq!(record.reason.as_deref(), Some("full_motif"));
        assert_eq!(record.natural_order, Some(true));
        assert_eq!(record.backend_candidate, Some(true));
        assert_eq!(record.subgraph_fusable, Some(true));
        assert_eq!(record.motif_nodes, Some(4));
        assert_eq!(record.fusion_outputs, Some(3));
        assert_eq!(record.weighted_sum_gap, Some(2));
        assert_eq!(record.weighted_sum_graph_gap, Some(2));
        assert_eq!(record.src_type.as_deref(), Some("q3_K"));
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
        let summary = summarize_log("stage1.log".into(), &records, &[], &[], &[], &[]);
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
        let summary = summarize_log("stage1.log".into(), &timing, &groups, &[], &[], &[]);
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
        assert_eq!(records[0].direction, SidebandDirection::Forward);
        assert_eq!(records[0].stage, "stage-0");
        assert_eq!(records[0].kind, "DecodeEmbd");
        assert_eq!(records[0].pos_start, 718);
        assert_eq!(records[0].sideband_bytes, 3072);
        assert_eq!(records[0].sideband_i32, 768);
    }

    #[test]
    fn parses_sideband_receive_record_with_prefix() {
        let records = parse_sideband_records(SIDEBAND_RECEIVE_LINE).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].direction, SidebandDirection::Receive);
        assert_eq!(records[0].stage, "stage-1");
        assert_eq!(records[0].kind, "DecodeEmbd");
        assert_eq!(records[0].pos_start, 718);
        assert_eq!(records[0].sideband_bytes, 3072);
        assert_eq!(records[0].sideband_i32, 768);
    }

    #[test]
    fn rejects_malformed_sideband_record() {
        let error = parse_sideband_record(
            SidebandDirection::Forward,
            "stage=stage-0 kind=DecodeEmbd pos_start=0",
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("missing tokens"));
    }

    #[test]
    fn summarizes_sideband_payload_ratios() {
        let timing = parse_timing_records(LINE).unwrap();
        let sideband = parse_sideband_records(SIDEBAND_LINE).unwrap();
        let summary = summarize_log("stage0.log".into(), &timing, &[], &sideband, &[], &[]);
        let stages = summary.sideband_records.get("stage-0").unwrap();
        let decode = stages.get(&Phase::Decode).unwrap();
        assert_eq!(decode.records, 1);
        assert_eq!(decode.forward_records, 1);
        assert_eq!(decode.receive_records, 0);
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
    fn summarizes_receive_sideband_records() {
        let timing = parse_timing_records(LINE).unwrap();
        let text = format!("{SIDEBAND_LINE}\n{SIDEBAND_RECEIVE_LINE}");
        let sideband = parse_sideband_records(&text).unwrap();
        let summary = summarize_log("stage0.log".into(), &timing, &[], &sideband, &[], &[]);
        let forward = summary
            .sideband_records
            .get("stage-0")
            .unwrap()
            .get(&Phase::Decode)
            .unwrap();
        let receive = summary
            .sideband_records
            .get("stage-1")
            .unwrap()
            .get(&Phase::Decode)
            .unwrap();
        assert_eq!(forward.forward_records, 1);
        assert_eq!(forward.receive_records, 0);
        assert_eq!(receive.forward_records, 0);
        assert_eq!(receive.receive_records, 1);
        assert_eq!(receive.sideband_bytes, 3072);
        assert_eq!(receive.sideband_i32, 768);
    }

    #[test]
    fn summarizes_sideband_padding_for_prefill() {
        let timing = parse_timing_records(LINE).unwrap();
        let sideband = parse_sideband_records(PADDED_PREFILL_SIDEBAND_LINE).unwrap();
        let summary = summarize_log("stage0.log".into(), &timing, &[], &sideband, &[], &[]);
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
            compact_get_rows: None,
            shared_expert: OpBucket {
                nodes: 1,
                elapsed_us: shared_expert_us,
            },
            ..PhaseSummary::default()
        }
    }
}
