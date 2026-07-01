use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::{
    cli::{GlmDsaAggregateReportCase, GlmDsaAggregateReportsArgs},
    glm_dsa_microbench_summary::{TimingDistributionSummary, summarize_elapsed_ms},
};

pub fn glm_dsa_aggregate_reports(args: GlmDsaAggregateReportsArgs) -> Result<()> {
    validate_args(&args)?;

    let mut groups: BTreeMap<AggregateKey, AggregateGroupBuilder> = BTreeMap::new();
    for report_path in &args.report {
        let report = read_report(report_path)?;
        let selected = select_case(&report, args.case).with_context(|| {
            format!("select {:?} case from {}", args.case, report_path.display())
        })?;
        let key = AggregateKey::new(&report, &selected);
        groups
            .entry(key)
            .or_insert_with(|| AggregateGroupBuilder::new(report_path, &report, &selected))
            .push(report_path, &report, selected);
    }

    let report = AggregateReport {
        command: "glm-dsa-aggregate-reports",
        case: args.case.as_str(),
        trim_fraction: args.trim_fraction,
        report_count: args.report.len(),
        groups: groups
            .into_values()
            .map(|group| group.finish(args.trim_fraction))
            .collect(),
    };
    write_report(args.output.as_deref(), &report)
}

fn validate_args(args: &GlmDsaAggregateReportsArgs) -> Result<()> {
    if args.report.is_empty() {
        bail!("at least one --report is required");
    }
    if !(0.0..0.5).contains(&args.trim_fraction) {
        bail!("--trim-fraction must be >= 0.0 and < 0.5");
    }
    Ok(())
}

fn read_report(path: &Path) -> Result<RawMicrobenchReport> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))
}

fn write_report(output: Option<&Path>, report: &AggregateReport) -> Result<()> {
    let encoded = format!("{}\n", serde_json::to_string_pretty(report)?);
    if let Some(output) = output {
        fs::write(output, encoded).with_context(|| format!("write {}", output.display()))?;
    } else {
        print!("{encoded}");
    }
    Ok(())
}

fn select_case(
    report: &RawMicrobenchReport,
    case: GlmDsaAggregateReportCase,
) -> Result<SelectedCase> {
    match case {
        GlmDsaAggregateReportCase::TopLevel => Ok(SelectedCase::from_top_level(report)),
        GlmDsaAggregateReportCase::Baseline => {
            let comparison = report
                .comparison
                .as_ref()
                .context("report does not contain a comparison")?;
            Ok(SelectedCase::from_raw_case(
                "baseline",
                &comparison.baseline,
            ))
        }
        GlmDsaAggregateReportCase::Candidate => {
            let comparison = report
                .comparison
                .as_ref()
                .context("report does not contain a comparison")?;
            Ok(SelectedCase::from_raw_case(
                "candidate",
                &comparison.candidate,
            ))
        }
    }
}

#[derive(Clone)]
struct SelectedCase {
    label: String,
    flags: RawFlags,
    measured_phase: Option<String>,
    timing_summary: RawTimingSummary,
    dispatch_summary: RawDispatchSummary,
    timings: Vec<RawIterationTiming>,
}

impl SelectedCase {
    fn from_top_level(report: &RawMicrobenchReport) -> Self {
        if let Some(representative) = &report.representative_profile {
            return Self {
                label: representative
                    .source
                    .clone()
                    .unwrap_or_else(|| "representative".to_string()),
                flags: report.flags.clone().unwrap_or_default(),
                measured_phase: measured_phase(
                    representative.timing_breakdown.as_ref(),
                    report.timing_breakdown.as_ref(),
                    report.tokens,
                ),
                timing_summary: representative.timing_summary.clone().unwrap_or_default(),
                dispatch_summary: representative
                    .metal_dispatch_summary
                    .clone()
                    .unwrap_or_default(),
                timings: Vec::new(),
            };
        }

        if report
            .profile_integrity
            .as_ref()
            .is_some_and(|integrity| integrity.diagnostic_timing_may_disable_route_fusion)
            && let Some(probe) = &report.optimized_dispatch_probe
        {
            return Self::from_raw_case("optimized_dispatch_probe", probe);
        }

        Self {
            label: "top-level".to_string(),
            flags: report.flags.clone().unwrap_or_default(),
            measured_phase: measured_phase(report.timing_breakdown.as_ref(), None, report.tokens),
            timing_summary: report.timing_summary.clone().unwrap_or_default(),
            dispatch_summary: report.metal_dispatch_summary.clone().unwrap_or_default(),
            timings: report.timings.clone(),
        }
    }

    fn from_raw_case(default_label: &'static str, case: &RawCaseReport) -> Self {
        Self {
            label: case
                .label
                .clone()
                .unwrap_or_else(|| default_label.to_string()),
            flags: case.flags.clone().unwrap_or_default(),
            measured_phase: measured_phase(case.timing_breakdown.as_ref(), None, None),
            timing_summary: case.timing_summary.clone().unwrap_or_default(),
            dispatch_summary: case.metal_dispatch_summary.clone().unwrap_or_default(),
            timings: case.timings.clone(),
        }
    }
}

fn measured_phase(
    primary: Option<&RawTimingBreakdown>,
    fallback: Option<&RawTimingBreakdown>,
    tokens: Option<usize>,
) -> Option<String> {
    primary
        .and_then(RawTimingBreakdown::measured_phase)
        .or_else(|| fallback.and_then(RawTimingBreakdown::measured_phase))
        .map(str::to_string)
        .or_else(|| tokens.map(infer_measured_phase).map(str::to_string))
}

fn infer_measured_phase(tokens: usize) -> &'static str {
    if tokens == 1 { "decode" } else { "prefill" }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct AggregateKey {
    model_id: Option<String>,
    layer_start: Option<u32>,
    layer_end: Option<u32>,
    position_start: Option<i64>,
    kv_warmup_tokens: Option<usize>,
    measured_phase: Option<String>,
    path: String,
    direct_sparse_attn: bool,
    compact_flash_attn: bool,
    direct_sparse_prefill: bool,
    metal_topk_moe_route_fusion: bool,
    metal_topk_moe_route_fusion_native_default: bool,
    moe_motif_coencode: bool,
    moe_down_weighted_fusion: bool,
    sparse_attn_threads: Option<u32>,
}

impl AggregateKey {
    fn new(report: &RawMicrobenchReport, selected: &SelectedCase) -> Self {
        Self {
            model_id: report.model_id.clone(),
            layer_start: report.layer_start,
            layer_end: report.layer_end,
            position_start: report.position_start,
            kv_warmup_tokens: report.kv_warmup_tokens,
            measured_phase: selected.measured_phase.clone(),
            path: path_name(selected).to_string(),
            direct_sparse_attn: selected.flags.direct_sparse_attn,
            compact_flash_attn: selected.flags.compact_flash_attn,
            direct_sparse_prefill: selected.flags.direct_sparse_prefill,
            metal_topk_moe_route_fusion: selected.flags.metal_topk_moe_route_fusion,
            metal_topk_moe_route_fusion_native_default: selected
                .flags
                .metal_topk_moe_route_fusion_native_default,
            moe_motif_coencode: selected.flags.moe_motif_coencode,
            moe_down_weighted_fusion: selected.flags.moe_down_weighted_fusion,
            sparse_attn_threads: selected.flags.sparse_attn_threads,
        }
    }
}

fn path_name(selected: &SelectedCase) -> &'static str {
    let flags = &selected.flags;
    if selected.dispatch_summary.proves_compact_flash() {
        return if flags.compact_flash_attn {
            "compact_flash_forced"
        } else {
            "compact_flash_native"
        };
    }
    if selected.dispatch_summary.proves_sparse_mask_flash() {
        return if flags.compact_flash_attn {
            "compact_flash_fallback"
        } else {
            "sparse_mask_flash"
        };
    }
    if selected.dispatch_summary.dense_sparse_mask_records() > 0 {
        return "sparse_mask";
    }

    match (
        flags.direct_sparse_attn,
        flags.compact_flash_attn,
        flags.direct_sparse_prefill,
    ) {
        (true, false, true) => "direct_sparse_prefill",
        (true, false, false) => "sparse_mask_flash",
        (false, true, _) => "compact_flash_fallback",
        (false, false, _) => "dense_fallback",
        (true, true, _) => "direct_sparse_and_compact_flash_fallback",
    }
}

struct AggregateGroupBuilder {
    key: AggregateKey,
    first_report: PathBuf,
    runs: Vec<AggregateRunReport>,
    sample_count: usize,
    run_means: Vec<f64>,
    pooled_timings: Vec<f64>,
    dispatch_family_summary: DispatchFamilySummary,
}

impl AggregateGroupBuilder {
    fn new(path: &Path, report: &RawMicrobenchReport, selected: &SelectedCase) -> Self {
        Self {
            key: AggregateKey::new(report, selected),
            first_report: path.to_path_buf(),
            runs: Vec::new(),
            sample_count: 0,
            run_means: Vec::new(),
            pooled_timings: Vec::new(),
            dispatch_family_summary: DispatchFamilySummary::default(),
        }
    }

    fn push(&mut self, path: &Path, report: &RawMicrobenchReport, selected: SelectedCase) {
        let elapsed_ms: Vec<f64> = selected
            .timings
            .iter()
            .map(|timing| timing.elapsed_ms)
            .filter(|elapsed| elapsed.is_finite())
            .collect();
        let sample_count = if elapsed_ms.is_empty() {
            selected.timing_summary.samples.unwrap_or(0)
        } else {
            elapsed_ms.len()
        };
        self.sample_count += sample_count;
        self.pooled_timings.extend(elapsed_ms);
        if let Some(mean) = selected.timing_summary.mean_ms
            && mean.is_finite()
        {
            self.run_means.push(mean);
        }
        let dispatch_family_summary = selected.dispatch_summary.family_summary();
        self.dispatch_family_summary.add(&dispatch_family_summary);
        self.runs.push(AggregateRunReport {
            path: path.to_path_buf(),
            model_id: report.model_id.clone(),
            label: selected.label,
            measured_phase: selected.measured_phase,
            sample_count,
            timing_summary: selected.timing_summary,
            dispatch_family_summary,
        });
    }

    fn finish(self, trim_fraction: f64) -> AggregateGroupReport {
        let run_mean_summary = summarize_elapsed_ms(self.run_means.iter().copied());
        let pooled_timing_summary = summarize_elapsed_ms(self.pooled_timings.iter().copied());
        AggregateGroupReport {
            key: self.key,
            first_report: self.first_report,
            run_count: self.runs.len(),
            sample_count: self.sample_count,
            run_mean_summary,
            pooled_timing_summary,
            dispatch_family_summary: self.dispatch_family_summary,
            trimmed_run_mean_ms: trimmed_mean(&self.run_means, trim_fraction),
            runs: self.runs,
        }
    }
}

fn trimmed_mean(values: &[f64], trim_fraction: f64) -> Option<f64> {
    let mut values: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return None;
    }

    let trim_count = ((values.len() as f64) * trim_fraction).floor() as usize;
    let start = trim_count.min(values.len());
    let end = values.len().saturating_sub(trim_count);
    let retained = values.get(start..end).unwrap_or_default();
    if retained.is_empty() {
        return None;
    }
    Some(retained.iter().sum::<f64>() / retained.len() as f64)
}

#[derive(Deserialize)]
struct RawMicrobenchReport {
    model_id: Option<String>,
    layer_start: Option<u32>,
    layer_end: Option<u32>,
    tokens: Option<usize>,
    position_start: Option<i64>,
    kv_warmup_tokens: Option<usize>,
    flags: Option<RawFlags>,
    timing_summary: Option<RawTimingSummary>,
    timing_breakdown: Option<RawTimingBreakdown>,
    metal_dispatch_summary: Option<RawDispatchSummary>,
    #[serde(default)]
    timings: Vec<RawIterationTiming>,
    representative_profile: Option<RawRepresentativeProfile>,
    profile_integrity: Option<RawProfileIntegrity>,
    optimized_dispatch_probe: Option<RawCaseReport>,
    comparison: Option<RawComparisonReport>,
}

#[derive(Clone, Default, Deserialize)]
struct RawFlags {
    #[serde(default)]
    direct_sparse_attn: bool,
    #[serde(default)]
    compact_flash_attn: bool,
    #[serde(default)]
    direct_sparse_prefill: bool,
    #[serde(default)]
    metal_topk_moe_route_fusion: bool,
    #[serde(default)]
    metal_topk_moe_route_fusion_native_default: bool,
    #[serde(default)]
    moe_motif_coencode: bool,
    #[serde(default)]
    moe_down_weighted_fusion: bool,
    sparse_attn_threads: Option<u32>,
}

#[derive(Clone, Default, Deserialize, Serialize)]
struct RawTimingSummary {
    samples: Option<usize>,
    mean_ms: Option<f64>,
    min_ms: Option<f64>,
    p50_ms: Option<f64>,
    p90_ms: Option<f64>,
    p95_ms: Option<f64>,
    p99_ms: Option<f64>,
    max_ms: Option<f64>,
    stdev_ms: Option<f64>,
    coefficient_of_variation: Option<f64>,
    slow_outlier_count: Option<usize>,
    slow_outlier_threshold_ms: Option<f64>,
}

#[derive(Clone, Default, Deserialize)]
struct RawTimingBreakdown {
    measured_phase: Option<String>,
}

impl RawTimingBreakdown {
    fn measured_phase(&self) -> Option<&str> {
        self.measured_phase
            .as_deref()
            .filter(|phase| matches!(*phase, "decode" | "prefill"))
    }
}

#[derive(Clone, Default, Deserialize)]
struct RawDispatchSummary {
    #[serde(default)]
    records: usize,
    #[serde(default)]
    topk_moe_route_fused_records: usize,
    #[serde(default)]
    topk_moe_route_encode_skipped_candidate_records: usize,
    #[serde(default)]
    flash_attn_ext_records: usize,
    #[serde(default)]
    flash_attn_ext_vec_records: usize,
    #[serde(default)]
    flash_attn_ext_tile_records: usize,
    #[serde(default)]
    flash_attn_ext_glm_dsa_shape_records: usize,
    #[serde(default)]
    dsa_compact_get_rows_fused_records: usize,
    #[serde(default)]
    get_rows_records: usize,
    #[serde(default)]
    get_rows_typed_records: usize,
    #[serde(default)]
    get_rows_promote_records: usize,
    #[serde(default)]
    dsa_sparse_attn_records: usize,
    #[serde(default)]
    mul_mat_id_records: usize,
    #[serde(default)]
    moe_weighted_sum_records: usize,
    #[serde(default)]
    mul_mv_id_weighted_sum_fused_q3_k_records: usize,
    #[serde(default)]
    dispatch_shapes: Vec<RawDispatchShape>,
}

impl RawDispatchSummary {
    fn family_summary(&self) -> DispatchFamilySummary {
        DispatchFamilySummary {
            records: self.records,
            dense_sparse_mask_records: self.dense_sparse_mask_records(),
            flash_attn_ext_records: self.flash_attn_ext_records,
            flash_attn_ext_vec_records: self.flash_attn_ext_vec_records,
            flash_attn_ext_tile_records: self.flash_attn_ext_tile_records,
            flash_attn_ext_glm_dsa_shape_records: self.flash_attn_ext_glm_dsa_shape_records,
            dsa_compact_get_rows_fused_records: self.dsa_compact_get_rows_fused_records,
            get_rows_records: self.get_rows_records,
            get_rows_typed_records: self.get_rows_typed_records,
            get_rows_promote_records: self.get_rows_promote_records,
            dsa_sparse_attn_records: self.dsa_sparse_attn_records,
            topk_moe_route_fused_records: self.topk_moe_route_fused_records,
            topk_moe_route_encode_skipped_candidate_records: self
                .topk_moe_route_encode_skipped_candidate_records,
            mul_mat_id_records: self.mul_mat_id_records,
            moe_weighted_sum_records: self.moe_weighted_sum_records,
            mul_mv_id_weighted_sum_fused_q3_k_records: self
                .mul_mv_id_weighted_sum_fused_q3_k_records,
        }
    }

    fn dense_sparse_mask_records(&self) -> usize {
        self.dispatch_shapes
            .iter()
            .filter(|shape| shape.op.as_deref() == Some("dsa_sparse_mask"))
            .map(|shape| shape.records.unwrap_or(1))
            .sum()
    }

    fn proves_compact_flash(&self) -> bool {
        self.flash_attn_ext_glm_dsa_shape_records > 0
            && self.flash_attn_ext_vec_records > 0
            && self.dsa_sparse_attn_records == 0
            && self.get_rows_promote_records == 0
            && (self.get_rows_typed_records > 0 || self.dsa_compact_get_rows_fused_records > 0)
    }

    fn proves_sparse_mask_flash(&self) -> bool {
        self.dense_sparse_mask_records() > 0
            && self.flash_attn_ext_glm_dsa_shape_records > 0
            && self.flash_attn_ext_vec_records > 0
            && self.dsa_sparse_attn_records == 0
    }
}

#[derive(Clone, Default, Deserialize)]
struct RawDispatchShape {
    op: Option<String>,
    records: Option<usize>,
}

#[derive(Clone, Deserialize)]
struct RawIterationTiming {
    elapsed_ms: f64,
}

#[derive(Deserialize)]
struct RawRepresentativeProfile {
    source: Option<String>,
    timing_summary: Option<RawTimingSummary>,
    timing_breakdown: Option<RawTimingBreakdown>,
    metal_dispatch_summary: Option<RawDispatchSummary>,
}

#[derive(Deserialize)]
struct RawProfileIntegrity {
    #[serde(default)]
    diagnostic_timing_may_disable_route_fusion: bool,
}

#[derive(Deserialize)]
struct RawComparisonReport {
    baseline: RawCaseReport,
    candidate: RawCaseReport,
}

#[derive(Deserialize)]
struct RawCaseReport {
    label: Option<String>,
    flags: Option<RawFlags>,
    timing_summary: Option<RawTimingSummary>,
    timing_breakdown: Option<RawTimingBreakdown>,
    metal_dispatch_summary: Option<RawDispatchSummary>,
    #[serde(default)]
    timings: Vec<RawIterationTiming>,
}

#[derive(Serialize)]
struct AggregateReport {
    command: &'static str,
    case: &'static str,
    trim_fraction: f64,
    report_count: usize,
    groups: Vec<AggregateGroupReport>,
}

#[derive(Serialize)]
struct AggregateGroupReport {
    key: AggregateKey,
    first_report: PathBuf,
    run_count: usize,
    sample_count: usize,
    #[serde(skip_serializing_if = "TimingDistributionSummary::is_empty")]
    run_mean_summary: TimingDistributionSummary,
    #[serde(skip_serializing_if = "TimingDistributionSummary::is_empty")]
    pooled_timing_summary: TimingDistributionSummary,
    #[serde(skip_serializing_if = "DispatchFamilySummary::is_empty")]
    dispatch_family_summary: DispatchFamilySummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    trimmed_run_mean_ms: Option<f64>,
    runs: Vec<AggregateRunReport>,
}

#[derive(Clone, Debug, Default, Serialize)]
struct DispatchFamilySummary {
    records: usize,
    dense_sparse_mask_records: usize,
    flash_attn_ext_records: usize,
    flash_attn_ext_vec_records: usize,
    flash_attn_ext_tile_records: usize,
    flash_attn_ext_glm_dsa_shape_records: usize,
    dsa_compact_get_rows_fused_records: usize,
    get_rows_records: usize,
    get_rows_typed_records: usize,
    get_rows_promote_records: usize,
    dsa_sparse_attn_records: usize,
    topk_moe_route_fused_records: usize,
    topk_moe_route_encode_skipped_candidate_records: usize,
    mul_mat_id_records: usize,
    moe_weighted_sum_records: usize,
    mul_mv_id_weighted_sum_fused_q3_k_records: usize,
}

impl DispatchFamilySummary {
    fn is_empty(summary: &Self) -> bool {
        summary.records == 0
            && summary.dense_sparse_mask_records == 0
            && summary.flash_attn_ext_records == 0
            && summary.flash_attn_ext_glm_dsa_shape_records == 0
            && summary.dsa_compact_get_rows_fused_records == 0
            && summary.get_rows_typed_records == 0
            && summary.get_rows_promote_records == 0
            && summary.dsa_sparse_attn_records == 0
            && summary.mul_mat_id_records == 0
            && summary.moe_weighted_sum_records == 0
            && summary.mul_mv_id_weighted_sum_fused_q3_k_records == 0
    }

    fn add(&mut self, other: &Self) {
        self.records += other.records;
        self.dense_sparse_mask_records += other.dense_sparse_mask_records;
        self.flash_attn_ext_records += other.flash_attn_ext_records;
        self.flash_attn_ext_vec_records += other.flash_attn_ext_vec_records;
        self.flash_attn_ext_tile_records += other.flash_attn_ext_tile_records;
        self.flash_attn_ext_glm_dsa_shape_records += other.flash_attn_ext_glm_dsa_shape_records;
        self.dsa_compact_get_rows_fused_records += other.dsa_compact_get_rows_fused_records;
        self.get_rows_records += other.get_rows_records;
        self.get_rows_typed_records += other.get_rows_typed_records;
        self.get_rows_promote_records += other.get_rows_promote_records;
        self.dsa_sparse_attn_records += other.dsa_sparse_attn_records;
        self.topk_moe_route_fused_records += other.topk_moe_route_fused_records;
        self.topk_moe_route_encode_skipped_candidate_records +=
            other.topk_moe_route_encode_skipped_candidate_records;
        self.mul_mat_id_records += other.mul_mat_id_records;
        self.moe_weighted_sum_records += other.moe_weighted_sum_records;
        self.mul_mv_id_weighted_sum_fused_q3_k_records +=
            other.mul_mv_id_weighted_sum_fused_q3_k_records;
    }
}

#[derive(Serialize)]
struct AggregateRunReport {
    path: PathBuf,
    model_id: Option<String>,
    label: String,
    measured_phase: Option<String>,
    sample_count: usize,
    timing_summary: RawTimingSummary,
    #[serde(skip_serializing_if = "DispatchFamilySummary::is_empty")]
    dispatch_family_summary: DispatchFamilySummary,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trimmed_mean_drops_each_tail() {
        let values = [10.0, 12.0, 14.0, 100.0, 8.0];
        let mean = trimmed_mean(&values, 0.20).unwrap();

        assert_eq!(mean, 12.0);
    }

    #[test]
    fn aggregate_groups_by_path_and_position() {
        let report = RawMicrobenchReport {
            model_id: Some("meshllm/test".to_string()),
            layer_start: Some(30),
            layer_end: Some(34),
            tokens: Some(2),
            position_start: Some(4096),
            kv_warmup_tokens: Some(4096),
            flags: Some(RawFlags {
                direct_sparse_attn: true,
                compact_flash_attn: false,
                direct_sparse_prefill: true,
                sparse_attn_threads: Some(256),
                ..RawFlags::default()
            }),
            timing_summary: Some(RawTimingSummary {
                samples: Some(2),
                mean_ms: Some(6.0),
                ..RawTimingSummary::default()
            }),
            timing_breakdown: None,
            metal_dispatch_summary: None,
            timings: vec![
                RawIterationTiming { elapsed_ms: 5.0 },
                RawIterationTiming { elapsed_ms: 7.0 },
            ],
            representative_profile: None,
            profile_integrity: None,
            optimized_dispatch_probe: None,
            comparison: None,
        };
        let selected = select_case(&report, GlmDsaAggregateReportCase::TopLevel).unwrap();
        let mut group = AggregateGroupBuilder::new(Path::new("a.json"), &report, &selected);
        group.push(Path::new("a.json"), &report, selected);

        let finished = group.finish(0.10);

        assert_eq!(finished.key.path, "direct_sparse_prefill");
        assert!(finished.key.direct_sparse_prefill);
        assert_eq!(finished.key.sparse_attn_threads, Some(256));
        assert_eq!(finished.key.measured_phase.as_deref(), Some("prefill"));
        assert_eq!(finished.runs[0].measured_phase.as_deref(), Some("prefill"));
        assert_eq!(finished.run_count, 1);
        assert_eq!(finished.sample_count, 2);
        assert_eq!(finished.trimmed_run_mean_ms, Some(6.0));
        assert_eq!(finished.pooled_timing_summary.mean_ms, Some(6.0));
    }

    #[test]
    fn top_level_selection_prefers_representative_profile_timing() {
        let report = RawMicrobenchReport {
            model_id: Some("meshllm/test".to_string()),
            layer_start: Some(30),
            layer_end: Some(46),
            tokens: Some(1),
            position_start: Some(32768),
            kv_warmup_tokens: Some(32768),
            flags: Some(RawFlags {
                direct_sparse_attn: true,
                compact_flash_attn: true,
                metal_topk_moe_route_fusion: true,
                moe_motif_coencode: true,
                moe_down_weighted_fusion: true,
                ..RawFlags::default()
            }),
            timing_summary: Some(RawTimingSummary {
                samples: Some(3),
                mean_ms: Some(153.0),
                ..RawTimingSummary::default()
            }),
            timing_breakdown: Some(RawTimingBreakdown {
                measured_phase: Some("decode".to_string()),
            }),
            metal_dispatch_summary: None,
            timings: vec![
                RawIterationTiming { elapsed_ms: 150.0 },
                RawIterationTiming { elapsed_ms: 153.0 },
                RawIterationTiming { elapsed_ms: 156.0 },
            ],
            representative_profile: Some(RawRepresentativeProfile {
                source: Some("optimized_dispatch_probe".to_string()),
                timing_summary: Some(RawTimingSummary {
                    samples: Some(3),
                    mean_ms: Some(1.55),
                    ..RawTimingSummary::default()
                }),
                timing_breakdown: Some(RawTimingBreakdown {
                    measured_phase: Some("decode".to_string()),
                }),
                metal_dispatch_summary: None,
            }),
            profile_integrity: None,
            optimized_dispatch_probe: None,
            comparison: None,
        };

        let selected = select_case(&report, GlmDsaAggregateReportCase::TopLevel).unwrap();
        let mut group = AggregateGroupBuilder::new(Path::new("a.json"), &report, &selected);
        group.push(Path::new("a.json"), &report, selected);
        let finished = group.finish(0.10);

        assert_eq!(finished.runs[0].label, "optimized_dispatch_probe");
        assert_eq!(finished.key.measured_phase.as_deref(), Some("decode"));
        assert_eq!(finished.runs[0].measured_phase.as_deref(), Some("decode"));
        assert_eq!(finished.runs[0].sample_count, 3);
        assert_eq!(finished.runs[0].timing_summary.mean_ms, Some(1.55));
        assert_eq!(finished.sample_count, 3);
        assert_eq!(finished.run_mean_summary.mean_ms, Some(1.55));
        assert_eq!(finished.pooled_timing_summary.samples, 0);
    }

    #[test]
    fn top_level_selection_uses_legacy_optimized_probe_when_diagnostic_is_perturbed() {
        let report = RawMicrobenchReport {
            model_id: Some("meshllm/test".to_string()),
            layer_start: Some(30),
            layer_end: Some(46),
            tokens: Some(1),
            position_start: Some(32768),
            kv_warmup_tokens: Some(32768),
            flags: Some(RawFlags {
                direct_sparse_attn: true,
                compact_flash_attn: true,
                metal_topk_moe_route_fusion: true,
                moe_motif_coencode: true,
                moe_down_weighted_fusion: true,
                ..RawFlags::default()
            }),
            timing_summary: Some(RawTimingSummary {
                samples: Some(3),
                mean_ms: Some(153.0),
                ..RawTimingSummary::default()
            }),
            timing_breakdown: Some(RawTimingBreakdown {
                measured_phase: Some("decode".to_string()),
            }),
            metal_dispatch_summary: None,
            timings: vec![
                RawIterationTiming { elapsed_ms: 150.0 },
                RawIterationTiming { elapsed_ms: 153.0 },
                RawIterationTiming { elapsed_ms: 156.0 },
            ],
            representative_profile: None,
            profile_integrity: Some(RawProfileIntegrity {
                diagnostic_timing_may_disable_route_fusion: true,
            }),
            optimized_dispatch_probe: Some(RawCaseReport {
                label: Some("optimized_dispatch_probe".to_string()),
                flags: Some(RawFlags {
                    direct_sparse_attn: true,
                    compact_flash_attn: true,
                    metal_topk_moe_route_fusion: true,
                    moe_motif_coencode: true,
                    moe_down_weighted_fusion: true,
                    ..RawFlags::default()
                }),
                timing_summary: Some(RawTimingSummary {
                    samples: Some(3),
                    mean_ms: Some(1.55),
                    ..RawTimingSummary::default()
                }),
                timing_breakdown: None,
                metal_dispatch_summary: None,
                timings: vec![
                    RawIterationTiming { elapsed_ms: 1.50 },
                    RawIterationTiming { elapsed_ms: 1.55 },
                    RawIterationTiming { elapsed_ms: 1.60 },
                ],
            }),
            comparison: None,
        };

        let selected = select_case(&report, GlmDsaAggregateReportCase::TopLevel).unwrap();
        let mut group = AggregateGroupBuilder::new(Path::new("a.json"), &report, &selected);
        group.push(Path::new("a.json"), &report, selected);
        let finished = group.finish(0.10);

        assert_eq!(finished.runs[0].label, "optimized_dispatch_probe");
        assert_eq!(finished.key.measured_phase.as_deref(), None);
        assert_eq!(finished.runs[0].sample_count, 3);
        assert_eq!(finished.sample_count, 3);
        assert_eq!(finished.runs[0].timing_summary.mean_ms, Some(1.55));
        assert_eq!(finished.run_mean_summary.mean_ms, Some(1.55));
        assert_eq!(finished.pooled_timing_summary.mean_ms, Some(1.55));
    }

    #[test]
    fn aggregate_key_separates_glm_fusion_ablations() {
        let route_on = report_with_fusion_flags(true, true);
        let route_off = report_with_fusion_flags(false, true);
        let down_off = report_with_fusion_flags(true, false);

        let route_on_case = select_case(&route_on, GlmDsaAggregateReportCase::TopLevel).unwrap();
        let route_off_case = select_case(&route_off, GlmDsaAggregateReportCase::TopLevel).unwrap();
        let down_off_case = select_case(&down_off, GlmDsaAggregateReportCase::TopLevel).unwrap();

        assert_ne!(
            AggregateKey::new(&route_on, &route_on_case),
            AggregateKey::new(&route_off, &route_off_case)
        );
        assert_ne!(
            AggregateKey::new(&route_on, &route_on_case),
            AggregateKey::new(&down_off, &down_off_case)
        );
    }

    #[test]
    fn aggregate_key_separates_prefill_sparse_path_from_dense_flash_path() {
        let direct_sparse = report_with_prefill_path(true, Some(256));
        let sparse_mask_flash = report_with_prefill_path(false, None);

        let direct_case = select_case(&direct_sparse, GlmDsaAggregateReportCase::TopLevel).unwrap();
        let flash_case =
            select_case(&sparse_mask_flash, GlmDsaAggregateReportCase::TopLevel).unwrap();

        let direct_key = AggregateKey::new(&direct_sparse, &direct_case);
        let flash_key = AggregateKey::new(&sparse_mask_flash, &flash_case);

        assert_ne!(direct_key, flash_key);
        assert_eq!(direct_key.path, "direct_sparse_prefill");
        assert_eq!(direct_key.sparse_attn_threads, Some(256));
        assert_eq!(flash_key.path, "sparse_mask_flash");
        assert_eq!(flash_key.sparse_attn_threads, None);
    }

    #[test]
    fn aggregate_key_uses_dispatch_proof_for_native_compact_flash() {
        let report = report_with_dispatch_summary(RawDispatchSummary {
            records: 48,
            flash_attn_ext_records: 3,
            flash_attn_ext_vec_records: 3,
            flash_attn_ext_glm_dsa_shape_records: 3,
            get_rows_records: 9,
            get_rows_typed_records: 9,
            get_rows_promote_records: 0,
            dsa_sparse_attn_records: 0,
            ..RawDispatchSummary::default()
        });
        let selected = select_case(&report, GlmDsaAggregateReportCase::TopLevel).unwrap();
        let key = AggregateKey::new(&report, &selected);

        assert_eq!(key.path, "compact_flash_native");
        assert!(!key.compact_flash_attn);
        assert!(key.direct_sparse_attn);
    }

    #[test]
    fn aggregate_key_marks_forced_compact_flash_fallback_without_dispatch_proof() {
        let report = RawMicrobenchReport {
            model_id: Some("meshllm/test".to_string()),
            layer_start: Some(6),
            layer_end: Some(10),
            tokens: Some(4),
            position_start: Some(1024),
            kv_warmup_tokens: Some(1024),
            flags: Some(RawFlags {
                direct_sparse_attn: false,
                compact_flash_attn: true,
                direct_sparse_prefill: false,
                metal_topk_moe_route_fusion: true,
                ..RawFlags::default()
            }),
            timing_summary: Some(RawTimingSummary {
                samples: Some(3),
                mean_ms: Some(14.7),
                ..RawTimingSummary::default()
            }),
            timing_breakdown: Some(RawTimingBreakdown {
                measured_phase: Some("prefill".to_string()),
            }),
            metal_dispatch_summary: Some(RawDispatchSummary {
                records: 208,
                flash_attn_ext_records: 16,
                flash_attn_ext_vec_records: 16,
                flash_attn_ext_glm_dsa_shape_records: 16,
                dispatch_shapes: vec![RawDispatchShape {
                    op: Some("dsa_sparse_mask".to_string()),
                    records: Some(32),
                }],
                ..RawDispatchSummary::default()
            }),
            timings: vec![],
            representative_profile: None,
            profile_integrity: None,
            optimized_dispatch_probe: None,
            comparison: None,
        };

        let selected = select_case(&report, GlmDsaAggregateReportCase::TopLevel).unwrap();
        let key = AggregateKey::new(&report, &selected);

        assert_eq!(key.path, "compact_flash_fallback");
        assert!(key.compact_flash_attn);
    }

    #[test]
    fn aggregate_key_uses_dispatch_proof_for_sparse_mask_flash() {
        let report = report_with_dispatch_summary(RawDispatchSummary {
            records: 208,
            flash_attn_ext_records: 16,
            flash_attn_ext_vec_records: 16,
            flash_attn_ext_glm_dsa_shape_records: 16,
            dispatch_shapes: vec![RawDispatchShape {
                op: Some("dsa_sparse_mask".to_string()),
                records: Some(32),
            }],
            ..RawDispatchSummary::default()
        });
        let selected = select_case(&report, GlmDsaAggregateReportCase::TopLevel).unwrap();
        let key = AggregateKey::new(&report, &selected);

        assert_eq!(key.path, "sparse_mask_flash");
        assert!(key.direct_sparse_attn);
        assert!(!key.compact_flash_attn);
    }

    #[test]
    fn aggregate_summarizes_dispatch_families() {
        let report = report_with_dispatch_summary(RawDispatchSummary {
            records: 12,
            topk_moe_route_fused_records: 2,
            flash_attn_ext_records: 3,
            flash_attn_ext_tile_records: 3,
            flash_attn_ext_glm_dsa_shape_records: 2,
            get_rows_typed_records: 4,
            dsa_sparse_attn_records: 1,
            mul_mat_id_records: 4,
            moe_weighted_sum_records: 1,
            dispatch_shapes: vec![
                RawDispatchShape {
                    op: Some("dsa_sparse_mask".to_string()),
                    records: Some(2),
                },
                RawDispatchShape {
                    op: Some("dsa_sparse_mask".to_string()),
                    records: Some(3),
                },
            ],
            ..RawDispatchSummary::default()
        });
        let selected = select_case(&report, GlmDsaAggregateReportCase::TopLevel).unwrap();
        let mut group = AggregateGroupBuilder::new(Path::new("a.json"), &report, &selected);
        group.push(Path::new("a.json"), &report, selected);

        let finished = group.finish(0.10);

        assert_eq!(finished.dispatch_family_summary.records, 12);
        assert_eq!(
            finished.dispatch_family_summary.dense_sparse_mask_records,
            5
        );
        assert_eq!(finished.dispatch_family_summary.flash_attn_ext_records, 3);
        assert_eq!(
            finished.dispatch_family_summary.flash_attn_ext_tile_records,
            3
        );
        assert_eq!(
            finished
                .dispatch_family_summary
                .flash_attn_ext_glm_dsa_shape_records,
            2
        );
        assert_eq!(finished.dispatch_family_summary.get_rows_typed_records, 4);
        assert_eq!(finished.dispatch_family_summary.dsa_sparse_attn_records, 1);
        assert_eq!(finished.dispatch_family_summary.mul_mat_id_records, 4);
        assert_eq!(finished.dispatch_family_summary.moe_weighted_sum_records, 1);
        assert_eq!(
            finished.runs[0]
                .dispatch_family_summary
                .dense_sparse_mask_records,
            5
        );
    }

    fn report_with_fusion_flags(
        metal_topk_moe_route_fusion: bool,
        moe_down_weighted_fusion: bool,
    ) -> RawMicrobenchReport {
        RawMicrobenchReport {
            model_id: Some("meshllm/test".to_string()),
            layer_start: Some(30),
            layer_end: Some(46),
            tokens: Some(1),
            position_start: Some(65536),
            kv_warmup_tokens: Some(65536),
            flags: Some(RawFlags {
                direct_sparse_attn: true,
                compact_flash_attn: true,
                metal_topk_moe_route_fusion,
                moe_motif_coencode: true,
                moe_down_weighted_fusion,
                ..RawFlags::default()
            }),
            timing_summary: Some(RawTimingSummary {
                samples: Some(1),
                mean_ms: Some(1.0),
                ..RawTimingSummary::default()
            }),
            timing_breakdown: None,
            metal_dispatch_summary: None,
            timings: vec![RawIterationTiming { elapsed_ms: 1.0 }],
            representative_profile: None,
            profile_integrity: None,
            optimized_dispatch_probe: None,
            comparison: None,
        }
    }

    fn report_with_prefill_path(
        direct_sparse_prefill: bool,
        sparse_attn_threads: Option<u32>,
    ) -> RawMicrobenchReport {
        RawMicrobenchReport {
            model_id: Some("meshllm/test".to_string()),
            layer_start: Some(31),
            layer_end: Some(32),
            tokens: Some(2304),
            position_start: Some(0),
            kv_warmup_tokens: Some(0),
            flags: Some(RawFlags {
                direct_sparse_attn: true,
                compact_flash_attn: false,
                direct_sparse_prefill,
                sparse_attn_threads,
                metal_topk_moe_route_fusion: true,
                moe_motif_coencode: true,
                moe_down_weighted_fusion: true,
                ..RawFlags::default()
            }),
            timing_summary: Some(RawTimingSummary {
                samples: Some(1),
                mean_ms: Some(1.0),
                ..RawTimingSummary::default()
            }),
            timing_breakdown: Some(RawTimingBreakdown {
                measured_phase: Some("prefill".to_string()),
            }),
            metal_dispatch_summary: None,
            timings: vec![RawIterationTiming { elapsed_ms: 1.0 }],
            representative_profile: None,
            profile_integrity: None,
            optimized_dispatch_probe: None,
            comparison: None,
        }
    }

    fn report_with_dispatch_summary(dispatch: RawDispatchSummary) -> RawMicrobenchReport {
        RawMicrobenchReport {
            model_id: Some("meshllm/test".to_string()),
            layer_start: Some(31),
            layer_end: Some(32),
            tokens: Some(2304),
            position_start: Some(0),
            kv_warmup_tokens: Some(0),
            flags: Some(RawFlags {
                direct_sparse_attn: true,
                compact_flash_attn: false,
                direct_sparse_prefill: false,
                metal_topk_moe_route_fusion: true,
                ..RawFlags::default()
            }),
            timing_summary: Some(RawTimingSummary {
                samples: Some(1),
                mean_ms: Some(1.0),
                ..RawTimingSummary::default()
            }),
            timing_breakdown: Some(RawTimingBreakdown {
                measured_phase: Some("prefill".to_string()),
            }),
            metal_dispatch_summary: Some(dispatch),
            timings: vec![RawIterationTiming { elapsed_ms: 1.0 }],
            representative_profile: None,
            profile_integrity: None,
            optimized_dispatch_probe: None,
            comparison: None,
        }
    }
}
