use std::collections::BTreeMap;

use serde::Serialize;

use crate::glm_dsa_op_report::{MetalDispatchRecord, TimingRecord};

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct TimingDistributionSummary {
    pub(crate) samples: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) mean_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) min_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) p50_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) p90_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) p95_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) p99_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stdev_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) coefficient_of_variation: Option<f64>,
    pub(crate) slow_outlier_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) slow_outlier_threshold_ms: Option<f64>,
}

impl TimingDistributionSummary {
    pub(crate) fn is_empty(summary: &Self) -> bool {
        summary.samples == 0
    }
}

pub(crate) fn summarize_elapsed_ms(
    values: impl IntoIterator<Item = f64>,
) -> TimingDistributionSummary {
    let mut values: Vec<f64> = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect();
    values.sort_by(f64::total_cmp);
    let samples = values.len();
    if samples == 0 {
        return TimingDistributionSummary::default();
    }

    let sum: f64 = values.iter().sum();
    let mean = sum / samples as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / samples as f64;
    let stdev = variance.sqrt();
    let p50 = percentile(&values, 0.50);
    let slow_outlier_threshold = p50 * 1.25;
    let slow_outlier_count = values
        .iter()
        .filter(|value| **value > slow_outlier_threshold)
        .count();

    TimingDistributionSummary {
        samples,
        mean_ms: Some(mean),
        min_ms: values.first().copied(),
        p50_ms: Some(p50),
        p90_ms: Some(percentile(&values, 0.90)),
        p95_ms: Some(percentile(&values, 0.95)),
        p99_ms: Some(percentile(&values, 0.99)),
        max_ms: values.last().copied(),
        stdev_ms: Some(stdev),
        coefficient_of_variation: if mean > f64::EPSILON {
            Some(stdev / mean)
        } else {
            None
        },
        slow_outlier_count,
        slow_outlier_threshold_ms: Some(slow_outlier_threshold),
    }
}

fn percentile(sorted_values: &[f64], quantile: f64) -> f64 {
    debug_assert!(!sorted_values.is_empty());
    let last_index = sorted_values.len() - 1;
    let index = ((last_index as f64) * quantile).round() as usize;
    sorted_values[index.min(last_index)]
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct GlmDsaDispatchSummary {
    pub(crate) records: usize,
    pub(crate) topk_moe_route_fused_records: usize,
    pub(crate) topk_moe_route_pack_records: usize,
    pub(crate) topk_moe_route_encode_records: usize,
    pub(crate) topk_moe_route_pack_candidate_records: usize,
    pub(crate) topk_moe_route_packed_candidate_records: usize,
    pub(crate) topk_moe_route_pack_skipped_candidate_records: usize,
    pub(crate) topk_moe_route_encode_candidate_records: usize,
    pub(crate) topk_moe_route_encode_fused_candidate_records: usize,
    pub(crate) topk_moe_route_encode_skipped_candidate_records: usize,
    pub(crate) dsa_sparse_attn_records: usize,
    pub(crate) mul_mat_id_records: usize,
    pub(crate) moe_weighted_sum_records: usize,
    pub(crate) moe_weighted_sum_f32x4_records: usize,
    pub(crate) routed_moe_gate_records: usize,
    pub(crate) routed_moe_up_records: usize,
    pub(crate) routed_moe_down_records: usize,
    pub(crate) routed_moe_down_q3_k_records: usize,
    pub(crate) routed_moe_down_expanded_grid_records: usize,
    pub(crate) max_grid_x: u64,
    pub(crate) max_grid_y: u64,
    pub(crate) max_grid_z: u64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) dispatch_shapes: Vec<DispatchShapeSummary>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) route_fusion_reasons: Vec<RouteFusionReasonSummary>,
}

impl GlmDsaDispatchSummary {
    pub(crate) fn is_empty(summary: &Self) -> bool {
        summary.records == 0
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct DispatchShapeSummary {
    pub(crate) op: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) kernel: Option<String>,
    pub(crate) tensor: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) src_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) dst_type: Option<String>,
    pub(crate) grid_x: u64,
    pub(crate) grid_y: u64,
    pub(crate) grid_z: u64,
    pub(crate) threads_x: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) threads_y: Option<u64>,
    pub(crate) records: usize,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RouteFusionReasonSummary {
    pub(crate) op: String,
    pub(crate) reason: String,
    pub(crate) records: usize,
}

pub(crate) fn summarize_metal_dispatch(records: &[MetalDispatchRecord]) -> GlmDsaDispatchSummary {
    let mut summary = GlmDsaDispatchSummary {
        records: records.len(),
        ..GlmDsaDispatchSummary::default()
    };
    let mut shapes = BTreeMap::<DispatchShapeKey, usize>::new();
    let mut route_fusion_reasons = BTreeMap::<RouteFusionReasonKey, usize>::new();

    for record in records {
        summary.max_grid_x = summary.max_grid_x.max(record.grid_x);
        summary.max_grid_y = summary.max_grid_y.max(record.grid_y);
        summary.max_grid_z = summary.max_grid_z.max(record.grid_z);

        match record.op.as_str() {
            "topk_moe_route_fused" => summary.topk_moe_route_fused_records += 1,
            "topk_moe_route_pack" => summary.topk_moe_route_pack_records += 1,
            "topk_moe_route_encode" => summary.topk_moe_route_encode_records += 1,
            "dsa_sparse_attn" => summary.dsa_sparse_attn_records += 1,
            "mul_mat_id" => summary.mul_mat_id_records += 1,
            "moe_weighted_sum" => {
                summary.moe_weighted_sum_records += 1;
                if record.kernel.as_deref() == Some("f32x4") {
                    summary.moe_weighted_sum_f32x4_records += 1;
                }
            }
            _ => {}
        }

        if is_topk_moe_route_pack_candidate(record) {
            summary.topk_moe_route_pack_candidate_records += 1;
            if record.reason.as_deref() == Some("packed") {
                summary.topk_moe_route_packed_candidate_records += 1;
            } else {
                summary.topk_moe_route_pack_skipped_candidate_records += 1;
            }
        }
        if is_topk_moe_route_encode_candidate(record) {
            summary.topk_moe_route_encode_candidate_records += 1;
            if record.reason.as_deref() == Some("fused") {
                summary.topk_moe_route_encode_fused_candidate_records += 1;
            } else {
                summary.topk_moe_route_encode_skipped_candidate_records += 1;
            }
        }

        if record.tensor.contains("ffn_moe_gate") {
            summary.routed_moe_gate_records += 1;
        }
        if record.tensor.contains("ffn_moe_up") {
            summary.routed_moe_up_records += 1;
        }
        if record.tensor.contains("ffn_moe_down") {
            summary.routed_moe_down_records += 1;
            if record.grid_x > 256 {
                summary.routed_moe_down_expanded_grid_records += 1;
            }
            if record.src_type.as_deref() == Some("q3_K") {
                summary.routed_moe_down_q3_k_records += 1;
            }
        }

        *shapes.entry(DispatchShapeKey::from(record)).or_insert(0) += 1;
        if let Some(reason) = &record.reason {
            *route_fusion_reasons
                .entry(RouteFusionReasonKey {
                    op: record.op.clone(),
                    reason: reason.clone(),
                })
                .or_insert(0) += 1;
        }
    }

    summary.dispatch_shapes = shapes
        .into_iter()
        .map(|(shape, records)| shape.into_summary(records))
        .collect();
    summary.route_fusion_reasons = route_fusion_reasons
        .into_iter()
        .map(|(reason, records)| reason.into_summary(records))
        .collect();
    summary
}

fn is_topk_moe_route_pack_candidate(record: &MetalDispatchRecord) -> bool {
    record.op == "topk_moe_route_pack" && record.reason.is_some()
}

fn is_topk_moe_route_encode_candidate(record: &MetalDispatchRecord) -> bool {
    record.op == "topk_moe_route_encode" && record.reason.is_some()
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct GlmDsaOpTimingSummary {
    pub(crate) records: usize,
    pub(crate) total_us: u64,
    pub(crate) indexer_topk: TimingBucketSummary,
    pub(crate) sparse_mask: TimingBucketSummary,
    pub(crate) dsa_sparse_attn: TimingBucketSummary,
    pub(crate) mla_attention: TimingBucketSummary,
    pub(crate) routed_moe: TimingBucketSummary,
    pub(crate) shared_expert: TimingBucketSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) dsa_sparse_attn_share_of_total: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) routed_moe_share_of_total: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) shared_expert_share_of_total: Option<f64>,
}

impl GlmDsaOpTimingSummary {
    pub(crate) fn is_empty(summary: &Self) -> bool {
        summary.records == 0
    }
}

pub(crate) fn summarize_glm_dsa_op_timing(records: &[TimingRecord]) -> GlmDsaOpTimingSummary {
    let mut summary = GlmDsaOpTimingSummary {
        records: records.len(),
        ..GlmDsaOpTimingSummary::default()
    };

    for record in records {
        summary.total_us += record.total_us;
        add_timing(
            &mut summary.indexer_topk,
            record.indexer_topk_nodes,
            record.indexer_topk_us,
        );
        add_timing(
            &mut summary.sparse_mask,
            record.sparse_mask_nodes,
            record.sparse_mask_us,
        );
        add_optional_timing(
            &mut summary.dsa_sparse_attn,
            record.dsa_sparse_attn_nodes,
            record.dsa_sparse_attn_us,
        );
        add_timing(
            &mut summary.mla_attention,
            record.mla_attention_nodes,
            record.mla_attention_us,
        );
        add_timing(
            &mut summary.routed_moe,
            record.routed_moe_nodes,
            record.routed_moe_us,
        );
        add_timing(
            &mut summary.shared_expert,
            record.shared_expert_nodes,
            record.shared_expert_us,
        );
    }

    finalize_timing_bucket(&mut summary.indexer_topk);
    finalize_timing_bucket(&mut summary.sparse_mask);
    finalize_timing_bucket(&mut summary.dsa_sparse_attn);
    finalize_timing_bucket(&mut summary.mla_attention);
    finalize_timing_bucket(&mut summary.routed_moe);
    finalize_timing_bucket(&mut summary.shared_expert);
    summary.dsa_sparse_attn_share_of_total =
        ratio(summary.dsa_sparse_attn.elapsed_us, summary.total_us);
    summary.routed_moe_share_of_total = ratio(summary.routed_moe.elapsed_us, summary.total_us);
    summary.shared_expert_share_of_total =
        ratio(summary.shared_expert.elapsed_us, summary.total_us);
    summary
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct RoutedMoeTimingSummary {
    pub(crate) records: usize,
    pub(crate) total_us: u64,
    pub(crate) routed_moe_nodes: u64,
    pub(crate) routed_moe_us: u64,
    pub(crate) route: TimingBucketSummary,
    pub(crate) gate_up: TimingBucketSummary,
    pub(crate) gate: TimingBucketSummary,
    pub(crate) up: TimingBucketSummary,
    pub(crate) activation: TimingBucketSummary,
    pub(crate) down: TimingBucketSummary,
    pub(crate) weighted: TimingBucketSummary,
    pub(crate) aggregate: TimingBucketSummary,
    pub(crate) weighted_or_aggregate: TimingBucketSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) routed_moe_share_of_total: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) down_share_of_routed_moe: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) weighted_share_of_routed_moe: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) weighted_or_aggregate_share_of_routed_moe: Option<f64>,
}

impl RoutedMoeTimingSummary {
    pub(crate) fn is_empty(summary: &Self) -> bool {
        summary.records == 0
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct TimingBucketSummary {
    pub(crate) nodes: u64,
    pub(crate) elapsed_us: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) avg_us_per_node: Option<f64>,
}

pub(crate) fn summarize_routed_moe_timing(records: &[TimingRecord]) -> RoutedMoeTimingSummary {
    let mut summary = RoutedMoeTimingSummary {
        records: records.len(),
        ..RoutedMoeTimingSummary::default()
    };

    for record in records {
        summary.total_us += record.total_us;
        summary.routed_moe_nodes += record.routed_moe_nodes;
        summary.routed_moe_us += record.routed_moe_us;
        add_optional_timing(
            &mut summary.route,
            record.routed_moe_route_nodes,
            record.routed_moe_route_us,
        );
        add_optional_timing(
            &mut summary.gate_up,
            record.routed_moe_gate_up_nodes,
            record.routed_moe_gate_up_us,
        );
        add_optional_timing(
            &mut summary.gate,
            record.routed_moe_gate_nodes,
            record.routed_moe_gate_us,
        );
        add_optional_timing(
            &mut summary.up,
            record.routed_moe_up_nodes,
            record.routed_moe_up_us,
        );
        add_optional_timing(
            &mut summary.activation,
            record.routed_moe_act_nodes,
            record.routed_moe_act_us,
        );
        add_optional_timing(
            &mut summary.down,
            record.routed_moe_down_nodes,
            record.routed_moe_down_us,
        );
        add_optional_timing(
            &mut summary.weighted,
            record.routed_moe_weighted_nodes,
            record.routed_moe_weighted_us,
        );
        add_optional_timing(
            &mut summary.aggregate,
            record.routed_moe_aggregate_nodes,
            record.routed_moe_aggregate_us,
        );
    }

    finalize_timing_bucket(&mut summary.route);
    finalize_timing_bucket(&mut summary.gate_up);
    finalize_timing_bucket(&mut summary.gate);
    finalize_timing_bucket(&mut summary.up);
    finalize_timing_bucket(&mut summary.activation);
    finalize_timing_bucket(&mut summary.down);
    finalize_timing_bucket(&mut summary.weighted);
    finalize_timing_bucket(&mut summary.aggregate);
    summary.weighted_or_aggregate = merge_timing_buckets(&summary.weighted, &summary.aggregate);
    summary.routed_moe_share_of_total = ratio(summary.routed_moe_us, summary.total_us);
    summary.down_share_of_routed_moe = ratio(summary.down.elapsed_us, summary.routed_moe_us);
    summary.weighted_share_of_routed_moe =
        ratio(summary.weighted.elapsed_us, summary.routed_moe_us);
    summary.weighted_or_aggregate_share_of_routed_moe = ratio(
        summary.weighted_or_aggregate.elapsed_us,
        summary.routed_moe_us,
    );
    summary
}

fn add_optional_timing(
    bucket: &mut TimingBucketSummary,
    nodes: Option<u64>,
    elapsed_us: Option<u64>,
) {
    add_timing(bucket, nodes.unwrap_or(0), elapsed_us.unwrap_or(0));
}

fn add_timing(bucket: &mut TimingBucketSummary, nodes: u64, elapsed_us: u64) {
    bucket.nodes += nodes;
    bucket.elapsed_us += elapsed_us;
}

fn finalize_timing_bucket(bucket: &mut TimingBucketSummary) {
    bucket.avg_us_per_node = ratio(bucket.elapsed_us, bucket.nodes);
}

fn merge_timing_buckets(
    left: &TimingBucketSummary,
    right: &TimingBucketSummary,
) -> TimingBucketSummary {
    let mut merged = TimingBucketSummary {
        nodes: left.nodes + right.nodes,
        elapsed_us: left.elapsed_us + right.elapsed_us,
        avg_us_per_node: None,
    };
    finalize_timing_bucket(&mut merged);
    merged
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DispatchShapeKey {
    op: String,
    kernel: Option<String>,
    tensor: String,
    src_type: Option<String>,
    dst_type: Option<String>,
    grid_x: u64,
    grid_y: u64,
    grid_z: u64,
    threads_x: u64,
    threads_y: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RouteFusionReasonKey {
    op: String,
    reason: String,
}

impl RouteFusionReasonKey {
    fn into_summary(self, records: usize) -> RouteFusionReasonSummary {
        RouteFusionReasonSummary {
            op: self.op,
            reason: self.reason,
            records,
        }
    }
}

impl DispatchShapeKey {
    fn from(record: &MetalDispatchRecord) -> Self {
        Self {
            op: record.op.clone(),
            kernel: record.kernel.clone(),
            tensor: record.tensor.clone(),
            src_type: record.src_type.clone(),
            dst_type: record.dst_type.clone(),
            grid_x: record.grid_x,
            grid_y: record.grid_y,
            grid_z: record.grid_z,
            threads_x: record.threads_x,
            threads_y: record.threads_y,
        }
    }

    fn into_summary(self, records: usize) -> DispatchShapeSummary {
        DispatchShapeSummary {
            op: self.op,
            kernel: self.kernel,
            tensor: self.tensor,
            src_type: self.src_type,
            dst_type: self.dst_type,
            grid_x: self.grid_x,
            grid_y: self.grid_y,
            grid_z: self.grid_z,
            threads_x: self.threads_x,
            threads_y: self.threads_y,
            records,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_summary_reports_tail_and_outliers() {
        let summary = summarize_elapsed_ms([10.0, 11.0, 9.0, 40.0]);

        assert_eq!(summary.samples, 4);
        assert_eq!(summary.min_ms, Some(9.0));
        assert_eq!(summary.max_ms, Some(40.0));
        assert_eq!(summary.slow_outlier_count, 1);
        assert!(summary.coefficient_of_variation.unwrap() > 0.5);
    }

    #[test]
    fn metal_dispatch_summary_counts_glm_dsa_shapes() {
        let mut fused_candidate =
            dispatch("topk_moe_route_encode", None, "blk.45.ffn_moe_probs", None);
        fused_candidate.reason = Some("fused".to_string());
        let mut skipped_candidate =
            dispatch("topk_moe_route_encode", None, "blk.46.ffn_moe_probs", None);
        skipped_candidate.reason = Some("shape_or_sequence".to_string());
        let records = vec![
            dispatch("topk_moe_route_fused", None, "route", None),
            fused_candidate,
            skipped_candidate,
            dispatch("moe_weighted_sum", Some("f32x4"), "weighted", None),
            dispatch(
                "mul_mat_id",
                None,
                "blk.45.ffn_moe_down.weight",
                Some("q3_K"),
            ),
            dispatch(
                "mul_mat_id",
                None,
                "blk.45.ffn_moe_down.weight",
                Some("q3_K"),
            ),
        ];

        let summary = summarize_metal_dispatch(&records);

        assert_eq!(summary.records, 6);
        assert_eq!(summary.topk_moe_route_fused_records, 1);
        assert_eq!(summary.topk_moe_route_encode_records, 2);
        assert_eq!(summary.topk_moe_route_pack_candidate_records, 0);
        assert_eq!(summary.topk_moe_route_packed_candidate_records, 0);
        assert_eq!(summary.topk_moe_route_pack_skipped_candidate_records, 0);
        assert_eq!(summary.topk_moe_route_encode_candidate_records, 2);
        assert_eq!(summary.topk_moe_route_encode_fused_candidate_records, 1);
        assert_eq!(summary.topk_moe_route_encode_skipped_candidate_records, 1);
        assert_eq!(summary.mul_mat_id_records, 2);
        assert_eq!(summary.moe_weighted_sum_f32x4_records, 1);
        assert_eq!(summary.routed_moe_down_q3_k_records, 2);
        assert_eq!(summary.routed_moe_down_expanded_grid_records, 2);
        assert_eq!(summary.route_fusion_reasons.len(), 2);
        assert_eq!(summary.route_fusion_reasons[0].reason, "fused");
        assert_eq!(summary.route_fusion_reasons[0].records, 1);
        assert_eq!(summary.route_fusion_reasons[1].reason, "shape_or_sequence");
        assert_eq!(summary.route_fusion_reasons[1].records, 1);
        assert_eq!(summary.dispatch_shapes.len(), 5);
    }

    #[test]
    fn routed_moe_timing_summary_reports_cost_split() {
        let summary = summarize_routed_moe_timing(&[
            timing_record(1_000, 600, Some((2, 200)), Some((1, 250)), Some((1, 50))),
            timing_record(2_000, 1_400, Some((2, 700)), Some((1, 350)), Some((1, 70))),
        ]);

        assert_eq!(summary.records, 2);
        assert_eq!(summary.total_us, 3_000);
        assert_eq!(summary.routed_moe_us, 2_000);
        assert_eq!(summary.down.nodes, 4);
        assert_eq!(summary.down.elapsed_us, 900);
        assert_eq!(summary.down.avg_us_per_node, Some(225.0));
        assert_eq!(summary.weighted.elapsed_us, 600);
        assert_eq!(summary.aggregate.elapsed_us, 120);
        assert_eq!(summary.weighted_or_aggregate.elapsed_us, 720);
        assert_eq!(summary.down_share_of_routed_moe, Some(0.45));
        assert_eq!(summary.weighted_share_of_routed_moe, Some(0.3));
        assert_eq!(
            summary.weighted_or_aggregate_share_of_routed_moe,
            Some(0.36)
        );
    }

    #[test]
    fn glm_dsa_op_timing_summary_reports_major_buckets() {
        let mut first = timing_record(1_000, 600, Some((2, 200)), Some((1, 250)), Some((1, 50)));
        first.dsa_sparse_attn_nodes = Some(1);
        first.dsa_sparse_attn_us = Some(150);
        first.mla_attention_nodes = 1;
        first.mla_attention_us = 100;
        first.shared_expert_nodes = 1;
        first.shared_expert_us = 50;

        let summary = summarize_glm_dsa_op_timing(&[first]);

        assert_eq!(summary.records, 1);
        assert_eq!(summary.total_us, 1_000);
        assert_eq!(summary.dsa_sparse_attn.elapsed_us, 150);
        assert_eq!(summary.routed_moe.elapsed_us, 600);
        assert_eq!(summary.shared_expert.elapsed_us, 50);
        assert_eq!(summary.dsa_sparse_attn_share_of_total, Some(0.15));
        assert_eq!(summary.routed_moe_share_of_total, Some(0.6));
    }

    fn timing_record(
        total_us: u64,
        routed_moe_us: u64,
        down: Option<(u64, u64)>,
        weighted: Option<(u64, u64)>,
        aggregate: Option<(u64, u64)>,
    ) -> TimingRecord {
        TimingRecord {
            stage: 0,
            tokens: 1,
            total_us,
            indexer_topk_nodes: 0,
            indexer_topk_us: 0,
            indexer_nodes: None,
            indexer_us: None,
            top_k_nodes: None,
            top_k_us: None,
            sparse_mask_nodes: 0,
            sparse_mask_us: 0,
            sparse_mask_fill_nodes: None,
            sparse_mask_fill_us: None,
            sparse_mask_topk_nodes: None,
            sparse_mask_topk_us: None,
            sparse_mask_add_nodes: None,
            sparse_mask_add_us: None,
            dsa_sparse_attn_nodes: None,
            dsa_sparse_attn_us: None,
            mla_attention_nodes: 0,
            mla_attention_us: 0,
            routed_moe_nodes: 0,
            routed_moe_us,
            routed_moe_route_nodes: None,
            routed_moe_route_us: None,
            routed_moe_gate_up_nodes: None,
            routed_moe_gate_up_us: None,
            routed_moe_gate_nodes: None,
            routed_moe_gate_us: None,
            routed_moe_up_nodes: None,
            routed_moe_up_us: None,
            routed_moe_act_nodes: None,
            routed_moe_act_us: None,
            routed_moe_down_nodes: down.map(|bucket| bucket.0),
            routed_moe_down_us: down.map(|bucket| bucket.1),
            routed_moe_weighted_nodes: weighted.map(|bucket| bucket.0),
            routed_moe_weighted_us: weighted.map(|bucket| bucket.1),
            routed_moe_aggregate_nodes: aggregate.map(|bucket| bucket.0),
            routed_moe_aggregate_us: aggregate.map(|bucket| bucket.1),
            shared_expert_nodes: 0,
            shared_expert_us: 0,
        }
    }

    fn dispatch(
        op: &str,
        kernel: Option<&str>,
        tensor: &str,
        src_type: Option<&str>,
    ) -> MetalDispatchRecord {
        MetalDispatchRecord {
            op: op.to_string(),
            kernel: kernel.map(str::to_string),
            tensor: tensor.to_string(),
            reason: None,
            parallel: None,
            q_type: None,
            k_type: None,
            v_type: None,
            mask_type: None,
            top_k_type: None,
            src_type: src_type.map(str::to_string),
            dst_type: None,
            q_width: None,
            v_width: None,
            batch: None,
            heads: None,
            stream: None,
            kv: None,
            top_k: None,
            top_stream: None,
            grid_x: if tensor.contains("ffn_moe_down") {
                1536
            } else {
                256
            },
            grid_y: 1,
            grid_z: 8,
            threads_x: 32,
            threads_y: Some(2),
        }
    }
}
