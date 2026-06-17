use std::collections::BTreeMap;

use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::frontend) enum NativeMtpVerification {
    #[default]
    NoPending,
    Accepted {
        draft: i32,
        target: i32,
    },
    Rejected {
        draft: i32,
        target: i32,
    },
}

impl NativeMtpVerification {
    pub(in crate::frontend) fn label(self) -> &'static str {
        match self {
            Self::NoPending => "none",
            Self::Accepted { .. } => "accepted",
            Self::Rejected { .. } => "rejected",
        }
    }

    fn accepted(self) -> bool {
        matches!(self, Self::Accepted { .. })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::frontend) struct NativeMtpN1Stats {
    pub(in crate::frontend) drafted_tokens: u64,
    pub(in crate::frontend) accepted_tokens: u64,
    pub(in crate::frontend) rejected_tokens: u64,
    pub(in crate::frontend) pending_tokens: u64,
    pub(in crate::frontend) verification_count: u64,
    pub(in crate::frontend) proposal_compute_us: i64,
    pub(in crate::frontend) verification_compute_us: i64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(in crate::frontend) struct NativeMtpBatchedTimingSample {
    pub(in crate::frontend) verification: NativeMtpVerification,
    pub(in crate::frontend) verify_elapsed_ms: f64,
    pub(in crate::frontend) stage0_compute_ms: f64,
    pub(in crate::frontend) runtime_lock_wait_ms: f64,
    pub(in crate::frontend) runtime_lock_hold_ms: f64,
    pub(in crate::frontend) activation_encode_ms: f64,
    pub(in crate::frontend) forward_write_ms: f64,
    pub(in crate::frontend) downstream_wait_ms: f64,
    pub(in crate::frontend) trim_elapsed_ms: f64,
    pub(in crate::frontend) trim_local_ms: f64,
    pub(in crate::frontend) trim_downstream_write_ms: f64,
    pub(in crate::frontend) trim_downstream_wait_ms: f64,
    pub(in crate::frontend) consumed_positions: usize,
    pub(in crate::frontend) committed_positions: usize,
    pub(in crate::frontend) verify_next_proposal_compute_us: Option<i64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(in crate::frontend) struct NativeMtpBatchedTimingStats {
    accepted: NativeMtpBatchedPathTimingStats,
    rejected: NativeMtpBatchedPathTimingStats,
    consumed_positions: u64,
    committed_positions: u64,
    trim_count: u64,
    trim_elapsed_ms: f64,
    trim_local_ms: f64,
    trim_downstream_write_ms: f64,
    trim_downstream_wait_ms: f64,
    verify_next_proposal_count: u64,
    verify_next_proposal_compute_us: i64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(in crate::frontend) struct NativeMtpMarginOutcomeStats {
    accepted: NativeMtpMarginPathStats,
    rejected: NativeMtpMarginPathStats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct NativeMtpBatchedPathTimingStats {
    count: u64,
    verify_elapsed_ms: f64,
    stage0_compute_ms: f64,
    runtime_lock_wait_ms: f64,
    runtime_lock_hold_ms: f64,
    activation_encode_ms: f64,
    forward_write_ms: f64,
    downstream_wait_ms: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct NativeMtpMarginPathStats {
    count: u64,
    sum: f64,
    min: f32,
    max: f32,
}

impl NativeMtpBatchedTimingStats {
    pub(in crate::frontend) fn record(&mut self, sample: NativeMtpBatchedTimingSample) {
        if sample.verification == NativeMtpVerification::NoPending {
            return;
        }
        if sample.verification.accepted() {
            self.accepted.record(sample);
        } else {
            self.rejected.record(sample);
        }
        self.consumed_positions = self
            .consumed_positions
            .saturating_add(sample.consumed_positions as u64);
        self.committed_positions = self
            .committed_positions
            .saturating_add(sample.committed_positions as u64);
        if sample.trim_elapsed_ms > 0.0
            || sample.trim_local_ms > 0.0
            || sample.trim_downstream_write_ms > 0.0
            || sample.trim_downstream_wait_ms > 0.0
        {
            self.trim_count = self.trim_count.saturating_add(1);
            self.trim_elapsed_ms += sample.trim_elapsed_ms;
            self.trim_local_ms += sample.trim_local_ms;
            self.trim_downstream_write_ms += sample.trim_downstream_write_ms;
            self.trim_downstream_wait_ms += sample.trim_downstream_wait_ms;
        }
        if let Some(proposal_compute_us) = sample.verify_next_proposal_compute_us {
            self.verify_next_proposal_count = self.verify_next_proposal_count.saturating_add(1);
            self.verify_next_proposal_compute_us = self
                .verify_next_proposal_compute_us
                .saturating_add(proposal_compute_us.max(0));
        }
    }

    pub(in crate::frontend) fn insert_attrs(self, attrs: &mut BTreeMap<String, Value>) {
        self.accepted
            .insert_attrs(attrs, "llama_stage.native_mtp.batched.accepted");
        self.rejected
            .insert_attrs(attrs, "llama_stage.native_mtp.batched.rejected");
        attrs.insert(
            "llama_stage.native_mtp.batched.consumed_positions".to_string(),
            json!(self.consumed_positions),
        );
        attrs.insert(
            "llama_stage.native_mtp.batched.committed_positions".to_string(),
            json!(self.committed_positions),
        );
        attrs.insert(
            "llama_stage.native_mtp.batched.trim_count".to_string(),
            json!(self.trim_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.batched.trim_elapsed_ms".to_string(),
            json!(self.trim_elapsed_ms),
        );
        attrs.insert(
            "llama_stage.native_mtp.batched.trim_local_ms".to_string(),
            json!(self.trim_local_ms),
        );
        attrs.insert(
            "llama_stage.native_mtp.batched.trim_downstream_write_ms".to_string(),
            json!(self.trim_downstream_write_ms),
        );
        attrs.insert(
            "llama_stage.native_mtp.batched.trim_downstream_wait_ms".to_string(),
            json!(self.trim_downstream_wait_ms),
        );
        attrs.insert(
            "llama_stage.native_mtp.batched.verify_next_proposal_count".to_string(),
            json!(self.verify_next_proposal_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.batched.verify_next_proposal_compute_us".to_string(),
            json!(self.verify_next_proposal_compute_us),
        );
        if self.verify_next_proposal_count > 0 {
            attrs.insert(
                "llama_stage.native_mtp.batched.verify_next_proposal_compute_us_avg".to_string(),
                json!(
                    self.verify_next_proposal_compute_us as f64
                        / self.verify_next_proposal_count as f64
                ),
            );
        }
    }
}

impl NativeMtpMarginOutcomeStats {
    pub(in crate::frontend) fn record(
        &mut self,
        margin: Option<f32>,
        verification: NativeMtpVerification,
    ) {
        let Some(margin) = margin else {
            return;
        };
        if !margin.is_finite() || verification == NativeMtpVerification::NoPending {
            return;
        }
        if verification.accepted() {
            self.accepted.record(margin);
        } else {
            self.rejected.record(margin);
        }
    }

    pub(in crate::frontend) fn insert_attrs(
        self,
        attrs: &mut BTreeMap<String, Value>,
        prefix: &str,
    ) {
        self.accepted
            .insert_attrs(attrs, &format!("{prefix}.accepted"));
        self.rejected
            .insert_attrs(attrs, &format!("{prefix}.rejected"));
    }
}

impl NativeMtpBatchedPathTimingStats {
    fn record(&mut self, sample: NativeMtpBatchedTimingSample) {
        self.count = self.count.saturating_add(1);
        self.verify_elapsed_ms += sample.verify_elapsed_ms;
        self.stage0_compute_ms += sample.stage0_compute_ms;
        self.runtime_lock_wait_ms += sample.runtime_lock_wait_ms;
        self.runtime_lock_hold_ms += sample.runtime_lock_hold_ms;
        self.activation_encode_ms += sample.activation_encode_ms;
        self.forward_write_ms += sample.forward_write_ms;
        self.downstream_wait_ms += sample.downstream_wait_ms;
    }

    fn insert_attrs(self, attrs: &mut BTreeMap<String, Value>, prefix: &str) {
        attrs.insert(format!("{prefix}_count"), json!(self.count));
        attrs.insert(
            format!("{prefix}_verify_elapsed_ms"),
            json!(self.verify_elapsed_ms),
        );
        attrs.insert(
            format!("{prefix}_stage0_compute_ms"),
            json!(self.stage0_compute_ms),
        );
        attrs.insert(
            format!("{prefix}_runtime_lock_wait_ms"),
            json!(self.runtime_lock_wait_ms),
        );
        attrs.insert(
            format!("{prefix}_runtime_lock_hold_ms"),
            json!(self.runtime_lock_hold_ms),
        );
        attrs.insert(
            format!("{prefix}_activation_encode_ms"),
            json!(self.activation_encode_ms),
        );
        attrs.insert(
            format!("{prefix}_forward_write_ms"),
            json!(self.forward_write_ms),
        );
        attrs.insert(
            format!("{prefix}_downstream_wait_ms"),
            json!(self.downstream_wait_ms),
        );
        if self.count > 0 {
            attrs.insert(
                format!("{prefix}_verify_elapsed_avg_ms"),
                json!(self.verify_elapsed_ms / self.count as f64),
            );
        }
    }
}

impl NativeMtpMarginPathStats {
    fn record(&mut self, margin: f32) {
        self.count = self.count.saturating_add(1);
        self.sum += f64::from(margin);
        if self.count == 1 {
            self.min = margin;
            self.max = margin;
        } else {
            self.min = self.min.min(margin);
            self.max = self.max.max(margin);
        }
    }

    fn insert_attrs(self, attrs: &mut BTreeMap<String, Value>, prefix: &str) {
        attrs.insert(format!("{prefix}_count"), json!(self.count));
        if self.count == 0 {
            return;
        }
        attrs.insert(format!("{prefix}_avg"), json!(self.sum / self.count as f64));
        attrs.insert(format!("{prefix}_min"), json!(self.min));
        attrs.insert(format!("{prefix}_max"), json!(self.max));
    }
}

impl NativeMtpN1Stats {
    pub(in crate::frontend) fn verified_tokens(self) -> u64 {
        self.accepted_tokens + self.rejected_tokens
    }

    pub(in crate::frontend) fn accept_rate(self) -> f64 {
        let verified = self.verified_tokens();
        if verified == 0 {
            0.0
        } else {
            self.accepted_tokens as f64 / verified as f64
        }
    }

    pub(in crate::frontend) fn insert_attrs(self, attrs: &mut BTreeMap<String, Value>) {
        if self.drafted_tokens == 0 && self.verified_tokens() == 0 {
            attrs.insert("llama_stage.native_mtp.enabled".to_string(), json!(false));
            return;
        }

        attrs.insert("llama_stage.native_mtp.enabled".to_string(), json!(true));
        attrs.insert(
            "llama_stage.native_mtp.drafted".to_string(),
            json!(self.drafted_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.accepted".to_string(),
            json!(self.accepted_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.rejected".to_string(),
            json!(self.rejected_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.pending".to_string(),
            json!(self.pending_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.accept_rate".to_string(),
            json!(self.accept_rate()),
        );
        attrs.insert(
            "llama_stage.native_mtp.proposal_compute_us".to_string(),
            json!(self.proposal_compute_us),
        );
        attrs.insert(
            "llama_stage.native_mtp.verification_compute_us".to_string(),
            json!(self.verification_compute_us),
        );
        attrs.insert(
            "llama_stage.native_mtp.verifications".to_string(),
            json!(self.verification_count),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attrs_include_disabled_and_enabled_shapes() {
        let mut attrs = BTreeMap::new();
        NativeMtpN1Stats::default().insert_attrs(&mut attrs);
        assert_eq!(
            attrs.get("llama_stage.native_mtp.enabled"),
            Some(&json!(false))
        );

        let stats = NativeMtpN1Stats {
            drafted_tokens: 1,
            accepted_tokens: 1,
            verification_count: 1,
            proposal_compute_us: 7,
            verification_compute_us: 9,
            ..NativeMtpN1Stats::default()
        };

        let mut attrs = BTreeMap::new();
        stats.insert_attrs(&mut attrs);
        assert_eq!(
            attrs.get("llama_stage.native_mtp.enabled"),
            Some(&json!(true))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.accept_rate"),
            Some(&json!(1.0))
        );
    }

    #[test]
    fn verification_labels_match_telemetry_values() {
        assert_eq!(NativeMtpVerification::NoPending.label(), "none");
        assert_eq!(
            NativeMtpVerification::Accepted {
                draft: 1,
                target: 1
            }
            .label(),
            "accepted"
        );
        assert_eq!(
            NativeMtpVerification::Rejected {
                draft: 1,
                target: 2
            }
            .label(),
            "rejected"
        );
    }

    #[test]
    fn batched_timing_stats_split_accepted_and_rejected_paths() {
        let mut stats = NativeMtpBatchedTimingStats::default();
        stats.record(NativeMtpBatchedTimingSample {
            verification: NativeMtpVerification::Accepted {
                draft: 1,
                target: 1,
            },
            verify_elapsed_ms: 10.0,
            stage0_compute_ms: 4.0,
            downstream_wait_ms: 3.0,
            consumed_positions: 2,
            committed_positions: 2,
            verify_next_proposal_compute_us: Some(11),
            ..NativeMtpBatchedTimingSample::default()
        });
        stats.record(NativeMtpBatchedTimingSample {
            verification: NativeMtpVerification::Rejected {
                draft: 2,
                target: 3,
            },
            verify_elapsed_ms: 20.0,
            trim_elapsed_ms: 5.0,
            trim_downstream_wait_ms: 2.0,
            consumed_positions: 2,
            committed_positions: 1,
            verify_next_proposal_compute_us: Some(13),
            ..NativeMtpBatchedTimingSample::default()
        });

        let mut attrs = BTreeMap::new();
        stats.insert_attrs(&mut attrs);

        assert_eq!(
            attrs.get("llama_stage.native_mtp.batched.accepted_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.batched.rejected_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.batched.accepted_verify_elapsed_ms"),
            Some(&json!(10.0))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.batched.rejected_verify_elapsed_ms"),
            Some(&json!(20.0))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.batched.trim_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.batched.consumed_positions"),
            Some(&json!(4))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.batched.committed_positions"),
            Some(&json!(3))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.batched.verify_next_proposal_count"),
            Some(&json!(2))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.batched.verify_next_proposal_compute_us"),
            Some(&json!(24))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.batched.verify_next_proposal_compute_us_avg"),
            Some(&json!(12.0))
        );
    }

    #[test]
    fn margin_outcome_stats_split_actual_verification_results() {
        let mut stats = NativeMtpMarginOutcomeStats::default();
        stats.record(
            Some(0.25),
            NativeMtpVerification::Accepted {
                draft: 1,
                target: 1,
            },
        );
        stats.record(
            Some(1.25),
            NativeMtpVerification::Accepted {
                draft: 2,
                target: 2,
            },
        );
        stats.record(
            Some(0.75),
            NativeMtpVerification::Rejected {
                draft: 3,
                target: 4,
            },
        );
        stats.record(
            None,
            NativeMtpVerification::Rejected {
                draft: 5,
                target: 6,
            },
        );
        stats.record(Some(9.0), NativeMtpVerification::NoPending);

        let mut attrs = BTreeMap::new();
        stats.insert_attrs(&mut attrs, "llama_stage.native_mtp.test_margin");

        assert_eq!(
            attrs.get("llama_stage.native_mtp.test_margin.accepted_count"),
            Some(&json!(2))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.test_margin.accepted_avg"),
            Some(&json!(0.75))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.test_margin.accepted_min"),
            Some(&json!(0.25_f32))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.test_margin.accepted_max"),
            Some(&json!(1.25_f32))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.test_margin.rejected_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.test_margin.rejected_avg"),
            Some(&json!(0.75))
        );
    }
}
