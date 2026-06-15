use std::collections::BTreeMap;

use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NativeMtpDraft {
    pub(super) token: i32,
    pub(super) proposal_compute_us: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingDraft {
    token: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeMtpVerification {
    NoPending,
    Accepted { draft: i32, target: i32 },
    Rejected { draft: i32, target: i32 },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct NativeMtpN1Stats {
    pub(super) drafted_tokens: u64,
    pub(super) accepted_tokens: u64,
    pub(super) rejected_tokens: u64,
    pub(super) pending_tokens: u64,
    pub(super) verification_count: u64,
    pub(super) proposal_compute_us: i64,
    pub(super) verification_compute_us: i64,
}

impl NativeMtpN1Stats {
    pub(super) fn verified_tokens(self) -> u64 {
        self.accepted_tokens + self.rejected_tokens
    }

    pub(super) fn accept_rate(self) -> f64 {
        let verified = self.verified_tokens();
        if verified == 0 {
            0.0
        } else {
            self.accepted_tokens as f64 / verified as f64
        }
    }

    pub(super) fn insert_attrs(self, attrs: &mut BTreeMap<String, Value>) {
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

impl NativeMtpVerification {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::NoPending => "none",
            Self::Accepted { .. } => "accepted",
            Self::Rejected { .. } => "rejected",
        }
    }
}

#[derive(Default)]
pub(super) struct NativeMtpN1Verifier {
    pending: Option<PendingDraft>,
    stats: NativeMtpN1Stats,
}

impl NativeMtpN1Verifier {
    pub(super) fn observe_target_token(
        &mut self,
        target_token: i32,
        verification_compute_us: i64,
        next_draft: Option<NativeMtpDraft>,
    ) -> NativeMtpVerification {
        let verification = self.verify_pending(target_token, verification_compute_us);
        self.observe_next_draft(next_draft);
        verification
    }

    pub(super) fn stats(&self) -> NativeMtpN1Stats {
        let mut stats = self.stats;
        stats.pending_tokens = u64::from(self.pending.is_some());
        stats
    }

    fn verify_pending(
        &mut self,
        target_token: i32,
        verification_compute_us: i64,
    ) -> NativeMtpVerification {
        let Some(pending) = self.pending.take() else {
            return NativeMtpVerification::NoPending;
        };

        self.stats.verification_count += 1;
        self.stats.verification_compute_us = self
            .stats
            .verification_compute_us
            .saturating_add(verification_compute_us);
        if pending.token == target_token {
            self.stats.accepted_tokens += 1;
            NativeMtpVerification::Accepted {
                draft: pending.token,
                target: target_token,
            }
        } else {
            self.stats.rejected_tokens += 1;
            NativeMtpVerification::Rejected {
                draft: pending.token,
                target: target_token,
            }
        }
    }

    fn observe_next_draft(&mut self, next_draft: Option<NativeMtpDraft>) {
        let Some(next_draft) = next_draft else {
            return;
        };
        self.stats.drafted_tokens += 1;
        self.stats.proposal_compute_us = self
            .stats
            .proposal_compute_us
            .saturating_add(next_draft.proposal_compute_us);
        self.pending = Some(PendingDraft {
            token: next_draft.token,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft(token: i32) -> NativeMtpDraft {
        NativeMtpDraft {
            token,
            proposal_compute_us: 7,
        }
    }

    #[test]
    fn no_draft_behaves_like_baseline() {
        let mut verifier = NativeMtpN1Verifier::default();

        let decision = verifier.observe_target_token(11, 5, None);

        assert_eq!(decision, NativeMtpVerification::NoPending);
        assert_eq!(verifier.stats(), NativeMtpN1Stats::default());
    }

    #[test]
    fn first_draft_is_pending_until_next_target_decode() {
        let mut verifier = NativeMtpN1Verifier::default();

        let decision = verifier.observe_target_token(11, 5, Some(draft(12)));

        assert_eq!(decision, NativeMtpVerification::NoPending);
        assert_eq!(
            verifier.stats(),
            NativeMtpN1Stats {
                drafted_tokens: 1,
                pending_tokens: 1,
                proposal_compute_us: 7,
                ..NativeMtpN1Stats::default()
            }
        );
    }

    #[test]
    fn matching_next_target_accepts_pending_draft() {
        let mut verifier = NativeMtpN1Verifier::default();
        verifier.observe_target_token(11, 5, Some(draft(12)));

        let decision = verifier.observe_target_token(12, 9, None);

        assert_eq!(
            decision,
            NativeMtpVerification::Accepted {
                draft: 12,
                target: 12,
            }
        );
        assert_eq!(
            verifier.stats(),
            NativeMtpN1Stats {
                drafted_tokens: 1,
                accepted_tokens: 1,
                verification_count: 1,
                proposal_compute_us: 7,
                verification_compute_us: 9,
                ..NativeMtpN1Stats::default()
            }
        );
    }

    #[test]
    fn different_next_target_rejects_pending_draft() {
        let mut verifier = NativeMtpN1Verifier::default();
        verifier.observe_target_token(11, 5, Some(draft(12)));

        let decision = verifier.observe_target_token(13, 9, None);

        assert_eq!(
            decision,
            NativeMtpVerification::Rejected {
                draft: 12,
                target: 13,
            }
        );
        assert_eq!(
            verifier.stats(),
            NativeMtpN1Stats {
                drafted_tokens: 1,
                rejected_tokens: 1,
                verification_count: 1,
                proposal_compute_us: 7,
                verification_compute_us: 9,
                ..NativeMtpN1Stats::default()
            }
        );
    }

    #[test]
    fn verifies_previous_draft_before_storing_next_draft() {
        let mut verifier = NativeMtpN1Verifier::default();
        verifier.observe_target_token(11, 5, Some(draft(12)));

        let decision = verifier.observe_target_token(12, 9, Some(draft(14)));

        assert_eq!(
            decision,
            NativeMtpVerification::Accepted {
                draft: 12,
                target: 12,
            }
        );
        assert_eq!(
            verifier.stats(),
            NativeMtpN1Stats {
                drafted_tokens: 2,
                accepted_tokens: 1,
                pending_tokens: 1,
                verification_count: 1,
                proposal_compute_us: 14,
                verification_compute_us: 9,
                ..NativeMtpN1Stats::default()
            }
        );
    }

    #[test]
    fn attrs_include_disabled_and_enabled_shapes() {
        let mut attrs = BTreeMap::new();
        NativeMtpN1Stats::default().insert_attrs(&mut attrs);
        assert_eq!(
            attrs.get("llama_stage.native_mtp.enabled"),
            Some(&json!(false))
        );

        let mut verifier = NativeMtpN1Verifier::default();
        verifier.observe_target_token(11, 5, Some(draft(12)));
        verifier.observe_target_token(12, 9, None);

        let mut attrs = BTreeMap::new();
        verifier.stats().insert_attrs(&mut attrs);
        assert_eq!(
            attrs.get("llama_stage.native_mtp.enabled"),
            Some(&json!(true))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.accept_rate"),
            Some(&json!(1.0))
        );
    }
}
