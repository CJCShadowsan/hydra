use std::collections::BTreeMap;

use serde_json::{Value, json};

const BATCHED_VERIFY_ENV: &str = "SKIPPY_NATIVE_MTP_BATCHED_VERIFY";
const SERIAL_STAGE0_VERIFY_ENV: &str = "SKIPPY_NATIVE_MTP_SERIAL_STAGE0_VERIFY";
const COMPARE_STAGE0_VERIFY_ENV: &str = "SKIPPY_NATIVE_MTP_COMPARE_STAGE0_VERIFY";
const ADAPTIVE_DISABLE_ENV: &str = "SKIPPY_NATIVE_MTP_ADAPTIVE_DISABLE";
const ADAPTIVE_DISABLE_MIN_VERIFY_ENV: &str = "SKIPPY_NATIVE_MTP_ADAPTIVE_DISABLE_MIN_VERIFY";
const ADAPTIVE_DISABLE_THRESHOLD_ENV: &str = "SKIPPY_NATIVE_MTP_ADAPTIVE_DISABLE_THRESHOLD";
const REJECT_COOLDOWN_TOKENS_ENV: &str = "SKIPPY_NATIVE_MTP_REJECT_COOLDOWN_TOKENS";
const REJECT_RECOVERY_SERIAL_ACCEPTS_ENV: &str = "SKIPPY_NATIVE_MTP_REJECT_RECOVERY_SERIAL_ACCEPTS";
const SERIAL_AFTER_GAP_REJECT_RECOVERY_SERIAL_ACCEPTS_ENV: &str =
    "SKIPPY_NATIVE_MTP_SERIAL_AFTER_GAP_REJECT_RECOVERY_SERIAL_ACCEPTS";
const VERIFY_NEXT_DRAFT_MIN_MARGIN_ENV: &str = "SKIPPY_NATIVE_MTP_VERIFY_NEXT_DRAFT_MIN_MARGIN";
const DEFER_REJECT_TRIM_ENV: &str = "SKIPPY_NATIVE_MTP_DEFER_REJECT_TRIM";
const SUPPRESS_COOLDOWN_DRAFTS_ENV: &str = "SKIPPY_NATIVE_MTP_SUPPRESS_COOLDOWN_DRAFTS";
const SUPPRESS_COOLDOWN_DRAFT_LIMIT_ENV: &str = "SKIPPY_NATIVE_MTP_SUPPRESS_COOLDOWN_DRAFT_LIMIT";
const MTP_DRAFT_MARGIN_SCALE: f32 = 1000.0;
const DEFAULT_ADAPTIVE_DISABLE_MIN_VERIFY: u64 = 32;
const DEFAULT_ADAPTIVE_DISABLE_THRESHOLD: f64 = 0.70;

pub(super) fn native_mtp_batched_verify_enabled() -> bool {
    native_mtp_batched_verify_enabled_from(std::env::var(BATCHED_VERIFY_ENV).ok().as_deref())
}

pub(super) fn native_mtp_serial_stage0_verify_enabled() -> bool {
    native_mtp_serial_stage0_verify_enabled_from(
        std::env::var(SERIAL_STAGE0_VERIFY_ENV).ok().as_deref(),
    )
}

pub(super) fn native_mtp_compare_stage0_verify_enabled() -> bool {
    native_mtp_compare_stage0_verify_enabled_from(
        std::env::var(COMPARE_STAGE0_VERIFY_ENV).ok().as_deref(),
    )
}

pub(super) fn native_mtp_adaptive_disable_config() -> NativeMtpAdaptiveDisableConfig {
    NativeMtpAdaptiveDisableConfig {
        enabled: native_mtp_adaptive_disable_enabled_from(
            std::env::var(ADAPTIVE_DISABLE_ENV).ok().as_deref(),
        ),
        min_verifications: parse_u64_env(
            ADAPTIVE_DISABLE_MIN_VERIFY_ENV,
            DEFAULT_ADAPTIVE_DISABLE_MIN_VERIFY,
        ),
        threshold: parse_threshold_env(
            ADAPTIVE_DISABLE_THRESHOLD_ENV,
            DEFAULT_ADAPTIVE_DISABLE_THRESHOLD,
        ),
    }
}

pub(super) fn native_mtp_reject_cooldown_tokens() -> usize {
    parse_usize_env(REJECT_COOLDOWN_TOKENS_ENV, 0)
}

pub(super) fn native_mtp_reject_recovery_serial_accepts() -> usize {
    parse_usize_env(REJECT_RECOVERY_SERIAL_ACCEPTS_ENV, 0)
}

pub(super) fn native_mtp_serial_after_gap_reject_recovery_serial_accepts() -> usize {
    parse_usize_env(SERIAL_AFTER_GAP_REJECT_RECOVERY_SERIAL_ACCEPTS_ENV, 0)
}

pub(super) fn native_mtp_verify_next_draft_min_margin() -> Option<f32> {
    parse_optional_f32_env(VERIFY_NEXT_DRAFT_MIN_MARGIN_ENV)
}

pub(super) fn native_mtp_defer_reject_trim_enabled() -> bool {
    native_mtp_defer_reject_trim_enabled_from(std::env::var(DEFER_REJECT_TRIM_ENV).ok().as_deref())
}

pub(super) fn native_mtp_suppress_cooldown_drafts_enabled() -> bool {
    native_mtp_suppress_cooldown_drafts_enabled_from(
        std::env::var(SUPPRESS_COOLDOWN_DRAFTS_ENV).ok().as_deref(),
    )
}

pub(super) fn native_mtp_suppress_cooldown_draft_limit() -> usize {
    parse_usize_env(SUPPRESS_COOLDOWN_DRAFT_LIMIT_ENV, 0)
}

fn native_mtp_batched_verify_enabled_from(value: Option<&str>) -> bool {
    !matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("0" | "false" | "off" | "disable" | "disabled" | "no")
    )
}

fn native_mtp_serial_stage0_verify_enabled_from(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "on" | "enable" | "enabled" | "yes")
    )
}

fn native_mtp_compare_stage0_verify_enabled_from(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "on" | "enable" | "enabled" | "yes")
    )
}

fn native_mtp_adaptive_disable_enabled_from(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "on" | "enable" | "enabled" | "yes")
    )
}

fn native_mtp_defer_reject_trim_enabled_from(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "on" | "enable" | "enabled" | "yes")
    )
}

fn native_mtp_suppress_cooldown_drafts_enabled_from(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "on" | "enable" | "enabled" | "yes")
    )
}

fn parse_u64_env(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn parse_usize_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_threshold_env(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| (0.0..=1.0).contains(value))
        .unwrap_or(default)
}

fn parse_optional_f32_env(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NativeMtpDraft {
    pub(super) token: i32,
    pub(super) proposal_compute_us: i64,
    pub(super) margin_milli: Option<i32>,
}

impl NativeMtpDraft {
    pub(super) fn from_prediction_tokens(tokens: &[i32]) -> Option<Self> {
        let token = *tokens.get(1)?;
        let proposal_compute_us = tokens.get(2).copied().unwrap_or_default();
        Some(Self {
            token,
            proposal_compute_us: i64::from(proposal_compute_us.max(0)),
            margin_milli: None,
        })
    }

    pub(super) fn from_verify_prediction_tokens(
        tokens: &[i32],
        verified_token_count: usize,
    ) -> Option<Self> {
        let token = *tokens.get(verified_token_count)?;
        let proposal_compute_us = tokens
            .get(verified_token_count.saturating_add(1))
            .copied()
            .unwrap_or_default();
        Some(Self {
            token,
            proposal_compute_us: i64::from(proposal_compute_us.max(0)),
            margin_milli: tokens.get(verified_token_count.saturating_add(2)).copied(),
        })
    }

    pub(super) fn margin(&self) -> Option<f32> {
        self.margin_milli
            .map(|margin| margin as f32 / MTP_DRAFT_MARGIN_SCALE)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingDraft {
    token: i32,
    origin: NativeMtpDraftOrigin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PendingNativeMtpDraft {
    pub(super) token: i32,
    pub(super) origin: NativeMtpDraftOrigin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeMtpDraftOrigin {
    InitialSerial,
    SerialAfterGap,
    VerifyNext,
}

impl NativeMtpDraftOrigin {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::InitialSerial => "initial_serial",
            Self::SerialAfterGap => "serial_after_gap",
            Self::VerifyNext => "verify_next",
        }
    }
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct NativeMtpAdaptiveDisableConfig {
    pub(super) enabled: bool,
    pub(super) min_verifications: u64,
    pub(super) threshold: f64,
}

#[derive(Debug)]
pub(super) struct NativeMtpAdaptiveDisable {
    config: NativeMtpAdaptiveDisableConfig,
    accepted: u64,
    rejected: u64,
    disabled_at_verification: Option<u64>,
}

impl NativeMtpAdaptiveDisable {
    pub(super) fn new(config: NativeMtpAdaptiveDisableConfig) -> Self {
        Self {
            config,
            accepted: 0,
            rejected: 0,
            disabled_at_verification: None,
        }
    }

    pub(super) fn observe(&mut self, verification: NativeMtpVerification) -> bool {
        if !self.config.enabled || self.disabled() {
            return false;
        }
        match verification {
            NativeMtpVerification::Accepted { .. } => self.accepted += 1,
            NativeMtpVerification::Rejected { .. } => self.rejected += 1,
            NativeMtpVerification::NoPending => return false,
        }
        let verified = self.verified();
        if verified >= self.config.min_verifications && self.accept_rate() < self.config.threshold {
            self.disabled_at_verification = Some(verified);
            return true;
        }
        false
    }

    pub(super) fn disabled(&self) -> bool {
        self.disabled_at_verification.is_some()
    }

    pub(super) fn insert_attrs(&self, attrs: &mut BTreeMap<String, Value>) {
        attrs.insert(
            "llama_stage.native_mtp.adaptive_disable.enabled".to_string(),
            json!(self.config.enabled),
        );
        attrs.insert(
            "llama_stage.native_mtp.adaptive_disable.disabled".to_string(),
            json!(self.disabled()),
        );
        attrs.insert(
            "llama_stage.native_mtp.adaptive_disable.min_verifications".to_string(),
            json!(self.config.min_verifications),
        );
        attrs.insert(
            "llama_stage.native_mtp.adaptive_disable.threshold".to_string(),
            json!(self.config.threshold),
        );
        attrs.insert(
            "llama_stage.native_mtp.adaptive_disable.accept_rate".to_string(),
            json!(self.accept_rate()),
        );
        if let Some(disabled_at) = self.disabled_at_verification {
            attrs.insert(
                "llama_stage.native_mtp.adaptive_disable.disabled_at_verification".to_string(),
                json!(disabled_at),
            );
        }
    }

    fn verified(&self) -> u64 {
        self.accepted + self.rejected
    }

    fn accept_rate(&self) -> f64 {
        let verified = self.verified();
        if verified == 0 {
            0.0
        } else {
            self.accepted as f64 / verified as f64
        }
    }
}

#[derive(Default)]
pub(super) struct NativeMtpN1Verifier {
    pending: Option<PendingDraft>,
    stats: NativeMtpN1Stats,
}

impl NativeMtpN1Verifier {
    pub(super) fn take_pending_draft(&mut self) -> Option<PendingNativeMtpDraft> {
        self.pending.take().map(|pending| PendingNativeMtpDraft {
            token: pending.token,
            origin: pending.origin,
        })
    }

    pub(super) fn clear_pending_draft(&mut self) {
        self.pending = None;
    }

    pub(super) fn observe_taken_draft_verification(
        &mut self,
        draft_token: i32,
        target_token: i32,
        verification_compute_us: i64,
    ) -> NativeMtpVerification {
        self.record_verification(draft_token, target_token, verification_compute_us)
    }

    pub(super) fn observe_target_token(
        &mut self,
        target_token: i32,
        verification_compute_us: i64,
        next_draft: Option<NativeMtpDraft>,
        next_draft_origin: NativeMtpDraftOrigin,
    ) -> NativeMtpVerification {
        let verification = self.verify_pending(target_token, verification_compute_us);
        self.observe_next_draft(next_draft, next_draft_origin);
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

        self.record_verification(pending.token, target_token, verification_compute_us)
    }

    fn record_verification(
        &mut self,
        draft_token: i32,
        target_token: i32,
        verification_compute_us: i64,
    ) -> NativeMtpVerification {
        self.stats.verification_count += 1;
        self.stats.verification_compute_us = self
            .stats
            .verification_compute_us
            .saturating_add(verification_compute_us);
        if draft_token == target_token {
            self.stats.accepted_tokens += 1;
            NativeMtpVerification::Accepted {
                draft: draft_token,
                target: target_token,
            }
        } else {
            self.stats.rejected_tokens += 1;
            NativeMtpVerification::Rejected {
                draft: draft_token,
                target: target_token,
            }
        }
    }

    pub(super) fn observe_next_draft(
        &mut self,
        next_draft: Option<NativeMtpDraft>,
        origin: NativeMtpDraftOrigin,
    ) {
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
            origin,
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
            margin_milli: None,
        }
    }

    fn observe(
        verifier: &mut NativeMtpN1Verifier,
        target_token: i32,
        verification_compute_us: i64,
        next_draft: Option<NativeMtpDraft>,
    ) -> NativeMtpVerification {
        verifier.observe_target_token(
            target_token,
            verification_compute_us,
            next_draft,
            NativeMtpDraftOrigin::InitialSerial,
        )
    }

    #[test]
    fn parses_prediction_token_sideband() {
        assert_eq!(
            NativeMtpDraft::from_prediction_tokens(&[11, 12, 34]),
            Some(NativeMtpDraft {
                token: 12,
                proposal_compute_us: 34,
                margin_milli: None,
            })
        );
        assert_eq!(NativeMtpDraft::from_prediction_tokens(&[11]), None);
    }

    #[test]
    fn parses_verify_prediction_token_sideband_after_verified_tokens() {
        assert_eq!(
            NativeMtpDraft::from_verify_prediction_tokens(&[10, 11, 12, 34], 2),
            Some(NativeMtpDraft {
                token: 12,
                proposal_compute_us: 34,
                margin_milli: None,
            })
        );
        assert_eq!(
            NativeMtpDraft::from_verify_prediction_tokens(&[10, 11, 12, -3], 2),
            Some(NativeMtpDraft {
                token: 12,
                proposal_compute_us: 0,
                margin_milli: None,
            })
        );
        assert_eq!(
            NativeMtpDraft::from_verify_prediction_tokens(&[10, 11, 12, 34, 567], 2),
            Some(NativeMtpDraft {
                token: 12,
                proposal_compute_us: 34,
                margin_milli: Some(567),
            })
        );
        let margin = NativeMtpDraft::from_verify_prediction_tokens(&[10, 11, 12, 34, 567], 2)
            .and_then(|draft| draft.margin())
            .expect("margin sideband");
        assert!((margin - 0.567).abs() < 0.000_001);
        assert_eq!(
            NativeMtpDraft::from_verify_prediction_tokens(&[10, 11], 2),
            None
        );
    }

    #[test]
    fn no_draft_behaves_like_baseline() {
        let mut verifier = NativeMtpN1Verifier::default();

        let decision = observe(&mut verifier, 11, 5, None);

        assert_eq!(decision, NativeMtpVerification::NoPending);
        assert_eq!(verifier.stats(), NativeMtpN1Stats::default());
    }

    #[test]
    fn first_draft_is_pending_until_next_target_decode() {
        let mut verifier = NativeMtpN1Verifier::default();

        let decision = observe(&mut verifier, 11, 5, Some(draft(12)));

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
        observe(&mut verifier, 11, 5, Some(draft(12)));

        let decision = observe(&mut verifier, 12, 9, None);

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
        observe(&mut verifier, 11, 5, Some(draft(12)));

        let decision = observe(&mut verifier, 13, 9, None);

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
        observe(&mut verifier, 11, 5, Some(draft(12)));

        let decision = observe(&mut verifier, 12, 9, Some(draft(14)));

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
    fn taken_pending_draft_can_be_recorded_as_batched_accept() {
        let mut verifier = NativeMtpN1Verifier::default();
        observe(&mut verifier, 11, 5, Some(draft(12)));

        let pending = verifier.take_pending_draft();
        let pending = pending.unwrap();
        assert_eq!(pending.origin, NativeMtpDraftOrigin::InitialSerial);
        let decision = verifier.observe_taken_draft_verification(pending.token, 12, 9);

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
    fn taken_pending_draft_can_be_recorded_as_batched_reject() {
        let mut verifier = NativeMtpN1Verifier::default();
        observe(&mut verifier, 11, 5, Some(draft(12)));

        let pending = verifier.take_pending_draft();
        let pending = pending.unwrap();
        assert_eq!(pending.origin, NativeMtpDraftOrigin::InitialSerial);
        let decision = verifier.observe_taken_draft_verification(pending.token, 13, 9);

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
    fn clear_pending_draft_drops_unverified_draft_without_changing_stats() {
        let mut verifier = NativeMtpN1Verifier::default();
        observe(&mut verifier, 11, 5, Some(draft(12)));

        verifier.clear_pending_draft();

        assert_eq!(
            verifier.stats(),
            NativeMtpN1Stats {
                drafted_tokens: 1,
                proposal_compute_us: 7,
                ..NativeMtpN1Stats::default()
            }
        );
        assert_eq!(
            observe(&mut verifier, 12, 9, None),
            NativeMtpVerification::NoPending
        );
    }

    #[test]
    fn adaptive_disable_triggers_after_minimum_low_acceptance_window() {
        let mut adaptive = NativeMtpAdaptiveDisable::new(NativeMtpAdaptiveDisableConfig {
            enabled: true,
            min_verifications: 4,
            threshold: 0.75,
        });

        assert!(!adaptive.observe(NativeMtpVerification::Accepted {
            draft: 1,
            target: 1,
        }));
        assert!(!adaptive.observe(NativeMtpVerification::Rejected {
            draft: 2,
            target: 3,
        }));
        assert!(!adaptive.observe(NativeMtpVerification::Accepted {
            draft: 4,
            target: 4,
        }));
        assert!(adaptive.observe(NativeMtpVerification::Rejected {
            draft: 5,
            target: 6,
        }));
        assert!(adaptive.disabled());
    }

    #[test]
    fn adaptive_disable_stays_enabled_for_high_acceptance_window() {
        let mut adaptive = NativeMtpAdaptiveDisable::new(NativeMtpAdaptiveDisableConfig {
            enabled: true,
            min_verifications: 4,
            threshold: 0.75,
        });

        for token in 0..4 {
            assert!(!adaptive.observe(NativeMtpVerification::Accepted {
                draft: token,
                target: token,
            }));
        }

        assert!(!adaptive.disabled());
    }

    #[test]
    fn adaptive_disable_flag_defaults_off_and_accepts_true_values() {
        assert!(!native_mtp_adaptive_disable_enabled_from(None));
        assert!(native_mtp_adaptive_disable_enabled_from(Some("1")));
        assert!(native_mtp_adaptive_disable_enabled_from(Some("true")));
        assert!(native_mtp_adaptive_disable_enabled_from(Some(" enabled ")));
        assert!(!native_mtp_adaptive_disable_enabled_from(Some("0")));
        assert!(!native_mtp_adaptive_disable_enabled_from(Some("false")));
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
        observe(&mut verifier, 11, 5, Some(draft(12)));
        observe(&mut verifier, 12, 9, None);

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

    #[test]
    fn batched_verify_flag_defaults_on_and_accepts_false_values() {
        assert!(native_mtp_batched_verify_enabled_from(None));
        assert!(native_mtp_batched_verify_enabled_from(Some("1")));
        assert!(native_mtp_batched_verify_enabled_from(Some("true")));
        assert!(!native_mtp_batched_verify_enabled_from(Some("0")));
        assert!(!native_mtp_batched_verify_enabled_from(Some("false")));
        assert!(!native_mtp_batched_verify_enabled_from(Some(" disabled ")));
    }

    #[test]
    fn serial_stage0_verify_flag_defaults_off_and_accepts_true_values() {
        assert!(!native_mtp_serial_stage0_verify_enabled_from(None));
        assert!(native_mtp_serial_stage0_verify_enabled_from(Some("1")));
        assert!(native_mtp_serial_stage0_verify_enabled_from(Some("true")));
        assert!(native_mtp_serial_stage0_verify_enabled_from(Some(
            " enabled "
        )));
        assert!(!native_mtp_serial_stage0_verify_enabled_from(Some("0")));
        assert!(!native_mtp_serial_stage0_verify_enabled_from(Some("false")));
    }

    #[test]
    fn compare_stage0_verify_flag_defaults_off_and_accepts_true_values() {
        assert!(!native_mtp_compare_stage0_verify_enabled_from(None));
        assert!(native_mtp_compare_stage0_verify_enabled_from(Some("1")));
        assert!(native_mtp_compare_stage0_verify_enabled_from(Some("true")));
        assert!(native_mtp_compare_stage0_verify_enabled_from(Some(
            " enabled "
        )));
        assert!(!native_mtp_compare_stage0_verify_enabled_from(Some("0")));
        assert!(!native_mtp_compare_stage0_verify_enabled_from(Some(
            "false"
        )));
    }

    #[test]
    fn reject_cooldown_tokens_defaults_zero() {
        assert_eq!(parse_usize_env("SKIPPY_TEST_MISSING_REJECT_COOLDOWN", 0), 0);
    }

    #[test]
    fn reject_recovery_serial_accepts_defaults_zero() {
        assert_eq!(parse_usize_env("SKIPPY_TEST_MISSING_REJECT_RECOVERY", 0), 0);
    }

    #[test]
    fn serial_after_gap_reject_recovery_serial_accepts_defaults_zero() {
        assert_eq!(
            parse_usize_env("SKIPPY_TEST_MISSING_GAP_REJECT_RECOVERY", 0),
            0
        );
    }

    #[test]
    fn verify_next_draft_min_margin_defaults_none() {
        assert_eq!(
            parse_optional_f32_env("SKIPPY_TEST_MISSING_VERIFY_NEXT_MARGIN"),
            None
        );
    }

    #[test]
    fn defer_reject_trim_flag_defaults_off_and_accepts_true_values() {
        assert!(!native_mtp_defer_reject_trim_enabled_from(None));
        assert!(native_mtp_defer_reject_trim_enabled_from(Some("1")));
        assert!(native_mtp_defer_reject_trim_enabled_from(Some("true")));
        assert!(native_mtp_defer_reject_trim_enabled_from(Some(" enabled ")));
        assert!(!native_mtp_defer_reject_trim_enabled_from(Some("0")));
        assert!(!native_mtp_defer_reject_trim_enabled_from(Some("false")));
    }

    #[test]
    fn suppress_cooldown_drafts_flag_defaults_off_and_accepts_true_values() {
        assert!(!native_mtp_suppress_cooldown_drafts_enabled_from(None));
        assert!(native_mtp_suppress_cooldown_drafts_enabled_from(Some("1")));
        assert!(native_mtp_suppress_cooldown_drafts_enabled_from(Some(
            "true"
        )));
        assert!(native_mtp_suppress_cooldown_drafts_enabled_from(Some(
            " enabled "
        )));
        assert!(!native_mtp_suppress_cooldown_drafts_enabled_from(Some("0")));
        assert!(!native_mtp_suppress_cooldown_drafts_enabled_from(Some(
            "false"
        )));
    }

    #[test]
    fn suppress_cooldown_draft_limit_defaults_zero() {
        assert_eq!(
            parse_usize_env("SKIPPY_TEST_MISSING_SUPPRESS_COOLDOWN_LIMIT", 0),
            0
        );
    }
}
