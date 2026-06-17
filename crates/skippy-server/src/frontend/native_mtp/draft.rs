const MTP_DRAFT_MARGIN_SCALE: f32 = 1000.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::frontend) struct NativeMtpDraft {
    pub(in crate::frontend) token: i32,
    pub(in crate::frontend) proposal_compute_us: i64,
    pub(in crate::frontend) margin_milli: Option<i32>,
}

impl NativeMtpDraft {
    pub(in crate::frontend) fn from_prediction_tokens(tokens: &[i32]) -> Option<Self> {
        let token = *tokens.get(1)?;
        let proposal_compute_us = tokens.get(2).copied().unwrap_or_default();
        Some(Self {
            token,
            proposal_compute_us: i64::from(proposal_compute_us.max(0)),
            margin_milli: tokens.get(3).copied(),
        })
    }

    pub(in crate::frontend) fn from_verify_prediction_tokens(
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

    pub(in crate::frontend) fn margin(&self) -> Option<f32> {
        margin_from_milli(self.margin_milli)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::frontend) struct PendingNativeMtpDraft {
    pub(in crate::frontend) token: i32,
    pub(in crate::frontend) margin_milli: Option<i32>,
    pub(in crate::frontend) origin: NativeMtpDraftOrigin,
}

impl PendingNativeMtpDraft {
    pub(in crate::frontend) fn margin(&self) -> Option<f32> {
        margin_from_milli(self.margin_milli)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::frontend) enum NativeMtpDraftOrigin {
    InitialSerial,
    SerialAfterGap,
    VerifyNext,
}

impl NativeMtpDraftOrigin {
    pub(in crate::frontend) fn label(self) -> &'static str {
        match self {
            Self::InitialSerial => "initial_serial",
            Self::SerialAfterGap => "serial_after_gap",
            Self::VerifyNext => "verify_next",
        }
    }
}

fn margin_from_milli(margin_milli: Option<i32>) -> Option<f32> {
    margin_milli.map(|margin| margin as f32 / MTP_DRAFT_MARGIN_SCALE)
}

pub(in crate::frontend) fn margin_passes_min_threshold(
    margin: Option<f32>,
    min_margin: Option<f32>,
) -> bool {
    min_margin.is_none_or(|min_margin| margin.is_none_or(|margin| margin >= min_margin))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(
            NativeMtpDraft::from_prediction_tokens(&[11, 12, 34, 567]),
            Some(NativeMtpDraft {
                token: 12,
                proposal_compute_us: 34,
                margin_milli: Some(567),
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
    fn pending_draft_margin_uses_same_scale_as_native_draft() {
        let pending = PendingNativeMtpDraft {
            token: 12,
            margin_milli: Some(1250),
            origin: NativeMtpDraftOrigin::VerifyNext,
        };

        assert_eq!(pending.margin(), Some(1.25));
        assert_eq!(pending.origin.label(), "verify_next");
    }

    #[test]
    fn missing_margin_passes_threshold_filter() {
        assert!(margin_passes_min_threshold(None, None));
        assert!(margin_passes_min_threshold(Some(0.25), None));
        assert!(margin_passes_min_threshold(None, Some(1.0)));
    }

    #[test]
    fn present_margin_is_filtered_by_threshold() {
        assert!(!margin_passes_min_threshold(Some(0.5), Some(1.0)));
        assert!(margin_passes_min_threshold(Some(1.0), Some(1.0)));
        assert!(margin_passes_min_threshold(Some(1.5), Some(1.0)));
    }
}
