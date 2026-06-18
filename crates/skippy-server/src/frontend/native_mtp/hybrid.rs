use openai_frontend::{OpenAiError, OpenAiResult};

use super::NativeMtpDecodeOptions;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::frontend) struct NativeMtpHybridProposal {
    tokens: Vec<i32>,
    ngram_span_available: bool,
    ngram_anchor_agreed: bool,
    ngram_anchor_disagreed: bool,
}

impl NativeMtpHybridProposal {
    pub(in crate::frontend) fn from_anchor(
        anchor: i32,
        context_tokens: &[i32],
        options: NativeMtpDecodeOptions,
        max_proposal_tokens: usize,
    ) -> Self {
        let max_proposal_tokens = max_proposal_tokens.max(1);
        if !options.ngram_hybrid || options.ngram_max_proposal_tokens == 0 {
            return Self::anchor_only(anchor);
        }

        let proposal_limit = max_proposal_tokens.min(options.ngram_max_proposal_tokens);
        let ngram_tokens =
            ngram_history_proposal(context_tokens, options.ngram_size, proposal_limit);
        let ngram_span_available = !ngram_tokens.is_empty();
        let ngram_anchor_agreed = ngram_tokens.first().is_some_and(|token| *token == anchor);
        let ngram_anchor_disagreed = ngram_span_available && !ngram_anchor_agreed;
        if ngram_anchor_agreed {
            return Self {
                tokens: ngram_tokens,
                ngram_span_available,
                ngram_anchor_agreed,
                ngram_anchor_disagreed,
            };
        }

        Self {
            tokens: vec![anchor],
            ngram_span_available,
            ngram_anchor_agreed,
            ngram_anchor_disagreed,
        }
    }

    pub(in crate::frontend) fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    pub(in crate::frontend) fn ngram_span_available(&self) -> bool {
        self.ngram_span_available
    }

    pub(in crate::frontend) fn ngram_anchor_agreed(&self) -> bool {
        self.ngram_anchor_agreed
    }

    pub(in crate::frontend) fn ngram_anchor_disagreed(&self) -> bool {
        self.ngram_anchor_disagreed
    }

    fn anchor_only(anchor: i32) -> Self {
        Self {
            tokens: vec![anchor],
            ngram_span_available: false,
            ngram_anchor_agreed: false,
            ngram_anchor_disagreed: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::frontend) struct NativeMtpBatchedDecision {
    pub(in crate::frontend) accepted_proposal_tokens: usize,
    pub(in crate::frontend) commit_count: usize,
    pub(in crate::frontend) rejected: bool,
}

pub(in crate::frontend) fn native_mtp_verify_inputs_for_proposals(
    current: i32,
    proposals: &[i32],
) -> Vec<i32> {
    let mut tokens = Vec::with_capacity(proposals.len().saturating_add(1));
    tokens.push(current);
    tokens.extend_from_slice(proposals);
    tokens
}

pub(in crate::frontend) fn classify_native_mtp_batched_verify<F>(
    proposal_tokens: &[i32],
    predicted_tokens: &[i32],
    generated_len: usize,
    max_new_tokens: usize,
    mut token_is_eog: F,
) -> OpenAiResult<NativeMtpBatchedDecision>
where
    F: FnMut(i32) -> OpenAiResult<bool>,
{
    let required_predictions = proposal_tokens.len().saturating_add(1);
    if predicted_tokens.len() < required_predictions {
        return Err(OpenAiError::backend(format!(
            "native MTP verify span returned too few tokens: got {} expected {}",
            predicted_tokens.len(),
            required_predictions
        )));
    }

    let mut accepted_proposal_tokens = 0usize;
    for (index, proposal_token) in proposal_tokens.iter().enumerate() {
        let predicted = predicted_tokens[index];
        let commit_count = index + 1;
        let accepted = predicted == *proposal_token;
        let reached_eog = token_is_eog(predicted)?;
        let reached_limit = generated_len + commit_count >= max_new_tokens;
        if !accepted {
            return Ok(NativeMtpBatchedDecision {
                accepted_proposal_tokens,
                commit_count,
                rejected: true,
            });
        }

        accepted_proposal_tokens += 1;
        if reached_eog || reached_limit {
            return Ok(NativeMtpBatchedDecision {
                accepted_proposal_tokens,
                commit_count,
                rejected: false,
            });
        }
    }

    let extra_commit_count = proposal_tokens.len().saturating_add(1);
    Ok(NativeMtpBatchedDecision {
        accepted_proposal_tokens,
        commit_count: extra_commit_count.min(max_new_tokens.saturating_sub(generated_len)),
        rejected: false,
    })
}

fn ngram_history_proposal(
    context_tokens: &[i32],
    ngram_size: usize,
    max_tokens: usize,
) -> Vec<i32> {
    if ngram_size == 0 || max_tokens == 0 || context_tokens.len() <= ngram_size {
        return Vec::new();
    }

    let suffix_start = context_tokens.len() - ngram_size;
    let suffix = &context_tokens[suffix_start..];
    for candidate_start in (0..suffix_start).rev() {
        let candidate_end = candidate_start + ngram_size;
        if &context_tokens[candidate_start..candidate_end] != suffix {
            continue;
        }
        let proposal_start = candidate_end;
        let proposal_end = proposal_start
            .saturating_add(max_tokens)
            .min(context_tokens.len());
        if proposal_start < proposal_end {
            return context_tokens[proposal_start..proposal_end].to_vec();
        }
    }

    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> NativeMtpDecodeOptions {
        NativeMtpDecodeOptions {
            batched_verify: true,
            reject_cooldown_tokens: 0,
            defer_reject_trim: false,
            suppress_cooldown_drafts: false,
            suppress_cooldown_draft_limit: 0,
            ngram_hybrid: true,
            ngram_size: 2,
            ngram_max_proposal_tokens: 4,
        }
    }

    #[test]
    fn hybrid_extends_when_ngram_first_token_matches_anchor() {
        let proposal =
            NativeMtpHybridProposal::from_anchor(3, &[1, 2, 3, 4, 5, 1, 2], options(), 4);

        assert_eq!(proposal.tokens(), &[3, 4, 5, 1]);
        assert!(proposal.ngram_span_available());
        assert!(proposal.ngram_anchor_agreed());
        assert!(!proposal.ngram_anchor_disagreed());
    }

    #[test]
    fn hybrid_keeps_only_anchor_when_ngram_first_token_disagrees() {
        let proposal =
            NativeMtpHybridProposal::from_anchor(9, &[1, 2, 3, 4, 5, 1, 2], options(), 4);

        assert_eq!(proposal.tokens(), &[9]);
        assert!(proposal.ngram_span_available());
        assert!(!proposal.ngram_anchor_agreed());
        assert!(proposal.ngram_anchor_disagreed());
    }

    #[test]
    fn hybrid_keeps_anchor_only_when_disabled() {
        let mut options = options();
        options.ngram_hybrid = false;

        let proposal = NativeMtpHybridProposal::from_anchor(3, &[1, 2, 3, 4, 1, 2], options, 4);

        assert_eq!(proposal.tokens(), &[3]);
        assert!(!proposal.ngram_span_available());
    }

    #[test]
    fn verify_inputs_include_current_and_all_proposals() {
        assert_eq!(
            native_mtp_verify_inputs_for_proposals(10, &[11, 12, 13]),
            vec![10, 11, 12, 13]
        );
    }

    #[test]
    fn classify_commits_extra_target_after_full_accept() {
        let decision =
            classify_native_mtp_batched_verify(&[11, 12], &[11, 12, 13], 0, 8, |_| Ok(false))
                .unwrap();

        assert_eq!(
            decision,
            NativeMtpBatchedDecision {
                accepted_proposal_tokens: 2,
                commit_count: 3,
                rejected: false,
            }
        );
    }

    #[test]
    fn classify_commits_rejected_target_and_trims_rest() {
        let decision =
            classify_native_mtp_batched_verify(&[11, 12, 13], &[11, 42, 99, 100], 0, 8, |_| {
                Ok(false)
            })
            .unwrap();

        assert_eq!(
            decision,
            NativeMtpBatchedDecision {
                accepted_proposal_tokens: 1,
                commit_count: 2,
                rejected: true,
            }
        );
    }

    #[test]
    fn classify_stops_without_extra_target_at_generation_limit() {
        let decision =
            classify_native_mtp_batched_verify(&[11, 12], &[11, 12, 13], 0, 2, |_| Ok(false))
                .unwrap();

        assert_eq!(
            decision,
            NativeMtpBatchedDecision {
                accepted_proposal_tokens: 2,
                commit_count: 2,
                rejected: false,
            }
        );
    }
}
