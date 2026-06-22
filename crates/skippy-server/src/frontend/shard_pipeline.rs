use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ShardPipelineConfig {
    pub(super) depth: usize,
    pub(super) max_decoded: usize,
    pub(super) prefill_token_count: usize,
    pub(super) current: i32,
    pub(super) decoded: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ShardVerifyWindow {
    pub(super) window_index: usize,
    pub(super) epoch: i32,
    pub(super) decoded_at_send: usize,
    pub(super) pos_start: usize,
    pub(super) decode_step: usize,
    pub(super) verify_inputs: Vec<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ShardCursor {
    pub(super) current: i32,
    pub(super) decoded: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ShardRejectRecovery {
    pub(super) stale_windows: usize,
}

#[derive(Debug)]
pub(super) struct ShardPipelineScheduler {
    depth: usize,
    max_decoded: usize,
    prefill_token_count: usize,
    cursor: ShardCursor,
    epoch: i32,
    next_window_index: usize,
    inflight: VecDeque<ShardVerifyWindow>,
    stale_debt: VecDeque<ShardVerifyWindow>,
}

impl ShardPipelineScheduler {
    pub(super) fn new(config: ShardPipelineConfig) -> Self {
        Self {
            depth: config.depth.max(1),
            max_decoded: config.max_decoded,
            prefill_token_count: config.prefill_token_count,
            cursor: ShardCursor {
                current: config.current,
                decoded: config.decoded,
            },
            epoch: 0,
            next_window_index: 0,
            inflight: VecDeque::new(),
            stale_debt: VecDeque::new(),
        }
    }

    pub(super) fn cursor(&self) -> ShardCursor {
        self.cursor
    }

    pub(super) fn can_prime_draft(&self) -> bool {
        self.cursor.decoded < self.max_decoded
    }

    pub(super) fn can_send(&self) -> bool {
        self.inflight.len().saturating_add(self.stale_debt.len()) < self.depth
            && self.can_prime_draft()
    }

    pub(super) fn proposal_limit(
        &self,
        max_speculative_window: usize,
        draft_window: usize,
    ) -> Option<usize> {
        let remaining = self.max_decoded.checked_sub(self.cursor.decoded)?;
        if remaining == 0 {
            return None;
        }
        Some(
            remaining
                .min(max_speculative_window)
                .min(draft_window)
                .max(1),
        )
    }

    pub(super) fn prepare_window(&self, draft_tokens: &[i32]) -> Option<ShardVerifyWindow> {
        if draft_tokens.is_empty() || !self.can_prime_draft() {
            return None;
        }
        let mut verify_inputs = Vec::with_capacity(draft_tokens.len() + 1);
        verify_inputs.push(self.cursor.current);
        verify_inputs.extend_from_slice(draft_tokens);
        Some(ShardVerifyWindow {
            window_index: self.next_window_index,
            epoch: self.epoch,
            decoded_at_send: self.cursor.decoded,
            pos_start: self.prefill_token_count + self.cursor.decoded,
            decode_step: self.cursor.decoded,
            verify_inputs,
        })
    }

    pub(super) fn mark_sent(&mut self, window: ShardVerifyWindow) {
        let Some(current) = window.verify_inputs.last().copied() else {
            return;
        };
        self.cursor.current = current;
        self.cursor.decoded = self
            .cursor
            .decoded
            .saturating_add(window.verify_inputs.len().saturating_sub(1));
        self.next_window_index = self.next_window_index.saturating_add(1);
        self.inflight.push_back(window);
    }

    pub(super) fn mark_inflight_returned(&mut self) -> Option<ShardVerifyWindow> {
        self.inflight.pop_front()
    }

    pub(super) fn reject_to(&mut self, current: i32, decoded: usize) -> ShardRejectRecovery {
        let stale_windows = self.inflight.len();
        self.stale_debt.append(&mut self.inflight);
        self.cursor = ShardCursor { current, decoded };
        self.epoch = self.epoch.saturating_add(1);
        ShardRejectRecovery { stale_windows }
    }

    pub(super) fn mark_stale_returned(&mut self) -> Option<ShardVerifyWindow> {
        self.stale_debt.pop_front()
    }

    pub(super) fn cancel_inflight(&mut self) -> ShardRejectRecovery {
        let stale_windows = self.inflight.len();
        self.stale_debt.append(&mut self.inflight);
        ShardRejectRecovery { stale_windows }
    }

    #[cfg(test)]
    fn inflight_len(&self) -> usize {
        self.inflight.len()
    }

    #[cfg(test)]
    fn stale_debt_len(&self) -> usize {
        self.stale_debt.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::speculative::{VerifySpanDecisionKind, classify_verify_span};

    #[test]
    fn shard_windows_overlap_by_current_token_and_advance_by_draft_len() {
        let mut scheduler = scheduler(3, 10, 7, 0);

        let first = scheduler.prepare_window(&[8, 9]).expect("first");
        assert_eq!(first.window_index, 0);
        assert_eq!(first.epoch, 0);
        assert_eq!(first.pos_start, 10);
        assert_eq!(first.decode_step, 0);
        assert_eq!(first.verify_inputs, vec![7, 8, 9]);

        scheduler.mark_sent(first);
        assert_eq!(
            scheduler.cursor(),
            ShardCursor {
                current: 9,
                decoded: 2
            }
        );

        let second = scheduler.prepare_window(&[10, 11]).expect("second");
        assert_eq!(second.window_index, 1);
        assert_eq!(second.epoch, 0);
        assert_eq!(second.pos_start, 12);
        assert_eq!(second.decode_step, 2);
        assert_eq!(second.verify_inputs, vec![9, 10, 11]);
    }

    #[test]
    fn reject_moves_remaining_inflight_to_stale_debt_and_rewinds_cursor() {
        let mut scheduler = scheduler(3, 10, 5, 0);
        send(&mut scheduler, &[6, 7]);
        send(&mut scheduler, &[8, 9]);
        send(&mut scheduler, &[10, 11]);

        assert_eq!(scheduler.mark_inflight_returned().unwrap().window_index, 0);
        let recovery = scheduler.reject_to(42, 1);

        assert_eq!(recovery.stale_windows, 2);
        assert_eq!(scheduler.inflight_len(), 0);
        assert_eq!(scheduler.stale_debt_len(), 2);
        assert_eq!(
            scheduler.cursor(),
            ShardCursor {
                current: 42,
                decoded: 1
            }
        );

        let fresh = scheduler.prepare_window(&[43]).expect("fresh window");
        assert_eq!(fresh.epoch, 1);
        assert_eq!(fresh.window_index, 3);
    }

    #[test]
    fn stale_debt_consumes_depth_but_allows_fresh_work_behind_fifo_returns() {
        let mut scheduler = scheduler(3, 12, 5, 0);
        send(&mut scheduler, &[6, 7]);
        send(&mut scheduler, &[8, 9]);
        send(&mut scheduler, &[10, 11]);
        scheduler.mark_inflight_returned();
        scheduler.reject_to(99, 1);

        assert!(scheduler.can_send());
        send(&mut scheduler, &[100, 101]);
        assert!(!scheduler.can_send());

        assert_eq!(scheduler.mark_stale_returned().unwrap().window_index, 1);
        assert!(scheduler.can_send());
    }

    #[test]
    fn proposal_limit_is_bounded_by_remaining_draft_and_spec_windows() {
        let mut scheduler = scheduler(3, 5, 1, 3);

        assert_eq!(scheduler.proposal_limit(8, 6), Some(2));
        assert_eq!(scheduler.proposal_limit(1, 6), Some(1));

        send(&mut scheduler, &[2, 3]);
        assert_eq!(scheduler.proposal_limit(8, 6), None);
    }

    #[test]
    fn cancel_inflight_marks_unread_windows_stale_for_drain() {
        let mut scheduler = scheduler(4, 12, 5, 0);
        send(&mut scheduler, &[6]);
        send(&mut scheduler, &[7]);

        let recovery = scheduler.cancel_inflight();

        assert_eq!(recovery.stale_windows, 2);
        assert_eq!(scheduler.inflight_len(), 0);
        assert_eq!(scheduler.stale_debt_len(), 2);
    }

    #[test]
    fn adversarial_rejects_still_emit_target_greedy_stream() {
        let target = target_tokens(80);
        let proof = run_shard_pipeline_proof(ProofConfig {
            depth: 4,
            draft_k: 3,
            max_decoded: 32,
            fault_positions: &[0, 5, 11, 12, 19],
            target: &target,
        });

        assert_eq!(proof.emitted, target[..32]);
        assert!(proof.max_inflight > 1);
        assert!(proof.rejected_windows >= 4);
        assert!(proof.stale_windows > 0);
        assert!(proof.early_reject_windows > 0);
        assert!(proof.tail_reject_windows > 0);
    }

    #[test]
    fn full_accept_path_defers_bonus_token_to_next_overlap() {
        let target = target_tokens(48);
        let proof = run_shard_pipeline_proof(ProofConfig {
            depth: 3,
            draft_k: 4,
            max_decoded: 24,
            fault_positions: &[],
            target: &target,
        });

        assert_eq!(proof.emitted, target[..24]);
        assert!(proof.full_accept_windows > 0);
        assert_eq!(proof.rejected_windows, 0);
        assert_eq!(proof.stale_windows, 0);
    }

    fn scheduler(
        depth: usize,
        prefill_token_count: usize,
        current: i32,
        decoded: usize,
    ) -> ShardPipelineScheduler {
        ShardPipelineScheduler::new(ShardPipelineConfig {
            depth,
            max_decoded: 5,
            prefill_token_count,
            current,
            decoded,
        })
    }

    fn send(scheduler: &mut ShardPipelineScheduler, draft_tokens: &[i32]) {
        let window = scheduler.prepare_window(draft_tokens).expect("window");
        scheduler.mark_sent(window);
    }

    #[derive(Clone, Copy)]
    struct ProofConfig<'a> {
        depth: usize,
        draft_k: usize,
        max_decoded: usize,
        fault_positions: &'a [usize],
        target: &'a [i32],
    }

    #[derive(Default)]
    struct ProofOutcome {
        emitted: Vec<i32>,
        max_inflight: usize,
        full_accept_windows: usize,
        early_reject_windows: usize,
        tail_reject_windows: usize,
        rejected_windows: usize,
        stale_windows: usize,
    }

    struct ProofWindow {
        window: ShardVerifyWindow,
        draft_tokens: Vec<i32>,
        predicted_tokens: Vec<i32>,
    }

    fn run_shard_pipeline_proof(config: ProofConfig<'_>) -> ProofOutcome {
        let mut scheduler = ShardPipelineScheduler::new(ShardPipelineConfig {
            depth: config.depth,
            max_decoded: config.max_decoded,
            prefill_token_count: 1,
            current: 999,
            decoded: 0,
        });
        let mut inflight = VecDeque::<ProofWindow>::new();
        let mut stale_debt = VecDeque::<ProofWindow>::new();
        let mut outcome = ProofOutcome::default();

        while outcome.emitted.len() < config.max_decoded {
            fill_proof_windows(config, &mut scheduler, &mut inflight, &mut outcome);
            if !stale_debt.is_empty() {
                stale_debt.pop_front();
                scheduler.mark_stale_returned();
                outcome.stale_windows += 1;
                continue;
            }
            let Some(proof_window) = inflight.pop_front() else {
                break;
            };
            scheduler.mark_inflight_returned();
            let rejected = commit_proof_window(config, proof_window, &mut scheduler, &mut outcome);
            if rejected {
                let recovery = scheduler.reject_to(
                    *outcome.emitted.last().expect("reject commits a correction"),
                    outcome.emitted.len(),
                );
                outcome.stale_windows += recovery.stale_windows;
                scheduler.cancel_inflight();
                stale_debt.append(&mut inflight);
            }
        }

        outcome
    }

    fn fill_proof_windows(
        config: ProofConfig<'_>,
        scheduler: &mut ShardPipelineScheduler,
        inflight: &mut VecDeque<ProofWindow>,
        outcome: &mut ProofOutcome,
    ) {
        while scheduler.can_send() {
            let cursor = scheduler.cursor();
            let draft_tokens = proof_draft_tokens(config, cursor.decoded);
            let Some(window) = scheduler.prepare_window(&draft_tokens) else {
                break;
            };
            let predicted_tokens =
                config.target[window.decoded_at_send..][..window.verify_inputs.len()].to_vec();
            scheduler.mark_sent(window.clone());
            inflight.push_back(ProofWindow {
                window,
                draft_tokens,
                predicted_tokens,
            });
            outcome.max_inflight = outcome.max_inflight.max(inflight.len());
        }
    }

    fn commit_proof_window(
        config: ProofConfig<'_>,
        proof_window: ProofWindow,
        scheduler: &mut ShardPipelineScheduler,
        outcome: &mut ProofOutcome,
    ) -> bool {
        let decision = classify_verify_span(
            &proof_window.draft_tokens,
            &proof_window.predicted_tokens,
            proof_window.window.decoded_at_send,
            config.max_decoded,
            |_| Ok(false),
        )
        .expect("synthetic verifier returns K+1 predictions");
        let commit_count = match decision.kind {
            VerifySpanDecisionKind::FullAccept => {
                outcome.full_accept_windows += 1;
                proof_window.draft_tokens.len()
            }
            VerifySpanDecisionKind::TailReject => {
                outcome.tail_reject_windows += 1;
                decision.commit_count
            }
            VerifySpanDecisionKind::EarlyReject | VerifySpanDecisionKind::EarlyRejectStop => {
                outcome.early_reject_windows += 1;
                decision.commit_count
            }
            VerifySpanDecisionKind::AcceptedStop => decision.commit_count,
        };
        outcome.emitted.extend(
            proof_window
                .predicted_tokens
                .iter()
                .take(commit_count)
                .take(config.max_decoded.saturating_sub(outcome.emitted.len())),
        );
        let rejected = decision.rejected();
        if rejected {
            outcome.rejected_windows += 1;
        } else {
            let cursor = scheduler.cursor();
            assert!(cursor.decoded >= outcome.emitted.len());
        }
        rejected
    }

    fn proof_draft_tokens(config: ProofConfig<'_>, decoded: usize) -> Vec<i32> {
        let remaining = config.max_decoded.saturating_sub(decoded);
        let count = remaining.min(config.draft_k);
        (0..count)
            .map(|offset| {
                let position = decoded + offset;
                let target = config.target[position];
                if config.fault_positions.contains(&position) {
                    target.saturating_add(10_000)
                } else {
                    target
                }
            })
            .collect()
    }

    fn target_tokens(count: usize) -> Vec<i32> {
        (0..count).map(|index| 100 + index as i32).collect()
    }
}
