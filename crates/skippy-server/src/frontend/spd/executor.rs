use std::collections::{BTreeMap, VecDeque};

use anyhow::{Context, Result, bail};
use skippy_runtime::spd::{SpdRollingScheduler, SpdRollingVerifyOutcome};

use crate::binary_transport::PredictionReturnOrigin;

use super::{PhaseTimer, SpdInlineProbe, SpdInlineProbePhase, SpdReplayProposalSource};

#[derive(Debug)]
pub(in crate::frontend) struct SpdRollingExecutor {
    logical_stage_count: usize,
    base_position: usize,
    scheduler: SpdRollingScheduler,
    pending_pre_step_scheduler: Option<SpdRollingScheduler>,
    speculative_context: Vec<i32>,
    in_flight: VecDeque<SpdRollingExecutorInFlight>,
    target_tokens: BTreeMap<usize, i32>,
    stats: SpdRollingExecutorStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SpdRollingExecutorInFlight {
    position: usize,
    proposed: i32,
    origin: PredictionReturnOrigin,
}

#[derive(Debug, Clone, PartialEq)]
pub(in crate::frontend) struct SpdRollingExecutorPreparedLaunch {
    pub(in crate::frontend) position: usize,
    pub(in crate::frontend) proposed: i32,
    pub(in crate::frontend) decode_step: usize,
    pub(in crate::frontend) chain_depth: usize,
    pub(in crate::frontend) probe: SpdInlineProbe,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(in crate::frontend) struct SpdRollingExecutorStats {
    pub(in crate::frontend) launches: usize,
    pub(in crate::frontend) launch_misses: usize,
    pub(in crate::frontend) launch_miss_in_flight_full: usize,
    pub(in crate::frontend) launch_miss_no_rows: usize,
    pub(in crate::frontend) launch_miss_no_proposal: usize,
    pub(in crate::frontend) launch_miss_shadow_not_seedable: usize,
    pub(in crate::frontend) launch_miss_shadow_missing_view: usize,
    pub(in crate::frontend) shadow_source_reseeds: usize,
    pub(in crate::frontend) launch_margin_rejects: usize,
    pub(in crate::frontend) max_in_flight: usize,
    pub(in crate::frontend) accepted_oldest: usize,
    pub(in crate::frontend) rejected_oldest: usize,
    pub(in crate::frontend) drained_younger: usize,
    pub(in crate::frontend) target_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::frontend) enum SpdRollingExecutorLaunchMissReason {
    InFlightFull,
    NoSpeculationRows,
    NoProposal,
    ShadowNotSeedable,
    ShadowMissingView,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::frontend) enum SpdRollingExecutorCommit {
    Accepted {
        completed_position: usize,
        position: usize,
        token: i32,
        origin: Option<PredictionReturnOrigin>,
        in_flight_after: usize,
    },
    Rejected {
        completed_position: usize,
        position: usize,
        speculated: i32,
        corrected: i32,
        origin: Option<PredictionReturnOrigin>,
        drained_younger: usize,
    },
}

impl SpdRollingExecutor {
    pub(in crate::frontend) fn new(
        logical_stage_count: usize,
        context_tokens: &[i32],
    ) -> Result<Self> {
        let first_position = context_tokens
            .len()
            .checked_sub(1)
            .context("SPD rolling executor requires non-empty context")?;
        let first_token = context_tokens
            .last()
            .copied()
            .context("SPD rolling executor requires current token")?;
        Ok(Self {
            logical_stage_count,
            base_position: first_position,
            scheduler: SpdRollingScheduler::new(logical_stage_count, first_position, first_token)?,
            pending_pre_step_scheduler: None,
            speculative_context: context_tokens.to_vec(),
            in_flight: VecDeque::new(),
            target_tokens: BTreeMap::new(),
            stats: SpdRollingExecutorStats::default(),
        })
    }

    pub(in crate::frontend) fn prepare_launch(
        &mut self,
        source: &mut SpdReplayProposalSource,
        decode_step: usize,
        phase: SpdInlineProbePhase,
        min_logit_margin: Option<f32>,
        trigger_hf_index: Option<u32>,
    ) -> Result<Option<SpdRollingExecutorPreparedLaunch>> {
        if self.in_flight.len() >= self.logical_stage_count {
            self.record_launch_miss(SpdRollingExecutorLaunchMissReason::InFlightFull);
            return Ok(None);
        }
        self.drop_stale_pending_pre_step_scheduler();
        let launch_scheduler = self
            .pending_pre_step_scheduler
            .as_ref()
            .unwrap_or(&self.scheduler);
        let Some(rows) = launch_scheduler.speculation_rows() else {
            self.record_launch_miss(SpdRollingExecutorLaunchMissReason::NoSpeculationRows);
            return Ok(None);
        };
        let timer = PhaseTimer::start();
        let proposal =
            source.propose_inline_for_rolling_context(&self.speculative_context, &rows)?;
        let elapsed_ms = timer.elapsed_ms();
        let Some(proposal) = proposal else {
            self.record_launch_miss(SpdRollingExecutorLaunchMissReason::NoProposal);
            return Ok(None);
        };
        let probe = SpdInlineProbe::from_proposal(
            phase,
            Some(&proposal),
            elapsed_ms,
            0.0,
            trigger_hf_index,
        );
        if !probe.allows_optimistic_decode(min_logit_margin) {
            self.stats.launch_margin_rejects += 1;
            return Ok(None);
        }
        let position = launch_scheduler.next_position();
        Ok(Some(SpdRollingExecutorPreparedLaunch {
            position,
            proposed: proposal.token,
            decode_step,
            chain_depth: self.in_flight.len(),
            probe,
        }))
    }

    pub(in crate::frontend) fn record_launch(
        &mut self,
        launch: &SpdRollingExecutorPreparedLaunch,
        origin: PredictionReturnOrigin,
    ) -> Result<()> {
        if self.speculative_context.len() != launch.position {
            bail!(
                "SPD rolling executor context length {} does not match launch position {}",
                self.speculative_context.len(),
                launch.position
            );
        }
        self.scheduler
            .insert_draft_at(launch.position, launch.proposed)?;
        if self
            .pending_pre_step_scheduler
            .as_ref()
            .is_some_and(|scheduler| scheduler.next_position() == launch.position)
        {
            self.pending_pre_step_scheduler = None;
        }
        self.speculative_context.push(launch.proposed);
        self.in_flight.push_back(SpdRollingExecutorInFlight {
            position: launch.position,
            proposed: launch.proposed,
            origin,
        });
        self.stats.launches += 1;
        self.stats.max_in_flight = self.stats.max_in_flight.max(self.in_flight.len());
        Ok(())
    }

    pub(in crate::frontend) fn record_launch_miss(
        &mut self,
        reason: SpdRollingExecutorLaunchMissReason,
    ) {
        self.stats.launch_misses += 1;
        match reason {
            SpdRollingExecutorLaunchMissReason::InFlightFull => {
                self.stats.launch_miss_in_flight_full += 1;
            }
            SpdRollingExecutorLaunchMissReason::NoSpeculationRows => {
                self.stats.launch_miss_no_rows += 1;
            }
            SpdRollingExecutorLaunchMissReason::NoProposal => {
                self.stats.launch_miss_no_proposal += 1;
            }
            SpdRollingExecutorLaunchMissReason::ShadowNotSeedable => {
                self.stats.launch_miss_shadow_not_seedable += 1;
            }
            SpdRollingExecutorLaunchMissReason::ShadowMissingView => {
                self.stats.launch_miss_shadow_missing_view += 1;
            }
        }
    }

    pub(in crate::frontend) fn record_shadow_source_reseed(&mut self) {
        self.stats.shadow_source_reseeds += 1;
    }

    pub(in crate::frontend) fn record_target_token(&mut self, position: usize, token: i32) {
        self.target_tokens.insert(position, token);
        self.stats.target_tokens += 1;
    }

    pub(in crate::frontend) fn advance_idle_to_accepted_context(
        &mut self,
        context_tokens: &[i32],
    ) -> Result<bool> {
        if !self.in_flight.is_empty() {
            return Ok(false);
        }
        let Some((accepted_position, accepted_token)) =
            context_tokens.len().checked_sub(1).and_then(|position| {
                context_tokens
                    .last()
                    .copied()
                    .map(|token| (position, token))
            })
        else {
            return Ok(false);
        };
        if self.base_position == accepted_position
            && self.speculative_context.as_slice() == context_tokens
            && self.scheduler.next_position() == context_tokens.len()
        {
            return Ok(false);
        }
        self.base_position = accepted_position;
        self.scheduler =
            SpdRollingScheduler::new(self.logical_stage_count, accepted_position, accepted_token)?;
        self.pending_pre_step_scheduler = None;
        self.speculative_context = context_tokens.to_vec();
        self.target_tokens
            .retain(|position, _| *position <= accepted_position);
        Ok(true)
    }

    pub(in crate::frontend) fn commit_ready_oldest(
        &mut self,
    ) -> Result<Option<SpdRollingExecutorCommit>> {
        let Some(target_position) = self.scheduler.oldest_target_position() else {
            return Ok(None);
        };
        let Some(target_token) = self.target_tokens.get(&target_position).copied() else {
            return Ok(None);
        };
        let pre_step_scheduler = self.scheduler.clone();
        match self.scheduler.verify_oldest(target_token) {
            SpdRollingVerifyOutcome::NotReady => Ok(None),
            SpdRollingVerifyOutcome::Accepted {
                completed_position,
                target_position,
                token,
                ..
            } => {
                self.pending_pre_step_scheduler
                    .get_or_insert(pre_step_scheduler);
                let accepted = self.pop_completed_in_flight(completed_position)?;
                self.base_position = completed_position;
                self.stats.accepted_oldest += 1;
                Ok(Some(SpdRollingExecutorCommit::Accepted {
                    completed_position,
                    position: target_position,
                    token,
                    origin: accepted.map(|in_flight| in_flight.origin),
                    in_flight_after: self.in_flight.len(),
                }))
            }
            SpdRollingVerifyOutcome::Rejected {
                completed_position,
                target_position,
                speculated,
                corrected,
                ..
            } => {
                let rejected = self.pop_completed_in_flight(completed_position)?;
                let drained_younger = self.in_flight.len();
                self.in_flight.clear();
                self.pending_pre_step_scheduler = None;
                self.reset_speculative_context(target_position, corrected)?;
                self.base_position = target_position;
                self.target_tokens
                    .retain(|position, _| *position <= target_position);
                self.stats.rejected_oldest += 1;
                self.stats.drained_younger += drained_younger;
                Ok(Some(SpdRollingExecutorCommit::Rejected {
                    completed_position,
                    position: target_position,
                    speculated,
                    corrected,
                    origin: rejected.map(|in_flight| in_flight.origin),
                    drained_younger,
                }))
            }
        }
    }

    pub(in crate::frontend) fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }

    pub(in crate::frontend) fn logical_stage_count(&self) -> usize {
        self.logical_stage_count
    }

    pub(in crate::frontend) fn stats(&self) -> SpdRollingExecutorStats {
        self.stats
    }

    fn pop_completed_in_flight(
        &mut self,
        completed_position: usize,
    ) -> Result<Option<SpdRollingExecutorInFlight>> {
        if completed_position <= self.base_position {
            return Ok(None);
        }
        let in_flight = self
            .in_flight
            .pop_front()
            .context("SPD rolling executor has no in-flight verifier for oldest target")?;
        if in_flight.position != completed_position {
            bail!(
                "SPD rolling executor oldest in-flight position {} does not match completed position {}",
                in_flight.position,
                completed_position
            );
        }
        Ok(Some(in_flight))
    }

    fn drop_stale_pending_pre_step_scheduler(&mut self) {
        let context_len = self.speculative_context.len();
        if self
            .pending_pre_step_scheduler
            .as_ref()
            .is_some_and(|scheduler| scheduler.next_position() != context_len)
        {
            self.pending_pre_step_scheduler = None;
        }
    }

    fn reset_speculative_context(&mut self, target_position: usize, corrected: i32) -> Result<()> {
        if target_position > self.speculative_context.len() {
            bail!(
                "SPD rolling executor target position {target_position} exceeds speculative context length {}",
                self.speculative_context.len()
            );
        }
        self.speculative_context.truncate(target_position);
        self.speculative_context.push(corrected);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waits_for_pipeline_fill_before_committing_oldest() {
        let mut executor = SpdRollingExecutor::new(4, &[10, 20]).unwrap();

        executor.record_target_token(2, 21);
        assert_eq!(executor.commit_ready_oldest().unwrap(), None);

        record_launch(&mut executor, 2, 21);
        assert_eq!(executor.commit_ready_oldest().unwrap(), None);
        record_launch(&mut executor, 3, 22);
        assert_eq!(executor.commit_ready_oldest().unwrap(), None);
        record_launch(&mut executor, 4, 23);

        assert_eq!(
            executor.commit_ready_oldest().unwrap(),
            Some(SpdRollingExecutorCommit::Accepted {
                completed_position: 1,
                position: 2,
                token: 21,
                origin: None,
                in_flight_after: 3,
            })
        );
        assert_eq!(executor.stats().accepted_oldest, 1);
        assert_eq!(executor.stats().max_in_flight, 3);
    }

    #[test]
    fn accepted_commit_reports_completed_verifier_origin_after_initial_window() {
        let mut executor = SpdRollingExecutor::new(3, &[10, 20]).unwrap();
        record_launch(&mut executor, 2, 21);
        record_launch(&mut executor, 3, 22);
        executor.record_target_token(2, 21);
        assert_eq!(
            executor.commit_ready_oldest().unwrap(),
            Some(SpdRollingExecutorCommit::Accepted {
                completed_position: 1,
                position: 2,
                token: 21,
                origin: None,
                in_flight_after: 2,
            })
        );
        assert_eq!(
            executor
                .pending_pre_step_scheduler
                .as_ref()
                .and_then(SpdRollingScheduler::speculation_rows)
                .map(|rows| rows.row_positions),
            Some(vec![1, 2, 3])
        );

        record_launch(&mut executor, 4, 23);
        assert!(executor.pending_pre_step_scheduler.is_none());
        executor.record_target_token(3, 22);

        assert_eq!(
            executor.commit_ready_oldest().unwrap(),
            Some(SpdRollingExecutorCommit::Accepted {
                completed_position: 2,
                position: 3,
                token: 22,
                origin: Some(origin(2)),
                in_flight_after: 2,
            })
        );
    }

    #[test]
    fn rejection_drains_younger_work_and_resets_speculative_context() {
        let mut executor = SpdRollingExecutor::new(3, &[10, 20]).unwrap();
        record_launch(&mut executor, 2, 21);
        record_launch(&mut executor, 3, 22);
        executor.record_target_token(2, 99);

        assert_eq!(
            executor.commit_ready_oldest().unwrap(),
            Some(SpdRollingExecutorCommit::Rejected {
                completed_position: 1,
                position: 2,
                speculated: 21,
                corrected: 99,
                origin: None,
                drained_younger: 2,
            })
        );
        assert_eq!(executor.in_flight_len(), 0);
        assert_eq!(executor.speculative_context.as_slice(), &[10, 20, 99]);
        assert_eq!(executor.stats().rejected_oldest, 1);
        assert_eq!(executor.stats().drained_younger, 2);
    }

    #[test]
    fn rejection_reset_accepts_fresh_launches() {
        let mut executor = SpdRollingExecutor::new(3, &[10, 20]).unwrap();
        record_launch(&mut executor, 2, 21);
        record_launch(&mut executor, 3, 22);
        executor.record_target_token(2, 99);
        assert!(matches!(
            executor.commit_ready_oldest().unwrap(),
            Some(SpdRollingExecutorCommit::Rejected { .. })
        ));

        record_launch(&mut executor, 3, 100);
        assert_eq!(executor.in_flight_len(), 1);
        assert_eq!(executor.speculative_context.as_slice(), &[10, 20, 99, 100]);
        assert_eq!(executor.stats().launches, 3);
        assert_eq!(executor.stats().max_in_flight, 2);
    }

    #[test]
    fn rejection_reset_treats_corrected_target_as_new_base() {
        let mut executor = SpdRollingExecutor::new(3, &[10, 20]).unwrap();
        record_launch(&mut executor, 2, 21);
        record_launch(&mut executor, 3, 22);
        executor.record_target_token(2, 99);
        assert!(matches!(
            executor.commit_ready_oldest().unwrap(),
            Some(SpdRollingExecutorCommit::Rejected { .. })
        ));

        record_launch(&mut executor, 3, 100);
        record_launch(&mut executor, 4, 101);
        executor.record_target_token(3, 100);

        assert_eq!(
            executor.commit_ready_oldest().unwrap(),
            Some(SpdRollingExecutorCommit::Accepted {
                completed_position: 2,
                position: 3,
                token: 100,
                origin: None,
                in_flight_after: 2,
            })
        );
    }

    #[test]
    fn idle_executor_catches_up_to_accepted_context_after_reset() {
        let mut executor = SpdRollingExecutor::new(3, &[10, 20]).unwrap();
        record_launch(&mut executor, 2, 21);
        record_launch(&mut executor, 3, 22);
        executor.record_target_token(2, 99);
        assert!(matches!(
            executor.commit_ready_oldest().unwrap(),
            Some(SpdRollingExecutorCommit::Rejected { .. })
        ));

        assert!(
            executor
                .advance_idle_to_accepted_context(&[10, 20, 99, 100])
                .unwrap()
        );

        assert_eq!(executor.in_flight_len(), 0);
        assert_eq!(executor.speculative_context.as_slice(), &[10, 20, 99, 100]);
        assert_eq!(executor.base_position, 3);
        record_launch(&mut executor, 4, 101);
        assert_eq!(executor.in_flight_len(), 1);
    }

    #[test]
    fn active_executor_does_not_catch_up_while_verifier_is_in_flight() {
        let mut executor = SpdRollingExecutor::new(3, &[10, 20]).unwrap();
        record_launch(&mut executor, 2, 21);

        assert!(
            !executor
                .advance_idle_to_accepted_context(&[10, 20, 99])
                .unwrap()
        );

        assert_eq!(executor.in_flight_len(), 1);
        assert_eq!(executor.speculative_context.as_slice(), &[10, 20, 21]);
        assert_eq!(executor.base_position, 1);
    }

    #[test]
    fn stale_pre_step_scheduler_is_dropped_after_context_advances() {
        let mut executor = SpdRollingExecutor::new(4, &[10, 20]).unwrap();
        record_launch(&mut executor, 2, 21);
        record_launch(&mut executor, 3, 22);

        executor.pending_pre_step_scheduler =
            Some(SpdRollingScheduler::new(4, executor.base_position, 20).unwrap());

        let stale_next = executor
            .pending_pre_step_scheduler
            .as_ref()
            .map(SpdRollingScheduler::next_position);
        assert_ne!(stale_next, Some(executor.speculative_context.len()));
        executor.drop_stale_pending_pre_step_scheduler();
        assert!(executor.pending_pre_step_scheduler.is_none());
    }

    fn record_launch(executor: &mut SpdRollingExecutor, position: usize, proposed: i32) {
        let launch = SpdRollingExecutorPreparedLaunch {
            position,
            proposed,
            decode_step: position,
            chain_depth: executor.in_flight_len(),
            probe: SpdInlineProbe::from_proposal(
                SpdInlineProbePhase::OptimisticCommit,
                None,
                0.0,
                0.0,
                None,
            ),
        };
        executor.record_launch(&launch, origin(position)).unwrap();
    }

    fn origin(pos_start: usize) -> PredictionReturnOrigin {
        PredictionReturnOrigin {
            kind: skippy_protocol::binary::WireMessageKind::VerifySpan,
            pos_start: i32::try_from(pos_start).unwrap(),
            token_count: 1,
            prompt_token_count: 1,
            decode_step: i32::try_from(pos_start).unwrap(),
            checkpoint_generation: i32::try_from(pos_start).unwrap(),
        }
    }
}
