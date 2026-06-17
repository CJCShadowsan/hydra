use std::collections::{BTreeMap, BTreeSet, VecDeque};

use anyhow::{Result, bail};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpdRollingScheduler {
    stage_count: usize,
    pipeline: VecDeque<SpdRollingEntry>,
    generated: BTreeMap<usize, i32>,
    next_position: usize,
    verified_up_to: usize,
    previous_evicted_position: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SpdRollingEntry {
    position: usize,
    token: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpdRollingSpeculationRows {
    pub evicted_prefix_position: Option<usize>,
    pub row_positions: Vec<usize>,
    pub row_i_stages: Vec<usize>,
    pub newest_position: usize,
    pub next_draft_position: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpdRollingInsertedDraft {
    pub position: usize,
    pub token: i32,
    pub pipeline_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpdRollingVerifyOutcome {
    NotReady,
    Accepted {
        completed_position: usize,
        target_position: usize,
        token: i32,
    },
    Rejected {
        completed_position: usize,
        target_position: usize,
        speculated: i32,
        corrected: i32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpdRollingSnapshot {
    pub logical_stage_count: usize,
    pub target_position: Option<usize>,
    pub next_position: Option<usize>,
    pub inserted_drafts: usize,
    pub missing_proposals: usize,
    pub first_missing_proposal_position: Option<usize>,
    pub out_of_order_proposals: usize,
    pub first_out_of_order_proposal_position: Option<usize>,
    pub verified_windows: usize,
    pub accepted_windows: usize,
    pub rejected_windows: usize,
    pub first_rejected_target_position: Option<usize>,
    pub pipeline_len: Option<usize>,
    pub verified_up_to: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpdRollingObserver {
    logical_stage_count: usize,
    scheduler: Option<SpdRollingScheduler>,
    target_tokens: BTreeMap<usize, i32>,
    proposed_positions: BTreeSet<usize>,
    pending_proposals: BTreeMap<usize, i32>,
    missing_proposal_positions: BTreeSet<usize>,
    out_of_order_proposal_positions: BTreeSet<usize>,
    inserted_drafts: usize,
    verified_windows: usize,
    accepted_windows: usize,
    rejected_windows: usize,
    first_rejected_target_position: Option<usize>,
    first_position: Option<usize>,
    last_target_position: Option<usize>,
    released_verified_up_to: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpdRollingDraftPlan {
    scheduler: SpdRollingScheduler,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpdRollingVerifiedDelta {
    pub start_position: usize,
    pub verified_up_to: usize,
    pub tokens: Vec<i32>,
}

impl SpdRollingObserver {
    pub fn new(logical_stage_count: usize) -> Self {
        Self {
            logical_stage_count,
            scheduler: None,
            target_tokens: BTreeMap::new(),
            proposed_positions: BTreeSet::new(),
            pending_proposals: BTreeMap::new(),
            missing_proposal_positions: BTreeSet::new(),
            out_of_order_proposal_positions: BTreeSet::new(),
            inserted_drafts: 0,
            verified_windows: 0,
            accepted_windows: 0,
            rejected_windows: 0,
            first_rejected_target_position: None,
            first_position: None,
            last_target_position: None,
            released_verified_up_to: None,
        }
    }

    pub fn observe_target(
        &mut self,
        target_position: usize,
        target: i32,
    ) -> Result<SpdRollingSnapshot> {
        let first_position = self.record_target(target_position, target)?;
        if target_position > first_position {
            self.mark_missing_position(target_position);
        }
        Ok(self.snapshot())
    }

    pub fn observe_probe(
        &mut self,
        target_position: usize,
        target: i32,
        proposed: Option<i32>,
    ) -> Result<SpdRollingSnapshot> {
        let first_position = self.record_target(target_position, target)?;
        if target_position == first_position {
            return Ok(self.snapshot());
        }
        let Some(proposed) = proposed else {
            self.mark_missing_position(target_position);
            return Ok(self.snapshot());
        };
        self.proposed_positions.insert(target_position);
        let Some(scheduler) = self.scheduler.as_mut() else {
            return Ok(self.snapshot());
        };
        let next_position = scheduler.next_position();
        if target_position > next_position {
            self.mark_missing_range(next_position, target_position);
            self.mark_out_of_order(target_position);
            self.pending_proposals.insert(target_position, proposed);
            return Ok(self.snapshot());
        }
        if target_position < next_position {
            self.mark_out_of_order(target_position);
            return Ok(self.snapshot());
        }
        scheduler.insert_draft_at(target_position, proposed)?;
        self.inserted_drafts += 1;
        self.verify_ready_windows();
        Ok(self.snapshot())
    }

    pub fn snapshot(&self) -> SpdRollingSnapshot {
        SpdRollingSnapshot {
            logical_stage_count: self.logical_stage_count,
            target_position: self.last_target_position,
            next_position: self
                .scheduler
                .as_ref()
                .map(SpdRollingScheduler::next_position),
            inserted_drafts: self.inserted_drafts,
            missing_proposals: self.missing_proposal_positions.len(),
            first_missing_proposal_position: self.missing_proposal_positions.first().copied(),
            out_of_order_proposals: self.out_of_order_proposal_positions.len(),
            first_out_of_order_proposal_position: self
                .out_of_order_proposal_positions
                .first()
                .copied(),
            verified_windows: self.verified_windows,
            accepted_windows: self.accepted_windows,
            rejected_windows: self.rejected_windows,
            first_rejected_target_position: self.first_rejected_target_position,
            pipeline_len: self
                .scheduler
                .as_ref()
                .map(SpdRollingScheduler::pipeline_len),
            verified_up_to: self
                .scheduler
                .as_ref()
                .map(SpdRollingScheduler::verified_up_to),
        }
    }

    pub fn trace_replay(&self) -> Option<SpdRollingTraceReplay> {
        let scheduler = self.scheduler.as_ref()?;
        let first_position = self.first_position?;
        let prefix = rolling_verified_prefix(scheduler, &self.target_tokens, first_position);
        Some(SpdRollingTraceReplay {
            inserted_drafts: self.inserted_drafts,
            missing_proposals: self.missing_proposal_positions.len(),
            first_missing_proposal_position: self.missing_proposal_positions.first().copied(),
            out_of_order_proposals: self.out_of_order_proposal_positions.len(),
            first_out_of_order_proposal_position: self
                .out_of_order_proposal_positions
                .first()
                .copied(),
            verified_windows: self.verified_windows,
            accepted_windows: self.accepted_windows,
            rejected_windows: self.rejected_windows,
            first_rejected_target_position: self.first_rejected_target_position,
            final_pipeline_len: scheduler.pipeline_len(),
            final_verified_up_to: scheduler.verified_up_to(),
            final_verified_prefix_tokens: prefix.tokens,
            verified_prefix_matches_target: prefix.first_mismatch_position.is_none(),
            first_verified_prefix_mismatch_position: prefix.first_mismatch_position,
        })
    }

    pub fn speculation_rows(&self) -> Option<SpdRollingSpeculationRows> {
        self.scheduler.as_ref()?.speculation_rows()
    }

    pub fn draft_plan(&self) -> Option<SpdRollingDraftPlan> {
        Some(SpdRollingDraftPlan {
            scheduler: self.scheduler.as_ref()?.clone(),
        })
    }

    pub fn take_verified_delta(&mut self) -> Option<SpdRollingVerifiedDelta> {
        let scheduler = self.scheduler.as_ref()?;
        let first_position = self.first_position?;
        let start_position = self.released_verified_up_to.unwrap_or(first_position);
        let verified_up_to = scheduler.verified_up_to();
        if start_position >= verified_up_to {
            return None;
        }
        let generated_tokens = scheduler
            .generated_tokens()
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        let mut tokens = Vec::with_capacity(verified_up_to - start_position);
        for position in start_position..verified_up_to {
            let Some(token) = generated_tokens.get(&position).copied() else {
                break;
            };
            tokens.push(token);
        }
        if tokens.is_empty() {
            return None;
        }
        let released_verified_up_to = start_position + tokens.len();
        self.released_verified_up_to = Some(released_verified_up_to);
        Some(SpdRollingVerifiedDelta {
            start_position,
            verified_up_to: released_verified_up_to,
            tokens,
        })
    }

    pub fn advance_to_accepted_context(&mut self, context_tokens: &[i32]) {
        let Some(first_position) = self.first_position else {
            return;
        };
        let accepted_up_to = context_tokens.len().saturating_sub(1);
        if accepted_up_to <= first_position {
            return;
        }
        let Some(scheduler) = self.scheduler.as_mut() else {
            return;
        };
        if scheduler.next_position() >= accepted_up_to {
            return;
        }
        for position in first_position..accepted_up_to {
            let Some(token) = context_tokens.get(position + 1).copied() else {
                break;
            };
            self.target_tokens.insert(position, token);
        }
        scheduler.reset_to_accepted_context(first_position, context_tokens);
        self.promote_pending_proposals();
        self.prune_out_of_order_before(accepted_up_to);
    }

    fn record_target(&mut self, target_position: usize, target: i32) -> Result<usize> {
        self.last_target_position = Some(target_position);
        self.target_tokens.insert(target_position, target);
        if self.scheduler.is_none() {
            self.scheduler = Some(SpdRollingScheduler::new(
                self.logical_stage_count,
                target_position,
                target,
            )?);
            self.first_position = Some(target_position);
        }
        self.verify_ready_windows();
        Ok(self
            .first_position
            .expect("first position is initialized with scheduler"))
    }

    fn mark_missing_position(&mut self, target_position: usize) {
        if !self.proposed_positions.contains(&target_position) {
            self.missing_proposal_positions.insert(target_position);
        }
    }

    fn mark_missing_range(&mut self, start: usize, end: usize) {
        if start >= end {
            return;
        }
        for target_position in start..end {
            self.mark_missing_position(target_position);
        }
    }

    fn mark_out_of_order(&mut self, target_position: usize) {
        self.out_of_order_proposal_positions.insert(target_position);
    }

    fn prune_out_of_order_before(&mut self, accepted_up_to: usize) {
        self.pending_proposals
            .retain(|position, _| *position >= accepted_up_to);
        self.out_of_order_proposal_positions
            .retain(|position| *position >= accepted_up_to);
    }

    fn promote_pending_proposals(&mut self) {
        loop {
            let Some(scheduler) = self.scheduler.as_mut() else {
                return;
            };
            let next_position = scheduler.next_position();
            let Some(proposed) = self.pending_proposals.remove(&next_position) else {
                return;
            };
            if scheduler.insert_draft_at(next_position, proposed).is_err() {
                self.pending_proposals.insert(next_position, proposed);
                return;
            }
            self.out_of_order_proposal_positions.remove(&next_position);
            self.missing_proposal_positions.remove(&next_position);
            self.inserted_drafts += 1;
            self.verify_ready_windows();
        }
    }

    fn verify_ready_windows(&mut self) {
        loop {
            let Some(scheduler) = self.scheduler.as_mut() else {
                return;
            };
            let Some(target_position) = scheduler.oldest_target_position() else {
                return;
            };
            let Some(target_token) = self.target_tokens.get(&target_position).copied() else {
                return;
            };
            match scheduler.verify_oldest(target_token) {
                SpdRollingVerifyOutcome::NotReady => return,
                SpdRollingVerifyOutcome::Accepted { .. } => {
                    self.verified_windows += 1;
                    self.accepted_windows += 1;
                }
                SpdRollingVerifyOutcome::Rejected {
                    target_position, ..
                } => {
                    self.verified_windows += 1;
                    self.rejected_windows += 1;
                    self.first_rejected_target_position = self
                        .first_rejected_target_position
                        .or(Some(target_position));
                }
            }
        }
    }
}

impl SpdRollingDraftPlan {
    pub fn speculation_rows(&self) -> Option<SpdRollingSpeculationRows> {
        self.scheduler.speculation_rows()
    }

    pub fn insert_draft(&mut self, token: i32) -> SpdRollingInsertedDraft {
        self.scheduler.insert_draft(token)
    }

    pub fn next_position(&self) -> usize {
        self.scheduler.next_position()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpdRollingTraceReplay {
    pub inserted_drafts: usize,
    pub missing_proposals: usize,
    pub first_missing_proposal_position: Option<usize>,
    pub out_of_order_proposals: usize,
    pub first_out_of_order_proposal_position: Option<usize>,
    pub verified_windows: usize,
    pub accepted_windows: usize,
    pub rejected_windows: usize,
    pub first_rejected_target_position: Option<usize>,
    pub final_pipeline_len: usize,
    pub final_verified_up_to: usize,
    pub final_verified_prefix_tokens: Vec<i32>,
    pub verified_prefix_matches_target: bool,
    pub first_verified_prefix_mismatch_position: Option<usize>,
}

impl SpdRollingTraceReplay {
    pub fn from_observed_trace(
        stage_count: usize,
        target_tokens: &BTreeMap<usize, i32>,
        proposals: &BTreeMap<usize, i32>,
    ) -> Result<Option<Self>> {
        let Some((&first_position, &first_token)) = target_tokens.first_key_value() else {
            return Ok(None);
        };
        let last_target_position = target_tokens
            .keys()
            .next_back()
            .copied()
            .expect("first target token is present");
        let mut observer = SpdRollingObserver::new(stage_count);
        observer.observe_target(first_position, first_token)?;
        for target_position in first_position + 1..=last_target_position {
            let Some(target) = target_tokens.get(&target_position).copied() else {
                continue;
            };
            observer.observe_probe(
                target_position,
                target,
                proposals.get(&target_position).copied(),
            )?;
            if let Some(accepted_context) =
                accepted_context_from_targets(first_position, target_position, target_tokens)
            {
                observer.advance_to_accepted_context(&accepted_context);
            }
        }
        Ok(observer.trace_replay())
    }
}

fn accepted_context_from_targets(
    first_position: usize,
    target_position: usize,
    target_tokens: &BTreeMap<usize, i32>,
) -> Option<Vec<i32>> {
    let mut context_tokens = vec![0; target_position + 2];
    for position in first_position..=target_position {
        let token = target_tokens.get(&position).copied()?;
        let slot = context_tokens.get_mut(position + 1)?;
        *slot = token;
    }
    Some(context_tokens)
}

struct SpdRollingVerifiedPrefix {
    tokens: Vec<i32>,
    first_mismatch_position: Option<usize>,
}

fn rolling_verified_prefix(
    scheduler: &SpdRollingScheduler,
    target_tokens: &BTreeMap<usize, i32>,
    first_position: usize,
) -> SpdRollingVerifiedPrefix {
    let generated_tokens = scheduler
        .generated_tokens()
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let mut tokens = Vec::new();
    let mut first_mismatch_position = None;
    for position in first_position..scheduler.verified_up_to() {
        let Some(generated) = generated_tokens.get(&position).copied() else {
            first_mismatch_position.get_or_insert(position);
            break;
        };
        tokens.push(generated);
        if target_tokens.get(&position).copied() != Some(generated) {
            first_mismatch_position.get_or_insert(position);
        }
    }
    SpdRollingVerifiedPrefix {
        tokens,
        first_mismatch_position,
    }
}

impl SpdRollingScheduler {
    pub fn new(stage_count: usize, first_position: usize, first_token: i32) -> Result<Self> {
        if stage_count == 0 {
            bail!("SPD rolling scheduler stage_count must be greater than zero");
        }
        let mut pipeline = VecDeque::new();
        pipeline.push_front(SpdRollingEntry {
            position: first_position,
            token: first_token,
        });
        let mut generated = BTreeMap::new();
        generated.insert(first_position, first_token);
        Ok(Self {
            stage_count,
            pipeline,
            generated,
            next_position: first_position + 1,
            verified_up_to: first_position + 1,
            previous_evicted_position: None,
        })
    }

    pub fn speculation_rows(&self) -> Option<SpdRollingSpeculationRows> {
        let newest_position = self.pipeline.front()?.position;
        let oldest_needed = newest_position.checked_sub(self.stage_count - 1)?;
        let mut row_positions = Vec::with_capacity(self.stage_count + 1);
        let mut row_i_stages = Vec::with_capacity(self.stage_count + 1);
        if let Some(position) = self.previous_evicted_position {
            row_positions.push(position);
            row_i_stages.push(self.stage_count);
        }
        for position in oldest_needed..=newest_position {
            row_positions.push(position);
            row_i_stages.push(newest_position - position);
        }
        Some(SpdRollingSpeculationRows {
            evicted_prefix_position: self.previous_evicted_position,
            row_positions,
            row_i_stages,
            newest_position,
            next_draft_position: self.next_position,
        })
    }

    pub fn insert_draft(&mut self, token: i32) -> SpdRollingInsertedDraft {
        let position = self.next_position;
        self.insert_draft_at_next_position(position, token)
    }

    pub fn insert_draft_at(
        &mut self,
        position: usize,
        token: i32,
    ) -> Result<SpdRollingInsertedDraft> {
        if position != self.next_position {
            bail!(
                "SPD rolling draft position {position} does not match next position {}",
                self.next_position
            );
        }
        Ok(self.insert_draft_at_next_position(position, token))
    }

    fn insert_draft_at_next_position(
        &mut self,
        position: usize,
        token: i32,
    ) -> SpdRollingInsertedDraft {
        self.generated.insert(position, token);
        self.pipeline
            .push_front(SpdRollingEntry { position, token });
        self.next_position += 1;
        SpdRollingInsertedDraft {
            position,
            token,
            pipeline_len: self.pipeline.len(),
        }
    }

    pub fn verify_oldest(&mut self, target_token: i32) -> SpdRollingVerifyOutcome {
        if self.pipeline.len() < self.stage_count {
            return SpdRollingVerifyOutcome::NotReady;
        }
        let completed = *self
            .pipeline
            .back()
            .expect("pipeline len checked before oldest verification");
        let target_position = completed.position + 1;
        let Some(speculated) = self.generated.get(&target_position).copied() else {
            return SpdRollingVerifyOutcome::NotReady;
        };
        if speculated == target_token {
            self.pipeline.pop_back();
            self.previous_evicted_position = Some(completed.position);
            self.verified_up_to = target_position + 1;
            return SpdRollingVerifyOutcome::Accepted {
                completed_position: completed.position,
                target_position,
                token: target_token,
            };
        }
        self.generated = self
            .generated
            .iter()
            .filter(|(position, _)| **position < target_position)
            .map(|(position, token)| (*position, *token))
            .collect();
        self.generated.insert(target_position, target_token);
        self.pipeline.clear();
        self.pipeline.push_front(SpdRollingEntry {
            position: target_position,
            token: target_token,
        });
        self.next_position = target_position + 1;
        self.verified_up_to = target_position + 1;
        self.previous_evicted_position = None;
        SpdRollingVerifyOutcome::Rejected {
            completed_position: completed.position,
            target_position,
            speculated,
            corrected: target_token,
        }
    }

    pub fn oldest_target_position(&self) -> Option<usize> {
        if self.pipeline.len() < self.stage_count {
            return None;
        }
        self.pipeline.back().map(|entry| entry.position + 1)
    }

    pub fn pipeline_entries_newest_first(&self) -> Vec<(usize, i32)> {
        self.pipeline
            .iter()
            .map(|entry| (entry.position, entry.token))
            .collect()
    }

    pub fn generated_tokens(&self) -> Vec<(usize, i32)> {
        self.generated
            .iter()
            .map(|(position, token)| (*position, *token))
            .collect()
    }

    pub fn next_position(&self) -> usize {
        self.next_position
    }

    pub fn pipeline_len(&self) -> usize {
        self.pipeline.len()
    }

    pub fn verified_up_to(&self) -> usize {
        self.verified_up_to
    }

    fn reset_to_accepted_context(&mut self, first_position: usize, context_tokens: &[i32]) {
        let accepted_up_to = context_tokens.len().saturating_sub(1);
        if accepted_up_to <= first_position {
            return;
        }
        let last_position = accepted_up_to - 1;
        let Some(last_token) = context_tokens.get(last_position + 1).copied() else {
            return;
        };
        self.generated = (first_position..=last_position)
            .filter_map(|position| {
                context_tokens
                    .get(position + 1)
                    .copied()
                    .map(|token| (position, token))
            })
            .collect();
        self.pipeline.clear();
        self.pipeline.push_front(SpdRollingEntry {
            position: last_position,
            token: last_token,
        });
        self.next_position = last_position + 1;
        self.verified_up_to = last_position + 1;
        self.previous_evicted_position = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_scheduler_accepts_oldest_after_pipeline_fill() {
        let mut scheduler = SpdRollingScheduler::new(3, 2, 100).unwrap();

        assert_eq!(
            scheduler.speculation_rows().unwrap(),
            SpdRollingSpeculationRows {
                evicted_prefix_position: None,
                row_positions: vec![0, 1, 2],
                row_i_stages: vec![2, 1, 0],
                newest_position: 2,
                next_draft_position: 3,
            }
        );
        assert_eq!(
            scheduler.insert_draft(101),
            SpdRollingInsertedDraft {
                position: 3,
                token: 101,
                pipeline_len: 2,
            }
        );
        assert_eq!(
            scheduler.verify_oldest(101),
            SpdRollingVerifyOutcome::NotReady
        );
        scheduler.insert_draft(102);

        assert_eq!(scheduler.oldest_target_position(), Some(3));
        assert_eq!(
            scheduler.verify_oldest(101),
            SpdRollingVerifyOutcome::Accepted {
                completed_position: 2,
                target_position: 3,
                token: 101,
            }
        );
        assert_eq!(
            scheduler.pipeline_entries_newest_first(),
            vec![(4, 102), (3, 101)]
        );
        assert_eq!(scheduler.verified_up_to(), 4);
        assert_eq!(
            scheduler.speculation_rows().unwrap(),
            SpdRollingSpeculationRows {
                evicted_prefix_position: Some(2),
                row_positions: vec![2, 2, 3, 4],
                row_i_stages: vec![3, 2, 1, 0],
                newest_position: 4,
                next_draft_position: 5,
            }
        );
    }

    #[test]
    fn rolling_scheduler_rejection_resets_pipeline_to_corrected_token() {
        let mut scheduler = SpdRollingScheduler::new(3, 2, 100).unwrap();
        scheduler.insert_draft(101);
        scheduler.insert_draft(102);
        assert!(matches!(
            scheduler.verify_oldest(101),
            SpdRollingVerifyOutcome::Accepted { .. }
        ));
        scheduler.insert_draft(103);

        assert_eq!(
            scheduler.verify_oldest(999),
            SpdRollingVerifyOutcome::Rejected {
                completed_position: 3,
                target_position: 4,
                speculated: 102,
                corrected: 999,
            }
        );
        assert_eq!(scheduler.pipeline_entries_newest_first(), vec![(4, 999)]);
        assert_eq!(
            scheduler.generated_tokens(),
            vec![(2, 100), (3, 101), (4, 999)]
        );
        assert_eq!(scheduler.next_position(), 5);
        assert_eq!(scheduler.verified_up_to(), 5);
        assert_eq!(
            scheduler.speculation_rows().unwrap(),
            SpdRollingSpeculationRows {
                evicted_prefix_position: None,
                row_positions: vec![2, 3, 4],
                row_i_stages: vec![2, 1, 0],
                newest_position: 4,
                next_draft_position: 5,
            }
        );
    }

    #[test]
    fn rolling_scheduler_rejects_out_of_order_draft_insert() {
        let mut scheduler = SpdRollingScheduler::new(3, 2, 100).unwrap();

        assert!(scheduler.insert_draft_at(4, 102).is_err());
        assert_eq!(
            scheduler.insert_draft_at(3, 101).unwrap(),
            SpdRollingInsertedDraft {
                position: 3,
                token: 101,
                pipeline_len: 2,
            }
        );
        assert_eq!(scheduler.next_position(), 4);
    }

    #[test]
    fn rolling_scheduler_requires_enough_prefill_positions_for_rows() {
        let scheduler = SpdRollingScheduler::new(4, 1, 10).unwrap();

        assert_eq!(scheduler.speculation_rows(), None);
    }

    #[test]
    fn rolling_trace_replay_reports_verified_prefix_after_rejection() {
        let target_tokens = BTreeMap::from([(0, 10), (1, 11), (2, 12), (3, 13), (4, 14)]);
        let proposals = BTreeMap::from([(1, 11), (2, 99), (3, 13), (4, 14)]);

        let replay = SpdRollingTraceReplay::from_observed_trace(4, &target_tokens, &proposals)
            .unwrap()
            .unwrap();

        assert_eq!(replay.inserted_drafts, 4);
        assert_eq!(replay.missing_proposals, 0);
        assert_eq!(replay.out_of_order_proposals, 0);
        assert_eq!(replay.verified_windows, 2);
        assert_eq!(replay.accepted_windows, 1);
        assert_eq!(replay.rejected_windows, 1);
        assert_eq!(replay.first_rejected_target_position, Some(2));
        assert_eq!(replay.final_pipeline_len, 1);
        assert_eq!(replay.final_verified_up_to, 5);
        assert_eq!(
            replay.final_verified_prefix_tokens,
            vec![10, 11, 12, 13, 14]
        );
        assert!(replay.verified_prefix_matches_target);
        assert_eq!(replay.first_verified_prefix_mismatch_position, None);
    }

    #[test]
    fn rolling_trace_replay_does_not_shift_over_missing_position() {
        let target_tokens = BTreeMap::from([(0, 10), (1, 11), (2, 12), (3, 13), (4, 14), (5, 15)]);
        let proposals = BTreeMap::from([(1, 11), (2, 12), (4, 14), (5, 15)]);

        let replay = SpdRollingTraceReplay::from_observed_trace(4, &target_tokens, &proposals)
            .unwrap()
            .unwrap();

        assert_eq!(replay.inserted_drafts, 4);
        assert_eq!(replay.missing_proposals, 1);
        assert_eq!(replay.first_missing_proposal_position, Some(3));
        assert_eq!(replay.out_of_order_proposals, 0);
        assert_eq!(replay.first_out_of_order_proposal_position, None);
        assert_eq!(replay.verified_windows, 0);
        assert_eq!(replay.final_pipeline_len, 3);
        assert_eq!(replay.final_verified_up_to, 4);
        assert_eq!(replay.final_verified_prefix_tokens, vec![10, 11, 12, 13]);
        assert!(replay.verified_prefix_matches_target);
    }

    #[test]
    fn rolling_trace_replay_does_not_fabricate_missing_target_tokens() {
        let target_tokens = BTreeMap::from([(0, 10), (1, 11), (3, 13), (4, 14)]);
        let proposals = BTreeMap::from([(1, 11), (3, 13), (4, 14)]);

        let replay = SpdRollingTraceReplay::from_observed_trace(3, &target_tokens, &proposals)
            .unwrap()
            .unwrap();

        assert_eq!(replay.final_verified_prefix_tokens, vec![10]);
        assert!(replay.verified_prefix_matches_target);
        assert_eq!(replay.final_verified_up_to, 1);
    }

    #[test]
    fn rolling_trace_replay_supports_non_zero_initial_position() {
        let target_tokens = BTreeMap::from([(2, 100), (3, 101), (4, 102)]);
        let proposals = BTreeMap::from([(3, 101), (4, 102)]);

        let replay = SpdRollingTraceReplay::from_observed_trace(3, &target_tokens, &proposals)
            .unwrap()
            .unwrap();

        assert_eq!(replay.inserted_drafts, 2);
        assert_eq!(replay.verified_windows, 1);
        assert_eq!(replay.accepted_windows, 1);
        assert_eq!(replay.final_verified_up_to, 4);
        assert_eq!(replay.final_verified_prefix_tokens, vec![100, 101]);
        assert!(replay.verified_prefix_matches_target);
    }

    #[test]
    fn rolling_observer_releases_verified_deltas_after_acceptance() {
        let mut observer = SpdRollingObserver::new(3);

        observer.observe_probe(0, 10, Some(99)).unwrap();
        assert_eq!(
            observer.take_verified_delta(),
            Some(SpdRollingVerifiedDelta {
                start_position: 0,
                verified_up_to: 1,
                tokens: vec![10],
            })
        );
        assert_eq!(observer.take_verified_delta(), None);

        observer.observe_probe(1, 11, Some(11)).unwrap();
        assert_eq!(observer.take_verified_delta(), None);
        observer.observe_probe(2, 12, Some(12)).unwrap();
        assert_eq!(
            observer.take_verified_delta(),
            Some(SpdRollingVerifiedDelta {
                start_position: 1,
                verified_up_to: 2,
                tokens: vec![11],
            })
        );
    }

    #[test]
    fn rolling_observer_catches_up_after_rejection_to_accepted_context() {
        let mut observer = SpdRollingObserver::new(4);
        let mut context_tokens = vec![0; 28];
        context_tokens[23] = 198;
        context_tokens[24] = 90700;
        context_tokens[25] = 8340;
        context_tokens[26] = 25;
        context_tokens[27] = 271;

        observer.observe_probe(23, 90700, Some(8160)).unwrap();
        observer.observe_probe(24, 8340, Some(264)).unwrap();
        observer.observe_probe(25, 25, Some(25)).unwrap();
        let rejected = observer.observe_probe(26, 271, Some(25)).unwrap();

        assert_eq!(rejected.first_rejected_target_position, Some(24));
        assert_eq!(rejected.next_position, Some(25));
        assert_eq!(rejected.pipeline_len, Some(1));

        observer.advance_to_accepted_context(&context_tokens);
        let caught_up = observer.snapshot();

        assert_eq!(caught_up.next_position, Some(27));
        assert_eq!(caught_up.pipeline_len, Some(1));
        assert_eq!(caught_up.verified_up_to, Some(27));
        assert_eq!(
            observer.speculation_rows().unwrap(),
            SpdRollingSpeculationRows {
                evicted_prefix_position: None,
                row_positions: vec![23, 24, 25, 26],
                row_i_stages: vec![3, 2, 1, 0],
                newest_position: 26,
                next_draft_position: 27,
            }
        );

        let refilled = observer.observe_probe(27, 16, Some(16)).unwrap();
        assert_eq!(refilled.out_of_order_proposals, 0);
        assert_eq!(refilled.next_position, Some(28));
    }

    #[test]
    fn rolling_observer_prunes_stale_gap_diagnostics_after_context_catch_up() {
        let mut observer = SpdRollingObserver::new(4);
        let mut context_tokens = vec![0; 29];
        context_tokens[24] = 90700;
        context_tokens[25] = 8340;
        context_tokens[26] = 25;
        context_tokens[27] = 271;
        context_tokens[28] = 16;

        observer.observe_probe(24, 8340, Some(883)).unwrap();
        let stale_gap = observer.observe_probe(26, 271, Some(271)).unwrap();

        assert_eq!(stale_gap.missing_proposals, 1);
        assert_eq!(stale_gap.first_missing_proposal_position, Some(25));
        assert_eq!(stale_gap.out_of_order_proposals, 1);
        assert_eq!(stale_gap.first_out_of_order_proposal_position, Some(26));

        observer.advance_to_accepted_context(&context_tokens);
        let caught_up = observer.snapshot();

        assert_eq!(caught_up.missing_proposals, 1);
        assert_eq!(caught_up.first_missing_proposal_position, Some(25));
        assert_eq!(caught_up.out_of_order_proposals, 0);
        assert_eq!(caught_up.first_out_of_order_proposal_position, None);
        assert_eq!(caught_up.next_position, Some(28));
    }

    #[test]
    fn rolling_observer_promotes_early_proposal_after_context_catch_up() {
        let mut observer = SpdRollingObserver::new(4);
        let mut context_tokens = vec![0; 27];
        context_tokens[25] = 90700;
        context_tokens[26] = 8340;

        observer.observe_probe(24, 90700, Some(760)).unwrap();
        let early = observer.observe_probe(26, 25, Some(25)).unwrap();

        assert_eq!(early.missing_proposals, 1);
        assert_eq!(early.first_missing_proposal_position, Some(25));
        assert_eq!(early.out_of_order_proposals, 1);
        assert_eq!(early.first_out_of_order_proposal_position, Some(26));
        assert_eq!(early.next_position, Some(25));

        observer.advance_to_accepted_context(&context_tokens);
        let caught_up = observer.snapshot();

        assert_eq!(caught_up.next_position, Some(27));
        assert_eq!(caught_up.pipeline_len, Some(2));
        assert_eq!(caught_up.inserted_drafts, 1);
        assert_eq!(caught_up.missing_proposals, 1);
        assert_eq!(caught_up.first_missing_proposal_position, Some(25));
        assert_eq!(caught_up.out_of_order_proposals, 0);
        assert_eq!(caught_up.first_out_of_order_proposal_position, None);
    }

    #[test]
    fn rolling_observer_exposes_reference_row_stage_roles() {
        let mut observer = SpdRollingObserver::new(3);

        observer.observe_probe(0, 10, Some(99)).unwrap();
        observer.observe_probe(1, 11, Some(11)).unwrap();
        observer.observe_probe(2, 12, Some(12)).unwrap();

        assert_eq!(
            observer.speculation_rows(),
            Some(SpdRollingSpeculationRows {
                evicted_prefix_position: Some(0),
                row_positions: vec![0, 0, 1, 2],
                row_i_stages: vec![3, 2, 1, 0],
                newest_position: 2,
                next_draft_position: 3,
            })
        );
    }

    #[test]
    fn rolling_draft_plan_advances_rows_without_mutating_observer() {
        let mut observer = SpdRollingObserver::new(3);
        observer.observe_probe(0, 10, Some(99)).unwrap();
        observer.observe_probe(1, 11, Some(11)).unwrap();
        observer.observe_probe(2, 12, Some(12)).unwrap();

        let before = observer.snapshot();
        let mut plan = observer.draft_plan().expect("scheduler initialized");

        assert_eq!(plan.next_position(), 3);
        assert_eq!(
            plan.speculation_rows().unwrap(),
            SpdRollingSpeculationRows {
                evicted_prefix_position: Some(0),
                row_positions: vec![0, 0, 1, 2],
                row_i_stages: vec![3, 2, 1, 0],
                newest_position: 2,
                next_draft_position: 3,
            }
        );

        plan.insert_draft(13);

        assert_eq!(plan.next_position(), 4);
        assert_eq!(
            plan.speculation_rows().unwrap(),
            SpdRollingSpeculationRows {
                evicted_prefix_position: Some(0),
                row_positions: vec![0, 1, 2, 3],
                row_i_stages: vec![3, 2, 1, 0],
                newest_position: 3,
                next_draft_position: 4,
            }
        );
        assert_eq!(observer.snapshot(), before);
    }

    #[test]
    fn rolling_observer_releases_corrected_token_after_rejection() {
        let mut observer = SpdRollingObserver::new(4);

        observer.observe_probe(0, 90700, Some(8160)).unwrap();
        assert_eq!(
            observer.take_verified_delta(),
            Some(SpdRollingVerifiedDelta {
                start_position: 0,
                verified_up_to: 1,
                tokens: vec![90700],
            })
        );
        observer.observe_probe(1, 8340, Some(264)).unwrap();
        observer.observe_probe(2, 25, Some(25)).unwrap();
        observer.observe_probe(3, 271, Some(25)).unwrap();

        assert_eq!(
            observer.take_verified_delta(),
            Some(SpdRollingVerifiedDelta {
                start_position: 1,
                verified_up_to: 2,
                tokens: vec![8340],
            })
        );
        assert_eq!(observer.take_verified_delta(), None);
    }
}
