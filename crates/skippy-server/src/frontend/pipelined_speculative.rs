use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

use super::embedded_generation::VerifySpanKvRepair;
use super::*;

pub(super) struct PipelinedSpeculativeTask<'a, 'b> {
    pub(super) request: &'a EmbeddedStageZeroGeneration<'b>,
    pub(super) downstream: &'a mut TcpStream,
    pub(super) session_key: &'a str,
    pub(super) request_id: u64,
    pub(super) session_id: u64,
    pub(super) prefill_token_count: usize,
    pub(super) wire_sampling: Option<WireSamplingConfig>,
    pub(super) draft: Arc<Mutex<DraftRunner>>,
    pub(super) draft_window: usize,
    pub(super) context_tokens: &'a mut Vec<i32>,
    pub(super) current: i32,
    pub(super) decoded_tokens: usize,
    pub(super) adaptive_window: &'a mut usize,
    pub(super) max_speculative_window: usize,
    pub(super) speculative_stats: &'a mut OpenAiSpeculativeStats,
}

#[derive(Default)]
pub(super) struct PipelinedSpeculativeOutcome {
    pub(super) current: i32,
    pub(super) decoded_tokens: usize,
    pub(super) reached_stop: bool,
    pub(super) stats: PipelinedSpeculativeDecodeStats,
}

#[derive(Default)]
pub(super) struct PipelinedSpeculativeDecodeStats {
    pub(super) stage0_compute_ms: f64,
    pub(super) runtime_lock_wait_ms: f64,
    pub(super) runtime_lock_wait_max_ms: f64,
    pub(super) runtime_lock_hold_ms: f64,
    pub(super) runtime_lock_hold_max_ms: f64,
    pub(super) runtime_lock_acquires: usize,
    pub(super) activation_encode_ms: f64,
    pub(super) output_activation_bytes: usize,
    pub(super) forward_activation_bytes: usize,
    pub(super) forward_write_ms: f64,
    pub(super) downstream_wait_ms: f64,
}

impl PipelinedSpeculativeDecodeStats {
    fn observe_execution(&mut self, execution: &EmbeddedStageExecution) {
        self.stage0_compute_ms += execution.stats.stage0_compute_ms;
        self.runtime_lock_wait_ms += execution.stats.runtime_lock_wait_ms;
        self.runtime_lock_wait_max_ms = self
            .runtime_lock_wait_max_ms
            .max(execution.stats.runtime_lock_wait_ms);
        self.runtime_lock_hold_ms += execution.stats.runtime_lock_hold_ms;
        self.runtime_lock_hold_max_ms = self
            .runtime_lock_hold_max_ms
            .max(execution.stats.runtime_lock_hold_ms);
        self.runtime_lock_acquires += 1;
        self.activation_encode_ms += execution.stats.activation_encode_ms;
        self.output_activation_bytes = self
            .output_activation_bytes
            .saturating_add(execution.stats.output_activation_bytes);
        self.forward_activation_bytes = self
            .forward_activation_bytes
            .saturating_add(execution.stats.forward_activation_bytes);
        self.forward_write_ms += execution.stats.forward_write_ms;
        self.downstream_wait_ms += execution.stats.downstream_wait_ms;
    }
}

struct InFlightVerify {
    window_index: usize,
    window_epoch: i32,
    decoded_at_send: usize,
    draft_tokens: Vec<i32>,
    verify_inputs: Vec<i32>,
    message: StageWireMessage,
    pending: EmbeddedStagePendingExecution,
}

#[derive(Default)]
struct PipelinedWindowQueues {
    inflight: VecDeque<InFlightVerify>,
    stale_debt: VecDeque<InFlightVerify>,
    primed_draft: Option<AsyncDraftWindow>,
    draft_exhausted: bool,
}

impl PipelinedWindowQueues {
    fn mark_remaining_inflight_stale(&mut self) -> usize {
        let stale_count = self.inflight.len();
        self.stale_debt.append(&mut self.inflight);
        self.draft_exhausted = false;
        stale_count
    }
}

struct PreparedDraftWindow {
    draft_tokens: Vec<i32>,
    draft_propose_ms: f64,
}

struct AsyncDraftWindow {
    handle: JoinHandle<Result<PreparedDraftWindow, String>>,
}

struct PipelinedDebugWindow {
    window_index: usize,
    window_epoch: i32,
    decoded_at_send: usize,
    message_decode_step: i32,
    identity_present: bool,
    identity_matches: bool,
    proposed_count: usize,
    verify_input_count: usize,
}

struct PipelinedWindowOutcome {
    rejected: bool,
    repair: Option<PipelinedRejectRepair>,
}

struct PipelinedRejectRepair {
    decoded_at_send: usize,
    current_at_send: i32,
    commit_tokens: Vec<i32>,
}

impl StageOpenAiBackend {
    pub(super) fn generate_pipelined_speculative_tokens(
        &self,
        mut task: PipelinedSpeculativeTask<'_, '_>,
        mut on_token: impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<PipelinedSpeculativeOutcome> {
        let depth = task.request.pipelined_speculative_depth.max(1);
        task.speculative_stats.pipelined_depth = depth;
        let max_tokens = task.request.max_tokens as usize;
        let mut queues = PipelinedWindowQueues::default();
        let mut outcome = PipelinedSpeculativeOutcome {
            current: task.current,
            decoded_tokens: task.decoded_tokens,
            reached_stop: false,
            stats: PipelinedSpeculativeDecodeStats::default(),
        };
        let mut scheduler = ShardPipelineScheduler::new(ShardPipelineConfig {
            depth,
            max_decoded: max_tokens,
            prefill_token_count: task.prefill_token_count,
            current: outcome.current,
            decoded: outcome.decoded_tokens,
        });

        while outcome.decoded_tokens < max_tokens && !outcome.reached_stop {
            if task
                .request
                .cancellation
                .is_some_and(openai_frontend::CancellationToken::is_cancelled)
            {
                break;
            }
            self.fill_pipelined_speculative_window(&mut task, &mut queues, &mut scheduler)?;
            if !queues.stale_debt.is_empty() {
                self.drain_one_stale_pipelined_window(
                    &mut task,
                    &mut queues.stale_debt,
                    &mut scheduler,
                    &mut outcome,
                )?;
                continue;
            }
            let Some(window) = queues.inflight.pop_front() else {
                break;
            };
            let returned = scheduler.mark_inflight_returned();
            debug_assert_pipelined_scheduler_window(&returned, &window);
            let window_outcome = self.receive_pipelined_speculative_window(
                &mut task,
                window,
                &mut outcome,
                &mut on_token,
            )?;
            if window_outcome.rejected || outcome.reached_stop {
                self.discard_async_draft_window(&mut task, queues.primed_draft.take())?;
                let recovery = scheduler.reject_to(outcome.current, outcome.decoded_tokens);
                let stale_count = queues.mark_remaining_inflight_stale();
                debug_assert_eq!(recovery.stale_windows, stale_count);
                self.recover_pipelined_speculation(&mut task, &outcome, window_outcome.repair)?;
            }
        }

        self.discard_async_draft_window(&mut task, queues.primed_draft.take())?;
        if !queues.inflight.is_empty() {
            let recovery = scheduler.cancel_inflight();
            let stale_count = queues.mark_remaining_inflight_stale();
            debug_assert_eq!(recovery.stale_windows, stale_count);
            self.recover_pipelined_speculation(&mut task, &outcome, None)?;
        }
        if !queues.stale_debt.is_empty() {
            self.drain_stale_pipelined_debt(
                &mut task,
                &mut queues.stale_debt,
                &mut scheduler,
                &mut outcome,
            )?;
        }
        task.speculative_stats.adaptive_window_final = *task.adaptive_window;
        Ok(outcome)
    }

    fn fill_pipelined_speculative_window(
        &self,
        task: &mut PipelinedSpeculativeTask<'_, '_>,
        queues: &mut PipelinedWindowQueues,
        scheduler: &mut ShardPipelineScheduler,
    ) -> OpenAiResult<()> {
        while scheduler.can_send() && !queues.draft_exhausted {
            self.ensure_pipelined_draft_window_primed(
                task,
                &mut queues.primed_draft,
                scheduler,
                &mut queues.draft_exhausted,
            )?;
            let Some(prepared) =
                self.join_pipelined_draft_window(task, queues.primed_draft.take())?
            else {
                queues.draft_exhausted = true;
                break;
            };
            self.send_prepared_pipelined_window(task, &mut queues.inflight, prepared, scheduler)?;
        }
        self.ensure_pipelined_draft_window_primed(
            task,
            &mut queues.primed_draft,
            scheduler,
            &mut queues.draft_exhausted,
        )?;
        Ok(())
    }

    fn ensure_pipelined_draft_window_primed(
        &self,
        task: &mut PipelinedSpeculativeTask<'_, '_>,
        primed_draft: &mut Option<AsyncDraftWindow>,
        scheduler: &ShardPipelineScheduler,
        draft_exhausted: &mut bool,
    ) -> OpenAiResult<()> {
        if primed_draft.is_some() || !scheduler.can_prime_draft() || *draft_exhausted {
            return Ok(());
        }
        let Some(proposal_limit) =
            scheduler.proposal_limit(task.max_speculative_window, task.draft_window)
        else {
            *draft_exhausted = true;
            return Ok(());
        };
        let cursor = scheduler.cursor();
        *primed_draft = self.spawn_pipelined_draft_window(task, cursor.current, proposal_limit)?;
        Ok(())
    }

    fn spawn_pipelined_draft_window(
        &self,
        task: &mut PipelinedSpeculativeTask<'_, '_>,
        current: i32,
        proposal_limit: usize,
    ) -> OpenAiResult<Option<AsyncDraftWindow>> {
        if proposal_limit == 0 {
            return Ok(None);
        }
        let draft = Arc::clone(&task.draft);
        let handle = std::thread::Builder::new()
            .name("skippy-pipelined-draft".to_string())
            .spawn(move || {
                let propose_timer = PhaseTimer::start();
                let mut draft = draft
                    .lock()
                    .map_err(|_| "draft model lock poisoned".to_string())?;
                let draft_tokens = draft
                    .propose(current, proposal_limit)
                    .map_err(|error| error.to_string())?;
                Ok(PreparedDraftWindow {
                    draft_tokens,
                    draft_propose_ms: propose_timer.elapsed_ms(),
                })
            })
            .map_err(openai_io_error)?;
        task.speculative_stats.pipelined_async_draft_windows += 1;
        Ok(Some(AsyncDraftWindow { handle }))
    }

    fn join_pipelined_draft_window(
        &self,
        task: &mut PipelinedSpeculativeTask<'_, '_>,
        draft: Option<AsyncDraftWindow>,
    ) -> OpenAiResult<Option<PreparedDraftWindow>> {
        let Some(draft) = draft else {
            return Ok(None);
        };
        let join_timer = PhaseTimer::start();
        let prepared = draft
            .handle
            .join()
            .map_err(|_| OpenAiError::backend("pipelined draft worker panicked"))?
            .map_err(|error| OpenAiError::backend(format!("pipelined draft failed: {error}")))?;
        task.speculative_stats.pipelined_async_draft_wait_ms += join_timer.elapsed_ms();
        if prepared.draft_tokens.is_empty() {
            return Ok(None);
        }
        Ok(Some(prepared))
    }

    fn discard_async_draft_window(
        &self,
        task: &mut PipelinedSpeculativeTask<'_, '_>,
        draft: Option<AsyncDraftWindow>,
    ) -> OpenAiResult<()> {
        if self.join_pipelined_draft_window(task, draft)?.is_some() {
            task.speculative_stats.pipelined_stale_draft_windows += 1;
        }
        Ok(())
    }

    fn send_prepared_pipelined_window(
        &self,
        task: &mut PipelinedSpeculativeTask<'_, '_>,
        inflight: &mut VecDeque<InFlightVerify>,
        prepared: PreparedDraftWindow,
        scheduler: &mut ShardPipelineScheduler,
    ) -> OpenAiResult<()> {
        let PreparedDraftWindow {
            draft_tokens,
            draft_propose_ms,
        } = prepared;
        if draft_tokens.is_empty() {
            return Ok(());
        }
        let Some(window) = scheduler.prepare_window(&draft_tokens) else {
            return Ok(());
        };
        let mut message = embedded_verify_message(
            task.request.wire_dtype,
            VerifySpanMessageArgs {
                request_id: task.request_id,
                session_id: task.session_id,
                prompt_token_count: task.request.prompt_token_ids.len(),
                pos_start: window.pos_start,
                decode_step: window.decode_step,
                tokens: &window.verify_inputs,
                sampling: task.wire_sampling.clone(),
                checkpoint: false,
            },
        )?;
        message.state.flags |= state_flags::IDENTIFIED_REPLY;
        message.state.seq_id = i32::try_from(window.window_index)
            .map_err(|_| OpenAiError::backend("pipelined window index exceeds i32"))?;
        message.state.checkpoint_generation = window.epoch;
        let pending = self.send_embedded_stage_message(
            task.request,
            task.downstream,
            task.session_key,
            &message,
            &window.verify_inputs,
        )?;
        let window_index = window.window_index;
        let window_epoch = window.epoch;
        let decoded_at_send = window.decoded_at_send;
        let verify_inputs = window.verify_inputs.clone();
        scheduler.mark_sent(window);
        task.speculative_stats.draft_propose_ms += draft_propose_ms;
        task.speculative_stats.windows += 1;
        task.speculative_stats.draft_tokens += draft_tokens.len();
        task.speculative_stats.primary_verify_requests += 1;
        task.speculative_stats.primary_verify_tokens += verify_inputs.len();
        task.speculative_stats.pipelined_sent_windows += 1;
        inflight.push_back(InFlightVerify {
            window_index,
            window_epoch,
            decoded_at_send,
            draft_tokens,
            verify_inputs,
            message,
            pending,
        });
        task.speculative_stats.pipelined_max_inflight_windows = task
            .speculative_stats
            .pipelined_max_inflight_windows
            .max(inflight.len());
        Ok(())
    }

    fn receive_pipelined_speculative_window(
        &self,
        task: &mut PipelinedSpeculativeTask<'_, '_>,
        window: InFlightVerify,
        outcome: &mut PipelinedSpeculativeOutcome,
        on_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<PipelinedWindowOutcome> {
        let InFlightVerify {
            window_index,
            window_epoch,
            decoded_at_send,
            draft_tokens,
            verify_inputs,
            message,
            pending,
        } = window;
        observe_pipelined_fifo_window(task.speculative_stats, decoded_at_send, &message);
        task.speculative_stats.pipelined_committed_windows += 1;
        let proposed_count = draft_tokens.len();
        let verify_input_count = verify_inputs.len();
        let current_at_send = verify_inputs
            .first()
            .copied()
            .ok_or_else(|| OpenAiError::backend("pipelined verify window has no current token"))?;
        let verify = self.receive_embedded_stage_execution(
            task.request,
            task.downstream,
            pending,
            &message,
            WireReplyKind::PredictedTokens,
        )?;
        let identity_matches = validate_pipelined_reply_identity(
            task.speculative_stats,
            &message,
            verify.reply_identity,
        )?;
        observe_pipelined_primary_verify(task.speculative_stats, &verify);
        outcome.stats.observe_execution(&verify);
        let decision = classify_verify_span(
            &draft_tokens,
            &verify.reply.predicted_tokens,
            outcome.decoded_tokens,
            task.request.max_tokens as usize,
            |token| token_is_eog_with_runtime(&self.runtime, token),
        )?;
        task.speculative_stats.observe_verify_decision(
            decision,
            &mut *task.adaptive_window,
            task.request.adaptive_speculative_window,
            task.max_speculative_window,
        );
        let commit_count = shard_linear_commit_count(&decision, draft_tokens.len());
        let commit_tokens = verify.reply.predicted_tokens[..commit_count].to_vec();
        let mut emitted_commit_tokens = Vec::with_capacity(commit_tokens.len());
        for token in commit_tokens.iter().copied() {
            let token_decode_step = outcome.decoded_tokens;
            outcome.current = token;
            outcome.decoded_tokens += 1;
            task.context_tokens.push(token);
            emitted_commit_tokens.push(token);
            self.emit_speculative_commit_token_debug(
                task.request.ids,
                token_decode_step,
                token,
                "VerifySpan",
                "pipelined-draft",
                Some(window_index),
            );
            if on_token(token)? == TokenControl::Stop {
                outcome.reached_stop = true;
            }
            if outcome.reached_stop || outcome.decoded_tokens >= task.request.max_tokens as usize {
                break;
            }
        }
        if self.telemetry.is_debug_enabled() {
            self.emit_pipelined_speculative_debug(
                task,
                PipelinedDebugWindow {
                    window_index,
                    window_epoch,
                    decoded_at_send,
                    message_decode_step: message.state.decode_step,
                    identity_present: verify.reply_identity.is_some(),
                    identity_matches,
                    proposed_count,
                    verify_input_count,
                },
                &verify,
                decision,
            );
        }
        let rejected = decision.rejected();
        let repair = rejected.then_some(PipelinedRejectRepair {
            decoded_at_send,
            current_at_send,
            commit_tokens: emitted_commit_tokens,
        });
        Ok(PipelinedWindowOutcome { rejected, repair })
    }

    fn drain_one_stale_pipelined_window(
        &self,
        task: &mut PipelinedSpeculativeTask<'_, '_>,
        stale_debt: &mut VecDeque<InFlightVerify>,
        scheduler: &mut ShardPipelineScheduler,
        outcome: &mut PipelinedSpeculativeOutcome,
    ) -> OpenAiResult<()> {
        let Some(window) = stale_debt.pop_front() else {
            return Ok(());
        };
        let returned = scheduler.mark_stale_returned();
        debug_assert_pipelined_scheduler_window(&returned, &window);
        observe_pipelined_fifo_window(
            task.speculative_stats,
            window.decoded_at_send,
            &window.message,
        );
        let stale = self.receive_embedded_stage_execution(
            task.request,
            task.downstream,
            window.pending,
            &window.message,
            WireReplyKind::PredictedTokens,
        )?;
        validate_pipelined_reply_identity(
            task.speculative_stats,
            &window.message,
            stale.reply_identity,
        )?;
        observe_pipelined_primary_verify(task.speculative_stats, &stale);
        outcome.stats.observe_execution(&stale);
        task.speculative_stats.pipelined_stale_windows += 1;
        Ok(())
    }

    fn drain_stale_pipelined_debt(
        &self,
        task: &mut PipelinedSpeculativeTask<'_, '_>,
        stale_debt: &mut VecDeque<InFlightVerify>,
        scheduler: &mut ShardPipelineScheduler,
        outcome: &mut PipelinedSpeculativeOutcome,
    ) -> OpenAiResult<()> {
        while !stale_debt.is_empty() {
            self.drain_one_stale_pipelined_window(task, stale_debt, scheduler, outcome)?;
        }
        Ok(())
    }

    fn recover_pipelined_speculation(
        &self,
        task: &mut PipelinedSpeculativeTask<'_, '_>,
        outcome: &PipelinedSpeculativeOutcome,
        repair: Option<PipelinedRejectRepair>,
    ) -> OpenAiResult<()> {
        if let Some(repair) = repair {
            let repair = self.repair_rejected_verify_span_kv(VerifySpanKvRepair {
                request: task.request,
                downstream: task.downstream,
                session_key: task.session_key,
                request_id: task.request_id,
                session_id: task.session_id,
                prefill_token_count: task.prefill_token_count,
                decoded_tokens: repair.decoded_at_send,
                current: repair.current_at_send,
                commit_tokens: &repair.commit_tokens,
                wire_sampling: task.wire_sampling.clone(),
            })?;
            task.speculative_stats.recovery_restores += 1;
            task.speculative_stats.recovery_ms += repair.elapsed_ms;
            task.speculative_stats.recovery_restore_local_ms += repair.trim.local_ms;
            task.speculative_stats.recovery_restore_downstream_write_ms +=
                repair.trim.downstream_write_ms;
            task.speculative_stats.recovery_restore_downstream_wait_ms +=
                repair.trim.downstream_wait_ms;
        } else {
            let trim = self.trim_embedded_stage_session_local(
                task.session_key,
                task.prefill_token_count + outcome.decoded_tokens,
            )?;
            task.speculative_stats.recovery_ms += trim.elapsed_ms;
            task.speculative_stats.recovery_restore_local_ms += trim.local_ms;
        }
        let draft_reset_timer = PhaseTimer::start();
        task.draft
            .lock()
            .map_err(|_| OpenAiError::backend("draft model lock poisoned"))?
            .reset_to_context(task.context_tokens.as_slice())
            .map_err(openai_backend_error)?;
        task.speculative_stats.draft_reset_ms += draft_reset_timer.elapsed_ms();
        Ok(())
    }

    fn emit_pipelined_speculative_debug(
        &self,
        task: &PipelinedSpeculativeTask<'_, '_>,
        window: PipelinedDebugWindow,
        verify: &EmbeddedStageExecution,
        decision: VerifySpanDecision,
    ) {
        let mut attrs = self.openai_attrs(task.request.ids);
        attrs.insert(
            "llama_stage.spec.pipelined_window_index".to_string(),
            json!(window.window_index),
        );
        attrs.insert(
            "llama_stage.spec.pipelined_window_epoch".to_string(),
            json!(window.window_epoch),
        );
        attrs.insert(
            "llama_stage.decode_step".to_string(),
            json!(window.decoded_at_send),
        );
        attrs.insert(
            "llama_stage.spec.message_decode_step".to_string(),
            json!(window.message_decode_step),
        );
        attrs.insert(
            "llama_stage.spec.pipelined_fifo_order_ok".to_string(),
            json!(pipelined_window_decode_step_matches(
                window.decoded_at_send,
                window.message_decode_step
            )),
        );
        attrs.insert(
            "llama_stage.spec.pipelined_identity_present".to_string(),
            json!(window.identity_present),
        );
        attrs.insert(
            "llama_stage.spec.pipelined_identity_matches".to_string(),
            json!(window.identity_matches),
        );
        attrs.insert("llama_stage.message_kind".to_string(), json!("VerifySpan"));
        attrs.insert("llama_stage.spec.pipelined".to_string(), json!(true));
        attrs.insert(
            "llama_stage.spec.proposed".to_string(),
            json!(window.proposed_count),
        );
        attrs.insert(
            "llama_stage.spec.verify_inputs".to_string(),
            json!(window.verify_input_count),
        );
        attrs.insert(
            "llama_stage.spec.accepted".to_string(),
            json!(decision.accepted_before_reject),
        );
        attrs.insert(
            "llama_stage.spec.rejected".to_string(),
            json!(decision.rejected()),
        );
        attrs.insert(
            "llama_stage.stage0_compute_ms".to_string(),
            json!(verify.stats.stage0_compute_ms),
        );
        attrs.insert(
            "llama_stage.forward_write_ms".to_string(),
            json!(verify.stats.forward_write_ms),
        );
        attrs.insert(
            "llama_stage.downstream_wait_ms".to_string(),
            json!(verify.stats.downstream_wait_ms),
        );
        self.telemetry
            .emit_debug("stage.openai_decode_verify_window", attrs);
    }
}

fn debug_assert_pipelined_scheduler_window(
    expected: &Option<ShardVerifyWindow>,
    actual: &InFlightVerify,
) {
    let Some(expected) = expected else {
        debug_assert!(false, "pipelined scheduler returned no window");
        return;
    };
    debug_assert_eq!(expected.window_index, actual.window_index);
    debug_assert_eq!(expected.epoch, actual.window_epoch);
    debug_assert_eq!(expected.decoded_at_send, actual.decoded_at_send);
    debug_assert_eq!(
        Some(expected.decode_step),
        usize::try_from(actual.message.state.decode_step).ok()
    );
    debug_assert_eq!(
        Some(expected.pos_start),
        usize::try_from(actual.message.pos_start).ok()
    );
    debug_assert_eq!(expected.verify_inputs, actual.verify_inputs);
}

fn observe_pipelined_fifo_window(
    stats: &mut OpenAiSpeculativeStats,
    decoded_at_send: usize,
    message: &StageWireMessage,
) {
    stats.pipelined_fifo_return_windows += 1;
    if !pipelined_window_decode_step_matches(decoded_at_send, message.state.decode_step) {
        stats.pipelined_fifo_return_violations += 1;
    }
}

fn validate_pipelined_reply_identity(
    stats: &mut OpenAiSpeculativeStats,
    message: &StageWireMessage,
    identity: Option<StageReplyIdentity>,
) -> OpenAiResult<bool> {
    let Some(identity) = identity else {
        stats.pipelined_identity_violations += 1;
        return Err(OpenAiError::backend(
            "pipelined verify reply did not include a window identity",
        ));
    };
    if !identity.matches_message(message) {
        stats.pipelined_identity_violations += 1;
        return Err(OpenAiError::backend(format!(
            "pipelined verify reply identity mismatch: expected {:?}, got {:?}",
            StageReplyIdentity::from_message(message),
            identity
        )));
    }
    Ok(true)
}

fn pipelined_window_decode_step_matches(decoded_at_send: usize, message_decode_step: i32) -> bool {
    usize::try_from(message_decode_step) == Ok(decoded_at_send)
}

fn observe_pipelined_primary_verify(
    stats: &mut OpenAiSpeculativeStats,
    verify: &EmbeddedStageExecution,
) {
    stats.primary_verify_elapsed_ms += verify.elapsed_ms;
    stats.primary_verify_stage0_compute_ms += verify.stats.stage0_compute_ms;
    stats.primary_verify_runtime_lock_wait_ms += verify.stats.runtime_lock_wait_ms;
    stats.primary_verify_runtime_lock_hold_ms += verify.stats.runtime_lock_hold_ms;
    stats.primary_verify_activation_encode_ms += verify.stats.activation_encode_ms;
    stats.primary_verify_forward_write_ms += verify.stats.forward_write_ms;
    stats.primary_verify_downstream_wait_ms += verify.stats.downstream_wait_ms;
    stats.primary_verify_output_activation_bytes = stats
        .primary_verify_output_activation_bytes
        .saturating_add(verify.stats.output_activation_bytes);
    stats.primary_verify_forward_activation_bytes = stats
        .primary_verify_forward_activation_bytes
        .saturating_add(verify.stats.forward_activation_bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipelined_full_accept_defers_tail_correction_to_next_window() {
        let decision = classify_verify_span(&[10, 11], &[10, 11, 12], 0, 8, |_| Ok(false)).unwrap();

        assert_eq!(decision.kind, VerifySpanDecisionKind::FullAccept);
        assert_eq!(decision.commit_count, 3);
        assert_eq!(shard_linear_commit_count(&decision, 2), 2);
    }

    #[test]
    fn pipelined_reject_commits_correction_immediately() {
        let decision =
            classify_verify_span(&[10, 11, 12], &[10, 99, 12, 13], 0, 8, |_| Ok(false)).unwrap();

        assert_eq!(decision.kind, VerifySpanDecisionKind::EarlyReject);
        assert_eq!(shard_linear_commit_count(&decision, 3), 2);
    }

    #[test]
    fn pipelined_fifo_window_identity_uses_decode_step() {
        assert!(pipelined_window_decode_step_matches(4, 4));
        assert!(!pipelined_window_decode_step_matches(4, 5));
        assert!(!pipelined_window_decode_step_matches(4, -1));
    }

    #[test]
    fn pipelined_reply_identity_matches_original_message() {
        let message = identified_test_verify_message(3, 8);
        let mut stats = OpenAiSpeculativeStats::default();

        assert!(
            validate_pipelined_reply_identity(
                &mut stats,
                &message,
                Some(StageReplyIdentity::from_message(&message)),
            )
            .unwrap()
        );
        assert_eq!(stats.pipelined_identity_violations, 0);
    }

    #[test]
    fn pipelined_reply_identity_is_required() {
        let message = identified_test_verify_message(3, 8);
        let mut stats = OpenAiSpeculativeStats::default();

        validate_pipelined_reply_identity(&mut stats, &message, None).unwrap_err();

        assert_eq!(stats.pipelined_identity_violations, 1);
    }

    #[test]
    fn pipelined_reject_marks_remaining_inflight_stale_in_fifo_order() {
        let mut queues = PipelinedWindowQueues {
            draft_exhausted: true,
            ..PipelinedWindowQueues::default()
        };
        queues.inflight.push_back(test_inflight_window(1, 2));
        queues.inflight.push_back(test_inflight_window(2, 4));
        queues.inflight.push_back(test_inflight_window(3, 6));

        assert_eq!(queues.mark_remaining_inflight_stale(), 3);

        assert!(queues.inflight.is_empty());
        assert!(!queues.draft_exhausted);
        assert_eq!(
            queues
                .stale_debt
                .iter()
                .map(|window| (window.window_index, window.decoded_at_send))
                .collect::<Vec<_>>(),
            vec![(1, 2), (2, 4), (3, 6)]
        );
    }

    fn test_inflight_window(window_index: usize, decoded_at_send: usize) -> InFlightVerify {
        let verify_inputs = vec![10, 11];
        let message = embedded_verify_message(
            WireActivationDType::F16,
            VerifySpanMessageArgs {
                request_id: 1,
                session_id: 2,
                prompt_token_count: 5,
                pos_start: 5 + decoded_at_send,
                decode_step: decoded_at_send,
                tokens: &verify_inputs,
                sampling: None,
                checkpoint: false,
            },
        )
        .expect("verify message");
        InFlightVerify {
            window_index,
            window_epoch: 0,
            decoded_at_send,
            draft_tokens: vec![11],
            verify_inputs,
            message,
            pending: EmbeddedStagePendingExecution {
                timer: PhaseTimer::start(),
                reply_stats: StageReplyStats::default(),
                stats: EmbeddedExecutionStats::default(),
            },
        }
    }

    fn identified_test_verify_message(
        window_index: usize,
        decoded_at_send: usize,
    ) -> StageWireMessage {
        let verify_inputs = vec![10, 11];
        let mut message = embedded_verify_message(
            WireActivationDType::F16,
            VerifySpanMessageArgs {
                request_id: 1,
                session_id: 2,
                prompt_token_count: 5,
                pos_start: 5 + decoded_at_send,
                decode_step: decoded_at_send,
                tokens: &verify_inputs,
                sampling: None,
                checkpoint: false,
            },
        )
        .expect("verify message");
        message.state.flags |= state_flags::IDENTIFIED_REPLY;
        message.state.seq_id = i32::try_from(window_index).unwrap();
        message.state.checkpoint_generation = 4;
        message
    }
}
