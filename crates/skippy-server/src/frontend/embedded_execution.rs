use super::*;

const DIRECT_RETURN_FALLBACK_POLL: Duration = Duration::from_millis(10);
const DIRECT_RETURN_FALLBACK_TIMEOUT: Duration = Duration::from_secs(300);

pub(super) struct EmbeddedTreeGatherPath<'a> {
    pub(super) request_id: u64,
    pub(super) session_id: u64,
    pub(super) source_leaf_index: u32,
    pub(super) dest_start: usize,
    pub(super) source_positions: &'a [u64],
    pub(super) token_ids: &'a [i32],
}

impl StageOpenAiBackend {
    pub(super) fn execute_embedded_stage_message(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        message: &StageWireMessage,
        token_ids: &[i32],
        expected_reply: WireReplyKind,
    ) -> OpenAiResult<EmbeddedStageExecution> {
        let pending =
            self.send_embedded_stage_message(request, downstream, session_key, message, token_ids)?;
        self.receive_embedded_stage_execution(request, downstream, pending, message, expected_reply)
    }

    pub(super) fn send_embedded_stage_message(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        message: &StageWireMessage,
        token_ids: &[i32],
    ) -> OpenAiResult<EmbeddedStagePendingExecution> {
        let timer = PhaseTimer::start();
        let mut stats = StageReplyStats::default();
        let stage0_timer = PhaseTimer::start();
        let output = {
            let lock_timer = PhaseTimer::start();
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            let lock_wait_ms = lock_timer.elapsed_ms();
            let hold_timer = PhaseTimer::start();
            crate::binary_transport::apply_lazy_tree_gather(&mut runtime, session_key, message)
                .map_err(openai_backend_error)?;
            align_embedded_stage0_session_if_ahead(&mut runtime, session_key, message)?;
            if message.kind == WireMessageKind::VerifySpan
                && (message.state.flags & state_flags::SKIP_VERIFY_CHECKPOINT) == 0
            {
                let checkpoint_timer = PhaseTimer::start();
                runtime
                    .checkpoint_session(session_key)
                    .map_err(openai_backend_error)?;
                let checkpoint_us = ms_to_us(checkpoint_timer.elapsed_ms());
                stats.checkpoint_local_us += checkpoint_us;
                stats.checkpoint_total_us += checkpoint_us;
                stats.verify_span_checkpointed_requests += 1;
            } else if message.kind == WireMessageKind::VerifySpan {
                stats.verify_span_skip_checkpoint_requests += 1;
            }
            let output = run_binary_stage_message(
                request.config,
                &mut runtime,
                BinaryStageMessageExecution {
                    session_id: session_key,
                    message,
                    token_ids,
                    input: None,
                    sample_final_prefill: false,
                    output_capacity: stage_output_activation_capacity(
                        request.config,
                        message.token_count,
                        request.activation_width,
                    )
                    .map_err(openai_backend_error)?,
                },
            )
            .map_err(openai_backend_error)?
            .2;
            let hold_ms = hold_timer.elapsed_ms();
            EmbeddedLocalOutput {
                output,
                runtime_lock_wait_ms: lock_wait_ms,
                runtime_lock_hold_ms: hold_ms,
            }
        };
        let stage0_compute_ms = stage0_timer.elapsed_ms();
        let forwarded = forwarded_stage_message_timed(
            request.config,
            message,
            &output.output,
            request.wire_dtype,
            request.activation_width,
        )
        .map_err(openai_backend_error)?;
        let write_timer = PhaseTimer::start();
        write_stage_message_conditioned(
            &mut *downstream,
            &forwarded.message,
            request.wire_dtype,
            request.downstream_wire_condition,
        )
        .map_err(openai_io_error)?;
        let forward_write_ms = write_timer.elapsed_ms();
        Ok(EmbeddedStagePendingExecution {
            timer,
            reply_stats: stats,
            stats: EmbeddedExecutionStats {
                stage0_compute_ms,
                runtime_lock_wait_ms: output.runtime_lock_wait_ms,
                runtime_lock_hold_ms: output.runtime_lock_hold_ms,
                activation_encode_ms: forwarded.activation_encode_ms,
                output_activation_bytes: output.output.payload.len(),
                forward_activation_bytes: forwarded.message.activation.len(),
                forward_write_ms,
                downstream_wait_ms: 0.0,
            },
        })
    }

    pub(super) fn receive_embedded_stage_execution(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        pending: EmbeddedStagePendingExecution,
        message: &StageWireMessage,
        expected_reply: WireReplyKind,
    ) -> OpenAiResult<EmbeddedStageExecution> {
        let wait_timer = PhaseTimer::start();
        let prediction_return = (!message_forces_downstream_reply(message))
            .then_some(request.prediction_return.as_ref())
            .flatten();
        let envelope =
            receive_embedded_stage_reply_envelope(downstream, prediction_return, expected_reply)?;
        let reply = envelope.reply;
        let downstream_wait_ms = wait_timer.elapsed_ms();
        let mut stats = pending.reply_stats;
        stats.merge(reply.stats);
        let mut execution_stats = pending.stats;
        execution_stats.downstream_wait_ms = downstream_wait_ms;
        if message.kind == WireMessageKind::VerifySpan {
            stats.verify_span_compute_us += ms_to_us(execution_stats.stage0_compute_ms);
            stats.verify_span_forward_write_us += ms_to_us(execution_stats.forward_write_ms);
            stats.verify_span_downstream_wait_us += ms_to_us(downstream_wait_ms);
            stats.verify_span_total_us += ms_to_us(pending.timer.elapsed_ms());
            stats.verify_span_stage_count += 1;
            stats.verify_span_request_count += 1;
            stats.verify_span_token_count += i64::from(message.token_count.max(0));
            stats.verify_span_max_tokens = stats
                .verify_span_max_tokens
                .max(i64::from(message.token_count.max(0)));
        }
        Ok(EmbeddedStageExecution {
            reply: StageReply { stats, ..reply },
            reply_identity: envelope.identity,
            stats: execution_stats,
            elapsed_ms: pending.timer.elapsed_ms(),
        })
    }

    pub(super) fn trim_embedded_stage_session(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        request_id: u64,
        session_id: u64,
        token_count: usize,
    ) -> OpenAiResult<EmbeddedSessionControl> {
        let timer = PhaseTimer::start();
        let local_timer = PhaseTimer::start();
        {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            runtime
                .trim_session(session_key, token_count as u64)
                .map_err(openai_backend_error)?;
        }
        let local_ms = local_timer.elapsed_ms();
        let message =
            embedded_trim_session_message(request.wire_dtype, request_id, session_id, token_count)?;
        let write_timer = PhaseTimer::start();
        write_stage_message_conditioned(
            &mut *downstream,
            &message,
            request.wire_dtype,
            request.downstream_wire_condition,
        )
        .map_err(openai_io_error)?;
        let downstream_write_ms = write_timer.elapsed_ms();
        let wait_timer = PhaseTimer::start();
        let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
        let downstream_wait_ms = wait_timer.elapsed_ms();
        if reply.kind != WireReplyKind::Ack {
            return Err(OpenAiError::backend(format!(
                "trim expected ACK from downstream, got {:?}",
                reply.kind
            )));
        }
        Ok(EmbeddedSessionControl {
            elapsed_ms: timer.elapsed_ms(),
            local_ms,
            downstream_write_ms,
            downstream_wait_ms,
        })
    }

    pub(super) fn gather_embedded_stage_tree_path(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        gather: EmbeddedTreeGatherPath<'_>,
    ) -> OpenAiResult<EmbeddedSessionControl> {
        let timer = PhaseTimer::start();
        let local_timer = PhaseTimer::start();
        {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            runtime
                .gather_tree_path(
                    session_key,
                    gather.source_leaf_index,
                    gather.dest_start as u64,
                    gather.source_positions,
                    gather.token_ids,
                )
                .map_err(openai_backend_error)?;
        }
        let local_ms = local_timer.elapsed_ms();
        let message = embedded_gather_tree_path_message(
            request.wire_dtype,
            gather.request_id,
            gather.session_id,
            gather.source_leaf_index,
            gather.dest_start,
            gather.source_positions,
            gather.token_ids,
        )?;
        let write_timer = PhaseTimer::start();
        write_stage_message_conditioned(
            &mut *downstream,
            &message,
            request.wire_dtype,
            request.downstream_wire_condition,
        )
        .map_err(openai_io_error)?;
        let downstream_write_ms = write_timer.elapsed_ms();
        let wait_timer = PhaseTimer::start();
        let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
        let downstream_wait_ms = wait_timer.elapsed_ms();
        if reply.kind != WireReplyKind::Ack {
            return Err(OpenAiError::backend(format!(
                "tree gather expected ACK from downstream, got {:?}",
                reply.kind
            )));
        }
        Ok(EmbeddedSessionControl {
            elapsed_ms: timer.elapsed_ms(),
            local_ms,
            downstream_write_ms,
            downstream_wait_ms,
        })
    }

    pub(super) fn trim_embedded_stage_session_local(
        &self,
        session_key: &str,
        token_count: usize,
    ) -> OpenAiResult<EmbeddedSessionControl> {
        let timer = PhaseTimer::start();
        let local_timer = PhaseTimer::start();
        {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            runtime
                .trim_session(session_key, token_count as u64)
                .map_err(openai_backend_error)?;
        }
        Ok(EmbeddedSessionControl {
            elapsed_ms: timer.elapsed_ms(),
            local_ms: local_timer.elapsed_ms(),
            downstream_write_ms: 0.0,
            downstream_wait_ms: 0.0,
        })
    }
}

fn align_embedded_stage0_session_if_ahead(
    runtime: &mut RuntimeState,
    session_key: &str,
    message: &StageWireMessage,
) -> OpenAiResult<()> {
    if !embedded_message_allows_session_auto_align(message) {
        return Ok(());
    }
    let Some(target_token_count) = embedded_message_pos_start_as_token_count(message) else {
        return Ok(());
    };
    runtime
        .align_session_to_token_count_if_ahead(session_key, target_token_count)
        .map(|_| ())
        .map_err(openai_backend_error)
}

fn embedded_message_allows_session_auto_align(message: &StageWireMessage) -> bool {
    matches!(
        message.kind,
        WireMessageKind::DecodeEmbd
            | WireMessageKind::DecodeReadout
            | WireMessageKind::DecodeLightCtx
            | WireMessageKind::VerifySpan
    )
}

fn embedded_message_pos_start_as_token_count(message: &StageWireMessage) -> Option<u64> {
    u64::try_from(message.pos_start).ok()
}

pub(crate) fn message_forces_downstream_reply(message: &StageWireMessage) -> bool {
    (message.state.flags & state_flags::FORCE_DOWNSTREAM_REPLY) != 0
}

pub(crate) fn receive_embedded_stage_reply(
    downstream: &mut TcpStream,
    prediction_return: Option<&PredictionReturnReceiver>,
    expected_reply: WireReplyKind,
) -> OpenAiResult<StageReply> {
    Ok(receive_embedded_stage_reply_envelope(downstream, prediction_return, expected_reply)?.reply)
}

pub(crate) fn receive_embedded_stage_reply_envelope(
    downstream: &mut TcpStream,
    prediction_return: Option<&PredictionReturnReceiver>,
    expected_reply: WireReplyKind,
) -> OpenAiResult<StageReplyEnvelope> {
    let Some(prediction_return) = prediction_return else {
        return receive_downstream_stage_reply(downstream, expected_reply);
    };
    poll_direct_or_downstream_reply(downstream, prediction_return, expected_reply)
}

fn poll_direct_or_downstream_reply(
    downstream: &mut TcpStream,
    prediction_return: &PredictionReturnReceiver,
    expected_reply: WireReplyKind,
) -> OpenAiResult<StageReplyEnvelope> {
    let previous_timeout = downstream.read_timeout().map_err(openai_io_error)?;
    downstream
        .set_read_timeout(Some(DIRECT_RETURN_FALLBACK_POLL))
        .map_err(openai_io_error)?;
    let started = Instant::now();
    loop {
        if let Some(reply) = prediction_return
            .try_recv_expected(expected_reply)
            .map_err(openai_backend_error)?
        {
            restore_downstream_read_timeout(downstream, previous_timeout)?;
            return Ok(reply);
        }
        if downstream_reply_available(downstream)? {
            restore_downstream_read_timeout(downstream, previous_timeout)?;
            return receive_downstream_stage_reply(downstream, expected_reply);
        }
        if started.elapsed() >= DIRECT_RETURN_FALLBACK_TIMEOUT {
            restore_downstream_read_timeout(downstream, previous_timeout)?;
            return Err(OpenAiError::backend(format!(
                "timed out waiting for {expected_reply:?} reply from direct return or downstream"
            )));
        }
    }
}

fn downstream_reply_available(downstream: &TcpStream) -> OpenAiResult<bool> {
    let mut byte = [0u8; 1];
    match downstream.peek(&mut byte) {
        Ok(0) => Err(OpenAiError::backend("downstream closed before stage reply")),
        Ok(_) => Ok(true),
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            ) =>
        {
            Ok(false)
        }
        Err(error) => Err(openai_io_error(error)),
    }
}

fn restore_downstream_read_timeout(
    downstream: &TcpStream,
    timeout: Option<Duration>,
) -> OpenAiResult<()> {
    downstream
        .set_read_timeout(timeout)
        .map_err(openai_io_error)
}

fn receive_downstream_stage_reply(
    downstream: &mut TcpStream,
    expected_reply: WireReplyKind,
) -> OpenAiResult<StageReplyEnvelope> {
    let envelope = recv_reply_envelope(&mut *downstream).map_err(openai_io_error)?;
    if envelope.reply.kind != expected_reply {
        return Err(OpenAiError::backend(format!(
            "expected {expected_reply:?} reply from downstream, got {:?}",
            envelope.reply.kind
        )));
    }
    Ok(envelope)
}
