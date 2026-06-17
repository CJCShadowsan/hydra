use super::*;

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
        let timer = PhaseTimer::start();
        let mut message = message.clone();
        self.mark_spd_tap_return(request, &mut message);
        let message = &message;
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
            if message.kind == WireMessageKind::VerifySpan
                && (message.state.flags & state_flags::SKIP_VERIFY_CHECKPOINT) == 0
            {
                let checkpoint_timer = PhaseTimer::start();
                runtime
                    .checkpoint_session_generation(session_key, message.state.checkpoint_generation)
                    .map_err(openai_backend_error)?;
                let checkpoint_us = ms_to_us(checkpoint_timer.elapsed_ms());
                stats.checkpoint_local_us += checkpoint_us;
                stats.checkpoint_total_us += checkpoint_us;
                stats.verify_span_checkpointed_requests += 1;
            } else if message.kind == WireMessageKind::VerifySpan {
                stats.verify_span_skip_checkpoint_requests += 1;
            }
            let output = run_binary_stage_message(
                &mut runtime,
                session_key,
                message,
                token_ids,
                None,
                false,
                stage_output_activation_capacity(
                    request.config,
                    message.token_count,
                    request.activation_width,
                )
                .map_err(openai_backend_error)?,
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
        self.record_spd_stage0_boundary_tap(request, message, &output.output);
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
        let wait_timer = PhaseTimer::start();
        let reply = self
            .recv_spd_aware_prediction_return(request, expected_reply)
            .map_err(openai_backend_error)?;
        let downstream_wait_ms = wait_timer.elapsed_ms();
        stats.merge(reply.stats);
        if message.kind == WireMessageKind::VerifySpan {
            stats.verify_span_compute_us += ms_to_us(stage0_compute_ms);
            stats.verify_span_forward_write_us += ms_to_us(forward_write_ms);
            stats.verify_span_downstream_wait_us += ms_to_us(downstream_wait_ms);
            stats.verify_span_total_us += ms_to_us(timer.elapsed_ms());
            stats.verify_span_stage_count += 1;
            stats.verify_span_request_count += 1;
            stats.verify_span_token_count += i64::from(message.token_count.max(0));
            stats.verify_span_max_tokens = stats
                .verify_span_max_tokens
                .max(i64::from(message.token_count.max(0)));
        }
        Ok(EmbeddedStageExecution {
            reply: StageReply { stats, ..reply },
            stats: EmbeddedExecutionStats {
                stage0_compute_ms,
                runtime_lock_wait_ms: output.runtime_lock_wait_ms,
                runtime_lock_hold_ms: output.runtime_lock_hold_ms,
                activation_encode_ms: forwarded.activation_encode_ms,
                output_activation_bytes: output.output.payload.len(),
                forward_activation_bytes: forwarded.message.activation.len(),
                forward_write_ms,
                downstream_wait_ms,
            },
            elapsed_ms: timer.elapsed_ms(),
        })
    }

    pub(super) fn start_embedded_stage_message(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        message: &StageWireMessage,
        token_ids: &[i32],
        spd_tap_return: bool,
    ) -> OpenAiResult<EmbeddedStageStart> {
        let timer = PhaseTimer::start();
        let mut message = message.clone();
        if spd_tap_return {
            self.mark_spd_tap_return(request, &mut message);
        }
        let message = &message;
        let mut reply_stats = StageReplyStats::default();
        let stage0_timer = PhaseTimer::start();
        let output = {
            let lock_timer = PhaseTimer::start();
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            let lock_wait_ms = lock_timer.elapsed_ms();
            let hold_timer = PhaseTimer::start();
            if message.kind == WireMessageKind::VerifySpan
                && (message.state.flags & state_flags::SKIP_VERIFY_CHECKPOINT) == 0
            {
                let checkpoint_timer = PhaseTimer::start();
                runtime
                    .checkpoint_session_generation(session_key, message.state.checkpoint_generation)
                    .map_err(openai_backend_error)?;
                let checkpoint_us = ms_to_us(checkpoint_timer.elapsed_ms());
                reply_stats.checkpoint_local_us += checkpoint_us;
                reply_stats.checkpoint_total_us += checkpoint_us;
                reply_stats.verify_span_checkpointed_requests += 1;
            } else if message.kind == WireMessageKind::VerifySpan {
                reply_stats.verify_span_skip_checkpoint_requests += 1;
            }
            let output = run_binary_stage_message(
                &mut runtime,
                session_key,
                message,
                token_ids,
                None,
                false,
                stage_output_activation_capacity(
                    request.config,
                    message.token_count,
                    request.activation_width,
                )
                .map_err(openai_backend_error)?,
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
        if spd_tap_return {
            self.record_spd_stage0_boundary_tap(request, message, &output.output);
        }
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
        Ok(EmbeddedStageStart {
            reply_stats,
            stats: EmbeddedExecutionStats {
                stage0_compute_ms,
                runtime_lock_wait_ms: output.runtime_lock_wait_ms,
                runtime_lock_hold_ms: output.runtime_lock_hold_ms,
                activation_encode_ms: forwarded.activation_encode_ms,
                output_activation_bytes: output.output.payload.len(),
                forward_activation_bytes: forwarded.message.activation.len(),
                forward_write_ms: write_timer.elapsed_ms(),
                downstream_wait_ms: 0.0,
            },
            elapsed_ms: timer.elapsed_ms(),
        })
    }

    pub(super) fn start_spd_optimistic_decode_for_probe(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        args: SpdOptimisticDecodeStart<'_>,
    ) -> OpenAiResult<Option<SpdOptimisticDecode>> {
        let Some(proposed) = args.probe.proposed else {
            return Ok(None);
        };
        if args.chain_depth >= args.chain_depth_limit
            || !args
                .probe
                .allows_optimistic_decode(request.spd_optimistic_min_logit_margin)
            || !spd_optimistic_position_has_output_budget(request, args.pos_start)
        {
            return Ok(None);
        }
        let request_spd_taps = true;
        if let Some(spd) = request.spd.as_ref() {
            spd.mark_pending_optimistic_tap_position(args.pos_start)
                .map_err(openai_backend_error)?;
        }
        let timer = PhaseTimer::start();
        let tokens = [proposed];
        let message = embedded_verify_message(
            request.wire_dtype,
            VerifySpanMessageArgs {
                request_id: request.ids.request_id,
                session_id: request.ids.session_id,
                prompt_token_count: request.prompt_token_ids.len(),
                pos_start: args.pos_start,
                decode_step: args.decode_step,
                checkpoint_generation: checkpoint_generation_from_position(args.pos_start)?,
                tokens: &tokens,
                checkpoint: true,
            },
        )?;
        let origin = PredictionReturnOrigin::from_message(&message);
        let execution = self.start_embedded_stage_message(
            request,
            args.downstream,
            args.session_key,
            &message,
            &tokens,
            request_spd_taps,
        )?;
        Ok(Some(SpdOptimisticDecode {
            proposed,
            proposed_logit: args.probe.proposed_logit,
            proposed_logit_margin: args.probe.proposed_logit_margin,
            requested_spd_taps: request_spd_taps,
            chain_depth: args.chain_depth,
            origin,
            timer,
            execution,
        }))
    }

    pub(super) fn restore_embedded_stage_session(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        request_id: u64,
        session_id: u64,
        checkpoint_generation: i32,
    ) -> OpenAiResult<EmbeddedSessionControl> {
        let timer = PhaseTimer::start();
        let local_timer = PhaseTimer::start();
        {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            runtime
                .restore_session_generation(session_key, checkpoint_generation)
                .map_err(openai_backend_error)?;
        }
        let local_ms = local_timer.elapsed_ms();
        let message = embedded_session_control_message(
            request.wire_dtype,
            WireMessageKind::RestoreSession,
            request_id,
            session_id,
            checkpoint_generation,
        );
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
                "restore expected ACK from downstream, got {:?}",
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
}

fn spd_optimistic_position_has_output_budget(
    request: &EmbeddedStageZeroGeneration<'_>,
    pos_start: usize,
) -> bool {
    let output_limit = request
        .prompt_token_ids
        .len()
        .saturating_add(request.max_tokens as usize);
    pos_start
        .checked_add(1)
        .is_some_and(|returned_position| returned_position < output_limit)
}
