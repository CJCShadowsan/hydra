use super::*;

fn combine_activation_frames(frames: &[ActivationFrame]) -> OpenAiResult<ActivationFrame> {
    let Some(first) = frames.first() else {
        return Err(OpenAiError::backend(
            "cannot combine empty activation frames",
        ));
    };
    let mut desc = first.desc;
    let mut payload = Vec::new();
    let mut token_count = 0u32;
    for frame in frames {
        if frame.desc.dtype != desc.dtype
            || frame.desc.layout != desc.layout
            || frame.desc.producer_stage_index != desc.producer_stage_index
            || frame.desc.layer_start != desc.layer_start
            || frame.desc.layer_end != desc.layer_end
            || frame.desc.sequence_count != desc.sequence_count
            || frame.desc.flags != desc.flags
        {
            return Err(OpenAiError::backend(
                "cannot combine incompatible activation frames",
            ));
        }
        token_count = token_count
            .checked_add(frame.desc.token_count)
            .ok_or_else(|| OpenAiError::backend("combined activation token count overflow"))?;
        payload.extend_from_slice(&frame.payload);
    }
    desc.token_count = token_count;
    desc.payload_bytes = payload.len() as u64;
    Ok(ActivationFrame { desc, payload })
}

#[derive(Debug)]
struct ActivationFrameComparison {
    desc_equal: bool,
    payload_equal: bool,
    batched_payload_bytes: usize,
    serial_payload_bytes: usize,
    byte_diff_count: usize,
    first_diff_byte: Option<usize>,
    first_diff_row: Option<usize>,
    f32_compared: usize,
    f32_max_abs_diff: f32,
    f32_mean_abs_diff: f64,
    row_byte_diff_counts: Vec<usize>,
    row_f32_compared: Vec<usize>,
    row_f32_max_abs_diff: Vec<f32>,
    row_f32_mean_abs_diff: Vec<f64>,
}

impl ActivationFrameComparison {
    fn compare(batched: &ActivationFrame, serial: &ActivationFrame, activation_width: i32) -> Self {
        let compared_bytes = batched.payload.len().min(serial.payload.len());
        let mut byte_diff_count = batched.payload.len().abs_diff(serial.payload.len());
        let mut first_diff_byte = None;
        for index in 0..compared_bytes {
            if batched.payload[index] != serial.payload[index] {
                byte_diff_count += 1;
                first_diff_byte.get_or_insert(index);
            }
        }

        let row_bytes = usize::try_from(activation_width)
            .ok()
            .and_then(|width| width.checked_mul(std::mem::size_of::<f32>()));
        let first_diff_row = first_diff_byte
            .zip(row_bytes)
            .and_then(|(byte, bytes)| (bytes > 0).then_some(byte / bytes));
        let (f32_compared, f32_max_abs_diff, f32_mean_abs_diff) =
            compare_activation_f32_payloads(&batched.payload, &serial.payload);
        let row_metrics = row_activation_metrics(
            &batched.payload,
            &serial.payload,
            row_bytes,
            batched.desc.token_count.min(serial.desc.token_count) as usize,
        );

        Self {
            desc_equal: batched.desc == serial.desc,
            payload_equal: batched.payload == serial.payload,
            batched_payload_bytes: batched.payload.len(),
            serial_payload_bytes: serial.payload.len(),
            byte_diff_count,
            first_diff_byte,
            first_diff_row,
            f32_compared,
            f32_max_abs_diff,
            f32_mean_abs_diff,
            row_byte_diff_counts: row_metrics
                .iter()
                .map(|metrics| metrics.byte_diff_count)
                .collect(),
            row_f32_compared: row_metrics
                .iter()
                .map(|metrics| metrics.f32_compared)
                .collect(),
            row_f32_max_abs_diff: row_metrics
                .iter()
                .map(|metrics| metrics.f32_max_abs_diff)
                .collect(),
            row_f32_mean_abs_diff: row_metrics
                .iter()
                .map(|metrics| metrics.f32_mean_abs_diff)
                .collect(),
        }
    }

    fn insert_attrs(&self, attrs: &mut BTreeMap<String, Value>) {
        attrs.insert(
            "llama_stage.native_mtp.stage0_compare.desc_equal".to_string(),
            json!(self.desc_equal),
        );
        attrs.insert(
            "llama_stage.native_mtp.stage0_compare.payload_equal".to_string(),
            json!(self.payload_equal),
        );
        attrs.insert(
            "llama_stage.native_mtp.stage0_compare.batched_payload_bytes".to_string(),
            json!(self.batched_payload_bytes),
        );
        attrs.insert(
            "llama_stage.native_mtp.stage0_compare.serial_payload_bytes".to_string(),
            json!(self.serial_payload_bytes),
        );
        attrs.insert(
            "llama_stage.native_mtp.stage0_compare.byte_diff_count".to_string(),
            json!(self.byte_diff_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.stage0_compare.f32_compared".to_string(),
            json!(self.f32_compared),
        );
        attrs.insert(
            "llama_stage.native_mtp.stage0_compare.f32_max_abs_diff".to_string(),
            json!(self.f32_max_abs_diff),
        );
        attrs.insert(
            "llama_stage.native_mtp.stage0_compare.f32_mean_abs_diff".to_string(),
            json!(self.f32_mean_abs_diff),
        );
        attrs.insert(
            "llama_stage.native_mtp.stage0_compare.row_count".to_string(),
            json!(self.row_byte_diff_counts.len()),
        );
        for row in 0..self.row_byte_diff_counts.len() {
            attrs.insert(
                format!("llama_stage.native_mtp.stage0_compare.row{row}.byte_diff_count"),
                json!(self.row_byte_diff_counts[row]),
            );
            attrs.insert(
                format!("llama_stage.native_mtp.stage0_compare.row{row}.f32_compared"),
                json!(self.row_f32_compared[row]),
            );
            attrs.insert(
                format!("llama_stage.native_mtp.stage0_compare.row{row}.f32_max_abs_diff"),
                json!(self.row_f32_max_abs_diff[row]),
            );
            attrs.insert(
                format!("llama_stage.native_mtp.stage0_compare.row{row}.f32_mean_abs_diff"),
                json!(self.row_f32_mean_abs_diff[row]),
            );
        }
        if let Some(first_diff_byte) = self.first_diff_byte {
            attrs.insert(
                "llama_stage.native_mtp.stage0_compare.first_diff_byte".to_string(),
                json!(first_diff_byte),
            );
        }
        if let Some(first_diff_row) = self.first_diff_row {
            attrs.insert(
                "llama_stage.native_mtp.stage0_compare.first_diff_row".to_string(),
                json!(first_diff_row),
            );
        }
    }
}

struct ActivationRowMetrics {
    byte_diff_count: usize,
    f32_compared: usize,
    f32_max_abs_diff: f32,
    f32_mean_abs_diff: f64,
}

fn row_activation_metrics(
    batched: &[u8],
    serial: &[u8],
    row_bytes: Option<usize>,
    row_count: usize,
) -> Vec<ActivationRowMetrics> {
    let Some(row_bytes) = row_bytes.filter(|bytes| *bytes > 0) else {
        return Vec::new();
    };
    let max_complete_rows = batched.len().min(serial.len()) / row_bytes;
    let compared_rows = row_count.min(max_complete_rows);
    (0..compared_rows)
        .map(|row| {
            let row_start = row * row_bytes;
            let row_end = row_start + row_bytes;
            let byte_diff_count = batched[row_start..row_end]
                .iter()
                .zip(&serial[row_start..row_end])
                .filter(|(left, right)| left != right)
                .count();
            let (f32_compared, f32_max_abs_diff, f32_mean_abs_diff) =
                compare_activation_f32_payloads(
                    &batched[row_start..row_end],
                    &serial[row_start..row_end],
                );
            ActivationRowMetrics {
                byte_diff_count,
                f32_compared,
                f32_max_abs_diff,
                f32_mean_abs_diff,
            }
        })
        .collect()
}

fn compare_activation_f32_payloads(batched: &[u8], serial: &[u8]) -> (usize, f32, f64) {
    let compared_f32 = batched.len().min(serial.len()) / std::mem::size_of::<f32>();
    if compared_f32 == 0 {
        return (0, 0.0, 0.0);
    }

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    for index in 0..compared_f32 {
        let byte_index = index * std::mem::size_of::<f32>();
        let batched_value = f32::from_ne_bytes(
            batched[byte_index..byte_index + std::mem::size_of::<f32>()]
                .try_into()
                .expect("slice length checked"),
        );
        let serial_value = f32::from_ne_bytes(
            serial[byte_index..byte_index + std::mem::size_of::<f32>()]
                .try_into()
                .expect("slice length checked"),
        );
        let abs = (batched_value - serial_value).abs();
        max_abs = max_abs.max(abs);
        sum_abs += f64::from(abs);
    }
    (compared_f32, max_abs, sum_abs / compared_f32 as f64)
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
        let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
        let downstream_wait_ms = wait_timer.elapsed_ms();
        if reply.kind != expected_reply {
            return Err(OpenAiError::backend(format!(
                "expected embedded stage {expected_reply:?} reply from downstream, got {:?}",
                reply.kind
            )));
        }
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

    pub(super) fn execute_embedded_verify_span_with_serial_stage0(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        message: &StageWireMessage,
        token_ids: &[i32],
        expected_reply: WireReplyKind,
    ) -> OpenAiResult<EmbeddedStageExecution> {
        if message.kind != WireMessageKind::VerifySpan {
            return Err(OpenAiError::backend(
                "serial stage0 verify execution requires VerifySpan",
            ));
        }
        if token_ids.is_empty() || token_ids.len() != message.token_count.max(0) as usize {
            return Err(OpenAiError::backend(
                "serial stage0 verify execution token count mismatch",
            ));
        }

        let timer = PhaseTimer::start();
        let mut stats = StageReplyStats::default();
        stats.verify_span_skip_checkpoint_requests += 1;
        let stage0_timer = PhaseTimer::start();
        let output = if native_mtp_compare_stage0_verify_enabled() {
            self.run_serial_stage0_verify_tokens_with_compare(
                request,
                session_key,
                message,
                token_ids,
            )?
        } else {
            self.run_serial_stage0_verify_tokens(request, session_key, message, token_ids)?
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
        let wait_timer = PhaseTimer::start();
        let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
        let downstream_wait_ms = wait_timer.elapsed_ms();
        if reply.kind != expected_reply {
            return Err(OpenAiError::backend(format!(
                "expected embedded stage {expected_reply:?} reply from downstream, got {:?}",
                reply.kind
            )));
        }
        stats.merge(reply.stats);
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

    fn run_serial_stage0_verify_tokens(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        session_key: &str,
        message: &StageWireMessage,
        token_ids: &[i32],
    ) -> OpenAiResult<EmbeddedLocalOutput> {
        let lock_timer = PhaseTimer::start();
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        let runtime_lock_wait_ms = lock_timer.elapsed_ms();
        let hold_timer = PhaseTimer::start();
        let output = Self::run_serial_stage0_verify_tokens_locked(
            request,
            &mut runtime,
            session_key,
            message,
            token_ids,
        )?;
        let runtime_lock_hold_ms = hold_timer.elapsed_ms();
        Ok(EmbeddedLocalOutput {
            output,
            runtime_lock_wait_ms,
            runtime_lock_hold_ms,
        })
    }

    fn run_serial_stage0_verify_tokens_with_compare(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        session_key: &str,
        message: &StageWireMessage,
        token_ids: &[i32],
    ) -> OpenAiResult<EmbeddedLocalOutput> {
        let lock_timer = PhaseTimer::start();
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        let runtime_lock_wait_ms = lock_timer.elapsed_ms();
        let hold_timer = PhaseTimer::start();
        runtime
            .checkpoint_session(session_key)
            .map_err(openai_backend_error)?;
        let output_capacity = stage_output_activation_capacity(
            request.config,
            message.token_count,
            request.activation_width,
        )
        .map_err(openai_backend_error)?;
        let batched = run_binary_stage_message(
            &mut runtime,
            session_key,
            message,
            token_ids,
            None,
            false,
            output_capacity,
        )
        .map_err(openai_backend_error)?
        .2;
        runtime
            .restore_session(session_key)
            .map_err(openai_backend_error)?;
        let serial = Self::run_serial_stage0_verify_tokens_locked(
            request,
            &mut runtime,
            session_key,
            message,
            token_ids,
        )?;
        let comparison =
            ActivationFrameComparison::compare(&batched, &serial, request.activation_width);
        let runtime_lock_hold_ms = hold_timer.elapsed_ms();
        drop(runtime);

        self.emit_stage0_verify_comparison(request, message, token_ids, &comparison);
        Ok(EmbeddedLocalOutput {
            output: serial,
            runtime_lock_wait_ms,
            runtime_lock_hold_ms,
        })
    }

    fn run_serial_stage0_verify_tokens_locked(
        request: &EmbeddedStageZeroGeneration<'_>,
        runtime: &mut RuntimeState,
        session_key: &str,
        message: &StageWireMessage,
        token_ids: &[i32],
    ) -> OpenAiResult<ActivationFrame> {
        let mut frames = Vec::with_capacity(token_ids.len());
        for (index, token) in token_ids.iter().copied().enumerate() {
            let pos_start = usize::try_from(message.pos_start)
                .map_err(|_| OpenAiError::backend("negative verify span position"))?
                .checked_add(index)
                .ok_or_else(|| OpenAiError::backend("verify span position overflow"))?;
            let decode_step = usize::try_from(message.state.decode_step)
                .map_err(|_| OpenAiError::backend("negative verify span decode step"))?
                .checked_add(index)
                .ok_or_else(|| OpenAiError::backend("verify span decode step overflow"))?;
            let decode_message = embedded_decode_message(
                request.wire_dtype,
                DecodeMessageArgs {
                    request_id: message.request_id,
                    session_id: message.session_id,
                    prompt_token_count: usize::try_from(message.state.prompt_token_count)
                        .map_err(|_| OpenAiError::backend("negative prompt token count"))?,
                    pos_start,
                    decode_step,
                    current: token,
                    sampling: message.sampling.clone(),
                },
            )?;
            let output = run_binary_stage_message(
                runtime,
                session_key,
                &decode_message,
                &[token],
                None,
                false,
                stage_output_activation_capacity(request.config, 1, request.activation_width)
                    .map_err(openai_backend_error)?,
            )
            .map_err(openai_backend_error)?
            .2;
            frames.push(output);
        }
        combine_activation_frames(&frames)
    }

    fn emit_stage0_verify_comparison(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        message: &StageWireMessage,
        token_ids: &[i32],
        comparison: &ActivationFrameComparison,
    ) {
        if !self.telemetry.is_debug_enabled() {
            return;
        }
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert("llama_stage.message_kind".to_string(), json!("VerifySpan"));
        attrs.insert(
            "llama_stage.decode_step".to_string(),
            json!(message.state.decode_step),
        );
        attrs.insert(
            "llama_stage.token_count".to_string(),
            json!(token_ids.len()),
        );
        attrs.insert(
            "llama_stage.native_mtp.stage0_compare.enabled".to_string(),
            json!(true),
        );
        comparison.insert_attrs(&mut attrs);
        self.telemetry
            .emit_debug("stage.openai_native_mtp_stage0_compare", attrs);
    }

    pub(super) fn restore_embedded_stage_session(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        request_id: u64,
        session_id: u64,
    ) -> OpenAiResult<EmbeddedSessionControl> {
        let timer = PhaseTimer::start();
        let local_timer = PhaseTimer::start();
        {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            runtime
                .restore_session(session_key)
                .map_err(openai_backend_error)?;
        }
        let local_ms = local_timer.elapsed_ms();
        let message = embedded_session_control_message(
            request.wire_dtype,
            WireMessageKind::RestoreSession,
            request_id,
            session_id,
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
