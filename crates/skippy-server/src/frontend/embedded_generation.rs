use super::*;

struct PendingSpdInlineProbe {
    decode_step: u32,
    current: i32,
    target: i32,
    probe: super::spd::SpdInlineProbe,
    rolling: Option<super::spd::SpdRollingTelemetry>,
}

fn merge_spd_rolling_executor_stats(
    speculative_stats: &mut OpenAiSpeculativeStats,
    executor: Option<&SpdRollingExecutor>,
) {
    let Some(executor) = executor else {
        return;
    };
    let stats = executor.stats();
    speculative_stats.spd_rolling_executor_launches = stats.launches;
    speculative_stats.spd_rolling_executor_launch_misses = stats.launch_misses;
    speculative_stats.spd_rolling_executor_launch_miss_in_flight_full =
        stats.launch_miss_in_flight_full;
    speculative_stats.spd_rolling_executor_launch_miss_no_rows = stats.launch_miss_no_rows;
    speculative_stats.spd_rolling_executor_launch_miss_no_proposal = stats.launch_miss_no_proposal;
    speculative_stats.spd_rolling_executor_launch_miss_shadow_not_seedable =
        stats.launch_miss_shadow_not_seedable;
    speculative_stats.spd_rolling_executor_launch_miss_shadow_missing_view =
        stats.launch_miss_shadow_missing_view;
    speculative_stats.spd_rolling_executor_shadow_source_reseeds = stats.shadow_source_reseeds;
    speculative_stats.spd_rolling_executor_margin_rejects = stats.launch_margin_rejects;
    speculative_stats.spd_rolling_executor_max_in_flight = stats.max_in_flight;
    speculative_stats.spd_rolling_executor_accepted_oldest = stats.accepted_oldest;
    speculative_stats.spd_rolling_executor_rejected_oldest = stats.rejected_oldest;
    speculative_stats.spd_rolling_executor_drained_younger = stats.drained_younger;
}

fn observe_spd_rolling_executor_target(
    speculative_stats: &mut OpenAiSpeculativeStats,
    executor: Option<&mut SpdRollingExecutor>,
    position: usize,
    token: i32,
) -> OpenAiResult<Vec<SpdRollingExecutorCommit>> {
    let Some(executor) = executor else {
        return Ok(Vec::new());
    };
    executor.record_target_token(position, token);
    let mut commits = Vec::new();
    while let Some(ready) = executor
        .commit_ready_oldest()
        .map_err(openai_backend_error)?
    {
        let rejected = matches!(ready, SpdRollingExecutorCommit::Rejected { .. });
        commits.push(ready);
        if rejected {
            break;
        }
    }
    merge_spd_rolling_executor_stats(speculative_stats, Some(executor));
    Ok(commits)
}

fn spd_rolling_executor_acceptance(
    commits: &[SpdRollingExecutorCommit],
    target_position: usize,
    target_token: i32,
    legacy_acceptance: bool,
) -> OpenAiResult<bool> {
    let Some(commit) = commits
        .iter()
        .copied()
        .find(|commit| spd_rolling_executor_commit_position(*commit) == target_position)
    else {
        return Ok(legacy_acceptance);
    };
    match commit {
        SpdRollingExecutorCommit::Accepted {
            position, token, ..
        } => {
            if token != target_token {
                return Err(OpenAiError::backend(format!(
                    "SPD rolling executor accepted token {token} at position {position}, expected target token {target_token}"
                )));
            }
            Ok(true)
        }
        SpdRollingExecutorCommit::Rejected {
            position,
            corrected,
            ..
        } => {
            if corrected != target_token {
                return Err(OpenAiError::backend(format!(
                    "SPD rolling executor rejected with corrected token {corrected} at position {position}, expected target token {target_token}"
                )));
            }
            Ok(false)
        }
    }
}

fn spd_rolling_executor_commit_position(commit: SpdRollingExecutorCommit) -> usize {
    match commit {
        SpdRollingExecutorCommit::Accepted { position, .. }
        | SpdRollingExecutorCommit::Rejected { position, .. } => position,
    }
}

fn advance_idle_spd_rolling_executor_to_context(
    executor: Option<&mut SpdRollingExecutor>,
    context_tokens: &[i32],
) -> OpenAiResult<()> {
    if let Some(executor) = executor {
        executor
            .advance_idle_to_accepted_context(context_tokens)
            .map_err(openai_backend_error)?;
    }
    Ok(())
}

struct SpdRollingStartArgs<'a> {
    request: &'a EmbeddedStageZeroGeneration<'a>,
    downstream: &'a mut TcpStream,
    session_key: &'a str,
    shadow_session: Option<&'a mut SpdRollingShadowSession>,
    source_materialized_token_count: usize,
    spd: &'a mut SpdReplayProposalSource,
    executor: &'a mut SpdRollingExecutor,
    decode_step: usize,
    phase: super::spd::SpdInlineProbePhase,
    trigger_hf_index: Option<u32>,
}

#[derive(Debug)]
struct SpdRollingShadowLane {
    session_id: u64,
    session_key: String,
    token_count: usize,
}

impl SpdRollingShadowLane {
    fn new(token_count: usize) -> OpenAiResult<Self> {
        let session_id =
            OPENAI_GENERATION_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if session_id > i32::MAX as u64 {
            return Err(OpenAiError::backend(
                "SPD rolling shadow session id exceeds i32",
            ));
        }
        Ok(Self {
            session_id,
            session_key: session_id.to_string(),
            token_count,
        })
    }

    fn execution_session(&self) -> SpdExecutionSession<'_> {
        SpdExecutionSession {
            session_id: self.session_id,
            session_key: &self.session_key,
        }
    }
}

struct SpdRollingShadowSession {
    work: Option<SpdRollingShadowLane>,
    snapshots: std::collections::BTreeMap<usize, SpdRollingShadowLane>,
    snapshot_retention: usize,
}

impl SpdRollingShadowSession {
    fn new(logical_stage_count: usize) -> Self {
        Self {
            work: None,
            snapshots: std::collections::BTreeMap::new(),
            // Keep enough pre-step views for recurrent models, where canonical
            // KV cannot be rewound to an older materialized prefix.
            snapshot_retention: logical_stage_count.saturating_mul(2).max(3),
        }
    }

    fn ensure_work_at(
        &mut self,
        backend: &StageOpenAiBackend,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        source_session_id: u64,
        source_materialized_token_count: usize,
        token_count: usize,
    ) -> OpenAiResult<()> {
        if self.work.is_none() {
            let lane = SpdRollingShadowLane::new(token_count)?;
            backend.copy_embedded_stage_session(
                request,
                downstream,
                source_session_id,
                lane.session_id,
                token_count as u64,
            )
            .map_err(|error| {
                OpenAiError::backend(format!(
                    "failed to seed SPD rolling work shadow {} from source {} at token_count {}: {}",
                    lane.session_id, source_session_id, token_count, error
                ))
            })?;
            self.work = Some(lane);
            self.snapshot_work_at(backend, request, downstream, token_count)?;
            return Ok(());
        }

        let work = self
            .work
            .as_ref()
            .ok_or_else(|| OpenAiError::backend("missing SPD rolling work shadow"))?;
        if work.token_count != token_count {
            if self.snapshots.contains_key(&token_count) {
                self.swap_work_to_snapshot(backend, request, downstream, token_count)?;
            } else if token_count == source_materialized_token_count {
                self.reseed_work_from_source(
                    backend,
                    request,
                    downstream,
                    source_session_id,
                    token_count,
                )?;
            } else {
                return Err(self.missing_snapshot_error(token_count));
            }
        }
        self.prune_snapshots(backend, request, downstream)?;
        self.snapshot_work_at(backend, request, downstream, token_count)?;
        Ok(())
    }

    fn missing_snapshot_error(&self, token_count: usize) -> OpenAiError {
        let work_token_count = self.work.as_ref().map(|work| work.token_count);
        let snapshot_token_counts = self.snapshots.keys().copied().collect::<Vec<_>>();
        OpenAiError::backend(format!(
            "SPD rolling work shadow is at token_count {work_token_count:?}, cannot launch at {token_count}; available snapshots: {snapshot_token_counts:?}"
        ))
    }

    fn preserve_or_drop_current_work(
        &mut self,
        backend: &StageOpenAiBackend,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
    ) -> OpenAiResult<()> {
        let Some(current_work) = self.work.take() else {
            return Ok(());
        };
        match self.snapshots.entry(current_work.token_count) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(current_work);
            }
            std::collections::btree_map::Entry::Occupied(_) => {
                backend.drop_embedded_stage_session(
                    request,
                    downstream,
                    request.ids.request_id,
                    current_work.session_id,
                )?;
            }
        }
        Ok(())
    }

    fn swap_work_to_snapshot(
        &mut self,
        backend: &StageOpenAiBackend,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        token_count: usize,
    ) -> OpenAiResult<()> {
        let replacement = self
            .snapshots
            .remove(&token_count)
            .ok_or_else(|| self.missing_snapshot_error(token_count))?;
        self.preserve_or_drop_current_work(backend, request, downstream)?;
        self.work = Some(replacement);
        Ok(())
    }

    fn reseed_work_from_source(
        &mut self,
        backend: &StageOpenAiBackend,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        source_session_id: u64,
        token_count: usize,
    ) -> OpenAiResult<()> {
        self.preserve_or_drop_current_work(backend, request, downstream)?;
        let lane = SpdRollingShadowLane::new(token_count)?;
        backend.copy_embedded_stage_session(
            request,
            downstream,
            source_session_id,
            lane.session_id,
            token_count as u64,
        )
        .map_err(|error| {
            OpenAiError::backend(format!(
                "failed to reseed SPD rolling work shadow {} from source {} at token_count {}: {}",
                lane.session_id, source_session_id, token_count, error
            ))
        })?;
        self.work = Some(lane);
        Ok(())
    }

    fn prune_snapshots(
        &mut self,
        backend: &StageOpenAiBackend,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
    ) -> OpenAiResult<()> {
        let newest = self
            .work
            .as_ref()
            .map(|work| work.token_count)
            .into_iter()
            .chain(self.snapshots.keys().copied())
            .max()
            .unwrap_or(0);
        let min_keep = newest.saturating_sub(self.snapshot_retention);
        let stale = self
            .snapshots
            .keys()
            .copied()
            .filter(|token_count| *token_count < min_keep)
            .collect::<Vec<_>>();
        for token_count in stale {
            if let Some(snapshot) = self.snapshots.remove(&token_count) {
                backend.drop_embedded_stage_session(
                    request,
                    downstream,
                    request.ids.request_id,
                    snapshot.session_id,
                )?;
            }
        }
        Ok(())
    }

    fn empty_with_same_retention(&self) -> Self {
        Self {
            work: None,
            snapshots: std::collections::BTreeMap::new(),
            snapshot_retention: self.snapshot_retention,
        }
    }

    fn work_execution_session(&self) -> OpenAiResult<SpdExecutionSession<'_>> {
        self.work
            .as_ref()
            .map(SpdRollingShadowLane::execution_session)
            .ok_or_else(|| OpenAiError::backend("missing SPD rolling work shadow"))
    }

    fn advance_work_to(&mut self, token_count: usize) -> OpenAiResult<()> {
        let work = self
            .work
            .as_mut()
            .ok_or_else(|| OpenAiError::backend("missing SPD rolling work shadow"))?;
        work.token_count = token_count;
        Ok(())
    }

    fn snapshot_work_at(
        &mut self,
        backend: &StageOpenAiBackend,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        token_count: usize,
    ) -> OpenAiResult<Option<EmbeddedSessionControl>> {
        if self.snapshots.contains_key(&token_count) {
            return Ok(None);
        }
        let work_session_id = self
            .work
            .as_ref()
            .ok_or_else(|| OpenAiError::backend("missing SPD rolling work shadow"))?
            .session_id;
        let lane = SpdRollingShadowLane::new(token_count)?;
        let control = backend.copy_embedded_stage_session(
            request,
            downstream,
            work_session_id,
            lane.session_id,
            token_count as u64,
        )
        .map_err(|error| {
            OpenAiError::backend(format!(
                "failed to snapshot SPD rolling work shadow {} into snapshot {} at token_count {}: {}",
                work_session_id, lane.session_id, token_count, error
            ))
        })?;
        self.snapshots.insert(token_count, lane);
        Ok(Some(control))
    }

    fn snapshot_at(&self, token_count: usize) -> Option<&SpdRollingShadowLane> {
        self.snapshots.get(&token_count)
    }

    fn work_at(&self, token_count: usize) -> Option<&SpdRollingShadowLane> {
        self.work
            .as_ref()
            .filter(|work| work.token_count == token_count)
    }

    fn has_view_at(&self, token_count: usize) -> bool {
        self.work_at(token_count).is_some() || self.snapshots.contains_key(&token_count)
    }

    fn drain_lanes(self) -> Vec<SpdRollingShadowLane> {
        self.work
            .into_iter()
            .chain(self.snapshots.into_values())
            .collect()
    }
}

fn merge_session_control(
    total: &mut Option<EmbeddedSessionControl>,
    control: EmbeddedSessionControl,
) {
    match total {
        Some(total) => {
            total.elapsed_ms += control.elapsed_ms;
            total.local_ms += control.local_ms;
            total.downstream_write_ms += control.downstream_write_ms;
            total.downstream_wait_ms += control.downstream_wait_ms;
        }
        None => *total = Some(control),
    }
}

fn start_spd_rolling_executor_decode(
    backend: &StageOpenAiBackend,
    mut args: SpdRollingStartArgs<'_>,
) -> OpenAiResult<Option<(SpdOptimisticDecode, super::spd::SpdInlineProbe)>> {
    let Some(launch) = args
        .executor
        .prepare_launch(
            args.spd,
            args.decode_step,
            args.phase,
            args.request.spd_optimistic_min_logit_margin,
            args.trigger_hf_index,
        )
        .map_err(openai_backend_error)?
    else {
        return Ok(None);
    };
    let decode_step = launch
        .position
        .checked_sub(args.request.prompt_token_ids.len())
        .ok_or_else(|| {
            OpenAiError::backend("SPD rolling executor launch position is inside prompt")
        })?;
    if args
        .shadow_session
        .as_ref()
        .is_some_and(|shadow| shadow.work.is_none())
        && launch.position > args.source_materialized_token_count
    {
        args.executor
            .record_launch_miss(SpdRollingExecutorLaunchMissReason::ShadowNotSeedable);
        return Ok(None);
    }
    if launch.position < args.source_materialized_token_count
        && args
            .shadow_session
            .as_ref()
            .is_some_and(|shadow| !shadow.has_view_at(launch.position))
    {
        args.executor
            .record_launch_miss(SpdRollingExecutorLaunchMissReason::ShadowMissingView);
        return Ok(None);
    }
    let source_prefix_reseed = launch.position == args.source_materialized_token_count
        && args
            .shadow_session
            .as_ref()
            .is_some_and(|shadow| !shadow.has_view_at(launch.position));
    let decode = {
        let shadow = args
            .shadow_session
            .as_deref_mut()
            .ok_or_else(|| OpenAiError::backend("SPD rolling executor requires shadow KV"))?;
        shadow.ensure_work_at(
            backend,
            args.request,
            args.downstream,
            args.request.ids.session_id,
            args.source_materialized_token_count,
            launch.position,
        )?;
        if source_prefix_reseed {
            args.executor.record_shadow_source_reseed();
        }
        let execution_session = Some(shadow.work_execution_session()?);
        backend.start_spd_optimistic_decode_for_probe(
            args.request,
            SpdOptimisticDecodeStart {
                downstream: args.downstream,
                session_key: args.session_key,
                execution_session,
                pos_start: launch.position,
                decode_step,
                chain_depth: launch.chain_depth,
                chain_depth_limit: args.executor.logical_stage_count(),
                probe: &launch.probe,
                checkpoint: false,
            },
        )?
    };
    let Some(decode) = decode else {
        return Ok(None);
    };
    if let Some(shadow) = args.shadow_session.as_deref_mut() {
        shadow.advance_work_to(launch.position.saturating_add(1))?;
    }
    args.executor
        .record_launch(&launch, decode.origin)
        .map_err(openai_backend_error)?;
    Ok(Some((decode, launch.probe)))
}

fn drain_spd_rolling_executor_replies(
    backend: &StageOpenAiBackend,
    request: &EmbeddedStageZeroGeneration<'_>,
    queued: &mut std::collections::VecDeque<SpdOptimisticDecode>,
) -> OpenAiResult<usize> {
    let mut drained = 0;
    while let Some(decode) = queued.pop_front() {
        backend
            .recv_spd_aware_prediction_return_for_origin(
                request,
                WireReplyKind::PredictedTokens,
                decode.origin,
            )
            .map_err(openai_backend_error)?;
        drained += 1;
    }
    Ok(drained)
}

fn promote_spd_rolling_shadow_session(
    backend: &StageOpenAiBackend,
    request: &EmbeddedStageZeroGeneration<'_>,
    downstream: &mut TcpStream,
    shadow_session: &mut Option<SpdRollingShadowSession>,
    token_count: usize,
) -> OpenAiResult<Option<EmbeddedSessionControl>> {
    let Some(shadow) = shadow_session.as_mut() else {
        return Ok(None);
    };
    let source_session_id = if let Some(snapshot) = shadow.snapshot_at(token_count) {
        snapshot.session_id
    } else if let Some(work) = shadow.work_at(token_count) {
        work.session_id
    } else {
        return Err(OpenAiError::backend(format!(
            "missing exact SPD rolling shadow snapshot for token_count {token_count}"
        )));
    };
    let mut total = None;
    merge_session_control(
        &mut total,
        backend.copy_embedded_stage_session(
            request,
            downstream,
            source_session_id,
            request.ids.session_id,
            token_count as u64,
        )
        .map_err(|error| {
            OpenAiError::backend(format!(
                "failed to promote SPD rolling shadow {} into canonical {} at token_count {}: {}",
                source_session_id, request.ids.session_id, token_count, error
            ))
        })?,
    );
    // Do not consume or prune snapshots here. The rolling scheduler can still
    // launch from a pre-step view behind the newest work lane while younger
    // verifier work is in flight; request cleanup drains remaining snapshots.
    Ok(total)
}

fn drop_spd_rolling_shadow_session(
    backend: &StageOpenAiBackend,
    request: &EmbeddedStageZeroGeneration<'_>,
    downstream: &mut TcpStream,
    shadow_session: &mut Option<SpdRollingShadowSession>,
) -> OpenAiResult<Option<EmbeddedSessionControl>> {
    let Some(shadow) = shadow_session.take() else {
        return Ok(None);
    };
    let mut total = None;
    for lane in shadow.drain_lanes() {
        merge_session_control(
            &mut total,
            backend.drop_embedded_stage_session(
                request,
                downstream,
                request.ids.request_id,
                lane.session_id,
            )?,
        );
    }
    Ok(total)
}

fn reset_spd_rolling_shadow_session(
    backend: &StageOpenAiBackend,
    request: &EmbeddedStageZeroGeneration<'_>,
    downstream: &mut TcpStream,
    shadow_session: &mut Option<SpdRollingShadowSession>,
) -> OpenAiResult<Option<EmbeddedSessionControl>> {
    let replacement = shadow_session
        .as_ref()
        .map(SpdRollingShadowSession::empty_with_same_retention)
        .unwrap_or_else(|| SpdRollingShadowSession::new(0));
    let drop = drop_spd_rolling_shadow_session(backend, request, downstream, shadow_session)?;
    *shadow_session = Some(replacement);
    Ok(drop)
}

impl StageOpenAiBackend {
    pub(super) fn generate_embedded_stage_zero_tokens(
        &self,
        request: EmbeddedStageZeroGeneration<'_>,
        mut on_token: impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<GenerationCacheStats> {
        if request.config.downstream.is_none() {
            return self.generate_local_tokens(
                LocalGeneration {
                    prompt_token_ids: request.prompt_token_ids,
                    max_tokens: request.max_tokens,
                    sampling: request.sampling,
                    chat_sampling_metadata: request.chat_sampling_metadata,
                    hook_request: request.hook_request,
                    hook_runtime: request.hook_runtime,
                    cancellation: request.cancellation,
                    ids: request.ids,
                },
                on_token,
            );
        }

        let wire_sampling = wire_sampling_config(request.sampling);
        let session_id = request.ids.session_id;
        let request_id = request.ids.request_id;
        let session_key = session_id.to_string();
        let lane_pool = request
            .lane_pool
            .as_ref()
            .ok_or_else(|| OpenAiError::backend("embedded stage 0 has no downstream lane pool"))?;
        let mut lane = lane_pool.checkout(request.ids)?;
        if let Some(prediction_return) = request.prediction_return.as_ref() {
            match crate::binary_transport::direct_return::open_downstream_prediction_return_stream(
                request.config,
                request_id,
                session_id,
                request.wire_dtype,
            ) {
                Ok(stream) => {
                    prediction_return.attach_opened_stream(stream);
                }
                Err(error) => {
                    eprintln!(
                        "direct prediction return upstream-opened sink unavailable: {error:#}"
                    );
                }
            }
        }
        let mut cache_stats = GenerationCacheStats::default();

        let result = (|| {
            let downstream = &mut lane.stream;
            let prefill_token_count = request.prompt_token_ids.len().saturating_sub(1);
            let prefill_timer = PhaseTimer::start();
            let mut prefill_chunks = 0usize;
            let mut prefill_min_chunk_size = usize::MAX;
            let mut prefill_max_chunk_size = 0usize;
            let mut prefill_stage0_compute_ms = 0.0;
            let mut prefill_runtime_lock_wait_ms = 0.0;
            let mut prefill_runtime_lock_wait_max_ms = 0.0_f64;
            let mut prefill_runtime_lock_hold_ms = 0.0;
            let mut prefill_runtime_lock_hold_max_ms = 0.0_f64;
            let mut prefill_runtime_lock_acquires = 0usize;
            let mut prefill_runtime_sessions_before = None;
            let mut prefill_runtime_sessions_after = None;
            let mut prefill_forward_write_ms = 0.0;
            let mut prefill_output_activation_bytes = 0usize;
            let mut prefill_forward_activation_bytes = 0usize;
            let mut prefill_downstream_wait_ms = 0.0;
            let mut pending_prefill_replies = 0usize;
            let mut prefill_credit_wait_count = 0usize;
            let mut prefill_deferred_replies_drained = 0usize;
            let mut prefill_pending_replies_max = 0usize;
            let mut prefill_stage0_cache_hits = 0usize;
            let mut prefill_stage0_cache_misses = 0usize;
            let mut prefill_stage0_cache_errors = 0usize;
            let mut prefill_chain_cache_restored = false;
            let mut prefill_chain_restored_tokens = 0usize;
            let mut prefill_chain_cache_stats = StageReplyStats::default();
            let mut prefill_stage0_full_recorded = false;
            let mut fused_first_decode = None;
            let mut prefill_planner = request.prefill_chunk_policy.planner();
            if let Some(seed) = lane_pool.prefill_transport_seed() {
                prefill_planner.observe(seed);
            }
            if prefill_token_count > 0 {
                let prefill_tokens = &request.prompt_token_ids[..prefill_token_count];
                if self.kv.is_some() {
                    cache_stats.status = "miss";
                }
                if request.max_tokens > 0 && request.draft.is_none() {
                    let current = *request
                        .prompt_token_ids
                        .last()
                        .expect("checked non-empty prompt");
                    if let Some(cached) = self.try_restore_embedded_split_exact_replay(
                        &request,
                        &session_key,
                        downstream,
                    )? {
                        prefill_chain_cache_restored = true;
                        prefill_chain_restored_tokens = request
                            .prompt_token_ids
                            .len()
                            .saturating_add(cached.predicted_tokens.len().saturating_sub(1));
                        prefill_chain_cache_stats = cached.reply_stats;
                        cache_stats.cached_prompt_tokens =
                            saturating_u32(request.prompt_token_ids.len());
                        cache_stats.matched_prefix_tokens =
                            saturating_u32(request.prompt_token_ids.len());
                        cache_stats.suffix_prefill_tokens = 0;
                        cache_stats.status = "hit";
                        cache_stats.hit_kind = Some("chain_exact_replay");
                        fused_first_decode = Some(cached);
                    } else if let Some(cached) = self
                        .try_restore_embedded_split_full_prompt_first_token(
                            &request,
                            &session_key,
                            downstream,
                        )?
                    {
                        prefill_chain_cache_restored = true;
                        prefill_chain_restored_tokens = request.prompt_token_ids.len();
                        prefill_chain_cache_stats = cached.reply_stats;
                        cache_stats.cached_prompt_tokens =
                            saturating_u32(request.prompt_token_ids.len());
                        cache_stats.matched_prefix_tokens =
                            saturating_u32(request.prompt_token_ids.len());
                        cache_stats.suffix_prefill_tokens = 0;
                        cache_stats.status = "hit";
                        cache_stats.hit_kind = Some("chain_full_prompt_first_token");
                        fused_first_decode = Some(cached);
                    } else if let Some(fused) = self.try_restore_embedded_split_prefill_and_decode(
                        &request,
                        &session_key,
                        downstream,
                        prefill_tokens,
                        current,
                        wire_sampling.clone(),
                    )? {
                        prefill_chain_cache_restored = true;
                        prefill_chain_restored_tokens = prefill_token_count;
                        prefill_chain_cache_stats = fused.reply_stats;
                        cache_stats.cached_prompt_tokens = saturating_u32(prefill_token_count);
                        cache_stats.matched_prefix_tokens = saturating_u32(prefill_token_count);
                        cache_stats.suffix_prefill_tokens = 0;
                        cache_stats.status = "hit";
                        cache_stats.hit_kind = Some("chain_fused_exact_prefix");
                        fused_first_decode = Some(fused);
                    }
                }
                let split_prefill_restore = if prefill_chain_cache_restored {
                    None
                } else {
                    self.try_restore_embedded_split_prefill(
                        &request,
                        &session_key,
                        downstream,
                        prefill_tokens,
                    )?
                };
                if let Some(restore) = split_prefill_restore {
                    prefill_chain_restored_tokens = restore.restored_tokens;
                    prefill_chain_cache_restored =
                        prefill_chain_restored_tokens >= prefill_tokens.len();
                    prefill_chain_cache_stats = restore.stats;
                    cache_stats.cached_prompt_tokens =
                        saturating_u32(prefill_chain_restored_tokens);
                    cache_stats.matched_prefix_tokens =
                        saturating_u32(prefill_chain_restored_tokens);
                    cache_stats.suffix_prefill_tokens = saturating_u32(
                        prefill_tokens
                            .len()
                            .saturating_sub(prefill_chain_restored_tokens),
                    );
                    cache_stats.status = "hit";
                    cache_stats.hit_kind = Some("chain_prefix");
                }
                let mut pos_start = prefill_chain_restored_tokens.min(prefill_tokens.len());
                let mut chunk_index = 0usize;
                while pos_start < prefill_tokens.len() {
                    if request
                        .cancellation
                        .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                    {
                        drain_embedded_prefill_replies(
                            downstream,
                            &mut pending_prefill_replies,
                            &mut prefill_chain_cache_stats,
                        )?;
                        return Ok(());
                    }
                    let chunk_size =
                        prefill_planner.chunk_size_for(chunk_index, prefill_token_count);
                    let end = pos_start
                        .saturating_add(chunk_size)
                        .min(prefill_tokens.len());
                    let chunk = &prefill_tokens[pos_start..end];
                    prefill_min_chunk_size = prefill_min_chunk_size.min(chunk.len());
                    prefill_max_chunk_size = prefill_max_chunk_size.max(chunk.len());
                    let mut message = embedded_prefill_message(
                        request.wire_dtype,
                        OpenAiPrefillChunk {
                            seq_id: chunk_index,
                            pos_start,
                            prefill_token_count,
                            tokens: chunk,
                            request_id,
                            session_id,
                        },
                    )?;
                    self.mark_spd_tap_return(&request, &mut message);
                    let stage0_timer = PhaseTimer::start();
                    let pending_prefill_replies_before = pending_prefill_replies;
                    let mut output = self.restore_embedded_stage0_prefill(
                        &session_key,
                        request.ids,
                        pos_start as u64,
                        chunk,
                        request.activation_width,
                    )?;
                    if output.is_some() {
                        prefill_stage0_cache_hits += 1;
                    } else {
                        prefill_stage0_cache_misses += usize::from(pos_start == 0);
                    }
                    let output = match output.take() {
                        Some(output) => output,
                        None => {
                            let lock_timer = PhaseTimer::start();
                            let mut runtime = self
                                .runtime
                                .lock()
                                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                            let lock_wait_ms = lock_timer.elapsed_ms();
                            prefill_runtime_lock_wait_ms += lock_wait_ms;
                            prefill_runtime_lock_wait_max_ms =
                                prefill_runtime_lock_wait_max_ms.max(lock_wait_ms);
                            prefill_runtime_lock_acquires += 1;
                            let lock_hold_timer = PhaseTimer::start();
                            prefill_runtime_sessions_before
                                .get_or_insert_with(|| runtime.session_stats());
                            let output = run_binary_stage_message(
                                &mut runtime,
                                &session_key,
                                &message,
                                chunk,
                                None,
                                false,
                                0,
                            )
                            .map_err(openai_backend_error)?
                            .2;
                            prefill_runtime_sessions_after = Some(runtime.session_stats());
                            let lock_hold_ms = lock_hold_timer.elapsed_ms();
                            prefill_runtime_lock_hold_ms += lock_hold_ms;
                            prefill_runtime_lock_hold_max_ms =
                                prefill_runtime_lock_hold_max_ms.max(lock_hold_ms);
                            output
                        }
                    };
                    if let Err(error) = self.record_embedded_stage0_prefill(
                        &session_key,
                        request.ids,
                        pos_start as u64,
                        chunk,
                        request.activation_width,
                        &output,
                    ) {
                        prefill_stage0_cache_errors += 1;
                        let mut attrs = self.openai_attrs(request.ids);
                        attrs.insert(
                            "skippy.kv.decision".to_string(),
                            json!("stage0_record_error"),
                        );
                        attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
                        self.telemetry
                            .emit("stage.openai_kv_record_decision", attrs);
                    }
                    self.record_spd_stage0_boundary_tap(&request, &message, &output);
                    let chunk_stage0_compute_ms = stage0_timer.elapsed_ms();
                    prefill_stage0_compute_ms += chunk_stage0_compute_ms;
                    let mut forwarded = forwarded_stage_message(
                        request.config,
                        &message,
                        &output,
                        request.wire_dtype,
                        request.activation_width,
                    )
                    .map_err(openai_backend_error)?;
                    self.mark_spd_tap_return(&request, &mut forwarded);
                    prefill_output_activation_bytes =
                        prefill_output_activation_bytes.saturating_add(output.payload.len());
                    prefill_forward_activation_bytes =
                        prefill_forward_activation_bytes.saturating_add(forwarded.activation.len());
                    let write_timer = PhaseTimer::start();
                    write_stage_message_conditioned(
                        &mut *downstream,
                        &forwarded,
                        request.wire_dtype,
                        request.downstream_wire_condition,
                    )
                    .map_err(openai_io_error)?;
                    let chunk_forward_write_ms = write_timer.elapsed_ms();
                    prefill_forward_write_ms += chunk_forward_write_ms;
                    let mut chunk_downstream_wait_ms = 0.0;
                    let mut chunk_deferred_replies_drained = 0usize;
                    let mut chunk_credit_wait_count = 0usize;
                    if request.prefill_reply_credit_limit == 0 {
                        let wait_timer = PhaseTimer::start();
                        let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
                        chunk_downstream_wait_ms = wait_timer.elapsed_ms();
                        if reply.kind != WireReplyKind::Ack {
                            return Err(OpenAiError::backend(format!(
                                "expected prefill ACK from downstream, got {:?}",
                                reply.kind
                            )));
                        }
                        prefill_chain_cache_stats.merge(reply.stats);
                    } else {
                        while pending_prefill_replies >= request.prefill_reply_credit_limit {
                            prefill_credit_wait_count = prefill_credit_wait_count.saturating_add(1);
                            chunk_credit_wait_count = chunk_credit_wait_count.saturating_add(1);
                            let drained = drain_one_embedded_prefill_reply(
                                downstream,
                                &mut pending_prefill_replies,
                                &mut prefill_chain_cache_stats,
                            )?;
                            prefill_deferred_replies_drained = prefill_deferred_replies_drained
                                .saturating_add(drained.drained_replies);
                            chunk_deferred_replies_drained = chunk_deferred_replies_drained
                                .saturating_add(drained.drained_replies);
                            chunk_downstream_wait_ms += drained.downstream_wait_ms;
                        }
                        pending_prefill_replies = pending_prefill_replies.saturating_add(1);
                        prefill_pending_replies_max =
                            prefill_pending_replies_max.max(pending_prefill_replies);
                    }
                    prefill_downstream_wait_ms += chunk_downstream_wait_ms;
                    prefill_planner.observe(PrefillChunkObservation {
                        compute_ms: chunk_stage0_compute_ms,
                        forward_write_ms: chunk_forward_write_ms,
                        downstream_wait_ms: chunk_downstream_wait_ms,
                    });
                    let mut chunk_attrs = self.openai_attrs(request.ids);
                    chunk_attrs
                        .insert("llama_stage.message_kind".to_string(), json!("PrefillEmbd"));
                    chunk_attrs.insert("llama_stage.seq_id".to_string(), json!(chunk_index));
                    chunk_attrs.insert("llama_stage.pos_start".to_string(), json!(pos_start));
                    chunk_attrs.insert("llama_stage.token_count".to_string(), json!(chunk.len()));
                    chunk_attrs.insert(
                        "llama_stage.stage0_compute_ms".to_string(),
                        json!(chunk_stage0_compute_ms),
                    );
                    chunk_attrs.insert(
                        "llama_stage.forward_write_ms".to_string(),
                        json!(chunk_forward_write_ms),
                    );
                    chunk_attrs.insert(
                        "llama_stage.downstream_wait_ms".to_string(),
                        json!(chunk_downstream_wait_ms),
                    );
                    chunk_attrs.insert(
                        "llama_stage.output_activation_bytes".to_string(),
                        json!(output.payload.len()),
                    );
                    chunk_attrs.insert(
                        "llama_stage.forward_activation_bytes".to_string(),
                        json!(forwarded.activation.len()),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_credit_limit".to_string(),
                        json!(request.prefill_reply_credit_limit),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_pending_replies_before".to_string(),
                        json!(pending_prefill_replies_before),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_pending_replies_after".to_string(),
                        json!(pending_prefill_replies),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_credit_wait_count".to_string(),
                        json!(chunk_credit_wait_count),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_deferred_replies_drained".to_string(),
                        json!(chunk_deferred_replies_drained),
                    );
                    self.telemetry.emit_debug_span(
                        "stage.openai_prefill_chunk",
                        chunk_attrs,
                        stage0_timer.start_unix_nanos,
                        now_unix_nanos() as u64,
                    );
                    prefill_chunks += 1;
                    pos_start = end;
                    chunk_index += 1;
                }
                let drained = drain_embedded_prefill_replies(
                    downstream,
                    &mut pending_prefill_replies,
                    &mut prefill_chain_cache_stats,
                )?;
                prefill_deferred_replies_drained =
                    prefill_deferred_replies_drained.saturating_add(drained.drained_replies);
                prefill_downstream_wait_ms += drained.downstream_wait_ms;
                lane_pool.observe_prefill_transport(
                    &prefill_chain_cache_stats,
                    prefill_stage0_compute_ms,
                    prefill_chunks,
                );
                if !prefill_chain_cache_restored {
                    prefill_stage0_full_recorded = self.record_embedded_stage0_full_prefill(
                        &session_key,
                        request.ids,
                        prefill_tokens,
                    )?;
                }
            }
            let mut prefill_attrs = self.openai_attrs(request.ids);
            prefill_attrs.insert(
                "llama_stage.prefill_token_count".to_string(),
                json!(prefill_token_count),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_chunk_count".to_string(),
                json!(prefill_chunks),
            );
            attrs_insert_prefill_chunk_policy(
                &mut prefill_attrs,
                request.prefill_chunk_policy,
                prefill_min_chunk_size,
                prefill_max_chunk_size,
            );
            prefill_attrs.insert(
                "llama_stage.stage0_compute_ms".to_string(),
                json!(prefill_stage0_compute_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(prefill_runtime_lock_wait_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_wait_max_ms".to_string(),
                json!(prefill_runtime_lock_wait_max_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(prefill_runtime_lock_hold_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_hold_max_ms".to_string(),
                json!(prefill_runtime_lock_hold_max_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(prefill_runtime_lock_acquires),
            );
            if let Some(stats) = prefill_runtime_sessions_before.as_ref() {
                Self::insert_runtime_session_stats(
                    &mut prefill_attrs,
                    "llama_stage.runtime_sessions_before",
                    stats,
                );
            }
            if let Some(stats) = prefill_runtime_sessions_after.as_ref() {
                Self::insert_runtime_session_stats(
                    &mut prefill_attrs,
                    "llama_stage.runtime_sessions_after",
                    stats,
                );
            }
            prefill_attrs.insert(
                "llama_stage.forward_write_ms".to_string(),
                json!(prefill_forward_write_ms),
            );
            prefill_attrs.insert(
                "llama_stage.output_activation_bytes".to_string(),
                json!(prefill_output_activation_bytes),
            );
            prefill_attrs.insert(
                "llama_stage.forward_activation_bytes".to_string(),
                json!(prefill_forward_activation_bytes),
            );
            prefill_attrs.insert(
                "llama_stage.downstream_wait_ms".to_string(),
                json!(prefill_downstream_wait_ms),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_edge_write_us_max".to_string(),
                json!(prefill_chain_cache_stats.prefill_edge_write_us_max),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_edge_wait_us_max".to_string(),
                json!(prefill_chain_cache_stats.prefill_edge_wait_us_max),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_edge_total_us_max".to_string(),
                json!(prefill_chain_cache_stats.prefill_edge_total_us_max),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_edge_stage_index".to_string(),
                json!(prefill_chain_cache_stats.prefill_edge_stage_index),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_edge_observation_count".to_string(),
                json!(prefill_chain_cache_stats.prefill_edge_observation_count),
            );
            prefill_attrs.insert(
                "skippy.prefill_credit_limit".to_string(),
                json!(request.prefill_reply_credit_limit),
            );
            prefill_attrs.insert(
                "skippy.prefill_pending_replies_max".to_string(),
                json!(prefill_pending_replies_max),
            );
            prefill_attrs.insert(
                "skippy.prefill_pending_replies_after".to_string(),
                json!(pending_prefill_replies),
            );
            prefill_attrs.insert(
                "skippy.prefill_credit_wait_count".to_string(),
                json!(prefill_credit_wait_count),
            );
            prefill_attrs.insert(
                "skippy.prefill_deferred_replies_drained".to_string(),
                json!(prefill_deferred_replies_drained),
            );
            prefill_attrs.insert(
                "skippy.kv.stage0_cache_hits".to_string(),
                json!(prefill_stage0_cache_hits),
            );
            prefill_attrs.insert(
                "skippy.kv.stage0_cache_misses".to_string(),
                json!(prefill_stage0_cache_misses),
            );
            prefill_attrs.insert(
                "skippy.kv.stage0_cache_errors".to_string(),
                json!(prefill_stage0_cache_errors),
            );
            prefill_attrs.insert(
                "skippy.kv.stage0_full_recorded".to_string(),
                json!(prefill_stage0_full_recorded),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_restored".to_string(),
                json!(prefill_chain_cache_restored),
            );
            prefill_attrs.insert(
                "skippy.kv.matched_prefix_tokens".to_string(),
                json!(prefill_chain_restored_tokens),
            );
            prefill_attrs.insert(
                "skippy.kv.suffix_prefill_tokens".to_string(),
                json!(prefill_token_count.saturating_sub(prefill_chain_restored_tokens)),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_hits".to_string(),
                json!(prefill_chain_cache_stats.kv_lookup_hits),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_misses".to_string(),
                json!(prefill_chain_cache_stats.kv_lookup_misses),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_errors".to_string(),
                json!(prefill_chain_cache_stats.kv_lookup_errors),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_hit_stage_mask".to_string(),
                json!(prefill_chain_cache_stats.kv_hit_stage_mask),
            );
            super::prefix_cache::insert_chain_prefix_cache_savings_attrs(
                &mut prefill_attrs,
                super::prefix_cache::chain_prefix_cache_savings(
                    &prefill_chain_cache_stats,
                    prefill_chain_restored_tokens,
                    request.wire_dtype,
                    request.activation_width,
                ),
            );
            self.emit_openai_phase("stage.openai_prefill", prefill_timer, prefill_attrs);

            let message = generation_config_message(
                request.wire_dtype,
                request_id,
                session_id,
                request.prompt_token_ids.len(),
                wire_sampling.clone(),
                request.chat_sampling_metadata,
            )?;
            write_stage_message_conditioned(
                &mut *downstream,
                &message,
                request.wire_dtype,
                request.downstream_wire_condition,
            )
            .map_err(openai_io_error)?;
            let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
            if reply.kind != WireReplyKind::Ack {
                return Err(OpenAiError::backend(format!(
                    "expected generation config ACK from downstream, got {:?}",
                    reply.kind
                )));
            }

            let decode_timer = PhaseTimer::start();
            let mut decoded_tokens = 0usize;
            let mut decode_stage0_compute_ms = 0.0;
            let mut decode_runtime_lock_wait_ms = 0.0;
            let mut decode_runtime_lock_wait_max_ms = 0.0_f64;
            let mut decode_runtime_lock_hold_ms = 0.0;
            let mut decode_runtime_lock_hold_max_ms = 0.0_f64;
            let mut decode_runtime_lock_acquires = 0usize;
            let mut decode_batch_size_max = 1usize;
            let mut decode_batch_wait_ms = 0.0;
            let mut decode_forward_write_ms = 0.0;
            let mut decode_forward_activation_encode_ms = 0.0;
            let mut decode_output_activation_bytes = 0usize;
            let mut decode_forward_activation_bytes = 0usize;
            let mut decode_downstream_wait_ms = 0.0;
            let mut current = *request
                .prompt_token_ids
                .last()
                .expect("checked non-empty prompt");
            let mut context_tokens = request.prompt_token_ids.to_vec();
            let mut exact_replay_tokens = Vec::new();
            let mut decode_message = ReusableDecodeMessage::new(
                request.wire_dtype,
                ReusableDecodeMessageArgs {
                    request_id,
                    session_id,
                    prompt_token_count: request.prompt_token_ids.len(),
                    base_pos_start: prefill_token_count,
                    sampling: wire_sampling.clone(),
                    sideband_capacity: skippy_protocol::binary::MAX_STAGE_SIDEBAND_VALUES,
                },
            )?;
            if request.spd.is_some() {
                decode_message.enable_spd_tap_return();
            }
            let mut fused_reached_stop = false;
            let mut native_mtp = NativeMtpN1Verifier::default();
            let native_mtp_options = NativeMtpDecodeOptions::from_env();
            let mut native_mtp_counters = NativeMtpDecodeCounters::default();
            let mut native_mtp_reject_cooldown_remaining = 0usize;
            let mut native_mtp_suppress_cooldown_drafts_remaining = 0usize;
            if let Some(mut fused) = fused_first_decode.take() {
                current = fused.predicted;
                let mut fused_native_mtp_draft = fused.native_mtp_draft.take();
                decode_stage0_compute_ms += fused.execution.stage0_compute_ms;
                decode_runtime_lock_wait_ms += fused.execution.runtime_lock_wait_ms;
                decode_runtime_lock_wait_max_ms =
                    decode_runtime_lock_wait_max_ms.max(fused.execution.runtime_lock_wait_ms);
                decode_runtime_lock_hold_ms += fused.execution.runtime_lock_hold_ms;
                decode_runtime_lock_hold_max_ms =
                    decode_runtime_lock_hold_max_ms.max(fused.execution.runtime_lock_hold_ms);
                decode_runtime_lock_acquires += 1;
                decode_forward_activation_encode_ms += fused.execution.activation_encode_ms;
                decode_output_activation_bytes = decode_output_activation_bytes
                    .saturating_add(fused.execution.output_activation_bytes);
                decode_forward_activation_bytes = decode_forward_activation_bytes
                    .saturating_add(fused.execution.forward_activation_bytes);
                decode_forward_write_ms += fused.execution.forward_write_ms;
                decode_downstream_wait_ms += fused.execution.downstream_wait_ms;
                for (index, token) in fused.predicted_tokens.iter().copied().enumerate() {
                    if decoded_tokens >= request.max_tokens as usize {
                        break;
                    }
                    current = token;
                    exact_replay_tokens.push(current);
                    context_tokens.push(current);
                    let native_mtp_decision = native_mtp.observe_target_token(
                        current,
                        if index == 0 {
                            ms_to_us(fused.execution.downstream_wait_ms)
                        } else {
                            0
                        },
                        if index == 0 {
                            fused_native_mtp_draft.take()
                        } else {
                            None
                        },
                        NativeMtpDraftOrigin::InitialSerial,
                    );
                    decoded_tokens += 1;
                    if self.telemetry.is_debug_enabled() {
                        let mut token_attrs = self.openai_attrs(request.ids);
                        token_attrs.insert("llama_stage.decode_step".to_string(), json!(index));
                        token_attrs.insert(
                            "llama_stage.decode_token_phase".to_string(),
                            json!(fused.token_phase),
                        );
                        token_attrs.insert(
                            "llama_stage.message_kind".to_string(),
                            json!(fused.message_kind),
                        );
                        token_attrs.insert(
                            "llama_stage.elapsed_ms".to_string(),
                            json!(if index == 0 { fused.elapsed_ms } else { 0.0 }),
                        );
                        token_attrs.insert(
                            "llama_stage.cached_replay_token_index".to_string(),
                            json!(index),
                        );
                        token_attrs.insert(
                            "llama_stage.cached_replay_token_count".to_string(),
                            json!(fused.predicted_tokens.len()),
                        );
                        token_attrs.insert(
                            "llama_stage.stage0_compute_ms".to_string(),
                            json!(if index == 0 {
                                fused.execution.stage0_compute_ms
                            } else {
                                0.0
                            }),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_wait_ms".to_string(),
                            json!(if index == 0 {
                                fused.execution.runtime_lock_wait_ms
                            } else {
                                0.0
                            }),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_hold_ms".to_string(),
                            json!(if index == 0 {
                                fused.execution.runtime_lock_hold_ms
                            } else {
                                0.0
                            }),
                        );
                        token_attrs.insert(
                            "llama_stage.output_activation_bytes".to_string(),
                            json!(if index == 0 {
                                fused.execution.output_activation_bytes
                            } else {
                                0
                            }),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_activation_bytes".to_string(),
                            json!(if index == 0 {
                                fused.execution.forward_activation_bytes
                            } else {
                                0
                            }),
                        );
                        token_attrs.insert(
                            "llama_stage.activation_encode_ms".to_string(),
                            json!(if index == 0 {
                                fused.execution.activation_encode_ms
                            } else {
                                0.0
                            }),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_write_ms".to_string(),
                            json!(if index == 0 {
                                fused.execution.forward_write_ms
                            } else {
                                0.0
                            }),
                        );
                        token_attrs.insert(
                            "llama_stage.downstream_wait_ms".to_string(),
                            json!(if index == 0 {
                                fused.execution.downstream_wait_ms
                            } else {
                                0.0
                            }),
                        );
                        token_attrs
                            .insert("llama_stage.predicted_token".to_string(), json!(current));
                        token_attrs.insert(
                            "llama_stage.native_mtp.verification".to_string(),
                            json!(native_mtp_decision.label()),
                        );
                        self.telemetry
                            .emit_debug("stage.openai_decode_token", token_attrs);
                    }
                    if on_token(current)? == TokenControl::Stop {
                        fused_reached_stop = true;
                        break;
                    }
                }
            }
            let max_speculative_window = request.speculative_window.max(1);
            let mut adaptive_window = if request.adaptive_speculative_window {
                max_speculative_window.min(4)
            } else {
                max_speculative_window
            };
            let mut speculative_stats = OpenAiSpeculativeStats {
                adaptive_window_start: adaptive_window,
                adaptive_window_final: adaptive_window,
                adaptive_window_max: max_speculative_window,
                adaptive_window_min: if request.draft.is_some() || request.spd.is_some() {
                    adaptive_window
                } else {
                    0
                },
                adaptive_window_max_seen: adaptive_window,
                adaptive_window_enabled: request.adaptive_speculative_window,
                ..OpenAiSpeculativeStats::default()
            };
            let mut spd_guard = match request.spd.as_ref() {
                Some(spd) if request.speculative_window > 0 => {
                    let spd_reset_timer = PhaseTimer::start();
                    let mut spd = spd
                        .source
                        .lock()
                        .map_err(|_| OpenAiError::backend("SPD source lock poisoned"))?;
                    spd.reset_to_context(&context_tokens)
                        .map_err(openai_backend_error)?;
                    speculative_stats.draft_reset_ms += spd_reset_timer.elapsed_ms();
                    let mut attrs = self.openai_attrs(request.ids);
                    attrs.insert(
                        "llama_stage.spd_manifest_path".to_string(),
                        json!(spd.manifest_path.display().to_string()),
                    );
                    attrs.insert(
                        "llama_stage.spd_model_path".to_string(),
                        json!(
                            spd.model_path
                                .as_ref()
                                .map(|path| path.display().to_string())
                        ),
                    );
                    attrs.insert(
                        "llama_stage.spd_model_source".to_string(),
                        json!(spd.model_source.clone()),
                    );
                    attrs.insert(
                        "llama_stage.speculative_window".to_string(),
                        json!(spd.window),
                    );
                    attrs.insert(
                        "llama_stage.adaptive_speculative_window".to_string(),
                        json!(request.adaptive_speculative_window),
                    );
                    self.emit_openai_phase("stage.openai_spd_reset", spd_reset_timer, attrs);
                    Some(spd)
                }
                _ => None,
            };
            let mut draft_guard = match request.draft.as_ref() {
                Some(draft) if request.speculative_window > 0 => {
                    let draft_reset_timer = PhaseTimer::start();
                    let mut draft = draft
                        .lock()
                        .map_err(|_| OpenAiError::backend("draft model lock poisoned"))?;
                    draft
                        .reset_to_context(&context_tokens)
                        .map_err(openai_backend_error)?;
                    speculative_stats.draft_reset_ms += draft_reset_timer.elapsed_ms();
                    let mut attrs = self.openai_attrs(request.ids);
                    attrs.insert(
                        "llama_stage.draft_model_path".to_string(),
                        json!(draft.path.display().to_string()),
                    );
                    attrs.insert(
                        "llama_stage.speculative_window".to_string(),
                        json!(draft.window),
                    );
                    attrs.insert(
                        "llama_stage.adaptive_speculative_window".to_string(),
                        json!(request.adaptive_speculative_window),
                    );
                    self.emit_openai_phase("stage.openai_draft_reset", draft_reset_timer, attrs);
                    Some(draft)
                }
                _ => None,
            };
            let spd_optimistic_chain_depth_limit = spd_guard
                .as_deref()
                .map(SpdReplayProposalSource::logical_stage_count)
                .unwrap_or(0)
                .saturating_sub(1);
            let mut spd_rolling_executor = None::<SpdRollingExecutor>;
            let mut spd_rolling_shadow_session = None::<SpdRollingShadowSession>;
            while decoded_tokens < request.max_tokens as usize {
                let decode_step = u32::try_from(decoded_tokens)
                    .map_err(|_| OpenAiError::backend("decode step exceeds u32"))?;
                if fused_reached_stop {
                    break;
                }
                if decoded_tokens >= request.max_tokens as usize {
                    break;
                }
                if request
                    .cancellation
                    .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                {
                    break;
                }
                let token_timer = PhaseTimer::start();
                let prefer_rolling_spd_optimistic_decode = request.spd_optimistic_decode
                    && draft_guard.is_none()
                    && spd_guard.is_some()
                    && request.sampling.temperature <= 0.0;
                let use_spd_rolling_executor =
                    request.spd_rolling_executor && prefer_rolling_spd_optimistic_decode;
                if use_spd_rolling_executor && spd_rolling_executor.is_none() {
                    let logical_stage_count = spd_guard
                        .as_deref()
                        .map(SpdReplayProposalSource::logical_stage_count)
                        .unwrap_or(0);
                    spd_rolling_executor = Some(
                        SpdRollingExecutor::new(logical_stage_count, &context_tokens)
                            .map_err(openai_backend_error)?,
                    );
                    spd_rolling_shadow_session =
                        Some(SpdRollingShadowSession::new(logical_stage_count));
                }
                let native_mtp_remaining =
                    (request.max_tokens as usize).saturating_sub(decoded_tokens);
                let can_run_native_mtp_batched_verify = native_mtp_options.batched_verify
                    && native_mtp_reject_cooldown_remaining == 0
                    && draft_guard.is_none()
                    && spd_guard.is_none()
                    && native_mtp_remaining >= 2;
                let pending_native_mtp_draft = can_run_native_mtp_batched_verify
                    .then(|| native_mtp.take_pending_draft())
                    .flatten();
                if let Some(pending_native_mtp_draft) = pending_native_mtp_draft {
                    let batched_token_timer =
                        self.telemetry.is_debug_enabled().then(PhaseTimer::start);
                    let native_mtp_draft_token = pending_native_mtp_draft.token;
                    let native_mtp_draft_origin = pending_native_mtp_draft.origin;
                    let verify_inputs = [current, native_mtp_draft_token];
                    let message = embedded_verify_message(
                        request.wire_dtype,
                        VerifySpanMessageArgs {
                            request_id,
                            session_id,
                            prompt_token_count: request.prompt_token_ids.len(),
                            pos_start: prefill_token_count + decoded_tokens,
                            decode_step: decoded_tokens,
                            checkpoint_generation: 0,
                            tokens: &verify_inputs,
                            sampling: wire_sampling.clone(),
                            checkpoint: false,
                        },
                    )?;
                    let verify = self.execute_embedded_stage_message(
                        &request,
                        downstream,
                        &session_key,
                        &message,
                        &verify_inputs,
                        WireReplyKind::PredictedTokens,
                    )?;
                    if verify.reply.predicted_tokens.len() < verify_inputs.len() {
                        return Err(OpenAiError::backend(format!(
                            "native MTP verify span returned too few tokens: got {} expected {}",
                            verify.reply.predicted_tokens.len(),
                            verify_inputs.len()
                        )));
                    }
                    let target_token = verify.reply.predicted_tokens[0];
                    let after_draft_token = verify.reply.predicted_tokens[1];
                    let verify_next_mtp_draft = NativeMtpDraft::from_verify_prediction_tokens(
                        &verify.reply.predicted_tokens,
                        verify_inputs.len(),
                    );
                    let native_mtp_decision = native_mtp.observe_taken_draft_verification(
                        native_mtp_draft_token,
                        target_token,
                        ms_to_us(verify.elapsed_ms),
                    );
                    let accepted =
                        matches!(native_mtp_decision, NativeMtpVerification::Accepted { .. });
                    native_mtp_counters
                        .observe_batched_verification(native_mtp_draft_origin, accepted);
                    let commit_tokens = [target_token, after_draft_token];
                    let commit_token_count = if accepted { 2 } else { 1 };
                    let consumed_positions = verify_inputs.len();
                    let mut committed_positions = 0usize;
                    let mut reached_stop = false;
                    for token in commit_tokens.into_iter().take(commit_token_count) {
                        current = token;
                        decoded_tokens += 1;
                        committed_positions += 1;
                        exact_replay_tokens.push(current);
                        context_tokens.push(current);
                        if on_token(current)? == TokenControl::Stop {
                            reached_stop = true;
                            break;
                        }
                        if decoded_tokens >= request.max_tokens as usize {
                            break;
                        }
                    }
                    if !accepted && native_mtp_options.reject_cooldown_tokens > 0 {
                        native_mtp_reject_cooldown_remaining =
                            native_mtp_options.reject_cooldown_tokens;
                        native_mtp_suppress_cooldown_drafts_remaining =
                            native_mtp_options.suppress_cooldown_draft_limit;
                        native_mtp.clear_pending_draft();
                    }
                    let verify_next_mtp_draft_available = verify_next_mtp_draft.is_some();
                    let verify_next_mtp_draft_adopted = accepted
                        && committed_positions == consumed_positions
                        && !reached_stop
                        && decoded_tokens < request.max_tokens as usize
                        && verify_next_mtp_draft.is_some();
                    native_mtp_counters.observe_verify_next_draft(
                        verify_next_mtp_draft_available,
                        verify_next_mtp_draft_adopted,
                    );
                    if verify_next_mtp_draft_adopted {
                        native_mtp.observe_next_draft(
                            verify_next_mtp_draft,
                            NativeMtpDraftOrigin::VerifyNext,
                        );
                    }
                    let mut trim_control = None;
                    match native_mtp_trim_action(committed_positions, consumed_positions) {
                        NativeMtpTrimAction::None => {}
                        NativeMtpTrimAction::FullSession => {
                            let target_token_count = prefill_token_count + decoded_tokens;
                            let defer_trim =
                                native_mtp_options.defer_reject_trim && !accepted && !reached_stop;
                            let trim = if defer_trim {
                                let trim = self.trim_embedded_stage_session_local(
                                    &session_key,
                                    target_token_count,
                                )?;
                                native_mtp_counters.observe_deferred_reject_trim(trim.local_ms);
                                trim
                            } else {
                                self.trim_embedded_stage_session(
                                    &request,
                                    downstream,
                                    &session_key,
                                    request_id,
                                    session_id,
                                    target_token_count,
                                )?
                            };
                            trim_control = Some(trim);
                        }
                    }
                    decode_stage0_compute_ms += verify.stats.stage0_compute_ms;
                    decode_runtime_lock_wait_ms += verify.stats.runtime_lock_wait_ms;
                    decode_runtime_lock_wait_max_ms =
                        decode_runtime_lock_wait_max_ms.max(verify.stats.runtime_lock_wait_ms);
                    decode_runtime_lock_hold_ms += verify.stats.runtime_lock_hold_ms;
                    decode_runtime_lock_hold_max_ms =
                        decode_runtime_lock_hold_max_ms.max(verify.stats.runtime_lock_hold_ms);
                    decode_runtime_lock_acquires += 1;
                    decode_forward_activation_encode_ms += verify.stats.activation_encode_ms;
                    decode_output_activation_bytes = decode_output_activation_bytes
                        .saturating_add(verify.stats.output_activation_bytes);
                    decode_forward_activation_bytes = decode_forward_activation_bytes
                        .saturating_add(verify.stats.forward_activation_bytes);
                    decode_forward_write_ms += verify.stats.forward_write_ms;
                    decode_downstream_wait_ms += verify.stats.downstream_wait_ms;
                    if let Some(batched_token_timer) = batched_token_timer {
                        let mut token_attrs = self.openai_attrs(request.ids);
                        token_attrs
                            .insert("llama_stage.decode_step".to_string(), json!(decode_step));
                        token_attrs
                            .insert("llama_stage.message_kind".to_string(), json!("VerifySpan"));
                        token_attrs.insert(
                            "llama_stage.native_mtp.batched_verification".to_string(),
                            json!(true),
                        );
                        token_attrs.insert(
                            "llama_stage.native_mtp.verification".to_string(),
                            json!(native_mtp_decision.label()),
                        );
                        token_attrs.insert(
                            "llama_stage.native_mtp.verify_elapsed_ms".to_string(),
                            json!(verify.elapsed_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.native_mtp.draft_token".to_string(),
                            json!(native_mtp_draft_token),
                        );
                        token_attrs.insert(
                            "llama_stage.native_mtp.pending_origin".to_string(),
                            json!(native_mtp_draft_origin.label()),
                        );
                        token_attrs.insert(
                            "llama_stage.native_mtp.target_token".to_string(),
                            json!(target_token),
                        );
                        token_attrs.insert(
                            "llama_stage.native_mtp.after_draft_token".to_string(),
                            json!(after_draft_token),
                        );
                        token_attrs.insert(
                            "llama_stage.native_mtp.verify_next_draft_available".to_string(),
                            json!(verify_next_mtp_draft_available),
                        );
                        token_attrs.insert(
                            "llama_stage.native_mtp.verify_next_draft_adopted".to_string(),
                            json!(verify_next_mtp_draft_adopted),
                        );
                        if let Some(next_draft) = verify_next_mtp_draft {
                            token_attrs.insert(
                                "llama_stage.native_mtp.verify_next_draft_token".to_string(),
                                json!(next_draft.token),
                            );
                            token_attrs.insert(
                                "llama_stage.native_mtp.verify_next_draft_compute_us".to_string(),
                                json!(next_draft.proposal_compute_us),
                            );
                        }
                        token_attrs.insert(
                            "llama_stage.native_mtp.consumed_positions".to_string(),
                            json!(consumed_positions),
                        );
                        token_attrs.insert(
                            "llama_stage.native_mtp.committed_positions".to_string(),
                            json!(committed_positions),
                        );
                        token_attrs.insert(
                            "llama_stage.native_mtp.reject_cooldown_tokens".to_string(),
                            json!(native_mtp_options.reject_cooldown_tokens),
                        );
                        token_attrs.insert(
                            "llama_stage.native_mtp.reject_cooldown_remaining".to_string(),
                            json!(native_mtp_reject_cooldown_remaining),
                        );
                        token_attrs.insert(
                            "llama_stage.native_mtp.defer_reject_trim".to_string(),
                            json!(native_mtp_options.defer_reject_trim),
                        );
                        if let Some(trim) = trim_control.as_ref() {
                            token_attrs.insert(
                                "llama_stage.native_mtp.trim_ms".to_string(),
                                json!(trim.elapsed_ms),
                            );
                            token_attrs.insert(
                                "llama_stage.native_mtp.trim_local_ms".to_string(),
                                json!(trim.local_ms),
                            );
                            token_attrs.insert(
                                "llama_stage.native_mtp.trim_downstream_write_ms".to_string(),
                                json!(trim.downstream_write_ms),
                            );
                            token_attrs.insert(
                                "llama_stage.native_mtp.trim_downstream_wait_ms".to_string(),
                                json!(trim.downstream_wait_ms),
                            );
                        }
                        token_attrs.insert(
                            "llama_stage.stage0_compute_ms".to_string(),
                            json!(verify.stats.stage0_compute_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_wait_ms".to_string(),
                            json!(verify.stats.runtime_lock_wait_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_hold_ms".to_string(),
                            json!(verify.stats.runtime_lock_hold_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.activation_encode_ms".to_string(),
                            json!(verify.stats.activation_encode_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_write_ms".to_string(),
                            json!(verify.stats.forward_write_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.downstream_wait_ms".to_string(),
                            json!(verify.stats.downstream_wait_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.output_activation_bytes".to_string(),
                            json!(verify.stats.output_activation_bytes),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_activation_bytes".to_string(),
                            json!(verify.stats.forward_activation_bytes),
                        );
                        self.emit_openai_phase(
                            "stage.openai_native_mtp_verify",
                            batched_token_timer,
                            token_attrs,
                        );
                    }
                    if reached_stop {
                        break;
                    }
                    continue;
                }
                if !prefer_rolling_spd_optimistic_decode
                    && (spd_guard.is_some() || draft_guard.is_some())
                {
                    let remaining = (request.max_tokens as usize).saturating_sub(decoded_tokens);
                    if remaining == 0 {
                        break;
                    }
                    let proposal_limit = remaining.min(adaptive_window);
                    let propose_timer = PhaseTimer::start();
                    let proposal = if let Some(spd) = spd_guard.as_deref_mut() {
                        propose_from_source(spd, current, proposal_limit)
                            .map_err(openai_backend_error)?
                    } else if let Some(draft) = draft_guard.as_deref_mut() {
                        propose_from_source(draft, current, proposal_limit)
                            .map_err(openai_backend_error)?
                    } else {
                        SpeculativeProposal::empty(proposal_limit)
                    };
                    let draft_propose_ms = propose_timer.elapsed_ms();
                    speculative_stats.draft_propose_ms += draft_propose_ms;
                    if !proposal.is_empty() {
                        let draft_tokens = proposal.tokens.as_slice();
                        let verify_inputs = verify_inputs_for_proposals(current, draft_tokens);
                        let verify_pos_start = prefill_token_count + decoded_tokens;
                        let message = embedded_verify_message(
                            request.wire_dtype,
                            VerifySpanMessageArgs {
                                request_id,
                                session_id,
                                prompt_token_count: request.prompt_token_ids.len(),
                                pos_start: verify_pos_start,
                                decode_step: decoded_tokens,
                                checkpoint_generation: 0,
                                tokens: &verify_inputs,
                                sampling: wire_sampling.clone(),
                                checkpoint: true,
                            },
                        )?;
                        if let Some(spd) = spd_guard.as_deref_mut() {
                            spd.mark_pending_verify_tap_positions(
                                verify_pos_start,
                                verify_inputs.len(),
                            )
                            .map_err(openai_backend_error)?;
                        }
                        let verify = self.execute_embedded_stage_message(
                            &request,
                            downstream,
                            &session_key,
                            &message,
                            &verify_inputs,
                            WireReplyKind::PredictedTokens,
                        )?;
                        speculative_stats.windows += 1;
                        speculative_stats.draft_tokens += draft_tokens.len();
                        speculative_stats.primary_verify_requests += 1;
                        speculative_stats.primary_verify_tokens += verify_inputs.len();
                        speculative_stats.primary_verify_elapsed_ms += verify.elapsed_ms;
                        speculative_stats.primary_verify_stage0_compute_ms +=
                            verify.stats.stage0_compute_ms;
                        speculative_stats.primary_verify_runtime_lock_wait_ms +=
                            verify.stats.runtime_lock_wait_ms;
                        speculative_stats.primary_verify_runtime_lock_hold_ms +=
                            verify.stats.runtime_lock_hold_ms;
                        speculative_stats.primary_verify_activation_encode_ms +=
                            verify.stats.activation_encode_ms;
                        speculative_stats.primary_verify_forward_write_ms +=
                            verify.stats.forward_write_ms;
                        speculative_stats.primary_verify_downstream_wait_ms +=
                            verify.stats.downstream_wait_ms;
                        speculative_stats.primary_verify_output_activation_bytes =
                            speculative_stats
                                .primary_verify_output_activation_bytes
                                .saturating_add(verify.stats.output_activation_bytes);
                        speculative_stats.primary_verify_forward_activation_bytes =
                            speculative_stats
                                .primary_verify_forward_activation_bytes
                                .saturating_add(verify.stats.forward_activation_bytes);
                        decode_stage0_compute_ms += verify.stats.stage0_compute_ms;
                        decode_runtime_lock_wait_ms += verify.stats.runtime_lock_wait_ms;
                        decode_runtime_lock_wait_max_ms =
                            decode_runtime_lock_wait_max_ms.max(verify.stats.runtime_lock_wait_ms);
                        decode_runtime_lock_hold_ms += verify.stats.runtime_lock_hold_ms;
                        decode_runtime_lock_hold_max_ms =
                            decode_runtime_lock_hold_max_ms.max(verify.stats.runtime_lock_hold_ms);
                        decode_runtime_lock_acquires += 1;
                        decode_forward_activation_encode_ms += verify.stats.activation_encode_ms;
                        decode_output_activation_bytes = decode_output_activation_bytes
                            .saturating_add(verify.stats.output_activation_bytes);
                        decode_forward_activation_bytes = decode_forward_activation_bytes
                            .saturating_add(verify.stats.forward_activation_bytes);
                        decode_forward_write_ms += verify.stats.forward_write_ms;
                        decode_downstream_wait_ms += verify.stats.downstream_wait_ms;
                        speculative_stats.checkpoint_ms +=
                            us_to_ms(verify.reply.stats.checkpoint_total_us);
                        let decision = classify_verify_span(
                            draft_tokens,
                            &verify.reply.predicted_tokens,
                            decoded_tokens,
                            request.max_tokens as usize,
                            |token| token_is_eog_with_runtime(&self.runtime, token),
                        )?;
                        speculative_stats.observe_verify_decision(
                            decision,
                            &mut adaptive_window,
                            request.adaptive_speculative_window,
                            max_speculative_window,
                        );
                        let mut commit_tokens =
                            verify.reply.predicted_tokens[..decision.commit_count].to_vec();
                        if decision.requires_repair() {
                            speculative_stats.recovery_restores += 1;
                            let restore = self.restore_embedded_stage_session(
                                &request,
                                downstream,
                                &session_key,
                                request_id,
                                session_id,
                                0,
                            )?;
                            speculative_stats.recovery_ms += restore.elapsed_ms;
                            speculative_stats.recovery_restore_ms += restore.elapsed_ms;
                            speculative_stats.recovery_restore_local_ms += restore.local_ms;
                            speculative_stats.recovery_restore_downstream_write_ms +=
                                restore.downstream_write_ms;
                            speculative_stats.recovery_restore_downstream_wait_ms +=
                                restore.downstream_wait_ms;
                            let repair_input_count = decision
                                .repair_input_count
                                .ok_or_else(|| OpenAiError::backend("missing repair count"))?;
                            if repair_input_count == 1 {
                                let repair_message = embedded_decode_message(
                                    request.wire_dtype,
                                    DecodeMessageArgs {
                                        request_id,
                                        session_id,
                                        prompt_token_count: request.prompt_token_ids.len(),
                                        pos_start: prefill_token_count + decoded_tokens,
                                        decode_step: decoded_tokens,
                                        current,
                                        sampling: wire_sampling.clone(),
                                    },
                                )?;
                                let repair = self.execute_embedded_stage_message(
                                    &request,
                                    downstream,
                                    &session_key,
                                    &repair_message,
                                    &[current],
                                    WireReplyKind::PredictedToken,
                                )?;
                                commit_tokens = vec![repair.reply.predicted];
                                decode_stage0_compute_ms += repair.stats.stage0_compute_ms;
                                decode_runtime_lock_wait_ms += repair.stats.runtime_lock_wait_ms;
                                decode_runtime_lock_wait_max_ms = decode_runtime_lock_wait_max_ms
                                    .max(repair.stats.runtime_lock_wait_ms);
                                decode_runtime_lock_hold_ms += repair.stats.runtime_lock_hold_ms;
                                decode_runtime_lock_hold_max_ms = decode_runtime_lock_hold_max_ms
                                    .max(repair.stats.runtime_lock_hold_ms);
                                decode_runtime_lock_acquires += 1;
                                decode_forward_activation_encode_ms +=
                                    repair.stats.activation_encode_ms;
                                decode_output_activation_bytes = decode_output_activation_bytes
                                    .saturating_add(repair.stats.output_activation_bytes);
                                decode_forward_activation_bytes = decode_forward_activation_bytes
                                    .saturating_add(repair.stats.forward_activation_bytes);
                                decode_forward_write_ms += repair.stats.forward_write_ms;
                                decode_downstream_wait_ms += repair.stats.downstream_wait_ms;
                                speculative_stats.recovery_decode_repairs += 1;
                                speculative_stats.recovery_ms += repair.elapsed_ms;
                                speculative_stats.recovery_decode_elapsed_ms += repair.elapsed_ms;
                            } else {
                                let repair_inputs = &verify_inputs[..repair_input_count];
                                let repair_message = embedded_verify_message(
                                    request.wire_dtype,
                                    VerifySpanMessageArgs {
                                        request_id,
                                        session_id,
                                        prompt_token_count: request.prompt_token_ids.len(),
                                        pos_start: prefill_token_count + decoded_tokens,
                                        decode_step: decoded_tokens,
                                        checkpoint_generation: 0,
                                        tokens: repair_inputs,
                                        sampling: wire_sampling.clone(),
                                        checkpoint: false,
                                    },
                                )?;
                                let repair = self.execute_embedded_stage_message(
                                    &request,
                                    downstream,
                                    &session_key,
                                    &repair_message,
                                    repair_inputs,
                                    WireReplyKind::PredictedTokens,
                                )?;
                                commit_tokens = repaired_commit_tokens(
                                    draft_tokens,
                                    decision.accepted_before_reject,
                                    repair_input_count,
                                    &repair.reply.predicted_tokens,
                                )?;
                                decode_stage0_compute_ms += repair.stats.stage0_compute_ms;
                                decode_runtime_lock_wait_ms += repair.stats.runtime_lock_wait_ms;
                                decode_runtime_lock_wait_max_ms = decode_runtime_lock_wait_max_ms
                                    .max(repair.stats.runtime_lock_wait_ms);
                                decode_runtime_lock_hold_ms += repair.stats.runtime_lock_hold_ms;
                                decode_runtime_lock_hold_max_ms = decode_runtime_lock_hold_max_ms
                                    .max(repair.stats.runtime_lock_hold_ms);
                                decode_runtime_lock_acquires += 1;
                                decode_forward_activation_encode_ms +=
                                    repair.stats.activation_encode_ms;
                                decode_output_activation_bytes = decode_output_activation_bytes
                                    .saturating_add(repair.stats.output_activation_bytes);
                                decode_forward_activation_bytes = decode_forward_activation_bytes
                                    .saturating_add(repair.stats.forward_activation_bytes);
                                decode_forward_write_ms += repair.stats.forward_write_ms;
                                decode_downstream_wait_ms += repair.stats.downstream_wait_ms;
                                speculative_stats.recovery_reverify_tokens += repair_inputs.len();
                                speculative_stats.recovery_ms += repair.elapsed_ms;
                                speculative_stats.recovery_reverify_elapsed_ms += repair.elapsed_ms;
                            }
                        }
                        let primary_rolling = if let Some(spd) = spd_guard.as_deref_mut() {
                            spd.observe_primary_verify_span(
                                context_tokens.len(),
                                draft_tokens,
                                &commit_tokens,
                            )
                            .map_err(openai_backend_error)?
                        } else {
                            None
                        };

                        let mut reached_stop = false;
                        let verify_commit_count = commit_tokens.len().max(1);
                        let verify_commit_count_f64 = verify_commit_count as f64;
                        for token in commit_tokens {
                            let committed_step = u32::try_from(decoded_tokens)
                                .map_err(|_| OpenAiError::backend("decode step exceeds u32"))?;
                            current = token;
                            decoded_tokens += 1;
                            context_tokens.push(current);
                            if self.telemetry.is_debug_enabled() {
                                let mut token_attrs = self.openai_attrs(request.ids);
                                token_attrs.insert(
                                    "llama_stage.decode_step".to_string(),
                                    json!(committed_step),
                                );
                                token_attrs.insert(
                                    "llama_stage.decode_token_phase".to_string(),
                                    json!(decode_token_phase(committed_step)),
                                );
                                token_attrs.insert(
                                    "llama_stage.stage0_compute_ms".to_string(),
                                    json!(verify.stats.stage0_compute_ms / verify_commit_count_f64),
                                );
                                token_attrs.insert(
                                    "llama_stage.runtime_lock_wait_ms".to_string(),
                                    json!(
                                        verify.stats.runtime_lock_wait_ms / verify_commit_count_f64
                                    ),
                                );
                                token_attrs.insert(
                                    "llama_stage.runtime_lock_hold_ms".to_string(),
                                    json!(
                                        verify.stats.runtime_lock_hold_ms / verify_commit_count_f64
                                    ),
                                );
                                token_attrs.insert(
                                    "llama_stage.output_activation_bytes".to_string(),
                                    json!(
                                        verify.stats.output_activation_bytes / verify_commit_count
                                    ),
                                );
                                token_attrs.insert(
                                    "llama_stage.forward_activation_bytes".to_string(),
                                    json!(
                                        verify.stats.forward_activation_bytes / verify_commit_count
                                    ),
                                );
                                token_attrs.insert(
                                    "llama_stage.activation_encode_ms".to_string(),
                                    json!(
                                        verify.stats.activation_encode_ms / verify_commit_count_f64
                                    ),
                                );
                                token_attrs.insert(
                                    "llama_stage.forward_write_ms".to_string(),
                                    json!(verify.stats.forward_write_ms / verify_commit_count_f64),
                                );
                                token_attrs.insert(
                                    "llama_stage.downstream_wait_ms".to_string(),
                                    json!(
                                        verify.stats.downstream_wait_ms / verify_commit_count_f64
                                    ),
                                );
                                token_attrs.insert(
                                    "llama_stage.predicted_token".to_string(),
                                    json!(current),
                                );
                                token_attrs.insert(
                                    "llama_stage.message_kind".to_string(),
                                    json!("VerifySpan"),
                                );
                                token_attrs.insert(
                                    "llama_stage.spec.proposal_source".to_string(),
                                    json!(proposal.source),
                                );
                                self.telemetry
                                    .emit_debug("stage.openai_decode_token", token_attrs);
                            }
                            if on_token(current)? == TokenControl::Stop {
                                reached_stop = true;
                            }
                            if reached_stop || decoded_tokens >= request.max_tokens as usize {
                                break;
                            }
                        }
                        speculative_stats.adaptive_window_final = adaptive_window;
                        let should_reset_spd = spd_guard.as_ref().is_some_and(|spd| {
                            spd.should_reset_after_verify(decision, reached_stop)
                        });
                        let should_reset_draft = draft_guard.as_ref().is_some_and(|draft| {
                            draft.should_reset_after_verify(decision, reached_stop)
                        });
                        if let Some(spd) = spd_guard.as_deref_mut() {
                            let draft_reset_timer = PhaseTimer::start();
                            if should_reset_spd {
                                spd.reset_to_verified_context(&context_tokens)
                                    .map_err(openai_backend_error)?;
                            } else {
                                spd.advance_to_accepted_context(&context_tokens)
                                    .map_err(openai_backend_error)?;
                            }
                            speculative_stats.draft_reset_ms += draft_reset_timer.elapsed_ms();
                        }
                        if should_reset_draft {
                            let draft_reset_timer = PhaseTimer::start();
                            if let Some(draft) = draft_guard.as_deref_mut() {
                                draft
                                    .reset_to_context(&context_tokens)
                                    .map_err(openai_backend_error)?;
                            }
                            speculative_stats.draft_reset_ms += draft_reset_timer.elapsed_ms();
                        }
                        let mut token_attrs = self.openai_attrs(request.ids);
                        token_attrs
                            .insert("llama_stage.decode_step".to_string(), json!(decode_step));
                        token_attrs
                            .insert("llama_stage.message_kind".to_string(), json!("VerifySpan"));
                        token_attrs.insert(
                            "llama_stage.spec.windows".to_string(),
                            json!(speculative_stats.windows),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.proposed".to_string(),
                            json!(draft_tokens.len()),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.accepted".to_string(),
                            json!(decision.accepted_before_reject),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.rejected".to_string(),
                            json!(decision.rejected()),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.draft_propose_ms".to_string(),
                            json!(draft_propose_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.proposal_source".to_string(),
                            json!(proposal.source),
                        );
                        if let Some(spd) = spd_guard.as_deref() {
                            spd.insert_last_proposal_stats_attrs(&mut token_attrs);
                        }
                        if let Some(rolling) = primary_rolling.as_ref() {
                            super::spd::SpdReplayProposalSource::insert_rolling_telemetry_attrs(
                                &mut token_attrs,
                                rolling,
                            );
                        }
                        token_attrs.insert(
                            "llama_stage.spec.proposal_limit".to_string(),
                            json!(proposal.requested_limit),
                        );
                        token_attrs.insert(
                            "llama_stage.stage0_compute_ms".to_string(),
                            json!(verify.stats.stage0_compute_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_wait_ms".to_string(),
                            json!(verify.stats.runtime_lock_wait_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_hold_ms".to_string(),
                            json!(verify.stats.runtime_lock_hold_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.activation_encode_ms".to_string(),
                            json!(verify.stats.activation_encode_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_write_ms".to_string(),
                            json!(verify.stats.forward_write_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.downstream_wait_ms".to_string(),
                            json!(verify.stats.downstream_wait_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.output_activation_bytes".to_string(),
                            json!(verify.stats.output_activation_bytes),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_activation_bytes".to_string(),
                            json!(verify.stats.forward_activation_bytes),
                        );
                        self.emit_openai_phase(
                            "stage.openai_decode_verify_window",
                            token_timer,
                            token_attrs,
                        );
                        if reached_stop {
                            break;
                        }
                        continue;
                    }
                }
                let uses_context_sideband = decode_uses_context_sideband(
                    &context_tokens,
                    current,
                    skippy_protocol::binary::MAX_STAGE_SIDEBAND_VALUES,
                );
                let records_replay_checkpoint = uses_context_sideband && context_tokens.len() > 1;
                let records_full_prompt_checkpoint = decode_step == 0
                    && uses_context_sideband
                    && context_tokens.len() == request.prompt_token_ids.len();
                let decode_step_index = usize::try_from(decode_step)
                    .map_err(|_| OpenAiError::backend("decode step exceeds usize"))?;
                let message = if uses_context_sideband {
                    decode_message.update_with_tokens(
                        decode_step_index,
                        current,
                        &context_tokens,
                    )?
                } else {
                    decode_message.update(decode_step_index, current)?
                };
                let stage0_timer = PhaseTimer::start();
                let batch_outcome = self
                    .decode_frame_batcher
                    .decode(
                        &session_key,
                        current,
                        request.sampling.enabled.then_some(request.sampling),
                        None,
                    )
                    .map_err(openai_backend_error)?;
                let token_runtime_lock_wait_ms = batch_outcome.runtime_lock_wait_ms;
                let token_runtime_lock_hold_ms = batch_outcome.runtime_lock_hold_ms;
                decode_runtime_lock_wait_ms += token_runtime_lock_wait_ms;
                decode_runtime_lock_wait_max_ms =
                    decode_runtime_lock_wait_max_ms.max(token_runtime_lock_wait_ms);
                decode_runtime_lock_hold_ms += token_runtime_lock_hold_ms;
                decode_runtime_lock_hold_max_ms =
                    decode_runtime_lock_hold_max_ms.max(token_runtime_lock_hold_ms);
                decode_runtime_lock_acquires += 1;
                decode_batch_size_max = decode_batch_size_max.max(batch_outcome.batch_size);
                decode_batch_wait_ms += batch_outcome.batch_wait_ms;
                let output = batch_outcome.output;
                let mut canonical_materialized_token_count = context_tokens.len();
                let stage0_compute_ms = stage0_timer.elapsed_ms();
                decode_stage0_compute_ms += stage0_compute_ms;
                self.record_spd_stage0_boundary_tap(&request, message, &output);
                let mut forwarded = forwarded_stage_message_timed(
                    request.config,
                    message,
                    &output,
                    request.wire_dtype,
                    request.activation_width,
                )
                .map_err(openai_backend_error)?;
                self.mark_spd_tap_return(&request, &mut forwarded.message);
                decode_forward_activation_encode_ms += forwarded.activation_encode_ms;
                decode_output_activation_bytes =
                    decode_output_activation_bytes.saturating_add(output.payload.len());
                decode_forward_activation_bytes = decode_forward_activation_bytes
                    .saturating_add(forwarded.message.activation.len());
                let write_timer = PhaseTimer::start();
                write_stage_message_conditioned(
                    &mut *downstream,
                    &forwarded.message,
                    request.wire_dtype,
                    request.downstream_wire_condition,
                )
                .map_err(openai_io_error)?;
                let forward_write_ms = write_timer.elapsed_ms();
                decode_forward_write_ms += forward_write_ms;
                let wait_timer = PhaseTimer::start();
                let mut optimistic_decode = None;
                let mut rolling_optimistic_decodes =
                    std::collections::VecDeque::<SpdOptimisticDecode>::new();
                let can_start_optimistic_spd = request.spd_optimistic_decode
                    && spd_guard.is_some()
                    && request.sampling.temperature <= 0.0
                    && context_tokens.len().checked_add(1).is_some_and(|position| {
                        position
                            < request
                                .prompt_token_ids
                                .len()
                                .saturating_add(request.max_tokens as usize)
                    });
                let rolling_executor_can_launch =
                    can_start_optimistic_spd && use_spd_rolling_executor;
                let prediction_return =
                    if spd_guard.is_some() {
                        if rolling_executor_can_launch {
                            let mut pre_target_probe = None;
                            let immediate_probe_position = context_tokens.len();
                            let rolling_reply = self
                        .recv_spd_aware_prediction_return_with_tap_action(
                            &request,
                            WireReplyKind::PredictedToken,
                            None,
                            |backend, trigger_hf_index| {
                                let Some(executor) = spd_rolling_executor.as_mut() else {
                                    return Ok(());
                                };
                                let Some(spd) = spd_guard.as_deref_mut() else {
                                    return Ok(());
                                };
                                let decode_step = decoded_tokens
                                    .checked_add(1)
                                    .and_then(|step| step.checked_add(executor.in_flight_len()))
                                    .context("SPD rolling executor decode step overflow")?;
                                let Some((mut decode, probe)) = start_spd_rolling_executor_decode(
                                    backend,
                                    SpdRollingStartArgs {
                                        request: &request,
                                        downstream,
                                        session_key: &session_key,
                                        shadow_session: spd_rolling_shadow_session.as_mut(),
                                        source_materialized_token_count:
                                            canonical_materialized_token_count,
                                        spd,
                                        executor,
                                        decode_step,
                                        phase: super::spd::SpdInlineProbePhase::PreTargetReply,
                                        trigger_hf_index,
                                    },
                                )
                                .map_err(|error| anyhow::anyhow!(error.to_string()))?
                                else {
                                    return Ok(());
                                };
                                let emit_probe = pre_target_probe.is_none()
                                    && decode.position == immediate_probe_position;
                                if emit_probe {
                                    pre_target_probe = Some(probe.clone());
                                    decode.inline_probe_emitted = true;
                                }
                                if optimistic_decode.is_none() {
                                    optimistic_decode = Some(decode);
                                } else {
                                    rolling_optimistic_decodes.push_back(decode);
                                }
                                Ok(())
                            },
                        )
                        .map_err(openai_backend_error)?;
                            Ok(SpdPredictionReturn {
                                reply: rolling_reply,
                                pre_target_probe,
                            })
                        } else if can_start_optimistic_spd && !use_spd_rolling_executor {
                            self.recv_spd_aware_prediction_return_with_probe_action(
                        &request,
                        WireReplyKind::PredictedToken,
                        spd_guard.as_deref_mut(),
                        current,
                        super::spd::SpdInlineProbePhase::PreTargetReply,
                        Some(
                            |backend: &StageOpenAiBackend, probe: &super::spd::SpdInlineProbe| {
                                let optimistic_pos_start = context_tokens.len();
                                optimistic_decode = backend
                                    .start_spd_optimistic_decode_for_probe(
                                        &request,
                                        SpdOptimisticDecodeStart {
                                            downstream,
                                            session_key: &session_key,
                                            execution_session: None,
                                            pos_start: optimistic_pos_start,
                                            decode_step: decoded_tokens + 1,
                                            chain_depth: 0,
                                            chain_depth_limit: spd_optimistic_chain_depth_limit,
                                            probe,
                                            checkpoint: true,
                                        },
                                    )
                                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                                Ok(())
                            },
                        ),
                    )
                        } else {
                            self.recv_spd_aware_prediction_return_with_probe(
                                &request,
                                WireReplyKind::PredictedToken,
                                spd_guard.as_deref_mut(),
                                current,
                            )
                        }
                        .map_err(openai_backend_error)?
                    } else {
                        SpdPredictionReturn {
                            reply: super::embedded_execution::receive_embedded_stage_reply(
                                downstream,
                                request.prediction_return.as_ref(),
                                WireReplyKind::PredictedToken,
                            )?,
                            pre_target_probe: None,
                        }
                    };
                let reply = prediction_return.reply;
                let downstream_wait_ms = wait_timer.elapsed_ms();
                decode_downstream_wait_ms += downstream_wait_ms;
                let mut optimistic_commit = None;
                let mut chained_optimistic_commits = Vec::new();
                let mut optimistic_commit_probes = Vec::new();
                let mut spd_advanced_for_current = false;
                // Rolling SPD row positions are token indices. Before pushing
                // reply.predicted, the target token will occupy context_tokens.len().
                let rolling_target_position = context_tokens.len();
                let mut rolling_executor_commits = observe_spd_rolling_executor_target(
                    &mut speculative_stats,
                    spd_rolling_executor.as_mut(),
                    rolling_target_position,
                    reply.predicted,
                )?;
                if let Some(probe) = prediction_return.pre_target_probe {
                    let rolling = if let Some(spd) = spd_guard.as_deref_mut() {
                        Some(
                            spd.observe_rolling_probe(
                                rolling_target_position,
                                reply.predicted,
                                probe.proposed,
                            )
                            .map_err(openai_backend_error)?,
                        )
                    } else {
                        None
                    };
                    if let Some(proposed) = probe.proposed {
                        speculative_stats
                            .observe_inline_verified_probe(proposed == reply.predicted);
                    }
                    if let Some(mut optimistic) = optimistic_decode.take() {
                        let optimistic_wait_timer = PhaseTimer::start();
                        let accepted = if use_spd_rolling_executor {
                            spd_rolling_executor_acceptance(
                                &rolling_executor_commits,
                                rolling_target_position,
                                reply.predicted,
                                optimistic.proposed == reply.predicted,
                            )?
                        } else {
                            optimistic.proposed == reply.predicted
                        };
                        let mut context_with_reply = context_tokens.clone();
                        context_with_reply.push(reply.predicted);
                        if use_spd_rolling_executor
                            && !optimistic.inline_probe_emitted
                            && optimistic.position == rolling_target_position
                        {
                            let rolling = if let Some(spd) = spd_guard.as_deref_mut() {
                                Some(
                                    spd.observe_rolling_probe(
                                        rolling_target_position,
                                        reply.predicted,
                                        optimistic.inline_probe.proposed,
                                    )
                                    .map_err(openai_backend_error)?,
                                )
                            } else {
                                None
                            };
                            if let Some(proposed) = optimistic.inline_probe.proposed {
                                speculative_stats
                                    .observe_inline_verified_probe(proposed == reply.predicted);
                            }
                            let probe = optimistic.inline_probe.clone();
                            speculative_stats.draft_propose_ms += self.emit_spd_inline_probe(
                                &request,
                                decode_step,
                                current,
                                reply.predicted,
                                probe,
                                rolling.as_ref(),
                            );
                            optimistic.inline_probe_emitted = true;
                        }
                        let optimistic_return = if accepted && optimistic.requested_spd_taps {
                            if let Some(spd) = spd_guard.as_deref_mut() {
                                let reset_timer = PhaseTimer::start();
                                spd.advance_to_accepted_context(&context_with_reply)
                                    .map_err(openai_backend_error)?;
                                speculative_stats.draft_reset_ms += reset_timer.elapsed_ms();
                                spd_advanced_for_current = true;
                                if use_spd_rolling_executor {
                                    let mut optimistic_commit_probe = None;
                                    let immediate_probe_position = context_with_reply.len();
                                    let rolling_reply = self
                                        .recv_spd_aware_prediction_return_with_tap_action(
                                            &request,
                                            WireReplyKind::PredictedTokens,
                                            Some(optimistic.origin),
                                            |backend, trigger_hf_index| {
                                                let Some(executor) =
                                                    spd_rolling_executor.as_mut()
                                                else {
                                                    return Ok(());
                                                };
                                                let decode_step = decoded_tokens
                                                    .checked_add(2)
                                                    .and_then(|step| {
                                                        step.checked_add(executor.in_flight_len())
                                                    })
                                                    .context(
                                                        "SPD rolling executor decode step overflow",
                                                    )?;
                                                let Some((mut decode, probe)) =
                                                    start_spd_rolling_executor_decode(
                                                        backend,
                                                        SpdRollingStartArgs {
                                                            request: &request,
                                                            downstream,
                                                            session_key: &session_key,
                                                            shadow_session:
                                                                spd_rolling_shadow_session.as_mut(),
                                                            source_materialized_token_count:
                                                                canonical_materialized_token_count,
                                                            spd,
                                                            executor,
                                                            decode_step,
                                                            phase: super::spd::SpdInlineProbePhase::OptimisticCommit,
                                                            trigger_hf_index,
                                                        },
                                                    )
                                                    .map_err(|error| {
                                                        anyhow::anyhow!(error.to_string())
                                                    })?
                                                else {
                                                    return Ok(());
                                                };
                                                let emit_probe =
                                                    optimistic_commit_probe.is_none()
                                                        && decode.position
                                                            == immediate_probe_position;
                                                if emit_probe {
                                                    optimistic_commit_probe = Some(probe.clone());
                                                    decode.inline_probe_emitted = true;
                                                }
                                                rolling_optimistic_decodes.push_back(decode);
                                                Ok(())
                                            },
                                        )
                                        .map_err(openai_backend_error)?;
                                    SpdPredictionReturn {
                                        reply: rolling_reply,
                                        pre_target_probe: optimistic_commit_probe,
                                    }
                                } else {
                                    self.recv_spd_aware_prediction_return_with_wait_action(
                                        &request,
                                        SpdPredictionReturnWait {
                                            expected: WireReplyKind::PredictedTokens,
                                            expected_origin: Some(optimistic.origin),
                                            spd_source: Some(spd),
                                            current: reply.predicted,
                                            probe_phase: super::spd::SpdInlineProbePhase::OptimisticCommit,
                                        },
                                        Some(
                                            |backend: &StageOpenAiBackend,
                                             probe: &super::spd::SpdInlineProbe| {
                                                let chain_pos_start = context_with_reply.len();
                                                if let Some(decode) = backend
                                                    .start_spd_optimistic_decode_for_probe(
                                                        &request,
                                                        SpdOptimisticDecodeStart {
                                                            downstream,
                                                            session_key: &session_key,
                                                            execution_session: None,
                                                            pos_start: chain_pos_start,
                                                            decode_step: decoded_tokens + 2,
                                                            chain_depth: 1,
                                                            chain_depth_limit:
                                                                spd_optimistic_chain_depth_limit,
                                                            probe,
                                                            checkpoint: true,
                                                        },
                                                    )
                                                    .map_err(|error| {
                                                        anyhow::anyhow!(error.to_string())
                                                    })?
                                                {
                                                    rolling_optimistic_decodes.push_back(decode);
                                                }
                                                Ok(())
                                            },
                                        ),
                                    )
                                    .map_err(openai_backend_error)?
                                }
                            } else {
                                self.recv_spd_aware_prediction_return_for_origin(
                                    &request,
                                    WireReplyKind::PredictedTokens,
                                    optimistic.origin,
                                )
                                .map_err(openai_backend_error)?
                            }
                        } else {
                            self.recv_spd_aware_prediction_return_for_origin(
                                &request,
                                WireReplyKind::PredictedTokens,
                                optimistic.origin,
                            )
                            .map_err(openai_backend_error)?
                        };
                        let optimistic_wait_ms = optimistic_wait_timer.elapsed_ms();
                        let optimistic_reply = optimistic_return.reply;
                        let optimistic_next = optimistic_reply
                            .predicted_tokens
                            .first()
                            .copied()
                            .ok_or_else(|| {
                                OpenAiError::backend(
                                    "optimistic SPD verify returned no predicted token",
                                )
                            })?;
                        let optimistic_target_commits = observe_spd_rolling_executor_target(
                            &mut speculative_stats,
                            spd_rolling_executor.as_mut(),
                            context_with_reply.len(),
                            optimistic_next,
                        )?;
                        rolling_executor_commits.extend(optimistic_target_commits);
                        if let Some(probe) = optimistic_return.pre_target_probe {
                            let rolling_target_position = context_with_reply.len();
                            let rolling = if let Some(spd) = spd_guard.as_deref_mut() {
                                Some(
                                    spd.observe_rolling_probe(
                                        rolling_target_position,
                                        optimistic_next,
                                        probe.proposed,
                                    )
                                    .map_err(openai_backend_error)?,
                                )
                            } else {
                                None
                            };
                            optimistic_commit_probes.push(PendingSpdInlineProbe {
                                decode_step: decode_step.saturating_add(1),
                                current: reply.predicted,
                                target: optimistic_next,
                                probe,
                                rolling,
                            });
                        }
                        let mut optimistic_reply_stats = optimistic.execution.reply_stats;
                        optimistic_reply_stats.merge(optimistic_reply.stats);
                        let optimistic_checkpoint_ms =
                            us_to_ms(optimistic_reply_stats.checkpoint_total_us);
                        speculative_stats.optimistic_checkpoint_ms += optimistic_checkpoint_ms;
                        let optimistic_elapsed_ms = optimistic.timer.elapsed_ms();
                        let optimistic_start_elapsed_ms = optimistic.execution.elapsed_ms;
                        let mut optimistic_stats = optimistic.execution.stats;
                        optimistic_stats.downstream_wait_ms = optimistic_wait_ms;
                        speculative_stats.optimistic_decode_requests += 1;
                        speculative_stats.optimistic_decode_elapsed_ms += optimistic_elapsed_ms;
                        speculative_stats.optimistic_decode_wait_ms += optimistic_wait_ms;
                        decode_stage0_compute_ms += optimistic_stats.stage0_compute_ms;
                        decode_runtime_lock_wait_ms += optimistic_stats.runtime_lock_wait_ms;
                        decode_runtime_lock_wait_max_ms = decode_runtime_lock_wait_max_ms
                            .max(optimistic_stats.runtime_lock_wait_ms);
                        decode_runtime_lock_hold_ms += optimistic_stats.runtime_lock_hold_ms;
                        decode_runtime_lock_hold_max_ms = decode_runtime_lock_hold_max_ms
                            .max(optimistic_stats.runtime_lock_hold_ms);
                        decode_runtime_lock_acquires += 1;
                        decode_forward_activation_encode_ms +=
                            optimistic_stats.activation_encode_ms;
                        decode_output_activation_bytes = decode_output_activation_bytes
                            .saturating_add(optimistic_stats.output_activation_bytes);
                        decode_forward_activation_bytes = decode_forward_activation_bytes
                            .saturating_add(optimistic_stats.forward_activation_bytes);
                        decode_forward_write_ms += optimistic_stats.forward_write_ms;
                        decode_downstream_wait_ms += optimistic_wait_ms;
                        if accepted {
                            if use_spd_rolling_executor
                                && let Some(promote) = promote_spd_rolling_shadow_session(
                                    self,
                                    &request,
                                    downstream,
                                    &mut spd_rolling_shadow_session,
                                    context_with_reply.len(),
                                )?
                            {
                                canonical_materialized_token_count = context_with_reply.len();
                                speculative_stats.recovery_ms += promote.elapsed_ms;
                            }
                            speculative_stats.optimistic_decode_accepted += 1;
                            optimistic_commit = Some((optimistic_next, optimistic_stats));
                        } else {
                            speculative_stats.optimistic_decode_rejected += 1;
                            if use_spd_rolling_executor {
                                drain_spd_rolling_executor_replies(
                                    self,
                                    &request,
                                    &mut rolling_optimistic_decodes,
                                )?;
                                merge_spd_rolling_executor_stats(
                                    &mut speculative_stats,
                                    spd_rolling_executor.as_ref(),
                                );
                                if let Some(drop) = reset_spd_rolling_shadow_session(
                                    self,
                                    &request,
                                    downstream,
                                    &mut spd_rolling_shadow_session,
                                )? {
                                    speculative_stats.recovery_ms += drop.elapsed_ms;
                                }
                            } else {
                                let restore = self.restore_embedded_stage_session(
                                    &request,
                                    downstream,
                                    &session_key,
                                    request_id,
                                    session_id,
                                    optimistic.origin.checkpoint_generation,
                                )?;
                                speculative_stats.recovery_restores += 1;
                                speculative_stats.recovery_ms += restore.elapsed_ms;
                                speculative_stats.recovery_restore_ms += restore.elapsed_ms;
                                speculative_stats.recovery_restore_local_ms += restore.local_ms;
                                speculative_stats.recovery_restore_downstream_write_ms +=
                                    restore.downstream_write_ms;
                                speculative_stats.recovery_restore_downstream_wait_ms +=
                                    restore.downstream_wait_ms;
                                speculative_stats.optimistic_restore_ms += restore.elapsed_ms;
                            }
                            if optimistic.requested_spd_taps
                                && let Some(spd) = spd_guard.as_deref_mut()
                            {
                                let reset_timer = PhaseTimer::start();
                                spd.reset_to_verified_context(&context_with_reply)
                                    .map_err(openai_backend_error)?;
                                speculative_stats.draft_reset_ms += reset_timer.elapsed_ms();
                            }
                        }
                        let mut attrs = self.openai_attrs(request.ids);
                        attrs.insert("llama_stage.decode_step".to_string(), json!(decode_step));
                        attrs.insert(
                            "llama_stage.spd_optimistic_proposed_token".to_string(),
                            json!(optimistic.proposed),
                        );
                        attrs.insert(
                            "llama_stage.spd_optimistic_proposed_logit".to_string(),
                            json!(optimistic.proposed_logit),
                        );
                        attrs.insert(
                            "llama_stage.spd_optimistic_logit_margin".to_string(),
                            json!(optimistic.proposed_logit_margin),
                        );
                        attrs.insert(
                            "llama_stage.spd_optimistic_tap_return".to_string(),
                            json!(optimistic.requested_spd_taps),
                        );
                        attrs.insert(
                            "llama_stage.spd_optimistic_target_token".to_string(),
                            json!(reply.predicted),
                        );
                        attrs.insert(
                            "llama_stage.spd_optimistic_accepted".to_string(),
                            json!(accepted),
                        );
                        attrs.insert(
                            "llama_stage.spd_optimistic_next_token".to_string(),
                            json!(optimistic_next),
                        );
                        attrs.insert(
                            "llama_stage.spd_optimistic_checkpoint_ms".to_string(),
                            json!(optimistic_checkpoint_ms),
                        );
                        attrs.insert(
                            "llama_stage.spd_optimistic_decode_elapsed_ms".to_string(),
                            json!(optimistic_elapsed_ms),
                        );
                        attrs.insert(
                            "llama_stage.spd_optimistic_start_elapsed_ms".to_string(),
                            json!(optimistic_start_elapsed_ms),
                        );
                        attrs.insert(
                            "llama_stage.spd_optimistic_decode_wait_ms".to_string(),
                            json!(optimistic_wait_ms),
                        );
                        attrs.insert(
                            "llama_stage.stage0_compute_ms".to_string(),
                            json!(optimistic_stats.stage0_compute_ms),
                        );
                        attrs.insert(
                            "llama_stage.forward_write_ms".to_string(),
                            json!(optimistic_stats.forward_write_ms),
                        );
                        attrs.insert(
                            "llama_stage.downstream_wait_ms".to_string(),
                            json!(optimistic_wait_ms),
                        );
                        self.telemetry
                            .emit_debug("stage.openai_spd_optimistic_decode", attrs);
                        if accepted {
                            let mut rolling_context = context_with_reply;
                            let mut expected_target = optimistic_next;
                            while let Some(mut chained) = rolling_optimistic_decodes.pop_front() {
                                let chain_decode_step = chained
                                    .position
                                    .checked_sub(request.prompt_token_ids.len())
                                    .and_then(|step| u32::try_from(step).ok())
                                    .ok_or_else(|| {
                                        OpenAiError::backend("optimistic decode step exceeds u32")
                                    })?;
                                let chain_target = expected_target;
                                let chain_accepted = if use_spd_rolling_executor {
                                    spd_rolling_executor_acceptance(
                                        &rolling_executor_commits,
                                        chained.position,
                                        chain_target,
                                        chained.proposed == chain_target,
                                    )?
                                } else {
                                    chained.proposed == chain_target
                                };
                                let mut context_with_expected = rolling_context.clone();
                                context_with_expected.push(chain_target);
                                if use_spd_rolling_executor
                                    && !chained.inline_probe_emitted
                                    && chained.position == rolling_context.len()
                                {
                                    let rolling_target_position = rolling_context.len();
                                    let rolling = if let Some(spd) = spd_guard.as_deref_mut() {
                                        Some(
                                            spd.observe_rolling_probe(
                                                rolling_target_position,
                                                chain_target,
                                                chained.inline_probe.proposed,
                                            )
                                            .map_err(openai_backend_error)?,
                                        )
                                    } else {
                                        None
                                    };
                                    if let Some(proposed) = chained.inline_probe.proposed {
                                        speculative_stats.observe_inline_verified_probe(
                                            proposed == chain_target,
                                        );
                                    }
                                    let probe_current =
                                        rolling_context.last().copied().unwrap_or(chain_target);
                                    let probe = chained.inline_probe.clone();
                                    speculative_stats.draft_propose_ms += self
                                        .emit_spd_inline_probe(
                                            &request,
                                            chain_decode_step,
                                            probe_current,
                                            chain_target,
                                            probe,
                                            rolling.as_ref(),
                                        );
                                    chained.inline_probe_emitted = true;
                                }
                                let chain_wait_timer = PhaseTimer::start();
                                let chain_return = if chain_accepted && chained.requested_spd_taps {
                                    if let Some(spd) = spd_guard.as_deref_mut() {
                                        let reset_timer = PhaseTimer::start();
                                        spd.advance_to_accepted_context(&context_with_expected)
                                            .map_err(openai_backend_error)?;
                                        speculative_stats.draft_reset_ms +=
                                            reset_timer.elapsed_ms();
                                        let next_chain_depth = chained.chain_depth + 1;
                                        if use_spd_rolling_executor {
                                            let mut optimistic_commit_probe = None;
                                            let immediate_probe_position =
                                                context_with_expected.len();
                                            let rolling_reply = self
                                                .recv_spd_aware_prediction_return_with_tap_action(
                                                    &request,
                                                    WireReplyKind::PredictedTokens,
                                                    Some(chained.origin),
                                                    |backend, trigger_hf_index| {
                                                        let Some(executor) =
                                                            spd_rolling_executor.as_mut()
                                                        else {
                                                            return Ok(());
                                                        };
                                                        let decode_step = decoded_tokens
                                                            .checked_add(2)
                                                            .and_then(|step| {
                                                                step.checked_add(
                                                                    executor.in_flight_len(),
                                                                )
                                                            })
                                                            .context(
                                                                "SPD rolling executor decode step overflow",
                                                            )?;
                                                        let Some((mut decode, probe)) =
                                                            start_spd_rolling_executor_decode(
                                                                backend,
                                                                SpdRollingStartArgs {
                                                                    request: &request,
                                                                    downstream,
                                                                    session_key: &session_key,
                                                                    shadow_session:
                                                                        spd_rolling_shadow_session
                                                                            .as_mut(),
                                                                    source_materialized_token_count:
                                                                        canonical_materialized_token_count,
                                                                    spd,
                                                                    executor,
                                                                    decode_step,
                                                                    phase: super::spd::SpdInlineProbePhase::OptimisticCommit,
                                                                    trigger_hf_index,
                                                                },
                                                            )
                                                            .map_err(|error| {
                                                                anyhow::anyhow!(error.to_string())
                                                            })?
                                                        else {
                                                            return Ok(());
                                                        };
                                                        let emit_probe =
                                                            optimistic_commit_probe.is_none()
                                                                && decode.position
                                                                    == immediate_probe_position;
                                                        if emit_probe {
                                                            optimistic_commit_probe =
                                                                Some(probe.clone());
                                                            decode.inline_probe_emitted = true;
                                                        }
                                                        rolling_optimistic_decodes
                                                            .push_back(decode);
                                                        Ok(())
                                                    },
                                                )
                                                .map_err(openai_backend_error)?;
                                            SpdPredictionReturn {
                                                reply: rolling_reply,
                                                pre_target_probe: optimistic_commit_probe,
                                            }
                                        } else {
                                            self.recv_spd_aware_prediction_return_with_wait_action(
                                                &request,
                                                SpdPredictionReturnWait {
                                                    expected: WireReplyKind::PredictedTokens,
                                                    expected_origin: Some(chained.origin),
                                                    spd_source: Some(spd),
                                                    current: chain_target,
                                                    probe_phase: super::spd::SpdInlineProbePhase::OptimisticCommit,
                                                },
                                                Some(
                                                    |backend: &StageOpenAiBackend,
                                                     probe: &super::spd::SpdInlineProbe| {
                                                        let next_pos_start =
                                                            context_with_expected.len();
                                                        if let Some(decode) = backend
                                                            .start_spd_optimistic_decode_for_probe(
                                                                &request,
                                                                SpdOptimisticDecodeStart {
                                                                    downstream,
                                                                    session_key: &session_key,
                                                                    execution_session: None,
                                                                    pos_start: next_pos_start,
                                                                    decode_step: decoded_tokens
                                                                        + 2
                                                                        + chained.chain_depth,
                                                                    chain_depth: next_chain_depth,
                                                                    chain_depth_limit:
                                                                        spd_optimistic_chain_depth_limit,
                                                                    probe,
                                                                    checkpoint: true,
                                                                },
                                                            )
                                                            .map_err(|error| {
                                                                anyhow::anyhow!(error.to_string())
                                                            })?
                                                        {
                                                            rolling_optimistic_decodes
                                                                .push_back(decode);
                                                        }
                                                        Ok(())
                                                    },
                                                ),
                                            )
                                            .map_err(openai_backend_error)?
                                        }
                                    } else {
                                        self.recv_spd_aware_prediction_return_for_origin(
                                            &request,
                                            WireReplyKind::PredictedTokens,
                                            chained.origin,
                                        )
                                        .map_err(openai_backend_error)?
                                    }
                                } else {
                                    self.recv_spd_aware_prediction_return_for_origin(
                                        &request,
                                        WireReplyKind::PredictedTokens,
                                        chained.origin,
                                    )
                                    .map_err(openai_backend_error)?
                                };
                                let chain_wait_ms = chain_wait_timer.elapsed_ms();
                                let chain_reply = chain_return.reply;
                                let chain_next = chain_reply
                                    .predicted_tokens
                                    .first()
                                    .copied()
                                    .ok_or_else(|| {
                                        OpenAiError::backend(
                                            "chained optimistic SPD verify returned no predicted token",
                                        )
                                    })?;
                                let chain_target_commits = observe_spd_rolling_executor_target(
                                    &mut speculative_stats,
                                    spd_rolling_executor.as_mut(),
                                    context_with_expected.len(),
                                    chain_next,
                                )?;
                                rolling_executor_commits.extend(chain_target_commits);
                                if let Some(probe) = chain_return.pre_target_probe {
                                    let rolling_target_position = context_with_expected.len();
                                    let rolling = if let Some(spd) = spd_guard.as_deref_mut() {
                                        Some(
                                            spd.observe_rolling_probe(
                                                rolling_target_position,
                                                chain_next,
                                                probe.proposed,
                                            )
                                            .map_err(openai_backend_error)?,
                                        )
                                    } else {
                                        None
                                    };
                                    optimistic_commit_probes.push(PendingSpdInlineProbe {
                                        decode_step: chain_decode_step,
                                        current: chain_target,
                                        target: chain_next,
                                        probe,
                                        rolling,
                                    });
                                }
                                let mut chain_reply_stats = chained.execution.reply_stats;
                                chain_reply_stats.merge(chain_reply.stats);
                                let chain_checkpoint_ms =
                                    us_to_ms(chain_reply_stats.checkpoint_total_us);
                                speculative_stats.optimistic_checkpoint_ms += chain_checkpoint_ms;
                                let chain_elapsed_ms = chained.timer.elapsed_ms();
                                let chain_start_elapsed_ms = chained.execution.elapsed_ms;
                                let mut chain_stats = chained.execution.stats;
                                chain_stats.downstream_wait_ms = chain_wait_ms;
                                speculative_stats.optimistic_decode_requests += 1;
                                speculative_stats.chained_optimistic_decode_requests += 1;
                                speculative_stats.optimistic_decode_elapsed_ms += chain_elapsed_ms;
                                speculative_stats.optimistic_decode_wait_ms += chain_wait_ms;
                                decode_stage0_compute_ms += chain_stats.stage0_compute_ms;
                                decode_runtime_lock_wait_ms += chain_stats.runtime_lock_wait_ms;
                                decode_runtime_lock_wait_max_ms = decode_runtime_lock_wait_max_ms
                                    .max(chain_stats.runtime_lock_wait_ms);
                                decode_runtime_lock_hold_ms += chain_stats.runtime_lock_hold_ms;
                                decode_runtime_lock_hold_max_ms = decode_runtime_lock_hold_max_ms
                                    .max(chain_stats.runtime_lock_hold_ms);
                                decode_runtime_lock_acquires += 1;
                                decode_forward_activation_encode_ms +=
                                    chain_stats.activation_encode_ms;
                                decode_output_activation_bytes = decode_output_activation_bytes
                                    .saturating_add(chain_stats.output_activation_bytes);
                                decode_forward_activation_bytes = decode_forward_activation_bytes
                                    .saturating_add(chain_stats.forward_activation_bytes);
                                decode_forward_write_ms += chain_stats.forward_write_ms;
                                decode_downstream_wait_ms += chain_wait_ms;
                                if chain_accepted {
                                    if use_spd_rolling_executor
                                        && let Some(promote) = promote_spd_rolling_shadow_session(
                                            self,
                                            &request,
                                            downstream,
                                            &mut spd_rolling_shadow_session,
                                            context_with_expected.len(),
                                        )?
                                    {
                                        canonical_materialized_token_count =
                                            context_with_expected.len();
                                        speculative_stats.recovery_ms += promote.elapsed_ms;
                                    }
                                    speculative_stats.optimistic_decode_accepted += 1;
                                    speculative_stats.chained_optimistic_decode_accepted += 1;
                                    chained_optimistic_commits.push((
                                        chain_next,
                                        chain_stats,
                                        chained.chain_depth,
                                    ));
                                    rolling_context = context_with_expected;
                                    expected_target = chain_next;
                                } else {
                                    speculative_stats.optimistic_decode_rejected += 1;
                                    speculative_stats.chained_optimistic_decode_rejected += 1;
                                    if use_spd_rolling_executor {
                                        drain_spd_rolling_executor_replies(
                                            self,
                                            &request,
                                            &mut rolling_optimistic_decodes,
                                        )?;
                                        merge_spd_rolling_executor_stats(
                                            &mut speculative_stats,
                                            spd_rolling_executor.as_ref(),
                                        );
                                        if let Some(drop) = reset_spd_rolling_shadow_session(
                                            self,
                                            &request,
                                            downstream,
                                            &mut spd_rolling_shadow_session,
                                        )? {
                                            speculative_stats.recovery_ms += drop.elapsed_ms;
                                        }
                                    } else {
                                        let restore = self.restore_embedded_stage_session(
                                            &request,
                                            downstream,
                                            &session_key,
                                            request_id,
                                            session_id,
                                            chained.origin.checkpoint_generation,
                                        )?;
                                        speculative_stats.recovery_restores += 1;
                                        speculative_stats.recovery_ms += restore.elapsed_ms;
                                        speculative_stats.recovery_restore_ms += restore.elapsed_ms;
                                        speculative_stats.recovery_restore_local_ms +=
                                            restore.local_ms;
                                        speculative_stats.recovery_restore_downstream_write_ms +=
                                            restore.downstream_write_ms;
                                        speculative_stats.recovery_restore_downstream_wait_ms +=
                                            restore.downstream_wait_ms;
                                        speculative_stats.optimistic_restore_ms +=
                                            restore.elapsed_ms;
                                    }
                                    if chained.requested_spd_taps
                                        && let Some(spd) = spd_guard.as_deref_mut()
                                    {
                                        let reset_timer = PhaseTimer::start();
                                        spd.reset_to_verified_context(&context_with_expected)
                                            .map_err(openai_backend_error)?;
                                        speculative_stats.draft_reset_ms +=
                                            reset_timer.elapsed_ms();
                                    }
                                }
                                let mut attrs = self.openai_attrs(request.ids);
                                attrs.insert(
                                    "llama_stage.decode_step".to_string(),
                                    json!(chain_decode_step),
                                );
                                attrs.insert(
                                    "llama_stage.spd_optimistic_chain".to_string(),
                                    json!(true),
                                );
                                attrs.insert(
                                    "llama_stage.spd_optimistic_chain_depth".to_string(),
                                    json!(chained.chain_depth),
                                );
                                attrs.insert(
                                    "llama_stage.spd_optimistic_proposed_token".to_string(),
                                    json!(chained.proposed),
                                );
                                attrs.insert(
                                    "llama_stage.spd_optimistic_proposed_logit".to_string(),
                                    json!(chained.proposed_logit),
                                );
                                attrs.insert(
                                    "llama_stage.spd_optimistic_logit_margin".to_string(),
                                    json!(chained.proposed_logit_margin),
                                );
                                attrs.insert(
                                    "llama_stage.spd_optimistic_tap_return".to_string(),
                                    json!(chained.requested_spd_taps),
                                );
                                attrs.insert(
                                    "llama_stage.spd_optimistic_target_token".to_string(),
                                    json!(chain_target),
                                );
                                attrs.insert(
                                    "llama_stage.spd_optimistic_accepted".to_string(),
                                    json!(chain_accepted),
                                );
                                attrs.insert(
                                    "llama_stage.spd_optimistic_next_token".to_string(),
                                    json!(chain_next),
                                );
                                attrs.insert(
                                    "llama_stage.spd_optimistic_checkpoint_ms".to_string(),
                                    json!(chain_checkpoint_ms),
                                );
                                attrs.insert(
                                    "llama_stage.spd_optimistic_decode_elapsed_ms".to_string(),
                                    json!(chain_elapsed_ms),
                                );
                                attrs.insert(
                                    "llama_stage.spd_optimistic_start_elapsed_ms".to_string(),
                                    json!(chain_start_elapsed_ms),
                                );
                                attrs.insert(
                                    "llama_stage.spd_optimistic_decode_wait_ms".to_string(),
                                    json!(chain_wait_ms),
                                );
                                attrs.insert(
                                    "llama_stage.stage0_compute_ms".to_string(),
                                    json!(chain_stats.stage0_compute_ms),
                                );
                                attrs.insert(
                                    "llama_stage.forward_write_ms".to_string(),
                                    json!(chain_stats.forward_write_ms),
                                );
                                attrs.insert(
                                    "llama_stage.downstream_wait_ms".to_string(),
                                    json!(chain_wait_ms),
                                );
                                self.telemetry
                                    .emit_debug("stage.openai_spd_optimistic_decode", attrs);
                                if !chain_accepted {
                                    break;
                                }
                            }
                        }
                    }
                    speculative_stats.draft_propose_ms += self.emit_spd_inline_probe(
                        &request,
                        decode_step,
                        current,
                        reply.predicted,
                        probe,
                        rolling.as_ref(),
                    );
                } else if spd_guard.is_some() {
                    let probe_timer = PhaseTimer::start();
                    let (proposal, proposal_miss) = {
                        let spd = spd_guard
                            .as_deref_mut()
                            .ok_or_else(|| OpenAiError::backend("missing SPD proposal source"))?;
                        let proposal = spd
                            .propose_inline_for_current_context(current)
                            .map_err(openai_backend_error)?;
                        let proposal_miss = if proposal.is_none() {
                            spd.inline_proposal_miss_for_current_context(current)
                                .map_err(openai_backend_error)?
                        } else {
                            None
                        };
                        (proposal, proposal_miss)
                    };
                    let proposed = proposal.as_ref().map(|proposal| proposal.token);
                    if let Some(proposed) = proposed {
                        speculative_stats
                            .observe_inline_verified_probe(proposed == reply.predicted);
                    }
                    let rolling = if let Some(spd) = spd_guard.as_deref_mut() {
                        Some(
                            spd.observe_rolling_probe(
                                rolling_target_position,
                                reply.predicted,
                                proposed,
                            )
                            .map_err(openai_backend_error)?,
                        )
                    } else {
                        None
                    };
                    speculative_stats.draft_propose_ms += self.emit_spd_inline_probe(
                        &request,
                        decode_step,
                        current,
                        reply.predicted,
                        super::spd::SpdInlineProbe::from_proposal(
                            super::spd::SpdInlineProbePhase::PostTargetReply,
                            proposal.as_ref(),
                            probe_timer.elapsed_ms(),
                            0.0,
                            None,
                        )
                        .with_proposal_miss(proposal_miss),
                        rolling.as_ref(),
                    );
                }
                if records_replay_checkpoint
                    && super::prefix_cache::request_allows_exact_replay(&request)
                {
                    self.record_embedded_stage0_replay_checkpoint(
                        super::prefix_cache::EmbeddedReplayCheckpointRecord {
                            session_id: &session_key,
                            ids: request.ids,
                            prompt_token_ids: request.prompt_token_ids,
                            checkpoint_token_ids: &context_tokens,
                            predicted_tokens: &exact_replay_tokens,
                            predicted: reply.predicted,
                            sampling: request.sampling,
                            chat_sampling_metadata: request.chat_sampling_metadata,
                        },
                    )?;
                } else if records_full_prompt_checkpoint {
                    self.record_embedded_stage0_full_prompt_first_token(
                        &session_key,
                        request.ids,
                        request.prompt_token_ids,
                        reply.predicted,
                    )?;
                }
                current = reply.predicted;
                let suppress_cooldown_draft_broad = native_mtp_options.suppress_cooldown_drafts
                    && native_mtp_reject_cooldown_remaining > 0;
                let suppress_cooldown_draft_limited = native_mtp_reject_cooldown_remaining > 0
                    && native_mtp_suppress_cooldown_drafts_remaining > 0;
                let suppress_cooldown_draft =
                    suppress_cooldown_draft_broad || suppress_cooldown_draft_limited;
                let native_mtp_draft = if suppress_cooldown_draft {
                    None
                } else {
                    NativeMtpDraft::from_prediction_tokens(&reply.predicted_tokens)
                };
                if suppress_cooldown_draft {
                    native_mtp.clear_pending_draft();
                    native_mtp_counters.observe_suppressed_cooldown_draft();
                    native_mtp_suppress_cooldown_drafts_remaining =
                        native_mtp_suppress_cooldown_drafts_remaining.saturating_sub(1);
                }
                let native_mtp_decision = native_mtp.observe_target_token(
                    current,
                    ms_to_us(downstream_wait_ms),
                    native_mtp_draft,
                    if native_mtp_counters.batched_verification_count() == 0 {
                        NativeMtpDraftOrigin::InitialSerial
                    } else {
                        NativeMtpDraftOrigin::SerialAfterGap
                    },
                );
                native_mtp_reject_cooldown_remaining =
                    native_mtp_reject_cooldown_remaining.saturating_sub(1);
                decoded_tokens += 1;
                exact_replay_tokens.push(current);
                context_tokens.push(current);
                if use_spd_rolling_executor {
                    advance_idle_spd_rolling_executor_to_context(
                        spd_rolling_executor.as_mut(),
                        &context_tokens,
                    )?;
                }
                if !spd_advanced_for_current && let Some(spd) = spd_guard.as_deref_mut() {
                    let reset_timer = PhaseTimer::start();
                    spd.advance_to_accepted_context(&context_tokens)
                        .map_err(openai_backend_error)?;
                    speculative_stats.draft_reset_ms += reset_timer.elapsed_ms();
                }
                if let Some((optimistic_token, _)) = optimistic_commit.as_ref()
                    && optimistic_commit_probes.is_empty()
                    && !use_spd_rolling_executor
                    && let Some(spd) = spd_guard.as_deref_mut()
                {
                    let optimistic_probe_step = u32::try_from(decoded_tokens)
                        .map_err(|_| OpenAiError::backend("decode step exceeds u32"))?;
                    let probe = {
                        let probe_timer = PhaseTimer::start();
                        let proposal = spd
                            .propose_inline_for_current_context(current)
                            .map_err(openai_backend_error)?;
                        let proposal_miss = if proposal.is_none() {
                            spd.inline_proposal_miss_for_current_context(current)
                                .map_err(openai_backend_error)?
                        } else {
                            None
                        };
                        super::spd::SpdInlineProbe::from_proposal(
                            super::spd::SpdInlineProbePhase::OptimisticCommit,
                            proposal.as_ref(),
                            probe_timer.elapsed_ms(),
                            0.0,
                            None,
                        )
                        .with_proposal_miss(proposal_miss)
                    };
                    let rolling_target_position = context_tokens.len();
                    let rolling = Some(
                        spd.observe_rolling_probe(
                            rolling_target_position,
                            *optimistic_token,
                            probe.proposed,
                        )
                        .map_err(openai_backend_error)?,
                    );
                    optimistic_commit_probes.push(PendingSpdInlineProbe {
                        decode_step: optimistic_probe_step,
                        current,
                        target: *optimistic_token,
                        probe,
                        rolling,
                    });
                }
                for pending_probe in optimistic_commit_probes.drain(..) {
                    let proposed = pending_probe.probe.proposed;
                    if let Some(proposed) = proposed {
                        speculative_stats
                            .observe_inline_verified_probe(proposed == pending_probe.target);
                    }
                    let rolling = pending_probe.rolling.as_ref();
                    speculative_stats.draft_propose_ms += self.emit_spd_inline_probe(
                        &request,
                        pending_probe.decode_step,
                        pending_probe.current,
                        pending_probe.target,
                        pending_probe.probe,
                        rolling,
                    );
                }
                let mut reached_stop = false;
                if self.telemetry.is_debug_enabled() {
                    let mut token_attrs = self.openai_attrs(request.ids);
                    token_attrs.insert("llama_stage.decode_step".to_string(), json!(decode_step));
                    token_attrs.insert(
                        "llama_stage.decode_token_phase".to_string(),
                        json!(decode_token_phase(decode_step)),
                    );
                    token_attrs.insert(
                        "llama_stage.stage0_compute_ms".to_string(),
                        json!(stage0_compute_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.runtime_lock_wait_ms".to_string(),
                        json!(token_runtime_lock_wait_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.runtime_lock_hold_ms".to_string(),
                        json!(token_runtime_lock_hold_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.decode_batch_size".to_string(),
                        json!(batch_outcome.batch_size),
                    );
                    token_attrs.insert(
                        "llama_stage.decode_batch_wait_ms".to_string(),
                        json!(batch_outcome.batch_wait_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.output_activation_bytes".to_string(),
                        json!(output.payload.len()),
                    );
                    token_attrs.insert(
                        "llama_stage.forward_activation_bytes".to_string(),
                        json!(forwarded.message.activation.len()),
                    );
                    token_attrs.insert(
                        "llama_stage.activation_encode_ms".to_string(),
                        json!(forwarded.activation_encode_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.forward_write_ms".to_string(),
                        json!(forward_write_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.downstream_wait_ms".to_string(),
                        json!(downstream_wait_ms),
                    );
                    token_attrs.insert("llama_stage.predicted_token".to_string(), json!(current));
                    token_attrs.insert("llama_stage.message_kind".to_string(), json!("DecodeEmbd"));
                    token_attrs.insert(
                        "llama_stage.native_mtp.verification".to_string(),
                        json!(native_mtp_decision.label()),
                    );
                    token_attrs.insert(
                        "llama_stage.native_mtp.suppress_cooldown_drafts".to_string(),
                        json!(native_mtp_options.suppress_cooldown_drafts),
                    );
                    token_attrs.insert(
                        "llama_stage.native_mtp.suppress_cooldown_draft_limit".to_string(),
                        json!(native_mtp_options.suppress_cooldown_draft_limit),
                    );
                    token_attrs.insert(
                        "llama_stage.native_mtp.cooldown_draft_suppressed".to_string(),
                        json!(suppress_cooldown_draft),
                    );
                    self.emit_openai_phase("stage.openai_decode_token", token_timer, token_attrs);
                }
                if on_token(current)? == TokenControl::Stop {
                    reached_stop = true;
                }
                if !reached_stop
                    && decoded_tokens < request.max_tokens as usize
                    && let Some((optimistic_token, optimistic_stats)) = optimistic_commit
                {
                    let optimistic_decode_step = u32::try_from(decoded_tokens)
                        .map_err(|_| OpenAiError::backend("decode step exceeds u32"))?;
                    let optimistic_target_position = context_tokens.len();
                    current = optimistic_token;
                    decoded_tokens += 1;
                    speculative_stats.optimistic_decode_committed_tokens += 1;
                    exact_replay_tokens.push(current);
                    context_tokens.push(current);
                    if use_spd_rolling_executor {
                        advance_idle_spd_rolling_executor_to_context(
                            spd_rolling_executor.as_mut(),
                            &context_tokens,
                        )?;
                    }
                    if self.telemetry.is_debug_enabled() {
                        let mut token_attrs = self.openai_attrs(request.ids);
                        token_attrs.insert(
                            "llama_stage.decode_step".to_string(),
                            json!(optimistic_decode_step),
                        );
                        token_attrs.insert(
                            "llama_stage.decode_token_phase".to_string(),
                            json!(decode_token_phase(optimistic_decode_step)),
                        );
                        token_attrs.insert(
                            "llama_stage.stage0_compute_ms".to_string(),
                            json!(optimistic_stats.stage0_compute_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_wait_ms".to_string(),
                            json!(optimistic_stats.runtime_lock_wait_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_hold_ms".to_string(),
                            json!(optimistic_stats.runtime_lock_hold_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.output_activation_bytes".to_string(),
                            json!(optimistic_stats.output_activation_bytes),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_activation_bytes".to_string(),
                            json!(optimistic_stats.forward_activation_bytes),
                        );
                        token_attrs.insert(
                            "llama_stage.activation_encode_ms".to_string(),
                            json!(optimistic_stats.activation_encode_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_write_ms".to_string(),
                            json!(optimistic_stats.forward_write_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.downstream_wait_ms".to_string(),
                            json!(optimistic_stats.downstream_wait_ms),
                        );
                        token_attrs
                            .insert("llama_stage.predicted_token".to_string(), json!(current));
                        token_attrs.insert(
                            "llama_stage.message_kind".to_string(),
                            json!("DecodeEmbdOptimistic"),
                        );
                        self.telemetry
                            .emit_debug("stage.openai_decode_token", token_attrs);
                    }
                    if let Some(spd) = spd_guard.as_deref_mut() {
                        let reset_timer = PhaseTimer::start();
                        spd.observe_rolling_target_token(optimistic_target_position, current)
                            .map_err(openai_backend_error)?;
                        spd.advance_to_accepted_context(&context_tokens)
                            .map_err(openai_backend_error)?;
                        speculative_stats.draft_reset_ms += reset_timer.elapsed_ms();
                    }
                    if on_token(current)? == TokenControl::Stop {
                        reached_stop = true;
                    }
                }
                for (chained_token, chained_stats, chain_depth) in chained_optimistic_commits {
                    if reached_stop || decoded_tokens >= request.max_tokens as usize {
                        break;
                    }
                    let chained_decode_step = u32::try_from(decoded_tokens)
                        .map_err(|_| OpenAiError::backend("decode step exceeds u32"))?;
                    let chained_target_position = context_tokens.len();
                    current = chained_token;
                    decoded_tokens += 1;
                    speculative_stats.optimistic_decode_committed_tokens += 1;
                    speculative_stats.chained_optimistic_decode_committed_tokens += 1;
                    exact_replay_tokens.push(current);
                    context_tokens.push(current);
                    if use_spd_rolling_executor {
                        advance_idle_spd_rolling_executor_to_context(
                            spd_rolling_executor.as_mut(),
                            &context_tokens,
                        )?;
                    }
                    if self.telemetry.is_debug_enabled() {
                        let mut token_attrs = self.openai_attrs(request.ids);
                        token_attrs.insert(
                            "llama_stage.decode_step".to_string(),
                            json!(chained_decode_step),
                        );
                        token_attrs.insert(
                            "llama_stage.decode_token_phase".to_string(),
                            json!(decode_token_phase(chained_decode_step)),
                        );
                        token_attrs.insert(
                            "llama_stage.stage0_compute_ms".to_string(),
                            json!(chained_stats.stage0_compute_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_wait_ms".to_string(),
                            json!(chained_stats.runtime_lock_wait_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_hold_ms".to_string(),
                            json!(chained_stats.runtime_lock_hold_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.output_activation_bytes".to_string(),
                            json!(chained_stats.output_activation_bytes),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_activation_bytes".to_string(),
                            json!(chained_stats.forward_activation_bytes),
                        );
                        token_attrs.insert(
                            "llama_stage.activation_encode_ms".to_string(),
                            json!(chained_stats.activation_encode_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_write_ms".to_string(),
                            json!(chained_stats.forward_write_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.downstream_wait_ms".to_string(),
                            json!(chained_stats.downstream_wait_ms),
                        );
                        token_attrs
                            .insert("llama_stage.predicted_token".to_string(), json!(current));
                        token_attrs.insert(
                            "llama_stage.message_kind".to_string(),
                            json!("DecodeEmbdOptimistic"),
                        );
                        token_attrs
                            .insert("llama_stage.spd_optimistic_chain".to_string(), json!(true));
                        token_attrs.insert(
                            "llama_stage.spd_optimistic_chain_depth".to_string(),
                            json!(chain_depth),
                        );
                        self.telemetry
                            .emit_debug("stage.openai_decode_token", token_attrs);
                    }
                    if let Some(spd) = spd_guard.as_deref_mut() {
                        let reset_timer = PhaseTimer::start();
                        spd.observe_rolling_target_token(chained_target_position, current)
                            .map_err(openai_backend_error)?;
                        spd.advance_to_accepted_context(&context_tokens)
                            .map_err(openai_backend_error)?;
                        speculative_stats.draft_reset_ms += reset_timer.elapsed_ms();
                    }
                    if on_token(current)? == TokenControl::Stop {
                        reached_stop = true;
                    }
                }
                if reached_stop {
                    break;
                }
            }
            if let Some(drop) = drop_spd_rolling_shadow_session(
                self,
                &request,
                downstream,
                &mut spd_rolling_shadow_session,
            )? {
                speculative_stats.recovery_ms += drop.elapsed_ms;
            }
            let mut decode_attrs = self.openai_attrs(request.ids);
            decode_attrs.insert(
                "llama_stage.decode_token_count".to_string(),
                json!(decoded_tokens),
            );
            decode_attrs.insert(
                "llama_stage.stage0_compute_ms".to_string(),
                json!(decode_stage0_compute_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(decode_runtime_lock_wait_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_wait_max_ms".to_string(),
                json!(decode_runtime_lock_wait_max_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(decode_runtime_lock_hold_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_hold_max_ms".to_string(),
                json!(decode_runtime_lock_hold_max_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(decode_runtime_lock_acquires),
            );
            decode_attrs.insert(
                "llama_stage.decode_batch_size_max".to_string(),
                json!(decode_batch_size_max),
            );
            decode_attrs.insert(
                "llama_stage.decode_batch_wait_ms".to_string(),
                json!(decode_batch_wait_ms),
            );
            decode_attrs.insert(
                "llama_stage.forward_write_ms".to_string(),
                json!(decode_forward_write_ms),
            );
            decode_attrs.insert(
                "llama_stage.activation_encode_ms".to_string(),
                json!(decode_forward_activation_encode_ms),
            );
            decode_attrs.insert(
                "llama_stage.output_activation_bytes".to_string(),
                json!(decode_output_activation_bytes),
            );
            decode_attrs.insert(
                "llama_stage.forward_activation_bytes".to_string(),
                json!(decode_forward_activation_bytes),
            );
            decode_attrs.insert(
                "llama_stage.downstream_wait_ms".to_string(),
                json!(decode_downstream_wait_ms),
            );
            if let Some(spd) = spd_guard.as_deref() {
                spd.insert_rolling_attrs(&mut decode_attrs);
                spd.insert_total_proposal_stats_attrs(&mut decode_attrs);
            }
            speculative_stats.insert_attrs(&mut decode_attrs);
            native_mtp.stats().insert_attrs(&mut decode_attrs);
            native_mtp_counters.insert_summary_attrs(&mut decode_attrs, native_mtp_options);
            self.emit_openai_summary("stage.openai_decode", decode_timer, decode_attrs);
            Ok(())
        })();

        let stop_result = write_stage_message(
            &mut lane.stream,
            &StageWireMessage::stop_with_identity(request.wire_dtype, request_id, session_id),
            request.wire_dtype,
        )
        .and_then(|_| recv_reply(&mut lane.stream).map(|reply| reply.kind))
        .and_then(|kind| {
            if kind == WireReplyKind::Ack {
                Ok(())
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("expected stop ACK, got {kind:?}"),
                ))
            }
        });
        let lock_timer = PhaseTimer::start();
        if let Ok(mut runtime) = self.runtime.lock() {
            let runtime_lock_wait_ms = lock_timer.elapsed_ms();
            if let Ok(drop_stats) = runtime.drop_session_timed(&session_key) {
                let mut attrs = self.openai_attrs(request.ids);
                attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(runtime_lock_wait_ms),
                );
                attrs.insert(
                    "llama_stage.session_reset_ms".to_string(),
                    json!(drop_stats.reset_ms),
                );
                attrs.insert(
                    "llama_stage.session_reset".to_string(),
                    json!(drop_stats.reset_session),
                );
                attrs.insert(
                    "llama_stage.lane_discarded".to_string(),
                    json!(drop_stats.lane_discarded),
                );
                if let Some(reason) = drop_stats.lane_discard_reason.as_deref() {
                    attrs.insert("llama_stage.lane_discard_reason".to_string(), json!(reason));
                }
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_after",
                    &drop_stats.stats_after,
                );
                self.telemetry
                    .emit_debug("stage.openai_session_stop", attrs);
            }
        }
        let lane_id = lane.id;
        let stop_result = stop_result.map_err(openai_io_error);
        match (&result, &stop_result) {
            (Ok(_), Ok(_)) => lane_pool.return_lane(lane),
            _ => lane_pool.replace_lane(lane_id),
        }
        if result.is_ok() {
            stop_result?;
        }
        result?;
        Ok(cache_stats)
    }
}

fn decode_uses_context_sideband(
    context_token_ids: &[i32],
    current: i32,
    sideband_capacity: usize,
) -> bool {
    context_token_ids.len() <= sideband_capacity
        && context_token_ids.last().copied() == Some(current)
}
