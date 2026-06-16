use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result, bail};
use skippy_runtime::spd::{
    GgufTokenEmbeddingTable, SpdHeadManifest, SpdLiveCurInRequest, SpdLiveTapRunner,
    SpdLiveTapRunnerConfig, SpdQwen3ForwardInput, SpdQwen3Head, SpdSafetensorsFile,
    SpdStageLayerRange, assemble_spd_live_cur_in_for_positions, plan_hidden_state_taps,
    sliding_spd_row_positions,
};
use skippy_runtime::{ActivationFrame, RuntimeActivationDType, RuntimeActivationLayout};

use super::*;

pub(super) const SPD_REPLAY_PROPOSAL_SOURCE: &str = "spd-replay";

pub(super) struct SpdReplayOpenArgs<'a> {
    pub(super) manifest_path: Option<&'a Path>,
    pub(super) fixture_path: Option<&'a Path>,
    pub(super) model_path: Option<&'a Path>,
    pub(super) config: &'a StageConfig,
    pub(super) topology: Option<&'a StageTopology>,
    pub(super) n_gpu_layers: Option<i32>,
    pub(super) replay_fallback: bool,
    pub(super) window: usize,
    pub(super) top_k: usize,
}

#[derive(Clone)]
pub(super) struct SpdReplayProposalState {
    pub(super) source: Arc<Mutex<SpdReplayProposalSource>>,
    taps: Arc<Mutex<SpdInlineTapCache>>,
}

pub(super) struct SpdReplayProposalSource {
    pub(super) manifest_path: PathBuf,
    pub(super) model_path: PathBuf,
    pub(super) window: usize,
    top_k: usize,
    row_count: usize,
    row_stage_ids: Vec<i64>,
    row_hf_indices: Vec<Vec<u32>>,
    required_hf_indices: Vec<u32>,
    hidden_size: usize,
    final_norm_weight: Vec<f32>,
    context_tokens: Vec<i32>,
    head: SpdQwen3Head,
    manifest: SpdHeadManifest,
    serving_file: SpdSafetensorsFile,
    live_taps: SpdLiveTapRunner,
    h0_embeddings: Option<GgufTokenEmbeddingTable>,
    inline_taps: Arc<Mutex<SpdInlineTapCache>>,
    replay_fallback: bool,
}

impl SpdReplayProposalSource {
    fn open(args: SpdReplayOpenArgs<'_>) -> Result<Self> {
        if args.window == 0 {
            bail!("--openai-speculative-window must be greater than zero when SPD is set");
        }
        if args.top_k == 0 {
            bail!("--openai-spd-top-k must be greater than zero");
        }
        let manifest_path = args.manifest_path.context("missing SPD manifest path")?;
        let fixture_path = args.fixture_path.context("missing SPD fixture path")?;
        let topology = args
            .topology
            .context("--openai-spd-manifest requires --topology for stage layer ranges")?;
        let model_path = resolve_spd_model_path(args.model_path, args.config)?;
        let head = SpdQwen3Head::open(manifest_path).context("open SPD Qwen head")?;
        let manifest = head.manifest().clone();
        let serving_file =
            SpdSafetensorsFile::open(manifest.serving_checkpoint_path(manifest_path)?)
                .context("open SPD serving checkpoint")?;
        let fixture_file =
            SpdSafetensorsFile::open(fixture_path).context("open SPD parity fixture")?;
        let hidden_size =
            usize::try_from(manifest.topology.hidden_size).context("SPD hidden_size too large")?;
        let vocab_size =
            usize::try_from(manifest.topology.vocab_size).context("SPD vocab_size too large")?;
        let row_count = fixture_cur_in_row_count(&fixture_file, hidden_size)?;
        let row_stage_ids = read_spd_row_stage_ids(&fixture_file, row_count)?;
        let row_hf_indices = read_spd_row_hf_indices(&fixture_file, row_count)?;
        let required_hf_indices = required_spd_hf_indices(&row_hf_indices);
        let final_norm_weight = read_spd_final_norm_weight(&fixture_file, hidden_size)?;
        let stage_ranges = spd_stage_ranges_from_topology(topology)?;
        let tap_plan = plan_hidden_state_taps(&manifest.topology, &stage_ranges)?;
        if tap_plan.requires_internal_taps() {
            bail!(
                "experimental SPD replay source requires boundary-aligned splits; missing hidden states {:?}",
                tap_plan.boundary_only_missing_hf_indices
            );
        }
        let live_taps = SpdLiveTapRunner::open(SpdLiveTapRunnerConfig {
            model_path: &model_path,
            stage_ranges: &stage_ranges,
            layer_end: stage_ranges
                .last()
                .map(|range| range.layer_end)
                .context("SPD topology has no stages")?,
            ctx_size: args.config.ctx_size,
            n_gpu_layers: args.n_gpu_layers.unwrap_or(args.config.n_gpu_layers),
            selected_backend_device: args
                .config
                .selected_device
                .as_ref()
                .map(|device| device.backend_device.clone()),
        })
        .context("open live SPD tap replay stages")?;
        let h0_embeddings =
            GgufTokenEmbeddingTable::open(&model_path, hidden_size, vocab_size).ok();
        let inline_taps = Arc::new(Mutex::new(SpdInlineTapCache::new(
            hidden_size,
            required_hf_indices.clone(),
        )));
        Ok(Self {
            manifest_path: manifest_path.to_path_buf(),
            model_path,
            window: args.window,
            top_k: args.top_k,
            row_count,
            row_stage_ids,
            row_hf_indices,
            required_hf_indices,
            hidden_size,
            final_norm_weight,
            context_tokens: Vec::new(),
            head,
            manifest,
            serving_file,
            live_taps,
            h0_embeddings,
            inline_taps,
            replay_fallback: args.replay_fallback,
        })
    }

    fn propose_one(&self, context_tokens: &[i32]) -> Result<Option<i32>> {
        let row_positions = sliding_spd_row_positions(context_tokens.len(), self.row_count)?;
        let Some(taps) = self.collect_taps_for_proposal(context_tokens, &row_positions)? else {
            return Ok(None);
        };
        let live_rows = assemble_spd_live_cur_in_for_positions(SpdLiveCurInRequest {
            manifest: &self.manifest,
            serving_file: &self.serving_file,
            taps: &taps,
            row_positions: &row_positions,
            row_stage_ids: &self.row_stage_ids,
            row_hf_indices: &self.row_hf_indices,
            hidden_size: self.hidden_size,
        })?;
        let topk = self.head.forward(
            SpdQwen3ForwardInput {
                cur_in: live_rows.cur_in,
                seq_len: self.row_count,
                position_ids: row_positions,
                final_norm_weight: self.final_norm_weight.clone(),
            },
            self.top_k,
        )?;
        topk.token_ids
            .first()
            .copied()
            .context("SPD head returned no proposal token")
            .and_then(|token| i32::try_from(token).context("SPD proposal token exceeds i32"))
            .map(Some)
    }

    fn collect_taps_for_proposal(
        &self,
        context_tokens: &[i32],
        row_positions: &[i64],
    ) -> Result<Option<BTreeMap<u32, ActivationFrame>>> {
        if let Some(mut taps) = self.complete_inline_taps(row_positions)? {
            if self.required_hf_indices.contains(&0) {
                taps.insert(0, self.collect_h0_tap(context_tokens, row_positions)?);
            }
            return Ok(Some(taps));
        }
        if !self.replay_fallback {
            return Ok(None);
        }
        let mut taps = self.live_taps.collect_taps(context_tokens)?;
        self.overlay_inline_taps(&mut taps, row_positions)?;
        Ok(Some(taps))
    }

    fn collect_h0_tap(
        &self,
        context_tokens: &[i32],
        row_positions: &[i64],
    ) -> Result<ActivationFrame> {
        if let Some(embeddings) = &self.h0_embeddings {
            return embeddings
                .frame_for_positions(context_tokens, row_positions)
                .context("collect GGUF token embedding h0 tap");
        }
        self.live_taps.collect_h0_tap(context_tokens)
    }

    pub(super) fn propose_inline_for_current_context(&self, current: i32) -> Result<Option<i32>> {
        if self.context_tokens.last().copied() != Some(current) {
            return Ok(None);
        }
        self.propose_one(&self.context_tokens)
    }

    fn complete_inline_taps(
        &self,
        row_positions: &[i64],
    ) -> Result<Option<BTreeMap<u32, ActivationFrame>>> {
        let inline_taps = self
            .inline_taps
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD inline tap cache lock poisoned"))?;
        inline_taps.complete_frames(
            row_positions,
            &non_h0_required_hf_indices(&self.required_hf_indices),
            self.hidden_size,
        )
    }

    fn overlay_inline_taps(
        &self,
        taps: &mut BTreeMap<u32, ActivationFrame>,
        row_positions: &[i64],
    ) -> Result<()> {
        let mut inline_taps = self
            .inline_taps
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD inline tap cache lock poisoned"))?;
        inline_taps.overlay_complete_frames(
            taps,
            row_positions,
            &self.required_hf_indices,
            self.hidden_size,
        )
    }
}

impl SpeculativeProposalSource for SpdReplayProposalSource {
    fn label(&self) -> &'static str {
        SPD_REPLAY_PROPOSAL_SOURCE
    }

    fn max_window(&self) -> usize {
        self.window
    }

    fn reset_to_context(&mut self, context_tokens: &[i32]) -> Result<()> {
        self.context_tokens = context_tokens.to_vec();
        self.inline_taps
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD inline tap cache lock poisoned"))?
            .clear();
        Ok(())
    }

    fn propose(&mut self, current: i32, max_tokens: usize) -> Result<Vec<i32>> {
        if self.context_tokens.last().copied() != Some(current) {
            self.context_tokens.push(current);
        }
        if self.context_tokens.len() < self.row_count {
            return Ok(Vec::new());
        }
        let mut proposals = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            let Some(proposal) = self.propose_one(&self.context_tokens)? else {
                break;
            };
            proposals.push(proposal);
            self.context_tokens.push(proposal);
        }
        Ok(proposals)
    }
}

pub(super) fn open_spd_replay_source(
    args: SpdReplayOpenArgs<'_>,
) -> Result<Option<SpdReplayProposalState>> {
    match (args.manifest_path, args.fixture_path) {
        (None, None) => Ok(None),
        (Some(_), Some(_)) => {
            let source = SpdReplayProposalSource::open(args)?;
            let taps = source.inline_taps.clone();
            Ok(Some(SpdReplayProposalState {
                source: Arc::new(Mutex::new(source)),
                taps,
            }))
        }
        _ => bail!("--openai-spd-manifest and --openai-spd-fixture must be set together"),
    }
}

impl StageOpenAiBackend {
    pub(super) fn mark_spd_tap_return(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        message: &mut StageWireMessage,
    ) {
        if request.spd.is_some() {
            message.state.flags |= state_flags::SPD_TAP_RETURN;
        }
    }

    pub(super) fn recv_spd_aware_prediction_return(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        expected: WireReplyKind,
    ) -> Result<StageReply> {
        Ok(self
            .recv_spd_aware_prediction_return_with_probe(request, expected, None, 0)?
            .reply)
    }

    pub(super) fn recv_spd_aware_prediction_return_with_probe(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        expected: WireReplyKind,
        spd_source: Option<&SpdReplayProposalSource>,
        current: i32,
    ) -> Result<SpdPredictionReturn> {
        let receiver = request
            .prediction_return
            .as_ref()
            .context("missing direct prediction return receiver")?;
        let mut pre_target_probe = None;
        let mut wait_after_probe_timer = None;
        loop {
            let reply = receiver.recv()?;
            if reply.kind == WireReplyKind::SpdTap {
                let trigger_hf_index = reply.spd_tap.as_ref().map(|tap| tap.hf_index);
                self.record_spd_direct_return_tap(request, &reply);
                if pre_target_probe.is_none()
                    && let Some(spd) = spd_source
                {
                    let probe_timer = PhaseTimer::start();
                    let proposed = spd.propose_inline_for_current_context(current)?;
                    let elapsed_ms = probe_timer.elapsed_ms();
                    if proposed.is_some() {
                        pre_target_probe = Some(SpdInlineProbe {
                            phase: SpdInlineProbePhase::PreTargetReply,
                            proposed,
                            elapsed_ms,
                            target_wait_after_probe_ms: 0.0,
                            trigger_hf_index,
                        });
                        wait_after_probe_timer = Some(PhaseTimer::start());
                    }
                }
                continue;
            }
            if reply.kind != expected {
                bail!(
                    "expected {expected:?} direct prediction return, got {:?}",
                    reply.kind
                );
            }
            if let (Some(probe), Some(wait_timer)) =
                (pre_target_probe.as_mut(), wait_after_probe_timer)
            {
                probe.target_wait_after_probe_ms = wait_timer.elapsed_ms();
            }
            return Ok(SpdPredictionReturn {
                reply,
                pre_target_probe,
            });
        }
    }

    pub(super) fn emit_spd_inline_probe(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        decode_step: u32,
        current: i32,
        target: i32,
        probe: SpdInlineProbe,
    ) -> f64 {
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert("llama_stage.decode_step".to_string(), json!(decode_step));
        attrs.insert(
            "llama_stage.elapsed_ms".to_string(),
            json!(probe.elapsed_ms),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_phase".to_string(),
            json!(probe.phase.as_str()),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_ready".to_string(),
            json!(probe.proposed.is_some()),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_current_token".to_string(),
            json!(current),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_proposed_token".to_string(),
            json!(probe.proposed),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_target_token".to_string(),
            json!(target),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_accepted".to_string(),
            json!(probe.proposed == Some(target)),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_target_wait_after_probe_ms".to_string(),
            json!(probe.target_wait_after_probe_ms),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_trigger_hf_index".to_string(),
            json!(probe.trigger_hf_index),
        );
        self.telemetry
            .emit_debug("stage.openai_spd_inline_probe", attrs);
        probe.elapsed_ms
    }

    pub(super) fn record_spd_stage0_boundary_tap(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        message: &StageWireMessage,
        frame: &ActivationFrame,
    ) {
        let Some(spd) = request.spd.as_ref() else {
            return;
        };
        let outcome = spd
            .taps
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD inline tap cache lock poisoned"))
            .and_then(|mut taps| taps.record_stage_output(request.config, message, frame));
        match outcome {
            Ok(Some(record)) => {
                let mut attrs = self.openai_attrs(request.ids);
                attrs.insert(
                    "llama_stage.spd_inline_tap_hf_index".to_string(),
                    json!(record.hf_index),
                );
                attrs.insert(
                    "llama_stage.spd_inline_tap_rows_recorded".to_string(),
                    json!(record.rows_recorded),
                );
                attrs.insert(
                    "llama_stage.spd_inline_tap_cached_rows".to_string(),
                    json!(record.cached_rows),
                );
                attrs.insert(
                    "llama_stage.spd_inline_tap_payload_bytes".to_string(),
                    json!(record.payload_bytes),
                );
                attrs.insert(
                    "llama_stage.spd_inline_tap_required".to_string(),
                    json!(record.required),
                );
                self.telemetry
                    .emit_debug("stage.openai_spd_tap_record", attrs);
            }
            Ok(None) => {}
            Err(error) => {
                let mut attrs = self.openai_attrs(request.ids);
                attrs.insert(
                    "llama_stage.spd_inline_tap_error".to_string(),
                    json!(error.to_string()),
                );
                self.telemetry
                    .emit_debug("stage.openai_spd_tap_record_failed", attrs);
            }
        }
    }

    fn record_spd_direct_return_tap(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        reply: &StageReply,
    ) {
        let Some(spd) = request.spd.as_ref() else {
            return;
        };
        let Some(tap) = reply.spd_tap.as_ref() else {
            let mut attrs = self.openai_attrs(request.ids);
            attrs.insert(
                "llama_stage.spd_inline_tap_error".to_string(),
                json!("missing SPD tap reply payload"),
            );
            self.telemetry
                .emit_debug("stage.openai_spd_tap_record_failed", attrs);
            return;
        };
        let outcome = spd
            .taps
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD inline tap cache lock poisoned"))
            .and_then(|mut taps| taps.record_returned_tap(tap));
        match outcome {
            Ok(record) => {
                let mut attrs = self.openai_attrs(request.ids);
                attrs.insert(
                    "llama_stage.spd_inline_tap_hf_index".to_string(),
                    json!(record.hf_index),
                );
                attrs.insert(
                    "llama_stage.spd_inline_tap_rows_recorded".to_string(),
                    json!(record.rows_recorded),
                );
                attrs.insert(
                    "llama_stage.spd_inline_tap_cached_rows".to_string(),
                    json!(record.cached_rows),
                );
                attrs.insert(
                    "llama_stage.spd_inline_tap_payload_bytes".to_string(),
                    json!(record.payload_bytes),
                );
                attrs.insert(
                    "llama_stage.spd_inline_tap_required".to_string(),
                    json!(record.required),
                );
                attrs.insert(
                    "llama_stage.spd_inline_tap_producer_stage_index".to_string(),
                    json!(tap.producer_stage_index),
                );
                self.telemetry
                    .emit_debug("stage.openai_spd_tap_record", attrs);
            }
            Err(error) => {
                let mut attrs = self.openai_attrs(request.ids);
                attrs.insert(
                    "llama_stage.spd_inline_tap_error".to_string(),
                    json!(error.to_string()),
                );
                attrs.insert(
                    "llama_stage.spd_inline_tap_hf_index".to_string(),
                    json!(tap.hf_index),
                );
                self.telemetry
                    .emit_debug("stage.openai_spd_tap_record_failed", attrs);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct SpdPredictionReturn {
    pub(super) reply: StageReply,
    pub(super) pre_target_probe: Option<SpdInlineProbe>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct SpdInlineProbe {
    pub(super) phase: SpdInlineProbePhase,
    pub(super) proposed: Option<i32>,
    pub(super) elapsed_ms: f64,
    pub(super) target_wait_after_probe_ms: f64,
    pub(super) trigger_hf_index: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SpdInlineProbePhase {
    PreTargetReply,
    PostTargetReply,
}

impl SpdInlineProbePhase {
    fn as_str(self) -> &'static str {
        match self {
            Self::PreTargetReply => "pre_target_reply",
            Self::PostTargetReply => "post_target_reply",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SpdInlineTapRecord {
    hf_index: u32,
    rows_recorded: usize,
    cached_rows: usize,
    payload_bytes: usize,
    required: bool,
}

struct SpdInlineTapCache {
    hidden_size: usize,
    required_hf_indices: BTreeSet<u32>,
    frames: BTreeMap<u32, SpdCachedTapFrame>,
}

impl SpdInlineTapCache {
    fn new(hidden_size: usize, required_hf_indices: Vec<u32>) -> Self {
        Self {
            hidden_size,
            required_hf_indices: required_hf_indices.into_iter().collect(),
            frames: BTreeMap::new(),
        }
    }

    fn clear(&mut self) {
        self.frames.clear();
    }

    fn record_stage_output(
        &mut self,
        config: &StageConfig,
        message: &StageWireMessage,
        frame: &ActivationFrame,
    ) -> Result<Option<SpdInlineTapRecord>> {
        let hf_index = config.layer_end;
        if frame.payload.is_empty() {
            return Ok(None);
        }
        validate_spd_inline_frame(frame, self.hidden_size)?;
        let token_count =
            usize::try_from(frame.desc.token_count).context("SPD tap token_count exceeds usize")?;
        if token_count == 0 {
            return Ok(None);
        }
        let positions = message_positions(message, token_count)?;
        self.record_rows(hf_index, positions, frame).map(Some)
    }

    fn record_returned_tap(&mut self, tap: &StageReplySpdTap) -> Result<SpdInlineTapRecord> {
        if tap.dtype != RuntimeActivationDType::F32 as i32 {
            bail!("SPD returned tap frame must be f32, got {}", tap.dtype);
        }
        if tap.layout != RuntimeActivationLayout::TokenMajor as i32 {
            bail!(
                "SPD returned tap frame must be token-major, got {}",
                tap.layout
            );
        }
        let frame = ActivationFrame {
            desc: skippy_runtime::ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F32,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: tap.producer_stage_index,
                layer_start: tap.layer_start,
                layer_end: tap.layer_end,
                token_count: tap.token_count,
                sequence_count: tap.sequence_count,
                payload_bytes: u64::try_from(tap.payload.len())
                    .context("SPD returned tap payload bytes exceed u64")?,
                flags: tap.flags,
            },
            payload: tap.payload.clone(),
        };
        let positions = tap
            .positions
            .iter()
            .copied()
            .map(|position| {
                u32::try_from(position)
                    .with_context(|| format!("negative SPD returned tap position {position}"))
            })
            .collect::<Result<Vec<_>>>()?;
        self.record_rows(tap.hf_index, positions, &frame)
    }

    fn record_rows(
        &mut self,
        hf_index: u32,
        positions: Vec<u32>,
        frame: &ActivationFrame,
    ) -> Result<SpdInlineTapRecord> {
        validate_spd_inline_frame(frame, self.hidden_size)?;
        let token_count =
            usize::try_from(frame.desc.token_count).context("SPD tap token_count exceeds usize")?;
        if positions.len() != token_count {
            bail!(
                "SPD inline tap positions length {} does not match token_count {}",
                positions.len(),
                token_count
            );
        }
        if positions.is_empty() {
            bail!("SPD inline tap has no positions");
        }
        let required = self.required_hf_indices.contains(&hf_index);
        let cached = self
            .frames
            .entry(hf_index)
            .or_insert_with(|| SpdCachedTapFrame::new(frame.desc));
        let row_bytes = self
            .hidden_size
            .checked_mul(std::mem::size_of::<f32>())
            .context("SPD inline tap row byte width overflow")?;
        for (row_index, position) in positions.iter().copied().enumerate() {
            let offset = row_index
                .checked_mul(row_bytes)
                .context("SPD inline tap payload offset overflow")?;
            cached
                .rows
                .insert(position, frame.payload[offset..offset + row_bytes].to_vec());
        }
        Ok(SpdInlineTapRecord {
            hf_index,
            rows_recorded: positions.len(),
            cached_rows: cached.rows.len(),
            payload_bytes: frame.payload.len(),
            required,
        })
    }

    fn overlay_complete_frames(
        &mut self,
        taps: &mut BTreeMap<u32, ActivationFrame>,
        row_positions: &[i64],
        required_hf_indices: &[u32],
        hidden_size: usize,
    ) -> Result<()> {
        if hidden_size != self.hidden_size {
            bail!(
                "SPD inline tap hidden size mismatch: cache {}, request {}",
                self.hidden_size,
                hidden_size
            );
        }
        for hf_index in required_hf_indices {
            let Some(frame) = self.frame_for_positions(*hf_index, row_positions)? else {
                continue;
            };
            taps.insert(*hf_index, frame);
        }
        Ok(())
    }

    fn complete_frames(
        &self,
        row_positions: &[i64],
        required_hf_indices: &[u32],
        hidden_size: usize,
    ) -> Result<Option<BTreeMap<u32, ActivationFrame>>> {
        if hidden_size != self.hidden_size {
            bail!(
                "SPD inline tap hidden size mismatch: cache {}, request {}",
                self.hidden_size,
                hidden_size
            );
        }
        let mut complete = BTreeMap::new();
        for hf_index in required_hf_indices {
            let Some(frame) = self.frame_for_positions(*hf_index, row_positions)? else {
                return Ok(None);
            };
            complete.insert(*hf_index, frame);
        }
        Ok(Some(complete))
    }

    fn frame_for_positions(
        &self,
        hf_index: u32,
        row_positions: &[i64],
    ) -> Result<Option<ActivationFrame>> {
        let Some(cached) = self.frames.get(&hf_index) else {
            return Ok(None);
        };
        let positions = row_positions
            .iter()
            .copied()
            .map(|position| {
                u32::try_from(position)
                    .with_context(|| format!("negative SPD inline tap position {position}"))
            })
            .collect::<Result<Vec<_>>>()?;
        if positions
            .iter()
            .any(|position| !cached.rows.contains_key(position))
        {
            return Ok(None);
        }
        let token_count = positions
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .context("SPD inline tap synthetic token_count overflow")?;
        let row_bytes = self
            .hidden_size
            .checked_mul(std::mem::size_of::<f32>())
            .context("SPD inline tap row byte width overflow")?;
        let total_bytes = usize::try_from(token_count)
            .context("SPD inline tap token_count exceeds usize")?
            .checked_mul(row_bytes)
            .context("SPD inline tap synthetic payload overflow")?;
        let mut payload = vec![0_u8; total_bytes];
        for position in positions {
            let row = cached
                .rows
                .get(&position)
                .context("missing cached SPD row after completeness check")?;
            let offset = usize::try_from(position)
                .context("SPD inline tap position exceeds usize")?
                .checked_mul(row_bytes)
                .context("SPD inline tap synthetic offset overflow")?;
            payload[offset..offset + row_bytes].copy_from_slice(row);
        }
        let mut desc = cached.desc;
        desc.token_count = token_count;
        desc.sequence_count = if token_count == 0 { 0 } else { 1 };
        desc.payload_bytes =
            u64::try_from(payload.len()).context("SPD inline tap payload bytes exceed u64")?;
        Ok(Some(ActivationFrame { desc, payload }))
    }
}

#[derive(Clone)]
struct SpdCachedTapFrame {
    desc: skippy_runtime::ActivationDesc,
    rows: BTreeMap<u32, Vec<u8>>,
}

impl SpdCachedTapFrame {
    fn new(desc: skippy_runtime::ActivationDesc) -> Self {
        Self {
            desc,
            rows: BTreeMap::new(),
        }
    }
}

fn resolve_spd_model_path(override_path: Option<&Path>, config: &StageConfig) -> Result<PathBuf> {
    if let Some(path) = override_path {
        ensure_model_file(path)?;
        return Ok(path.to_path_buf());
    }
    for value in [&config.source_model_path, &config.model_path]
        .into_iter()
        .flatten()
    {
        let path = PathBuf::from(value);
        if path.is_file() {
            return Ok(path);
        }
    }
    bail!(
        "SPD replay source requires a full GGUF via --openai-spd-model-path, source_model_path, or model_path"
    )
}

fn required_spd_hf_indices(row_hf_indices: &[Vec<u32>]) -> Vec<u32> {
    row_hf_indices
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn non_h0_required_hf_indices(required_hf_indices: &[u32]) -> Vec<u32> {
    required_hf_indices
        .iter()
        .copied()
        .filter(|hf_index| *hf_index != 0)
        .collect()
}

fn validate_spd_inline_frame(frame: &ActivationFrame, hidden_size: usize) -> Result<()> {
    if frame.desc.dtype != RuntimeActivationDType::F32 {
        bail!(
            "SPD inline tap frame must be f32, got {:?}",
            frame.desc.dtype
        );
    }
    if frame.desc.layout != RuntimeActivationLayout::TokenMajor {
        bail!(
            "SPD inline tap frame must be token-major, got {:?}",
            frame.desc.layout
        );
    }
    let token_count =
        usize::try_from(frame.desc.token_count).context("SPD tap token_count exceeds usize")?;
    let expected = token_count
        .checked_mul(hidden_size)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("SPD inline tap expected payload byte count overflow")?;
    if frame.payload.len() != expected {
        bail!(
            "SPD inline tap payload has {} bytes, expected {} for {} tokens x hidden {}",
            frame.payload.len(),
            expected,
            token_count,
            hidden_size
        );
    }
    Ok(())
}

fn message_positions(message: &StageWireMessage, token_count: usize) -> Result<Vec<u32>> {
    if message.positions.len() == token_count {
        return message
            .positions
            .iter()
            .copied()
            .map(|position| {
                u32::try_from(position)
                    .with_context(|| format!("negative SPD inline tap position {position}"))
            })
            .collect();
    }
    if !message.positions.is_empty() {
        bail!(
            "SPD inline tap positions length {} does not match token_count {}",
            message.positions.len(),
            token_count
        );
    }
    let start = u32::try_from(message.pos_start).context("negative SPD inline tap pos_start")?;
    (0..token_count)
        .map(|offset| {
            let offset = u32::try_from(offset).context("SPD inline tap offset exceeds u32")?;
            start
                .checked_add(offset)
                .context("SPD inline tap position overflow")
        })
        .collect()
}

fn ensure_model_file(path: &Path) -> Result<()> {
    if !path.is_file() {
        bail!("SPD replay source model does not exist: {}", path.display());
    }
    Ok(())
}

fn spd_stage_ranges_from_topology(topology: &StageTopology) -> Result<Vec<SpdStageLayerRange>> {
    let mut ranges = topology
        .stages
        .iter()
        .map(|stage| SpdStageLayerRange::new(stage.stage_index, stage.layer_start, stage.layer_end))
        .collect::<Vec<_>>();
    ranges.sort_by_key(|range| (range.layer_start, range.layer_end, range.stage_index));
    if ranges.is_empty() {
        bail!("SPD topology has no stages");
    }
    Ok(ranges)
}

fn fixture_cur_in_row_count(fixture: &SpdSafetensorsFile, hidden_size: usize) -> Result<usize> {
    let shape = &fixture.index.tensor("cur_in")?.shape;
    if shape.len() != 3 || shape[0] != 1 || shape[2] != hidden_size as u64 {
        bail!(
            "SPD fixture cur_in shape {:?} is not [1, rows, hidden]",
            shape
        );
    }
    usize::try_from(shape[1]).context("SPD fixture row count exceeds usize")
}

fn read_spd_row_stage_ids(fixture: &SpdSafetensorsFile, row_count: usize) -> Result<Vec<i64>> {
    let row_stage_ids = fixture.read_tensor_i64("row_i_stages")?;
    if row_stage_ids.len() != row_count {
        bail!(
            "SPD fixture row_i_stages length {} does not match row count {}",
            row_stage_ids.len(),
            row_count
        );
    }
    Ok(row_stage_ids)
}

fn read_spd_row_hf_indices(
    fixture: &SpdSafetensorsFile,
    row_count: usize,
) -> Result<Vec<Vec<u32>>> {
    (0..row_count)
        .map(|row_index| {
            fixture
                .read_tensor_i64(&format!("tap_row_{row_index}_hf_indices"))?
                .into_iter()
                .map(|value| {
                    u32::try_from(value).with_context(|| {
                        format!("SPD fixture row {row_index} has negative hf index")
                    })
                })
                .collect()
        })
        .collect()
}

fn read_spd_final_norm_weight(
    fixture: &SpdSafetensorsFile,
    hidden_size: usize,
) -> Result<Vec<f32>> {
    let final_norm_weight = fixture.read_tensor_f32("final_norm_weight")?;
    if final_norm_weight.len() != hidden_size {
        bail!(
            "SPD fixture final_norm_weight length {} does not match hidden size {}",
            final_norm_weight.len(),
            hidden_size
        );
    }
    Ok(final_norm_weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::{
        LoadMode, StageTopologyEntry,
        binary::{StageStateHeader, WireActivationDType, WireMessageKind},
    };
    use skippy_runtime::{ActivationDesc, RuntimeActivationDType, RuntimeActivationLayout};

    #[test]
    fn stage_ranges_from_topology_are_layer_sorted() {
        let topology = StageTopology {
            topology_id: "topology".to_string(),
            model_id: "model".to_string(),
            stages: vec![
                stage_entry("stage-1", 1, 8, 16),
                stage_entry("stage-0", 0, 0, 8),
            ],
        };

        let ranges = spd_stage_ranges_from_topology(&topology).unwrap();

        assert_eq!(
            ranges,
            vec![
                SpdStageLayerRange::new(0, 0, 8),
                SpdStageLayerRange::new(1, 8, 16)
            ]
        );
    }

    #[test]
    fn inline_tap_cache_rebuilds_positioned_activation_frame() {
        let mut cache = SpdInlineTapCache::new(2, vec![8]);
        let config = stage_config("stage-0", 0, 0, 8);
        let message = stage_message(2, 2, Vec::new());
        let frame = activation_frame(0, 8, &[1.0, 2.0, 3.0, 4.0]);

        let record = cache
            .record_stage_output(&config, &message, &frame)
            .unwrap()
            .unwrap();
        assert_eq!(
            record,
            SpdInlineTapRecord {
                hf_index: 8,
                rows_recorded: 2,
                cached_rows: 2,
                payload_bytes: 16,
                required: true,
            }
        );

        let rebuilt = cache
            .frame_for_positions(8, &[2, 3])
            .unwrap()
            .expect("positions should be complete");
        assert_eq!(rebuilt.desc.token_count, 4);
        assert_eq!(f32_row(&rebuilt, 2, 2), vec![1.0, 2.0]);
        assert_eq!(f32_row(&rebuilt, 3, 2), vec![3.0, 4.0]);
    }

    #[test]
    fn inline_tap_cache_overlays_complete_required_frames() {
        let mut cache = SpdInlineTapCache::new(2, vec![8]);
        let config = stage_config("stage-0", 0, 0, 8);
        let message = stage_message(2, 2, vec![2, 3]);
        let frame = activation_frame(0, 8, &[5.0, 6.0, 7.0, 8.0]);
        cache
            .record_stage_output(&config, &message, &frame)
            .unwrap()
            .unwrap();
        let mut taps = BTreeMap::new();

        cache
            .overlay_complete_frames(&mut taps, &[2, 3], &[8], 2)
            .unwrap();

        let overlaid = taps.get(&8).expect("expected overlaid tap");
        assert_eq!(f32_row(overlaid, 2, 2), vec![5.0, 6.0]);
        assert_eq!(f32_row(overlaid, 3, 2), vec![7.0, 8.0]);
    }

    #[test]
    fn inline_tap_cache_returns_complete_required_frames() {
        let mut cache = SpdInlineTapCache::new(2, vec![10, 20]);
        let message = stage_message(4, 2, Vec::new());
        cache
            .record_stage_output(
                &stage_config("stage-1", 1, 8, 10),
                &message,
                &activation_frame(8, 10, &[1.0, 2.0, 3.0, 4.0]),
            )
            .unwrap()
            .unwrap();
        cache
            .record_stage_output(
                &stage_config("stage-3", 3, 16, 20),
                &message,
                &activation_frame(16, 20, &[5.0, 6.0, 7.0, 8.0]),
            )
            .unwrap()
            .unwrap();

        let complete = cache
            .complete_frames(&[4, 5], &[10, 20], 2)
            .unwrap()
            .expect("all required taps are complete");

        assert_eq!(complete.keys().copied().collect::<Vec<_>>(), vec![10, 20]);
        assert_eq!(f32_row(complete.get(&10).unwrap(), 4, 2), vec![1.0, 2.0]);
        assert_eq!(f32_row(complete.get(&20).unwrap(), 5, 2), vec![7.0, 8.0]);
    }

    #[test]
    fn inline_tap_cache_reports_missing_required_frame() {
        let mut cache = SpdInlineTapCache::new(2, vec![10, 20]);
        cache
            .record_stage_output(
                &stage_config("stage-1", 1, 8, 10),
                &stage_message(4, 1, Vec::new()),
                &activation_frame(8, 10, &[1.0, 2.0]),
            )
            .unwrap()
            .unwrap();

        assert!(cache.complete_frames(&[4], &[10, 20], 2).unwrap().is_none());
    }

    #[test]
    fn inline_tap_cache_records_unrequired_stage_output_without_overlaying_it() {
        let mut cache = SpdInlineTapCache::new(2, vec![8]);
        let config = stage_config("stage-1", 1, 8, 10);
        let message = stage_message(0, 1, Vec::new());
        let frame = activation_frame(8, 10, &[1.0, 2.0]);

        let record = cache
            .record_stage_output(&config, &message, &frame)
            .unwrap()
            .unwrap();
        assert_eq!(
            record,
            SpdInlineTapRecord {
                hf_index: 10,
                rows_recorded: 1,
                cached_rows: 1,
                payload_bytes: 8,
                required: false,
            }
        );
        let mut taps = BTreeMap::new();
        cache
            .overlay_complete_frames(&mut taps, &[0], &[8], 2)
            .unwrap();
        assert!(taps.is_empty());
    }

    #[test]
    fn inline_tap_cache_records_returned_downstream_tap() {
        let mut cache = SpdInlineTapCache::new(2, vec![10]);
        let tap = StageReplySpdTap {
            hf_index: 10,
            producer_stage_index: 1,
            layer_start: 8,
            layer_end: 10,
            token_count: 2,
            sequence_count: 1,
            dtype: RuntimeActivationDType::F32 as i32,
            layout: RuntimeActivationLayout::TokenMajor as i32,
            flags: 0,
            positions: vec![4, 5],
            payload: f32_payload(&[9.0, 10.0, 11.0, 12.0]),
        };

        let record = cache.record_returned_tap(&tap).unwrap();

        assert_eq!(
            record,
            SpdInlineTapRecord {
                hf_index: 10,
                rows_recorded: 2,
                cached_rows: 2,
                payload_bytes: 16,
                required: true,
            }
        );
        let overlaid = cache
            .frame_for_positions(10, &[4, 5])
            .unwrap()
            .expect("returned tap rows should be complete");
        assert_eq!(f32_row(&overlaid, 4, 2), vec![9.0, 10.0]);
        assert_eq!(f32_row(&overlaid, 5, 2), vec![11.0, 12.0]);
    }

    #[test]
    fn inline_tap_cache_clear_drops_recorded_rows() {
        let mut cache = SpdInlineTapCache::new(2, vec![8]);
        let config = stage_config("stage-0", 0, 0, 8);
        let message = stage_message(2, 1, Vec::new());
        let frame = activation_frame(0, 8, &[1.0, 2.0]);
        cache
            .record_stage_output(&config, &message, &frame)
            .unwrap()
            .unwrap();

        cache.clear();

        assert!(cache.frame_for_positions(8, &[2]).unwrap().is_none());
    }

    fn stage_entry(
        stage_id: &str,
        stage_index: u32,
        layer_start: u32,
        layer_end: u32,
    ) -> StageTopologyEntry {
        StageTopologyEntry {
            stage_id: stage_id.to_string(),
            stage_index,
            host: None,
            endpoint: "127.0.0.1:0".to_string(),
            layer_start,
            layer_end,
            load_mode: LoadMode::RuntimeSlice,
        }
    }

    fn stage_config(
        stage_id: &str,
        stage_index: u32,
        layer_start: u32,
        layer_end: u32,
    ) -> StageConfig {
        StageConfig {
            run_id: "run".to_string(),
            topology_id: "topology".to_string(),
            model_id: "model".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/model.gguf".to_string()),
            projector_path: None,
            stage_id: stage_id.to_string(),
            stage_index,
            layer_start,
            layer_end,
            ctx_size: 128,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: 0,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: skippy_protocol::FlashAttentionType::Auto,
            filter_tensors_on_load: true,
            selected_device: None,
            kv_cache: None,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
        }
    }

    fn stage_message(pos_start: i32, token_count: i32, positions: Vec<i32>) -> StageWireMessage {
        StageWireMessage {
            kind: WireMessageKind::PrefillEmbd,
            pos_start,
            token_count,
            state: StageStateHeader::new(WireMessageKind::PrefillEmbd, WireActivationDType::F32),
            request_id: 1,
            session_id: 2,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![1; usize::try_from(token_count).unwrap()],
            positions,
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    fn activation_frame(layer_start: i32, layer_end: i32, values: &[f32]) -> ActivationFrame {
        let payload = f32_payload(values);
        ActivationFrame {
            desc: ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F32,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: 0,
                layer_start,
                layer_end,
                token_count: u32::try_from(values.len() / 2).unwrap(),
                sequence_count: 1,
                payload_bytes: u64::try_from(payload.len()).unwrap(),
                flags: 0,
            },
            payload,
        }
    }

    fn f32_payload(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn f32_row(frame: &ActivationFrame, row_index: usize, width: usize) -> Vec<f32> {
        let offset = row_index * width * std::mem::size_of::<f32>();
        frame.payload[offset..offset + width * std::mem::size_of::<f32>()]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }
}
