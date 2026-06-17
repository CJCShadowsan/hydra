use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result, bail};
use skippy_runtime::spd::{
    GgufTokenEmbeddingTable, SpdHeadManifest, SpdLiveCurInRequest, SpdLiveTapRunner,
    SpdLiveTapRunnerConfig, SpdQwen3ForwardCache, SpdQwen3ForwardInput, SpdQwen3Head,
    SpdRollingDraftPlan, SpdRollingObserver, SpdRollingSnapshot, SpdRollingSpeculationRows,
    SpdRollingVerifiedDelta, SpdSafetensorsFile, SpdStageLayerRange, SpdTapInputProjector,
    assemble_spd_live_cur_in_for_positions, plan_hidden_state_taps, required_spd_hf_indices,
    required_spd_hf_indices_for_topology, sliding_spd_row_positions, spd_fixture_cur_in_row_count,
    spd_fixture_row_hf_indices, spd_hf_indices_for_stage_id,
};
use skippy_runtime::{ActivationFrame, RuntimeActivationDType, RuntimeActivationLayout};

use super::*;

mod cache;
mod telemetry;
#[cfg(test)]
mod tests;

use self::{
    cache::{
        SpdInlineTapCache, SpdInlineTapLifecycle, SpdTapRecordOutcome, inline_required_hf_indices,
        retained_tap_prefix_len_for_context_update,
    },
    telemetry::{
        insert_proposal_stats_attrs, insert_rolling_attrs, insert_rolling_speculation_rows_attrs,
        insert_rolling_verified_delta_attrs,
    },
};

pub(super) use self::telemetry::SpdRollingTelemetry;

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
    tap_lifecycle: Arc<Mutex<SpdInlineTapLifecycle>>,
}

pub(super) struct SpdReplayProposalSource {
    pub(super) manifest_path: PathBuf,
    pub(super) model_path: PathBuf,
    pub(super) window: usize,
    top_k: usize,
    row_count: usize,
    row_stage_ids: Vec<i64>,
    row_hf_indices: Vec<Vec<u32>>,
    hidden_size: usize,
    final_norm_weight: Vec<f32>,
    context_tokens: Vec<i32>,
    head: SpdQwen3Head,
    head_cache: Mutex<SpdQwen3ForwardCache>,
    manifest: SpdHeadManifest,
    serving_file: SpdSafetensorsFile,
    tap_projector: SpdTapInputProjector,
    live_taps: SpdLiveTapRunner,
    h0_embeddings: Option<GgufTokenEmbeddingTable>,
    inline_taps: Arc<Mutex<SpdInlineTapCache>>,
    tap_lifecycle: Arc<Mutex<SpdInlineTapLifecycle>>,
    logical_stage_count: usize,
    rolling: SpdRollingObserver,
    last_proposal_stats: SpdProposalSourceStats,
    total_proposal_stats: SpdProposalSourceStats,
    replay_fallback: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct SpdInlineProposal {
    pub(super) token: i32,
    pub(super) logit: Option<f32>,
    pub(super) logit_margin: Option<f32>,
    pub(super) cache_used: bool,
    pub(super) cache_prefix_len: Option<usize>,
    pub(super) tap_source: SpdTapCollectionSource,
    pub(super) tap_collect_ms: f64,
    pub(super) cur_in_ms: f64,
    pub(super) forward_ms: f64,
    pub(super) proposal_rows: SpdProposalRows,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct SpdProposalRows {
    pub(super) row_positions: Vec<i64>,
    pub(super) row_i_stages: Vec<i64>,
    pub(super) evicted_prefix_position: Option<usize>,
    pub(super) newest_position: Option<usize>,
    pub(super) next_draft_position: Option<usize>,
}

impl SpdProposalRows {
    fn from_probe_rows(
        context_len: usize,
        row_positions: &[i64],
        row_i_stages: &[i64],
        rolling_rows: Option<&SpdRollingSpeculationRows>,
    ) -> Result<Self> {
        let newest_position = row_positions
            .last()
            .copied()
            .map(|position| usize::try_from(position).context("negative SPD proposal row position"))
            .transpose()?;
        Ok(Self {
            row_positions: row_positions.to_vec(),
            row_i_stages: row_i_stages.to_vec(),
            evicted_prefix_position: rolling_rows.and_then(|rows| rows.evicted_prefix_position),
            newest_position: rolling_rows
                .map(|rows| rows.newest_position)
                .or(newest_position),
            next_draft_position: rolling_rows
                .map(|rows| rows.next_draft_position)
                .or(Some(context_len)),
        })
    }

    fn from_rolling_rows(rows: &SpdRollingSpeculationRows) -> Result<Self> {
        Ok(Self {
            row_positions: rows
                .row_positions
                .iter()
                .copied()
                .map(i64::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("SPD rolling row position exceeds i64")?,
            row_i_stages: rows
                .row_i_stages
                .iter()
                .copied()
                .map(i64::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("SPD rolling row stage exceeds i64")?,
            evicted_prefix_position: rows.evicted_prefix_position,
            newest_position: Some(rows.newest_position),
            next_draft_position: Some(rows.next_draft_position),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SpdInlineProposalMiss {
    reason: &'static str,
    missing_taps: BTreeMap<u32, Vec<i64>>,
    proposal_rows: Option<SpdProposalRows>,
}

impl SpdInlineProposalMiss {
    fn empty(reason: &'static str) -> Self {
        Self {
            reason,
            missing_taps: BTreeMap::new(),
            proposal_rows: None,
        }
    }

    fn for_rows(
        reason: &'static str,
        rows: &SpdRollingSpeculationRows,
        missing_taps: BTreeMap<u32, Vec<i64>>,
    ) -> Result<Self> {
        Ok(Self {
            reason,
            missing_taps,
            proposal_rows: Some(SpdProposalRows::from_rolling_rows(rows)?),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SpdTapCollectionSource {
    Inline,
    ReplayFallback,
}

impl SpdTapCollectionSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Inline => "inline",
            Self::ReplayFallback => "replay_fallback",
        }
    }
}

#[derive(Debug)]
struct SpdTapCollection {
    taps: BTreeMap<u32, ActivationFrame>,
    source: SpdTapCollectionSource,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct SpdProposalSourceStats {
    requested_limit: usize,
    attempts: usize,
    proposed: usize,
    inline_tap_hits: usize,
    replay_fallbacks: usize,
    cache_hits: usize,
    cache_misses: usize,
    tap_collect_ms: f64,
    cur_in_ms: f64,
    forward_ms: f64,
    last_cache_prefix_len: Option<usize>,
    max_cache_prefix_len: Option<usize>,
}

impl SpdProposalSourceStats {
    fn for_limit(requested_limit: usize) -> Self {
        Self {
            requested_limit,
            ..Self::default()
        }
    }

    fn observe_proposal(&mut self, proposal: &SpdInlineProposal) {
        self.proposed += 1;
        match proposal.tap_source {
            SpdTapCollectionSource::Inline => self.inline_tap_hits += 1,
            SpdTapCollectionSource::ReplayFallback => self.replay_fallbacks += 1,
        }
        if proposal.cache_used {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }
        self.tap_collect_ms += proposal.tap_collect_ms;
        self.cur_in_ms += proposal.cur_in_ms;
        self.forward_ms += proposal.forward_ms;
        if let Some(prefix_len) = proposal.cache_prefix_len {
            self.last_cache_prefix_len = Some(prefix_len);
            self.max_cache_prefix_len =
                Some(self.max_cache_prefix_len.unwrap_or(0).max(prefix_len));
        }
    }

    fn merge(&mut self, window: &Self) {
        self.requested_limit += window.requested_limit;
        self.attempts += window.attempts;
        self.proposed += window.proposed;
        self.inline_tap_hits += window.inline_tap_hits;
        self.replay_fallbacks += window.replay_fallbacks;
        self.cache_hits += window.cache_hits;
        self.cache_misses += window.cache_misses;
        self.tap_collect_ms += window.tap_collect_ms;
        self.cur_in_ms += window.cur_in_ms;
        self.forward_ms += window.forward_ms;
        self.last_cache_prefix_len = window.last_cache_prefix_len.or(self.last_cache_prefix_len);
        if let Some(prefix_len) = window.max_cache_prefix_len {
            self.max_cache_prefix_len =
                Some(self.max_cache_prefix_len.unwrap_or(0).max(prefix_len));
        }
    }
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
        let row_count = spd_fixture_cur_in_row_count(&fixture_file, manifest.topology.hidden_size)?;
        let row_stage_ids = read_spd_row_stage_ids(&fixture_file, row_count)?;
        let row_hf_indices = spd_fixture_row_hf_indices(&fixture_file, row_count)?;
        let tap_projector = SpdTapInputProjector::from_topology(&manifest.topology, &serving_file)
            .context("load SPD tap projection weights")?;
        let required_hf_indices = required_spd_hf_indices_for_topology(&manifest.topology);
        let final_norm_weight = read_spd_final_norm_weight(&fixture_file, hidden_size)?;
        let logical_stage_count = usize::try_from(manifest.topology.num_stages)
            .context("SPD manifest num_stages exceeds usize")?;
        if logical_stage_count == 0 {
            bail!("SPD manifest num_stages must be greater than zero");
        }
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
        let tap_lifecycle = Arc::new(Mutex::new(SpdInlineTapLifecycle::default()));
        Ok(Self {
            manifest_path: manifest_path.to_path_buf(),
            model_path,
            window: args.window,
            top_k: args.top_k,
            row_count,
            row_stage_ids,
            row_hf_indices,
            hidden_size,
            final_norm_weight,
            context_tokens: Vec::new(),
            head_cache: Mutex::new(head.new_forward_cache()),
            head,
            manifest,
            serving_file,
            tap_projector,
            live_taps,
            h0_embeddings,
            inline_taps,
            tap_lifecycle,
            logical_stage_count,
            rolling: SpdRollingObserver::new(logical_stage_count),
            last_proposal_stats: SpdProposalSourceStats::default(),
            total_proposal_stats: SpdProposalSourceStats::default(),
            replay_fallback: args.replay_fallback,
        })
    }

    fn propose_one(&self, context_tokens: &[i32]) -> Result<Option<SpdInlineProposal>> {
        let row_positions = sliding_spd_row_positions(context_tokens.len(), self.row_count)?;
        self.propose_from_row_metadata(
            context_tokens,
            &row_positions,
            &self.row_stage_ids,
            &self.row_hf_indices,
            None,
        )
    }

    fn propose_one_from_rolling_rows(
        &self,
        context_tokens: &[i32],
        rows: &SpdRollingSpeculationRows,
    ) -> Result<Option<SpdInlineProposal>> {
        let rows = self.resolve_rolling_rows(rows)?;
        if rows.row_positions.is_empty()
            || rows.row_positions.len() != rows.row_i_stages.len()
            || rows
                .row_positions
                .iter()
                .any(|position| *position >= context_tokens.len())
        {
            return Ok(None);
        }
        let row_positions = rows
            .row_positions
            .iter()
            .copied()
            .map(i64::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("SPD rolling row position exceeds i64")?;
        let row_stage_ids = rows
            .row_i_stages
            .iter()
            .copied()
            .map(i64::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("SPD rolling row stage exceeds i64")?;
        let row_hf_indices = rows
            .row_i_stages
            .iter()
            .copied()
            .map(|stage_id| {
                u32::try_from(stage_id)
                    .context("SPD rolling row stage exceeds u32")
                    .and_then(|stage_id| {
                        spd_hf_indices_for_stage_id(&self.manifest.topology, stage_id)
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        self.propose_from_row_metadata(
            context_tokens,
            &row_positions,
            &row_stage_ids,
            &row_hf_indices,
            Some(&rows),
        )
    }

    fn propose_from_row_metadata(
        &self,
        context_tokens: &[i32],
        row_positions: &[i64],
        row_stage_ids: &[i64],
        row_hf_indices: &[Vec<u32>],
        rolling_rows: Option<&SpdRollingSpeculationRows>,
    ) -> Result<Option<SpdInlineProposal>> {
        let required_hf_indices = required_spd_hf_indices(row_hf_indices);
        let tap_timer = PhaseTimer::start();
        let Some(tap_collection) = self.collect_taps_for_proposal(
            context_tokens,
            row_positions,
            row_hf_indices,
            &required_hf_indices,
        )?
        else {
            return Ok(None);
        };
        let tap_collect_ms = tap_timer.elapsed_ms();
        let cur_in_timer = PhaseTimer::start();
        let live_rows = assemble_spd_live_cur_in_for_positions(SpdLiveCurInRequest {
            manifest: &self.manifest,
            serving_file: &self.serving_file,
            tap_projector: Some(&self.tap_projector),
            taps: &tap_collection.taps,
            row_positions,
            row_stage_ids,
            row_hf_indices,
            hidden_size: self.hidden_size,
        })?;
        let cur_in_ms = cur_in_timer.elapsed_ms();
        let forward_timer = PhaseTimer::start();
        let (topk, cache_used, cache_prefix_len) =
            self.forward_head_for_rows(context_tokens, row_positions, live_rows.cur_in)?;
        let forward_ms = forward_timer.elapsed_ms();
        let mut proposal = spd_inline_proposal_from_topk(&topk)?;
        proposal.cache_used = cache_used;
        proposal.cache_prefix_len = cache_prefix_len;
        proposal.tap_source = tap_collection.source;
        proposal.tap_collect_ms = tap_collect_ms;
        proposal.cur_in_ms = cur_in_ms;
        proposal.forward_ms = forward_ms;
        proposal.proposal_rows = SpdProposalRows::from_probe_rows(
            context_tokens.len(),
            row_positions,
            row_stage_ids,
            rolling_rows,
        )?;
        Ok(Some(proposal))
    }

    fn forward_head_for_rows(
        &self,
        context_tokens: &[i32],
        row_positions: &[i64],
        cur_in: Vec<f32>,
    ) -> Result<(
        skippy_runtime::spd::SpdQwen3FixtureTopK,
        bool,
        Option<usize>,
    )> {
        let input = SpdQwen3ForwardInput {
            cur_in,
            seq_len: row_positions.len(),
            position_ids: row_positions.to_vec(),
            fixed_stage_ids: None,
            final_norm_weight: self.final_norm_weight.clone(),
        };
        let mut cache = self
            .head_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD Qwen head cache lock poisoned"))?;
        if self.prefill_head_cache_for_rows(context_tokens, row_positions, &mut cache)? {
            let cache_prefix_len = cache.cached_prefix_len();
            let topk = self
                .head
                .forward_with_cache(input, self.top_k, &mut cache)?;
            return Ok((topk, true, Some(cache_prefix_len)));
        }
        Ok((self.head.forward(input, self.top_k)?, false, None))
    }

    fn prefill_head_cache_for_rows(
        &self,
        context_tokens: &[i32],
        row_positions: &[i64],
        cache: &mut SpdQwen3ForwardCache,
    ) -> Result<bool> {
        let Some(min_position) = row_positions.iter().copied().min() else {
            return Ok(false);
        };
        let min_position =
            usize::try_from(min_position).context("negative SPD rolling row position")?;
        let cached_prefix_len = cache.cached_prefix_len().min(min_position);
        if cached_prefix_len >= min_position {
            return Ok(true);
        }
        let prefix_positions = (cached_prefix_len..min_position)
            .map(i64::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("SPD prefix position exceeds i64")?;
        self.prefill_head_cache_positions(context_tokens, &prefix_positions, cache)
    }

    fn prefill_head_cache_positions(
        &self,
        context_tokens: &[i32],
        row_positions: &[i64],
        cache: &mut SpdQwen3ForwardCache,
    ) -> Result<bool> {
        if row_positions.is_empty() {
            return Ok(true);
        }
        let stage_id = self.manifest.topology.num_stages;
        let row_stage_ids = vec![i64::from(stage_id); row_positions.len()];
        let qwen_stage_id = usize::try_from(stage_id).context("SPD stage count exceeds usize")?;
        let stage_hf_indices = spd_hf_indices_for_stage_id(&self.manifest.topology, stage_id)?;
        let row_hf_indices = vec![stage_hf_indices; row_positions.len()];
        let required_hf_indices = required_spd_hf_indices(&row_hf_indices);
        let required_inline_hf_indices = inline_required_hf_indices(&required_hf_indices);
        let Some(mut taps) =
            self.complete_inline_taps(row_positions, &required_inline_hf_indices)?
        else {
            return Ok(false);
        };
        if required_hf_indices.contains(&0) {
            taps.insert(0, self.collect_h0_tap(context_tokens, row_positions)?);
        }
        let live_rows = assemble_spd_live_cur_in_for_positions(SpdLiveCurInRequest {
            manifest: &self.manifest,
            serving_file: &self.serving_file,
            tap_projector: Some(&self.tap_projector),
            taps: &taps,
            row_positions,
            row_stage_ids: &row_stage_ids,
            row_hf_indices: &row_hf_indices,
            hidden_size: self.hidden_size,
        })?;
        self.head.prefill_cache(
            SpdQwen3ForwardInput {
                cur_in: live_rows.cur_in,
                seq_len: row_positions.len(),
                position_ids: row_positions.to_vec(),
                fixed_stage_ids: Some(vec![qwen_stage_id; row_positions.len()]),
                final_norm_weight: self.final_norm_weight.clone(),
            },
            cache,
        )?;
        Ok(true)
    }

    fn collect_taps_for_proposal(
        &self,
        context_tokens: &[i32],
        row_positions: &[i64],
        row_hf_indices: &[Vec<u32>],
        required_hf_indices: &[u32],
    ) -> Result<Option<SpdTapCollection>> {
        let required_inline_hf_indices = inline_required_hf_indices(required_hf_indices);
        if let Some(mut taps) = self.complete_inline_taps_for_rows(
            row_positions,
            row_hf_indices,
            &required_inline_hf_indices,
        )? {
            if required_hf_indices.contains(&0) {
                taps.insert(0, self.collect_h0_tap(context_tokens, row_positions)?);
            }
            return Ok(Some(SpdTapCollection {
                taps,
                source: SpdTapCollectionSource::Inline,
            }));
        }
        if !self.replay_fallback {
            return Ok(None);
        }
        let mut taps = self.live_taps.collect_taps(context_tokens)?;
        self.overlay_inline_taps_for_rows(
            &mut taps,
            row_positions,
            row_hf_indices,
            &required_inline_hf_indices,
        )?;
        Ok(Some(SpdTapCollection {
            taps,
            source: SpdTapCollectionSource::ReplayFallback,
        }))
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

    pub(super) fn propose_inline_for_current_context(
        &mut self,
        current: i32,
    ) -> Result<Option<SpdInlineProposal>> {
        let proposal = self.propose_inline_for_current_context_untracked(current)?;
        self.record_inline_probe_attempt(proposal.as_ref());
        Ok(proposal)
    }

    fn propose_inline_for_current_context_untracked(
        &self,
        current: i32,
    ) -> Result<Option<SpdInlineProposal>> {
        if self.context_tokens.last().copied() != Some(current) {
            return Ok(None);
        }
        if let Some(rows) = self.resolved_rolling_speculation_rows()?
            && let Some(proposal) =
                self.propose_one_from_rolling_rows(&self.context_tokens, &rows)?
        {
            return Ok(Some(proposal));
        }
        self.propose_one(&self.context_tokens)
    }

    pub(super) fn inline_proposal_miss_for_current_context(
        &self,
        current: i32,
    ) -> Result<Option<SpdInlineProposalMiss>> {
        if self.context_tokens.last().copied() != Some(current) {
            return Ok(Some(SpdInlineProposalMiss::empty(
                "context_current_mismatch",
            )));
        }
        if let Some(rows) = self.resolved_rolling_speculation_rows()? {
            return self.inline_proposal_miss_for_rolling_rows(&rows);
        }
        Ok(Some(SpdInlineProposalMiss::empty(
            "rolling_rows_unavailable",
        )))
    }

    fn inline_proposal_miss_for_rolling_rows(
        &self,
        rows: &SpdRollingSpeculationRows,
    ) -> Result<Option<SpdInlineProposalMiss>> {
        if rows.row_positions.is_empty() || rows.row_positions.len() != rows.row_i_stages.len() {
            return Ok(Some(SpdInlineProposalMiss::empty("rolling_rows_invalid")));
        }
        if rows
            .row_positions
            .iter()
            .any(|position| *position >= self.context_tokens.len())
        {
            return Ok(Some(SpdInlineProposalMiss::empty(
                "rolling_rows_outside_context",
            )));
        }
        let row_positions = rows
            .row_positions
            .iter()
            .copied()
            .map(i64::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("SPD rolling row position exceeds i64")?;
        let row_hf_indices = rows
            .row_i_stages
            .iter()
            .copied()
            .map(|stage_id| {
                u32::try_from(stage_id)
                    .context("SPD rolling row stage exceeds u32")
                    .and_then(|stage_id| {
                        spd_hf_indices_for_stage_id(&self.manifest.topology, stage_id)
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let required_hf_indices = required_spd_hf_indices(&row_hf_indices);
        let required_inline_hf_indices = inline_required_hf_indices(&required_hf_indices);
        let missing_taps = self
            .inline_taps
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD inline tap cache lock poisoned"))?
            .missing_required_rows_for_row_hf_indices(
                &row_positions,
                &row_hf_indices,
                &required_inline_hf_indices,
            )?;
        if missing_taps.is_empty() {
            return Ok(Some(SpdInlineProposalMiss::for_rows(
                "rolling_rows_complete_without_proposal",
                rows,
                BTreeMap::new(),
            )?));
        }
        Ok(Some(SpdInlineProposalMiss::for_rows(
            "missing_inline_taps",
            rows,
            missing_taps,
        )?))
    }

    fn record_inline_probe_attempt(&mut self, proposal: Option<&SpdInlineProposal>) {
        let mut stats = SpdProposalSourceStats::for_limit(1);
        stats.attempts = 1;
        if let Some(proposal) = proposal {
            stats.observe_proposal(proposal);
        }
        self.last_proposal_stats = stats;
        self.total_proposal_stats.merge(&self.last_proposal_stats);
    }

    pub(super) fn advance_to_accepted_context(&mut self, context_tokens: &[i32]) -> Result<()> {
        if self.context_tokens.starts_with(context_tokens) {
            self.tap_lifecycle
                .lock()
                .map_err(|_| anyhow::anyhow!("SPD tap lifecycle lock poisoned"))?
                .accept_context_len(context_tokens.len());
            return Ok(());
        }
        let accepted_extension = context_tokens.starts_with(&self.context_tokens);
        let retained_prefix_len =
            retained_tap_prefix_len_for_context_update(&self.context_tokens, context_tokens, true);
        self.context_tokens = context_tokens.to_vec();
        self.rolling.advance_to_accepted_context(context_tokens);
        if !accepted_extension {
            self.inline_taps
                .lock()
                .map_err(|_| anyhow::anyhow!("SPD inline tap cache lock poisoned"))?
                .retain_positions_before(retained_prefix_len);
        }
        self.head_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD Qwen head cache lock poisoned"))?
            .crop_to_position(
                i64::try_from(retained_prefix_len).context("SPD retained prefix exceeds i64")?,
            );
        self.tap_lifecycle
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD tap lifecycle lock poisoned"))?
            .accept_context_len(context_tokens.len());
        Ok(())
    }

    pub(super) fn reset_to_verified_context(&mut self, context_tokens: &[i32]) -> Result<()> {
        let retained_prefix_len =
            retained_tap_prefix_len_for_context_update(&self.context_tokens, context_tokens, false);
        self.context_tokens = context_tokens.to_vec();
        self.rolling.advance_to_accepted_context(context_tokens);
        self.inline_taps
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD inline tap cache lock poisoned"))?
            .retain_positions_before(retained_prefix_len);
        self.head_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD Qwen head cache lock poisoned"))?
            .crop_to_position(
                i64::try_from(retained_prefix_len).context("SPD retained prefix exceeds i64")?,
            );
        self.tap_lifecycle
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD tap lifecycle lock poisoned"))?
            .reset_context_len(context_tokens.len());
        Ok(())
    }

    pub(super) fn mark_pending_verify_tap_positions(
        &mut self,
        pos_start: usize,
        token_count: usize,
    ) -> Result<()> {
        let positions = (0..token_count)
            .map(|offset| {
                pos_start
                    .checked_add(offset)
                    .context("SPD verify tap position overflow")
            })
            .collect::<Result<Vec<_>>>()?;
        self.tap_lifecycle
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD tap lifecycle lock poisoned"))?
            .mark_pending_future_positions(positions);
        Ok(())
    }

    pub(super) fn observe_primary_verify_span(
        &mut self,
        first_target_position: usize,
        draft_tokens: &[i32],
        target_tokens: &[i32],
    ) -> Result<Option<SpdRollingTelemetry>> {
        let mut telemetry = None;
        for (offset, (draft, target)) in draft_tokens.iter().zip(target_tokens).enumerate() {
            let target_position = first_target_position
                .checked_add(offset)
                .context("SPD primary verify rolling position overflow")?;
            telemetry = Some(self.observe_rolling_probe(target_position, *target, Some(*draft))?);
        }
        Ok(telemetry)
    }

    pub(super) fn observe_rolling_target_token(
        &mut self,
        target_position: usize,
        target: i32,
    ) -> Result<SpdRollingTelemetry> {
        let snapshot = self.rolling.observe_target(target_position, target)?;
        let speculation_rows = self.resolved_rolling_speculation_rows()?;
        let verified_delta = self.rolling.take_verified_delta();
        Ok(SpdRollingTelemetry {
            snapshot,
            speculation_rows,
            verified_delta,
        })
    }

    pub(super) fn observe_rolling_probe(
        &mut self,
        target_position: usize,
        target: i32,
        proposed: Option<i32>,
    ) -> Result<SpdRollingTelemetry> {
        let snapshot = self
            .rolling
            .observe_probe(target_position, target, proposed)?;
        let speculation_rows = self.resolved_rolling_speculation_rows()?;
        let verified_delta = self.rolling.take_verified_delta();
        Ok(SpdRollingTelemetry {
            snapshot,
            speculation_rows,
            verified_delta,
        })
    }

    pub(super) fn insert_rolling_attrs(&self, attrs: &mut BTreeMap<String, Value>) {
        insert_rolling_attrs(&self.rolling.snapshot(), attrs);
    }

    pub(super) fn insert_last_proposal_stats_attrs(&self, attrs: &mut BTreeMap<String, Value>) {
        insert_proposal_stats_attrs("window", &self.last_proposal_stats, attrs);
    }

    pub(super) fn insert_total_proposal_stats_attrs(&self, attrs: &mut BTreeMap<String, Value>) {
        insert_proposal_stats_attrs("total", &self.total_proposal_stats, attrs);
    }

    pub(super) fn insert_rolling_telemetry_attrs(
        attrs: &mut BTreeMap<String, Value>,
        rolling: &SpdRollingTelemetry,
    ) {
        insert_rolling_attrs(&rolling.snapshot, attrs);
        if let Some(rows) = rolling.speculation_rows.as_ref() {
            insert_rolling_speculation_rows_attrs(rows, attrs);
        }
        if let Some(delta) = rolling.verified_delta.as_ref() {
            insert_rolling_verified_delta_attrs(delta, attrs);
        }
    }

    fn resolved_rolling_speculation_rows(&self) -> Result<Option<SpdRollingSpeculationRows>> {
        self.rolling
            .speculation_rows()
            .map(|rows| self.resolve_rolling_rows(&rows))
            .transpose()
    }

    fn resolve_rolling_rows(
        &self,
        rows: &SpdRollingSpeculationRows,
    ) -> Result<SpdRollingSpeculationRows> {
        let mut resolved = rows.clone();
        resolved.row_i_stages = {
            let inline_taps = self
                .inline_taps
                .lock()
                .map_err(|_| anyhow::anyhow!("SPD inline tap cache lock poisoned"))?;
            resolve_rolling_row_stage_roles(&self.manifest.topology, rows, &inline_taps)?
        };
        Ok(resolved)
    }

    fn complete_inline_taps(
        &self,
        row_positions: &[i64],
        required_hf_indices: &[u32],
    ) -> Result<Option<BTreeMap<u32, ActivationFrame>>> {
        let inline_taps = self
            .inline_taps
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD inline tap cache lock poisoned"))?;
        inline_taps.complete_frames(
            row_positions,
            &non_h0_required_hf_indices(required_hf_indices),
            self.hidden_size,
        )
    }

    fn complete_inline_taps_for_rows(
        &self,
        row_positions: &[i64],
        row_hf_indices: &[Vec<u32>],
        required_hf_indices: &[u32],
    ) -> Result<Option<BTreeMap<u32, ActivationFrame>>> {
        let inline_taps = self
            .inline_taps
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD inline tap cache lock poisoned"))?;
        inline_taps.complete_frames_for_row_hf_indices(
            row_positions,
            row_hf_indices,
            &non_h0_required_hf_indices(required_hf_indices),
            self.hidden_size,
        )
    }

    fn overlay_inline_taps_for_rows(
        &self,
        taps: &mut BTreeMap<u32, ActivationFrame>,
        row_positions: &[i64],
        row_hf_indices: &[Vec<u32>],
        required_hf_indices: &[u32],
    ) -> Result<()> {
        let mut inline_taps = self
            .inline_taps
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD inline tap cache lock poisoned"))?;
        inline_taps.overlay_complete_frames_for_row_hf_indices(
            taps,
            row_positions,
            row_hf_indices,
            required_hf_indices,
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
        let retained_prefix_len =
            retained_tap_prefix_len_for_context_update(&self.context_tokens, context_tokens, false);
        self.context_tokens = context_tokens.to_vec();
        self.rolling = SpdRollingObserver::new(self.logical_stage_count);
        self.last_proposal_stats = SpdProposalSourceStats::default();
        self.total_proposal_stats = SpdProposalSourceStats::default();
        self.inline_taps
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD inline tap cache lock poisoned"))?
            .retain_positions_before(retained_prefix_len);
        self.head_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD Qwen head cache lock poisoned"))?
            .crop_to_position(
                i64::try_from(retained_prefix_len).context("SPD retained prefix exceeds i64")?,
            );
        self.tap_lifecycle
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD tap lifecycle lock poisoned"))?
            .reset_context_len(context_tokens.len());
        Ok(())
    }

    fn propose(&mut self, current: i32, max_tokens: usize) -> Result<Vec<i32>> {
        if self.context_tokens.last().copied() != Some(current) {
            self.context_tokens.push(current);
        }
        if self.context_tokens.len() < self.row_count {
            self.last_proposal_stats = SpdProposalSourceStats::for_limit(max_tokens);
            return Ok(Vec::new());
        }
        let mut proposals = Vec::with_capacity(max_tokens);
        let mut stats = SpdProposalSourceStats::for_limit(max_tokens);
        let mut rolling_plan = self.rolling.draft_plan();
        for _ in 0..max_tokens {
            stats.attempts += 1;
            let Some(proposal) = self.propose_one_for_primary(&rolling_plan)? else {
                break;
            };
            stats.observe_proposal(&proposal);
            proposals.push(proposal.token);
            self.context_tokens.push(proposal.token);
            if let Some(plan) = rolling_plan.as_mut() {
                plan.insert_draft(proposal.token);
            }
        }
        self.last_proposal_stats = stats;
        self.total_proposal_stats.merge(&self.last_proposal_stats);
        Ok(proposals)
    }
}

impl SpdReplayProposalSource {
    pub(super) fn logical_stage_count(&self) -> usize {
        self.logical_stage_count
    }

    fn propose_one_for_primary(
        &self,
        rolling_plan: &Option<SpdRollingDraftPlan>,
    ) -> Result<Option<SpdInlineProposal>> {
        let Some(plan) = rolling_plan.as_ref() else {
            return self.propose_one(&self.context_tokens);
        };
        let Some(rows) = plan.speculation_rows() else {
            return Ok(None);
        };
        self.propose_one_from_rolling_rows(&self.context_tokens, &rows)
    }
}

fn spd_inline_proposal_from_topk(
    topk: &skippy_runtime::spd::SpdQwen3FixtureTopK,
) -> Result<SpdInlineProposal> {
    let token = topk
        .token_ids
        .first()
        .copied()
        .context("SPD head returned no proposal token")
        .and_then(|token| i32::try_from(token).context("SPD proposal token exceeds i32"))?;
    let logit = topk.logits.first().copied();
    let logit_margin = match (topk.logits.first(), topk.logits.get(1)) {
        (Some(first), Some(second)) => Some(*first - *second),
        _ => None,
    };
    Ok(SpdInlineProposal {
        token,
        logit,
        logit_margin,
        cache_used: false,
        cache_prefix_len: None,
        tap_source: SpdTapCollectionSource::Inline,
        tap_collect_ms: 0.0,
        cur_in_ms: 0.0,
        forward_ms: 0.0,
        proposal_rows: SpdProposalRows::default(),
    })
}

pub(super) fn open_spd_replay_source(
    args: SpdReplayOpenArgs<'_>,
) -> Result<Option<SpdReplayProposalState>> {
    match (args.manifest_path, args.fixture_path) {
        (None, None) => Ok(None),
        (Some(_), Some(_)) => {
            let source = SpdReplayProposalSource::open(args)?;
            let taps = source.inline_taps.clone();
            let tap_lifecycle = source.tap_lifecycle.clone();
            Ok(Some(SpdReplayProposalState {
                source: Arc::new(Mutex::new(source)),
                taps,
                tap_lifecycle,
            }))
        }
        _ => bail!("--openai-spd-manifest and --openai-spd-fixture must be set together"),
    }
}

impl SpdReplayProposalState {
    pub(super) fn mark_pending_optimistic_tap_position(&self, position: usize) -> Result<()> {
        self.tap_lifecycle
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD tap lifecycle lock poisoned"))?
            .mark_pending_optimistic_position(position);
        Ok(())
    }

    fn record_returned_tap(&self, tap: &StageReplySpdTap) -> Result<SpdTapRecordOutcome> {
        let decision = self
            .tap_lifecycle
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD tap lifecycle lock poisoned"))?
            .record_decision(tap)?;
        if let Some(ignored) = decision.ignored {
            return Ok(SpdTapRecordOutcome::Ignored(ignored));
        }
        let record = self
            .taps
            .lock()
            .map_err(|_| anyhow::anyhow!("SPD inline tap cache lock poisoned"))?
            .record_returned_tap(tap)?;
        Ok(SpdTapRecordOutcome::Recorded(record))
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
        spd_source: Option<&mut SpdReplayProposalSource>,
        current: i32,
    ) -> Result<SpdPredictionReturn> {
        self.recv_spd_aware_prediction_return_with_probe_action(
            request,
            expected,
            spd_source,
            current,
            SpdInlineProbePhase::PreTargetReply,
            None::<fn(&StageOpenAiBackend, &SpdInlineProbe) -> Result<()>>,
        )
    }

    pub(super) fn recv_spd_aware_prediction_return_for_origin(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        expected: WireReplyKind,
        expected_origin: PredictionReturnOrigin,
    ) -> Result<SpdPredictionReturn> {
        self.recv_spd_aware_prediction_return_with_wait_action(
            request,
            SpdPredictionReturnWait {
                expected,
                expected_origin: Some(expected_origin),
                spd_source: None,
                current: 0,
                probe_phase: SpdInlineProbePhase::PreTargetReply,
            },
            None::<fn(&StageOpenAiBackend, &SpdInlineProbe) -> Result<()>>,
        )
    }

    pub(super) fn recv_spd_aware_prediction_return_with_probe_action<F>(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        expected: WireReplyKind,
        spd_source: Option<&mut SpdReplayProposalSource>,
        current: i32,
        probe_phase: SpdInlineProbePhase,
        on_pre_target_probe: Option<F>,
    ) -> Result<SpdPredictionReturn>
    where
        F: FnOnce(&StageOpenAiBackend, &SpdInlineProbe) -> Result<()>,
    {
        self.recv_spd_aware_prediction_return_with_wait_action(
            request,
            SpdPredictionReturnWait {
                expected,
                expected_origin: None,
                spd_source,
                current,
                probe_phase,
            },
            on_pre_target_probe,
        )
    }

    pub(super) fn recv_spd_aware_prediction_return_with_wait_action<F>(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        mut wait: SpdPredictionReturnWait<'_>,
        mut on_pre_target_probe: Option<F>,
    ) -> Result<SpdPredictionReturn>
    where
        F: FnOnce(&StageOpenAiBackend, &SpdInlineProbe) -> Result<()>,
    {
        let receiver = request
            .prediction_return
            .as_ref()
            .context("missing direct prediction return receiver")?;
        let mut pre_target_probe = None;
        let mut wait_after_probe_timer = None;
        loop {
            let item = if let Some(origin) = wait.expected_origin {
                receiver.recv_item_matching(|item| {
                    item.reply.kind == WireReplyKind::SpdTap
                        || item.matches_origin(wait.expected, origin)
                })?
            } else {
                receiver.recv_item()?
            };
            let reply = item.reply;
            if reply.kind == WireReplyKind::SpdTap {
                let trigger_hf_index = reply.spd_tap.as_ref().map(|tap| tap.hf_index);
                self.record_spd_direct_return_tap(request, &reply);
                if pre_target_probe.is_none()
                    && let Some(spd) = wait.spd_source.as_mut()
                {
                    let probe_timer = PhaseTimer::start();
                    let proposal = spd.propose_inline_for_current_context(wait.current)?;
                    let elapsed_ms = probe_timer.elapsed_ms();
                    if let Some(proposal) = proposal {
                        pre_target_probe = Some(SpdInlineProbe::from_proposal(
                            wait.probe_phase,
                            Some(&proposal),
                            elapsed_ms,
                            0.0,
                            trigger_hf_index,
                        ));
                        wait_after_probe_timer = Some(PhaseTimer::start());
                        if let (Some(callback), Some(probe)) =
                            (on_pre_target_probe.take(), pre_target_probe.as_ref())
                        {
                            callback(self, probe)?;
                        }
                    } else {
                        let proposal_miss =
                            spd.inline_proposal_miss_for_current_context(wait.current)?;
                        self.emit_spd_inline_probe_miss(
                            request,
                            wait.probe_phase,
                            wait.current,
                            trigger_hf_index,
                            elapsed_ms,
                            proposal_miss.as_ref(),
                        );
                    }
                }
                continue;
            }
            if reply.kind != wait.expected {
                bail!(
                    "expected {:?} direct prediction return, got {:?}",
                    wait.expected,
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
        rolling: Option<&SpdRollingTelemetry>,
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
            "llama_stage.spd_inline_probe_proposed_logit".to_string(),
            json!(probe.proposed_logit),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_logit_margin".to_string(),
            json!(probe.proposed_logit_margin),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_cache_used".to_string(),
            json!(probe.cache_used),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_cache_prefix_len".to_string(),
            json!(probe.cache_prefix_len),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_tap_source".to_string(),
            json!(probe.tap_source.map(SpdTapCollectionSource::as_str)),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_tap_collect_ms".to_string(),
            json!(probe.tap_collect_ms),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_cur_in_ms".to_string(),
            json!(probe.cur_in_ms),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_forward_ms".to_string(),
            json!(probe.forward_ms),
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
        attrs.insert(
            "llama_stage.spd_inline_probe_proposal_row_positions".to_string(),
            json!(&probe.proposal_rows.row_positions),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_proposal_row_i_stages".to_string(),
            json!(&probe.proposal_rows.row_i_stages),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_proposal_row_evicted_prefix_position".to_string(),
            json!(probe.proposal_rows.evicted_prefix_position),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_proposal_row_newest_position".to_string(),
            json!(probe.proposal_rows.newest_position),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_proposal_row_next_draft_position".to_string(),
            json!(probe.proposal_rows.next_draft_position),
        );
        if let Some(miss) = probe.proposal_miss.as_ref() {
            attrs.insert(
                "llama_stage.spd_inline_probe_miss_reason".to_string(),
                json!(miss.reason),
            );
            attrs.insert(
                "llama_stage.spd_inline_probe_missing_taps".to_string(),
                json!(&miss.missing_taps),
            );
        }
        if let Some(rolling) = rolling {
            insert_rolling_attrs(&rolling.snapshot, &mut attrs);
            if let Some(rows) = rolling.speculation_rows.as_ref() {
                insert_rolling_speculation_rows_attrs(rows, &mut attrs);
            }
            if let Some(delta) = rolling.verified_delta.as_ref() {
                insert_rolling_verified_delta_attrs(delta, &mut attrs);
            }
        }
        self.telemetry
            .emit_debug("stage.openai_spd_inline_probe", attrs);
        probe.elapsed_ms
    }

    fn emit_spd_inline_probe_miss(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        phase: SpdInlineProbePhase,
        current: i32,
        trigger_hf_index: Option<u32>,
        elapsed_ms: f64,
        proposal_miss: Option<&SpdInlineProposalMiss>,
    ) {
        if !self.telemetry.is_debug_enabled() {
            return;
        }
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert(
            "llama_stage.spd_inline_probe_phase".to_string(),
            json!(phase.as_str()),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_ready".to_string(),
            json!(false),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_current_token".to_string(),
            json!(current),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_trigger_hf_index".to_string(),
            json!(trigger_hf_index),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_elapsed_ms".to_string(),
            json!(elapsed_ms),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_miss_reason".to_string(),
            json!(proposal_miss.map(|miss| miss.reason)),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_missing_taps".to_string(),
            json!(proposal_miss.map(|miss| &miss.missing_taps)),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_proposal_row_positions".to_string(),
            json!(
                proposal_miss
                    .and_then(|miss| miss.proposal_rows.as_ref().map(|rows| &rows.row_positions))
            ),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_proposal_row_i_stages".to_string(),
            json!(
                proposal_miss
                    .and_then(|miss| miss.proposal_rows.as_ref().map(|rows| &rows.row_i_stages))
            ),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_proposal_row_evicted_prefix_position".to_string(),
            json!(
                proposal_miss
                    .and_then(|miss| miss.proposal_rows.as_ref())
                    .and_then(|rows| rows.evicted_prefix_position)
            ),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_proposal_row_newest_position".to_string(),
            json!(
                proposal_miss
                    .and_then(|miss| miss.proposal_rows.as_ref())
                    .and_then(|rows| rows.newest_position)
            ),
        );
        attrs.insert(
            "llama_stage.spd_inline_probe_proposal_row_next_draft_position".to_string(),
            json!(
                proposal_miss
                    .and_then(|miss| miss.proposal_rows.as_ref())
                    .and_then(|rows| rows.next_draft_position)
            ),
        );
        self.telemetry
            .emit_debug("stage.openai_spd_inline_probe_miss", attrs);
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
                    "llama_stage.spd_inline_tap_positions".to_string(),
                    json!(&record.positions),
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
        let Some(tap) = reply.spd_tap.as_ref() else {
            self.emit_missing_spd_tap_payload(request);
            return;
        };
        self.record_spd_direct_tap_payload(request, tap);
    }

    fn emit_missing_spd_tap_payload(&self, request: &EmbeddedStageZeroGeneration<'_>) {
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert(
            "llama_stage.spd_inline_tap_error".to_string(),
            json!("missing SPD tap reply payload"),
        );
        self.telemetry
            .emit_debug("stage.openai_spd_tap_record_failed", attrs);
    }

    fn record_spd_direct_tap_payload(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        tap: &StageReplySpdTap,
    ) {
        let Some(spd) = request.spd.as_ref() else {
            return;
        };
        let outcome = spd.record_returned_tap(tap);
        match outcome {
            Ok(SpdTapRecordOutcome::Recorded(record)) => {
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
                    "llama_stage.spd_inline_tap_positions".to_string(),
                    json!(&record.positions),
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
            Ok(SpdTapRecordOutcome::Ignored(ignored)) => {
                let mut attrs = self.openai_attrs(request.ids);
                attrs.insert(
                    "llama_stage.spd_inline_tap_hf_index".to_string(),
                    json!(tap.hf_index),
                );
                attrs.insert(
                    "llama_stage.spd_inline_tap_ignored_reason".to_string(),
                    json!(ignored.reason),
                );
                attrs.insert(
                    "llama_stage.spd_inline_tap_positions".to_string(),
                    json!(ignored.positions),
                );
                attrs.insert(
                    "llama_stage.spd_inline_tap_accepted_context_len".to_string(),
                    json!(ignored.accepted_context_len),
                );
                attrs.insert(
                    "llama_stage.spd_inline_tap_pending_position".to_string(),
                    json!(ignored.pending_positions.first().copied()),
                );
                attrs.insert(
                    "llama_stage.spd_inline_tap_pending_positions".to_string(),
                    json!(ignored.pending_positions),
                );
                self.telemetry
                    .emit_debug("stage.openai_spd_tap_ignored", attrs);
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

pub(super) struct SpdPredictionReturnWait<'a> {
    pub(super) expected: WireReplyKind,
    pub(super) expected_origin: Option<PredictionReturnOrigin>,
    pub(super) spd_source: Option<&'a mut SpdReplayProposalSource>,
    pub(super) current: i32,
    pub(super) probe_phase: SpdInlineProbePhase,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct SpdInlineProbe {
    pub(super) phase: SpdInlineProbePhase,
    pub(super) proposed: Option<i32>,
    pub(super) proposed_logit: Option<f32>,
    pub(super) proposed_logit_margin: Option<f32>,
    pub(super) cache_used: bool,
    pub(super) cache_prefix_len: Option<usize>,
    pub(super) tap_source: Option<SpdTapCollectionSource>,
    pub(super) tap_collect_ms: f64,
    pub(super) cur_in_ms: f64,
    pub(super) forward_ms: f64,
    pub(super) elapsed_ms: f64,
    pub(super) target_wait_after_probe_ms: f64,
    pub(super) trigger_hf_index: Option<u32>,
    pub(super) proposal_rows: SpdProposalRows,
    proposal_miss: Option<SpdInlineProposalMiss>,
}

impl SpdInlineProbe {
    pub(super) fn from_proposal(
        phase: SpdInlineProbePhase,
        proposal: Option<&SpdInlineProposal>,
        elapsed_ms: f64,
        target_wait_after_probe_ms: f64,
        trigger_hf_index: Option<u32>,
    ) -> Self {
        Self {
            phase,
            proposed: proposal.map(|proposal| proposal.token),
            proposed_logit: proposal.and_then(|proposal| proposal.logit),
            proposed_logit_margin: proposal.and_then(|proposal| proposal.logit_margin),
            cache_used: proposal
                .map(|proposal| proposal.cache_used)
                .unwrap_or(false),
            cache_prefix_len: proposal.and_then(|proposal| proposal.cache_prefix_len),
            tap_source: proposal.map(|proposal| proposal.tap_source),
            tap_collect_ms: proposal
                .map(|proposal| proposal.tap_collect_ms)
                .unwrap_or(0.0),
            cur_in_ms: proposal.map(|proposal| proposal.cur_in_ms).unwrap_or(0.0),
            forward_ms: proposal.map(|proposal| proposal.forward_ms).unwrap_or(0.0),
            elapsed_ms,
            target_wait_after_probe_ms,
            trigger_hf_index,
            proposal_rows: proposal
                .map(|proposal| proposal.proposal_rows.clone())
                .unwrap_or_default(),
            proposal_miss: None,
        }
    }

    pub(super) fn with_proposal_miss(mut self, miss: Option<SpdInlineProposalMiss>) -> Self {
        if self.proposed.is_none() {
            self.proposal_miss = miss;
        }
        self
    }

    pub(super) fn allows_optimistic_decode(&self, min_logit_margin: Option<f32>) -> bool {
        match min_logit_margin {
            Some(minimum) => self
                .proposed_logit_margin
                .is_some_and(|margin| margin >= minimum),
            None => self.proposed.is_some(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SpdInlineProbePhase {
    PreTargetReply,
    PostTargetReply,
    OptimisticCommit,
}

impl SpdInlineProbePhase {
    fn as_str(self) -> &'static str {
        match self {
            Self::PreTargetReply => "pre_target_reply",
            Self::PostTargetReply => "post_target_reply",
            Self::OptimisticCommit => "optimistic_commit",
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

fn resolve_rolling_row_stage_roles(
    topology: &skippy_runtime::spd::SpdHeadTopology,
    rows: &SpdRollingSpeculationRows,
    inline_taps: &SpdInlineTapCache,
) -> Result<Vec<usize>> {
    let has_evicted_prefix = rows.evicted_prefix_position.is_some();
    rows.row_positions
        .iter()
        .copied()
        .zip(rows.row_i_stages.iter().copied())
        .map(|(position, nominal)| {
            resolve_rolling_row_stage_role(
                topology,
                position,
                nominal,
                has_evicted_prefix,
                inline_taps,
            )
        })
        .collect()
}

fn resolve_rolling_row_stage_role(
    topology: &skippy_runtime::spd::SpdHeadTopology,
    position: usize,
    nominal: usize,
    has_evicted_prefix: bool,
    inline_taps: &SpdInlineTapCache,
) -> Result<usize> {
    let stage_count =
        usize::try_from(topology.num_stages).context("SPD num_stages exceeds usize")?;
    if nominal > stage_count {
        bail!("SPD rolling row stage {nominal} exceeds manifest stage count {stage_count}");
    }
    if nominal == 0 || !topology.trained_with_use_deepest {
        return Ok(nominal);
    }
    if nominal == stage_count {
        return Ok(stage_count);
    }
    let deepest_fused = if has_evicted_prefix {
        stage_count.saturating_sub(1).max(1)
    } else {
        stage_count
    };
    let search_hi = if nominal == stage_count {
        stage_count
    } else {
        deepest_fused
    };
    for stage_id in (1..=search_hi).rev() {
        let hf_indices = spd_hf_indices_for_stage_id(
            topology,
            u32::try_from(stage_id).context("SPD stage id exceeds u32")?,
        )?;
        let inline_hf_indices = inline_required_hf_indices(&hf_indices);
        if inline_taps.has_rows_for_hf_indices(position, &inline_hf_indices) {
            return Ok(stage_id);
        }
    }
    Ok(nominal)
}
