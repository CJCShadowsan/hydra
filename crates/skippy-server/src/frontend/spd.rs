use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result, bail};
use skippy_runtime::spd::{
    SpdHeadManifest, SpdLiveCurInRequest, SpdLiveTapRunner, SpdLiveTapRunnerConfig,
    SpdQwen3ForwardInput, SpdQwen3Head, SpdSafetensorsFile, SpdStageLayerRange,
    assemble_spd_live_cur_in_for_positions, plan_hidden_state_taps, sliding_spd_row_positions,
};

use super::*;

pub(super) const SPD_REPLAY_PROPOSAL_SOURCE: &str = "spd-replay";

pub(super) struct SpdReplayOpenArgs<'a> {
    pub(super) manifest_path: Option<&'a Path>,
    pub(super) fixture_path: Option<&'a Path>,
    pub(super) model_path: Option<&'a Path>,
    pub(super) config: &'a StageConfig,
    pub(super) topology: Option<&'a StageTopology>,
    pub(super) n_gpu_layers: Option<i32>,
    pub(super) window: usize,
    pub(super) top_k: usize,
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
    manifest: SpdHeadManifest,
    serving_file: SpdSafetensorsFile,
    live_taps: SpdLiveTapRunner,
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
        let row_count = fixture_cur_in_row_count(&fixture_file, hidden_size)?;
        let row_stage_ids = read_spd_row_stage_ids(&fixture_file, row_count)?;
        let row_hf_indices = read_spd_row_hf_indices(&fixture_file, row_count)?;
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
            head,
            manifest,
            serving_file,
            live_taps,
        })
    }

    fn propose_one(&self, context_tokens: &[i32]) -> Result<i32> {
        let row_positions = sliding_spd_row_positions(context_tokens.len(), self.row_count)?;
        let taps = self.live_taps.collect_taps(context_tokens)?;
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
            let proposal = self.propose_one(&self.context_tokens)?;
            proposals.push(proposal);
            self.context_tokens.push(proposal);
        }
        Ok(proposals)
    }
}

pub(super) fn open_spd_replay_source(
    args: SpdReplayOpenArgs<'_>,
) -> Result<Option<Arc<Mutex<SpdReplayProposalSource>>>> {
    match (args.manifest_path, args.fixture_path) {
        (None, None) => Ok(None),
        (Some(_), Some(_)) => Ok(Some(Arc::new(Mutex::new(SpdReplayProposalSource::open(
            args,
        )?)))),
        _ => bail!("--openai-spd-manifest and --openai-spd-fixture must be set together"),
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
    use skippy_protocol::{LoadMode, StageTopologyEntry};

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
}
