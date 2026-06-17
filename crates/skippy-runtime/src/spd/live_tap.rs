use std::{collections::BTreeMap, path::Path};

use anyhow::{Context, Result, bail};

use super::{
    SpdHeadManifest, SpdSafetensorsFile, SpdStageLayerRange, SpdTapInputProjector,
    project_spd_tap_input_row,
};
use crate::{ActivationFrame, GGML_TYPE_F16, RuntimeConfig, RuntimeLoadMode, StageModel};

pub struct SpdLiveTapRunnerConfig<'a> {
    pub model_path: &'a Path,
    pub stage_ranges: &'a [SpdStageLayerRange],
    pub layer_end: u32,
    pub ctx_size: u32,
    pub n_gpu_layers: i32,
    pub selected_backend_device: Option<String>,
}

pub struct SpdLiveTapRunner {
    h0: StageModel,
    stages: Vec<SpdLiveStage>,
}

struct SpdLiveStage {
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    include_output: bool,
    model: StageModel,
}

pub struct SpdLiveCurInRequest<'a> {
    pub manifest: &'a SpdHeadManifest,
    pub serving_file: &'a SpdSafetensorsFile,
    pub tap_projector: Option<&'a SpdTapInputProjector>,
    pub taps: &'a BTreeMap<u32, ActivationFrame>,
    pub row_positions: &'a [i64],
    pub row_stage_ids: &'a [i64],
    pub row_hf_indices: &'a [Vec<u32>],
    pub hidden_size: usize,
}

pub struct SpdLiveCurInRows {
    pub cur_in: Vec<f32>,
}

impl SpdLiveTapRunner {
    pub fn open(config: SpdLiveTapRunnerConfig<'_>) -> Result<Self> {
        let h0 = open_live_stage_model(&config, 0, 0, 0, false, true)
            .context("open embedding-only SPD h0 tap stage")?;
        let stages = config
            .stage_ranges
            .iter()
            .map(|range| {
                // SPD tap replay needs the final boundary hidden state too. Target
                // logits are verified through a separate full-model session.
                let include_output = false;
                let model = open_live_stage_model(
                    &config,
                    range.stage_index,
                    range.layer_start,
                    range.layer_end,
                    include_output,
                    false,
                )?;
                Ok(SpdLiveStage {
                    stage_index: range.stage_index,
                    layer_start: range.layer_start,
                    layer_end: range.layer_end,
                    include_output,
                    model,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { h0, stages })
    }

    pub fn collect_taps(&self, context_tokens: &[i32]) -> Result<BTreeMap<u32, ActivationFrame>> {
        let mut taps = BTreeMap::new();
        taps.insert(0, self.collect_h0_tap(context_tokens)?);

        let mut input = None;
        for stage in &self.stages {
            let output = run_live_stage_model(
                &stage.model,
                stage.stage_index,
                stage.layer_start,
                stage.layer_end,
                context_tokens,
                input.as_ref(),
            )
            .with_context(|| {
                format!(
                    "run live SPD tap stage {} {}..{}",
                    stage.stage_index, stage.layer_start, stage.layer_end
                )
            })?;
            if !stage.include_output {
                taps.insert(stage.layer_end, output.clone());
                input = Some(output);
            }
        }
        Ok(taps)
    }

    pub fn collect_h0_tap(&self, context_tokens: &[i32]) -> Result<ActivationFrame> {
        run_live_stage_model(&self.h0, 0, 0, 0, context_tokens, None)
            .context("run embedding-only SPD h0 tap")
    }
}

pub fn assemble_spd_live_cur_in_for_positions(
    request: SpdLiveCurInRequest<'_>,
) -> Result<SpdLiveCurInRows> {
    validate_live_cur_in_request(&request)?;
    let mut cur_in = Vec::with_capacity(request.row_positions.len() * request.hidden_size);
    for row_index in 0..request.row_positions.len() {
        let position = request.row_positions[row_index];
        let stage_id = u32::try_from(request.row_stage_ids[row_index])
            .with_context(|| format!("SPD row {row_index} has negative stage id"))?;
        let hf_indices = &request.row_hf_indices[row_index];
        let concat_hidden =
            concat_live_hidden(request.taps, hf_indices, position, request.hidden_size)?;
        let projection = project_live_tap_input(&request, stage_id, hf_indices, &concat_hidden)?;
        cur_in.extend_from_slice(&projection.projected);
    }
    Ok(SpdLiveCurInRows { cur_in })
}

fn project_live_tap_input(
    request: &SpdLiveCurInRequest<'_>,
    stage_id: u32,
    hf_indices: &[u32],
    concat_hidden: &[f32],
) -> Result<super::SpdTapInputProjection> {
    if let Some(projector) = request.tap_projector {
        return projector.project(stage_id, hf_indices, concat_hidden);
    }
    project_spd_tap_input_row(
        &request.manifest.topology,
        request.serving_file,
        stage_id,
        hf_indices,
        concat_hidden,
    )
}

pub fn sliding_spd_row_positions(context_len: usize, row_count: usize) -> Result<Vec<i64>> {
    if context_len < row_count {
        bail!("context length {context_len} is shorter than SPD row count {row_count}");
    }
    let start = context_len - row_count;
    (start..context_len)
        .map(|position| i64::try_from(position).context("SPD row position exceeds i64"))
        .collect()
}

fn open_live_stage_model(
    config: &SpdLiveTapRunnerConfig<'_>,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    include_output: bool,
    embedding_only: bool,
) -> Result<StageModel> {
    let runtime_config = RuntimeConfig {
        stage_index,
        layer_start,
        layer_end,
        ctx_size: config.ctx_size,
        lane_count: 1,
        n_batch: None,
        n_ubatch: None,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: config.n_gpu_layers,
        selected_backend_device: config.selected_backend_device.clone(),
        cache_type_k: GGML_TYPE_F16,
        cache_type_v: GGML_TYPE_F16,
        flash_attn_type: crate::FlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::RuntimeSlice,
        projector_path: None,
        include_embeddings: layer_start == 0 || embedding_only,
        include_output,
        filter_tensors_on_load: true,
    };
    StageModel::open(config.model_path, &runtime_config).with_context(|| {
        format!("open SPD live tap stage {stage_index} {layer_start}..{layer_end}")
    })
}

fn run_live_stage_model(
    model: &StageModel,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    context_tokens: &[i32],
    input: Option<&ActivationFrame>,
) -> Result<ActivationFrame> {
    let mut session = model.create_session().with_context(|| {
        format!("create SPD live tap stage {stage_index} {layer_start}..{layer_end} session")
    })?;
    let positions = sequential_positions(context_tokens.len())?;
    session.prefill_chunk_frame_with_positions(context_tokens, &positions, input, 0)
}

fn sequential_positions(token_count: usize) -> Result<Vec<i32>> {
    (0..token_count)
        .map(|position| i32::try_from(position).context("SPD tap position exceeds i32"))
        .collect()
}

fn validate_live_cur_in_request(request: &SpdLiveCurInRequest<'_>) -> Result<()> {
    if request.row_positions.len() != request.row_stage_ids.len()
        || request.row_positions.len() != request.row_hf_indices.len()
    {
        bail!(
            "SPD live row metadata length mismatch: positions {}, stages {}, hf rows {}",
            request.row_positions.len(),
            request.row_stage_ids.len(),
            request.row_hf_indices.len()
        );
    }
    Ok(())
}

fn concat_live_hidden(
    taps: &BTreeMap<u32, ActivationFrame>,
    hf_indices: &[u32],
    position: i64,
    hidden_size: usize,
) -> Result<Vec<f32>> {
    let mut concat = Vec::with_capacity(hf_indices.len() * hidden_size);
    for hf_index in hf_indices {
        let frame = taps
            .get(hf_index)
            .with_context(|| format!("missing live Skippy tap for HF hidden-state {hf_index}"))?;
        concat.extend_from_slice(&live_hidden_row(frame, position, hidden_size)?);
    }
    Ok(concat)
}

fn live_hidden_row(frame: &ActivationFrame, position: i64, hidden_size: usize) -> Result<Vec<f32>> {
    let position = usize::try_from(position).context("negative live tap position")?;
    let token_count =
        usize::try_from(frame.desc.token_count).context("token count exceeds usize")?;
    if position >= token_count {
        bail!("live tap position {position} is outside token_count {token_count}");
    }
    let row_bytes = hidden_size
        .checked_mul(std::mem::size_of::<f32>())
        .context("live activation row byte width overflow")?;
    let expected_payload_bytes = token_count
        .checked_mul(row_bytes)
        .context("live activation payload byte count overflow")?;
    if frame.payload.len() != expected_payload_bytes {
        bail!(
            "live activation payload for {}..{} has {} bytes, expected {} for {} tokens x hidden {}",
            frame.desc.layer_start,
            frame.desc.layer_end,
            frame.payload.len(),
            expected_payload_bytes,
            token_count,
            hidden_size
        );
    }
    let offset = position
        .checked_mul(row_bytes)
        .context("live activation row offset overflow")?;
    Ok(frame.payload[offset..offset + row_bytes]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_positions_use_trailing_context_window() {
        assert_eq!(sliding_spd_row_positions(8, 4).unwrap(), vec![4, 5, 6, 7]);
    }

    #[test]
    fn sliding_positions_reject_short_context() {
        let error = sliding_spd_row_positions(3, 4).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("context length 3 is shorter than SPD row count 4")
        );
    }
}
