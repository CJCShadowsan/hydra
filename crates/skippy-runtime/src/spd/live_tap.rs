use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};

use super::{
    GgufTokenEmbeddingTable, SpdHeadManifest, SpdSafetensorsFile, SpdStageLayerRange,
    SpdTapInputProjector, project_spd_tap_input_row,
};
use crate::{
    ACTIVATION_FLAG_FINAL_NORMED, ActivationFrame, GGML_TYPE_F16, RuntimeConfig, RuntimeLoadMode,
    StageModel,
    package::{PackageStageRequest, select_layer_package_parts},
};

pub enum SpdLiveTapModelSource<'a> {
    Gguf(&'a Path),
    LayerPackage {
        package_ref: &'a str,
        model_id: &'a str,
        topology_id: &'a str,
    },
}

pub struct SpdLiveTapRunnerConfig<'a> {
    pub model_source: SpdLiveTapModelSource<'a>,
    pub stage_ranges: &'a [SpdStageLayerRange],
    pub layer_end: u32,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub ctx_size: u32,
    pub n_gpu_layers: i32,
    pub selected_backend_device: Option<String>,
    pub stage_backend_devices: Vec<String>,
    pub stream_stages: bool,
}

pub struct SpdLiveTapRunner {
    h0: SpdLiveH0Source,
    model_source: SpdOwnedLiveTapModelSource,
    stage_ranges: Vec<SpdStageLayerRange>,
    layer_end: u32,
    hidden_size: usize,
    vocab_size: usize,
    ctx_size: u32,
    n_gpu_layers: i32,
    selected_backend_device: Option<String>,
    stage_backend_devices: Vec<String>,
    stages: SpdLiveStageSet,
}

enum SpdOwnedLiveTapModelSource {
    Gguf(PathBuf),
    LayerPackage {
        package_ref: String,
        model_id: String,
        topology_id: String,
    },
}

impl SpdOwnedLiveTapModelSource {
    fn from_borrowed(source: &SpdLiveTapModelSource<'_>) -> Self {
        match source {
            SpdLiveTapModelSource::Gguf(path) => Self::Gguf((*path).to_path_buf()),
            SpdLiveTapModelSource::LayerPackage {
                package_ref,
                model_id,
                topology_id,
            } => Self::LayerPackage {
                package_ref: (*package_ref).to_string(),
                model_id: (*model_id).to_string(),
                topology_id: (*topology_id).to_string(),
            },
        }
    }

    fn as_borrowed(&self) -> SpdLiveTapModelSource<'_> {
        match self {
            Self::Gguf(path) => SpdLiveTapModelSource::Gguf(path.as_path()),
            Self::LayerPackage {
                package_ref,
                model_id,
                topology_id,
            } => SpdLiveTapModelSource::LayerPackage {
                package_ref,
                model_id,
                topology_id,
            },
        }
    }
}

enum SpdLiveStageSet {
    Resident(Vec<SpdLiveStage>),
    Streaming,
}

enum SpdLiveH0Source {
    Embeddings(GgufTokenEmbeddingTable),
    Stage(StageModel),
}

struct SpdLiveStage {
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
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
    pub terminal_hidden_hf_index: Option<u32>,
    pub final_norm_weight: Option<&'a [f32]>,
}

pub struct SpdLiveCurInRows {
    pub cur_in: Vec<f32>,
}

impl SpdLiveTapRunner {
    pub fn open(config: SpdLiveTapRunnerConfig<'_>) -> Result<Self> {
        validate_stage_backend_devices(&config)?;
        let h0 = open_h0_source(&config)?;
        let stages = if config.stream_stages {
            SpdLiveStageSet::Streaming
        } else {
            SpdLiveStageSet::Resident(
                config
                    .stage_ranges
                    .iter()
                    .map(|range| {
                        let model = open_live_stage_model(
                            &config,
                            range.stage_index,
                            range.layer_start,
                            range.layer_end,
                            false,
                            false,
                        )?;
                        Ok(SpdLiveStage {
                            stage_index: range.stage_index,
                            layer_start: range.layer_start,
                            layer_end: range.layer_end,
                            model,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
        };
        Ok(Self {
            h0,
            model_source: SpdOwnedLiveTapModelSource::from_borrowed(&config.model_source),
            stage_ranges: config.stage_ranges.to_vec(),
            layer_end: config.layer_end,
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            ctx_size: config.ctx_size,
            n_gpu_layers: config.n_gpu_layers,
            selected_backend_device: config.selected_backend_device.clone(),
            stage_backend_devices: config.stage_backend_devices.clone(),
            stages,
        })
    }

    pub fn collect_taps(&self, context_tokens: &[i32]) -> Result<BTreeMap<u32, ActivationFrame>> {
        let mut taps = BTreeMap::new();
        taps.insert(0, self.collect_h0_tap(context_tokens)?);

        match &self.stages {
            SpdLiveStageSet::Resident(stages) => {
                self.collect_resident_stage_taps(context_tokens, stages, &mut taps)?;
            }
            SpdLiveStageSet::Streaming => {
                self.collect_streaming_stage_taps(context_tokens, &mut taps)?;
            }
        }
        Ok(taps)
    }

    fn collect_resident_stage_taps(
        &self,
        context_tokens: &[i32],
        stages: &[SpdLiveStage],
        taps: &mut BTreeMap<u32, ActivationFrame>,
    ) -> Result<()> {
        let mut input = None;
        for stage in stages {
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
            taps.insert(stage.layer_end, output.clone());
            input = Some(output);
        }
        Ok(())
    }

    fn collect_streaming_stage_taps(
        &self,
        context_tokens: &[i32],
        taps: &mut BTreeMap<u32, ActivationFrame>,
    ) -> Result<()> {
        let config = self.stage_open_config();
        let mut input = None;
        for range in &self.stage_ranges {
            let model = open_live_stage_model(
                &config,
                range.stage_index,
                range.layer_start,
                range.layer_end,
                false,
                false,
            )
            .with_context(|| {
                format!(
                    "open streaming SPD tap stage {} {}..{}",
                    range.stage_index, range.layer_start, range.layer_end
                )
            })?;
            let output = run_live_stage_model(
                &model,
                range.stage_index,
                range.layer_start,
                range.layer_end,
                context_tokens,
                input.as_ref(),
            )
            .with_context(|| {
                format!(
                    "run streaming SPD tap stage {} {}..{}",
                    range.stage_index, range.layer_start, range.layer_end
                )
            })?;
            taps.insert(range.layer_end, output.clone());
            input = Some(output);
        }
        Ok(())
    }

    fn stage_open_config(&self) -> SpdLiveTapRunnerConfig<'_> {
        SpdLiveTapRunnerConfig {
            model_source: self.model_source.as_borrowed(),
            stage_ranges: &self.stage_ranges,
            layer_end: self.layer_end,
            hidden_size: self.hidden_size,
            vocab_size: self.vocab_size,
            ctx_size: self.ctx_size,
            n_gpu_layers: self.n_gpu_layers,
            selected_backend_device: self.selected_backend_device.clone(),
            stage_backend_devices: self.stage_backend_devices.clone(),
            stream_stages: false,
        }
    }

    pub fn collect_h0_tap(&self, context_tokens: &[i32]) -> Result<ActivationFrame> {
        let row_positions = all_context_row_positions(context_tokens.len())?;
        self.collect_h0_tap_for_positions(context_tokens, &row_positions)
    }

    pub fn collect_h0_tap_for_positions(
        &self,
        context_tokens: &[i32],
        row_positions: &[i64],
    ) -> Result<ActivationFrame> {
        match &self.h0 {
            SpdLiveH0Source::Embeddings(embeddings) => embeddings
                .frame_for_positions(context_tokens, row_positions)
                .context("collect token embedding SPD h0 tap"),
            SpdLiveH0Source::Stage(model) => {
                run_live_stage_model(model, 0, 0, 0, context_tokens, None)
                    .context("run embedding-only SPD h0 tap")
            }
        }
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
        let mut concat_hidden =
            concat_live_hidden(request.taps, hf_indices, position, request.hidden_size)?;
        apply_terminal_final_norm(&mut concat_hidden, &request, hf_indices)?;
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
    let load_mode = match &config.model_source {
        SpdLiveTapModelSource::Gguf(_) => RuntimeLoadMode::RuntimeSlice,
        SpdLiveTapModelSource::LayerPackage { .. } => RuntimeLoadMode::LayerPackage,
    };
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
        selected_backend_device: selected_backend_device_for_stage(config, stage_index)?,
        cache_type_k: GGML_TYPE_F16,
        cache_type_v: GGML_TYPE_F16,
        flash_attn_type: crate::FlashAttentionType::Auto,
        load_mode,
        projector_path: None,
        include_embeddings: layer_start == 0 || embedding_only,
        include_output,
        filter_tensors_on_load: true,
        use_mmap: true,
        use_mmap_prefetch: false,
        use_mmap_buffer: true,
        glm_dsa_policy: None,
    };
    match &config.model_source {
        SpdLiveTapModelSource::Gguf(model_path) => StageModel::open(model_path, &runtime_config)
            .with_context(|| {
                format!("open SPD live tap stage {stage_index} {layer_start}..{layer_end}")
            }),
        SpdLiveTapModelSource::LayerPackage {
            package_ref,
            model_id,
            topology_id,
        } => open_live_stage_package_parts(LiveStagePackageOpen {
            package_ref,
            model_id,
            topology_id,
            stage_index,
            layer_start,
            layer_end,
            include_output,
            embedding_only,
            runtime_config: &runtime_config,
        }),
    }
}

fn validate_stage_backend_devices(config: &SpdLiveTapRunnerConfig<'_>) -> Result<()> {
    if !config.stage_backend_devices.is_empty()
        && config.stage_backend_devices.len() != config.stage_ranges.len()
    {
        bail!(
            "SPD live tap stage backend device count {} must match stage count {}",
            config.stage_backend_devices.len(),
            config.stage_ranges.len()
        );
    }
    Ok(())
}

fn selected_backend_device_for_stage(
    config: &SpdLiveTapRunnerConfig<'_>,
    stage_index: u32,
) -> Result<Option<String>> {
    if config.stage_backend_devices.is_empty() {
        return Ok(config.selected_backend_device.clone());
    }
    let stage_index = usize::try_from(stage_index).context("SPD stage index exceeds usize")?;
    config
        .stage_backend_devices
        .get(stage_index)
        .cloned()
        .with_context(|| format!("missing SPD backend device for stage {stage_index}"))
        .map(Some)
}

fn open_h0_source(config: &SpdLiveTapRunnerConfig<'_>) -> Result<SpdLiveH0Source> {
    match &config.model_source {
        SpdLiveTapModelSource::Gguf(model_path) => {
            match GgufTokenEmbeddingTable::open(model_path, config.hidden_size, config.vocab_size) {
                Ok(table) => Ok(SpdLiveH0Source::Embeddings(table)),
                Err(table_error) => open_live_stage_model(config, 0, 0, 0, false, true)
                    .with_context(|| {
                        format!(
                            "open embedding-only SPD h0 tap stage after GGUF token embedding table failed: {table_error:#}"
                        )
                    })
                    .map(SpdLiveH0Source::Stage),
            }
        }
        SpdLiveTapModelSource::LayerPackage { .. } => {
            open_package_h0_embeddings(config).map(SpdLiveH0Source::Embeddings)
        }
    }
}

fn open_package_h0_embeddings(
    config: &SpdLiveTapRunnerConfig<'_>,
) -> Result<GgufTokenEmbeddingTable> {
    let SpdLiveTapModelSource::LayerPackage {
        package_ref,
        model_id,
        topology_id,
    } = &config.model_source
    else {
        bail!("package h0 embeddings require a layer package model source");
    };
    let parts = select_layer_package_parts(&PackageStageRequest {
        model_id: (*model_id).to_string(),
        topology_id: (*topology_id).to_string(),
        package_ref: (*package_ref).to_string(),
        stage_id: "spd-live-tap-h0".to_string(),
        layer_start: 0,
        layer_end: 1,
        include_embeddings: true,
        include_output: false,
    })
    .context("select layer package token embedding parts for SPD h0")?;
    let mut failures = Vec::new();
    for path in parts
        .selected_parts
        .iter()
        .zip(parts.absolute_paths.iter())
        .filter_map(|(part, path)| (part.role == "embeddings").then_some(path))
    {
        match GgufTokenEmbeddingTable::open(path, config.hidden_size, config.vocab_size) {
            Ok(table) => return Ok(table),
            Err(error) => failures.push(format!("{}: {error:#}", path.display())),
        }
    }
    bail!(
        "selected layer package parts did not contain a usable token_embd.weight for SPD h0: {}",
        failures.join("; ")
    )
}

struct LiveStagePackageOpen<'a> {
    package_ref: &'a str,
    model_id: &'a str,
    topology_id: &'a str,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    include_output: bool,
    embedding_only: bool,
    runtime_config: &'a RuntimeConfig,
}

fn open_live_stage_package_parts(args: LiveStagePackageOpen<'_>) -> Result<StageModel> {
    let parts = select_layer_package_parts(&PackageStageRequest {
        model_id: args.model_id.to_string(),
        topology_id: args.topology_id.to_string(),
        package_ref: args.package_ref.to_string(),
        stage_id: format!("spd-live-tap-{}", args.stage_index),
        layer_start: args.layer_start,
        layer_end: args.layer_end,
        include_embeddings: args.layer_start == 0 || args.embedding_only,
        include_output: args.include_output,
    })
    .with_context(|| {
        format!(
            "select SPD live tap package parts for stage {}",
            args.stage_index
        )
    })?;
    StageModel::open_from_parts(&parts.absolute_paths, args.runtime_config).with_context(|| {
        format!(
            "open SPD live tap package stage {} {}..{}",
            args.stage_index, args.layer_start, args.layer_end
        )
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

fn all_context_row_positions(token_count: usize) -> Result<Vec<i64>> {
    (0..token_count)
        .map(|position| i64::try_from(position).context("SPD h0 row position exceeds i64"))
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
    if request.terminal_hidden_hf_index.is_some() != request.final_norm_weight.is_some() {
        bail!(
            "SPD terminal hidden final norm requires both terminal_hidden_hf_index and final_norm_weight"
        );
    }
    match request.final_norm_weight {
        Some(weight) if weight.len() != request.hidden_size => {
            bail!(
                "SPD terminal final norm weight length {} does not match hidden size {}",
                weight.len(),
                request.hidden_size
            );
        }
        _ => {}
    }
    Ok(())
}

fn apply_terminal_final_norm(
    concat_hidden: &mut [f32],
    request: &SpdLiveCurInRequest<'_>,
    hf_indices: &[u32],
) -> Result<()> {
    let (Some(terminal_hf_index), Some(final_norm_weight)) =
        (request.terminal_hidden_hf_index, request.final_norm_weight)
    else {
        return Ok(());
    };
    for (component_index, hf_index) in hf_indices.iter().enumerate() {
        if *hf_index != terminal_hf_index {
            continue;
        }
        if terminal_frame_is_final_normed(request.taps, terminal_hf_index)? {
            continue;
        }
        let start = component_index
            .checked_mul(request.hidden_size)
            .context("SPD terminal final norm component offset overflow")?;
        let end = start + request.hidden_size;
        qwen_final_rms_norm_in_place(&mut concat_hidden[start..end], final_norm_weight);
    }
    Ok(())
}

fn terminal_frame_is_final_normed(
    taps: &BTreeMap<u32, ActivationFrame>,
    terminal_hf_index: u32,
) -> Result<bool> {
    let frame = taps.get(&terminal_hf_index).with_context(|| {
        format!("missing live Skippy terminal tap for HF hidden-state {terminal_hf_index}")
    })?;
    Ok((frame.desc.flags & ACTIVATION_FLAG_FINAL_NORMED) != 0)
}

fn qwen_final_rms_norm_in_place(values: &mut [f32], weight: &[f32]) {
    let sum_sq: f32 = values.iter().map(|value| value * value).sum();
    let scale = (sum_sq / values.len() as f32 + 1.0e-6).sqrt().recip();
    for (value, weight) in values.iter_mut().zip(weight) {
        *value *= scale * *weight;
    }
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

    #[test]
    fn terminal_frame_final_norm_flag_is_detected() {
        let mut taps = BTreeMap::new();
        taps.insert(
            36,
            activation_frame_with_flags(36, ACTIVATION_FLAG_FINAL_NORMED),
        );

        assert!(terminal_frame_is_final_normed(&taps, 36).unwrap());
    }

    #[test]
    fn terminal_frame_without_final_norm_flag_requires_runtime_norm() {
        let mut taps = BTreeMap::new();
        taps.insert(36, activation_frame_with_flags(36, 0));

        assert!(!terminal_frame_is_final_normed(&taps, 36).unwrap());
    }

    fn activation_frame_with_flags(hf_index: i32, flags: u64) -> ActivationFrame {
        ActivationFrame {
            desc: crate::ActivationDesc {
                version: 1,
                dtype: crate::RuntimeActivationDType::F32,
                layout: crate::RuntimeActivationLayout::TokenMajor,
                producer_stage_index: 1,
                layer_start: hf_index - 1,
                layer_end: hf_index,
                token_count: 1,
                sequence_count: 1,
                payload_bytes: 4,
                flags,
            },
            payload: 1.0_f32.to_le_bytes().to_vec(),
        }
    }
}
