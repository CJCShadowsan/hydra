use std::{
    path::{Path, PathBuf},
    sync::Arc,
    thread,
    time::Instant,
};

use anyhow::{Context, Result, bail};

use super::{SpdHeadManifest, SpdSafetensorsFile};

const QWEN3_RMS_NORM_EPS: f32 = 1.0e-6;
const QWEN35_ROPE_THETA: f32 = 10_000_000.0;
const QWEN35_PARTIAL_ROTARY_FACTOR: f32 = 0.25;
const PARALLEL_LINEAR_MIN_DOT_OPS: usize = 2_000_000;

#[derive(Debug, Clone, PartialEq)]
pub struct SpdQwen3FixtureTopK {
    pub draft_indices: Vec<i64>,
    pub token_ids: Vec<i64>,
    pub logits: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpdQwen3FixtureParity {
    pub rust: SpdQwen3FixtureTopK,
    pub python: SpdQwen3FixtureTopK,
    pub diagnostics: SpdQwen3FixtureDiagnostics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpdQwen3CachedFixtureParity {
    pub rust: SpdQwen3FixtureTopK,
    pub python: SpdQwen3FixtureTopK,
    pub diagnostics: SpdQwen3CachedFixtureDiagnostics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpdQwen3FixtureDiagnostics {
    pub layer_input_max_abs_diff: Vec<f32>,
    pub layer_query_max_abs_diff: Vec<f32>,
    pub spec_query_max_abs_diff: f32,
    pub final_hidden_max_abs_diff: f32,
    pub python_top_logit_values_at_rust_indices: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpdQwen3CachedFixtureDiagnostics {
    pub cache_prefix_len: usize,
    pub spec_query_max_abs_diff: f32,
    pub final_hidden_max_abs_diff: f32,
    pub logits_max_abs_diff: f32,
    pub python_top_logit_values_at_rust_indices: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpdQwen3ForwardInput {
    pub cur_in: Vec<f32>,
    pub seq_len: usize,
    pub position_ids: Vec<i64>,
    pub fixed_stage_ids: Option<Vec<usize>>,
    pub final_norm_weight: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpdQwen3ForwardTiming {
    pub fixed_stage_projection_ms: f64,
    pub decoder_layer_ms: Vec<f64>,
    pub final_norm_ms: f64,
    pub lm_head_topk_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpdQwen3TimedForward {
    pub topk: SpdQwen3FixtureTopK,
    pub timing: SpdQwen3ForwardTiming,
}

#[derive(Debug, Clone)]
pub struct SpdQwen3ForwardCache {
    layers: Vec<SpdQwen3LayerKvCache>,
    kv_width: usize,
}

#[derive(Debug, Clone, Default)]
struct SpdQwen3LayerKvCache {
    positions: Vec<i64>,
    k: Vec<f32>,
    v: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SpdQwen3Head {
    manifest_path: PathBuf,
    manifest: SpdHeadManifest,
    shape: SpdQwen3Shape,
    weights: Arc<SpdQwen3Weights>,
}

impl SpdQwen3Head {
    pub fn open(manifest_path: impl AsRef<Path>) -> Result<Self> {
        let manifest_path = manifest_path.as_ref().to_path_buf();
        let manifest = SpdHeadManifest::from_path(&manifest_path)?;
        manifest.ensure_serving_checkpoint_for_runtime(&manifest_path)?;
        let serving_file =
            SpdSafetensorsFile::open(manifest.serving_checkpoint_path(&manifest_path)?)?;
        let shape = SpdQwen3Shape::from_manifest_and_weights(&manifest, &serving_file)?;
        let weights = Arc::new(SpdQwen3Weights::load(&serving_file, &shape)?);
        Ok(Self {
            manifest_path,
            manifest,
            shape,
            weights,
        })
    }

    pub fn manifest(&self) -> &SpdHeadManifest {
        &self.manifest
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    pub fn new_forward_cache(&self) -> SpdQwen3ForwardCache {
        SpdQwen3ForwardCache::new(&self.shape)
    }

    pub fn forward(
        &self,
        input: SpdQwen3ForwardInput,
        top_k: usize,
    ) -> Result<SpdQwen3FixtureTopK> {
        let final_hidden = run_forward_hidden(&self.weights, input, &self.shape)?;
        lm_head_topk(
            &self.weights,
            &final_hidden,
            top_k,
            self.manifest.topology.draft_token_ids.as_deref(),
        )
    }

    pub fn prefill_cache(
        &self,
        input: SpdQwen3ForwardInput,
        cache: &mut SpdQwen3ForwardCache,
    ) -> Result<()> {
        prefill_forward_cache(&self.weights, input, &self.shape, cache)
    }

    pub fn forward_with_cache(
        &self,
        input: SpdQwen3ForwardInput,
        top_k: usize,
        cache: &mut SpdQwen3ForwardCache,
    ) -> Result<SpdQwen3FixtureTopK> {
        let final_hidden = run_forward_hidden_with_cache(&self.weights, input, &self.shape, cache)?;
        lm_head_topk(
            &self.weights,
            &final_hidden,
            top_k,
            self.manifest.topology.draft_token_ids.as_deref(),
        )
    }

    pub fn forward_timed(
        &self,
        input: SpdQwen3ForwardInput,
        top_k: usize,
    ) -> Result<SpdQwen3TimedForward> {
        let total_timer = Instant::now();
        let core = run_forward_core(&self.weights, input, &self.shape, ForwardMode::Timed)?;
        let lm_head_timer = Instant::now();
        let topk = lm_head_topk(
            &self.weights,
            &core.final_hidden,
            top_k,
            self.manifest.topology.draft_token_ids.as_deref(),
        )?;
        let timing = SpdQwen3ForwardTiming {
            fixed_stage_projection_ms: core.timing.fixed_stage_projection_ms,
            decoder_layer_ms: core.timing.decoder_layer_ms,
            final_norm_ms: core.timing.final_norm_ms,
            lm_head_topk_ms: elapsed_ms(lm_head_timer),
            total_ms: elapsed_ms(total_timer),
        };
        Ok(SpdQwen3TimedForward { topk, timing })
    }
}

impl SpdQwen3ForwardCache {
    fn new(shape: &SpdQwen3Shape) -> Self {
        Self {
            layers: vec![SpdQwen3LayerKvCache::default(); shape.num_spec_layers],
            kv_width: shape.num_key_value_heads * shape.head_dim,
        }
    }

    pub fn crop_to_position(&mut self, position: i64) {
        for layer in &mut self.layers {
            layer.crop_to_position(position, self.kv_width);
        }
    }

    pub fn cached_prefix_len(&self) -> usize {
        self.layers
            .first()
            .map(|layer| layer.cached_prefix_len())
            .unwrap_or(0)
    }
}

impl SpdQwen3LayerKvCache {
    fn crop_to_position(&mut self, position: i64, kv_width: usize) {
        let keep = self
            .positions
            .iter()
            .position(|cached_position| *cached_position >= position)
            .unwrap_or(self.positions.len());
        self.positions.truncate(keep);
        self.k.truncate(keep * kv_width);
        self.v.truncate(keep * kv_width);
    }

    fn cached_prefix_len(&self) -> usize {
        self.positions
            .iter()
            .copied()
            .enumerate()
            .take_while(|(idx, position)| usize::try_from(*position).ok() == Some(*idx))
            .count()
    }

    fn append(
        &mut self,
        position_ids: &[i64],
        k: &[f32],
        v: &[f32],
        kv_width: usize,
    ) -> Result<()> {
        if k.len() != position_ids.len() * kv_width || v.len() != position_ids.len() * kv_width {
            bail!("SPD cache KV length must match positions * KV width");
        }
        if let (Some(last), Some(first_new)) = (self.positions.last(), position_ids.first())
            && last >= first_new
        {
            bail!("SPD cache append positions must be after existing cache");
        }
        self.positions.extend_from_slice(position_ids);
        self.k.extend_from_slice(k);
        self.v.extend_from_slice(v);
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct SpdQwen3Shape {
    hidden_size: usize,
    num_stages: usize,
    num_spec_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_key_value_groups: usize,
    head_dim: usize,
    rotary_dim: usize,
}

#[derive(Debug, Clone)]
struct SpdQwen3ForwardTrace {
    logits: Vec<f32>,
    layer_inputs: Vec<Vec<f32>>,
    layer_queries: Vec<Vec<f32>>,
    spec_query: Vec<f32>,
    final_hidden: Vec<f32>,
}

struct SpdQwen3ForwardCore {
    layer_inputs: Option<Vec<Vec<f32>>>,
    layer_queries: Option<Vec<Vec<f32>>>,
    spec_query: Vec<f32>,
    final_hidden: Vec<f32>,
    timing: SpdQwen3ForwardTiming,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForwardMode {
    Fast,
    Timed,
    Trace,
}

#[derive(Debug)]
struct SpdQwen3Weights {
    fixed_stage_per_layer_projs: Vec<Vec<Vec<f32>>>,
    layers: Vec<SpdQwen3LayerWeights>,
    lm_head: Vec<f32>,
}

#[derive(Debug)]
struct SpdQwen3LayerWeights {
    input_layernorm: Vec<f32>,
    q_proj: Vec<f32>,
    q_norm: Vec<f32>,
    k_proj: Vec<f32>,
    k_norm: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
}

impl SpdQwen3Weights {
    fn load(serving_file: &SpdSafetensorsFile, shape: &SpdQwen3Shape) -> Result<Self> {
        let mut fixed_stage_per_layer_projs = Vec::with_capacity(shape.num_spec_layers);
        for layer in 0..shape.num_spec_layers {
            let mut projections = Vec::with_capacity(shape.num_stages);
            for projection_idx in 0..shape.num_stages {
                projections.push(serving_file.read_tensor_f32(&format!(
                    "fixed_stage_per_layer_projs.{layer}.{projection_idx}.weight"
                ))?);
            }
            fixed_stage_per_layer_projs.push(projections);
        }

        let mut layers = Vec::with_capacity(shape.num_spec_layers);
        for layer in 0..shape.num_spec_layers {
            layers.push(SpdQwen3LayerWeights {
                input_layernorm: serving_file
                    .read_tensor_f32(&format!("spec_layers.{layer}.input_layernorm.weight"))?,
                q_proj: serving_file
                    .read_tensor_f32(&format!("spec_layers.{layer}.self_attn.q_proj.weight"))?,
                q_norm: serving_file
                    .read_tensor_f32(&format!("spec_layers.{layer}.self_attn.q_norm.weight"))?,
                k_proj: serving_file
                    .read_tensor_f32(&format!("spec_layers.{layer}.self_attn.k_proj.weight"))?,
                k_norm: serving_file
                    .read_tensor_f32(&format!("spec_layers.{layer}.self_attn.k_norm.weight"))?,
                v_proj: serving_file
                    .read_tensor_f32(&format!("spec_layers.{layer}.self_attn.v_proj.weight"))?,
                o_proj: serving_file
                    .read_tensor_f32(&format!("spec_layers.{layer}.self_attn.o_proj.weight"))?,
                post_attention_layernorm: serving_file.read_tensor_f32(&format!(
                    "spec_layers.{layer}.post_attention_layernorm.weight"
                ))?,
                gate_proj: serving_file
                    .read_tensor_f32(&format!("spec_layers.{layer}.mlp.gate_proj.weight"))?,
                up_proj: serving_file
                    .read_tensor_f32(&format!("spec_layers.{layer}.mlp.up_proj.weight"))?,
                down_proj: serving_file
                    .read_tensor_f32(&format!("spec_layers.{layer}.mlp.down_proj.weight"))?,
            });
        }

        Ok(Self {
            fixed_stage_per_layer_projs,
            layers,
            lm_head: serving_file.read_tensor_f32("lm_head.weight")?,
        })
    }
}

pub fn run_qwen3_fixture_parity(
    manifest_path: impl AsRef<Path>,
    fixture_path: impl AsRef<Path>,
    top_k: usize,
) -> Result<SpdQwen3FixtureParity> {
    let manifest_path = manifest_path.as_ref();
    let fixture_path = fixture_path.as_ref();
    let head = SpdQwen3Head::open(manifest_path)?;
    let fixture_file = SpdSafetensorsFile::open(fixture_path)?;
    let trace = run_fixture_forward(&head.weights, &fixture_file, &head.shape)?;
    let rust = topk_from_logits(
        &trace.logits,
        top_k,
        head.manifest.topology.draft_token_ids.as_deref(),
    )?;
    let python = python_topk_from_fixture(&fixture_file)?;
    let diagnostics = fixture_diagnostics(&fixture_file, &trace, &rust)?;
    Ok(SpdQwen3FixtureParity {
        rust,
        python,
        diagnostics,
    })
}

pub fn run_qwen3_cached_fixture_parity(
    manifest_path: impl AsRef<Path>,
    fixture_path: impl AsRef<Path>,
    top_k: usize,
) -> Result<Option<SpdQwen3CachedFixtureParity>> {
    let manifest_path = manifest_path.as_ref();
    let fixture_path = fixture_path.as_ref();
    let head = SpdQwen3Head::open(manifest_path)?;
    let fixture_file = SpdSafetensorsFile::open(fixture_path)?;
    if !fixture_file
        .index
        .tensors
        .contains_key("python_cached_logits")
    {
        return Ok(None);
    }
    let mut cache = head.new_forward_cache();
    let (trace, cache_prefix_len) =
        run_cached_fixture_forward(&head.weights, &fixture_file, &head.shape, &mut cache)?;
    let rust = topk_from_logits(
        &trace.logits,
        top_k,
        head.manifest.topology.draft_token_ids.as_deref(),
    )?;
    let python = python_cached_topk_from_fixture(&fixture_file)?;
    let diagnostics = cached_fixture_diagnostics(&fixture_file, &trace, &rust, cache_prefix_len)?;
    Ok(Some(SpdQwen3CachedFixtureParity {
        rust,
        python,
        diagnostics,
    }))
}

pub fn run_qwen3_forward_from_inputs(
    manifest_path: impl AsRef<Path>,
    input: SpdQwen3ForwardInput,
    top_k: usize,
) -> Result<SpdQwen3FixtureTopK> {
    SpdQwen3Head::open(manifest_path)?.forward(input, top_k)
}

fn run_fixture_forward(
    weights: &SpdQwen3Weights,
    fixture_file: &SpdSafetensorsFile,
    shape: &SpdQwen3Shape,
) -> Result<SpdQwen3ForwardTrace> {
    run_forward_trace(
        weights,
        fixture_forward_input(fixture_file, shape, "cur_in", "position_ids")?,
        shape,
    )
}

fn run_cached_fixture_forward(
    weights: &SpdQwen3Weights,
    fixture_file: &SpdSafetensorsFile,
    shape: &SpdQwen3Shape,
    cache: &mut SpdQwen3ForwardCache,
) -> Result<(SpdQwen3ForwardTrace, usize)> {
    if fixture_file
        .index
        .tensors
        .contains_key("cached_prefill_cur_in")
    {
        let prefill = fixture_forward_input(
            fixture_file,
            shape,
            "cached_prefill_cur_in",
            "cached_prefill_position_ids",
        )?;
        prefill_forward_cache(weights, prefill, shape, cache)?;
    }
    let cache_prefix_len = cache.cached_prefix_len();
    let input = fixture_forward_input(fixture_file, shape, "cur_in", "position_ids")?;
    let trace = run_forward_trace_with_cache(weights, input, shape, cache)?;
    Ok((trace, cache_prefix_len))
}

fn fixture_forward_input(
    fixture_file: &SpdSafetensorsFile,
    shape: &SpdQwen3Shape,
    cur_in_name: &str,
    position_ids_name: &str,
) -> Result<SpdQwen3ForwardInput> {
    let cur_in = fixture_file.read_tensor_f32(cur_in_name)?;
    let cur_shape = &fixture_file.index.tensor(cur_in_name)?.shape;
    if cur_shape.len() != 3 || cur_shape[0] != 1 || cur_shape[2] != shape.hidden_size as u64 {
        bail!(
            "SPD fixture {cur_in_name} shape {:?} is not [1, seq, hidden]",
            cur_shape,
        );
    }
    let seq_len = usize::try_from(cur_shape[1]).context("SPD fixture sequence length too large")?;
    let position_ids = fixture_file.read_tensor_i64(position_ids_name)?;
    if position_ids.len() != seq_len {
        bail!(
            "SPD fixture {position_ids_name} length {} must match {cur_in_name} seq_len {}",
            position_ids.len(),
            seq_len
        );
    }
    let final_norm_weight = fixture_file.read_tensor_f32("final_norm_weight")?;
    if final_norm_weight.len() != shape.hidden_size {
        bail!(
            "SPD fixture final_norm_weight length {} must match hidden_size {}",
            final_norm_weight.len(),
            shape.hidden_size
        );
    }
    let fixed_stage_ids =
        (cur_in_name == "cached_prefill_cur_in").then(|| vec![shape.num_stages; seq_len]);

    Ok(SpdQwen3ForwardInput {
        cur_in,
        seq_len,
        position_ids,
        fixed_stage_ids,
        final_norm_weight,
    })
}

fn run_forward_trace(
    weights: &SpdQwen3Weights,
    input: SpdQwen3ForwardInput,
    shape: &SpdQwen3Shape,
) -> Result<SpdQwen3ForwardTrace> {
    let core = run_forward_core(weights, input, shape, ForwardMode::Trace)?;
    let final_hidden = core.final_hidden;
    let logits = lm_head_logits(weights, &final_hidden)?;
    Ok(SpdQwen3ForwardTrace {
        logits,
        layer_inputs: core.layer_inputs.unwrap_or_default(),
        layer_queries: core.layer_queries.unwrap_or_default(),
        spec_query: core.spec_query,
        final_hidden,
    })
}

fn run_forward_trace_with_cache(
    weights: &SpdQwen3Weights,
    input: SpdQwen3ForwardInput,
    shape: &SpdQwen3Shape,
    cache: &mut SpdQwen3ForwardCache,
) -> Result<SpdQwen3ForwardTrace> {
    let core = run_forward_core_with_cache(weights, input, shape, cache)?;
    let final_hidden = core.final_hidden;
    let logits = lm_head_logits(weights, &final_hidden)?;
    Ok(SpdQwen3ForwardTrace {
        logits,
        layer_inputs: Vec::new(),
        layer_queries: Vec::new(),
        spec_query: core.spec_query,
        final_hidden,
    })
}

fn run_forward_hidden(
    weights: &SpdQwen3Weights,
    input: SpdQwen3ForwardInput,
    shape: &SpdQwen3Shape,
) -> Result<Vec<f32>> {
    Ok(run_forward_core(weights, input, shape, ForwardMode::Fast)?.final_hidden)
}

fn run_forward_hidden_with_cache(
    weights: &SpdQwen3Weights,
    input: SpdQwen3ForwardInput,
    shape: &SpdQwen3Shape,
    cache: &mut SpdQwen3ForwardCache,
) -> Result<Vec<f32>> {
    Ok(run_forward_core_with_cache(weights, input, shape, cache)?.final_hidden)
}

fn run_forward_core(
    weights: &SpdQwen3Weights,
    input: SpdQwen3ForwardInput,
    shape: &SpdQwen3Shape,
    mode: ForwardMode,
) -> Result<SpdQwen3ForwardCore> {
    validate_forward_input(&input, shape)?;
    let cur_in = input.cur_in;
    let seq_len = input.seq_len;
    let position_ids = input.position_ids;
    let stage_ids = input
        .fixed_stage_ids
        .unwrap_or_else(|| infer_stage_ids(seq_len, shape.num_stages));
    let final_norm_weight = input.final_norm_weight;
    let original_hidden = cur_in.clone();
    let mut base_fixed = cur_in.clone();
    let mut query = row(&cur_in, seq_len - 1, shape.hidden_size).to_vec();
    let collect_trace = mode == ForwardMode::Trace;
    let mut layer_inputs = collect_trace.then(|| Vec::with_capacity(shape.num_spec_layers));
    let mut layer_queries = collect_trace.then(|| Vec::with_capacity(shape.num_spec_layers));
    let mut timing = SpdQwen3ForwardTiming {
        fixed_stage_projection_ms: 0.0,
        decoder_layer_ms: Vec::with_capacity(shape.num_spec_layers),
        final_norm_ms: 0.0,
        lm_head_topk_ms: 0.0,
        total_ms: 0.0,
    };

    for layer in 0..shape.num_spec_layers {
        let fixed_timer = (mode == ForwardMode::Timed).then(Instant::now);
        apply_fixed_stage_projections(weights, &mut base_fixed, &stage_ids, layer, shape)?;
        if let Some(timer) = fixed_timer {
            timing.fixed_stage_projection_ms += elapsed_ms(timer);
        }
        let mut full_in = original_hidden.clone();
        copy_fixed_rows(&mut full_in, &base_fixed, &stage_ids, shape);
        full_in[(seq_len - 1) * shape.hidden_size..seq_len * shape.hidden_size]
            .copy_from_slice(&query);
        if let Some(layer_inputs) = layer_inputs.as_mut() {
            layer_inputs.push(full_in.clone());
        }
        let layer_timer = (mode == ForwardMode::Timed).then(Instant::now);
        query = decoder_layer_query(weights, &full_in, &position_ids, layer, shape)?;
        if let Some(timer) = layer_timer {
            timing.decoder_layer_ms.push(elapsed_ms(timer));
        }
        if let Some(layer_queries) = layer_queries.as_mut() {
            layer_queries.push(query.clone());
        }
    }

    let spec_query = query.clone();
    let norm_timer = (mode == ForwardMode::Timed).then(Instant::now);
    qwen35_final_norm_in_place(&mut query, &final_norm_weight, QWEN3_RMS_NORM_EPS);
    if let Some(timer) = norm_timer {
        timing.final_norm_ms = elapsed_ms(timer);
    }
    let final_hidden = query.clone();
    Ok(SpdQwen3ForwardCore {
        layer_inputs,
        layer_queries,
        spec_query,
        final_hidden,
        timing,
    })
}

fn run_forward_core_with_cache(
    weights: &SpdQwen3Weights,
    input: SpdQwen3ForwardInput,
    shape: &SpdQwen3Shape,
    cache: &mut SpdQwen3ForwardCache,
) -> Result<SpdQwen3ForwardCore> {
    validate_forward_input(&input, shape)?;
    validate_cache_shape(cache, shape)?;
    let min_position = input.position_ids.iter().copied().min().unwrap_or(0);
    cache.crop_to_position(min_position);

    let cur_in = input.cur_in;
    let seq_len = input.seq_len;
    let position_ids = input.position_ids;
    let stage_ids = input
        .fixed_stage_ids
        .unwrap_or_else(|| infer_stage_ids(seq_len, shape.num_stages));
    let final_norm_weight = input.final_norm_weight;
    let original_hidden = cur_in.clone();
    let mut base_fixed = cur_in.clone();
    let mut query = row(&cur_in, seq_len - 1, shape.hidden_size).to_vec();
    let timing = SpdQwen3ForwardTiming {
        fixed_stage_projection_ms: 0.0,
        decoder_layer_ms: Vec::with_capacity(shape.num_spec_layers),
        final_norm_ms: 0.0,
        lm_head_topk_ms: 0.0,
        total_ms: 0.0,
    };

    for layer in 0..shape.num_spec_layers {
        apply_fixed_stage_projections(weights, &mut base_fixed, &stage_ids, layer, shape)?;
        let mut full_in = original_hidden.clone();
        copy_fixed_rows(&mut full_in, &base_fixed, &stage_ids, shape);
        full_in[(seq_len - 1) * shape.hidden_size..seq_len * shape.hidden_size]
            .copy_from_slice(&query);
        query = decoder_layer_query_with_cache(
            weights,
            &full_in,
            &position_ids,
            layer,
            shape,
            &mut cache.layers[layer],
        )?;
    }

    let spec_query = query.clone();
    qwen35_final_norm_in_place(&mut query, &final_norm_weight, QWEN3_RMS_NORM_EPS);
    let final_hidden = query.clone();
    Ok(SpdQwen3ForwardCore {
        layer_inputs: None,
        layer_queries: None,
        spec_query,
        final_hidden,
        timing,
    })
}

fn prefill_forward_cache(
    weights: &SpdQwen3Weights,
    input: SpdQwen3ForwardInput,
    shape: &SpdQwen3Shape,
    cache: &mut SpdQwen3ForwardCache,
) -> Result<()> {
    validate_forward_input(&input, shape)?;
    validate_cache_shape(cache, shape)?;
    let min_position = input.position_ids.iter().copied().min().unwrap_or(0);
    cache.crop_to_position(min_position);

    let cur_in = input.cur_in;
    let position_ids = input.position_ids;
    let stage_ids = input
        .fixed_stage_ids
        .unwrap_or_else(|| vec![shape.num_stages; input.seq_len]);
    let mut base_fixed = cur_in;
    for layer in 0..shape.num_spec_layers {
        apply_fixed_stage_projections(weights, &mut base_fixed, &stage_ids, layer, shape)?;
        append_layer_kv_to_cache(
            weights,
            &base_fixed,
            &position_ids,
            layer,
            shape,
            &mut cache.layers[layer],
        )?;
    }
    Ok(())
}

fn validate_forward_input(input: &SpdQwen3ForwardInput, shape: &SpdQwen3Shape) -> Result<()> {
    if input.cur_in.len() != input.seq_len * shape.hidden_size {
        bail!(
            "SPD forward cur_in length {} must match seq_len {} * hidden_size {}",
            input.cur_in.len(),
            input.seq_len,
            shape.hidden_size
        );
    }
    if input.position_ids.len() != input.seq_len {
        bail!(
            "SPD forward position_ids length {} must match seq_len {}",
            input.position_ids.len(),
            input.seq_len
        );
    }
    if let Some(fixed_stage_ids) = &input.fixed_stage_ids {
        if fixed_stage_ids.len() != input.seq_len {
            bail!(
                "SPD forward fixed_stage_ids length {} must match seq_len {}",
                fixed_stage_ids.len(),
                input.seq_len
            );
        }
        if let Some(stage_id) = fixed_stage_ids
            .iter()
            .copied()
            .find(|stage_id| *stage_id > shape.num_stages)
        {
            bail!(
                "SPD forward fixed_stage_id {} exceeds num_stages {}",
                stage_id,
                shape.num_stages
            );
        }
    }
    if input.final_norm_weight.len() != shape.hidden_size {
        bail!(
            "SPD forward final_norm_weight length {} must match hidden_size {}",
            input.final_norm_weight.len(),
            shape.hidden_size
        );
    }
    Ok(())
}

fn validate_cache_shape(cache: &SpdQwen3ForwardCache, shape: &SpdQwen3Shape) -> Result<()> {
    let kv_width = shape.num_key_value_heads * shape.head_dim;
    if cache.layers.len() != shape.num_spec_layers || cache.kv_width != kv_width {
        bail!("SPD Qwen forward cache does not match head shape");
    }
    Ok(())
}

fn apply_fixed_stage_projections(
    weights: &SpdQwen3Weights,
    base_fixed: &mut [f32],
    stage_ids: &[usize],
    layer: usize,
    shape: &SpdQwen3Shape,
) -> Result<()> {
    for (row_idx, stage_id) in stage_ids.iter().enumerate() {
        if *stage_id == 0 {
            continue;
        }
        let projection_idx = shape
            .num_stages
            .checked_sub(*stage_id)
            .context("SPD stage id exceeds num_stages")?;
        let weight = weights
            .fixed_stage_per_layer_projs
            .get(layer)
            .and_then(|layer_weights| layer_weights.get(projection_idx))
            .context("missing cached SPD fixed-stage projection")?;
        let input = row(base_fixed, row_idx, shape.hidden_size).to_vec();
        let output = row_mut(base_fixed, row_idx, shape.hidden_size);
        linear_into(weight, shape.hidden_size, &input, output)?;
    }
    Ok(())
}

fn copy_fixed_rows(
    full_in: &mut [f32],
    base_fixed: &[f32],
    stage_ids: &[usize],
    shape: &SpdQwen3Shape,
) {
    for (row_idx, stage_id) in stage_ids.iter().enumerate() {
        if *stage_id == 0 {
            continue;
        }
        row_mut(full_in, row_idx, shape.hidden_size).copy_from_slice(row(
            base_fixed,
            row_idx,
            shape.hidden_size,
        ));
    }
}

fn decoder_layer_query(
    weights: &SpdQwen3Weights,
    full_in: &[f32],
    position_ids: &[i64],
    layer: usize,
    shape: &SpdQwen3Shape,
) -> Result<Vec<f32>> {
    let seq_len = position_ids.len();
    let layer_weights = weights
        .layers
        .get(layer)
        .context("missing cached SPD decoder layer")?;
    let mut normed = full_in.to_vec();
    for token in 0..seq_len {
        rms_norm_in_place(
            row_mut(&mut normed, token, shape.hidden_size),
            &layer_weights.input_layernorm,
            QWEN3_RMS_NORM_EPS,
        );
    }

    let query_row = seq_len - 1;
    let q = project_query(layer_weights, &normed, query_row, shape)?;
    let k = project_kv(layer_weights, &normed, shape, KvProjection::K)?;
    let v = project_kv(layer_weights, &normed, shape, KvProjection::V)?;
    let attn = attention_query(&q, &k, &v, position_ids, shape);
    let mut attn_hidden = vec![0.0; shape.hidden_size];
    linear_into(
        &layer_weights.o_proj,
        shape.num_attention_heads * shape.head_dim,
        &attn,
        &mut attn_hidden,
    )?;

    let mut hidden = row(full_in, query_row, shape.hidden_size).to_vec();
    add_in_place(&mut hidden, &attn_hidden);

    let mut mlp_in = hidden.clone();
    rms_norm_in_place(
        &mut mlp_in,
        &layer_weights.post_attention_layernorm,
        QWEN3_RMS_NORM_EPS,
    );
    let mlp_out = mlp(layer_weights, &mlp_in, shape)?;
    add_in_place(&mut hidden, &mlp_out);
    Ok(hidden)
}

fn decoder_layer_query_with_cache(
    weights: &SpdQwen3Weights,
    full_in: &[f32],
    position_ids: &[i64],
    layer: usize,
    shape: &SpdQwen3Shape,
    cache: &mut SpdQwen3LayerKvCache,
) -> Result<Vec<f32>> {
    let seq_len = position_ids.len();
    let layer_weights = weights
        .layers
        .get(layer)
        .context("missing cached SPD decoder layer")?;
    let normed = norm_decoder_input(full_in, layer_weights, shape);
    let query_row = seq_len - 1;
    let q = project_query(layer_weights, &normed, query_row, shape)?;
    append_layer_kv_to_cache(weights, full_in, position_ids, layer, shape, cache)?;
    let query_position = *position_ids
        .get(query_row)
        .context("missing SPD query position")?;
    let attn = attention_query_cached(&q, cache, query_position, shape);
    let mut attn_hidden = vec![0.0; shape.hidden_size];
    linear_into(
        &layer_weights.o_proj,
        shape.num_attention_heads * shape.head_dim,
        &attn,
        &mut attn_hidden,
    )?;

    let mut hidden = row(full_in, query_row, shape.hidden_size).to_vec();
    add_in_place(&mut hidden, &attn_hidden);

    let mut mlp_in = hidden.clone();
    rms_norm_in_place(
        &mut mlp_in,
        &layer_weights.post_attention_layernorm,
        QWEN3_RMS_NORM_EPS,
    );
    let mlp_out = mlp(layer_weights, &mlp_in, shape)?;
    add_in_place(&mut hidden, &mlp_out);
    Ok(hidden)
}

fn append_layer_kv_to_cache(
    weights: &SpdQwen3Weights,
    full_in: &[f32],
    position_ids: &[i64],
    layer: usize,
    shape: &SpdQwen3Shape,
    cache: &mut SpdQwen3LayerKvCache,
) -> Result<()> {
    let layer_weights = weights
        .layers
        .get(layer)
        .context("missing cached SPD decoder layer")?;
    let normed = norm_decoder_input(full_in, layer_weights, shape);
    let mut k = project_kv(layer_weights, &normed, shape, KvProjection::K)?;
    apply_rotary_kv(&mut k, position_ids, shape);
    let v = project_kv(layer_weights, &normed, shape, KvProjection::V)?;
    cache.append(
        position_ids,
        &k,
        &v,
        shape.num_key_value_heads * shape.head_dim,
    )
}

fn norm_decoder_input(
    full_in: &[f32],
    layer_weights: &SpdQwen3LayerWeights,
    shape: &SpdQwen3Shape,
) -> Vec<f32> {
    let seq_len = full_in.len() / shape.hidden_size;
    let mut normed = full_in.to_vec();
    for token in 0..seq_len {
        rms_norm_in_place(
            row_mut(&mut normed, token, shape.hidden_size),
            &layer_weights.input_layernorm,
            QWEN3_RMS_NORM_EPS,
        );
    }
    normed
}

fn project_query(
    layer_weights: &SpdQwen3LayerWeights,
    normed: &[f32],
    query_row: usize,
    shape: &SpdQwen3Shape,
) -> Result<Vec<f32>> {
    let mut q = vec![0.0; shape.num_attention_heads * shape.head_dim];
    linear_into(
        &layer_weights.q_proj,
        shape.hidden_size,
        row(normed, query_row, shape.hidden_size),
        &mut q,
    )?;
    for head in 0..shape.num_attention_heads {
        rms_norm_in_place(
            &mut q[head * shape.head_dim..(head + 1) * shape.head_dim],
            &layer_weights.q_norm,
            QWEN3_RMS_NORM_EPS,
        );
    }
    Ok(q)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KvProjection {
    K,
    V,
}

fn project_kv(
    layer_weights: &SpdQwen3LayerWeights,
    normed: &[f32],
    shape: &SpdQwen3Shape,
    projection: KvProjection,
) -> Result<Vec<f32>> {
    let seq_len = normed.len() / shape.hidden_size;
    let weight = match projection {
        KvProjection::K => &layer_weights.k_proj,
        KvProjection::V => &layer_weights.v_proj,
    };
    let mut output = vec![0.0; seq_len * shape.num_key_value_heads * shape.head_dim];
    for token in 0..seq_len {
        linear_into(
            weight,
            shape.hidden_size,
            row(normed, token, shape.hidden_size),
            &mut output[token * shape.num_key_value_heads * shape.head_dim
                ..(token + 1) * shape.num_key_value_heads * shape.head_dim],
        )?;
    }
    if projection == KvProjection::K {
        for token in 0..seq_len {
            for head in 0..shape.num_key_value_heads {
                let start = (token * shape.num_key_value_heads + head) * shape.head_dim;
                rms_norm_in_place(
                    &mut output[start..start + shape.head_dim],
                    &layer_weights.k_norm,
                    QWEN3_RMS_NORM_EPS,
                );
            }
        }
    }
    Ok(output)
}

fn attention_query(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    position_ids: &[i64],
    shape: &SpdQwen3Shape,
) -> Vec<f32> {
    let seq_len = position_ids.len();
    let query_position = *position_ids.last().unwrap_or(&0);
    let mut q = q.to_vec();
    apply_rotary_query(&mut q, query_position, shape);

    let mut k = k.to_vec();
    for (token, position) in position_ids.iter().enumerate().take(seq_len) {
        for head in 0..shape.num_key_value_heads {
            let start = (token * shape.num_key_value_heads + head) * shape.head_dim;
            apply_rotary_head(&mut k[start..start + shape.head_dim], *position, shape);
        }
    }

    let mut output = vec![0.0; shape.num_attention_heads * shape.head_dim];
    let scale = (shape.head_dim as f32).powf(-0.5);
    for head in 0..shape.num_attention_heads {
        let kv_head = head / shape.num_key_value_groups;
        let q_head = &q[head * shape.head_dim..(head + 1) * shape.head_dim];
        let mut scores = vec![0.0; seq_len];
        for (token, score) in scores.iter_mut().enumerate().take(seq_len) {
            let k_start = (token * shape.num_key_value_heads + kv_head) * shape.head_dim;
            *score = dot(q_head, &k[k_start..k_start + shape.head_dim]) * scale;
        }
        softmax_in_place(&mut scores);
        round_slice_to_bf16(&mut scores);
        let out_head = &mut output[head * shape.head_dim..(head + 1) * shape.head_dim];
        for (token, score) in scores.iter().enumerate().take(seq_len) {
            let v_start = (token * shape.num_key_value_heads + kv_head) * shape.head_dim;
            axpy(*score, &v[v_start..v_start + shape.head_dim], out_head);
        }
        round_slice_to_bf16(out_head);
    }
    output
}

fn attention_query_cached(
    q: &[f32],
    cache: &SpdQwen3LayerKvCache,
    query_position: i64,
    shape: &SpdQwen3Shape,
) -> Vec<f32> {
    let mut q = q.to_vec();
    apply_rotary_query(&mut q, query_position, shape);
    let eligible = cache
        .positions
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(idx, position)| (position <= query_position).then_some(idx))
        .collect::<Vec<_>>();
    let mut output = vec![0.0; shape.num_attention_heads * shape.head_dim];
    if eligible.is_empty() {
        return output;
    }
    let scale = (shape.head_dim as f32).powf(-0.5);
    for head in 0..shape.num_attention_heads {
        let kv_head = head / shape.num_key_value_groups;
        let q_head = &q[head * shape.head_dim..(head + 1) * shape.head_dim];
        let mut scores = vec![0.0; eligible.len()];
        for (score_idx, token) in eligible.iter().copied().enumerate() {
            let k_start = (token * shape.num_key_value_heads + kv_head) * shape.head_dim;
            scores[score_idx] = dot(q_head, &cache.k[k_start..k_start + shape.head_dim]) * scale;
        }
        softmax_in_place(&mut scores);
        round_slice_to_bf16(&mut scores);
        let out_head = &mut output[head * shape.head_dim..(head + 1) * shape.head_dim];
        for (score, token) in scores.iter().copied().zip(eligible.iter().copied()) {
            let v_start = (token * shape.num_key_value_heads + kv_head) * shape.head_dim;
            axpy(score, &cache.v[v_start..v_start + shape.head_dim], out_head);
        }
        round_slice_to_bf16(out_head);
    }
    output
}

fn mlp(
    layer_weights: &SpdQwen3LayerWeights,
    input: &[f32],
    shape: &SpdQwen3Shape,
) -> Result<Vec<f32>> {
    let intermediate = layer_weights.gate_proj.len() / shape.hidden_size;
    let mut gate = vec![0.0; intermediate];
    linear_into(
        &layer_weights.gate_proj,
        shape.hidden_size,
        input,
        &mut gate,
    )?;
    for value in &mut gate {
        *value = round_to_bf16(silu(*value));
    }

    let mut up = vec![0.0; intermediate];
    linear_into(&layer_weights.up_proj, shape.hidden_size, input, &mut up)?;
    for (gate_value, up_value) in gate.iter_mut().zip(up) {
        *gate_value = round_to_bf16(*gate_value * up_value);
    }

    let mut output = vec![0.0; shape.hidden_size];
    linear_into(&layer_weights.down_proj, intermediate, &gate, &mut output)?;
    Ok(output)
}

fn lm_head_logits(weights: &SpdQwen3Weights, hidden: &[f32]) -> Result<Vec<f32>> {
    let vocab = weights.lm_head.len() / hidden.len();
    let mut logits = vec![0.0; vocab];
    linear_into(&weights.lm_head, hidden.len(), hidden, &mut logits)?;
    Ok(logits)
}

fn lm_head_topk(
    weights: &SpdQwen3Weights,
    hidden: &[f32],
    top_k: usize,
    draft_token_ids: Option<&[u32]>,
) -> Result<SpdQwen3FixtureTopK> {
    if hidden.is_empty() {
        bail!("SPD lm_head hidden input must not be empty");
    }
    if !weights.lm_head.len().is_multiple_of(hidden.len()) {
        bail!(
            "SPD lm_head shape mismatch: weight len {}, hidden {}",
            weights.lm_head.len(),
            hidden.len()
        );
    }
    let vocab = weights.lm_head.len() / hidden.len();
    let limit = top_k.min(vocab);
    let pairs = parallel_lm_head_topk_pairs(&weights.lm_head, hidden, limit);
    topk_pairs_to_result(pairs, draft_token_ids)
}

fn parallel_lm_head_topk_pairs(weight: &[f32], hidden: &[f32], limit: usize) -> Vec<(usize, f32)> {
    if limit == 0 {
        return Vec::new();
    }
    let vocab = weight.len() / hidden.len();
    let workers = thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1)
        .min(vocab);
    let rows_per_worker = vocab.div_ceil(workers);
    let mut merged = Vec::with_capacity(limit);
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for chunk_idx in 0..workers {
            let first_row = chunk_idx * rows_per_worker;
            if first_row >= vocab {
                break;
            }
            let row_count = rows_per_worker.min(vocab - first_row);
            let weight_start = first_row * hidden.len();
            let weight_end = weight_start + row_count * hidden.len();
            let weight_chunk = &weight[weight_start..weight_end];
            handles.push(
                scope.spawn(move || lm_head_topk_chunk(weight_chunk, hidden, first_row, limit)),
            );
        }
        for handle in handles {
            let partial = handle.join().expect("SPD lm_head top-k worker panicked");
            merge_topk_pairs(&mut merged, partial, limit);
        }
    });
    merged
}

fn lm_head_topk_chunk(
    weight: &[f32],
    hidden: &[f32],
    first_row: usize,
    limit: usize,
) -> Vec<(usize, f32)> {
    let mut best = Vec::with_capacity(limit);
    for (offset, weight_row) in weight.chunks_exact(hidden.len()).enumerate() {
        let logit = round_to_bf16(dot(weight_row, hidden));
        insert_topk_pair(&mut best, (first_row + offset, logit), limit);
    }
    best
}

fn topk_from_logits(
    logits: &[f32],
    top_k: usize,
    draft_token_ids: Option<&[u32]>,
) -> Result<SpdQwen3FixtureTopK> {
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    pairs.truncate(top_k);
    topk_pairs_to_result(pairs, draft_token_ids)
}

fn topk_pairs_to_result(
    pairs: Vec<(usize, f32)>,
    draft_token_ids: Option<&[u32]>,
) -> Result<SpdQwen3FixtureTopK> {
    let draft_indices: Vec<i64> = pairs.iter().map(|(idx, _)| *idx as i64).collect();
    let token_ids = match draft_token_ids {
        Some(ids) => draft_indices
            .iter()
            .map(|idx| {
                let idx = usize::try_from(*idx).context("negative draft index")?;
                ids.get(idx)
                    .copied()
                    .map(i64::from)
                    .with_context(|| format!("draft index {idx} missing from draft_token_ids"))
            })
            .collect::<Result<Vec<_>>>()?,
        None => draft_indices.clone(),
    };
    let logits = pairs.iter().map(|(_, value)| *value).collect();
    Ok(SpdQwen3FixtureTopK {
        draft_indices,
        token_ids,
        logits,
    })
}

fn merge_topk_pairs(target: &mut Vec<(usize, f32)>, source: Vec<(usize, f32)>, limit: usize) {
    for pair in source {
        insert_topk_pair(target, pair, limit);
    }
}

fn insert_topk_pair(best: &mut Vec<(usize, f32)>, candidate: (usize, f32), limit: usize) {
    if limit == 0 {
        return;
    }
    let position = best
        .iter()
        .position(|existing| topk_pair_precedes(&candidate, existing))
        .unwrap_or(best.len());
    if position < limit {
        best.insert(position, candidate);
        if best.len() > limit {
            best.pop();
        }
    } else if best.len() < limit {
        best.push(candidate);
    }
}

fn topk_pair_precedes(left: &(usize, f32), right: &(usize, f32)) -> bool {
    right
        .1
        .total_cmp(&left.1)
        .then_with(|| left.0.cmp(&right.0))
        .is_lt()
}

fn python_topk_from_fixture(fixture_file: &SpdSafetensorsFile) -> Result<SpdQwen3FixtureTopK> {
    Ok(SpdQwen3FixtureTopK {
        draft_indices: fixture_file.read_tensor_i64("python_topk_draft_indices")?,
        token_ids: fixture_file.read_tensor_i64("python_topk_token_ids")?,
        logits: fixture_file.read_tensor_f32("python_topk_logits")?,
    })
}

fn python_cached_topk_from_fixture(
    fixture_file: &SpdSafetensorsFile,
) -> Result<SpdQwen3FixtureTopK> {
    Ok(SpdQwen3FixtureTopK {
        draft_indices: fixture_file.read_tensor_i64("python_cached_topk_draft_indices")?,
        token_ids: fixture_file.read_tensor_i64("python_cached_topk_token_ids")?,
        logits: fixture_file.read_tensor_f32("python_cached_topk_logits")?,
    })
}

fn fixture_diagnostics(
    fixture_file: &SpdSafetensorsFile,
    trace: &SpdQwen3ForwardTrace,
    rust_topk: &SpdQwen3FixtureTopK,
) -> Result<SpdQwen3FixtureDiagnostics> {
    let mut layer_input_max_abs_diff = Vec::with_capacity(trace.layer_inputs.len());
    for (idx, rust_layer) in trace.layer_inputs.iter().enumerate() {
        let python_layer = fixture_file.read_tensor_f32(&format!("python_layer_{idx}_full_in"))?;
        layer_input_max_abs_diff.push(max_abs_diff(rust_layer, &python_layer)?);
    }
    let mut layer_query_max_abs_diff = Vec::with_capacity(trace.layer_queries.len());
    for (idx, rust_layer) in trace.layer_queries.iter().enumerate() {
        let python_layer = fixture_file.read_tensor_f32(&format!("python_layer_{idx}_query"))?;
        layer_query_max_abs_diff.push(max_abs_diff(rust_layer, &python_layer)?);
    }
    let python_spec_query = fixture_file.read_tensor_f32("python_spec_query")?;
    let python_final_hidden = fixture_file.read_tensor_f32("python_final_hidden")?;
    let python_logits = fixture_file.read_tensor_f32("python_logits")?;
    let python_top_logit_values_at_rust_indices = rust_topk
        .draft_indices
        .iter()
        .map(|idx| {
            let idx = usize::try_from(*idx).context("negative rust draft index")?;
            python_logits
                .get(idx)
                .copied()
                .with_context(|| format!("rust draft index {idx} missing from python logits"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(SpdQwen3FixtureDiagnostics {
        layer_input_max_abs_diff,
        layer_query_max_abs_diff,
        spec_query_max_abs_diff: max_abs_diff(&trace.spec_query, &python_spec_query)?,
        final_hidden_max_abs_diff: max_abs_diff(&trace.final_hidden, &python_final_hidden)?,
        python_top_logit_values_at_rust_indices,
    })
}

fn cached_fixture_diagnostics(
    fixture_file: &SpdSafetensorsFile,
    trace: &SpdQwen3ForwardTrace,
    rust_topk: &SpdQwen3FixtureTopK,
    cache_prefix_len: usize,
) -> Result<SpdQwen3CachedFixtureDiagnostics> {
    let python_spec_query = fixture_file.read_tensor_f32("python_cached_spec_query")?;
    let python_final_hidden = fixture_file.read_tensor_f32("python_cached_final_hidden")?;
    let python_logits = fixture_file.read_tensor_f32("python_cached_logits")?;
    let python_top_logit_values_at_rust_indices = rust_topk
        .draft_indices
        .iter()
        .map(|idx| {
            let idx = usize::try_from(*idx).context("negative rust draft index")?;
            python_logits.get(idx).copied().with_context(|| {
                format!("rust draft index {idx} missing from python cached logits")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(SpdQwen3CachedFixtureDiagnostics {
        cache_prefix_len,
        spec_query_max_abs_diff: max_abs_diff(&trace.spec_query, &python_spec_query)?,
        final_hidden_max_abs_diff: max_abs_diff(&trace.final_hidden, &python_final_hidden)?,
        logits_max_abs_diff: max_abs_diff(&trace.logits, &python_logits)?,
        python_top_logit_values_at_rust_indices,
    })
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        bail!(
            "SPD diagnostic vector length mismatch: {} vs {}",
            left.len(),
            right.len()
        );
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max))
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

impl SpdQwen3Shape {
    fn from_manifest_and_weights(
        manifest: &SpdHeadManifest,
        serving_file: &SpdSafetensorsFile,
    ) -> Result<Self> {
        let hidden_size = manifest.topology.hidden_size as usize;
        let q_shape = &serving_file
            .index
            .tensor("spec_layers.0.self_attn.q_proj.weight")?
            .shape;
        let k_shape = &serving_file
            .index
            .tensor("spec_layers.0.self_attn.k_proj.weight")?
            .shape;
        let q_norm_shape = &serving_file
            .index
            .tensor("spec_layers.0.self_attn.q_norm.weight")?
            .shape;
        if q_shape.len() != 2 || k_shape.len() != 2 || q_norm_shape.len() != 1 {
            bail!("unsupported SPD Qwen attention tensor shapes");
        }
        let head_dim = q_norm_shape[0] as usize;
        let q_out = q_shape[0] as usize;
        let k_out = k_shape[0] as usize;
        if q_shape[1] != hidden_size as u64 || k_shape[1] != hidden_size as u64 {
            bail!("SPD Qwen projection input dims must match hidden_size");
        }
        if !q_out.is_multiple_of(head_dim) || !k_out.is_multiple_of(head_dim) {
            bail!("SPD Qwen projection output dims must be divisible by head_dim");
        }
        let num_attention_heads = q_out / head_dim;
        let num_key_value_heads = k_out / head_dim;
        if !num_attention_heads.is_multiple_of(num_key_value_heads) {
            bail!("SPD Qwen attention heads must be divisible by KV heads");
        }
        Ok(Self {
            hidden_size,
            num_stages: manifest.topology.num_stages as usize,
            num_spec_layers: manifest.topology.num_spec_layers as usize,
            num_attention_heads,
            num_key_value_heads,
            num_key_value_groups: num_attention_heads / num_key_value_heads,
            head_dim,
            rotary_dim: (head_dim as f32 * QWEN35_PARTIAL_ROTARY_FACTOR) as usize,
        })
    }
}

fn infer_stage_ids(seq_len: usize, num_stages: usize) -> Vec<usize> {
    if seq_len == num_stages + 1 {
        return (0..=num_stages).rev().collect();
    }
    if seq_len == num_stages {
        return (0..num_stages).rev().collect();
    }
    vec![num_stages; seq_len]
}

fn row(values: &[f32], row_idx: usize, width: usize) -> &[f32] {
    &values[row_idx * width..(row_idx + 1) * width]
}

fn row_mut(values: &mut [f32], row_idx: usize, width: usize) -> &mut [f32] {
    &mut values[row_idx * width..(row_idx + 1) * width]
}

fn linear_into(
    weight: &[f32],
    input_width: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<()> {
    if input.len() != input_width {
        bail!(
            "SPD linear input width mismatch: expected {}, got {}",
            input_width,
            input.len()
        );
    }
    if weight.len() != output.len() * input_width {
        bail!(
            "SPD linear weight shape mismatch: weight len {}, output {}, input {}",
            weight.len(),
            output.len(),
            input_width
        );
    }
    if should_parallelize_linear(output.len(), input_width) {
        parallel_linear_into(weight, input_width, input, output);
        return Ok(());
    }
    serial_linear_into(weight, input_width, input, output);
    Ok(())
}

fn should_parallelize_linear(output_width: usize, input_width: usize) -> bool {
    thread::available_parallelism().is_ok_and(|parallelism| parallelism.get() > 1)
        && output_width.saturating_mul(input_width) >= PARALLEL_LINEAR_MIN_DOT_OPS
}

fn serial_linear_into(weight: &[f32], input_width: usize, input: &[f32], output: &mut [f32]) {
    for (out_idx, out) in output.iter_mut().enumerate() {
        let weight_row = &weight[out_idx * input_width..(out_idx + 1) * input_width];
        *out = round_to_bf16(dot(weight_row, input));
    }
}

fn parallel_linear_into(weight: &[f32], input_width: usize, input: &[f32], output: &mut [f32]) {
    let workers = thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1)
        .min(output.len());
    let rows_per_worker = output.len().div_ceil(workers);
    thread::scope(|scope| {
        for (chunk_idx, output_chunk) in output.chunks_mut(rows_per_worker).enumerate() {
            let first_row = chunk_idx * rows_per_worker;
            let weight_start = first_row * input_width;
            let weight_end = weight_start + output_chunk.len() * input_width;
            let weight_chunk = &weight[weight_start..weight_end];
            scope.spawn(move || {
                serial_linear_into(weight_chunk, input_width, input, output_chunk);
            });
        }
    });
}

fn rms_norm_in_place(values: &mut [f32], weight: &[f32], eps: f32) {
    let sum_sq: f32 = values.iter().map(|value| value * value).sum();
    let scale = (sum_sq / values.len() as f32 + eps).sqrt().recip();
    for (value, weight) in values.iter_mut().zip(weight) {
        *value = round_to_bf16(*value * scale * *weight);
    }
}

fn qwen35_final_norm_in_place(values: &mut [f32], weight: &[f32], eps: f32) {
    let sum_sq: f32 = values.iter().map(|value| value * value).sum();
    let scale = (sum_sq / values.len() as f32 + eps).sqrt().recip();
    for (value, weight) in values.iter_mut().zip(weight) {
        *value = round_to_bf16(*value * scale * (1.0 + *weight));
    }
}

fn apply_rotary_query(q: &mut [f32], position: i64, shape: &SpdQwen3Shape) {
    for head in 0..shape.num_attention_heads {
        let start = head * shape.head_dim;
        apply_rotary_head(&mut q[start..start + shape.head_dim], position, shape);
    }
}

fn apply_rotary_kv(k: &mut [f32], position_ids: &[i64], shape: &SpdQwen3Shape) {
    for (token, position) in position_ids.iter().copied().enumerate() {
        for head in 0..shape.num_key_value_heads {
            let start = (token * shape.num_key_value_heads + head) * shape.head_dim;
            apply_rotary_head(&mut k[start..start + shape.head_dim], position, shape);
        }
    }
}

fn apply_rotary_head(values: &mut [f32], position: i64, shape: &SpdQwen3Shape) {
    let rotary_dim = shape.rotary_dim;
    let half = rotary_dim / 2;
    for pair in 0..half {
        let freq = rope_frequency(pair, position, rotary_dim);
        let cos = freq.cos();
        let sin = freq.sin();
        let left = values[pair];
        let right = values[pair + half];
        values[pair] = round_to_bf16(left * cos - right * sin);
        values[pair + half] = round_to_bf16(right * cos + left * sin);
    }
}

fn rope_frequency(pair: usize, position: i64, rotary_dim: usize) -> f32 {
    let exponent = (2 * pair) as f32 / rotary_dim as f32;
    position as f32 / QWEN35_ROPE_THETA.powf(exponent)
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum != 0.0 {
        for value in values {
            *value /= sum;
        }
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn add_in_place(left: &mut [f32], right: &[f32]) {
    for (left, right) in left.iter_mut().zip(right) {
        *left = round_to_bf16(*left + *right);
    }
}

fn axpy(scale: f32, input: &[f32], output: &mut [f32]) {
    for (out, input) in output.iter_mut().zip(input) {
        *out += scale * *input;
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn round_slice_to_bf16(values: &mut [f32]) {
    for value in values {
        *value = round_to_bf16(*value);
    }
}

fn round_to_bf16(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb) & 0xffff_0000;
    f32::from_bits(rounded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_forward_matches_stateless_without_prefix() {
        let shape = test_shape();
        let weights = test_weights(&shape);
        let input = test_input(&shape, &[0, 1, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let stateless = run_forward_hidden(&weights, input.clone(), &shape).unwrap();
        let mut cache = SpdQwen3ForwardCache::new(&shape);
        let cached = run_forward_hidden_with_cache(&weights, input, &shape, &mut cache).unwrap();

        assert_eq!(cached, stateless);
        assert_eq!(cache.layers[0].positions, vec![0, 1, 2]);
    }

    #[test]
    fn prefill_cache_appends_fixed_rows_and_forward_extends_it() {
        let shape = test_shape();
        let weights = test_weights(&shape);
        let mut cache = SpdQwen3ForwardCache::new(&shape);
        let prefill = test_input(&shape, &[0, 1], &[1.0, 0.0, 0.0, 1.0]);
        prefill_forward_cache(&weights, prefill, &shape, &mut cache).unwrap();
        assert_eq!(cache.cached_prefix_len(), 2);

        let forward = test_input(&shape, &[2, 3, 4], &[1.0, 1.0, 2.0, 1.0, 1.0, 2.0]);
        run_forward_hidden_with_cache(&weights, forward, &shape, &mut cache).unwrap();
        assert_eq!(cache.layers[0].positions, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn cached_forward_crops_replaced_rolling_rows() {
        let shape = test_shape();
        let weights = test_weights(&shape);
        let mut cache = SpdQwen3ForwardCache::new(&shape);
        prefill_forward_cache(
            &weights,
            test_input(&shape, &[0, 1], &[1.0, 0.0, 0.0, 1.0]),
            &shape,
            &mut cache,
        )
        .unwrap();
        run_forward_hidden_with_cache(
            &weights,
            test_input(&shape, &[2, 3, 4], &[1.0, 1.0, 2.0, 1.0, 1.0, 2.0]),
            &shape,
            &mut cache,
        )
        .unwrap();
        run_forward_hidden_with_cache(
            &weights,
            test_input(&shape, &[3, 4, 5], &[2.0, 1.0, 1.0, 2.0, 3.0, 1.0]),
            &shape,
            &mut cache,
        )
        .unwrap();

        assert_eq!(cache.layers[0].positions, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn forward_rejects_stage_id_length_mismatch() {
        let shape = test_shape();
        let weights = test_weights(&shape);
        let mut input = test_input(&shape, &[0, 1, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        input.fixed_stage_ids = Some(vec![shape.num_stages; input.seq_len - 1]);

        let error = run_forward_hidden(&weights, input, &shape)
            .expect_err("missing stage id should fail validation")
            .to_string();

        assert!(error.contains("fixed_stage_ids length"));
    }

    #[test]
    fn forward_rejects_stage_id_above_topology() {
        let shape = test_shape();
        let weights = test_weights(&shape);
        let mut input = test_input(&shape, &[0, 1, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        input.fixed_stage_ids = Some(vec![shape.num_stages + 1, shape.num_stages, 0]);

        let error = run_forward_hidden(&weights, input, &shape)
            .expect_err("out-of-range stage id should fail validation")
            .to_string();

        assert!(error.contains("exceeds num_stages"));
    }

    fn test_input(
        shape: &SpdQwen3Shape,
        positions: &[i64],
        cur_in: &[f32],
    ) -> SpdQwen3ForwardInput {
        assert_eq!(cur_in.len(), positions.len() * shape.hidden_size);
        SpdQwen3ForwardInput {
            cur_in: cur_in.to_vec(),
            seq_len: positions.len(),
            position_ids: positions.to_vec(),
            fixed_stage_ids: None,
            final_norm_weight: vec![0.0; shape.hidden_size],
        }
    }

    fn test_shape() -> SpdQwen3Shape {
        SpdQwen3Shape {
            hidden_size: 2,
            num_stages: 2,
            num_spec_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            num_key_value_groups: 1,
            head_dim: 2,
            rotary_dim: 0,
        }
    }

    fn test_weights(shape: &SpdQwen3Shape) -> SpdQwen3Weights {
        SpdQwen3Weights {
            fixed_stage_per_layer_projs: vec![vec![identity(shape.hidden_size); shape.num_stages]],
            layers: vec![SpdQwen3LayerWeights {
                input_layernorm: vec![1.0; shape.hidden_size],
                q_proj: identity(shape.hidden_size),
                q_norm: vec![1.0; shape.head_dim],
                k_proj: identity(shape.hidden_size),
                k_norm: vec![1.0; shape.head_dim],
                v_proj: identity(shape.hidden_size),
                o_proj: identity(shape.hidden_size),
                post_attention_layernorm: vec![1.0; shape.hidden_size],
                gate_proj: vec![0.0; shape.hidden_size * shape.hidden_size],
                up_proj: vec![0.0; shape.hidden_size * shape.hidden_size],
                down_proj: vec![0.0; shape.hidden_size * shape.hidden_size],
            }],
            lm_head: identity(shape.hidden_size),
        }
    }

    fn identity(width: usize) -> Vec<f32> {
        let mut values = vec![0.0; width * width];
        for idx in 0..width {
            values[idx * width + idx] = 1.0;
        }
        values
    }
}
