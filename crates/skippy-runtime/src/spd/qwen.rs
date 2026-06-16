use std::path::Path;

use anyhow::{Context, Result, bail};

use super::{SpdHeadManifest, SpdSafetensorsFile};

const QWEN3_RMS_NORM_EPS: f32 = 1.0e-6;
const QWEN35_ROPE_THETA: f32 = 10_000_000.0;
const QWEN35_PARTIAL_ROTARY_FACTOR: f32 = 0.25;

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
pub struct SpdQwen3FixtureDiagnostics {
    pub layer_input_max_abs_diff: Vec<f32>,
    pub layer_query_max_abs_diff: Vec<f32>,
    pub spec_query_max_abs_diff: f32,
    pub final_hidden_max_abs_diff: f32,
    pub python_top_logit_values_at_rust_indices: Vec<f32>,
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

pub fn run_qwen3_fixture_parity(
    manifest_path: impl AsRef<Path>,
    fixture_path: impl AsRef<Path>,
    top_k: usize,
) -> Result<SpdQwen3FixtureParity> {
    let manifest_path = manifest_path.as_ref();
    let fixture_path = fixture_path.as_ref();
    let manifest = SpdHeadManifest::from_path(manifest_path)?;
    manifest.ensure_serving_checkpoint_for_runtime(manifest_path)?;
    let serving_path = manifest.serving_checkpoint_path(manifest_path)?;
    let serving_file = SpdSafetensorsFile::open(&serving_path)?;
    let fixture_file = SpdSafetensorsFile::open(fixture_path)?;
    let shape = SpdQwen3Shape::from_manifest_and_weights(&manifest, &serving_file)?;
    let trace = run_fixture_forward(&serving_file, &fixture_file, &shape)?;
    let rust = topk_from_logits(
        &trace.logits,
        top_k,
        manifest.topology.draft_token_ids.as_deref(),
    )?;
    let python = python_topk_from_fixture(&fixture_file)?;
    let diagnostics = fixture_diagnostics(&fixture_file, &trace, &rust)?;
    Ok(SpdQwen3FixtureParity {
        rust,
        python,
        diagnostics,
    })
}

fn run_fixture_forward(
    serving_file: &SpdSafetensorsFile,
    fixture_file: &SpdSafetensorsFile,
    shape: &SpdQwen3Shape,
) -> Result<SpdQwen3ForwardTrace> {
    let cur_in = fixture_file.read_tensor_f32("cur_in")?;
    let cur_shape = &fixture_file.index.tensor("cur_in")?.shape;
    if cur_shape.len() != 3 || cur_shape[0] != 1 || cur_shape[2] != shape.hidden_size as u64 {
        bail!(
            "SPD fixture cur_in shape {:?} is not [1, seq, hidden]",
            cur_shape
        );
    }
    let seq_len = usize::try_from(cur_shape[1]).context("SPD fixture sequence length too large")?;
    let position_ids = fixture_file.read_tensor_i64("position_ids")?;
    if position_ids.len() != seq_len {
        bail!(
            "SPD fixture position_ids length {} must match cur_in seq_len {}",
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

    let stage_ids = infer_stage_ids(seq_len, shape.num_stages);
    let original_hidden = cur_in.clone();
    let mut base_fixed = cur_in.clone();
    let mut query = row(&cur_in, seq_len - 1, shape.hidden_size).to_vec();
    let mut layer_inputs = Vec::with_capacity(shape.num_spec_layers);
    let mut layer_queries = Vec::with_capacity(shape.num_spec_layers);

    for layer in 0..shape.num_spec_layers {
        apply_fixed_stage_projections(serving_file, &mut base_fixed, &stage_ids, layer, shape)?;
        let mut full_in = original_hidden.clone();
        copy_fixed_rows(&mut full_in, &base_fixed, &stage_ids, shape);
        full_in[(seq_len - 1) * shape.hidden_size..seq_len * shape.hidden_size]
            .copy_from_slice(&query);
        layer_inputs.push(full_in.clone());
        query = decoder_layer_query(serving_file, &full_in, &position_ids, layer, shape)?;
        layer_queries.push(query.clone());
    }

    let spec_query = query.clone();
    qwen35_final_norm_in_place(&mut query, &final_norm_weight, QWEN3_RMS_NORM_EPS);
    let final_hidden = query.clone();
    let logits = lm_head_logits(serving_file, &query)?;
    Ok(SpdQwen3ForwardTrace {
        logits,
        layer_inputs,
        layer_queries,
        spec_query,
        final_hidden,
    })
}

fn apply_fixed_stage_projections(
    serving_file: &SpdSafetensorsFile,
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
        let weight = serving_file.read_tensor_f32(&format!(
            "fixed_stage_per_layer_projs.{layer}.{projection_idx}.weight"
        ))?;
        let input = row(base_fixed, row_idx, shape.hidden_size).to_vec();
        let output = row_mut(base_fixed, row_idx, shape.hidden_size);
        linear_into(&weight, shape.hidden_size, &input, output)?;
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
    serving_file: &SpdSafetensorsFile,
    full_in: &[f32],
    position_ids: &[i64],
    layer: usize,
    shape: &SpdQwen3Shape,
) -> Result<Vec<f32>> {
    let seq_len = position_ids.len();
    let input_norm_weight =
        serving_file.read_tensor_f32(&format!("spec_layers.{layer}.input_layernorm.weight"))?;
    let mut normed = full_in.to_vec();
    for token in 0..seq_len {
        rms_norm_in_place(
            row_mut(&mut normed, token, shape.hidden_size),
            &input_norm_weight,
            QWEN3_RMS_NORM_EPS,
        );
    }

    let query_row = seq_len - 1;
    let q = project_query(serving_file, &normed, query_row, layer, shape)?;
    let k = project_kv(serving_file, &normed, layer, shape, "k_proj")?;
    let v = project_kv(serving_file, &normed, layer, shape, "v_proj")?;
    let attn = attention_query(&q, &k, &v, position_ids, shape);
    let o_proj =
        serving_file.read_tensor_f32(&format!("spec_layers.{layer}.self_attn.o_proj.weight"))?;
    let mut attn_hidden = vec![0.0; shape.hidden_size];
    linear_into(
        &o_proj,
        shape.num_attention_heads * shape.head_dim,
        &attn,
        &mut attn_hidden,
    )?;

    let mut hidden = row(full_in, query_row, shape.hidden_size).to_vec();
    add_in_place(&mut hidden, &attn_hidden);

    let post_norm_weight = serving_file.read_tensor_f32(&format!(
        "spec_layers.{layer}.post_attention_layernorm.weight"
    ))?;
    let mut mlp_in = hidden.clone();
    rms_norm_in_place(&mut mlp_in, &post_norm_weight, QWEN3_RMS_NORM_EPS);
    let mlp_out = mlp(serving_file, &mlp_in, layer, shape)?;
    add_in_place(&mut hidden, &mlp_out);
    Ok(hidden)
}

fn project_query(
    serving_file: &SpdSafetensorsFile,
    normed: &[f32],
    query_row: usize,
    layer: usize,
    shape: &SpdQwen3Shape,
) -> Result<Vec<f32>> {
    let q_proj =
        serving_file.read_tensor_f32(&format!("spec_layers.{layer}.self_attn.q_proj.weight"))?;
    let mut q = vec![0.0; shape.num_attention_heads * shape.head_dim];
    linear_into(
        &q_proj,
        shape.hidden_size,
        row(normed, query_row, shape.hidden_size),
        &mut q,
    )?;
    let q_norm =
        serving_file.read_tensor_f32(&format!("spec_layers.{layer}.self_attn.q_norm.weight"))?;
    for head in 0..shape.num_attention_heads {
        rms_norm_in_place(
            &mut q[head * shape.head_dim..(head + 1) * shape.head_dim],
            &q_norm,
            QWEN3_RMS_NORM_EPS,
        );
    }
    Ok(q)
}

fn project_kv(
    serving_file: &SpdSafetensorsFile,
    normed: &[f32],
    layer: usize,
    shape: &SpdQwen3Shape,
    projection: &str,
) -> Result<Vec<f32>> {
    let seq_len = normed.len() / shape.hidden_size;
    let weight = serving_file.read_tensor_f32(&format!(
        "spec_layers.{layer}.self_attn.{projection}.weight"
    ))?;
    let mut output = vec![0.0; seq_len * shape.num_key_value_heads * shape.head_dim];
    for token in 0..seq_len {
        linear_into(
            &weight,
            shape.hidden_size,
            row(normed, token, shape.hidden_size),
            &mut output[token * shape.num_key_value_heads * shape.head_dim
                ..(token + 1) * shape.num_key_value_heads * shape.head_dim],
        )?;
    }
    if projection == "k_proj" {
        let k_norm = serving_file
            .read_tensor_f32(&format!("spec_layers.{layer}.self_attn.k_norm.weight"))?;
        for token in 0..seq_len {
            for head in 0..shape.num_key_value_heads {
                let start = (token * shape.num_key_value_heads + head) * shape.head_dim;
                rms_norm_in_place(
                    &mut output[start..start + shape.head_dim],
                    &k_norm,
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

fn mlp(
    serving_file: &SpdSafetensorsFile,
    input: &[f32],
    layer: usize,
    shape: &SpdQwen3Shape,
) -> Result<Vec<f32>> {
    let gate_weight =
        serving_file.read_tensor_f32(&format!("spec_layers.{layer}.mlp.gate_proj.weight"))?;
    let intermediate = gate_weight.len() / shape.hidden_size;
    let mut gate = vec![0.0; intermediate];
    linear_into(&gate_weight, shape.hidden_size, input, &mut gate)?;
    for value in &mut gate {
        *value = round_to_bf16(silu(*value));
    }

    let up_weight =
        serving_file.read_tensor_f32(&format!("spec_layers.{layer}.mlp.up_proj.weight"))?;
    let mut up = vec![0.0; intermediate];
    linear_into(&up_weight, shape.hidden_size, input, &mut up)?;
    for (gate_value, up_value) in gate.iter_mut().zip(up) {
        *gate_value = round_to_bf16(*gate_value * up_value);
    }

    let down_weight =
        serving_file.read_tensor_f32(&format!("spec_layers.{layer}.mlp.down_proj.weight"))?;
    let mut output = vec![0.0; shape.hidden_size];
    linear_into(&down_weight, intermediate, &gate, &mut output)?;
    Ok(output)
}

fn lm_head_logits(serving_file: &SpdSafetensorsFile, hidden: &[f32]) -> Result<Vec<f32>> {
    let lm_head = serving_file.read_tensor_f32("lm_head.weight")?;
    let vocab = lm_head.len() / hidden.len();
    let mut logits = vec![0.0; vocab];
    linear_into(&lm_head, hidden.len(), hidden, &mut logits)?;
    Ok(logits)
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

fn python_topk_from_fixture(fixture_file: &SpdSafetensorsFile) -> Result<SpdQwen3FixtureTopK> {
    Ok(SpdQwen3FixtureTopK {
        draft_indices: fixture_file.read_tensor_i64("python_topk_draft_indices")?,
        token_ids: fixture_file.read_tensor_i64("python_topk_token_ids")?,
        logits: fixture_file.read_tensor_f32("python_topk_logits")?,
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
    for (out_idx, out) in output.iter_mut().enumerate() {
        let weight_row = &weight[out_idx * input_width..(out_idx + 1) * input_width];
        *out = round_to_bf16(dot(weight_row, input));
    }
    Ok(())
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
