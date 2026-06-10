//! Llama / Qwen forward pass in Rust over `mlx-c`.
//!
//! This is a direct transcription of the reference Python (`mlx-lm/models/
//! llama.py`, `qwen3.py`) and Swift (`mlx-swift-lm`) implementations: the same
//! op sequence (proj → reshape → RoPE → SDPA → proj; SwiGLU MLP; RMSNorm;
//! residuals) over the same fused kernels. Qwen adds q/k/v attention bias
//! (Qwen2) and q/k norm (Qwen3); those are detected from config + weights.
//!
//! Pipeline parallelism: each rank only runs the layers it owns (per
//! [`Pipeline`]); activation hand-off via the [`Group`] is done by the caller in
//! the runtime, so this forward takes the already-received hidden state and
//! returns this stage's output.

use crate::Result;
use crate::array::{Array, Stream};
use crate::distributed::Pipeline;
use crate::models::config::{Family, ModelConfig};
use crate::nn::{Embedding, Linear, RmsNorm, Weights};
use crate::ops;

/// A loaded Llama-family model bound to a set of weights.
pub struct LlamaModel<'w> {
    cfg: &'w ModelConfig,
    weights: &'w Weights,
    family: Family,
    /// Layer ownership for pipeline parallelism (full range when single rank).
    pipeline: Pipeline,
}

/// Per-layer KV cache (keys/values concatenated along the sequence axis).
#[derive(Default)]
pub struct LayerCache {
    pub keys: Option<Array>,
    pub values: Option<Array>,
    pub offset: i32,
}

impl<'w> LlamaModel<'w> {
    pub fn new(cfg: &'w ModelConfig, weights: &'w Weights, pipeline: Pipeline) -> Self {
        LlamaModel {
            cfg,
            weights,
            family: cfg.family(),
            pipeline,
        }
    }

    /// Fresh per-layer caches for the layers this rank owns.
    pub fn new_cache(&self) -> Vec<LayerCache> {
        (0..self.pipeline.range.len())
            .map(|_| LayerCache::default())
            .collect()
    }

    /// Embed token ids `[B, L]` (i32) into hidden states. Only meaningful on the
    /// first forward stage (which owns the embedding).
    pub fn embed(&self, ids: &Array, s: &Stream) -> Result<Array> {
        Embedding::load(self.weights, "model.embed_tokens")?.forward(ids, s)
    }

    /// Run this rank's owned layers over hidden state `h` `[B, L, D]`.
    pub fn forward_layers(
        &self,
        mut h: Array,
        cache: &mut [LayerCache],
        s: &Stream,
    ) -> Result<Array> {
        for (local_idx, global_idx) in
            (self.pipeline.range.start..self.pipeline.range.end).enumerate()
        {
            h = self.layer(global_idx, h, &mut cache[local_idx], s)?;
        }
        Ok(h)
    }

    /// Final norm + LM head → logits `[B, L, vocab]`. Only the output stage
    /// (rank 0) calls this.
    pub fn head(&self, h: &Array, s: &Stream) -> Result<Array> {
        let normed =
            RmsNorm::load(self.weights, "model.norm", self.cfg.rms_norm_eps)?.forward(h, s)?;
        if self.cfg.tie_word_embeddings || !self.weights.contains("lm_head.weight") {
            Embedding::load(self.weights, "model.embed_tokens")?.as_linear(&normed, s)
        } else {
            Linear::load(self.weights, "lm_head")?.forward(&normed, s)
        }
    }

    fn layer(&self, idx: usize, h: Array, cache: &mut LayerCache, s: &Stream) -> Result<Array> {
        let p = format!("model.layers.{idx}");

        // Attention block with residual.
        let normed = RmsNorm::load(
            self.weights,
            &format!("{p}.input_layernorm"),
            self.cfg.rms_norm_eps,
        )?
        .forward(&h, s)?;
        let attn = self.attention(&p, &normed, cache, s)?;
        let h = ops::add(&h, &attn, s)?;

        // MLP block with residual.
        let normed = RmsNorm::load(
            self.weights,
            &format!("{p}.post_attention_layernorm"),
            self.cfg.rms_norm_eps,
        )?
        .forward(&h, s)?;
        let mlp = self.mlp(&p, &normed, s)?;
        ops::add(&h, &mlp, s)
    }

    fn attention(
        &self,
        prefix: &str,
        x: &Array,
        cache: &mut LayerCache,
        s: &Stream,
    ) -> Result<Array> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);
        let n_heads = self.cfg.num_attention_heads;
        let n_kv = self.cfg.kv_heads();
        let hd = self.cfg.head_dim();

        let mut q =
            Linear::load(self.weights, &format!("{prefix}.self_attn.q_proj"))?.forward(x, s)?;
        let mut k =
            Linear::load(self.weights, &format!("{prefix}.self_attn.k_proj"))?.forward(x, s)?;
        let v = Linear::load(self.weights, &format!("{prefix}.self_attn.v_proj"))?.forward(x, s)?;

        // [B, L, H, hd] -> [B, H, L, hd]
        q = ops::transpose(
            &ops::reshape(&q, &[b, l, n_heads, hd], s)?,
            &[0, 2, 1, 3],
            s,
        )?;
        k = ops::transpose(&ops::reshape(&k, &[b, l, n_kv, hd], s)?, &[0, 2, 1, 3], s)?;
        let mut v = ops::transpose(&ops::reshape(&v, &[b, l, n_kv, hd], s)?, &[0, 2, 1, 3], s)?;

        // Qwen3: per-head q/k RMSNorm before RoPE.
        if self.family == Family::Qwen3 {
            if self
                .weights
                .contains(&format!("{prefix}.self_attn.q_norm.weight"))
            {
                q = RmsNorm::load(
                    self.weights,
                    &format!("{prefix}.self_attn.q_norm"),
                    self.cfg.rms_norm_eps,
                )?
                .forward(&q, s)?;
            }
            if self
                .weights
                .contains(&format!("{prefix}.self_attn.k_norm.weight"))
            {
                k = RmsNorm::load(
                    self.weights,
                    &format!("{prefix}.self_attn.k_norm"),
                    self.cfg.rms_norm_eps,
                )?
                .forward(&k, s)?;
            }
        }

        // RoPE at the cache offset.
        let offset = cache.offset;
        q = ops::rope(&q, hd, false, self.cfg.rope_theta, 1.0, offset, s)?;
        k = ops::rope(&k, hd, false, self.cfg.rope_theta, 1.0, offset, s)?;

        // Append to KV cache along the sequence axis (axis 2).
        if let (Some(pk), Some(pv)) = (cache.keys.as_ref(), cache.values.as_ref()) {
            k = ops::concatenate(&[pk, &k], 2, s)?;
            v = ops::concatenate(&[pv, &v], 2, s)?;
        }
        cache.offset += l;

        let out =
            ops::scaled_dot_product_attention(&q, &k, &v, self.cfg.attention_scale(), "causal", s)?;
        // store updated cache (clone handles by re-reading: we keep k/v)
        cache.keys = Some(clone_ref(&k, s)?);
        cache.values = Some(clone_ref(&v, s)?);

        // [B, H, L, hd] -> [B, L, H*hd]
        let out = ops::transpose(&out, &[0, 2, 1, 3], s)?;
        let out = ops::reshape(&out, &[b, l, n_heads * hd], s)?;
        Linear::load(self.weights, &format!("{prefix}.self_attn.o_proj"))?.forward(&out, s)
    }

    fn mlp(&self, prefix: &str, x: &Array, s: &Stream) -> Result<Array> {
        let gate = Linear::load(self.weights, &format!("{prefix}.mlp.gate_proj"))?.forward(x, s)?;
        let up = Linear::load(self.weights, &format!("{prefix}.mlp.up_proj"))?.forward(x, s)?;
        let act = ops::silu(&gate, s)?;
        let h = ops::multiply(&act, &up, s)?;
        Linear::load(self.weights, &format!("{prefix}.mlp.down_proj"))?.forward(&h, s)
    }
}

/// Produce an independent handle to the same logical array (adds a no-op into
/// the graph). Used to retain KV state after the SDPA consumes k/v.
fn clone_ref(a: &Array, s: &Stream) -> Result<Array> {
    // astype to the same dtype yields a fresh handle without changing values.
    a.astype(a.dtype(), s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_sizing_matches_owned_layers() {
        let cfg = ModelConfig {
            model_type: "llama".into(),
            architectures: vec![],
            hidden_size: 64,
            num_hidden_layers: 8,
            num_attention_heads: 8,
            num_key_value_heads: Some(8),
            intermediate_size: 128,
            vocab_size: 100,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            head_dim: None,
            attention_bias: false,
            tie_word_embeddings: true,
            max_position_embeddings: 4096,
        };
        let weights = Weights::new();
        // 2 ranks: rank 0 owns last 4 layers.
        let pipe = Pipeline::plan(0, 2, 8);
        let model = LlamaModel::new(&cfg, &weights, pipe);
        assert_eq!(model.new_cache().len(), 4);
    }
}
