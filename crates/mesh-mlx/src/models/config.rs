//! Model configuration parsed from a Hugging Face `config.json`.

use serde::Deserialize;

/// The subset of `config.json` fields the supported architectures need.
/// Field names match HF; defaults cover common omissions.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub architectures: Vec<String>,

    pub hidden_size: i32,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i32,
    #[serde(default)]
    pub num_key_value_heads: Option<i32>,
    pub intermediate_size: i32,
    pub vocab_size: i32,

    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub head_dim: Option<i32>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: i32,
}

fn default_rms_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_max_pos() -> i32 {
    4096
}

impl ModelConfig {
    /// Number of KV heads (defaults to attention heads if unset → MHA).
    pub fn kv_heads(&self) -> i32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Per-head dimension.
    pub fn head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Attention scale `1/sqrt(head_dim)`.
    pub fn attention_scale(&self) -> f32 {
        (self.head_dim() as f32).powf(-0.5)
    }

    /// The model family, normalised from `model_type` / `architectures`.
    pub fn family(&self) -> Family {
        let mt = self.model_type.to_lowercase();
        if mt.contains("qwen3") {
            Family::Qwen3
        } else if mt.contains("qwen2") || mt.contains("qwen") {
            Family::Qwen2
        } else if mt.contains("llama") || mt.contains("mistral") {
            Family::Llama
        } else {
            // Fall back to inspecting architectures.
            let arch = self
                .architectures
                .first()
                .map(|s| s.to_lowercase())
                .unwrap_or_default();
            if arch.contains("qwen3") {
                Family::Qwen3
            } else if arch.contains("qwen2") {
                Family::Qwen2
            } else {
                Family::Llama
            }
        }
    }
}

/// Supported architecture families (initial set).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Family {
    /// Llama / Mistral (RMSNorm, RoPE, SwiGLU, GQA).
    Llama,
    /// Qwen2 (Llama-like with attention biases on q/k/v).
    Qwen2,
    /// Qwen3 (adds q/k RMSNorm).
    Qwen3,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_llama_config_and_derives() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 128256
        }"#;
        let cfg: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.kv_heads(), 8);
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.family(), Family::Llama);
        assert!((cfg.attention_scale() - 0.088388346).abs() < 1e-6);
    }

    #[test]
    fn detects_qwen3() {
        let json = r#"{
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 3072,
            "vocab_size": 151936
        }"#;
        let cfg: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.family(), Family::Qwen3);
        assert_eq!(cfg.kv_heads(), 8);
    }
}
