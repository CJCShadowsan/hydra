//! Tokenizer wrapper over the HF `tokenizers` crate, plus chat templating.

use crate::{MlxError, Result};
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

/// A loaded tokenizer.
pub struct Tokenizer {
    inner: HfTokenizer,
    eos_ids: Vec<u32>,
}

impl Tokenizer {
    /// Load `tokenizer.json` from a model directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("tokenizer.json");
        let inner = HfTokenizer::from_file(&path)
            .map_err(|e| MlxError::Tokenizer(format!("load {}: {e}", path.display())))?;

        // Resolve EOS token id(s) from tokenizer_config / generation_config.
        let eos_ids = resolve_eos_ids(dir, &inner);
        Ok(Tokenizer { inner, eos_ids })
    }

    /// Encode text to token ids.
    pub fn encode(&self, text: &str) -> Result<Vec<i32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| MlxError::Tokenizer(format!("encode: {e}")))?;
        Ok(enc.get_ids().iter().map(|&u| u as i32).collect())
    }

    /// Decode token ids to text.
    pub fn decode(&self, ids: &[i32]) -> Result<String> {
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.inner
            .decode(&ids, true)
            .map_err(|e| MlxError::Tokenizer(format!("decode: {e}")))
    }

    /// Whether `id` is an end-of-sequence token.
    pub fn is_eos(&self, id: i32) -> bool {
        self.eos_ids.contains(&(id as u32))
    }
}

/// Apply a minimal chat template. Many MLX-community repos ship a Jinja
/// template; supporting full Jinja is out of scope for the first cut, so we use
/// the widely-compatible ChatML-ish framing and fall back to raw text. The
/// runtime can be extended to honour the repo's `chat_template` later.
pub fn apply_chat_template(system: Option<&str>, user: &str) -> String {
    let mut out = String::new();
    if let Some(sys) = system {
        out.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
    }
    out.push_str(&format!("<|im_start|>user\n{user}<|im_end|>\n"));
    out.push_str("<|im_start|>assistant\n");
    out
}

fn resolve_eos_ids(dir: &Path, tok: &HfTokenizer) -> Vec<u32> {
    let mut ids = Vec::new();
    // generation_config.json may list eos_token_id (int or array).
    if let Ok(text) = std::fs::read_to_string(dir.join("generation_config.json"))
        && let Ok(json) = serde_json::from_str::<serde_json::Value>(&text)
    {
        collect_eos(&json["eos_token_id"], &mut ids);
    }
    // Common ChatML end token.
    if let Some(id) = tok.token_to_id("<|im_end|>") {
        ids.push(id);
    }
    if let Some(id) = tok.token_to_id("</s>") {
        ids.push(id);
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn collect_eos(v: &serde_json::Value, out: &mut Vec<u32>) {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_u64() {
                out.push(i as u32);
            }
        }
        serde_json::Value::Array(a) => {
            for e in a {
                collect_eos(e, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_template_frames_roles() {
        let p = apply_chat_template(Some("be brief"), "hi");
        assert!(p.contains("system\nbe brief"));
        assert!(p.contains("user\nhi"));
        assert!(p.trim_end().ends_with("assistant"));
    }

    #[test]
    fn collect_eos_handles_int_and_array() {
        let mut v = vec![];
        collect_eos(&serde_json::json!(2), &mut v);
        collect_eos(&serde_json::json!([100, 101]), &mut v);
        assert_eq!(v, vec![2, 100, 101]);
    }
}
