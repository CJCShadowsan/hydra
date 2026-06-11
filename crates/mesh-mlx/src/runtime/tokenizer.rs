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
        let ids: Vec<u32> = ids
            .iter()
            .map(|&i| {
                u32::try_from(i).map_err(|_| MlxError::Tokenizer(format!("negative token id {i}")))
            })
            .collect::<Result<_>>()?;
        self.inner
            .decode(&ids, true)
            .map_err(|e| MlxError::Tokenizer(format!("decode: {e}")))
    }

    /// Whether `id` is an end-of-sequence token.
    pub fn is_eos(&self, id: i32) -> bool {
        self.eos_ids.contains(&(id as u32))
    }
}

/// A single chat turn fed to [`render_chat`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

impl ChatTurn {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// Render a full multi-turn conversation with the widely-compatible ChatML
/// framing, preserving message order so context and prior assistant turns are
/// not dropped.
///
/// Many MLX-community repos ship a Jinja `chat_template`; honouring the repo's
/// own template is a later refinement. ChatML framing works for the
/// Qwen/Llama-style families this runtime currently transcribes. Unknown roles
/// are passed through verbatim so tool/other roles are at least visible to the
/// model rather than silently discarded.
pub fn render_chat(turns: &[ChatTurn]) -> String {
    let mut out = String::new();
    for turn in turns {
        out.push_str("<|im_start|>");
        out.push_str(&turn.role);
        out.push('\n');
        out.push_str(&turn.content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Convenience wrapper for the common system + single-user shape.
pub fn apply_chat_template(system: Option<&str>, user: &str) -> String {
    let mut turns = Vec::new();
    if let Some(sys) = system {
        turns.push(ChatTurn::new("system", sys));
    }
    turns.push(ChatTurn::new("user", user));
    render_chat(&turns)
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

    #[test]
    fn render_chat_preserves_multi_turn_order() {
        // Full history must survive: system, prior user/assistant turns, then
        // the latest user turn — not just the last message.
        let turns = vec![
            ChatTurn::new("system", "be brief"),
            ChatTurn::new("user", "hi"),
            ChatTurn::new("assistant", "hello"),
            ChatTurn::new("user", "and now?"),
        ];
        let p = render_chat(&turns);
        let sys = p.find("system\nbe brief").expect("system present");
        let first_user = p.find("user\nhi").expect("first user present");
        let asst = p.find("assistant\nhello").expect("assistant turn present");
        let last_user = p.find("user\nand now?").expect("last user present");
        // Order is preserved.
        assert!(sys < first_user && first_user < asst && asst < last_user);
        // Generation prompt is appended.
        assert!(p.trim_end().ends_with("assistant"));
    }

    #[test]
    fn render_chat_passes_unknown_roles_through() {
        let turns = vec![ChatTurn::new("tool", "result: 42")];
        let p = render_chat(&turns);
        assert!(p.contains("tool\nresult: 42"));
    }

    #[test]
    fn apply_chat_template_matches_render_chat() {
        let a = apply_chat_template(Some("s"), "u");
        let b = render_chat(&[ChatTurn::new("system", "s"), ChatTurn::new("user", "u")]);
        assert_eq!(a, b);
    }
}
