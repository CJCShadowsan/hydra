//! Hugging Face download with selective safetensors fetching.
//!
//! Mirrors `mlx-lm`: always fetch metadata (config, tokenizer, index), then
//! fetch either the whole repo or only the stage's safetensors shards. Uses the
//! repo's [`model_hf::HfModelRepository`] wrapper (the patched `hf-hub` fork +
//! shared cache), so behaviour matches the rest of mesh-llm.

use crate::distributed::Pipeline;
use crate::loader::{self, DownloadScope};
use crate::{MlxError, Result};
use model_hf::HfModelRepository;
use std::path::PathBuf;

/// A reference to a model to serve. MLX consumes safetensors; GGUF is rejected.
#[derive(Debug, Clone)]
pub struct ModelRef {
    pub repo: String,
    pub revision: String,
}

impl ModelRef {
    pub fn new(repo: impl Into<String>) -> Self {
        ModelRef {
            repo: repo.into(),
            revision: "main".to_string(),
        }
    }

    pub fn revision(mut self, rev: impl Into<String>) -> Self {
        self.revision = rev.into();
        self
    }
}

/// Metadata files always needed (config, tokenizer, weight index). Not all
/// repos have every file, so fetching is best-effort per file.
const METADATA_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "generation_config.json",
    "special_tokens_map.json",
];

/// Download what a stage needs and return a local model directory + files.
///
/// Two-pass like `sharded_load`: metadata first (to read config + index and
/// decide the layer split), then the safetensors shards for this stage.
pub async fn fetch(model: &ModelRef, pipeline: &Pipeline) -> Result<loader::LoadedModel> {
    let hf = HfModelRepository::from_env()
        .map_err(|e| MlxError::Download(format!("hf repository: {e}")))?;

    // Pass 1: metadata (best-effort; the directory is inferred from any file).
    let mut dir: Option<PathBuf> = None;
    for f in METADATA_FILES {
        if let Ok(path) = hf.download_file(&model.repo, &model.revision, f).await
            && dir.is_none()
        {
            dir = path.parent().map(|p| p.to_path_buf());
        }
    }
    let dir = dir.ok_or_else(|| {
        MlxError::Download(format!("no metadata found for repo '{}'", model.repo))
    })?;

    let config = loader::read_config(&dir)?;

    // Reject GGUF-only repos defensively.
    if dir.join("model.gguf").exists() && !dir.join("model.safetensors").exists() {
        return Err(MlxError::Download(
            "repo provides GGUF, not safetensors; route to the llama.cpp lane".into(),
        ));
    }

    // Pass 2: decide shard files from the index, then download them by name.
    let scope = DownloadScope::for_pipeline(pipeline.size);
    let shard_files = loader::shard_files_for_stage(&dir, pipeline, scope)?;
    let mut local = Vec::with_capacity(shard_files.len());
    for f in &shard_files {
        let name = f
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| MlxError::Download("bad shard file name".into()))?;
        let path = hf
            .download_file(&model.repo, &model.revision, name)
            .await
            .map_err(|e| MlxError::Download(format!("fetch {name}: {e}")))?;
        local.push(path);
    }

    Ok(loader::LoadedModel {
        dir,
        config,
        shard_files: local,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_ref_builder() {
        let m = ModelRef::new("mlx-community/Qwen3-0.6B-4bit").revision("abc");
        assert_eq!(m.repo, "mlx-community/Qwen3-0.6B-4bit");
        assert_eq!(m.revision, "abc");
    }

    #[test]
    fn default_revision_is_main() {
        assert_eq!(ModelRef::new("a/b").revision, "main");
    }
}
