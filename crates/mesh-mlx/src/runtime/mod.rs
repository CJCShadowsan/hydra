//! High-level runtime: load a model and generate text, single-node or
//! distributed. Ties together download → load → tokenizer → forward/generate.

mod generate;
mod tokenizer;

pub use generate::{decode_step, generate_local};
pub use tokenizer::{Tokenizer, apply_chat_template};

use crate::Result;
use crate::array::Stream;
use crate::distributed::{Group, Pipeline};
use crate::download::{self, ModelRef};
use crate::loader;
use crate::mesh::ParallelismMode;
use crate::models::{LlamaModel, ModelConfig};
use crate::nn::Weights;

/// A loaded, ready-to-serve MLX engine for one model on this node.
///
/// Owns the parsed config, the loaded weights for this stage, the tokenizer,
/// and the pipeline topology. Generation borrows these to build the model.
pub struct Engine {
    pub config: ModelConfig,
    pub weights: Weights,
    pub tokenizer: Tokenizer,
    pub pipeline: Pipeline,
    pub stream: Stream,
}

impl Engine {
    /// Download (selectively) and load a model for single-node serving.
    pub async fn load_single(model: &ModelRef) -> Result<Self> {
        let pipeline = Pipeline::plan(0, 1, 0); // total layers filled after config
        Self::load_with_pipeline(model, pipeline).await
    }

    /// Download and load for a given pipeline topology (rank/size known from a
    /// live [`Group`]). The total layer count comes from the config.
    pub async fn load_with_pipeline(model: &ModelRef, mut pipeline: Pipeline) -> Result<Self> {
        // First fetch metadata to learn the layer count, then re-plan the
        // pipeline with the real total and fetch this stage's shards.
        let meta = download::fetch(model, &pipeline).await?;
        pipeline = Pipeline::plan(pipeline.rank, pipeline.size, meta.config.num_hidden_layers);

        // Re-resolve shard files now that we know the true layer split.
        let scope = loader::DownloadScope::for_pipeline(pipeline.size);
        let shard_files = loader::shard_files_for_stage(&meta.dir, &pipeline, scope)?;

        // Safetensors load is a host op — evaluate it on the CPU stream. Inference
        // then runs on the GPU stream.
        let load_stream = Stream::cpu();
        let weights = loader::load_weights(&shard_files, &load_stream)?;
        let stream = Stream::gpu();
        let tokenizer = Tokenizer::from_dir(&meta.dir)?;

        Ok(Engine {
            config: meta.config,
            weights,
            tokenizer,
            pipeline,
            stream,
        })
    }

    /// Build the model bound to this engine's loaded weights.
    pub fn model(&self) -> LlamaModel<'_> {
        LlamaModel::new(&self.config, &self.weights, self.pipeline.clone())
    }

    /// Generate a completion for a chat prompt (single-node greedy).
    pub fn chat(&self, system: Option<&str>, user: &str, max_tokens: usize) -> Result<String> {
        let prompt = apply_chat_template(system, user);
        let ids = self.tokenizer.encode(&prompt)?;
        let model = self.model();
        let out = generate_local(
            &model,
            &self.pipeline,
            &ids,
            max_tokens,
            |t| self.tokenizer.is_eos(t),
            &self.stream,
        )?;
        self.tokenizer.decode(&out)
    }

    /// The parallelism mode this engine is configured for.
    pub fn mode(&self) -> ParallelismMode {
        match self.pipeline.size {
            s if s <= 1 => ParallelismMode::Single,
            _ => ParallelismMode::Pipeline,
        }
    }
}

/// Initialise a distributed group for a backend, returning the pipeline plan
/// once the layer count is known. The caller passes total layers from config.
pub fn group_pipeline(group: &Group, total_layers: usize) -> Pipeline {
    Pipeline::from_group(group, total_layers)
}
