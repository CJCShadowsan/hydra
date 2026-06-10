//! High-level runtime: load a model and generate text, single-node or
//! distributed. Ties together download → load → tokenizer → forward/generate.

mod generate;
mod server;
mod tokenizer;

pub use generate::{generate_distributed, generate_local};
pub use server::{ServerHandle, ServerState, router, serve, spawn};
pub use tokenizer::{Tokenizer, apply_chat_template};

use crate::Result;
use crate::array::Stream;
use crate::distributed::{Backend, Group, Pipeline};
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

    /// Load a model into a **distributed** group, choosing pipeline or tensor
    /// parallelism per `mode`.
    ///
    /// The caller must have already initialised the MLX distributed environment
    /// (hostfile + rank + backend); see [`DistributedEngine::join`], which wires
    /// the env, inits the [`Group`], and calls this. Pipeline mode shards by
    /// layers (each rank downloads only its stage); tensor mode loads the full
    /// repo and slices each projection per rank.
    pub async fn load_distributed(
        model: &ModelRef,
        group: &Group,
        mode: ParallelismMode,
    ) -> Result<Self> {
        match mode {
            ParallelismMode::Tensor => Self::load_tensor_parallel(model, group).await,
            ParallelismMode::Pipeline => {
                let total = 0; // re-planned from config inside load_with_pipeline
                let pipeline = Pipeline::plan(group.rank(), group.size(), total);
                Self::load_with_pipeline(model, pipeline).await
            }
            ParallelismMode::Single => Self::load_single(model).await,
        }
    }

    /// Load a model for tensor-parallel serving across a live [`Group`]. The
    /// per-rank weight shards are sliced after loading. All ranks call this.
    pub async fn load_tensor_parallel(model: &ModelRef, group: &Group) -> Result<Self> {
        // Tensor parallel: every rank loads the full repo (single pipeline
        // stage), then slices its shard of each projection.
        let pipeline = Pipeline::plan(0, 1, 0);
        let mut engine = Self::load_with_pipeline(model, pipeline).await?;
        let load_stream = Stream::cpu();
        loader::shard_tensor_parallel(
            &mut engine.weights,
            &engine.config,
            group.rank(),
            group.size(),
            &load_stream,
        )?;
        Ok(engine)
    }

    /// Build the model bound to this engine's loaded weights.
    pub fn model(&self) -> LlamaModel<'_> {
        LlamaModel::new(&self.config, &self.weights, self.pipeline.clone())
    }

    /// Build the model with tensor parallelism enabled over `group`.
    pub fn model_tensor_parallel<'g>(&'g self, group: &'g Group) -> LlamaModel<'g> {
        LlamaModel::new(&self.config, &self.weights, self.pipeline.clone())
            .with_tensor_parallel(group)
    }

    /// Generate a completion for a chat prompt (single-node greedy).
    pub fn chat(&self, system: Option<&str>, user: &str, max_tokens: usize) -> Result<String> {
        let prompt = apply_chat_template(system, user);
        let ids = self.tokenizer.encode(&prompt)?;
        self.complete_ids(&ids, max_tokens, None)
    }

    /// Generate a completion for a chat prompt across a live distributed
    /// [`Group`] (pipeline parallelism). All ranks must call this in lock-step.
    pub fn chat_distributed(
        &self,
        group: &Group,
        system: Option<&str>,
        user: &str,
        max_tokens: usize,
    ) -> Result<String> {
        let prompt = apply_chat_template(system, user);
        let ids = self.tokenizer.encode(&prompt)?;
        self.complete_ids(&ids, max_tokens, Some(group))
    }

    /// Core completion: routes to single-node, pipeline, or tensor-parallel
    /// generation and decodes the result.
    ///
    /// - No group, or group size 1 → single-node local generate.
    /// - Group with `pipeline.size > 1` → pipeline-parallel generate (send/recv).
    /// - Group with `pipeline.size == 1` → tensor-parallel generate (the group
    ///   is the tensor group; layers are sharded and all-reduce per layer).
    pub fn complete_ids(
        &self,
        ids: &[i32],
        max_tokens: usize,
        group: Option<&Group>,
    ) -> Result<String> {
        let eos = |t: i32| self.tokenizer.is_eos(t);
        let out = match group {
            Some(g) if self.pipeline.size > 1 => {
                // Pipeline parallelism.
                let model = self.model();
                generate_distributed(
                    &model,
                    &self.pipeline,
                    g,
                    ids,
                    max_tokens,
                    eos,
                    &self.stream,
                )?
            }
            Some(g) if g.size() > 1 => {
                // Tensor parallelism: sharded model, plain local loop — the
                // all-reduces inside the layers do the cross-rank work, and the
                // loop is identical on every rank (greedy is deterministic).
                let model = self.model_tensor_parallel(g);
                generate_local(&model, &self.pipeline, ids, max_tokens, eos, &self.stream)?
            }
            _ => {
                let model = self.model();
                generate_local(&model, &self.pipeline, ids, max_tokens, eos, &self.stream)?
            }
        };
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

/// A distributed MLX node: a live [`Group`] plus the model [`Engine`] loaded for
/// this rank. Holding both together keeps the group alive for the engine's
/// lifetime (collectives borrow the group).
///
/// Construction ([`DistributedEngine::join`]) performs the full discovery →
/// serving handoff: it writes the rank-ordered hostfile, sets the MLX
/// environment (`MLX_HOSTFILE`, `MLX_RANK`) that the ring/jaccl backends read,
/// initialises the [`Group`], and loads the model sharded per the chosen
/// [`ParallelismMode`]. MLX then opens its own TCP ring / RDMA mesh to the
/// hostfile peers — mesh only supplied the addresses.
pub struct DistributedEngine {
    pub group: Group,
    pub engine: Engine,
    pub mode: ParallelismMode,
}

/// Parameters for joining a distributed MLX group.
pub struct JoinParams {
    /// Rank-ordered hostfile JSON (MLX `load_nodes` format).
    pub hostfile_json: String,
    /// This node's rank in the ring.
    pub rank: usize,
    /// Which MLX backend to initialise (ring/jaccl/mpi).
    pub backend: Backend,
    /// The parallelism mode (pipeline/tensor).
    pub mode: ParallelismMode,
}

impl DistributedEngine {
    /// Join an MLX distributed group and load the model for this rank.
    ///
    /// Writes `hostfile_json` to a temp file, points `MLX_HOSTFILE`/`MLX_RANK`
    /// at it, initialises the group on `backend`, and loads the model per
    /// `mode`. The hostfile path is kept alive for the process lifetime.
    pub async fn join(model: &ModelRef, params: JoinParams) -> Result<Self> {
        use std::io::Write;

        // Persist the hostfile and expose it to the MLX backend via env. The
        // file is intentionally leaked (kept for the process lifetime) because
        // MLX re-reads it lazily during collective setup.
        let mut path = std::env::temp_dir();
        path.push(format!("mesh-mlx-hosts-{}.json", std::process::id()));
        {
            let mut f = std::fs::File::create(&path)
                .map_err(|e| crate::MlxError::Distributed(format!("write hostfile: {e}")))?;
            f.write_all(params.hostfile_json.as_bytes())
                .map_err(|e| crate::MlxError::Distributed(format!("write hostfile: {e}")))?;
        }
        // SAFETY: set before any MLX distributed init on this process; the
        // runtime is single-threaded at this point in startup.
        unsafe {
            std::env::set_var("MLX_HOSTFILE", &path);
            std::env::set_var("MLX_RANK", params.rank.to_string());
        }

        let group = Group::init(params.backend, true)?;
        let engine = Engine::load_distributed(model, &group, params.mode).await?;
        Ok(DistributedEngine {
            group,
            engine,
            mode: params.mode,
        })
    }

    /// Generate a completion across the group (all ranks call in lock-step).
    pub fn chat(&self, system: Option<&str>, user: &str, max_tokens: usize) -> Result<String> {
        let prompt = apply_chat_template(system, user);
        let ids = self.engine.tokenizer.encode(&prompt)?;
        self.engine
            .complete_ids(&ids, max_tokens, Some(&self.group))
    }
}
