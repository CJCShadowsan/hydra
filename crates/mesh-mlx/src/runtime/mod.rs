//! High-level runtime: load a model and generate text, single-node or
//! distributed. Ties together download → load → tokenizer → forward/generate.

mod generate;
mod server;
mod tokenizer;

pub use generate::{generate_distributed, generate_local};
pub use server::{ServerHandle, ServerState, router, serve, spawn};
pub use tokenizer::{ChatTurn, Tokenizer, apply_chat_template, render_chat};

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
/// the pipeline topology, and the **parallelism mode it was loaded for**.
/// Generation borrows these to build the model; routing in [`Engine::complete_ids`]
/// branches on the persisted mode, never re-inferred from topology sizes.
pub struct Engine {
    pub config: ModelConfig,
    pub weights: Weights,
    pub tokenizer: Tokenizer,
    pub pipeline: Pipeline,
    pub stream: Stream,
    mode: ParallelismMode,
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

        // Loading by pipeline topology implies Single for a one-stage plan and
        // Pipeline otherwise; `load_tensor_parallel` overrides this to Tensor
        // after slicing the weights.
        let mode = if pipeline.size > 1 {
            ParallelismMode::Pipeline
        } else {
            ParallelismMode::Single
        };
        Ok(Engine {
            config: meta.config,
            weights,
            tokenizer,
            pipeline,
            stream,
            mode,
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
        engine.mode = ParallelismMode::Tensor;
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

    /// Generate a completion for a full multi-turn conversation (single-node).
    /// Preserves message order so system + prior turns are not dropped.
    pub fn chat_turns(&self, turns: &[ChatTurn], max_tokens: usize) -> Result<String> {
        let prompt = render_chat(turns);
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
    /// Routing follows the **persisted load mode** ([`Engine::mode`]), not a
    /// re-inference from topology sizes:
    ///
    /// - `Single` (or no group) → single-node local generate.
    /// - `Pipeline` + group → pipeline-parallel generate (send/recv).
    /// - `Tensor` + group → tensor-parallel generate (sharded weights; the
    ///   all-reduces inside the layers do the cross-rank work).
    ///
    /// A distributed mode without a group is an error — sharded weights cannot
    /// produce correct output locally.
    pub fn complete_ids(
        &self,
        ids: &[i32],
        max_tokens: usize,
        group: Option<&Group>,
    ) -> Result<String> {
        let eos = |t: i32| self.tokenizer.is_eos(t);
        let out = match (self.mode, group) {
            (ParallelismMode::Single, _) => {
                let model = self.model();
                generate_local(&model, &self.pipeline, ids, max_tokens, eos, &self.stream)?
            }
            (ParallelismMode::Pipeline, Some(g)) => {
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
            (ParallelismMode::Tensor, Some(g)) => {
                // Sharded model, plain local loop — the all-reduces inside the
                // layers do the cross-rank work, and the loop is identical on
                // every rank (greedy is deterministic).
                let model = self.model_tensor_parallel(g);
                generate_local(&model, &self.pipeline, ids, max_tokens, eos, &self.stream)?
            }
            (mode, None) => {
                return Err(crate::MlxError::Distributed(format!(
                    "engine loaded for {mode:?} parallelism but no group supplied"
                )));
            }
        };
        self.tokenizer.decode(&out)
    }

    /// The parallelism mode this engine was loaded for.
    pub fn mode(&self) -> ParallelismMode {
        self.mode
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
        // Persist the hostfile and expose it to the MLX backend via env. The
        // file is intentionally leaked (kept for the process lifetime) because
        // MLX re-reads it lazily during collective setup.
        let path = write_hostfile(&params.hostfile_json)?;
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

    /// This node's rank within the group.
    pub fn rank(&self) -> i32 {
        self.group.rank()
    }

    /// Whether this node is the group leader (rank 0) that serves the OpenAI
    /// API and drives generation. Worker ranks run [`DistributedEngine::run_worker_loop`].
    pub fn is_leader(&self) -> bool {
        self.group.rank() == LEADER_RANK
    }

    /// Generate a completion across the group, driven by the leader (rank 0).
    ///
    /// Only rank 0 receives the HTTP request. To keep the group in lock-step,
    /// the leader broadcasts the prompt + token budget to every worker (which
    /// are parked in [`run_worker_loop`]) before all ranks run the identical
    /// generation. This mirrors MLX's own distributed server: the request
    /// crosses ranks once, each rank tokenizes the shared prompt locally, and
    /// deterministic greedy sampling keeps the per-step token identical on
    /// every rank.
    ///
    /// Must be called on the leader only; workers participate via the loop.
    pub fn chat_turns(&self, turns: &[ChatTurn], max_tokens: usize) -> Result<String> {
        debug_assert!(self.is_leader(), "chat_turns must be driven by the leader");
        let prompt = render_chat(turns);
        self.broadcast_and_generate(&prompt, max_tokens)
    }

    /// Convenience wrapper for the system + single-user shape.
    pub fn chat(&self, system: Option<&str>, user: &str, max_tokens: usize) -> Result<String> {
        let prompt = apply_chat_template(system, user);
        self.broadcast_and_generate(&prompt, max_tokens)
    }

    /// Leader path: encode the request, broadcast it to workers, then run the
    /// lock-step generation that the workers also run.
    fn broadcast_and_generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let req = WorkRequest {
            prompt: prompt.to_string(),
            max_tokens,
        };
        self.group
            .broadcast_bytes(LEADER_RANK, &req.encode(), &self.engine.stream)?;
        self.run_one(&req)
    }

    /// Worker path: park until the leader broadcasts a request, run the
    /// matching lock-step generation, and repeat. A zero-length broadcast (the
    /// shutdown sentinel the leader sends on drop) ends the loop.
    ///
    /// This is the piece that makes multi-node real: without it the worker
    /// ranks never enter the collectives the leader's generation blocks on, so
    /// the leader would deadlock on the first request. Returns when the leader
    /// signals shutdown.
    pub fn run_worker_loop(&self) -> Result<()> {
        debug_assert!(!self.is_leader(), "only worker ranks run the worker loop");
        loop {
            let bytes = self
                .group
                .broadcast_bytes(LEADER_RANK, &[], &self.engine.stream)?;
            match WorkRequest::decode(&bytes) {
                // Empty broadcast: leader is shutting the group down.
                None => return Ok(()),
                Some(req) => {
                    // Run the same generation the leader runs; the output is
                    // discarded on workers (only the leader replies over HTTP),
                    // but every rank must execute it to satisfy the collectives.
                    let _ = self.run_one(&req)?;
                }
            }
        }
    }

    /// Run one generation for `req` on this rank, in lock-step with the group.
    fn run_one(&self, req: &WorkRequest) -> Result<String> {
        let ids = self.engine.tokenizer.encode(&req.prompt)?;
        self.engine
            .complete_ids(&ids, req.max_tokens, Some(&self.group))
    }

    /// Tell worker ranks to exit their loop. Called once on the leader during
    /// shutdown; broadcasts the zero-length sentinel the workers watch for.
    pub fn signal_shutdown(&self) {
        if self.is_leader() {
            let _ = self
                .group
                .broadcast_bytes(LEADER_RANK, &[], &self.engine.stream);
        }
    }
}

/// A running worker rank: owns the distributed engine and runs the lock-step
/// worker loop on a blocking task until the leader signals shutdown.
///
/// Worker ranks (rank != 0) do **not** serve the OpenAI API — only the leader
/// does. They park here driving every generation the leader broadcasts, which
/// is what keeps the group's collectives synchronised. Without a running worker
/// per non-leader rank, the leader deadlocks on its first request.
pub struct WorkerHandle {
    task: tokio::task::JoinHandle<Result<()>>,
}

impl WorkerHandle {
    /// Spawn the worker loop for `engine` on a blocking task. The engine is
    /// moved onto that task (MLX handles are `Send`); the loop returns when the
    /// leader broadcasts the shutdown sentinel or a collective fails.
    pub fn spawn(engine: DistributedEngine) -> Self {
        let task = tokio::task::spawn_blocking(move || engine.run_worker_loop());
        Self { task }
    }

    /// Wait for the worker loop to finish (after the leader shuts the group
    /// down). The loop also ends on its own when the leader exits.
    pub async fn join(self) -> Result<()> {
        match self.task.await {
            Ok(r) => r,
            Err(e) => Err(crate::MlxError::Distributed(format!(
                "worker loop task: {e}"
            ))),
        }
    }

    /// Abort the worker loop without waiting (used when the node is torn down
    /// before the group reaches a clean shutdown).
    pub fn abort(&self) {
        self.task.abort();
    }
}

/// The leader (request-receiving, generation-driving) rank.
const LEADER_RANK: i32 = 0;

/// A unit of work broadcast from the leader to every worker rank: the rendered
/// prompt and the token budget. Encoded as `[u32 max_tokens][utf8 prompt]`.
struct WorkRequest {
    prompt: String,
    max_tokens: usize,
}

impl WorkRequest {
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + self.prompt.len());
        let max = u32::try_from(self.max_tokens).unwrap_or(u32::MAX);
        out.extend_from_slice(&max.to_le_bytes());
        out.extend_from_slice(self.prompt.as_bytes());
        out
    }

    /// Decode a broadcast payload. `None` for the empty shutdown sentinel or a
    /// malformed (too-short) buffer, which is treated as shutdown rather than
    /// risking a desynchronised generation.
    fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }
        let max_tokens = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let prompt = String::from_utf8_lossy(&bytes[4..]).into_owned();
        Some(WorkRequest { prompt, max_tokens })
    }
}

/// Write the hostfile to a fresh, exclusively created temp file with an
/// unguessable name. `create_new` fails closed if the path already exists, so a
/// pre-created file or symlink in the shared temp directory cannot be reused.
fn write_hostfile(hostfile_json: &str) -> Result<std::path::PathBuf> {
    use std::io::Write;

    let nonce = {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        // RandomState seeds from OS randomness; enough entropy for a
        // non-guessable filename without pulling in a rand dependency.
        let mut h = RandomState::new().build_hasher();
        h.write_u32(std::process::id());
        h.finish()
    };
    let mut path = std::env::temp_dir();
    path.push(format!(
        "mesh-mlx-hosts-{}-{nonce:016x}.json",
        std::process::id()
    ));
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| {
            crate::MlxError::Distributed(format!("create hostfile {}: {e}", path.display()))
        })?;
    f.write_all(hostfile_json.as_bytes())
        .map_err(|e| crate::MlxError::Distributed(format!("write hostfile: {e}")))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn work_request_round_trips() {
        let req = WorkRequest {
            prompt: "<|im_start|>user\nhi<|im_end|>\n".to_string(),
            max_tokens: 128,
        };
        let bytes = req.encode();
        let decoded = WorkRequest::decode(&bytes).expect("decodes");
        assert_eq!(decoded.prompt, req.prompt);
        assert_eq!(decoded.max_tokens, 128);
    }

    #[test]
    fn work_request_empty_is_shutdown_sentinel() {
        // The empty broadcast (and any too-short buffer) decodes to None so the
        // worker loop treats it as shutdown rather than a desynced generation.
        assert!(WorkRequest::decode(&[]).is_none());
        assert!(WorkRequest::decode(&[0, 1, 2]).is_none());
    }

    #[test]
    fn work_request_preserves_unicode_prompt() {
        let req = WorkRequest {
            prompt: "héllo → 世界".to_string(),
            max_tokens: 1,
        };
        let decoded = WorkRequest::decode(&req.encode()).expect("decodes");
        assert_eq!(decoded.prompt, "héllo → 世界");
        assert_eq!(decoded.max_tokens, 1);
    }

    #[test]
    fn leader_rank_is_zero() {
        assert_eq!(LEADER_RANK, 0);
    }
}
