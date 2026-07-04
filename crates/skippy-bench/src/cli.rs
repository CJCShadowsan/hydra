use std::{net::SocketAddr, path::PathBuf};

use clap::{ArgAction, Parser, Subcommand, ValueEnum};

pub const DEFAULT_LOCAL_MODEL_ID: &str = "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M";
pub const DEFAULT_RUN_MAX_NEW_TOKENS: usize = 1;

#[derive(Parser)]
#[command(about = "Llama stage benchmark launcher")]
pub struct Cli {
    #[command(subcommand)]
    pub command: CommandKind,
}

#[derive(Subcommand)]
#[allow(clippy::enum_variant_names, clippy::large_enum_variant)]
pub enum CommandKind {
    LocalSingle(LocalSingleArgs),
    LocalSplitInprocess(LocalSplitInprocessArgs),
    LocalSplitBinary(LocalSplitBinaryArgs),
    LocalSplitCompare(LocalSplitCompareArgs),
    LocalSplitChainInprocess(LocalSplitChainInprocessArgs),
    LocalSplitChainBinary(LocalSplitChainBinaryArgs),
    #[command(name = "verify-span-local")]
    VerifySpanLocal(VerifySpanLocalArgs),
    #[command(name = "chat-corpus")]
    ChatCorpus(ChatCorpusArgs),
    #[command(name = "token-lengths")]
    TokenLengths(TokenLengthsArgs),
    #[command(name = "focused-runtime")]
    FocusedRuntime(FocusedRuntimeArgs),
    #[command(name = "drive-existing")]
    DriveExisting(DriveExistingArgs),
    #[command(name = "glm-dsa-layer-microbench")]
    GlmDsaLayerMicrobench(GlmDsaLayerMicrobenchArgs),
    #[command(name = "glm-dsa-op-report")]
    GlmDsaOpReport(GlmDsaOpReportArgs),
    #[command(name = "glm-dsa-op-compare")]
    GlmDsaOpCompare(GlmDsaOpCompareArgs),
    #[command(name = "glm-dsa-aggregate-reports")]
    GlmDsaAggregateReports(GlmDsaAggregateReportsArgs),
    Run(RunArgs),
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum FocusedRuntimeScenario {
    ColdStartup,
    FirstToken,
    SteadyDecode,
    KvWarmReuse,
}

impl FocusedRuntimeScenario {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ColdStartup => "cold-startup",
            Self::FirstToken => "first-token",
            Self::SteadyDecode => "steady-decode",
            Self::KvWarmReuse => "kv-warm-reuse",
        }
    }
}

#[derive(Parser)]
pub struct FocusedRuntimeArgs {
    #[arg(long, value_enum, default_value_t = FocusedRuntimeScenario::SteadyDecode)]
    pub scenario: FocusedRuntimeScenario,
    #[arg(
        long,
        help = "Write the compact focused-runtime report here. Defaults to <run-dir>/focused-runtime-report.json for real runs."
    )]
    pub focused_output: Option<PathBuf>,
    #[arg(
        long,
        help = "Emit a synthetic focused-runtime schema report and exit without launching models. Intended for CI smoke validation."
    )]
    pub schema_smoke: bool,
    #[command(flatten)]
    pub run: RunArgs,
}

#[derive(Parser)]
pub struct TokenLengthsArgs {
    #[arg(long)]
    pub model_path: PathBuf,
    #[arg(long)]
    pub prompt_corpus: PathBuf,
    #[arg(long, default_value_t = 8192)]
    pub ctx_size: u32,
    #[arg(long, visible_alias = "max-new-tokens", default_value_t = 512)]
    pub generation_limit: u32,
    #[arg(long, default_value_t = 40)]
    pub layer_end: u32,
    #[arg(long, default_value_t = 0)]
    pub n_gpu_layers: i32,
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    pub enable_thinking: bool,
    #[arg(long)]
    pub output_tsv: PathBuf,
    #[arg(long)]
    pub summary_json: Option<PathBuf>,
}

#[derive(Parser)]
pub struct DriveExistingArgs {
    #[arg(
        long,
        help = "Existing skippy-bench run directory containing deployment-plan.json."
    )]
    pub run_dir: PathBuf,
    #[arg(
        long,
        help = "Full GGUF model path for prompt tokenization. If omitted with --stage-load-mode layer-package, --stage-model must point at the local layer package."
    )]
    pub model_path: Option<PathBuf>,
    #[arg(
        long,
        help = "Layer-package directory used for tokenizer metadata when --model-path is omitted."
    )]
    pub stage_model: Option<PathBuf>,
    #[arg(long, default_value = "layer-package")]
    pub stage_load_mode: String,
    #[arg(long, default_value_t = 131072)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    pub n_gpu_layers: i32,
    #[arg(long, default_value = "f16")]
    pub activation_wire_dtype: String,
    #[arg(long, default_value = "Hello")]
    pub prompt: String,
    #[arg(long)]
    pub prompt_corpus: Option<PathBuf>,
    #[arg(long)]
    pub prompt_limit: Option<usize>,
    #[arg(long)]
    pub prompt_token_ids: Option<String>,
    #[arg(long, help = "Maximum generated tokens per prompt. Defaults to 1.")]
    pub max_new_tokens: Option<usize>,
    #[arg(long)]
    pub prefill_chunk_size: Option<usize>,
    #[arg(long)]
    pub prefill_chunk_threshold: Option<usize>,
    #[arg(long)]
    pub prefill_chunk_schedule: Option<String>,
    #[arg(long, default_value_t = 60)]
    pub startup_timeout_secs: u64,
    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        help = "Before driving prompts, verify all stages still answer binary readiness probes."
    )]
    pub check_stage_readiness: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Probe each upstream->downstream stage link before driving prompts."
    )]
    pub stage_connectivity_probe: bool,
    #[arg(long, default_value_t = 180)]
    pub stage_connectivity_probe_attempts: u32,
    #[arg(long, default_value_t = 2)]
    pub stage_connectivity_probe_timeout_secs: u64,
    #[arg(long, default_value_t = 1000)]
    pub stage_connectivity_probe_retry_delay_ms: u64,
    #[arg(long, default_value_t = false)]
    pub stage_connectivity_diagnostics: bool,
    #[arg(
        long,
        help = "Driver result output. Defaults to <run-dir>/driver-result-reuse.json."
    )]
    pub output: Option<PathBuf>,
}

#[derive(Parser)]
pub struct GlmDsaOpReportArgs {
    #[arg(long, required = true)]
    pub log: Vec<PathBuf>,
    #[arg(
        long,
        help = "Only include the first N timing records from each log. Use this for one request when a REPL log contains follow-up prompts."
    )]
    pub first_records: Option<usize>,
    #[arg(
        long,
        help = "Fail unless IndexShare trace proves Full producer top-k generation and Shared consumer reuse."
    )]
    pub require_indexshare_producer_consumer: bool,
    #[arg(
        long,
        help = "Fail unless every decode stage uses compact K/V gather and has zero sparse-mask nodes."
    )]
    pub require_compact_decode_no_sparse_mask: bool,
    #[arg(
        long,
        help = "Fail unless decode policy logs prove compact fallback was selected with backend support evidence."
    )]
    pub require_compact_decode_policy_evidence: bool,
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum GlmDsaAggregateReportCase {
    TopLevel,
    Baseline,
    Candidate,
}

impl GlmDsaAggregateReportCase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TopLevel => "top-level",
            Self::Baseline => "baseline",
            Self::Candidate => "candidate",
        }
    }
}

#[derive(Parser)]
pub struct GlmDsaAggregateReportsArgs {
    #[arg(long, required = true)]
    pub report: Vec<PathBuf>,
    #[arg(long, value_enum, default_value = "top-level")]
    pub case: GlmDsaAggregateReportCase,
    #[arg(
        long,
        default_value_t = 0.10,
        help = "Fraction of run means to trim from each tail when computing trimmed_mean_ms."
    )]
    pub trim_fraction: f64,
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Parser)]
pub struct GlmDsaLayerMicrobenchArgs {
    #[arg(
        long,
        help = "Local Skippy layer-package directory containing model-package.json."
    )]
    pub stage_model: PathBuf,
    #[arg(long, default_value = "meshllm/GLM-5.2-Q2_K-MTP-Q8-layers")]
    pub model_id: String,
    #[arg(long, default_value_t = 30)]
    pub layer_start: u32,
    #[arg(long, default_value_t = 31)]
    pub layer_end: u32,
    #[arg(long, default_value_t = 131072)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = 6144)]
    pub activation_width: u32,
    #[arg(long, default_value_t = 1)]
    pub tokens: usize,
    #[arg(
        long,
        default_value_t = 0,
        help = "Starting token position for GLM-DSA layer microbench inputs."
    )]
    pub position_start: i32,
    #[arg(
        long,
        default_value_t = 0,
        help = "Populate the target stage KV cache with this many synthetic prefix tokens before each timed run. The prefix must end exactly at position_start."
    )]
    pub kv_warmup_tokens: usize,
    #[arg(
        long,
        help = "Override the synthetic KV warmup chunk size. Defaults to explicit n_ubatch/n_batch when set, otherwise the conservative 128-token warmup chunk."
    )]
    pub kv_warmup_chunk_tokens: Option<usize>,
    #[arg(
        long,
        default_value_t = false,
        help = "Populate the target KV cache by importing synthetic zero KV pages instead of running warmup prefill. GLM-DSA microbench only."
    )]
    pub synthetic_kv_warmup: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Warm the synthetic KV prefix once, checkpoint it, and restore that checkpoint before each measured GLM-DSA decode sample."
    )]
    pub reuse_kv_warmup_checkpoint: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Warm the synthetic KV prefix once, then measure consecutive decode samples on the same session without restoring the KV position."
    )]
    pub reuse_kv_warmup_stream: bool,
    #[arg(long, default_value_t = 3)]
    pub iterations: usize,
    #[arg(long, default_value_t = 1)]
    pub warmup: usize,
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    pub n_gpu_layers: i32,
    #[arg(long)]
    pub n_batch: Option<u32>,
    #[arg(long)]
    pub n_ubatch: Option<u32>,
    #[arg(long, default_value = "f16")]
    pub cache_type_k: String,
    #[arg(long, default_value = "f16")]
    pub cache_type_v: String,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub direct_sparse_attn: bool,
    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::SetTrue,
        help = "Do not set the GLM-DSA direct sparse decode env toggle; prove llama.cpp's native default policy instead."
    )]
    pub native_default_direct_sparse_attn: bool,
    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::Set,
        help = "Enable the experimental GLM-DSA compact top-k K/V + flash-attention path for the candidate run."
    )]
    pub compact_flash_attn: bool,
    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::SetTrue,
        help = "Allow llama.cpp's native GLM-DSA compact flash-attention policy to select the compact path without forcing it on."
    )]
    pub allow_compact_flash_auto: bool,
    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::Set,
        help = "Enable the experimental native Metal GLM-DSA selected-row flash path that fuses compact GET_ROWS into flash attention."
    )]
    pub selected_row_flash: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub direct_sparse_prefill: bool,
    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::SetTrue,
        help = "Do not set the GLM-DSA direct sparse prefill env toggle; prove llama.cpp's native default policy instead."
    )]
    pub native_default_direct_sparse_prefill: bool,
    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::SetTrue,
        help = "Set SKIPPY_GLM_DSA_ENABLE_UNPROVEN_LARGE_DIRECT_SPARSE_PREFILL for candidate runs that intentionally test the large-prefill memory-guard path."
    )]
    pub enable_unproven_large_direct_sparse_prefill: bool,
    #[arg(
        long,
        help = "Set SKIPPY_GLM_DSA_DIRECT_SPARSE_PREFILL_MAX_TOKENS for this microbench run. Larger prefill batches require the explicit unproven-large opt-in."
    )]
    pub direct_sparse_prefill_max_tokens: Option<u32>,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub fused_sparse_mask: bool,
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    pub parallel_lightning_indexer: bool,
    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::Set,
        help = "Set SKIPPY_GLM_DSA_EXPERIMENTAL_MASKED_TOP_K for candidate runs that fuse GLM-DSA indexer mask semantics into top-k."
    )]
    pub masked_top_k: bool,
    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::Set,
        help = "Set SKIPPY_GLM_DSA_EXPERIMENTAL_INDEXER_TOP_K for candidate runs that fuse GLM-DSA Lightning Indexer, mask add, and top-k."
    )]
    pub indexer_top_k: bool,
    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::Set,
        help = "Set SKIPPY_GLM_DSA_EXPERIMENTAL_DECODE_CLIP_TOP_K for candidate decode runs that clip indexer scores to visible KV before top-k."
    )]
    pub decode_clip_top_k: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub op_timing: bool,
    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::Set,
        help = "Capture native Metal dispatch logs without enabling per-op timing callbacks."
    )]
    pub metal_dispatch_log: bool,
    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::Set,
        help = "Enable native GLM-DSA tensor trace digests for routed-MoE route tensors so baseline/candidate route outputs can be compared."
    )]
    pub trace_route_tensors: bool,
    #[arg(
        long,
        default_value = "ffn_moe_topk,ffn_moe_weights,ffn_moe_weights_norm,ffn_moe_weights_scaled",
        help = "Comma-separated native tensor-name substrings traced when --trace-route-tensors is enabled."
    )]
    pub trace_route_tensor_filter: String,
    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        help = "Enable the native Metal GLM-DSA top-k MoE route fusion."
    )]
    pub metal_topk_moe_route_fusion: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Do not inject route-fusion env overrides; clear route-fusion env knobs so the run proves llama.cpp's native Metal GLM-DSA default."
    )]
    pub metal_topk_moe_route_fusion_native_default: bool,
    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::Set,
        help = "Enable the experimental native Metal GLM-DSA MoE motif co-encode path."
    )]
    pub moe_motif_coencode: bool,
    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::Set,
        help = "Enable the experimental native Metal GLM-DSA routed down + weighted-sum fusion."
    )]
    pub moe_down_weighted_fusion: bool,
    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::Set,
        help = "Enable the experimental native Metal GLM-DSA routed down path that computes weighted expert slots in parallel before reduction."
    )]
    pub moe_down_weighted_parallel: bool,
    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::Set,
        help = "Enable the diagnostic native Metal GLM-DSA routed down path that writes unweighted expert slots before the normal weighted-sum reduction."
    )]
    pub moe_down_unweighted_slots: bool,
    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::Set,
        help = "Enable the experimental native Metal GLM-DSA q2_K routed gate/up SwigLU fusion."
    )]
    pub moe_q2_gate_up_swiglu: bool,
    #[arg(
        long,
        help = "Set SKIPPY_GLM_DSA_SPARSE_ATTN_THREADS for the candidate run. Use with --compare-metal-sparse-attn-threads-baseline to compare Metal sparse-attention thread counts."
    )]
    pub sparse_attn_threads: Option<u32>,
    #[arg(
        long,
        help = "Set SKIPPY_GLM_DSA_SPARSE_ATTN_DECODE_GROUP_HEADS for the candidate run. Valid values are 2 or 4."
    )]
    pub sparse_attn_group_heads: Option<u32>,
    #[arg(
        long,
        help = "Set LLAMA_GLM_DSA_PARALLEL_LIGHTNING_INDEXER_THREADS for the candidate run. Valid values are 32, 64, 128, 256, 512, or 1024."
    )]
    pub lightning_indexer_threads: Option<u32>,
    #[arg(
        long,
        help = "Set LLAMA_GLM_DSA_INDEXSHARE_FREQ for this microbench run so every Nth GLM-DSA layer recomputes top-k and intervening layers reuse it."
    )]
    pub indexshare_freq: Option<u32>,
    #[arg(
        long,
        help = "Set LLAMA_GLM_DSA_INDEXSHARE_PATTERN for this microbench run. Use F/S characters for Full/reused-Shared GLM-DSA layers."
    )]
    pub indexshare_pattern: Option<String>,
    #[arg(
        long,
        help = "Set SKIPPY_GLM_DSA_DENSE_SPARSE_MASK_MAX_BYTES for this microbench run. Used for direct-sparse decision telemetry and memory guard experiments."
    )]
    pub dense_sparse_mask_max_bytes: Option<u64>,
    #[arg(
        long,
        help = "Set SKIPPY_GLM_DSA_DIRECT_SPARSE_PREFILL_MIN_KV_TOPK_RATIO for this microbench run. Lower this only when intentionally forcing direct sparse prefill proof runs."
    )]
    pub direct_sparse_prefill_min_kv_topk_ratio: Option<u32>,
    #[arg(
        long,
        default_value_t = false,
        help = "Fail unless the optimized dispatch profile has at least one GLM-DSA route-fusion encode candidate, no skipped encode candidates, and at least one fused route dispatch."
    )]
    pub require_optimized_route_fusion: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Fail unless the run proves large-prefill GLM-DSA direct sparse attention avoided sparse-mask timing nodes and used the Metal correctness-capped sparse-attention dispatch."
    )]
    pub require_direct_sparse_prefill_proof: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Fail unless the run proves decode-shaped GLM-DSA direct sparse attention avoided sparse-mask timing nodes and dispatched native sparse attention."
    )]
    pub require_direct_sparse_decode_proof: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Fail unless the run proves GLM-DSA direct sparse attention executed a partial top-k shape where visible KV is larger than the top-k sideband width."
    )]
    pub require_partial_top_k_proof: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Fail unless the candidate proves an optimized compact GLM-DSA decode path: either compact flash with typed/fused get-rows or fused top-1 attention, with no promoted get-rows and no old dsa_sparse_attn dispatch."
    )]
    pub require_compact_flash_proof: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Fail unless the candidate proves GLM-DSA MoE aggregation used the Metal moe_weighted_sum f32x4 path."
    )]
    pub require_moe_weighted_sum_proof: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Fail unless the optimized dispatch profile proves natural-order GLM-DSA MoE motifs are backend candidates for native Metal fusion."
    )]
    pub require_moe_motif_proof: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Fail unless native llama.cpp logs prove a GLM-DSA Full layer produced top-k and a Shared layer consumed it in the same local stage."
    )]
    pub require_native_indexshare_proof: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Fail unless the run proves a real GLM-DSA top-k sideband was carried through Skippy stage wire and consumed by a Shared consumer layer."
    )]
    pub require_real_top_k_shared_consumer_proof: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Run a dense-mask fallback baseline and compare it with the requested direct sparse settings."
    )]
    pub compare_dense_fallback: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Run native dense-mask flash prefill as the baseline and forced direct-sparse prefill as the candidate."
    )]
    pub compare_dense_flash_prefill: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Run a CPU direct-sparse baseline and compare it with the requested backend settings."
    )]
    pub compare_cpu_direct_sparse: bool,
    #[arg(
        long,
        help = "Run a same-backend Metal direct-sparse baseline with this SKIPPY_GLM_DSA_SPARSE_ATTN_THREADS value and compare it with the candidate run."
    )]
    pub compare_metal_sparse_attn_threads_baseline: Option<u32>,
    #[arg(
        long,
        default_value_t = false,
        help = "Run the same GLM-DSA Metal compact-flash case with selected-row flash disabled as the baseline and enabled as the candidate."
    )]
    pub compare_selected_row_flash: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Run the same GLM-DSA case with parallel Lightning Indexer disabled as the baseline and enabled as the candidate."
    )]
    pub compare_parallel_lightning_indexer: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Compare baseline GLM-DSA indexer ADD+TOP_K against the experimental masked top-k fusion."
    )]
    pub compare_masked_top_k: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Compare baseline GLM-DSA indexer + mask ADD + TOP_K against the experimental fused indexer-top-k path."
    )]
    pub compare_indexer_top_k: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Compare baseline GLM-DSA indexer mask ADD+TOP_K against decode-only visible-KV score clipping before top-k."
    )]
    pub compare_decode_clip_top_k: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Run the same GLM-DSA Metal case with routed down + weighted-sum fusion disabled as the baseline and enabled as the candidate."
    )]
    pub compare_moe_down_weighted_fusion: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Run the same GLM-DSA Metal case with routed down + weighted slots disabled as the baseline and the expert-parallel weighted-slots path enabled as the candidate."
    )]
    pub compare_moe_down_weighted_parallel: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Run the same GLM-DSA Metal case with routed down + weighted slots disabled as the baseline and the diagnostic unweighted-slots path enabled as the candidate."
    )]
    pub compare_moe_down_unweighted_slots: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Run the same GLM-DSA Metal case with q2_K routed gate/up SwigLU fusion disabled as the baseline and enabled as the candidate."
    )]
    pub compare_moe_q2_gate_up_swiglu: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Compare native in-graph GLM-DSA Full->Shared execution with a local producer stage feeding the Shared consumer stage."
    )]
    pub compare_native_indexshare_producer_consumer: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "When comparing native GLM-DSA IndexShare producer/consumer execution, skip the poisoned-sideband sensitivity rerun. Use only for faster timing gates after the poisoned proof has already passed."
    )]
    pub skip_native_indexshare_poison: bool,
    #[arg(long, default_value_t = 1.0e-3)]
    pub parity_atol: f32,
    #[arg(long, default_value_t = 1.0e-3)]
    pub parity_rtol: f32,
    #[arg(
        long,
        default_value_t = false,
        help = "Allow another glm-dsa-layer-microbench process to run concurrently. Disabled by default because overlapping Metal runs contaminate timing evidence."
    )]
    pub allow_concurrent: bool,
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Parser)]
pub struct GlmDsaOpCompareArgs {
    #[arg(
        long,
        required = true,
        help = "Baseline glm-dsa-op-report JSON. Repeat for multiple per-stage reports."
    )]
    pub baseline_report: Vec<PathBuf>,
    #[arg(
        long,
        required = true,
        help = "Candidate glm-dsa-op-report JSON. Repeat for multiple per-stage reports."
    )]
    pub candidate_report: Vec<PathBuf>,
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Parser)]
pub struct VerifySpanLocalArgs {
    #[arg(long)]
    pub model_path: PathBuf,
    #[arg(long, default_value_t = 48)]
    pub layer_end: u32,
    #[arg(long)]
    pub split_layer: Option<u32>,
    #[arg(long, default_value_t = 4096)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    pub n_gpu_layers: i32,
    #[arg(long, default_value = "f16")]
    pub cache_type_k: String,
    #[arg(long, default_value = "f16")]
    pub cache_type_v: String,
    #[arg(long)]
    pub n_batch: Option<u32>,
    #[arg(long)]
    pub n_ubatch: Option<u32>,
    #[arg(long, default_value_t = 64)]
    pub iterations: usize,
    #[arg(long, default_value_t = 8)]
    pub warmup: usize,
    #[arg(
        long,
        default_value = "Write a Rust function that parses a list of integers and returns the median."
    )]
    pub prompt: String,
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Parser)]
pub struct ChatCorpusArgs {
    #[arg(long, default_value = "http://127.0.0.1:9337/v1")]
    pub base_url: String,
    #[arg(long, default_value = DEFAULT_LOCAL_MODEL_ID)]
    pub model: String,
    #[arg(long, default_value = "Hello")]
    pub prompt: String,
    #[arg(long)]
    pub prompt_corpus: Option<PathBuf>,
    #[arg(long)]
    pub prompt_id: Option<String>,
    #[arg(long)]
    pub category: Option<String>,
    #[arg(long)]
    pub length_bucket: Option<String>,
    #[arg(long)]
    pub prompt_limit: Option<usize>,
    #[arg(long, default_value_t = 16)]
    pub max_tokens: u32,
    #[arg(long, default_value_t = 1)]
    pub concurrency_depth: usize,
    #[arg(long)]
    pub stream: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub include_usage: bool,
    #[arg(long, default_value_t = 600)]
    pub request_timeout_secs: u64,
    #[arg(long)]
    pub output: Option<PathBuf>,
    #[arg(long, default_value = "chat-corpus-session")]
    pub session_prefix: String,
    #[arg(long)]
    pub temperature: Option<f32>,
    #[arg(long)]
    pub top_p: Option<f32>,
    #[arg(long)]
    pub top_k: Option<i32>,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long)]
    pub enable_thinking: Option<bool>,
    #[arg(long)]
    pub reasoning_effort: Option<String>,
}

#[derive(Parser)]
pub struct RunArgs {
    #[arg(long, default_value = "target/debug/metrics-server")]
    pub metrics_server_bin: PathBuf,
    #[arg(long, default_value = "target/debug/skippy-server")]
    pub stage_server_bin: PathBuf,
    #[arg(
        long,
        help = "Comma-separated unique stage hosts. Distributed lab runs require one separate node per stage."
    )]
    pub hosts: String,
    #[arg(long)]
    pub run_id: Option<String>,
    #[arg(long, default_value = "distributed-layer-package")]
    pub topology_id: String,
    #[arg(long, default_value = DEFAULT_LOCAL_MODEL_ID)]
    pub model_id: String,
    #[arg(long)]
    pub model_path: Option<PathBuf>,
    #[arg(long)]
    pub stage_model: Option<PathBuf>,
    #[arg(long, default_value = "layer-package")]
    pub stage_load_mode: String,
    #[arg(
        long,
        default_value = "14,27",
        help = "Comma-separated layer boundaries. Lab runs must be evenly balanced; Qwen3.6 40 layers on three hosts uses 14,27."
    )]
    pub splits: String,
    #[arg(long, default_value_t = 40)]
    pub layer_end: u32,
    #[arg(long, default_value_t = 128)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    pub n_gpu_layers: i32,
    #[arg(long, default_value = "f16")]
    pub cache_type_k: String,
    #[arg(long, default_value = "f16")]
    pub cache_type_v: String,
    #[arg(long, default_value_t = 2048)]
    pub activation_width: i32,
    #[arg(long, default_value = "f16")]
    pub activation_wire_dtype: String,
    #[arg(long, default_value = "Hello")]
    pub prompt: String,
    #[arg(long)]
    pub prompt_corpus: Option<PathBuf>,
    #[arg(long)]
    pub prompt_limit: Option<usize>,
    #[arg(long)]
    pub prompt_token_ids: Option<String>,
    #[arg(long, help = "Maximum generated tokens per prompt. Defaults to 1.")]
    pub max_new_tokens: Option<usize>,
    #[arg(long)]
    pub prefill_chunk_size: Option<usize>,
    #[arg(
        long,
        help = "Only split prefill into chunks when the prefill token count is above this threshold."
    )]
    pub prefill_chunk_threshold: Option<usize>,
    #[arg(
        long,
        help = "Comma-separated MIN_TOKENS:CHUNK_SIZE overrides for adaptive prefill chunking, for example 513:512."
    )]
    pub prefill_chunk_schedule: Option<String>,
    #[arg(long, default_value = "127.0.0.1:18080")]
    pub metrics_http_addr: SocketAddr,
    #[arg(long, default_value = "127.0.0.1:14317")]
    pub metrics_otlp_grpc_addr: SocketAddr,
    #[arg(long)]
    pub metrics_otlp_grpc_url: Option<String>,
    #[arg(long)]
    pub db: Option<PathBuf>,
    #[arg(long)]
    pub output: Option<PathBuf>,
    #[arg(long, default_value = "/Volumes/External/skippy-runtime-bench")]
    pub work_dir: PathBuf,
    #[arg(long, default_value = "/tmp/skippy-runtime-bench")]
    pub remote_root: String,
    #[arg(long)]
    pub remote_root_map: Option<String>,
    #[arg(long)]
    pub remote_shared_root_map: Option<String>,
    #[arg(
        long,
        help = "Comma-separated host=PATH overrides for --stage-model when hosts see a shared layer package at different mount points."
    )]
    pub stage_model_path_map: Option<String>,
    #[arg(long)]
    pub endpoint_host_map: Option<String>,
    #[arg(long, default_value = "0.0.0.0")]
    pub remote_bind_host: String,
    #[arg(long, default_value_t = 19031)]
    pub first_stage_port: u16,
    #[arg(long)]
    pub execute_remote: bool,
    #[arg(long)]
    pub keep_remote: bool,
    #[arg(long)]
    pub rsync_model_artifacts: bool,
    #[arg(long)]
    pub child_logs: bool,
    #[arg(long, default_value_t = 60)]
    pub startup_timeout_secs: u64,
    #[arg(long, default_value_t = 4)]
    pub stage_max_inflight: usize,
    #[arg(long)]
    pub stage_reply_credit_limit: Option<usize>,
    #[arg(
        long,
        help = "Pass --async-prefill-forward to every binary stage server."
    )]
    pub stage_async_prefill_forward: bool,
    #[arg(
        long,
        default_value_t = 0.0,
        help = "Pass artificial downstream wire delay in milliseconds to every binary stage server."
    )]
    pub stage_downstream_wire_delay_ms: f64,
    #[arg(
        long,
        help = "Pass artificial downstream activation bandwidth cap in megabits per second to every binary stage server."
    )]
    pub stage_downstream_wire_mbps: Option<f64>,
    #[arg(
        long,
        default_value_t = 8192,
        help = "Bounded per-stage telemetry queue capacity. Larger debug corpus runs should keep this above expected burst size."
    )]
    pub stage_telemetry_queue_capacity: usize,
    #[arg(
        long,
        default_value = "summary",
        help = "Stage telemetry volume: off, summary, or debug. Perf runs should use summary."
    )]
    pub stage_telemetry_level: String,
    #[arg(
        long,
        default_value_t = false,
        help = "Allow intentionally unbalanced stage layer counts for heterogeneous lab hosts."
    )]
    pub allow_unbalanced_stages: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Disable mmap-backed backend buffers in generated stage configs. Useful when large materialized Metal stages stall in residency registration."
    )]
    pub stage_disable_mmap_buffer: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "After all binary stages are listening, probe every upstream->downstream stage link with skippy-server probe-downstream before running prompts."
    )]
    pub stage_connectivity_probe: bool,
    #[arg(
        long,
        default_value_t = 180,
        help = "Number of skippy-server probe-downstream attempts for each upstream->downstream stage link."
    )]
    pub stage_connectivity_probe_attempts: u32,
    #[arg(
        long,
        default_value_t = 2,
        help = "Per-attempt timeout in seconds for each stage connectivity probe."
    )]
    pub stage_connectivity_probe_timeout_secs: u64,
    #[arg(
        long,
        default_value_t = 1000,
        help = "Delay between failed stage connectivity probe attempts in milliseconds."
    )]
    pub stage_connectivity_probe_retry_delay_ms: u64,
    #[arg(
        long,
        default_value_t = false,
        help = "When a stage connectivity probe fails, append best-effort route, interface, ARP, and downstream listener diagnostics to the error."
    )]
    pub stage_connectivity_diagnostics: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "When a remote distributed run fails, leave launched stage processes alive for manual probing instead of cleaning them up."
    )]
    pub keep_remote_on_failure: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Enable llama.cpp GLM-DSA op/group timing logs in every launched stage."
    )]
    pub glm_dsa_op_timing: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Enable llama.cpp GLM-DSA direct sparse-attention execution for decode-shaped microbatches in every launched stage."
    )]
    pub glm_dsa_direct_sparse_attn: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Also enable llama.cpp GLM-DSA direct sparse-attention execution for prefill microbatches. Use only for sparse-prefill experiments."
    )]
    pub glm_dsa_direct_sparse_prefill: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Enable native Metal GLM-DSA top-k MoE route fusion in every launched stage."
    )]
    pub glm_dsa_metal_topk_moe_route_fusion: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Enable native Metal GLM-DSA dispatch-shape logs in every launched stage."
    )]
    pub glm_dsa_metal_dispatch_log: bool,
    #[arg(
        long,
        help = "Set LLAMA_GLM_DSA_INDEXSHARE_FREQ in every launched stage so every Nth GLM-DSA layer recomputes top-k and intervening layers reuse it."
    )]
    pub glm_dsa_indexshare_freq: Option<u32>,
    #[arg(
        long,
        help = "Set LLAMA_GLM_DSA_INDEXSHARE_PATTERN in every launched stage. Use F/S characters for Full/reused-Shared GLM-DSA layers."
    )]
    pub glm_dsa_indexshare_pattern: Option<String>,
    #[arg(
        long,
        help = "Set SKIPPY_GLM_DSA_DENSE_SPARSE_MASK_MAX_BYTES in every launched stage. When GLM-DSA direct sparse prefill is enabled, larger dense sparse-mask fallbacks are routed to direct sparse attention."
    )]
    pub glm_dsa_dense_sparse_mask_max_bytes: Option<u64>,
}

#[derive(Parser)]
pub struct LocalSingleArgs {
    #[arg(long, default_value = "target/debug/metrics-server")]
    pub metrics_server_bin: PathBuf,
    #[arg(long, default_value = "target/debug/skippy-server")]
    pub stage_server_bin: PathBuf,
    #[arg(long)]
    pub model_path: PathBuf,
    #[arg(long)]
    pub run_id: Option<String>,
    #[arg(long, default_value = "single-stage-runtime")]
    pub topology_id: String,
    #[arg(long, default_value = DEFAULT_LOCAL_MODEL_ID)]
    pub model_id: String,
    #[arg(long, default_value = "127.0.0.1:18080")]
    pub metrics_http_addr: SocketAddr,
    #[arg(long, default_value = "127.0.0.1:14317")]
    pub metrics_otlp_grpc_addr: SocketAddr,
    #[arg(long, default_value = "127.0.0.1:19001")]
    pub stage_bind_addr: SocketAddr,
    #[arg(long, default_value_t = 128)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = 0)]
    pub n_gpu_layers: i32,
    #[arg(long, default_value = "f16")]
    pub cache_type_k: String,
    #[arg(long, default_value = "f16")]
    pub cache_type_v: String,
    #[arg(long, default_value_t = 0)]
    pub layer_start: u32,
    #[arg(long, default_value_t = 30)]
    pub layer_end: u32,
    #[arg(long, default_value = "Hello")]
    pub prompt: String,
    #[arg(long, default_value_t = 1)]
    pub max_new_tokens: usize,
    #[arg(long)]
    pub db: Option<PathBuf>,
    #[arg(long)]
    pub output: Option<PathBuf>,
    #[arg(long)]
    pub child_logs: bool,
    #[arg(long, default_value_t = 60)]
    pub startup_timeout_secs: u64,
}

#[derive(Parser)]
pub struct LocalSplitInprocessArgs {
    #[arg(long)]
    pub model_path: PathBuf,
    #[arg(long, default_value_t = 15)]
    pub split_layer: u32,
    #[arg(long, default_value_t = 30)]
    pub layer_end: u32,
    #[arg(long, default_value_t = 128)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = 0)]
    pub n_gpu_layers: i32,
    #[arg(long, default_value = "Hello")]
    pub prompt: String,
}

#[derive(Parser)]
pub struct LocalSplitBinaryArgs {
    #[arg(long, default_value = "target/debug/skippy-server")]
    pub stage_server_bin: PathBuf,
    #[arg(long)]
    pub model_path: PathBuf,
    #[arg(
        long,
        default_value = "runtime-slice",
        help = "Stage load mode for --model-path. Use layer-package or layer-package-mmap when --model-path points at a Skippy layer package directory."
    )]
    pub stage_load_mode: String,
    #[arg(long, default_value = DEFAULT_LOCAL_MODEL_ID)]
    pub model_id: String,
    #[arg(long, default_value_t = 15)]
    pub split_layer: u32,
    #[arg(long, default_value_t = 30)]
    pub layer_end: u32,
    #[arg(long, default_value_t = 128)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = 0)]
    pub n_gpu_layers: i32,
    #[arg(long, default_value = "Hello")]
    pub prompt: String,
    #[arg(long, default_value = "127.0.0.1:19011")]
    pub stage1_bind_addr: SocketAddr,
    #[arg(long, default_value = "f16")]
    pub activation_wire_dtype: String,
    #[arg(long)]
    pub child_logs: bool,
    #[arg(long, default_value_t = 60)]
    pub startup_timeout_secs: u64,
}

#[derive(Parser)]
pub struct LocalSplitCompareArgs {
    #[arg(long, default_value = "target/debug/skippy-server")]
    pub stage_server_bin: PathBuf,
    #[arg(long)]
    pub model_path: PathBuf,
    #[arg(long, default_value = DEFAULT_LOCAL_MODEL_ID)]
    pub model_id: String,
    #[arg(long, default_value_t = 15)]
    pub split_layer: u32,
    #[arg(long, default_value_t = 30)]
    pub layer_end: u32,
    #[arg(long, default_value_t = 128)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = 0)]
    pub n_gpu_layers: i32,
    #[arg(long, default_value = "Hello")]
    pub prompt: String,
    #[arg(long, default_value = "127.0.0.1:19021")]
    pub stage1_bind_addr: SocketAddr,
    #[arg(long, default_value = "f16")]
    pub activation_wire_dtype: String,
    #[arg(long)]
    pub child_logs: bool,
    #[arg(long, default_value_t = 60)]
    pub startup_timeout_secs: u64,
    #[arg(long)]
    pub allow_mismatch: bool,
}

#[derive(Parser)]
pub struct LocalSplitChainInprocessArgs {
    #[arg(long)]
    pub model_path: PathBuf,
    #[arg(
        long,
        default_value = "runtime-slice",
        help = "Stage load mode for --model-path. Use layer-package or layer-package-mmap when --model-path points at a Skippy layer package directory."
    )]
    pub stage_load_mode: String,
    #[arg(long, default_value = DEFAULT_LOCAL_MODEL_ID)]
    pub model_id: String,
    #[arg(long, default_value_t = 10)]
    pub split_layer_1: u32,
    #[arg(long, default_value_t = 20)]
    pub split_layer_2: u32,
    #[arg(long, default_value_t = 30)]
    pub layer_end: u32,
    #[arg(long, default_value_t = 128)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = 0)]
    pub n_gpu_layers: i32,
    #[arg(long, default_value = "Hello")]
    pub prompt: String,
    #[arg(
        long,
        default_value_t = 0,
        help = "Prefill this many prompt tokens through the in-process chain, then decode the next token. The prompt must tokenize to at least N+1 tokens."
    )]
    pub prefill_token_count: u32,
    #[arg(
        long,
        help = "Make the final stage include the output head and require a predicted token. Use only when --layer-end is the model output boundary."
    )]
    pub final_output: bool,
}

#[derive(Parser)]
pub struct LocalSplitChainBinaryArgs {
    #[arg(long, default_value = "target/debug/skippy-server")]
    pub stage_server_bin: PathBuf,
    #[arg(long)]
    pub model_path: PathBuf,
    #[arg(
        long,
        default_value = "runtime-slice",
        help = "Stage load mode for --model-path. Use layer-package or layer-package-mmap when --model-path points at a Skippy layer package directory."
    )]
    pub stage_load_mode: String,
    #[arg(long, default_value = DEFAULT_LOCAL_MODEL_ID)]
    pub model_id: String,
    #[arg(long, default_value_t = 10)]
    pub split_layer_1: u32,
    #[arg(long, default_value_t = 20)]
    pub split_layer_2: u32,
    #[arg(long, default_value_t = 30)]
    pub layer_end: u32,
    #[arg(long, default_value_t = 128)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = 0)]
    pub n_gpu_layers: i32,
    #[arg(long, default_value = "Hello")]
    pub prompt: String,
    #[arg(long, default_value = "127.0.0.1:19031")]
    pub stage1_bind_addr: SocketAddr,
    #[arg(long, default_value = "127.0.0.1:19032")]
    pub stage2_bind_addr: SocketAddr,
    #[arg(long, default_value = "f16")]
    pub activation_wire_dtype: String,
    #[arg(long)]
    pub child_logs: bool,
    #[arg(long, default_value_t = 60)]
    pub startup_timeout_secs: u64,
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use clap::Parser;

    use super::{Cli, CommandKind, FocusedRuntimeScenario};

    #[test]
    fn parses_focused_runtime_schema_smoke_command() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "focused-runtime",
            "--schema-smoke",
            "--scenario",
            "first-token",
            "--hosts",
            "host-a,host-b",
            "--splits",
            "1",
            "--layer-end",
            "2",
            "--max-new-tokens",
            "4",
        ])
        .unwrap();

        let CommandKind::FocusedRuntime(args) = cli.command else {
            panic!("expected focused-runtime subcommand");
        };

        assert!(args.schema_smoke);
        assert!(matches!(args.scenario, FocusedRuntimeScenario::FirstToken));
        assert_eq!(args.run.hosts, "host-a,host-b");
        assert_eq!(args.run.splits, "1");
        assert_eq!(args.run.layer_end, 2);
        assert_eq!(args.run.max_new_tokens, Some(4));
    }

    #[test]
    fn focused_runtime_keeps_omitted_max_new_tokens_unset() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "focused-runtime",
            "--schema-smoke",
            "--hosts",
            "host-a,host-b",
            "--splits",
            "1",
            "--layer-end",
            "2",
        ])
        .unwrap();

        let CommandKind::FocusedRuntime(args) = cli.command else {
            panic!("expected focused-runtime subcommand");
        };

        assert_eq!(args.run.max_new_tokens, None);
    }

    #[test]
    fn parses_drive_existing_command() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "drive-existing",
            "--run-dir",
            "/tmp/skippy-runtime-bench/run-1",
            "--stage-model",
            "/models/package",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "16",
            "--stage-connectivity-probe",
            "--stage-connectivity-probe-attempts",
            "180",
        ])
        .unwrap();

        let CommandKind::DriveExisting(args) = cli.command else {
            panic!("expected drive-existing subcommand");
        };

        assert_eq!(
            args.run_dir,
            PathBuf::from("/tmp/skippy-runtime-bench/run-1")
        );
        assert_eq!(args.stage_model, Some(PathBuf::from("/models/package")));
        assert_eq!(args.prompt, "hello");
        assert_eq!(args.max_new_tokens, Some(16));
        assert!(args.stage_connectivity_probe);
        assert_eq!(args.stage_connectivity_probe_attempts, 180);
    }

    #[test]
    fn parses_verify_span_local_command() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "verify-span-local",
            "--model-path",
            "/tmp/model.gguf",
            "--layer-end",
            "48",
            "--iterations",
            "3",
            "--warmup",
            "1",
            "--n-gpu-layers",
            "-1",
        ])
        .unwrap();

        let CommandKind::VerifySpanLocal(args) = cli.command else {
            panic!("expected verify-span-local subcommand");
        };

        assert_eq!(args.model_path, PathBuf::from("/tmp/model.gguf"));
        assert_eq!(args.layer_end, 48);
        assert_eq!(args.split_layer, None);
        assert_eq!(args.iterations, 3);
        assert_eq!(args.warmup, 1);
        assert_eq!(args.n_gpu_layers, -1);
    }

    #[test]
    fn parses_verify_span_local_split_layer() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "verify-span-local",
            "--model-path",
            "/tmp/model.gguf",
            "--split-layer",
            "24",
        ])
        .unwrap();

        let CommandKind::VerifySpanLocal(args) = cli.command else {
            panic!("expected verify-span-local subcommand");
        };

        assert_eq!(args.split_layer, Some(24));
    }

    #[test]
    fn parses_local_split_binary_layer_package_mode() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "local-split-binary",
            "--model-path",
            "/tmp/glm52-layers",
            "--stage-load-mode",
            "layer-package-mmap",
            "--split-layer",
            "31",
            "--layer-end",
            "32",
        ])
        .unwrap();

        let CommandKind::LocalSplitBinary(args) = cli.command else {
            panic!("expected local-split-binary subcommand");
        };

        assert_eq!(args.model_path, PathBuf::from("/tmp/glm52-layers"));
        assert_eq!(args.stage_load_mode, "layer-package-mmap");
        assert_eq!(args.split_layer, 31);
        assert_eq!(args.layer_end, 32);
    }

    #[test]
    fn parses_local_split_chain_binary_layer_package_mode() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "local-split-chain-binary",
            "--model-path",
            "/tmp/glm52-layers",
            "--stage-load-mode",
            "layer-package",
            "--split-layer-1",
            "6",
            "--split-layer-2",
            "7",
            "--layer-end",
            "8",
        ])
        .unwrap();

        let CommandKind::LocalSplitChainBinary(args) = cli.command else {
            panic!("expected local-split-chain-binary subcommand");
        };

        assert_eq!(args.model_path, PathBuf::from("/tmp/glm52-layers"));
        assert_eq!(args.stage_load_mode, "layer-package");
        assert_eq!(args.split_layer_1, 6);
        assert_eq!(args.split_layer_2, 7);
        assert_eq!(args.layer_end, 8);
    }

    #[test]
    fn parses_local_split_chain_inprocess_layer_package_mode() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "local-split-chain-inprocess",
            "--model-path",
            "/tmp/glm52-layers",
            "--stage-load-mode",
            "layer-package",
            "--split-layer-1",
            "6",
            "--split-layer-2",
            "7",
            "--layer-end",
            "8",
            "--prefill-token-count",
            "16",
            "--final-output",
        ])
        .unwrap();

        let CommandKind::LocalSplitChainInprocess(args) = cli.command else {
            panic!("expected local-split-chain-inprocess subcommand");
        };

        assert_eq!(args.model_path, PathBuf::from("/tmp/glm52-layers"));
        assert_eq!(args.stage_load_mode, "layer-package");
        assert_eq!(args.split_layer_1, 6);
        assert_eq!(args.split_layer_2, 7);
        assert_eq!(args.layer_end, 8);
        assert_eq!(args.prefill_token_count, 16);
        assert!(args.final_output);
    }

    #[test]
    fn parses_glm_dsa_layer_microbench_command() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "glm-dsa-layer-microbench",
            "--stage-model",
            "/tmp/glm52-layers",
            "--layer-start",
            "30",
            "--layer-end",
            "31",
            "--tokens",
            "128",
            "--position-start",
            "255",
            "--kv-warmup-tokens",
            "255",
            "--kv-warmup-chunk-tokens",
            "64",
            "--synthetic-kv-warmup",
            "--reuse-kv-warmup-checkpoint",
            "--indexshare-freq",
            "4",
            "--indexshare-pattern",
            "FSSS",
            "--dense-sparse-mask-max-bytes",
            "1048576",
            "--enable-unproven-large-direct-sparse-prefill",
            "--direct-sparse-prefill-max-tokens",
            "12",
            "--native-default-direct-sparse-attn",
            "--allow-compact-flash-auto",
            "--require-optimized-route-fusion",
            "--require-moe-weighted-sum-proof",
            "--require-moe-motif-proof",
            "--require-native-indexshare-proof",
            "--require-direct-sparse-decode-proof",
            "--compare-native-indexshare-producer-consumer",
            "--skip-native-indexshare-poison",
        ])
        .unwrap();

        let CommandKind::GlmDsaLayerMicrobench(args) = cli.command else {
            panic!("expected glm-dsa-layer-microbench subcommand");
        };

        assert_eq!(args.stage_model, PathBuf::from("/tmp/glm52-layers"));
        assert_eq!(args.layer_start, 30);
        assert_eq!(args.layer_end, 31);
        assert_eq!(args.tokens, 128);
        assert_eq!(args.position_start, 255);
        assert_eq!(args.kv_warmup_tokens, 255);
        assert_eq!(args.kv_warmup_chunk_tokens, Some(64));
        assert!(args.synthetic_kv_warmup);
        assert!(args.reuse_kv_warmup_checkpoint);
        assert_eq!(args.indexshare_freq, Some(4));
        assert_eq!(args.indexshare_pattern.as_deref(), Some("FSSS"));
        assert_eq!(args.dense_sparse_mask_max_bytes, Some(1_048_576));
        assert!(args.enable_unproven_large_direct_sparse_prefill);
        assert_eq!(args.direct_sparse_prefill_max_tokens, Some(12));
        assert!(args.native_default_direct_sparse_attn);
        assert!(args.allow_compact_flash_auto);
        assert!(args.require_optimized_route_fusion);
        assert!(args.require_moe_weighted_sum_proof);
        assert!(args.require_moe_motif_proof);
        assert!(args.require_native_indexshare_proof);
        assert!(args.require_direct_sparse_decode_proof);
        assert!(args.compare_native_indexshare_producer_consumer);
        assert!(args.skip_native_indexshare_poison);
        assert!(!args.compare_dense_fallback);
        assert!(!args.compare_dense_flash_prefill);
        assert!(!args.compare_cpu_direct_sparse);
        assert_eq!(args.sparse_attn_threads, None);
        assert_eq!(args.compare_metal_sparse_attn_threads_baseline, None);
        assert!(!args.allow_concurrent);
    }

    #[test]
    fn parses_glm_dsa_layer_microbench_concurrency_override() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "glm-dsa-layer-microbench",
            "--stage-model",
            "/tmp/glm52-layers",
            "--allow-concurrent",
        ])
        .unwrap();

        let CommandKind::GlmDsaLayerMicrobench(args) = cli.command else {
            panic!("expected glm-dsa-layer-microbench subcommand");
        };

        assert!(args.allow_concurrent);
    }

    #[test]
    fn parses_glm_dsa_layer_microbench_dense_flash_prefill_comparison() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "glm-dsa-layer-microbench",
            "--stage-model",
            "/tmp/glm52-layers",
            "--compare-dense-flash-prefill",
        ])
        .unwrap();

        let CommandKind::GlmDsaLayerMicrobench(args) = cli.command else {
            panic!("expected glm-dsa-layer-microbench subcommand");
        };

        assert!(args.compare_dense_flash_prefill);
        assert!(!args.compare_dense_fallback);
        assert!(!args.compare_cpu_direct_sparse);
        assert_eq!(args.compare_metal_sparse_attn_threads_baseline, None);
    }

    #[test]
    fn parses_glm_dsa_layer_microbench_cpu_direct_comparison() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "glm-dsa-layer-microbench",
            "--stage-model",
            "/tmp/glm52-layers",
            "--compare-cpu-direct-sparse",
        ])
        .unwrap();

        let CommandKind::GlmDsaLayerMicrobench(args) = cli.command else {
            panic!("expected glm-dsa-layer-microbench subcommand");
        };

        assert!(args.compare_cpu_direct_sparse);
        assert!(!args.compare_dense_fallback);
        assert_eq!(args.compare_metal_sparse_attn_threads_baseline, None);
    }

    #[test]
    fn parses_glm_dsa_layer_microbench_metal_thread_comparison() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "glm-dsa-layer-microbench",
            "--stage-model",
            "/tmp/glm52-layers",
            "--sparse-attn-threads",
            "256",
            "--compare-metal-sparse-attn-threads-baseline",
            "32",
        ])
        .unwrap();

        let CommandKind::GlmDsaLayerMicrobench(args) = cli.command else {
            panic!("expected glm-dsa-layer-microbench subcommand");
        };

        assert_eq!(args.sparse_attn_threads, Some(256));
        assert_eq!(args.compare_metal_sparse_attn_threads_baseline, Some(32));
        assert!(!args.compare_dense_fallback);
        assert!(!args.compare_cpu_direct_sparse);
    }

    #[test]
    fn parses_glm_dsa_aggregate_reports_command() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "glm-dsa-aggregate-reports",
            "--report",
            "/tmp/direct-a.json",
            "--report",
            "/tmp/direct-b.json",
            "--case",
            "candidate",
            "--trim-fraction",
            "0.2",
            "--output",
            "/tmp/aggregate.json",
        ])
        .unwrap();

        let CommandKind::GlmDsaAggregateReports(args) = cli.command else {
            panic!("expected glm-dsa-aggregate-reports subcommand");
        };

        assert_eq!(
            args.report,
            vec![
                PathBuf::from("/tmp/direct-a.json"),
                PathBuf::from("/tmp/direct-b.json"),
            ]
        );
        assert!(matches!(
            args.case,
            super::GlmDsaAggregateReportCase::Candidate
        ));
        assert_eq!(args.trim_fraction, 0.2);
        assert_eq!(args.output, Some(PathBuf::from("/tmp/aggregate.json")));
    }
}
