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
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    pub direct_sparse_prefill: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub fused_sparse_mask: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub parallel_lightning_indexer: bool,
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
        help = "Fail unless the optimized dispatch profile has at least one GLM-DSA route-fusion encode candidate, no skipped encode candidates, and at least one fused route dispatch."
    )]
    pub require_optimized_route_fusion: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Run a dense-mask fallback baseline and compare it with the requested direct sparse settings."
    )]
    pub compare_dense_fallback: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Run a CPU direct-sparse baseline and compare it with the requested backend settings."
    )]
    pub compare_cpu_direct_sparse: bool,
    #[arg(long, default_value_t = 1.0e-3)]
    pub parity_atol: f32,
    #[arg(long, default_value_t = 1.0e-3)]
    pub parity_rtol: f32,
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
pub struct LocalSplitChainBinaryArgs {
    #[arg(long, default_value = "target/debug/skippy-server")]
    pub stage_server_bin: PathBuf,
    #[arg(long)]
    pub model_path: PathBuf,
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
            "--require-optimized-route-fusion",
            "--compare-dense-fallback",
        ])
        .unwrap();

        let CommandKind::GlmDsaLayerMicrobench(args) = cli.command else {
            panic!("expected glm-dsa-layer-microbench subcommand");
        };

        assert_eq!(args.stage_model, PathBuf::from("/tmp/glm52-layers"));
        assert_eq!(args.layer_start, 30);
        assert_eq!(args.layer_end, 31);
        assert_eq!(args.tokens, 128);
        assert!(args.require_optimized_route_fusion);
        assert!(args.compare_dense_fallback);
        assert!(!args.compare_cpu_direct_sparse);
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
    }
}
