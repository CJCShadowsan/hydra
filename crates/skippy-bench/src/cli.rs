use std::{net::SocketAddr, path::PathBuf};

use clap::{Parser, Subcommand, ValueEnum};

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
    #[command(name = "chat-corpus")]
    ChatCorpus(ChatCorpusArgs),
    #[command(name = "token-lengths")]
    TokenLengths(TokenLengthsArgs),
    #[command(name = "spd-fixture-parity")]
    SpdFixtureParity(SpdFixtureParityArgs),
    #[command(name = "spd-live-tap-parity")]
    SpdLiveTapParity(SpdLiveTapParityArgs),
    #[command(name = "spd-openai-smoke")]
    SpdOpenAiSmoke(SpdOpenAiSmokeArgs),
    #[command(name = "spd-openai-check")]
    SpdOpenAiCheck(SpdOpenAiCheckArgs),
    #[command(name = "focused-runtime")]
    FocusedRuntime(FocusedRuntimeArgs),
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
pub struct SpdFixtureParityArgs {
    #[arg(long)]
    pub manifest: PathBuf,
    #[arg(long)]
    pub fixture: PathBuf,
    #[arg(long, default_value_t = 8)]
    pub top_k: usize,
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Parser)]
pub struct SpdLiveTapParityArgs {
    #[arg(long)]
    pub manifest: PathBuf,
    #[arg(long)]
    pub fixture: PathBuf,
    #[arg(long)]
    pub model_path: PathBuf,
    #[arg(long, value_delimiter = ',')]
    pub splits: Vec<u32>,
    #[arg(long, default_value_t = 32)]
    pub layer_end: u32,
    #[arg(long, default_value_t = 128)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = 0)]
    pub n_gpu_layers: i32,
    #[arg(long)]
    pub selected_backend_device: Option<String>,
    #[arg(long, default_value_t = 8)]
    pub top_k: usize,
    #[arg(long, default_value_t = 1)]
    pub verify_steps: usize,
    #[arg(long, default_value_t = 0.05)]
    pub cur_in_tol: f32,
    #[arg(long, default_value_t = 0.25)]
    pub logits_tol: f32,
    #[arg(long)]
    pub output: Option<PathBuf>,
    #[arg(
        long,
        help = "Write product live-tap SPD training rows into this corpus directory. Emits rows.f32 plus rows.jsonl metadata."
    )]
    pub product_corpus_dir: Option<PathBuf>,
}

#[derive(Parser)]
pub struct SpdOpenAiCheckArgs {
    #[arg(long)]
    pub report: PathBuf,
    #[arg(long, default_value_t = 24)]
    pub min_accepted: u64,
    #[arg(long, default_value_t = 4)]
    pub expected_logical_stage_count: u64,
    #[arg(long, default_value_t = 4)]
    pub min_max_inflight: u64,
    #[arg(long, default_value_t = 0)]
    pub max_rejected_oldest: u64,
    #[arg(long, default_value_t = 0)]
    pub max_drained_younger: u64,
    #[arg(long, default_value_t = 0)]
    pub max_rolling_trace_missing_proposals: u64,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub require_content_match: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub require_rolling_executor: bool,
    #[arg(long)]
    pub max_spd_decode_ms: Option<f64>,
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Parser)]
pub struct SpdOpenAiSmokeArgs {
    #[arg(long, default_value = "target/release/skippy-server")]
    pub stage_server_bin: PathBuf,
    #[arg(long)]
    pub manifest: PathBuf,
    #[arg(long)]
    pub fixture: PathBuf,
    #[arg(long)]
    pub model_path: PathBuf,
    #[arg(long, default_value = "local/spd-openai-smoke")]
    pub model_id: String,
    #[arg(long, value_delimiter = ',')]
    pub splits: Vec<u32>,
    #[arg(long, default_value_t = 32)]
    pub layer_end: u32,
    #[arg(long, default_value_t = 128)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = 0)]
    pub n_gpu_layers: i32,
    #[arg(long)]
    pub selected_backend_device: Option<String>,
    #[arg(
        long,
        value_delimiter = ',',
        help = "Optional stage host placement for SPD OpenAI smoke. Use 'local' for the coordinator host; other values are passed to ssh/rsync. Hosts repeat cyclically across stages."
    )]
    pub stage_hosts: Vec<String>,
    #[arg(
        long,
        default_value_t = 20031,
        help = "First binary stage port used when --stage-hosts is set."
    )]
    pub stage_port_base: u16,
    #[arg(
        long,
        default_value = "0.0.0.0",
        help = "Bind host for remote binary stages when --stage-hosts is set."
    )]
    pub remote_bind_host: String,
    #[arg(
        long,
        default_value = "/tmp/skippy-spd-openai-smoke",
        help = "Remote root directory for copied configs/logs/binaries when --stage-hosts includes remote hosts."
    )]
    pub remote_root: String,
    #[arg(
        long,
        help = "Comma-separated HOST=ENDPOINT_HOST overrides for topology endpoints, e.g. local=192.168.1.10,worker=192.168.1.11."
    )]
    pub endpoint_host_map: Option<String>,
    #[arg(
        long,
        help = "Comma-separated HOST=PATH overrides for the GGUF file or layer-package directory on remote hosts."
    )]
    pub remote_model_path_map: Option<String>,
    #[arg(
        long,
        default_value_t = false,
        help = "Rsync the GGUF model to each remote host under --remote-root for this smoke run."
    )]
    pub rsync_model_artifacts: bool,
    #[arg(long, default_value_t = 2560)]
    pub activation_width: i32,
    #[arg(long, default_value = "f16")]
    pub activation_wire_dtype: String,
    #[arg(long, default_value_t = 8)]
    pub max_tokens: u32,
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,
    #[arg(
        long,
        default_value = "Write a Python function named add that returns the sum of two integers."
    )]
    pub prompt: String,
    #[arg(
        long,
        default_value_t = false,
        action = clap::ArgAction::Set,
        help = "Set chat_template_kwargs.enable_thinking on OpenAI smoke requests. Defaults false to match exported SPD parity fixtures."
    )]
    pub enable_thinking: bool,
    #[arg(
        long,
        help = "Optional prompt file. Reads non-empty plain-text lines, JSON strings, or JSON objects with prompt/text/content fields."
    )]
    pub prompt_file: Option<PathBuf>,
    #[arg(long)]
    pub prompt_limit: Option<usize>,
    #[arg(
        long,
        default_value_t = 1,
        help = "Measured case iterations per prompt. Each iteration launches isolated stage processes."
    )]
    pub repeat_count: usize,
    #[arg(
        long,
        default_value_t = 0,
        help = "Warmup case iterations per prompt. Warmups are included in cases with warmup=true but excluded from summary."
    )]
    pub warmup_count: usize,
    #[arg(long, default_value_t = 1)]
    pub openai_generation_concurrency: usize,
    #[arg(long, default_value_t = 1)]
    pub max_inflight: usize,
    #[arg(
        long,
        default_value_t = 0.0,
        help = "Artificial downstream write delay in milliseconds per binary stage message."
    )]
    pub downstream_wire_delay_ms: f64,
    #[arg(
        long,
        help = "Artificial downstream activation bandwidth cap in megabits per second."
    )]
    pub downstream_wire_mbps: Option<f64>,
    #[arg(long, default_value_t = 4)]
    pub speculative_window: usize,
    #[arg(long, default_value_t = 1)]
    pub spd_top_k: usize,
    #[arg(long, default_value_t = 0, allow_hyphen_values = true)]
    pub spd_n_gpu_layers: i32,
    #[arg(
        long,
        default_value_t = false,
        help = "Allow the SPD source to run slow local full-context tap replay when inline taps are incomplete."
    )]
    pub spd_replay_fallback: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub optimistic_decode: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Pass --openai-spd-rolling-executor to the SPD stage-0 server for native rolling-executor validation."
    )]
    pub spd_rolling_executor: bool,
    #[arg(
        long,
        help = "Only start optimistic SPD target decode when the inline top-1/top-2 logit margin is at least this value. Use --spd-top-k 2 or higher to produce margins."
    )]
    pub optimistic_min_logit_margin: Option<f32>,
    #[arg(
        long,
        default_value_t = true,
        action = clap::ArgAction::Set,
        help = "Derive downstream SPD tap-return allowlist from fixture rows. Disable to preserve legacy all-taps behavior."
    )]
    pub derive_tap_allowlist: bool,
    #[arg(long, value_delimiter = ',')]
    pub spd_tap_return_hf_indices: Vec<u32>,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub run_baseline: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub run_spd: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Validate artifacts, topology, tap coverage, and remote placement without starting stages."
    )]
    pub preflight_only: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Write the report but do not fail when paired baseline/SPD content differs."
    )]
    pub allow_content_mismatch: bool,
    #[arg(long, default_value_t = 120)]
    pub startup_timeout_secs: u64,
    #[arg(long, default_value_t = 180)]
    pub request_timeout_secs: u64,
    #[arg(long)]
    pub work_dir: Option<PathBuf>,
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
    #[arg(long)]
    pub selected_backend_device: Option<String>,
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
    #[arg(long)]
    pub selected_backend_device: Option<String>,
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
    #[arg(long)]
    pub selected_backend_device: Option<String>,
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
    #[arg(long, value_delimiter = ',')]
    pub splits: Vec<u32>,
    #[arg(long, default_value_t = 30)]
    pub layer_end: u32,
    #[arg(long, default_value_t = 128)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = 0)]
    pub n_gpu_layers: i32,
    #[arg(long)]
    pub selected_backend_device: Option<String>,
    #[arg(long, default_value = "Hello")]
    pub prompt: String,
    #[arg(long, default_value = "127.0.0.1:19031")]
    pub stage1_bind_addr: SocketAddr,
    #[arg(long, default_value = "127.0.0.1:19032")]
    pub stage2_bind_addr: SocketAddr,
    #[arg(long, default_value_t = 19031)]
    pub stage_bind_base_port: u16,
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
    fn parses_local_split_chain_splits() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "local-split-chain-binary",
            "--model-path",
            "model.gguf",
            "--splits",
            "8,10,16,20,24,31",
            "--layer-end",
            "32",
            "--stage-bind-base-port",
            "19131",
        ])
        .unwrap();

        let CommandKind::LocalSplitChainBinary(args) = cli.command else {
            panic!("expected local-split-chain-binary subcommand");
        };

        assert_eq!(args.splits, vec![8, 10, 16, 20, 24, 31]);
        assert_eq!(args.layer_end, 32);
        assert_eq!(args.stage_bind_base_port, 19131);
    }

    #[test]
    fn parses_spd_fixture_parity() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "spd-fixture-parity",
            "--manifest",
            "skippy-spd-head.json",
            "--fixture",
            "spd-parity-fixture.safetensors",
            "--top-k",
            "4",
        ])
        .unwrap();

        let CommandKind::SpdFixtureParity(args) = cli.command else {
            panic!("expected spd-fixture-parity subcommand");
        };

        assert_eq!(args.manifest, PathBuf::from("skippy-spd-head.json"));
        assert_eq!(
            args.fixture,
            PathBuf::from("spd-parity-fixture.safetensors")
        );
        assert_eq!(args.top_k, 4);
    }

    #[test]
    fn parses_spd_live_tap_parity() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "spd-live-tap-parity",
            "--manifest",
            "skippy-spd-head.json",
            "--fixture",
            "spd-parity-fixture.safetensors",
            "--model-path",
            "model.gguf",
            "--splits",
            "8,10,16,20,24,31",
            "--layer-end",
            "32",
            "--selected-backend-device",
            "CPU0",
            "--product-corpus-dir",
            "/tmp/spd-product-corpus",
        ])
        .unwrap();

        let CommandKind::SpdLiveTapParity(args) = cli.command else {
            panic!("expected spd-live-tap-parity subcommand");
        };

        assert_eq!(args.manifest, PathBuf::from("skippy-spd-head.json"));
        assert_eq!(
            args.fixture,
            PathBuf::from("spd-parity-fixture.safetensors")
        );
        assert_eq!(args.model_path, PathBuf::from("model.gguf"));
        assert_eq!(args.splits, vec![8, 10, 16, 20, 24, 31]);
        assert_eq!(args.layer_end, 32);
        assert_eq!(args.selected_backend_device.as_deref(), Some("CPU0"));
        assert_eq!(
            args.product_corpus_dir,
            Some(PathBuf::from("/tmp/spd-product-corpus"))
        );
        assert_eq!(args.verify_steps, 1);
        assert_eq!(args.cur_in_tol, 0.05);
        assert_eq!(args.logits_tol, 0.25);
    }

    #[test]
    fn parses_spd_openai_smoke_enable_thinking() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "spd-openai-smoke",
            "--manifest",
            "skippy-spd-head.json",
            "--fixture",
            "spd-parity-fixture.safetensors",
            "--model-path",
            "model.gguf",
            "--splits",
            "8,10,16,20,24,31",
            "--enable-thinking",
            "true",
        ])
        .unwrap();

        let CommandKind::SpdOpenAiSmoke(args) = cli.command else {
            panic!("expected spd-openai-smoke subcommand");
        };

        assert!(args.enable_thinking);

        let cli = Cli::try_parse_from([
            "skippy-bench",
            "spd-openai-smoke",
            "--manifest",
            "skippy-spd-head.json",
            "--fixture",
            "spd-parity-fixture.safetensors",
            "--model-path",
            "model.gguf",
            "--splits",
            "8,10,16,20,24,31",
        ])
        .unwrap();

        let CommandKind::SpdOpenAiSmoke(args) = cli.command else {
            panic!("expected spd-openai-smoke subcommand");
        };

        assert!(!args.enable_thinking);
    }

    #[test]
    fn parses_spd_openai_smoke_allow_content_mismatch() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "spd-openai-smoke",
            "--manifest",
            "skippy-spd-head.json",
            "--fixture",
            "spd-parity-fixture.safetensors",
            "--model-path",
            "model.gguf",
            "--splits",
            "8,10,16,20,24,31",
            "--allow-content-mismatch",
        ])
        .unwrap();

        let CommandKind::SpdOpenAiSmoke(args) = cli.command else {
            panic!("expected spd-openai-smoke subcommand");
        };

        assert!(args.allow_content_mismatch);

        let cli = Cli::try_parse_from([
            "skippy-bench",
            "spd-openai-smoke",
            "--manifest",
            "skippy-spd-head.json",
            "--fixture",
            "spd-parity-fixture.safetensors",
            "--model-path",
            "model.gguf",
            "--splits",
            "8,10,16,20,24,31",
        ])
        .unwrap();

        let CommandKind::SpdOpenAiSmoke(args) = cli.command else {
            panic!("expected spd-openai-smoke subcommand");
        };

        assert!(!args.allow_content_mismatch);
    }

    #[test]
    fn parses_spd_openai_smoke_spd_replay_fallback() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "spd-openai-smoke",
            "--manifest",
            "skippy-spd-head.json",
            "--fixture",
            "spd-parity-fixture.safetensors",
            "--model-path",
            "model.gguf",
            "--splits",
            "8,10,16,20,24,31",
            "--spd-replay-fallback",
        ])
        .unwrap();

        let CommandKind::SpdOpenAiSmoke(args) = cli.command else {
            panic!("expected spd-openai-smoke subcommand");
        };

        assert!(args.spd_replay_fallback);

        let cli = Cli::try_parse_from([
            "skippy-bench",
            "spd-openai-smoke",
            "--manifest",
            "skippy-spd-head.json",
            "--fixture",
            "spd-parity-fixture.safetensors",
            "--model-path",
            "model.gguf",
            "--splits",
            "8,10,16,20,24,31",
        ])
        .unwrap();

        let CommandKind::SpdOpenAiSmoke(args) = cli.command else {
            panic!("expected spd-openai-smoke subcommand");
        };

        assert!(!args.spd_replay_fallback);
    }

    #[test]
    fn parses_spd_openai_smoke_remote_stage_options() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "spd-openai-smoke",
            "--manifest",
            "skippy-spd-head.json",
            "--fixture",
            "spd-parity-fixture.safetensors",
            "--model-path",
            "model.gguf",
            "--splits",
            "8,10,16,20,24,31",
            "--stage-hosts",
            "local,worker",
            "--stage-port-base",
            "21031",
            "--endpoint-host-map",
            "local=host-a,worker=host-b",
            "--remote-model-path-map",
            "worker=/models/model.gguf",
            "--rsync-model-artifacts",
        ])
        .unwrap();

        let CommandKind::SpdOpenAiSmoke(args) = cli.command else {
            panic!("expected spd-openai-smoke subcommand");
        };

        assert_eq!(args.stage_hosts, ["local", "worker"]);
        assert_eq!(args.stage_port_base, 21031);
        assert_eq!(
            args.endpoint_host_map.as_deref(),
            Some("local=host-a,worker=host-b")
        );
        assert_eq!(
            args.remote_model_path_map.as_deref(),
            Some("worker=/models/model.gguf")
        );
        assert!(args.rsync_model_artifacts);
    }

    #[test]
    fn parses_spd_openai_smoke_preflight_only() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "spd-openai-smoke",
            "--manifest",
            "skippy-spd-head.json",
            "--fixture",
            "spd-parity-fixture.safetensors",
            "--model-path",
            "model.gguf",
            "--splits",
            "8,10,16,20,24,31",
            "--preflight-only",
        ])
        .unwrap();

        let CommandKind::SpdOpenAiSmoke(args) = cli.command else {
            panic!("expected spd-openai-smoke subcommand");
        };

        assert!(args.preflight_only);
    }

    #[test]
    fn parses_spd_openai_check() {
        let cli = Cli::try_parse_from([
            "skippy-bench",
            "spd-openai-check",
            "--report",
            "report.json",
            "--min-accepted",
            "24",
            "--expected-logical-stage-count",
            "4",
            "--min-max-inflight",
            "4",
            "--max-rejected-oldest",
            "1",
            "--max-drained-younger",
            "3",
            "--max-rolling-trace-missing-proposals",
            "9",
            "--max-spd-decode-ms",
            "1500",
        ])
        .unwrap();

        let CommandKind::SpdOpenAiCheck(args) = cli.command else {
            panic!("expected spd-openai-check subcommand");
        };

        assert_eq!(args.report, PathBuf::from("report.json"));
        assert_eq!(args.min_accepted, 24);
        assert_eq!(args.expected_logical_stage_count, 4);
        assert_eq!(args.min_max_inflight, 4);
        assert_eq!(args.max_rejected_oldest, 1);
        assert_eq!(args.max_drained_younger, 3);
        assert_eq!(args.max_rolling_trace_missing_proposals, 9);
        assert_eq!(args.max_spd_decode_ms, Some(1500.0));
        assert!(args.require_content_match);
        assert!(args.require_rolling_executor);
    }
}
