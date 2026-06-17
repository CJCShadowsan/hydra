use std::{net::SocketAddr, path::PathBuf};

use crate::telemetry::TelemetryLevel;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(about = "Llama staged-runtime server")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    Serve(ServeArgs),
    ServeBinary(ServeBinaryArgs),
    #[command(name = "serve-openai")]
    ServeOpenAi(ServeOpenAiArgs),
    ExampleConfig,
}

#[derive(Parser)]
pub struct ServeArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub topology: Option<PathBuf>,
    #[arg(long)]
    pub bind_addr: Option<SocketAddr>,
    #[arg(long)]
    pub metrics_otlp_grpc: Option<String>,
    #[arg(long, default_value_t = 1024)]
    pub telemetry_queue_capacity: usize,
    #[arg(long, value_enum, default_value_t = TelemetryLevel::Summary)]
    pub telemetry_level: TelemetryLevel,
}

#[derive(Parser)]
pub struct ServeBinaryArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub topology: Option<PathBuf>,
    #[arg(long)]
    pub bind_addr: Option<SocketAddr>,
    #[arg(long)]
    pub activation_width: i32,
    #[arg(long, default_value = "f16")]
    pub activation_wire_dtype: String,
    #[arg(long)]
    pub metrics_otlp_grpc: Option<String>,
    #[arg(long, default_value_t = 1024)]
    pub telemetry_queue_capacity: usize,
    #[arg(long, value_enum, default_value_t = TelemetryLevel::Summary)]
    pub telemetry_level: TelemetryLevel,
    #[arg(long, default_value_t = 4)]
    pub max_inflight: usize,
    #[arg(long)]
    pub reply_credit_limit: Option<usize>,
    #[arg(
        long,
        help = "Forward eligible non-final prefill activation frames on a bounded background writer. Enabled by default."
    )]
    pub async_prefill_forward: bool,
    #[arg(
        long,
        help = "Disable async forwarding for eligible non-final prefill activation frames."
    )]
    pub no_async_prefill_forward: bool,
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
    #[arg(long, default_value_t = 60)]
    pub downstream_connect_timeout_secs: u64,
    #[arg(
        long,
        help = "Also serve the OpenAI-compatible HTTP surface from this stage process. Intended for stage 0."
    )]
    pub openai_bind_addr: Option<SocketAddr>,
    #[arg(
        long,
        help = "Served OpenAI model id. Defaults to the stage config model_id."
    )]
    pub openai_model_id: Option<String>,
    #[arg(long, default_value_t = 16)]
    pub openai_default_max_tokens: u32,
    #[arg(
        long,
        default_value_t = 1,
        help = "Maximum number of concurrent OpenAI chat generation requests hosted by this stage."
    )]
    pub openai_generation_concurrency: usize,
    #[arg(long, default_value_t = 256)]
    pub openai_prefill_chunk_size: usize,
    #[arg(
        long,
        default_value = "adaptive-ramp",
        help = "OpenAI prefill chunk policy: fixed, schedule, or adaptive-ramp. Passing --openai-prefill-chunk-schedule keeps legacy schedule behavior."
    )]
    pub openai_prefill_chunk_policy: String,
    #[arg(
        long,
        help = "Comma-separated OpenAI prefill chunk schedule. Example: 128,256,512 sends the first chunk at 128 tokens, second at 256, and repeats 512 after that."
    )]
    pub openai_prefill_chunk_schedule: Option<String>,
    #[arg(long, default_value_t = 128)]
    pub openai_prefill_adaptive_start: usize,
    #[arg(long, default_value_t = 128)]
    pub openai_prefill_adaptive_step: usize,
    #[arg(long, default_value_t = 384)]
    pub openai_prefill_adaptive_max: usize,
    #[arg(
        long,
        help = "Draft GGUF to use for speculative decoding in the embedded stage-0 OpenAI surface."
    )]
    pub openai_draft_model_path: Option<PathBuf>,
    #[arg(
        long,
        help = "Experimental SPD head manifest to use as the embedded stage-0 speculative proposal source."
    )]
    pub openai_spd_manifest: Option<PathBuf>,
    #[arg(
        long,
        help = "Experimental SPD parity fixture exported for the same head. Used for row metadata and final norm weights."
    )]
    pub openai_spd_fixture: Option<PathBuf>,
    #[arg(
        long,
        help = "Full GGUF to replay live SPD taps from. Defaults to source_model_path, then model_path, from the stage config."
    )]
    pub openai_spd_model_path: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    pub openai_spd_top_k: usize,
    #[arg(
        long,
        help = "Override n_gpu_layers for experimental SPD replay tap models. Defaults to the stage config n_gpu_layers."
    )]
    pub openai_spd_n_gpu_layers: Option<i32>,
    #[arg(
        long,
        help = "Allow the experimental SPD source to run slow local full-context tap replay when inline taps are incomplete."
    )]
    pub openai_spd_replay_fallback: bool,
    #[arg(
        long,
        help = "Experimentally start one target decode from an inline SPD proposal before the current target reply arrives. Only used for deterministic sampling."
    )]
    pub openai_spd_optimistic_decode: bool,
    #[arg(
        long,
        help = "Only start optimistic SPD target decode when the inline top-1/top-2 logit margin is at least this value. Requires --openai-spd-top-k >= 2 to produce margins."
    )]
    pub openai_spd_optimistic_min_logit_margin: Option<f32>,
    #[arg(long, default_value_t = 4)]
    pub openai_speculative_window: usize,
    #[arg(long)]
    pub openai_adaptive_speculative_window: bool,
    #[arg(
        long,
        help = "Override n_gpu_layers for the embedded OpenAI draft model. Defaults to the stage config n_gpu_layers."
    )]
    pub openai_draft_n_gpu_layers: Option<i32>,
}

#[derive(Parser)]
pub struct ServeOpenAiArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub topology: Option<PathBuf>,
    #[arg(long, default_value = "127.0.0.1:9337")]
    pub bind_addr: SocketAddr,
    #[arg(
        long,
        help = "Served model id to advertise and accept, for example org/repo:Q4_K_M. Defaults to config model_id."
    )]
    pub model_id: Option<String>,
    #[arg(long, default_value_t = 16)]
    pub default_max_tokens: u32,
    #[arg(
        long,
        default_value_t = 1,
        help = "Maximum number of concurrent chat generation requests."
    )]
    pub generation_concurrency: usize,
    #[arg(
        long,
        help = "Deprecated and unsupported. Direct prediction return requires embedded stage-0 OpenAI serving via serve-binary --openai-bind-addr."
    )]
    pub first_stage_addr: Option<String>,
    #[arg(long, default_value_t = 256)]
    pub prefill_chunk_size: usize,
    #[arg(
        long,
        default_value = "adaptive-ramp",
        help = "Prefill chunk policy for split OpenAI serving: fixed, schedule, or adaptive-ramp. Passing --prefill-chunk-schedule keeps legacy schedule behavior."
    )]
    pub prefill_chunk_policy: String,
    #[arg(
        long,
        help = "Comma-separated prefill chunk schedule for split OpenAI serving. Example: 128,256,512 sends the first chunk at 128 tokens, second at 256, and repeats 512 after that."
    )]
    pub prefill_chunk_schedule: Option<String>,
    #[arg(long, default_value_t = 128)]
    pub prefill_adaptive_start: usize,
    #[arg(long, default_value_t = 128)]
    pub prefill_adaptive_step: usize,
    #[arg(long, default_value_t = 384)]
    pub prefill_adaptive_max: usize,
    #[arg(long, default_value = "f32")]
    pub activation_wire_dtype: String,
    #[arg(long, default_value_t = 60)]
    pub startup_timeout_secs: u64,
    #[arg(long)]
    pub metrics_otlp_grpc: Option<String>,
    #[arg(long, default_value_t = 1024)]
    pub telemetry_queue_capacity: usize,
    #[arg(long, value_enum, default_value_t = TelemetryLevel::Summary)]
    pub telemetry_level: TelemetryLevel,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_prefill_policy_defaults_to_adaptive_ramp() {
        let cli = Cli::try_parse_from([
            "skippy-server",
            "serve-binary",
            "--config",
            "stage.json",
            "--activation-width",
            "2048",
        ])
        .unwrap();

        let Command::ServeBinary(args) = cli.command else {
            panic!("expected serve-binary command");
        };
        assert_eq!(args.openai_prefill_chunk_policy, "adaptive-ramp");
        assert_eq!(args.openai_prefill_adaptive_start, 128);
        assert_eq!(args.openai_prefill_adaptive_step, 128);
        assert_eq!(args.openai_prefill_adaptive_max, 384);

        let cli = Cli::try_parse_from(["skippy-server", "serve-openai", "--config", "stage.json"])
            .unwrap();

        let Command::ServeOpenAi(args) = cli.command else {
            panic!("expected serve-openai command");
        };
        assert_eq!(args.prefill_chunk_policy, "adaptive-ramp");
        assert_eq!(args.prefill_adaptive_start, 128);
        assert_eq!(args.prefill_adaptive_step, 128);
        assert_eq!(args.prefill_adaptive_max, 384);
    }

    #[test]
    fn serve_binary_parses_experimental_spd_options() {
        let cli = Cli::try_parse_from([
            "skippy-server",
            "serve-binary",
            "--config",
            "stage.json",
            "--topology",
            "topology.json",
            "--activation-width",
            "2560",
            "--openai-bind-addr",
            "127.0.0.1:9337",
            "--openai-spd-manifest",
            "skippy-spd-head.json",
            "--openai-spd-fixture",
            "spd-parity-fixture.safetensors",
            "--openai-spd-model-path",
            "model.gguf",
            "--openai-spd-top-k",
            "4",
            "--openai-spd-n-gpu-layers",
            "99",
            "--openai-spd-replay-fallback",
            "--openai-spd-optimistic-decode",
            "--openai-spd-optimistic-min-logit-margin",
            "5.5",
            "--openai-speculative-window",
            "2",
        ])
        .unwrap();

        let Command::ServeBinary(args) = cli.command else {
            panic!("expected serve-binary command");
        };

        assert_eq!(
            args.openai_spd_manifest,
            Some(PathBuf::from("skippy-spd-head.json"))
        );
        assert_eq!(
            args.openai_spd_fixture,
            Some(PathBuf::from("spd-parity-fixture.safetensors"))
        );
        assert_eq!(
            args.openai_spd_model_path,
            Some(PathBuf::from("model.gguf"))
        );
        assert_eq!(args.openai_spd_top_k, 4);
        assert_eq!(args.openai_spd_n_gpu_layers, Some(99));
        assert!(args.openai_spd_replay_fallback);
        assert!(args.openai_spd_optimistic_decode);
        assert_eq!(args.openai_spd_optimistic_min_logit_margin, Some(5.5));
        assert_eq!(args.openai_speculative_window, 2);
    }
}
