use std::{
    fs::{self, File},
    io::{ErrorKind, Write},
    net::{SocketAddr, TcpListener},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::json;
use skippy_protocol::binary::{
    StageReplyStats, WireMessageKind, read_stage_message, send_ready, send_reply_ack_with_stats,
    send_reply_predicted_tokens_with_stats, send_reply_predicted_with_stats,
};

use crate::{
    cli::{FlashAttentionArg, GlmDsaStage0TraceArgs, StageLoadMode},
    report::{GlmDsaStage0TraceReport, GlmDsaTimingReport, GlmDsaTraceVariantReport},
    support::{ChildGuard, connect_ready, parse_wire_dtype},
};

#[derive(Debug, Clone)]
struct FakeDownstreamMessage {
    kind: WireMessageKind,
    token_count: i32,
    top_k_count: usize,
}

struct FakeDownstreamGuard {
    stop: Arc<AtomicBool>,
    messages: Arc<Mutex<Vec<FakeDownstreamMessage>>>,
    handle: Option<JoinHandle<Result<()>>>,
}

#[derive(Clone, Copy)]
struct TraceVariant {
    name: &'static str,
    direct_sparse_attn: bool,
    fused_sparse_mask: bool,
}

pub fn glm_dsa_stage0_trace(args: GlmDsaStage0TraceArgs) -> Result<()> {
    ensure_supported_args(&args)?;
    let run_id = generate_glm_dsa_run_id();
    let case_root = args
        .case_root
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join(&run_id));
    fs::create_dir_all(&case_root)
        .with_context(|| format!("create case root {}", case_root.display()))?;

    let fused = run_variant(
        &args,
        &run_id,
        &case_root,
        TraceVariant {
            name: "fused",
            direct_sparse_attn: false,
            fused_sparse_mask: true,
        },
    )?;
    let direct = run_variant(
        &args,
        &run_id,
        &case_root,
        TraceVariant {
            name: "direct",
            direct_sparse_attn: true,
            fused_sparse_mask: false,
        },
    )?;

    let both_variants_completed = fused.prompt_success && direct.prompt_success;
    let fused_prefill_speedup_vs_direct =
        speedup(fused.prompt_prefill_tok_s, direct.prompt_prefill_tok_s);
    let fused_glm_dsa_op_speedup_vs_direct = timing_speedup(
        fused.avg_128_token_timing.as_ref(),
        direct.avg_128_token_timing.as_ref(),
    );
    let report = GlmDsaStage0TraceReport {
        mode: "glm-dsa-stage0-trace",
        status: if both_variants_completed {
            "pass"
        } else {
            "fail"
        },
        run_id,
        model_id: args
            .runtime
            .model_id
            .clone()
            .unwrap_or_else(|| "local/glm-dsa-stage0-trace".to_string()),
        model_path: stage_model_path(&args)?.to_string_lossy().into_owned(),
        case_root: case_root.to_string_lossy().into_owned(),
        stage_layer_end: args.stage_layer_end,
        activation_width: args.activation_width,
        activation_wire_dtype: args.activation_wire_dtype,
        prefill_chunk_size: args.prefill_chunk_size,
        max_new_tokens: args.max_new_tokens,
        trace_filter: args.trace_filter,
        both_variants_completed,
        fused_prefill_speedup_vs_direct,
        fused_glm_dsa_op_speedup_vs_direct,
        variants: vec![fused, direct],
    };
    emit_report(&report, args.output.report_out.as_deref())?;
    if !report.both_variants_completed {
        bail!("GLM-DSA stage0 trace did not complete both variants");
    }
    Ok(())
}

fn ensure_supported_args(args: &GlmDsaStage0TraceArgs) -> Result<()> {
    if args.runtime.stage_load_mode != StageLoadMode::LayerPackage {
        bail!("glm-dsa-stage0-trace currently requires --stage-load-mode layer-package");
    }
    if args.stage_layer_end == 0 {
        bail!("--stage-layer-end must be greater than zero");
    }
    parse_wire_dtype(&args.activation_wire_dtype)?;
    Ok(())
}

fn run_variant(
    args: &GlmDsaStage0TraceArgs,
    run_id: &str,
    case_root: &Path,
    variant: TraceVariant,
) -> Result<GlmDsaTraceVariantReport> {
    let variant_root = case_root.join(variant.name);
    fs::create_dir_all(&variant_root)
        .with_context(|| format!("create variant root {}", variant_root.display()))?;
    let stage_config_path = variant_root.join("stage0.json");
    let stage_log_path = variant_root.join("stage0.log");
    let prompt_log_path = variant_root.join("prompt.log");
    write_stage_config(args, run_id, variant.name, &stage_config_path)?;

    let fake = FakeDownstreamGuard::start(args.fake_downstream_bind_addr, args.activation_width)
        .context("start fake downstream")?;
    let _stage =
        start_stage0(args, &stage_config_path, &stage_log_path, variant).context("start stage0")?;
    drop(
        connect_ready(args.stage0_bind_addr, args.server.startup_timeout_secs)
            .context("stage0 did not become ready")?,
    );

    let prompt_output = run_prompt(args, &prompt_log_path).context("run skippy-prompt")?;
    let fake_messages = fake.finish()?;
    let stage_log = fs::read_to_string(&stage_log_path).unwrap_or_default();
    let prompt_log = fs::read_to_string(&prompt_log_path).unwrap_or_default();
    let avg_128_token_timing = avg_128_token_timing(&stage_log);
    let (prompt_prefill_tok_s, prompt_decode_tok_s) = parse_prompt_speeds(&prompt_log);
    let trace_line_count = stage_log.matches("glm_dsa_tensor_trace").count();
    let timing_line_count = stage_log.matches("glm_dsa_op_timing").count();
    let fake_downstream_top_k_message_count = fake_messages
        .iter()
        .filter(|message| message.top_k_count > 0)
        .count();
    let fake_downstream_prefill_message_count = fake_messages
        .iter()
        .filter(|message| message.kind.is_prefill())
        .count();
    let fake_downstream_decode_message_count = fake_messages
        .iter()
        .filter(|message| message.kind == WireMessageKind::DecodeEmbd)
        .count();
    let fake_downstream_prefill_token_count = fake_messages
        .iter()
        .filter(|message| message.kind.is_prefill())
        .map(|message| usize::try_from(message.token_count.max(0)).unwrap_or(0))
        .sum();
    let fake_downstream_max_top_k_count = fake_messages
        .iter()
        .map(|message| message.top_k_count)
        .max()
        .unwrap_or(0);

    Ok(GlmDsaTraceVariantReport {
        variant: variant.name,
        direct_sparse_attn: variant.direct_sparse_attn,
        fused_sparse_mask: variant.fused_sparse_mask,
        prompt_exit_code: prompt_output.status.code(),
        prompt_success: prompt_output.status.success(),
        stage_log: stage_log_path.to_string_lossy().into_owned(),
        prompt_log: prompt_log_path.to_string_lossy().into_owned(),
        fake_downstream_message_count: fake_messages.len(),
        fake_downstream_prefill_message_count,
        fake_downstream_decode_message_count,
        fake_downstream_prefill_token_count,
        fake_downstream_top_k_message_count,
        fake_downstream_max_top_k_count,
        trace_line_count,
        timing_line_count,
        prompt_prefill_tok_s,
        prompt_decode_tok_s,
        avg_128_token_timing,
    })
}

fn write_stage_config(
    args: &GlmDsaStage0TraceArgs,
    run_id: &str,
    variant: &str,
    path: &Path,
) -> Result<()> {
    let model_path = stage_model_path(args)?;
    let model_id = args
        .runtime
        .model_id
        .clone()
        .unwrap_or_else(|| "local/glm-dsa-stage0-trace".to_string());
    let config = json!({
        "run_id": run_id,
        "topology_id": format!("glm-dsa-stage0-trace-{variant}"),
        "model_id": model_id,
        "model_path": model_path,
        "stage_id": "stage-0",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": args.stage_layer_end,
        "ctx_size": args.runtime.ctx_size,
        "n_batch": args.runtime.n_batch,
        "n_ubatch": args.runtime.n_ubatch,
        "n_gpu_layers": args.runtime.n_gpu_layers,
        "flash_attn_type": protocol_flash_attn(args.runtime.flash_attn),
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "filter_tensors_on_load": true,
        "use_mmap": true,
        "load_mode": "layer-package",
        "bind_addr": args.stage0_bind_addr,
        "upstream": null,
        "downstream": {
            "stage_id": "fake-stage-1",
            "stage_index": 1,
            "endpoint": format!("tcp://{}", args.fake_downstream_bind_addr),
        },
    });
    fs::write(path, serde_json::to_vec_pretty(&config)?)
        .with_context(|| format!("write stage config {}", path.display()))
}

fn start_stage0(
    args: &GlmDsaStage0TraceArgs,
    config_path: &Path,
    log_path: &Path,
    variant: TraceVariant,
) -> Result<ChildGuard> {
    let log = File::create(log_path).with_context(|| format!("create {}", log_path.display()))?;
    let err_log = log
        .try_clone()
        .with_context(|| format!("clone {}", log_path.display()))?;
    let mut command = Command::new(&args.server.stage_server_bin);
    command
        .args([
            "serve-binary",
            "--config",
            path_str(config_path)?,
            "--activation-width",
            &args.activation_width.to_string(),
            "--activation-wire-dtype",
            &args.activation_wire_dtype,
            "--max-inflight",
            &args.server.max_inflight.to_string(),
        ])
        .env("SKIPPY_GLM_DSA_OP_TIMING", "1")
        .env("SKIPPY_GLM_DSA_TENSOR_TRACE", "1")
        .env("SKIPPY_GLM_DSA_TENSOR_TRACE_STATS", "1")
        .env(
            "SKIPPY_GLM_DSA_TENSOR_TRACE_VALUES",
            args.trace_values.to_string(),
        )
        .env(
            "SKIPPY_GLM_DSA_TENSOR_TRACE_NODES",
            args.trace_nodes.to_string(),
        )
        .env("SKIPPY_GLM_DSA_TENSOR_TRACE_FILTER", &args.trace_filter)
        .env(
            "SKIPPY_GLM_DSA_TENSOR_TRACE_STATS_MAX_BYTES",
            args.trace_stats_max_bytes.to_string(),
        )
        .env(
            "SKIPPY_GLM_DSA_ENABLE_DIRECT_SPARSE_ATTN",
            if variant.direct_sparse_attn { "1" } else { "0" },
        )
        .env(
            "SKIPPY_GLM_DSA_ENABLE_FUSED_SPARSE_MASK",
            if variant.fused_sparse_mask { "1" } else { "0" },
        )
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(err_log));
    ChildGuard::spawn(command)
}

fn run_prompt(
    args: &GlmDsaStage0TraceArgs,
    prompt_log_path: &Path,
) -> Result<std::process::Output> {
    let model_path = stage_model_path(args)?;
    let mut child = Command::new(&args.prompt_bin)
        .args([
            "binary",
            "--model-path",
            path_str(&model_path)?,
            "--tokenizer-load-mode",
            "layer-package",
            "--tokenizer-layer-start",
            "0",
            "--tokenizer-layer-end",
            "1",
            "--first-stage-addr",
            &args.stage0_bind_addr.to_string(),
            "--ctx-size",
            &args.runtime.ctx_size.to_string(),
            "--n-gpu-layers",
            "0",
            "--activation-width",
            &args.activation_width.to_string(),
            "--activation-wire-dtype",
            &args.activation_wire_dtype,
            "--prefill-chunk-size",
            &args.prefill_chunk_size.to_string(),
            "--max-new-tokens",
            &args.max_new_tokens.to_string(),
            "--decode-timeout-secs",
            &args.server.startup_timeout_secs.to_string(),
            "--trace-token-ids",
            "--no-think",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawn {}", args.prompt_bin.display()))?;
    {
        let stdin = child.stdin.as_mut().context("open skippy-prompt stdin")?;
        writeln!(stdin, "{}", args.runtime.prompt)?;
        writeln!(stdin, ":quit")?;
    }
    let output = child.wait_with_output().context("wait for skippy-prompt")?;
    let mut log = Vec::new();
    log.extend_from_slice(&output.stdout);
    log.extend_from_slice(&output.stderr);
    fs::write(prompt_log_path, log)
        .with_context(|| format!("write prompt log {}", prompt_log_path.display()))?;
    Ok(output)
}

impl FakeDownstreamGuard {
    fn start(addr: SocketAddr, activation_width: i32) -> Result<Self> {
        let listener = TcpListener::bind(addr).with_context(|| format!("bind fake {addr}"))?;
        listener
            .set_nonblocking(true)
            .with_context(|| format!("set fake {addr} nonblocking"))?;
        let stop = Arc::new(AtomicBool::new(false));
        let messages = Arc::new(Mutex::new(Vec::new()));
        let (ready_tx, ready_rx) = mpsc::channel();
        let thread_stop = Arc::clone(&stop);
        let thread_messages = Arc::clone(&messages);
        let handle = thread::spawn(move || -> Result<()> {
            ready_tx.send(()).ok();
            while !thread_stop.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream
                            .set_nonblocking(false)
                            .context("set fake downstream stream blocking")?;
                        send_ready(&mut stream).context("send fake downstream ready")?;
                        loop {
                            let message = match read_stage_message(&mut stream, activation_width) {
                                Ok(message) => message,
                                Err(error) if error.kind() == ErrorKind::UnexpectedEof => break,
                                Err(error) => {
                                    return Err(anyhow!(error).context("read stage message"));
                                }
                            };
                            let summary = FakeDownstreamMessage {
                                kind: message.kind,
                                token_count: message.token_count,
                                top_k_count: message.raw_bytes.len() / std::mem::size_of::<i32>(),
                            };
                            thread_messages
                                .lock()
                                .expect("fake downstream messages lock poisoned")
                                .push(summary.clone());
                            match message.kind {
                                WireMessageKind::Stop => {
                                    send_reply_ack_with_stats(
                                        &mut stream,
                                        StageReplyStats::default(),
                                    )?;
                                    return Ok(());
                                }
                                WireMessageKind::VerifySpan => {
                                    let count = usize::try_from(message.token_count.max(1))
                                        .unwrap_or(1)
                                        .min(8);
                                    let tokens = vec![2; count];
                                    send_reply_predicted_tokens_with_stats(
                                        &mut stream,
                                        &tokens,
                                        StageReplyStats::default(),
                                    )?;
                                }
                                kind if kind.requires_predicted_reply() => {
                                    send_reply_predicted_with_stats(
                                        &mut stream,
                                        2,
                                        StageReplyStats::default(),
                                    )?;
                                }
                                _ => {
                                    send_reply_ack_with_stats(
                                        &mut stream,
                                        StageReplyStats::default(),
                                    )?;
                                }
                            }
                        }
                    }
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(20));
                    }
                    Err(error) => return Err(anyhow!(error).context("accept fake downstream")),
                }
            }
            Ok(())
        });
        ready_rx
            .recv_timeout(Duration::from_secs(2))
            .context("fake downstream thread did not start")?;
        Ok(Self {
            stop,
            messages,
            handle: Some(handle),
        })
    }

    fn finish(mut self) -> Result<Vec<FakeDownstreamMessage>> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| anyhow!("fake downstream thread panicked"))??;
        }
        Ok(self
            .messages
            .lock()
            .expect("fake downstream messages lock poisoned")
            .clone())
    }
}

impl Drop for FakeDownstreamGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn avg_128_token_timing(stage_log: &str) -> Option<GlmDsaTimingReport> {
    let mut count = 0usize;
    let mut total_us = 0.0;
    let mut indexer_topk_us = 0.0;
    let mut sparse_mask_us = 0.0;
    let mut dsa_sparse_attn_us = 0.0;
    let mut mla_attention_us = 0.0;
    for line in stage_log
        .lines()
        .filter(|line| line.contains("glm_dsa_op_timing"))
    {
        if timing_value(line, "tokens")? != 128.0 {
            continue;
        }
        count += 1;
        total_us += timing_value(line, "total_us")?;
        indexer_topk_us += timing_value(line, "indexer_topk_us")?;
        sparse_mask_us += timing_value(line, "sparse_mask_us")?;
        dsa_sparse_attn_us += timing_value(line, "dsa_sparse_attn_us").unwrap_or(0.0);
        mla_attention_us += timing_value(line, "mla_attention_us")?;
    }
    if count == 0 {
        return None;
    }
    let count_f = count as f64;
    Some(GlmDsaTimingReport {
        chunk_count: count,
        total_us: total_us / count_f,
        indexer_topk_us: indexer_topk_us / count_f,
        sparse_mask_us: sparse_mask_us / count_f,
        dsa_sparse_attn_us: dsa_sparse_attn_us / count_f,
        mla_attention_us: mla_attention_us / count_f,
    })
}

fn timing_value(line: &str, key: &str) -> Option<f64> {
    line.split_whitespace().find_map(|field| {
        let (field_key, value) = field.split_once('=')?;
        (field_key == key)
            .then(|| value.parse::<f64>().ok())
            .flatten()
    })
}

fn parse_prompt_speeds(prompt_log: &str) -> (Option<f64>, Option<f64>) {
    let mut prefill = None;
    let mut decode = None;
    for line in prompt_log.lines() {
        if !line.trim_start().starts_with("speed") {
            continue;
        }
        prefill = speed_value(line, "prefill");
        decode = speed_value(line, "decode");
    }
    (prefill, decode)
}

fn speed_value(line: &str, key: &str) -> Option<f64> {
    line.split_whitespace().find_map(|field| {
        let (field_key, value) = field.split_once('=')?;
        (field_key == key)
            .then(|| value.parse::<f64>().ok())
            .flatten()
    })
}

fn speedup(fused: Option<f64>, direct: Option<f64>) -> Option<f64> {
    let fused = fused?;
    let direct = direct?;
    (direct > 0.0).then_some(fused / direct)
}

fn timing_speedup(
    fused: Option<&GlmDsaTimingReport>,
    direct: Option<&GlmDsaTimingReport>,
) -> Option<f64> {
    let fused = fused?;
    let direct = direct?;
    (fused.total_us > 0.0).then_some(direct.total_us / fused.total_us)
}

fn stage_model_path(args: &GlmDsaStage0TraceArgs) -> Result<PathBuf> {
    Ok(args
        .runtime
        .stage_model
        .clone()
        .unwrap_or_else(|| args.runtime.model.clone()))
}

fn protocol_flash_attn(value: FlashAttentionArg) -> &'static str {
    match value {
        FlashAttentionArg::Auto => "auto",
        FlashAttentionArg::Disabled => "disabled",
        FlashAttentionArg::Enabled => "enabled",
    }
}

fn emit_report<T: serde::Serialize>(report: &T, report_out: Option<&Path>) -> Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    println!("{json}");
    if let Some(path) = report_out {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("create report directory {}", parent.display()))?;
        }
        fs::write(path, format!("{json}\n"))
            .with_context(|| format!("write correctness report {}", path.display()))?;
    }
    Ok(())
}

fn generate_glm_dsa_run_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_millis();
    format!("glm-dsa-stage0-trace-{millis}")
}

fn path_str(path: &Path) -> Result<&str> {
    path.to_str()
        .with_context(|| format!("path is not valid UTF-8: {}", path.display()))
}
