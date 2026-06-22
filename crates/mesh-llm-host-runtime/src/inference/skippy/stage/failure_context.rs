use skippy_runtime::NativeLogTail;

use super::StageLoadRequest;

const NATIVE_LOG_TAIL_LINES: usize = 32;
const NATIVE_LOG_TAIL_LINE_CHARS: usize = 500;

pub(super) fn stage_load_failure_context(
    load: &StageLoadRequest,
    error: &str,
    last_error: Option<&str>,
) -> String {
    let native_tail =
        skippy_runtime::native_log_tail(NATIVE_LOG_TAIL_LINES, NATIVE_LOG_TAIL_LINE_CHARS)
            .ok()
            .flatten();
    format_stage_load_failure_context(load, error, last_error, native_tail.as_ref())
}

pub(super) fn format_stage_load_failure_context(
    load: &StageLoadRequest,
    error: &str,
    last_error: Option<&str>,
    native_tail: Option<&NativeLogTail>,
) -> String {
    let mut context = base_stage_load_failure_context(load, error, last_error);
    append_native_log_tail(&mut context, native_tail);
    context
}

fn base_stage_load_failure_context(
    load: &StageLoadRequest,
    error: &str,
    last_error: Option<&str>,
) -> String {
    let source_bytes = load
        .source_model_bytes
        .map(|bytes| bytes.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let device = load
        .selected_device
        .as_ref()
        .map(|device| device.backend_device.as_str())
        .unwrap_or("auto");
    format!(
        "split stage load failed: model={} topology={} run={} stage={} index={} layers={}..{} mode={:?} bind={} ctx={} lanes={} source_bytes={} device={} error={} last_error={}",
        load.model_id,
        load.topology_id,
        load.run_id,
        load.stage_id,
        load.stage_index,
        load.layer_start,
        load.layer_end,
        load.load_mode,
        load.bind_addr,
        load.ctx_size,
        load.lane_count,
        source_bytes,
        device,
        error,
        last_error.unwrap_or("none"),
    )
}

fn append_native_log_tail(context: &mut String, native_tail: Option<&NativeLogTail>) {
    let Some(native_tail) = native_tail else {
        return;
    };
    context.push_str(" native_log_path=");
    context.push_str(&sanitize_context_value(&native_tail.path.to_string_lossy()));
    if native_tail.lines.is_empty() {
        return;
    }
    context.push_str(" native_log_tail=");
    context.push_str(&sanitize_context_value(&native_tail.lines.join(" | ")));
}

fn sanitize_context_value(value: &str) -> String {
    value
        .chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect()
}
