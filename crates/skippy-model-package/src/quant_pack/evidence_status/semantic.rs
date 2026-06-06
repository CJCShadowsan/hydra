use std::fs;

use super::{
    EvidenceCommandInput, EvidenceOutputStatus, EvidencePlanRuntime, EvidencePlanTopology,
};

pub(super) struct EvidenceSemanticContext<'a> {
    pub(super) topology: &'a EvidencePlanTopology,
    pub(super) runtime: &'a EvidencePlanRuntime,
}

pub(super) fn completed_command_failure(
    command: &EvidenceCommandInput,
    outputs: &[EvidenceOutputStatus],
    context: &EvidenceSemanticContext<'_>,
) -> Option<String> {
    match command_semantic_kind(command) {
        "certify" | "skippy-quant-pack-certification" => certification_output_failure(outputs),
        "chat-corpus"
        | "skippy-bench-chat-corpus"
        | "long-context-chat-corpus"
        | "skippy-bench-long-context-chat-corpus" => chat_corpus_output_failure(outputs),
        "focused-runtime" | "skippy-bench-focused-runtime" => {
            focused_runtime_output_failure(outputs, FocusedRuntimeMode::Executed, context)
        }
        "focused-runtime-schema-smoke" | "skippy-bench-focused-runtime-schema-smoke" => {
            focused_runtime_output_failure(outputs, FocusedRuntimeMode::SchemaSmoke, context)
        }
        "local-split-chain" | "skippy-bench-local-split-chain" => {
            local_split_chain_output_failure(outputs, context)
        }
        "rank-after-evidence" | "rank-after-evidence-all" | "skippy-quant-pack-rank" => {
            rank_output_failure(outputs)
        }
        "token-lengths" | "skippy-bench-token-lengths" => {
            token_lengths_output_failure(outputs, context)
        }
        _ => None,
    }
}

pub(super) fn command_semantic_kind(command: &EvidenceCommandInput) -> &str {
    command
        .evidence_type
        .as_deref()
        .filter(|evidence_type| !evidence_type.trim().is_empty())
        .unwrap_or(&command.id)
}

fn local_split_chain_output_failure(
    outputs: &[EvidenceOutputStatus],
    context: &EvidenceSemanticContext<'_>,
) -> Option<String> {
    let output = outputs.iter().find(|output| output.exists)?;
    let value = match read_json_output(output) {
        Ok(value) => value,
        Err(failure) => return Some(failure),
    };
    if value.get("mode").and_then(serde_json::Value::as_str) != Some("local-split-chain-binary") {
        return Some(format!(
            "{}: local split report mode missing or invalid",
            output.path
        ));
    }
    if value.get("predicted_token").is_none() {
        return Some(format!(
            "{}: local split predicted_token missing",
            output.path
        ));
    }
    if let Some(failure) = local_split_topology_failure(output, &value, context) {
        return Some(failure);
    }
    if let Some(failure) = local_split_runtime_failure(output, &value, context) {
        return Some(failure);
    }
    let has_payload = value
        .get("stages")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .any(|stage| {
            stage
                .get("wire_payload_bytes")
                .and_then(serde_json::Value::as_u64)
                .is_some_and(|bytes| bytes > 0)
                && stage
                    .get("payload_bytes")
                    .and_then(serde_json::Value::as_u64)
                    .is_some_and(|bytes| bytes > 0)
        });
    if !has_payload {
        return Some(format!(
            "{}: local split payload and wire payload bytes missing",
            output.path
        ));
    }
    None
}

fn local_split_topology_failure(
    output: &EvidenceOutputStatus,
    value: &serde_json::Value,
    context: &EvidenceSemanticContext<'_>,
) -> Option<String> {
    if let Some(expected) = context.topology.stage_count {
        let actual = value
            .get("stages")
            .and_then(serde_json::Value::as_array)
            .map(Vec::len);
        if actual != Some(expected) {
            return Some(format!(
                "{}: local split stage_count {:?} != expected {}",
                output.path, actual, expected
            ));
        }
    }
    if let Some(expected) = context.topology.layer_end {
        let actual = value.get("layer_end").and_then(serde_json::Value::as_u64);
        if actual != Some(u64::from(expected)) {
            return Some(format!(
                "{}: local split layer_end {:?} != expected {}",
                output.path, actual, expected
            ));
        }
    }
    let expected_splits = context.topology.split_boundaries()?;
    let Some(actual_splits) = value
        .get("splits")
        .and_then(serde_json::Value::as_array)
        .map(|splits| {
            splits
                .iter()
                .filter_map(serde_json::Value::as_u64)
                .map(|split| split.to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
    else {
        return Some(format!("{}: local split splits missing", output.path));
    };
    if actual_splits != expected_splits {
        return Some(format!(
            "{}: local split splits {actual_splits} != expected {expected_splits}",
            output.path
        ));
    }
    None
}

fn local_split_runtime_failure(
    output: &EvidenceOutputStatus,
    value: &serde_json::Value,
    context: &EvidenceSemanticContext<'_>,
) -> Option<String> {
    let expected = context.runtime.activation_wire_dtype.as_deref()?;
    let actual = value.get("wire_dtype").and_then(serde_json::Value::as_str);
    if actual != Some(expected) {
        return Some(format!(
            "{}: local split wire_dtype {:?} != expected {}",
            output.path, actual, expected
        ));
    }
    None
}

fn certification_output_failure(outputs: &[EvidenceOutputStatus]) -> Option<String> {
    outputs
        .iter()
        .find(|output| output.exists)
        .and_then(certification_status_failure)
}

fn certification_status_failure(output: &EvidenceOutputStatus) -> Option<String> {
    let contents = match fs::read_to_string(&output.resolved_path) {
        Ok(contents) => contents,
        Err(error) => {
            return Some(format!(
                "{}: cannot read certification output: {error}",
                output.path
            ));
        }
    };
    let value = match serde_json::from_str::<serde_json::Value>(&contents) {
        Ok(value) => value,
        Err(error) => {
            return Some(format!(
                "{}: cannot parse certification output: {error}",
                output.path
            ));
        }
    };
    let status = value
        .get("status")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("missing");
    if status == "failed" {
        return Some(format!("{}: certification status failed", output.path));
    }
    if status == "missing" {
        return Some(format!("{}: certification status missing", output.path));
    }
    None
}

fn chat_corpus_output_failure(outputs: &[EvidenceOutputStatus]) -> Option<String> {
    let output = outputs.iter().find(|output| output.exists)?;
    let value = match read_json_output(output) {
        Ok(value) => value,
        Err(failure) => return Some(failure),
    };
    let errors = value
        .get("summary")
        .and_then(|summary| summary.get("errors"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default();
    if errors > 0 {
        return Some(format!("{}: chat corpus errors {errors}", output.path));
    }
    None
}

#[derive(Clone, Copy)]
enum FocusedRuntimeMode {
    Executed,
    SchemaSmoke,
}

impl FocusedRuntimeMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Executed => "executed",
            Self::SchemaSmoke => "schema-smoke",
        }
    }
}

fn focused_runtime_output_failure(
    outputs: &[EvidenceOutputStatus],
    expected_mode: FocusedRuntimeMode,
    context: &EvidenceSemanticContext<'_>,
) -> Option<String> {
    let output = outputs.iter().find(|output| output.exists)?;
    let value = match read_json_output(output) {
        Ok(value) => value,
        Err(failure) => return Some(failure),
    };
    let mode = value
        .get("mode")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("missing");
    if mode != expected_mode.as_str() {
        return Some(format!(
            "{}: focused-runtime mode {mode} != {}",
            output.path,
            expected_mode.as_str()
        ));
    }
    if focused_runtime_generated_tps(&value).is_none() {
        return Some(format!(
            "{}: focused-runtime generated throughput missing or non-positive",
            output.path
        ));
    }
    if focused_runtime_decode_p50_ms(&value).is_none() {
        return Some(format!(
            "{}: focused-runtime decode p50 missing",
            output.path
        ));
    }
    if let Some(failure) = focused_runtime_topology_failure(output, &value, context) {
        return Some(failure);
    }
    focused_runtime_runtime_failure(output, &value, context)
}

fn focused_runtime_topology_failure(
    output: &EvidenceOutputStatus,
    value: &serde_json::Value,
    context: &EvidenceSemanticContext<'_>,
) -> Option<String> {
    if let Some(expected) = context.topology.stage_count {
        let actual = nested_u64(value, &["topology", "stage_count"])
            .or_else(|| value.get("stage_count").and_then(serde_json::Value::as_u64));
        if actual != Some(expected as u64) {
            return Some(format!(
                "{}: focused-runtime stage_count {:?} != expected {}",
                output.path, actual, expected
            ));
        }
    }
    if let Some(expected) = context.topology.layer_end {
        let actual = nested_u64(value, &["topology", "layer_end"]);
        if actual != Some(u64::from(expected)) {
            return Some(format!(
                "{}: focused-runtime layer_end {:?} != expected {}",
                output.path, actual, expected
            ));
        }
    }
    let expected_splits = context.topology.split_boundaries()?;
    let actual_splits = nested_str(value, &["topology", "splits"]);
    if actual_splits != Some(expected_splits.as_str()) {
        return Some(format!(
            "{}: focused-runtime splits {:?} != expected {}",
            output.path, actual_splits, expected_splits
        ));
    }
    None
}

fn focused_runtime_runtime_failure(
    output: &EvidenceOutputStatus,
    value: &serde_json::Value,
    context: &EvidenceSemanticContext<'_>,
) -> Option<String> {
    let runtime = &context.runtime;
    if let Some(expected) = runtime.ctx_size {
        let actual = nested_u64(value, &["runtime", "ctx_size"]);
        if actual != Some(u64::from(expected)) {
            return Some(format!(
                "{}: focused-runtime ctx_size {:?} != expected {}",
                output.path, actual, expected
            ));
        }
    }
    if let Some(expected) = runtime.n_gpu_layers {
        let actual = nested_i64(value, &["runtime", "n_gpu_layers"]);
        if actual != Some(i64::from(expected)) {
            return Some(format!(
                "{}: focused-runtime n_gpu_layers {:?} != expected {}",
                output.path, actual, expected
            ));
        }
    }
    for (field, expected) in [
        ("cache_type_k", runtime.cache_type_k.as_deref()),
        ("cache_type_v", runtime.cache_type_v.as_deref()),
        (
            "activation_wire_dtype",
            runtime.activation_wire_dtype.as_deref(),
        ),
    ] {
        let Some(expected) = expected else {
            continue;
        };
        let actual = nested_str(value, &["runtime", field]);
        if actual != Some(expected) {
            return Some(format!(
                "{}: focused-runtime {field} {:?} != expected {}",
                output.path, actual, expected
            ));
        }
    }
    None
}

fn focused_runtime_generated_tps(value: &serde_json::Value) -> Option<f64> {
    value
        .get("throughput_tokens_per_second")
        .and_then(|throughput| throughput.get("generated"))
        .and_then(value_as_f64)
        .filter(|value| *value > 0.0)
}

fn focused_runtime_decode_p50_ms(value: &serde_json::Value) -> Option<f64> {
    value
        .get("latency_ms")
        .and_then(|latency| latency.get("decode_elapsed_ms_p50"))
        .and_then(value_as_f64)
}

fn token_lengths_output_failure(
    outputs: &[EvidenceOutputStatus],
    context: &EvidenceSemanticContext<'_>,
) -> Option<String> {
    outputs
        .iter()
        .filter(|output| output.exists)
        .find(|output| output.path.ends_with(".json"))
        .and_then(|output| token_lengths_summary_failure(output, context))
}

fn token_lengths_summary_failure(
    output: &EvidenceOutputStatus,
    context: &EvidenceSemanticContext<'_>,
) -> Option<String> {
    let value = match read_json_output(output) {
        Ok(value) => value,
        Err(failure) => return Some(failure),
    };
    if let Some(expected) = context.runtime.ctx_size {
        let actual = value.get("ctx_size").and_then(serde_json::Value::as_u64);
        if actual != Some(u64::from(expected)) {
            return Some(format!(
                "{}: token length ctx_size {:?} != expected {}",
                output.path, actual, expected
            ));
        }
    }
    let exceeds_context = value
        .get("exceeds_context")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default();
    if exceeds_context > 0 {
        return Some(format!(
            "{}: token length rows exceed context {exceeds_context}",
            output.path
        ));
    }
    None
}

fn rank_output_failure(outputs: &[EvidenceOutputStatus]) -> Option<String> {
    let output = outputs.iter().find(|output| output.exists)?;
    let value = match read_json_output(output) {
        Ok(value) => value,
        Err(failure) => return Some(failure),
    };
    if value.get("kind").and_then(serde_json::Value::as_str) != Some("skippy_quant_pack_rank") {
        return Some(format!(
            "{}: rank report kind missing or invalid",
            output.path
        ));
    }
    let Some(candidates) = value
        .get("candidates")
        .and_then(serde_json::Value::as_array)
    else {
        return Some(format!("{}: rank report candidates missing", output.path));
    };
    let candidate_count = value
        .get("candidate_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default() as usize;
    if candidate_count == 0 {
        return Some(format!("{}: rank report candidate_count is 0", output.path));
    }
    if candidate_count != candidates.len() {
        return Some(format!(
            "{}: rank report candidate_count {candidate_count} != candidates length {}",
            output.path,
            candidates.len()
        ));
    }
    None
}

fn value_as_f64(value: &serde_json::Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_u64().map(|value| value as f64))
        .or_else(|| value.as_i64().map(|value| value as f64))
}

fn nested_u64(value: &serde_json::Value, path: &[&str]) -> Option<u64> {
    nested_value(value, path).and_then(serde_json::Value::as_u64)
}

fn nested_i64(value: &serde_json::Value, path: &[&str]) -> Option<i64> {
    nested_value(value, path).and_then(serde_json::Value::as_i64)
}

fn nested_str<'a>(value: &'a serde_json::Value, path: &[&str]) -> Option<&'a str> {
    nested_value(value, path).and_then(serde_json::Value::as_str)
}

fn nested_value<'a>(value: &'a serde_json::Value, path: &[&str]) -> Option<&'a serde_json::Value> {
    path.iter().try_fold(value, |current, key| current.get(key))
}

fn read_json_output(output: &EvidenceOutputStatus) -> Result<serde_json::Value, String> {
    let contents = match fs::read_to_string(&output.resolved_path) {
        Ok(contents) => contents,
        Err(error) => {
            return Err(format!("{}: cannot read output: {error}", output.path));
        }
    };
    match serde_json::from_str::<serde_json::Value>(&contents) {
        Ok(value) => Ok(value),
        Err(error) => Err(format!("{}: cannot parse output: {error}", output.path)),
    }
}
