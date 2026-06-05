use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use skippy_ffi::TensorRole;
use skippy_runtime::TensorInfo;

use super::*;

#[test]
fn profile_reports_decode_scaffold_and_stage_bytes() {
    let dir = temp_dir("profile-stage-bytes");
    write_manifest(&dir);
    let args = ProfileArgs {
        package: dir.clone(),
        stages: 2,
        phase: ProfilePhase::Decode,
        existing_kv_tokens: 32_768,
        generated_tokens: 1,
        batch_size: 1,
        kv_type: "f16".to_string(),
        backend: Some("metal".to_string()),
        device: Some("metal:test".to_string()),
        samples: 20,
        warmup_samples: 3,
        timing_source: TimingSourceKind::Static,
        out: None,
    };

    let report = profile_package(&args).expect("profile package");

    assert_eq!(report.kind, "skippy_agent_quant_profile");
    assert!(matches!(report.input_kind, ProfileInputKind::LayerPackage));
    assert_eq!(report.request_shape.phase as u8, ProfilePhase::Decode as u8);
    assert_eq!(report.measurement.samples, 20);
    assert_eq!(report.measurement.warmup_samples, 3);
    assert_eq!(report.measurement_status.status, "not_measured");
    assert_eq!(report.summary.stage_count, 2);
    assert_eq!(report.summary.layer_artifact_bytes, 100);
    assert_eq!(report.summary.shared_artifact_bytes, 35);
    assert_eq!(report.layers.len(), 4);
    assert_eq!(report.stages.len(), 2);
    assert_eq!(report.stages[0].layer_start, 0);
    assert_eq!(report.stages[0].layer_end, 2);
    assert_eq!(report.stages[0].artifact_bytes, 65);
    assert_eq!(report.stages[1].layer_start, 2);
    assert_eq!(report.stages[1].layer_end, 4);
    assert_eq!(report.stages[1].artifact_bytes, 80);

    fs::remove_dir_all(dir).ok();
}

#[test]
fn direct_gguf_profile_synthesizes_layer_and_stage_bytes() {
    let tensors = vec![
        tensor("tokenizer.ggml.tokens", None, TensorRole::Tokenizer, 7),
        tensor("token_embd.weight", None, TensorRole::Embedding, 11),
        tensor("blk.0.attn_q.weight", Some(0), TensorRole::Layer, 13),
        tensor("blk.0.attn_k.weight", Some(0), TensorRole::Layer, 17),
        tensor("blk.1.attn_q.weight", Some(1), TensorRole::Layer, 19),
        tensor("blk.2.attn_q.weight", Some(2), TensorRole::Layer, 23),
        tensor("output_norm.weight", None, TensorRole::FinalNorm, 29),
        tensor("output.weight", None, TensorRole::Output, 31),
    ];
    let timing_report = ProfileTimingReport {
        measurement_status: MeasurementStatus {
            status: "not_measured".to_string(),
            reason: "test".to_string(),
        },
        layer_timings: BTreeMap::new(),
        stage_timings: BTreeMap::new(),
        estimated_tokens_per_second: None,
    };

    let layer_count = direct_layer_count(&tensors).expect("layer count");
    let layers = direct_layer_profiles(&tensors, Path::new("model.gguf"), "sha", &timing_report);
    let shared = direct_shared_profile(&tensors, Path::new("model.gguf"), "sha");
    let stages = direct_stage_profiles(&tensors, layer_count, 2, &timing_report);

    assert_eq!(layer_count, 3);
    assert_eq!(layers.len(), 3);
    assert_eq!(layers[0].artifact.tensor_bytes, 30);
    assert_eq!(shared.metadata.tensor_bytes, 7);
    assert_eq!(shared.embeddings.tensor_bytes, 11);
    assert_eq!(shared.output.tensor_bytes, 60);
    assert_eq!(stages.len(), 2);
    assert_eq!(stages[0].layer_start, 0);
    assert_eq!(stages[0].layer_end, 2);
    assert_eq!(stages[0].artifact_bytes, 67);
    assert_eq!(stages[1].layer_start, 2);
    assert_eq!(stages[1].layer_end, 3);
    assert_eq!(stages[1].artifact_bytes, 90);
}

#[test]
fn profile_rejects_too_many_stages() {
    let dir = temp_dir("profile-too-many-stages");
    write_manifest(&dir);
    let args = ProfileArgs {
        package: dir.clone(),
        stages: 5,
        phase: ProfilePhase::Decode,
        existing_kv_tokens: 8192,
        generated_tokens: 1,
        batch_size: 1,
        kv_type: "f16".to_string(),
        backend: None,
        device: None,
        samples: 20,
        warmup_samples: 3,
        timing_source: TimingSourceKind::Static,
        out: None,
    };

    let error = profile_package(&args).expect_err("stage count should fail");

    assert!(
        error
            .to_string()
            .contains("--stages 5 exceeds package layer_count 4")
    );
    fs::remove_dir_all(dir).ok();
}

#[test]
fn local_stage_timing_rejects_layer_package_inputs() {
    let request_shape = local_stage_request(ProfilePhase::Decode);
    let measurement = local_stage_measurement();
    let input = local_stage_input(
        ProfileInputKind::LayerPackage,
        1,
        &request_shape,
        &measurement,
    );

    let error = LocalStageTimingSource
        .profile(&input)
        .expect_err("layer packages are not accepted yet");

    assert!(
        error
            .to_string()
            .contains("currently supports direct GGUF inputs only")
    );
}

#[test]
fn local_stage_timing_rejects_multi_stage_inputs() {
    let request_shape = local_stage_request(ProfilePhase::Decode);
    let measurement = local_stage_measurement();
    let input = local_stage_input(
        ProfileInputKind::DirectGguf,
        2,
        &request_shape,
        &measurement,
    );

    let error = LocalStageTimingSource
        .profile(&input)
        .expect_err("multi-stage local timing is not accepted yet");

    assert!(error.to_string().contains("currently supports --stages 1"));
}

#[test]
fn local_stage_timing_rejects_non_decode_phase() {
    let request_shape = local_stage_request(ProfilePhase::Prefill);
    let measurement = local_stage_measurement();
    let input = local_stage_input(
        ProfileInputKind::DirectGguf,
        1,
        &request_shape,
        &measurement,
    );

    let error = LocalStageTimingSource
        .profile(&input)
        .expect_err("non-decode local timing is not accepted yet");

    assert!(
        error
            .to_string()
            .contains("currently supports --phase decode")
    );
}

#[test]
fn local_stage_timing_marks_runtime_execution_unwired() {
    let request_shape = local_stage_request(ProfilePhase::Decode);
    let measurement = local_stage_measurement();
    let input = local_stage_input(
        ProfileInputKind::DirectGguf,
        1,
        &request_shape,
        &measurement,
    );

    let error = LocalStageTimingSource
        .profile(&input)
        .expect_err("runtime execution should not be wired yet");

    assert!(
        error
            .to_string()
            .contains("runtime execution is not wired yet")
    );
}

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("skippy-profile-{name}-{nanos}"));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn local_stage_input<'a>(
    input_kind: ProfileInputKind,
    stage_count: usize,
    request_shape: &'a RequestShape,
    measurement: &'a MeasurementConfig,
) -> ProfileTimingInput<'a> {
    static PACKAGE: &str = "model.gguf";
    ProfileTimingInput {
        package: Path::new(PACKAGE),
        input_kind,
        stage_count,
        request_shape,
        measurement,
    }
}

fn local_stage_request(phase: ProfilePhase) -> RequestShape {
    RequestShape {
        phase,
        existing_kv_tokens: 8192,
        generated_tokens: 1,
        batch_size: 1,
        kv_type: "f16".to_string(),
        backend: None,
        device: None,
    }
}

fn local_stage_measurement() -> MeasurementConfig {
    MeasurementConfig {
        source: TimingSourceKind::LocalStage,
        warmup_samples: 3,
        samples: 20,
    }
}

fn write_manifest(dir: &Path) {
    fs::write(
        dir.join("model-package.json"),
        r#"{
  "model_id": "test/model:Q4_K_M",
  "layer_count": 4,
  "activation_width": 1024,
  "shared": {
    "metadata": {
      "path": "shared/metadata.gguf",
      "tensor_count": 1,
      "tensor_bytes": 10,
      "artifact_bytes": 10,
      "sha256": "metadata"
    },
    "embeddings": {
      "path": "shared/embeddings.gguf",
      "tensor_count": 1,
      "tensor_bytes": 11,
      "artifact_bytes": 20,
      "sha256": "embeddings"
    },
    "output": {
      "path": "shared/output.gguf",
      "tensor_count": 1,
      "tensor_bytes": 12,
      "artifact_bytes": 5,
      "sha256": "output"
    }
  },
  "layers": [
    {
      "layer_index": 0,
      "path": "layers/layer-00000.gguf",
      "tensor_count": 2,
      "tensor_bytes": 21,
      "artifact_bytes": 15,
      "sha256": "layer0"
    },
    {
      "layer_index": 1,
      "path": "layers/layer-00001.gguf",
      "tensor_count": 2,
      "tensor_bytes": 22,
      "artifact_bytes": 20,
      "sha256": "layer1"
    },
    {
      "layer_index": 2,
      "path": "layers/layer-00002.gguf",
      "tensor_count": 2,
      "tensor_bytes": 23,
      "artifact_bytes": 25,
      "sha256": "layer2"
    },
    {
      "layer_index": 3,
      "path": "layers/layer-00003.gguf",
      "tensor_count": 2,
      "tensor_bytes": 24,
      "artifact_bytes": 40,
      "sha256": "layer3"
    }
  ],
  "skippy_abi_version": "0.1.25"
}"#,
    )
    .expect("write manifest");
}

fn tensor(name: &str, layer_index: Option<u32>, role: TensorRole, byte_size: u64) -> TensorInfo {
    TensorInfo {
        name: name.to_string(),
        layer_index,
        role,
        ggml_type: 0,
        byte_size,
        element_count: byte_size,
    }
}
