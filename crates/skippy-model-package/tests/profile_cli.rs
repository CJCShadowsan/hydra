use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

#[test]
fn profile_cli_emits_decode_scaffold_json() {
    let package_dir = temp_dir("profile-cli");
    write_manifest(&package_dir);

    let output = Command::new(env!("CARGO_BIN_EXE_skippy-model-package"))
        .arg("profile")
        .arg(&package_dir)
        .arg("--stages")
        .arg("2")
        .arg("--phase")
        .arg("decode")
        .arg("--existing-kv-tokens")
        .arg("32768")
        .arg("--warmup-samples")
        .arg("4")
        .arg("--samples")
        .arg("24")
        .arg("--timing-source")
        .arg("static")
        .output()
        .expect("run skippy-model-package profile");

    assert!(
        output.status.success(),
        "profile command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let json: Value = serde_json::from_slice(&output.stdout).expect("parse profile json");
    assert_eq!(json["kind"], "skippy_agent_quant_profile");
    assert_eq!(json["input_kind"], "layer_package");
    assert_eq!(json["request_shape"]["phase"], "decode");
    assert_eq!(json["request_shape"]["existing_kv_tokens"], 32768);
    assert_eq!(json["measurement"]["source"], "static");
    assert_eq!(json["measurement"]["warmup_samples"], 4);
    assert_eq!(json["measurement"]["samples"], 24);
    assert_eq!(json["measurement_status"]["status"], "not_measured");
    assert_eq!(json["summary"]["stage_count"], 2);
    assert_eq!(json["summary"]["layer_artifact_bytes"], 100);
    assert_eq!(json["summary"]["shared_artifact_bytes"], 35);
    assert_eq!(json["layers"].as_array().expect("layers array").len(), 4);
    assert_eq!(json["stages"].as_array().expect("stages array").len(), 2);
    assert_eq!(json["stages"][0]["artifact_bytes"], 65);
    assert_eq!(json["stages"][1]["artifact_bytes"], 80);

    fs::remove_dir_all(package_dir).ok();
}

#[test]
fn profile_cli_rejects_local_stage_for_package_input() {
    let package_dir = temp_dir("profile-cli-local-stage");
    write_manifest(&package_dir);

    let output = Command::new(env!("CARGO_BIN_EXE_skippy-model-package"))
        .arg("profile")
        .arg(&package_dir)
        .arg("--timing-source")
        .arg("local-stage")
        .output()
        .expect("run skippy-model-package profile");

    assert!(
        !output.status.success(),
        "profile command unexpectedly passed"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("direct GGUF inputs only"),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    fs::remove_dir_all(package_dir).ok();
}

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("skippy-model-package-{name}-{nanos}"));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
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
