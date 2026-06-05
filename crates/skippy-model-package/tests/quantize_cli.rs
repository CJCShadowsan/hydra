use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

#[test]
fn quantize_emit_only_records_tensor_type_entries_in_manifest() {
    let run_dir = temp_dir("quantize-cli");
    let plan = run_dir.join("quant-plan.json");
    write_quant_plan(&plan);

    let output = Command::new(env!("CARGO_BIN_EXE_skippy-model-package"))
        .arg("quantize")
        .arg("source.gguf")
        .arg("--plan")
        .arg(&plan)
        .arg("--candidate")
        .arg("ffn-compressed-attention-protected")
        .arg("--out-dir")
        .arg(&run_dir)
        .arg("--emit-only")
        .arg("--keep-split")
        .output()
        .expect("run skippy-model-package quantize");

    assert!(
        output.status.success(),
        "quantize command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let tensor_types =
        fs::read_to_string(run_dir.join("tensor-types.txt")).expect("read tensor-types.txt");
    assert_eq!(
        tensor_types,
        concat!(
            "blk\\.(1|2)\\.attn_.*\\.weight=q4_k\n",
            "blk\\.(1|2)\\.ffn_.*\\.weight=q3_k\n"
        )
    );

    let manifest: Value = serde_json::from_slice(&output.stdout).expect("parse manifest json");
    assert_eq!(manifest["kind"], "skippy_quantize_run");
    assert_eq!(manifest["tensor_type_entry_count"], 2);
    assert_eq!(manifest["keep_split"], true);
    assert_eq!(
        manifest["tensor_type_entries"][0],
        "blk\\.(1|2)\\.attn_.*\\.weight=q4_k"
    );
    assert_eq!(
        manifest["tensor_type_entries"][1],
        "blk\\.(1|2)\\.ffn_.*\\.weight=q3_k"
    );
    assert!(manifest["quantized_model"].is_null());
    assert!(manifest["command"].is_null());

    fs::remove_dir_all(run_dir).ok();
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

fn write_quant_plan(path: &Path) {
    fs::write(
        path,
        r#"{
  "source": {
    "path": "source.gguf",
    "sha256": "source-sha",
    "inferred_source_quant": "Q4_K_M"
  },
  "profile": "coding-agent",
  "candidates": [
    {
      "id": "ffn-compressed-attention-protected",
      "layout_hash": "layout-hash",
      "name": "FFN compressed attention protected",
      "status": "experimental",
      "strategy": "stage-aware-ffn-compressed-attention-protected",
      "default_quant": "Q4_K_M",
      "groups": [
        {
          "name": "middle-attention-protected",
          "quant": "Q4_K_M",
          "selector": {
            "kind": "tensor_name_pattern",
            "patterns": ["blk\\.(1|2)\\.attn_.*\\.weight"]
          },
          "reason": "test attention"
        },
        {
          "name": "middle-ffn-compressed",
          "quant": "Q3_K_M",
          "selector": {
            "kind": "tensor_name_pattern",
            "patterns": ["blk\\.(1|2)\\.ffn_.*\\.weight"]
          },
          "reason": "test ffn"
        }
      ],
      "stage_hints": [],
      "notes": []
    }
  ]
}"#,
    )
    .expect("write quant plan");
}
