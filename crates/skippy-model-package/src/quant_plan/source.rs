use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use model_ref::split_gguf_shard_info;
use serde::Serialize;
use sha2::{Digest, Sha256};
use skippy_runtime::{ModelInfo, TensorInfo};

#[derive(Debug)]
pub(super) struct QuantSourceInspection {
    pub(super) source_sha256: String,
    pub(super) source_sha256_kind: String,
    pub(super) shards: Vec<SourceShardSummary>,
    pub(super) tensors: Vec<TensorInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct SourceShardSummary {
    path: String,
    sha256: String,
    tensor_count: usize,
    tensor_bytes: u64,
}

pub(super) fn inspect_quant_source(path: &Path) -> Result<QuantSourceInspection> {
    let shard_paths = resolve_gguf_shard_paths(path)?;
    let mut shards = Vec::with_capacity(shard_paths.len());
    let mut tensors = Vec::new();
    for shard_path in &shard_paths {
        let shard_tensors = inspect_shard_tensors(shard_path)?;
        shards.push(SourceShardSummary {
            path: shard_path.display().to_string(),
            sha256: file_sha256(shard_path)?,
            tensor_count: shard_tensors.len(),
            tensor_bytes: shard_tensors.iter().map(|tensor| tensor.byte_size).sum(),
        });
        tensors.extend(shard_tensors);
    }
    let (source_sha256, source_sha256_kind) = source_identity_hash(&shards);
    Ok(QuantSourceInspection {
        source_sha256,
        source_sha256_kind,
        shards: if shards.len() > 1 { shards } else { Vec::new() },
        tensors,
    })
}

fn inspect_shard_tensors(path: &Path) -> Result<Vec<TensorInfo>> {
    let model =
        ModelInfo::open(path).with_context(|| format!("open GGUF metadata {}", path.display()))?;
    model
        .tensors()
        .with_context(|| format!("inspect GGUF tensors {}", path.display()))
}

fn source_identity_hash(shards: &[SourceShardSummary]) -> (String, String) {
    if let [single] = shards {
        return (single.sha256.clone(), "file".to_string());
    }

    let mut hasher = Sha256::new();
    hash_str(&mut hasher, "skippy-split-gguf-source-v1");
    for shard in shards {
        hash_str(&mut hasher, &shard.path);
        hash_str(&mut hasher, &shard.sha256);
        hasher.update(shard.tensor_count.to_le_bytes());
        hasher.update(shard.tensor_bytes.to_le_bytes());
    }
    (
        format!("{:x}", hasher.finalize()),
        "split_shard_set".to_string(),
    )
}

fn resolve_gguf_shard_paths(path: &Path) -> Result<Vec<PathBuf>> {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return Ok(vec![path.to_path_buf()]);
    };
    let Some(shard) = split_gguf_shard_info(file_name) else {
        return Ok(vec![path.to_path_buf()]);
    };
    let total = shard
        .total
        .parse::<usize>()
        .with_context(|| format!("parse GGUF shard total from {file_name}"))?;
    if total <= 1 {
        return Ok(vec![path.to_path_buf()]);
    }

    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    let mut paths = Vec::with_capacity(total);
    for part in 1..=total {
        let shard_name = format!("{}-{part:05}-of-{}.gguf", shard.prefix, shard.total);
        let shard_path = parent.join(shard_name);
        if !shard_path.exists() {
            bail!(
                "split GGUF shard {} is missing sibling {}",
                path.display(),
                shard_path.display()
            );
        }
        paths.push(shard_path);
    }
    Ok(paths)
}

fn file_sha256(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_str(hasher: &mut Sha256, value: &str) {
    hasher.update(value.len().to_le_bytes());
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_gguf_source_resolves_all_sibling_shards() {
        let dir = unique_test_dir("quant-plan-split-source");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        for part in 1..=3 {
            std::fs::write(
                dir.join(format!("Model-Q4_K_M-{part:05}-of-00003.gguf")),
                b"test",
            )
            .expect("write shard");
        }

        let input = dir.join("Model-Q4_K_M-00001-of-00003.gguf");
        let paths = resolve_gguf_shard_paths(&input).expect("resolve shards");

        assert_eq!(paths.len(), 3);
        assert_eq!(
            paths[2].file_name().and_then(|name| name.to_str()),
            Some("Model-Q4_K_M-00003-of-00003.gguf")
        );
        std::fs::remove_dir_all(dir).expect("cleanup temp dir");
    }

    #[test]
    fn split_source_identity_depends_on_every_shard() {
        let first = vec![
            shard_summary("model-00001-of-00002.gguf", "a", 1),
            shard_summary("model-00002-of-00002.gguf", "b", 2),
        ];
        let second = vec![
            shard_summary("model-00001-of-00002.gguf", "a", 1),
            shard_summary("model-00002-of-00002.gguf", "changed", 2),
        ];

        let (first_hash, first_kind) = source_identity_hash(&first);
        let (second_hash, second_kind) = source_identity_hash(&second);

        assert_eq!(first_kind, "split_shard_set");
        assert_eq!(second_kind, "split_shard_set");
        assert_ne!(first_hash, second_hash);
    }

    fn shard_summary(path: &str, sha256: &str, tensor_count: usize) -> SourceShardSummary {
        SourceShardSummary {
            path: path.to_string(),
            sha256: sha256.to_string(),
            tensor_count,
            tensor_bytes: tensor_count as u64 * 10,
        }
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "{}-{}",
            name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ))
    }
}
