use crate::{ModelProfile, profile_gguf_path};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct HfCacheModelProfile {
    pub model_ref: String,
    pub path: PathBuf,
    pub profile: ModelProfile,
}

pub fn profile_hf_cache(cache_root: impl AsRef<Path>) -> Result<Vec<HfCacheModelProfile>> {
    let cache_root = cache_root.as_ref();
    let mut rows = Vec::new();
    for model_dir in read_dirs(cache_root)? {
        let Some(repo_id) = repo_id_from_cache_folder(&model_dir) else {
            continue;
        };
        let snapshots = model_dir.join("snapshots");
        if !snapshots.is_dir() {
            continue;
        }
        for revision_dir in read_dirs(&snapshots)? {
            collect_snapshot_profiles(&repo_id, &revision_dir, &mut rows)?;
        }
    }
    rows.sort_by(|left, right| {
        left.model_ref
            .cmp(&right.model_ref)
            .then_with(|| left.path.cmp(&right.path))
    });
    rows.dedup_by(|left, right| left.model_ref == right.model_ref);
    Ok(rows)
}

fn collect_snapshot_profiles(
    repo_id: &str,
    revision_dir: &Path,
    rows: &mut Vec<HfCacheModelProfile>,
) -> Result<()> {
    let mut paths = Vec::new();
    collect_gguf_paths(revision_dir, &mut paths)?;
    paths.sort();
    for path in paths {
        let relative_file = path
            .strip_prefix(revision_dir)
            .with_context(|| format!("derive relative path for {}", path.display()))?
            .to_string_lossy()
            .replace('\\', "/");
        if should_skip_relative_file(repo_id, &relative_file) {
            continue;
        }
        let model_ref =
            model_resolver::format_huggingface_display_ref(repo_id, None, &relative_file);
        let profile = match profile_gguf_path(&path) {
            Ok(profile) => profile,
            Err(err) => {
                eprintln!("skip {}: {err:#}", path.display());
                continue;
            }
        };
        rows.push(HfCacheModelProfile {
            model_ref,
            path,
            profile,
        });
    }
    Ok(())
}

fn collect_gguf_paths(dir: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    for entry in read_entries(dir)? {
        let path = entry.path();
        let file_type = std::fs::symlink_metadata(&path)
            .with_context(|| format!("read metadata for {}", path.display()))?
            .file_type();
        if file_type.is_dir() {
            collect_gguf_paths(&path, paths)?;
        } else if (file_type.is_file() || file_type.is_symlink())
            && path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            paths.push(path);
        }
    }
    Ok(())
}

fn should_skip_relative_file(repo_id: &str, relative_file: &str) -> bool {
    if repo_id.ends_with("-layers")
        && (relative_file.starts_with("shared/") || relative_file.starts_with("layers/"))
    {
        return true;
    }
    model_ref::split_gguf_shard_info(relative_file).is_some_and(|shard| shard.part != "00001")
}

fn repo_id_from_cache_folder(path: &Path) -> Option<String> {
    path.file_name()?
        .to_str()?
        .strip_prefix("models--")
        .map(|value| value.replace("--", "/"))
}

fn read_dirs(path: &Path) -> Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    for entry in read_entries(path)? {
        let path = entry.path();
        if std::fs::symlink_metadata(&path)
            .with_context(|| format!("read metadata for {}", path.display()))?
            .file_type()
            .is_dir()
        {
            dirs.push(path);
        }
    }
    dirs.sort();
    Ok(dirs)
}

fn read_entries(path: &Path) -> Result<Vec<std::fs::DirEntry>> {
    std::fs::read_dir(path)
        .with_context(|| format!("read {}", path.display()))?
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("read entries from {}", path.display()))
}
