use crate::models;

use super::CatalogModelHint;
use serde::Serialize;
use std::path::{Path, PathBuf};

const DISK_DOWNLOAD_HEADROOM_PERCENT: u64 = 5;

#[derive(Clone, Debug, Serialize)]
pub(super) struct DiskPlan {
    pub(super) cache_dir: String,
    pub(super) free_bytes: Option<u64>,
    pub(super) full_model_required_bytes: u64,
    pub(super) fits_full_model: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) layer_package_total_required_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) one_layer_required_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) fits_one_layer: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) planned_split_node_required_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) fits_planned_split_node: Option<bool>,
}

pub(super) fn build_disk_plan(
    hint: &CatalogModelHint,
    split_node_count: Option<usize>,
) -> DiskPlan {
    let cache_dir = models::huggingface_hub_cache_dir();
    let free_bytes = available_disk_bytes(&cache_dir);
    let full_model_required_bytes = disk_required_bytes(hint.model_bytes);
    let layer_package_total_required_bytes = hint.package_total_bytes.map(disk_required_bytes);
    let one_layer_required_bytes = layer_package_total_required_bytes
        .zip(hint.layer_count)
        .and_then(|(total, layers)| (layers > 0).then(|| total.div_ceil(u64::from(layers))));
    let planned_split_node_required_bytes = layer_package_total_required_bytes
        .zip(split_node_count)
        .and_then(|(total, nodes)| (nodes > 0).then(|| total.div_ceil(nodes as u64)));

    DiskPlan {
        cache_dir: cache_dir.display().to_string(),
        free_bytes,
        full_model_required_bytes,
        fits_full_model: fits_disk(free_bytes, full_model_required_bytes),
        layer_package_total_required_bytes,
        one_layer_required_bytes,
        fits_one_layer: one_layer_required_bytes
            .and_then(|required| fits_disk(free_bytes, required)),
        planned_split_node_required_bytes,
        fits_planned_split_node: planned_split_node_required_bytes
            .and_then(|required| fits_disk(free_bytes, required)),
    }
}

fn disk_required_bytes(bytes: u64) -> u64 {
    bytes
        .saturating_mul(100 + DISK_DOWNLOAD_HEADROOM_PERCENT)
        .div_ceil(100)
}

fn fits_disk(free_bytes: Option<u64>, required_bytes: u64) -> Option<bool> {
    free_bytes.map(|free| free >= required_bytes)
}

fn available_disk_bytes(path: &Path) -> Option<u64> {
    let probe_path = nearest_existing_path(path)?;
    stat_available_bytes(&probe_path)
}

fn nearest_existing_path(path: &Path) -> Option<PathBuf> {
    let mut current = path;
    loop {
        if current.exists() {
            return Some(current.to_path_buf());
        }
        current = current.parent()?;
    }
}

#[cfg(unix)]
fn stat_available_bytes(path: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), stat.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    let stat = unsafe { stat.assume_init() };
    let blocks = u128::from(stat.f_bavail);
    let block_size = u128::from(stat.f_frsize);
    blocks
        .checked_mul(block_size)
        .and_then(|bytes| u64::try_from(bytes).ok())
}

#[cfg(not(unix))]
fn stat_available_bytes(_path: &Path) -> Option<u64> {
    None
}
