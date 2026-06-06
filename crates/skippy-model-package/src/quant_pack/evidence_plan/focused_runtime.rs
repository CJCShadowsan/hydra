use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, clap::Args)]
pub(super) struct FocusedRuntimeEvidenceArgs {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) metrics_server_bin: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) stage_server_bin: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) lab_preflight_script: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) lab_preflight_hosts: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) lab_preflight_min_free_gb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) lab_preflight_ports: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) lab_preflight_ssh_opts: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) work_dir: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) remote_root: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) remote_root_map: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) remote_shared_root_map: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) endpoint_host_map: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) ssh_opts: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) metrics_otlp_grpc_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) remote_bind_host: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) first_stage_port: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) startup_timeout_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) stage_max_inflight: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) stage_reply_credit_limit: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) stage_downstream_wire_delay_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) stage_downstream_wire_mbps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) stage_telemetry_queue_capacity: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    pub(super) stage_telemetry_level: Option<String>,
    #[serde(skip_serializing_if = "is_false")]
    #[arg(long)]
    pub(super) rsync_model_artifacts: bool,
    #[serde(skip_serializing_if = "is_false")]
    #[arg(long)]
    pub(super) keep_remote: bool,
    #[serde(skip_serializing_if = "is_false")]
    #[arg(long)]
    pub(super) child_logs: bool,
    #[serde(skip_serializing_if = "is_false")]
    #[arg(long)]
    pub(super) stage_async_prefill_forward: bool,
    #[serde(skip_serializing_if = "is_false")]
    #[arg(long)]
    pub(super) allow_uneven_stage_ranges: bool,
}

impl FocusedRuntimeEvidenceArgs {
    pub(super) fn is_default(&self) -> bool {
        self == &Self::default()
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}
