use crate::network_cost::NetworkCostSnapshot;
use serde::{Deserialize, Serialize};
use std::env;

const DEFAULT_TTFT_BUDGET_MS: f64 = 750.0;
const DEFAULT_TPOT_BUDGET_MS: f64 = 80.0;
const DEFAULT_AFFINITY_OVERRIDE_THRESHOLD_MS: f64 = 75.0;
const DEFAULT_STALE_AFTER_MS: u64 = 20_000;
const DEFAULT_CACHE_AFFINITY_CREDIT_MS: f64 = 75.0;
const DEFAULT_FAILURE_PENALTY_MS: f64 = 500.0;
const DEFAULT_UNKNOWN_REMOTE_PENALTY_MS: f64 = 50.0;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerMode {
    Off,
    #[default]
    Shadow,
    Active,
}

impl SchedulerMode {
    pub fn from_env_value(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "disabled" => Some(Self::Off),
            "shadow" | "dry_run" | "dry-run" => Some(Self::Shadow),
            "active" | "enabled" => Some(Self::Active),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetKind {
    Local,
    Remote,
    Endpoint,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SchedulerConfig {
    #[serde(default)]
    pub mode: SchedulerMode,
    #[serde(default = "default_ttft_budget_ms")]
    pub ttft_budget_ms: f64,
    #[serde(default = "default_tpot_budget_ms")]
    pub tpot_budget_ms: f64,
    #[serde(default = "default_affinity_override_threshold_ms")]
    pub affinity_override_threshold_ms: f64,
    #[serde(default = "default_stale_after_ms")]
    pub stale_after_ms: u64,
    #[serde(default = "default_cache_affinity_credit_ms")]
    pub cache_affinity_credit_ms: f64,
    #[serde(default = "default_failure_penalty_ms")]
    pub failure_penalty_ms: f64,
    #[serde(default = "default_unknown_remote_penalty_ms")]
    pub unknown_remote_penalty_ms: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            mode: SchedulerMode::Shadow,
            ttft_budget_ms: DEFAULT_TTFT_BUDGET_MS,
            tpot_budget_ms: DEFAULT_TPOT_BUDGET_MS,
            affinity_override_threshold_ms: DEFAULT_AFFINITY_OVERRIDE_THRESHOLD_MS,
            stale_after_ms: DEFAULT_STALE_AFTER_MS,
            cache_affinity_credit_ms: DEFAULT_CACHE_AFFINITY_CREDIT_MS,
            failure_penalty_ms: DEFAULT_FAILURE_PENALTY_MS,
            unknown_remote_penalty_ms: DEFAULT_UNKNOWN_REMOTE_PENALTY_MS,
        }
    }
}

impl SchedulerConfig {
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Ok(mode) = env::var("MESH_LLM_SCHEDULER_MODE") {
            if let Some(mode) = SchedulerMode::from_env_value(&mode) {
                config.mode = mode;
            }
        }
        set_env_f64(
            "MESH_LLM_SCHEDULER_TTFT_BUDGET_MS",
            &mut config.ttft_budget_ms,
        );
        set_env_f64(
            "MESH_LLM_SCHEDULER_TPOT_BUDGET_MS",
            &mut config.tpot_budget_ms,
        );
        set_env_f64(
            "MESH_LLM_SCHEDULER_AFFINITY_OVERRIDE_THRESHOLD_MS",
            &mut config.affinity_override_threshold_ms,
        );
        set_env_u64(
            "MESH_LLM_SCHEDULER_STALE_AFTER_MS",
            &mut config.stale_after_ms,
        );
        set_env_f64(
            "MESH_LLM_SCHEDULER_CACHE_AFFINITY_CREDIT_MS",
            &mut config.cache_affinity_credit_ms,
        );
        set_env_f64(
            "MESH_LLM_SCHEDULER_FAILURE_PENALTY_MS",
            &mut config.failure_penalty_ms,
        );
        set_env_f64(
            "MESH_LLM_SCHEDULER_UNKNOWN_REMOTE_PENALTY_MS",
            &mut config.unknown_remote_penalty_ms,
        );
        config
    }

    pub fn status_snapshot(&self) -> SchedulerStatusSnapshot {
        SchedulerStatusSnapshot {
            mode: self.mode,
            ttft_budget_ms: self.ttft_budget_ms,
            tpot_budget_ms: self.tpot_budget_ms,
            affinity_override_threshold_ms: self.affinity_override_threshold_ms,
            stale_after_ms: self.stale_after_ms,
            cache_affinity_credit_ms: self.cache_affinity_credit_ms,
            failure_penalty_ms: self.failure_penalty_ms,
            unknown_remote_penalty_ms: self.unknown_remote_penalty_ms,
        }
    }

    pub fn decide(
        &self,
        existing_target: Option<&str>,
        candidates: &[SchedulerTargetCandidate],
    ) -> SchedulerDecision {
        let mut scored = candidates
            .iter()
            .map(|candidate| score_candidate(candidate, self))
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| {
            left.total_ms
                .partial_cmp(&right.total_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.target.cmp(&right.target))
        });

        let best = scored.first().cloned();
        let existing_score = existing_target.and_then(|target| {
            scored
                .iter()
                .find(|score| score.target == target)
                .cloned()
        });
        let best_target = best.as_ref().map(|score| score.target.clone());

        if self.mode == SchedulerMode::Off {
            return SchedulerDecision {
                mode: self.mode,
                selected_target: existing_target.map(str::to_string).or(best_target.clone()),
                shadow_target: None,
                existing_target: existing_target.map(str::to_string),
                active_override: false,
                reason: "scheduler disabled; existing route preserved".to_string(),
                candidates: scored,
            };
        }

        if self.mode == SchedulerMode::Shadow {
            return SchedulerDecision {
                mode: self.mode,
                selected_target: existing_target.map(str::to_string).or(best_target.clone()),
                shadow_target: best_target,
                existing_target: existing_target.map(str::to_string),
                active_override: false,
                reason: "shadow mode; scored route recorded without changing target".to_string(),
                candidates: scored,
            };
        }

        let Some(best) = best else {
            return SchedulerDecision {
                mode: self.mode,
                selected_target: existing_target.map(str::to_string),
                shadow_target: None,
                existing_target: existing_target.map(str::to_string),
                active_override: false,
                reason: "no eligible candidates; existing route preserved".to_string(),
                candidates: scored,
            };
        };

        let should_override = match existing_score.as_ref() {
            Some(existing) if existing.target == best.target => false,
            Some(existing) if existing.affinity_cached => {
                best.total_ms + self.affinity_override_threshold_ms < existing.total_ms
            }
            Some(existing) => best.total_ms < existing.total_ms,
            None => true,
        };

        let selected_target = if should_override {
            Some(best.target.clone())
        } else {
            existing_target.map(str::to_string).or(Some(best.target.clone()))
        };
        let reason = if should_override {
            format!(
                "active mode; selected {} with predicted {:.1} ms",
                best.target, best.total_ms
            )
        } else {
            "active mode; existing affinity/route stayed within threshold".to_string()
        };

        SchedulerDecision {
            mode: self.mode,
            selected_target,
            shadow_target: None,
            existing_target: existing_target.map(str::to_string),
            active_override: should_override,
            reason,
            candidates: scored,
        }
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct SchedulerStatusSnapshot {
    pub mode: SchedulerMode,
    pub ttft_budget_ms: f64,
    pub tpot_budget_ms: f64,
    pub affinity_override_threshold_ms: f64,
    pub stale_after_ms: u64,
    pub cache_affinity_credit_ms: f64,
    pub failure_penalty_ms: f64,
    pub unknown_remote_penalty_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SchedulerTargetCandidate {
    pub target: String,
    pub kind: TargetKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network: Option<NetworkCostSnapshot>,
    #[serde(default)]
    pub affinity_cached: bool,
    #[serde(default)]
    pub cache_ready: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_prefill_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_decode_pressure_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_cold_start_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_kv_transfer_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_artifact_ms: Option<f64>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct SchedulerCandidateScore {
    pub target: String,
    pub kind: TargetKind,
    pub total_ms: f64,
    pub queue_ms: f64,
    pub network_ms: f64,
    pub prefill_or_cache_miss_ms: f64,
    pub kv_transfer_ms: f64,
    pub cold_start_ms: f64,
    pub decode_pressure_ms: f64,
    pub artifact_materialization_ms: f64,
    pub cache_affinity_credit_ms: f64,
    pub failure_penalty_ms: f64,
    pub stale_metrics: bool,
    pub affinity_cached: bool,
    pub cache_ready: bool,
    pub cache_fetch_used: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct SchedulerDecision {
    pub mode: SchedulerMode,
    pub selected_target: Option<String>,
    pub shadow_target: Option<String>,
    pub existing_target: Option<String>,
    pub active_override: bool,
    pub reason: String,
    pub candidates: Vec<SchedulerCandidateScore>,
}

fn score_candidate(
    candidate: &SchedulerTargetCandidate,
    config: &SchedulerConfig,
) -> SchedulerCandidateScore {
    let fresh_network = candidate
        .network
        .as_ref()
        .filter(|network| network.last_updated_ms_ago <= config.stale_after_ms);
    let stale_metrics = candidate
        .network
        .as_ref()
        .map(|network| network.last_updated_ms_ago > config.stale_after_ms)
        .unwrap_or(true);
    let queue_ms = fresh_network
        .and_then(|network| network.avg_queue_wait_ms)
        .unwrap_or(0.0);
    let network_ms = fresh_network
        .and_then(|network| network.avg_rtt_ms)
        .or_else(|| fresh_network.and_then(|network| network.avg_attempt_ms).map(|ms| ms * 0.1))
        .unwrap_or_else(|| match candidate.kind {
            TargetKind::Local => 0.0,
            TargetKind::Remote => config.unknown_remote_penalty_ms,
            TargetKind::Endpoint => config.unknown_remote_penalty_ms * 0.5,
        });
    let recompute_ms = candidate
        .estimated_prefill_ms
        .or_else(|| fresh_network.and_then(|network| network.avg_ttft_ms))
        .unwrap_or(0.0);
    let raw_kv_transfer_ms = candidate
        .estimated_kv_transfer_ms
        .or_else(|| fresh_network.and_then(|network| network.avg_kv_transfer_ms))
        .unwrap_or(0.0);
    let cache_fetch_used = candidate.cache_ready
        && (raw_kv_transfer_ms == 0.0 || recompute_ms == 0.0 || raw_kv_transfer_ms < recompute_ms);
    let prefill_or_cache_miss_ms = if cache_fetch_used { 0.0 } else { recompute_ms };
    let kv_transfer_ms = if cache_fetch_used {
        raw_kv_transfer_ms
    } else {
        0.0
    };
    let cold_start_ms = candidate.estimated_cold_start_ms.unwrap_or(0.0);
    let decode_pressure_ms = candidate
        .estimated_decode_pressure_ms
        .or_else(|| fresh_network.and_then(|network| network.avg_itl_ms))
        .unwrap_or(0.0);
    let artifact_materialization_ms = candidate
        .estimated_artifact_ms
        .or_else(|| fresh_network.and_then(|network| network.avg_artifact_materialization_ms))
        .unwrap_or(0.0);
    let failure_penalty_ms = fresh_network
        .map(|network| (1.0 - network.success_rate.clamp(0.0, 1.0)) * config.failure_penalty_ms)
        .unwrap_or_else(|| {
            if candidate.kind == TargetKind::Local {
                0.0
            } else {
                config.unknown_remote_penalty_ms
            }
        });
    let cache_affinity_credit_ms = if candidate.affinity_cached {
        config.cache_affinity_credit_ms
    } else {
        fresh_network
            .and_then(|network| network.cache_hit_rate)
            .map(|rate| rate.clamp(0.0, 1.0) * config.cache_affinity_credit_ms)
            .unwrap_or(0.0)
    };
    let total_ms = queue_ms
        + network_ms
        + prefill_or_cache_miss_ms
        + kv_transfer_ms
        + cold_start_ms
        + decode_pressure_ms
        + artifact_materialization_ms
        + failure_penalty_ms
        - cache_affinity_credit_ms;

    SchedulerCandidateScore {
        target: candidate.target.clone(),
        kind: candidate.kind,
        total_ms,
        queue_ms,
        network_ms,
        prefill_or_cache_miss_ms,
        kv_transfer_ms,
        cold_start_ms,
        decode_pressure_ms,
        artifact_materialization_ms,
        cache_affinity_credit_ms,
        failure_penalty_ms,
        stale_metrics,
        affinity_cached: candidate.affinity_cached,
        cache_ready: candidate.cache_ready,
        cache_fetch_used,
    }
}

fn set_env_f64(name: &str, field: &mut f64) {
    if let Ok(value) = env::var(name)
        && let Ok(parsed) = value.parse::<f64>()
        && parsed.is_finite()
        && parsed >= 0.0
    {
        *field = parsed;
    }
}

fn set_env_u64(name: &str, field: &mut u64) {
    if let Ok(value) = env::var(name)
        && let Ok(parsed) = value.parse::<u64>()
    {
        *field = parsed;
    }
}

fn default_ttft_budget_ms() -> f64 {
    DEFAULT_TTFT_BUDGET_MS
}

fn default_tpot_budget_ms() -> f64 {
    DEFAULT_TPOT_BUDGET_MS
}

fn default_affinity_override_threshold_ms() -> f64 {
    DEFAULT_AFFINITY_OVERRIDE_THRESHOLD_MS
}

fn default_stale_after_ms() -> u64 {
    DEFAULT_STALE_AFTER_MS
}

fn default_cache_affinity_credit_ms() -> f64 {
    DEFAULT_CACHE_AFFINITY_CREDIT_MS
}

fn default_failure_penalty_ms() -> f64 {
    DEFAULT_FAILURE_PENALTY_MS
}

fn default_unknown_remote_penalty_ms() -> f64 {
    DEFAULT_UNKNOWN_REMOTE_PENALTY_MS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn network(target: &str, attempt_ms: f64, success_rate: f64) -> NetworkCostSnapshot {
        NetworkCostSnapshot {
            model: Some("qwen".to_string()),
            target: target.to_string(),
            kind: "remote".to_string(),
            sample_count: 10,
            success_count: (success_rate * 10.0) as u64,
            failure_count: ((1.0 - success_rate) * 10.0) as u64,
            success_rate,
            avg_queue_wait_ms: Some(10.0),
            avg_attempt_ms: Some(attempt_ms),
            avg_rtt_ms: Some(attempt_ms),
            avg_jitter_ms: None,
            avg_bandwidth_mbps: None,
            avg_ttft_ms: None,
            avg_itl_ms: None,
            avg_tokens_per_second: None,
            avg_kv_transfer_ms: None,
            avg_artifact_materialization_ms: None,
            cache_hit_rate: None,
            last_updated_unix_ms: 1,
            last_updated_ms_ago: 10,
            stale: false,
        }
    }

    #[test]
    fn shadow_scores_without_overriding() {
        let config = SchedulerConfig::default();
        let decision = config.decide(
            Some("slow"),
            &[
                SchedulerTargetCandidate {
                    target: "slow".to_string(),
                    kind: TargetKind::Remote,
                    network: Some(network("slow", 200.0, 1.0)),
                    affinity_cached: true,
                    cache_ready: true,
                    estimated_prefill_ms: None,
                    estimated_decode_pressure_ms: None,
                    estimated_cold_start_ms: None,
                    estimated_kv_transfer_ms: None,
                    estimated_artifact_ms: None,
                },
                SchedulerTargetCandidate {
                    target: "fast".to_string(),
                    kind: TargetKind::Remote,
                    network: Some(network("fast", 20.0, 1.0)),
                    affinity_cached: false,
                    cache_ready: true,
                    estimated_prefill_ms: None,
                    estimated_decode_pressure_ms: None,
                    estimated_cold_start_ms: None,
                    estimated_kv_transfer_ms: None,
                    estimated_artifact_ms: None,
                },
            ],
        );
        assert_eq!(decision.selected_target.as_deref(), Some("slow"));
        assert_eq!(decision.shadow_target.as_deref(), Some("fast"));
        assert!(!decision.active_override);
    }

    #[test]
    fn active_overrides_affinity_only_past_threshold() {
        let config = SchedulerConfig {
            mode: SchedulerMode::Active,
            affinity_override_threshold_ms: 25.0,
            ..SchedulerConfig::default()
        };
        let decision = config.decide(
            Some("slow"),
            &[
                SchedulerTargetCandidate {
                    target: "slow".to_string(),
                    kind: TargetKind::Remote,
                    network: Some(network("slow", 200.0, 1.0)),
                    affinity_cached: true,
                    cache_ready: true,
                    estimated_prefill_ms: None,
                    estimated_decode_pressure_ms: None,
                    estimated_cold_start_ms: None,
                    estimated_kv_transfer_ms: None,
                    estimated_artifact_ms: None,
                },
                SchedulerTargetCandidate {
                    target: "fast".to_string(),
                    kind: TargetKind::Remote,
                    network: Some(network("fast", 40.0, 1.0)),
                    affinity_cached: false,
                    cache_ready: true,
                    estimated_prefill_ms: None,
                    estimated_decode_pressure_ms: None,
                    estimated_cold_start_ms: None,
                    estimated_kv_transfer_ms: None,
                    estimated_artifact_ms: None,
                },
            ],
        );
        assert_eq!(decision.selected_target.as_deref(), Some("fast"));
        assert!(decision.active_override);
    }

    #[test]
    fn cache_fetch_is_used_only_when_it_beats_recompute() {
        let config = SchedulerConfig {
            mode: SchedulerMode::Active,
            ..SchedulerConfig::default()
        };
        let decision = config.decide(
            None,
            &[
                SchedulerTargetCandidate {
                    target: "fetch".to_string(),
                    kind: TargetKind::Remote,
                    network: Some(network("fetch", 10.0, 1.0)),
                    affinity_cached: false,
                    cache_ready: true,
                    estimated_prefill_ms: Some(200.0),
                    estimated_decode_pressure_ms: None,
                    estimated_cold_start_ms: None,
                    estimated_kv_transfer_ms: Some(40.0),
                    estimated_artifact_ms: None,
                },
                SchedulerTargetCandidate {
                    target: "recompute".to_string(),
                    kind: TargetKind::Remote,
                    network: Some(network("recompute", 10.0, 1.0)),
                    affinity_cached: false,
                    cache_ready: true,
                    estimated_prefill_ms: Some(60.0),
                    estimated_decode_pressure_ms: None,
                    estimated_cold_start_ms: None,
                    estimated_kv_transfer_ms: Some(120.0),
                    estimated_artifact_ms: None,
                },
            ],
        );

        let fetch = decision
            .candidates
            .iter()
            .find(|candidate| candidate.target == "fetch")
            .unwrap();
        assert!(fetch.cache_fetch_used);
        assert_eq!(fetch.kv_transfer_ms, 40.0);
        assert_eq!(fetch.prefill_or_cache_miss_ms, 0.0);

        let recompute = decision
            .candidates
            .iter()
            .find(|candidate| candidate.target == "recompute")
            .unwrap();
        assert!(!recompute.cache_fetch_used);
        assert_eq!(recompute.kv_transfer_ms, 0.0);
        assert_eq!(recompute.prefill_or_cache_miss_ms, 60.0);
    }
}
