use serde::Serialize;
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const DEFAULT_MAX_TARGETS: usize = 2048;
const DEFAULT_TTL: Duration = Duration::from_secs(20 * 60);
const EWMA_ALPHA: f64 = 0.25;

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize)]
pub struct TargetNetworkKey {
    pub model: Option<String>,
    pub target: String,
    pub kind: String,
}

impl TargetNetworkKey {
    pub fn new(model: Option<&str>, target: impl Into<String>, kind: impl Into<String>) -> Self {
        Self {
            model: model.map(str::to_string),
            target: target.into(),
            kind: kind.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct NetworkCostConfig {
    pub max_targets: usize,
    pub ttl: Duration,
}

impl Default for NetworkCostConfig {
    fn default() -> Self {
        Self {
            max_targets: DEFAULT_MAX_TARGETS,
            ttl: DEFAULT_TTL,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NetworkCostObservation {
    pub queue_wait_ms: Option<f64>,
    pub attempt_ms: Option<f64>,
    pub rtt_ms: Option<f64>,
    pub jitter_ms: Option<f64>,
    pub bandwidth_mbps: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub itl_ms: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub kv_transfer_ms: Option<f64>,
    pub artifact_materialization_ms: Option<f64>,
    pub cache_hit: Option<bool>,
    pub success: bool,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct NetworkCostSnapshot {
    pub model: Option<String>,
    pub target: String,
    pub kind: String,
    pub sample_count: u64,
    pub success_count: u64,
    pub failure_count: u64,
    pub success_rate: f64,
    pub avg_queue_wait_ms: Option<f64>,
    pub avg_attempt_ms: Option<f64>,
    pub avg_rtt_ms: Option<f64>,
    pub avg_jitter_ms: Option<f64>,
    pub avg_bandwidth_mbps: Option<f64>,
    pub avg_ttft_ms: Option<f64>,
    pub avg_itl_ms: Option<f64>,
    pub avg_tokens_per_second: Option<f64>,
    pub avg_kv_transfer_ms: Option<f64>,
    pub avg_artifact_materialization_ms: Option<f64>,
    pub cache_hit_rate: Option<f64>,
    pub last_updated_unix_ms: u64,
    pub last_updated_ms_ago: u64,
    pub stale: bool,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct NetworkAdvisoryHint {
    pub advisory_version: u32,
    pub model: Option<String>,
    pub target: String,
    pub kind: String,
    pub ttl_ms: u64,
    pub observed_unix_ms: u64,
    pub success_rate: f64,
    pub avg_queue_wait_ms: Option<f64>,
    pub avg_rtt_ms: Option<f64>,
    pub avg_ttft_ms: Option<f64>,
    pub avg_itl_ms: Option<f64>,
    pub avg_tokens_per_second: Option<f64>,
    pub cache_hit_rate: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct NetworkCostStatusSnapshot {
    pub tracked_targets: usize,
    pub stale_targets: usize,
    pub max_targets: usize,
    pub ttl_secs: u64,
    pub advisory_hints: Vec<NetworkAdvisoryHint>,
    pub targets: Vec<NetworkCostSnapshot>,
}

#[derive(Clone, Debug, Default)]
struct TargetCostEntry {
    sample_count: u64,
    success_count: u64,
    failure_count: u64,
    queue_wait_ms: Option<f64>,
    attempt_ms: Option<f64>,
    rtt_ms: Option<f64>,
    jitter_ms: Option<f64>,
    bandwidth_mbps: Option<f64>,
    ttft_ms: Option<f64>,
    itl_ms: Option<f64>,
    tokens_per_second: Option<f64>,
    kv_transfer_ms: Option<f64>,
    artifact_materialization_ms: Option<f64>,
    cache_hit_count: u64,
    cache_observation_count: u64,
    last_updated: SystemTime,
}

#[derive(Clone, Debug)]
pub struct NetworkCostCollector {
    inner: std::sync::Arc<std::sync::Mutex<HashMap<TargetNetworkKey, TargetCostEntry>>>,
    config: NetworkCostConfig,
}

impl Default for NetworkCostCollector {
    fn default() -> Self {
        Self::new(NetworkCostConfig::default())
    }
}

impl NetworkCostCollector {
    pub fn new(config: NetworkCostConfig) -> Self {
        Self {
            inner: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            config,
        }
    }

    pub fn observe(&self, key: TargetNetworkKey, observation: NetworkCostObservation) {
        let mut entries = self.inner.lock().unwrap();
        prune_expired(&mut entries, self.config.ttl);
        let entry = entries.entry(key).or_default();
        entry.sample_count = entry.sample_count.saturating_add(1);
        if observation.success {
            entry.success_count = entry.success_count.saturating_add(1);
        } else {
            entry.failure_count = entry.failure_count.saturating_add(1);
        }
        update_ewma(&mut entry.queue_wait_ms, observation.queue_wait_ms);
        update_ewma(&mut entry.attempt_ms, observation.attempt_ms);
        update_ewma(&mut entry.rtt_ms, observation.rtt_ms);
        update_ewma(&mut entry.jitter_ms, observation.jitter_ms);
        update_ewma(&mut entry.bandwidth_mbps, observation.bandwidth_mbps);
        update_ewma(&mut entry.ttft_ms, observation.ttft_ms);
        update_ewma(&mut entry.itl_ms, observation.itl_ms);
        update_ewma(&mut entry.tokens_per_second, observation.tokens_per_second);
        update_ewma(&mut entry.kv_transfer_ms, observation.kv_transfer_ms);
        update_ewma(
            &mut entry.artifact_materialization_ms,
            observation.artifact_materialization_ms,
        );
        if let Some(cache_hit) = observation.cache_hit {
            entry.cache_observation_count = entry.cache_observation_count.saturating_add(1);
            if cache_hit {
                entry.cache_hit_count = entry.cache_hit_count.saturating_add(1);
            }
        }
        entry.last_updated = SystemTime::now();
        prune_over_capacity(&mut entries, self.config.max_targets);
    }

    pub fn observe_probe(
        &self,
        key: TargetNetworkKey,
        rtt_ms: Option<f64>,
        jitter_ms: Option<f64>,
        bandwidth_mbps: Option<f64>,
        success: bool,
    ) {
        self.observe(
            key,
            NetworkCostObservation {
                rtt_ms,
                jitter_ms,
                bandwidth_mbps,
                success,
                ..NetworkCostObservation::default()
            },
        );
    }

    pub fn snapshot_for(&self, key: &TargetNetworkKey) -> Option<NetworkCostSnapshot> {
        let mut entries = self.inner.lock().unwrap();
        prune_expired(&mut entries, self.config.ttl);
        entries.get(key).map(|entry| snapshot_entry(key, entry, self.config.ttl))
    }

    pub fn status_snapshot(&self) -> NetworkCostStatusSnapshot {
        let mut entries = self.inner.lock().unwrap();
        prune_expired(&mut entries, self.config.ttl);
        let mut targets = entries
            .iter()
            .map(|(key, entry)| snapshot_entry(key, entry, self.config.ttl))
            .collect::<Vec<_>>();
        targets.sort_by(|a, b| {
            a.stale
                .cmp(&b.stale)
                .then_with(|| a.last_updated_ms_ago.cmp(&b.last_updated_ms_ago))
                .then_with(|| a.kind.cmp(&b.kind))
                .then_with(|| a.target.cmp(&b.target))
        });
        let stale_targets = targets.iter().filter(|target| target.stale).count();
        let advisory_hints = advisory_hints_from_snapshots(&targets, self.config.ttl, 32);
        NetworkCostStatusSnapshot {
            tracked_targets: targets.len(),
            stale_targets,
            max_targets: self.config.max_targets,
            ttl_secs: self.config.ttl.as_secs(),
            advisory_hints,
            targets,
        }
    }

    pub fn advisory_hints(&self, max_hints: usize) -> Vec<NetworkAdvisoryHint> {
        let mut entries = self.inner.lock().unwrap();
        prune_expired(&mut entries, self.config.ttl);
        let mut targets = entries
            .iter()
            .map(|(key, entry)| snapshot_entry(key, entry, self.config.ttl))
            .collect::<Vec<_>>();
        targets.sort_by(|a, b| {
            a.last_updated_ms_ago
                .cmp(&b.last_updated_ms_ago)
                .then_with(|| a.kind.cmp(&b.kind))
                .then_with(|| a.target.cmp(&b.target))
        });
        advisory_hints_from_snapshots(&targets, self.config.ttl, max_hints)
    }
}

fn update_ewma(current: &mut Option<f64>, next: Option<f64>) {
    let Some(next) = next.filter(|value| value.is_finite() && *value >= 0.0) else {
        return;
    };
    *current = Some(match current {
        Some(current) => (*current * (1.0 - EWMA_ALPHA)) + (next * EWMA_ALPHA),
        None => next,
    });
}

fn prune_expired(entries: &mut HashMap<TargetNetworkKey, TargetCostEntry>, ttl: Duration) {
    let now = SystemTime::now();
    entries.retain(|_, entry| {
        now.duration_since(entry.last_updated)
            .map(|age| age <= ttl)
            .unwrap_or(true)
    });
}

fn prune_over_capacity(entries: &mut HashMap<TargetNetworkKey, TargetCostEntry>, max_targets: usize) {
    if entries.len() <= max_targets {
        return;
    }
    let mut keys = entries
        .iter()
        .map(|(key, entry)| (key.clone(), entry.last_updated))
        .collect::<Vec<_>>();
    keys.sort_by_key(|(_, last_updated)| *last_updated);
    let remove_count = entries.len().saturating_sub(max_targets);
    for (key, _) in keys.into_iter().take(remove_count) {
        entries.remove(&key);
    }
}

fn snapshot_entry(
    key: &TargetNetworkKey,
    entry: &TargetCostEntry,
    ttl: Duration,
) -> NetworkCostSnapshot {
    let now = SystemTime::now();
    let last_updated_ms_ago = now
        .duration_since(entry.last_updated)
        .map(|age| age.as_millis() as u64)
        .unwrap_or(0);
    let last_updated_unix_ms = entry
        .last_updated
        .duration_since(UNIX_EPOCH)
        .map(|age| age.as_millis() as u64)
        .unwrap_or(0);
    let cache_hit_rate = if entry.cache_observation_count == 0 {
        None
    } else {
        Some(entry.cache_hit_count as f64 / entry.cache_observation_count as f64)
    };
    NetworkCostSnapshot {
        model: key.model.clone(),
        target: key.target.clone(),
        kind: key.kind.clone(),
        sample_count: entry.sample_count,
        success_count: entry.success_count,
        failure_count: entry.failure_count,
        success_rate: if entry.sample_count == 0 {
            0.0
        } else {
            entry.success_count as f64 / entry.sample_count as f64
        },
        avg_queue_wait_ms: entry.queue_wait_ms,
        avg_attempt_ms: entry.attempt_ms,
        avg_rtt_ms: entry.rtt_ms,
        avg_jitter_ms: entry.jitter_ms,
        avg_bandwidth_mbps: entry.bandwidth_mbps,
        avg_ttft_ms: entry.ttft_ms,
        avg_itl_ms: entry.itl_ms,
        avg_tokens_per_second: entry.tokens_per_second,
        avg_kv_transfer_ms: entry.kv_transfer_ms,
        avg_artifact_materialization_ms: entry.artifact_materialization_ms,
        cache_hit_rate,
        last_updated_unix_ms,
        last_updated_ms_ago,
        stale: last_updated_ms_ago > ttl.as_millis() as u64,
    }
}

fn advisory_hints_from_snapshots(
    snapshots: &[NetworkCostSnapshot],
    ttl: Duration,
    max_hints: usize,
) -> Vec<NetworkAdvisoryHint> {
    snapshots
        .iter()
        .filter(|snapshot| !snapshot.stale)
        .take(max_hints)
        .map(|snapshot| NetworkAdvisoryHint {
            advisory_version: 1,
            model: snapshot.model.clone(),
            target: snapshot.target.clone(),
            kind: snapshot.kind.clone(),
            ttl_ms: ttl.as_millis() as u64,
            observed_unix_ms: snapshot.last_updated_unix_ms,
            success_rate: snapshot.success_rate,
            avg_queue_wait_ms: snapshot.avg_queue_wait_ms,
            avg_rtt_ms: snapshot.avg_rtt_ms,
            avg_ttft_ms: snapshot.avg_ttft_ms,
            avg_itl_ms: snapshot.avg_itl_ms,
            avg_tokens_per_second: snapshot.avg_tokens_per_second,
            cache_hit_rate: snapshot.cache_hit_rate,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_ewma_and_success_rate() {
        let collector = NetworkCostCollector::default();
        let key = TargetNetworkKey::new(Some("qwen"), "peer-a", "remote");
        collector.observe(
            key.clone(),
            NetworkCostObservation {
                attempt_ms: Some(100.0),
                success: true,
                cache_hit: Some(true),
                ..NetworkCostObservation::default()
            },
        );
        collector.observe(
            key.clone(),
            NetworkCostObservation {
                attempt_ms: Some(200.0),
                success: false,
                cache_hit: Some(false),
                ..NetworkCostObservation::default()
            },
        );

        let snapshot = collector.snapshot_for(&key).unwrap();
        assert_eq!(snapshot.sample_count, 2);
        assert_eq!(snapshot.success_count, 1);
        assert_eq!(snapshot.success_rate, 0.5);
        assert_eq!(snapshot.cache_hit_rate, Some(0.5));
        assert!(snapshot.avg_attempt_ms.unwrap() > 100.0);
        assert_eq!(collector.advisory_hints(8).len(), 1);

        collector.observe_probe(key.clone(), Some(12.0), Some(1.0), Some(1000.0), true);
        let probed = collector.snapshot_for(&key).unwrap();
        assert!(probed.avg_rtt_ms.is_some());
    }

    #[test]
    fn prunes_over_capacity() {
        let collector = NetworkCostCollector::new(NetworkCostConfig {
            max_targets: 1,
            ttl: Duration::from_secs(60),
        });
        collector.observe(
            TargetNetworkKey::new(None, "a", "remote"),
            NetworkCostObservation {
                success: true,
                ..NetworkCostObservation::default()
            },
        );
        collector.observe(
            TargetNetworkKey::new(None, "b", "remote"),
            NetworkCostObservation {
                success: true,
                ..NetworkCostObservation::default()
            },
        );
        assert_eq!(collector.status_snapshot().tracked_targets, 1);
    }
}
