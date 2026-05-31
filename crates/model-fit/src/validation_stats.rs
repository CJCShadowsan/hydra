use std::cmp::Ordering;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ThroughputSampleStats {
    pub raw_sample_count: usize,
    pub raw_median: Option<f64>,
    pub raw_min: Option<f64>,
    pub raw_max: Option<f64>,
    pub raw_spread: Option<f64>,
    pub clean_sample_count: usize,
    pub clean_median: Option<f64>,
    pub clean_min: Option<f64>,
    pub clean_max: Option<f64>,
    pub clean_spread: Option<f64>,
    pub outlier_count: usize,
}

pub fn throughput_sample_stats(samples: &[f64], max_spread: f64) -> ThroughputSampleStats {
    let raw_samples = sorted_positive_finite_samples(samples);
    let Some(raw) = BasicStats::from_sorted(&raw_samples) else {
        return ThroughputSampleStats::default();
    };
    let clean = denoise_samples(&raw_samples, max_spread).unwrap_or_else(|| raw.clone());
    ThroughputSampleStats {
        raw_sample_count: raw_samples.len(),
        raw_median: Some(raw.median),
        raw_min: Some(raw.min),
        raw_max: Some(raw.max),
        raw_spread: Some(raw.spread),
        clean_sample_count: clean.sample_count,
        clean_median: Some(clean.median),
        clean_min: Some(clean.min),
        clean_max: Some(clean.max),
        clean_spread: Some(clean.spread),
        outlier_count: raw_samples.len().saturating_sub(clean.sample_count),
    }
}

fn sorted_positive_finite_samples(samples: &[f64]) -> Vec<f64> {
    let mut samples = samples
        .iter()
        .copied()
        .filter(|sample| sample.is_finite() && *sample > 0.0)
        .collect::<Vec<_>>();
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    samples
}

fn denoise_samples(samples: &[f64], max_spread: f64) -> Option<BasicStats> {
    let raw = BasicStats::from_sorted(samples)?;
    if samples.len() < 3 || raw.spread <= max_spread {
        return None;
    }

    let mut best = None;
    for drop_index in 0..samples.len() {
        let mut candidate = Vec::with_capacity(samples.len() - 1);
        candidate.extend_from_slice(&samples[..drop_index]);
        candidate.extend_from_slice(&samples[drop_index + 1..]);
        let Some(stats) = BasicStats::from_sorted(&candidate) else {
            continue;
        };
        if removed_sample_is_outlier(samples[drop_index], stats.median, max_spread)
            && stats.spread <= max_spread
            && best
                .as_ref()
                .is_none_or(|best_stats: &BasicStats| stats.spread < best_stats.spread)
        {
            best = Some(stats);
        }
    }
    best
}

fn removed_sample_is_outlier(sample: f64, clean_median: f64, max_spread: f64) -> bool {
    if clean_median <= 0.0 {
        return false;
    }
    let relative_delta = (sample - clean_median).abs() / clean_median;
    relative_delta > max_spread.max(0.05)
}

#[derive(Clone, Debug, PartialEq)]
struct BasicStats {
    sample_count: usize,
    median: f64,
    min: f64,
    max: f64,
    spread: f64,
}

impl BasicStats {
    fn from_sorted(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let median = median(samples);
        let min = samples[0];
        let max = *samples.last().expect("samples is not empty");
        let spread = if median > 0.0 {
            (max - min) / median
        } else {
            0.0
        };
        Some(Self {
            sample_count: samples.len(),
            median,
            min,
            max,
            spread,
        })
    }
}

fn median(samples: &[f64]) -> f64 {
    let mid = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[mid - 1] + samples[mid]) / 2.0
    } else {
        samples[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_single_cold_decode_outlier() {
        let stats = throughput_sample_stats(&[26.7, 51.2, 51.6], 0.10);

        assert_eq!(stats.raw_sample_count, 3);
        assert_eq!(stats.clean_sample_count, 2);
        assert_eq!(stats.outlier_count, 1);
        assert!((stats.clean_median.unwrap() - 51.4).abs() < 0.0001);
        assert!(stats.clean_spread.unwrap() < 0.01);
    }

    #[test]
    fn keeps_samples_when_no_single_drop_makes_them_stable() {
        let stats = throughput_sample_stats(&[20.0, 30.0, 45.0], 0.10);

        assert_eq!(stats.raw_sample_count, 3);
        assert_eq!(stats.clean_sample_count, 3);
        assert_eq!(stats.outlier_count, 0);
        assert!(stats.clean_spread.unwrap() > 0.10);
    }
}
