use std::{
    io,
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::Duration,
};

use anyhow::{Result, bail};
use skippy_protocol::binary::{StageWireMessage, WireActivationDType, write_stage_message};

#[derive(Clone, Copy, Debug)]
pub struct WireCondition {
    delay_ms: f64,
    jitter_ms: f64,
    mbps: Option<f64>,
}

static JITTER_STATE: AtomicU64 = AtomicU64::new(0x9e37_79b9_7f4a_7c15);

impl WireCondition {
    pub fn new(delay_ms: f64, mbps: Option<f64>) -> Result<Self> {
        Self::with_jitter(delay_ms, 0.0, mbps)
    }

    pub fn with_jitter(delay_ms: f64, jitter_ms: f64, mbps: Option<f64>) -> Result<Self> {
        if delay_ms < 0.0 {
            bail!("downstream wire delay must not be negative");
        }
        if jitter_ms < 0.0 {
            bail!("downstream wire jitter must not be negative");
        }
        if mbps.is_some_and(|value| value <= 0.0) {
            bail!("downstream wire mbps must be greater than zero");
        }
        Ok(Self {
            delay_ms,
            jitter_ms,
            mbps,
        })
    }

    pub fn enabled(&self) -> bool {
        self.delay_ms > 0.0 || self.jitter_ms > 0.0 || self.mbps.is_some()
    }

    pub fn delay_ms(&self) -> f64 {
        self.delay_ms
    }

    pub fn jitter_ms(&self) -> f64 {
        self.jitter_ms
    }

    pub fn mbps(&self) -> Option<f64> {
        self.mbps
    }

    fn sleep_for(&self, message: &StageWireMessage) {
        let delay_seconds = self.delay_ms / 1000.0;
        let jitter_seconds = next_jitter_unit() * self.jitter_ms / 1000.0;
        let bandwidth_seconds = self
            .mbps
            .map(|mbps| message.estimated_wire_bytes() as f64 / (mbps * 125_000.0))
            .unwrap_or(0.0);
        let seconds = delay_seconds + jitter_seconds + bandwidth_seconds;
        if seconds > 0.0 {
            thread::sleep(Duration::from_secs_f64(seconds));
        }
    }
}

fn next_jitter_unit() -> f64 {
    let mut value = JITTER_STATE.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value >> 11) as f64 / ((1_u64 << 53) as f64)
}

pub(crate) fn write_stage_message_conditioned(
    writer: impl io::Write,
    message: &StageWireMessage,
    dtype: WireActivationDType,
    condition: WireCondition,
) -> io::Result<()> {
    condition.sleep_for(message);
    write_stage_message(writer, message, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_condition_new_defaults_jitter_to_zero() {
        let condition = WireCondition::new(12.0, Some(100.0)).unwrap();

        assert_eq!(condition.delay_ms(), 12.0);
        assert_eq!(condition.jitter_ms(), 0.0);
        assert_eq!(condition.mbps(), Some(100.0));
    }

    #[test]
    fn wire_condition_accepts_jitter() {
        let condition = WireCondition::with_jitter(12.0, 8.0, None).unwrap();

        assert!(condition.enabled());
        assert_eq!(condition.delay_ms(), 12.0);
        assert_eq!(condition.jitter_ms(), 8.0);
        assert_eq!(condition.mbps(), None);
    }

    #[test]
    fn wire_condition_rejects_invalid_values() {
        assert!(WireCondition::with_jitter(-1.0, 0.0, None).is_err());
        assert!(WireCondition::with_jitter(0.0, -1.0, None).is_err());
        assert!(WireCondition::with_jitter(0.0, 0.0, Some(0.0)).is_err());
    }
}
