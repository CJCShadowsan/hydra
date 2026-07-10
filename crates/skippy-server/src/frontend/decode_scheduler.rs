use std::collections::VecDeque;

use super::{OpenAiError, OpenAiResult};

const PIPELINE_DEPTH_ENV: &str = "SKIPPY_VERIFY_WINDOW_PIPELINE_DEPTH";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VerifyWindowPipelineConfig {
    depth: usize,
}

impl VerifyWindowPipelineConfig {
    pub(super) fn from_env() -> Self {
        Self {
            depth: std::env::var(PIPELINE_DEPTH_ENV)
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(1)
                .max(1),
        }
    }

    pub(super) fn enabled(self) -> bool {
        self.depth > 1
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct VerifyWindow {
    pub(super) id: i32,
    pub(super) base_position: usize,
    pub(super) decode_step: usize,
}

#[derive(Debug)]
pub(super) struct VerifyWindowScheduler {
    config: VerifyWindowPipelineConfig,
    next_id: i32,
    in_flight: VecDeque<VerifyWindow>,
    stale_discard_count: usize,
}

impl VerifyWindowScheduler {
    pub(super) fn new(config: VerifyWindowPipelineConfig) -> Self {
        Self {
            config,
            next_id: 1,
            in_flight: VecDeque::new(),
            stale_discard_count: 0,
        }
    }

    pub(super) fn enabled(&self) -> bool {
        self.config.enabled()
    }

    pub(super) fn has_capacity(&self) -> bool {
        self.in_flight.len() < self.config.depth
    }

    pub(super) fn open(
        &mut self,
        base_position: usize,
        decode_step: usize,
    ) -> OpenAiResult<VerifyWindow> {
        if !self.has_capacity() {
            return Err(OpenAiError::backend(
                "verify window pipeline depth exceeded",
            ));
        }
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| OpenAiError::backend("verify window id overflow"))?;
        let window = VerifyWindow {
            id,
            base_position,
            decode_step,
        };
        self.in_flight.push_back(window.clone());
        Ok(window)
    }

    pub(super) fn complete_next(&mut self, reply_window_id: i32) -> OpenAiResult<VerifyWindow> {
        let Some(window) = self.in_flight.front() else {
            return Err(OpenAiError::backend(
                "verify window reply arrived with no in-flight window",
            ));
        };
        if window.id != reply_window_id {
            return Err(OpenAiError::backend(format!(
                "verify window reply out of order: got {reply_window_id}, expected {}",
                window.id
            )));
        }
        Ok(self.in_flight.pop_front().expect("checked non-empty queue"))
    }

    #[cfg(test)]
    pub(super) fn discard_stale(&mut self) -> usize {
        let discarded = self.in_flight.len();
        self.in_flight.clear();
        self.stale_discard_count = self.stale_discard_count.saturating_add(discarded);
        discarded
    }

    pub(super) fn record_stale_discarded(&mut self, count: usize) {
        self.stale_discard_count = self.stale_discard_count.saturating_add(count);
    }

    pub(super) fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }

    pub(super) fn stale_discard_count(&self) -> usize {
        self.stale_discard_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_depth_and_requires_fifo_reply_ids() {
        let config = VerifyWindowPipelineConfig { depth: 2 };
        let mut scheduler = VerifyWindowScheduler::new(config);
        let first = scheduler.open(10, 0).unwrap();
        let second = scheduler.open(11, 1).unwrap();

        assert!(scheduler.open(12, 2).is_err());
        assert!(scheduler.complete_next(second.id).is_err());
        assert_eq!(scheduler.in_flight_len(), 2);
        assert_eq!(scheduler.complete_next(first.id).unwrap(), first);
        assert_eq!(scheduler.complete_next(second.id).unwrap(), second);
        assert_eq!(first.id, 1);
    }

    #[test]
    fn discards_stale_windows_after_divergence() {
        let config = VerifyWindowPipelineConfig { depth: 3 };
        let mut scheduler = VerifyWindowScheduler::new(config);
        scheduler.open(10, 0).unwrap();
        scheduler.open(11, 1).unwrap();
        scheduler.open(12, 2).unwrap();

        assert_eq!(scheduler.discard_stale(), 3);
        assert_eq!(scheduler.stale_discard_count(), 3);
        assert_eq!(scheduler.in_flight_len(), 0);
    }
}
