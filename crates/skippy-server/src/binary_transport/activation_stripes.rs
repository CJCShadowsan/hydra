use std::{
    collections::BTreeMap,
    io,
    net::TcpStream,
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use skippy_protocol::binary::{StageActivationStripeReassembler, read_activation_stripe_chunk};

#[derive(Clone, Default)]
pub(crate) struct ActivationStripeHub {
    inner: Arc<(Mutex<ActivationStripeState>, Condvar)>,
}

#[derive(Default)]
struct ActivationStripeState {
    frames: BTreeMap<ActivationStripeFrameKey, StageActivationStripeReassembler>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct ActivationStripeFrameKey {
    request_id: u64,
    session_id: u64,
    frame_id: u64,
}

impl ActivationStripeFrameKey {
    pub(crate) fn new(request_id: u64, session_id: u64, frame_id: u64) -> Self {
        Self {
            request_id,
            session_id,
            frame_id,
        }
    }
}

impl ActivationStripeHub {
    pub(crate) fn push_stream_chunks(&self, mut stream: TcpStream) -> Result<()> {
        loop {
            match read_activation_stripe_chunk(&mut stream) {
                Ok(chunk) => self.push_chunk(chunk)?,
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
                Err(error) => return Err(error).context("read activation stripe chunk"),
            }
        }
    }

    pub(crate) fn take_complete(
        &self,
        key: ActivationStripeFrameKey,
        timeout: Duration,
    ) -> Result<Vec<u8>> {
        let started = Instant::now();
        let (lock, ready) = &*self.inner;
        let mut state = lock
            .lock()
            .map_err(|_| anyhow!("activation stripe hub lock poisoned"))?;
        loop {
            if state
                .frames
                .get(&key)
                .is_some_and(StageActivationStripeReassembler::is_complete)
            {
                let frame = state
                    .frames
                    .remove(&key)
                    .context("missing complete striped activation frame")?;
                return frame.finish().map_err(Into::into);
            }

            let elapsed = started.elapsed();
            if elapsed >= timeout {
                bail!(
                    "timed out waiting for striped activation frame request={} session={} frame={}",
                    key.request_id,
                    key.session_id,
                    key.frame_id
                );
            }
            let remaining = timeout.saturating_sub(elapsed);
            let wait = ready
                .wait_timeout(state, remaining)
                .map_err(|_| anyhow!("activation stripe hub lock poisoned"))?;
            state = wait.0;
        }
    }

    fn push_chunk(&self, chunk: skippy_protocol::binary::StageActivationStripeChunk) -> Result<()> {
        let key = ActivationStripeFrameKey::new(chunk.request_id, chunk.session_id, chunk.frame_id);
        let (lock, ready) = &*self.inner;
        let mut state = lock
            .lock()
            .map_err(|_| anyhow!("activation stripe hub lock poisoned"))?;
        if let Some(frame) = state.frames.get_mut(&key) {
            frame.push(chunk)?;
        } else {
            state
                .frames
                .insert(key, StageActivationStripeReassembler::new(chunk)?);
        }
        ready.notify_all();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::binary::stripe_activation_payload;

    #[test]
    fn hub_reassembles_complete_frame() {
        let hub = ActivationStripeHub::default();
        let payload = (0..128).map(|value| value as u8).collect::<Vec<_>>();
        let mut chunks = stripe_activation_payload(7, 11, 13, &payload, 32).unwrap();
        chunks.reverse();

        for chunk in chunks {
            hub.push_chunk(chunk).unwrap();
        }

        let reassembled = hub
            .take_complete(
                ActivationStripeFrameKey::new(7, 11, 13),
                Duration::from_millis(10),
            )
            .unwrap();
        assert_eq!(reassembled, payload);
    }
}
