use std::{io, thread, time::Duration};

use anyhow::{Result, bail};
use skippy_protocol::binary::{StageWireMessage, WireActivationDType, write_stage_message};

#[cfg(test)]
use skippy_protocol::binary::{
    state_flags, stripe_activation_payload, write_activation_stripe_chunk,
};

#[derive(Clone, Copy, Debug)]
pub struct WireCondition {
    delay_ms: f64,
    mbps: Option<f64>,
}

impl WireCondition {
    pub fn new(delay_ms: f64, mbps: Option<f64>) -> Result<Self> {
        if delay_ms < 0.0 {
            bail!("downstream wire delay must not be negative");
        }
        if mbps.is_some_and(|value| value <= 0.0) {
            bail!("downstream wire mbps must be greater than zero");
        }
        Ok(Self { delay_ms, mbps })
    }

    fn sleep_for(&self, message: &StageWireMessage) {
        let delay_seconds = self.delay_ms / 1000.0;
        let bandwidth_seconds = self
            .mbps
            .map(|mbps| message.estimated_wire_bytes() as f64 / (mbps * 125_000.0))
            .unwrap_or(0.0);
        let seconds = delay_seconds + bandwidth_seconds;
        if seconds > 0.0 {
            thread::sleep(Duration::from_secs_f64(seconds));
        }
    }
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
fn should_stripe_activation(message: &StageWireMessage, threshold_bytes: usize) -> bool {
    threshold_bytes > 0
        && message.kind.is_prefill()
        && message.state.source_stage_index >= 0
        && message.activation.len() >= threshold_bytes
}

#[cfg(test)]
fn write_stage_message_striped(
    control_writer: &mut dyn io::Write,
    stripe_writers: &mut [&mut dyn io::Write],
    message: &StageWireMessage,
    dtype: WireActivationDType,
    frame_id: u64,
    max_chunk_bytes: usize,
    condition: WireCondition,
) -> io::Result<usize> {
    if stripe_writers.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "striped activation requires at least one stripe writer",
        ));
    }
    let chunks = stripe_activation_payload(
        message.request_id,
        message.session_id,
        frame_id,
        &message.activation,
        max_chunk_bytes,
    )?;
    let mut control = message.clone();
    control.activation.clear();
    control.state.flags |= state_flags::STRIPED_ACTIVATION;
    write_stage_message_conditioned(control_writer, &control, dtype, condition)?;

    for (index, chunk) in chunks.iter().enumerate() {
        let writer_index = index % stripe_writers.len();
        write_activation_stripe_chunk(&mut stripe_writers[writer_index], chunk)?;
    }
    Ok(chunks.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::binary::{
        StageStateHeader, WireMessageKind, read_activation_stripe_chunk, read_stage_message,
    };
    use std::io::Cursor;

    fn prefill_message(activation: Vec<u8>) -> StageWireMessage {
        let mut state =
            StageStateHeader::new(WireMessageKind::PrefillEmbd, WireActivationDType::F32);
        state.source_stage_index = 0;
        StageWireMessage {
            kind: WireMessageKind::PrefillEmbd,
            pos_start: 0,
            token_count: 4,
            state,
            request_id: 7,
            session_id: 11,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![1, 2, 3, 4],
            positions: Vec::new(),
            activation,
            raw_bytes: Vec::new(),
        }
    }

    #[test]
    fn should_stripe_only_large_prefill_activations_from_upstream_stage() {
        assert!(should_stripe_activation(&prefill_message(vec![1; 128]), 64));
        assert!(!should_stripe_activation(&prefill_message(vec![1; 32]), 64));

        let mut decode = prefill_message(vec![1; 128]);
        decode.kind = WireMessageKind::DecodeEmbd;
        assert!(!should_stripe_activation(&decode, 64));

        let mut driver_origin = prefill_message(vec![1; 128]);
        driver_origin.state.source_stage_index = -1;
        assert!(!should_stripe_activation(&driver_origin, 64));
    }

    #[test]
    fn striped_writer_emits_control_message_and_round_robin_chunks() {
        let message = prefill_message((0..128).map(|value| value as u8).collect());
        let mut control = Vec::new();
        let mut stripe_a = Vec::new();
        let mut stripe_b = Vec::new();
        let mut writers: Vec<&mut dyn io::Write> = vec![&mut stripe_a, &mut stripe_b];

        let chunk_count = write_stage_message_striped(
            &mut control,
            &mut writers,
            &message,
            WireActivationDType::F32,
            99,
            32,
            WireCondition::new(0.0, None).unwrap(),
        )
        .unwrap();

        assert_eq!(chunk_count, 4);
        let control_message = read_stage_message(Cursor::new(control), 4).unwrap();
        assert!(control_message.state.uses_striped_activation());
        assert!(control_message.activation.is_empty());
        assert_eq!(control_message.tokens, vec![1, 2, 3, 4]);

        let first_a = read_activation_stripe_chunk(Cursor::new(&stripe_a)).unwrap();
        let first_b = read_activation_stripe_chunk(Cursor::new(&stripe_b)).unwrap();
        assert_eq!(first_a.chunk_index, 0);
        assert_eq!(first_b.chunk_index, 1);
        assert_eq!(first_a.frame_id, 99);
        assert_eq!(first_b.frame_id, 99);
    }
}
