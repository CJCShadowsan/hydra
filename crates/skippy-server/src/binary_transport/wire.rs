use std::{io, net::TcpStream, thread, time::Duration};

use anyhow::{Result, bail};
use skippy_protocol::binary::{
    StageWireMessage, WireActivationDType, state_flags, stripe_activation_payload,
    write_activation_stripe_chunk, write_stage_message,
};
use std::io::Error;

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

pub(crate) fn should_stripe_activation(message: &StageWireMessage, threshold_bytes: usize) -> bool {
    threshold_bytes > 0
        && message.kind.is_prefill()
        && message.state.source_stage_index >= 0
        && message.activation.len() >= threshold_bytes
}

pub(crate) fn write_stage_message_striped_tcp(
    control_writer: &mut TcpStream,
    stripe_writers: &mut [TcpStream],
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
    let (control, chunks) = striped_control_and_chunks(message, frame_id, max_chunk_bytes)?;
    write_stage_message_conditioned(control_writer, &control, dtype, condition)?;

    let mut groups = vec![Vec::new(); stripe_writers.len()];
    for (index, chunk) in chunks.iter().cloned().enumerate() {
        let writer_index = index % stripe_writers.len();
        groups[writer_index].push(chunk);
    }
    thread::scope(|scope| -> io::Result<()> {
        let mut handles = Vec::with_capacity(stripe_writers.len());
        for (writer, group) in stripe_writers.iter_mut().zip(groups) {
            handles.push(scope.spawn(move || -> io::Result<()> {
                for chunk in &group {
                    write_activation_stripe_chunk(&mut *writer, chunk)?;
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| Error::other("activation stripe writer panicked"))??;
        }
        Ok(())
    })?;
    Ok(chunks.len())
}

#[cfg(test)]
fn write_stage_message_striped_buffered(
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
    let (control, chunks) = striped_control_and_chunks(message, frame_id, max_chunk_bytes)?;
    write_stage_message_conditioned(control_writer, &control, dtype, condition)?;

    for (index, chunk) in chunks.iter().enumerate() {
        let writer_index = index % stripe_writers.len();
        write_activation_stripe_chunk(&mut stripe_writers[writer_index], chunk)?;
    }
    Ok(chunks.len())
}

fn striped_control_and_chunks(
    message: &StageWireMessage,
    frame_id: u64,
    max_chunk_bytes: usize,
) -> io::Result<(
    StageWireMessage,
    Vec<skippy_protocol::binary::StageActivationStripeChunk>,
)> {
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
    control.state.checkpoint_generation = i32::try_from(frame_id)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "stripe frame id exceeds i32"))?;
    Ok((control, chunks))
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

        let chunk_count = write_stage_message_striped_buffered(
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
        assert_eq!(control_message.state.checkpoint_generation, 99);
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
