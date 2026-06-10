use std::io::{self, Read, Write};

use super::{MAX_STAGE_ACTIVATION_BYTES, invalid_data, invalid_input};

pub const ACTIVATION_STRIPE_CHUNK_MAGIC: i32 = 0x5354_5249; // "STRI"
pub const ACTIVATION_STRIPE_CHUNK_VERSION: i32 = 1;
pub const MAX_STAGE_ACTIVATION_STRIPE_CHUNKS: usize = 4096;
pub const MAX_STAGE_ACTIVATION_STRIPE_CHUNK_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageActivationStripeChunk {
    pub request_id: u64,
    pub session_id: u64,
    pub frame_id: u64,
    pub chunk_index: u32,
    pub chunk_count: u32,
    pub total_bytes: u64,
    pub offset: u64,
    pub payload: Vec<u8>,
}

impl StageActivationStripeChunk {
    pub fn end_offset(&self) -> io::Result<u64> {
        self.offset
            .checked_add(
                u64::try_from(self.payload.len())
                    .map_err(|_| invalid_data("stripe chunk byte count overflow"))?,
            )
            .ok_or_else(|| invalid_data("stripe chunk byte range overflow"))
    }
}

pub fn stripe_activation_payload(
    request_id: u64,
    session_id: u64,
    frame_id: u64,
    payload: &[u8],
    max_chunk_bytes: usize,
) -> io::Result<Vec<StageActivationStripeChunk>> {
    if max_chunk_bytes == 0 {
        return Err(invalid_input(
            "stripe chunk byte limit must be greater than zero",
        ));
    }
    if max_chunk_bytes > MAX_STAGE_ACTIVATION_STRIPE_CHUNK_BYTES {
        return Err(invalid_input("stripe chunk byte limit exceeds maximum"));
    }
    if payload.len() > MAX_STAGE_ACTIVATION_BYTES {
        return Err(invalid_input(
            "striped activation byte count exceeds maximum",
        ));
    }
    if payload.is_empty() {
        return Ok(Vec::new());
    }

    let chunk_count = payload.len().div_ceil(max_chunk_bytes);
    if chunk_count > MAX_STAGE_ACTIVATION_STRIPE_CHUNKS {
        return Err(invalid_input("stripe chunk count exceeds maximum"));
    }
    let chunk_count_u32 = u32::try_from(chunk_count)
        .map_err(|_| invalid_input("stripe chunk count exceeds maximum"))?;
    let total_bytes = u64::try_from(payload.len())
        .map_err(|_| invalid_input("striped activation byte count exceeds maximum"))?;

    payload
        .chunks(max_chunk_bytes)
        .enumerate()
        .map(|(chunk_index, chunk)| {
            Ok(StageActivationStripeChunk {
                request_id,
                session_id,
                frame_id,
                chunk_index: u32::try_from(chunk_index)
                    .map_err(|_| invalid_input("stripe chunk count exceeds maximum"))?,
                chunk_count: chunk_count_u32,
                total_bytes,
                offset: u64::try_from(chunk_index.saturating_mul(max_chunk_bytes))
                    .map_err(|_| invalid_input("stripe chunk offset exceeds maximum"))?,
                payload: chunk.to_vec(),
            })
        })
        .collect()
}

pub fn write_activation_stripe_chunk(
    mut writer: impl Write,
    chunk: &StageActivationStripeChunk,
) -> io::Result<()> {
    validate_chunk(chunk)?;
    write_i32(&mut writer, ACTIVATION_STRIPE_CHUNK_MAGIC)?;
    write_i32(&mut writer, ACTIVATION_STRIPE_CHUNK_VERSION)?;
    write_u64(&mut writer, chunk.request_id)?;
    write_u64(&mut writer, chunk.session_id)?;
    write_u64(&mut writer, chunk.frame_id)?;
    write_u32(&mut writer, chunk.chunk_index)?;
    write_u32(&mut writer, chunk.chunk_count)?;
    write_u64(&mut writer, chunk.total_bytes)?;
    write_u64(&mut writer, chunk.offset)?;
    write_u32(
        &mut writer,
        u32::try_from(chunk.payload.len())
            .map_err(|_| invalid_input("stripe chunk byte count exceeds maximum"))?,
    )?;
    writer.write_all(&chunk.payload)
}

pub fn read_activation_stripe_chunk(
    mut reader: impl Read,
) -> io::Result<StageActivationStripeChunk> {
    let magic = read_i32(&mut reader)?;
    if magic != ACTIVATION_STRIPE_CHUNK_MAGIC {
        return Err(invalid_data("activation stripe chunk magic mismatch"));
    }
    let version = read_i32(&mut reader)?;
    if version != ACTIVATION_STRIPE_CHUNK_VERSION {
        return Err(invalid_data("unsupported activation stripe chunk version"));
    }
    let request_id = read_u64(&mut reader)?;
    let session_id = read_u64(&mut reader)?;
    let frame_id = read_u64(&mut reader)?;
    let chunk_index = read_u32(&mut reader)?;
    let chunk_count = read_u32(&mut reader)?;
    let total_bytes = read_u64(&mut reader)?;
    let offset = read_u64(&mut reader)?;
    let payload_len = checked_u32_len(
        read_u32(&mut reader)?,
        MAX_STAGE_ACTIVATION_STRIPE_CHUNK_BYTES,
        "stripe chunk byte count exceeds maximum",
    )?;
    let mut payload = vec![0_u8; payload_len];
    reader.read_exact(&mut payload)?;
    let chunk = StageActivationStripeChunk {
        request_id,
        session_id,
        frame_id,
        chunk_index,
        chunk_count,
        total_bytes,
        offset,
        payload,
    };
    validate_chunk(&chunk)?;
    Ok(chunk)
}

#[derive(Debug)]
pub struct StageActivationStripeReassembler {
    request_id: u64,
    session_id: u64,
    frame_id: u64,
    chunk_count: u32,
    total_bytes: u64,
    received_bytes: u64,
    received_ranges: Vec<(u64, u64)>,
    chunks: Vec<Option<StageActivationStripeChunk>>,
}

impl StageActivationStripeReassembler {
    pub fn new(first: StageActivationStripeChunk) -> io::Result<Self> {
        validate_chunk(&first)?;
        let chunk_count = first.chunk_count;
        let chunks_len = usize::try_from(chunk_count)
            .map_err(|_| invalid_data("stripe chunk count exceeds maximum"))?;
        let mut reassembler = Self {
            request_id: first.request_id,
            session_id: first.session_id,
            frame_id: first.frame_id,
            chunk_count,
            total_bytes: first.total_bytes,
            received_bytes: 0,
            received_ranges: Vec::with_capacity(chunks_len),
            chunks: vec![None; chunks_len],
        };
        reassembler.push(first)?;
        Ok(reassembler)
    }

    pub fn push(&mut self, chunk: StageActivationStripeChunk) -> io::Result<bool> {
        validate_chunk(&chunk)?;
        self.validate_matching_frame(&chunk)?;
        let index = usize::try_from(chunk.chunk_index)
            .map_err(|_| invalid_data("stripe chunk index exceeds maximum"))?;
        if self.chunks[index].is_some() {
            return Err(invalid_data("duplicate activation stripe chunk"));
        }
        let end_offset = chunk.end_offset()?;
        if self
            .received_ranges
            .iter()
            .any(|(start, end)| ranges_overlap(chunk.offset, end_offset, *start, *end))
        {
            return Err(invalid_data("overlapping activation stripe chunk"));
        }
        self.received_bytes = self
            .received_bytes
            .checked_add(
                u64::try_from(chunk.payload.len())
                    .map_err(|_| invalid_data("stripe chunk byte count overflow"))?,
            )
            .ok_or_else(|| invalid_data("striped activation byte count overflow"))?;
        self.received_ranges.push((chunk.offset, end_offset));
        self.chunks[index] = Some(chunk);
        Ok(self.is_complete())
    }

    pub fn is_complete(&self) -> bool {
        self.received_bytes == self.total_bytes && self.chunks.iter().all(Option::is_some)
    }

    pub fn finish(self) -> io::Result<Vec<u8>> {
        if !self.is_complete() {
            return Err(invalid_data("activation stripe frame is incomplete"));
        }
        let total_len = usize::try_from(self.total_bytes)
            .map_err(|_| invalid_data("striped activation byte count exceeds maximum"))?;
        let mut payload = vec![0_u8; total_len];
        for chunk in self.chunks.into_iter().flatten() {
            let start = usize::try_from(chunk.offset)
                .map_err(|_| invalid_data("stripe chunk offset exceeds maximum"))?;
            let end = start
                .checked_add(chunk.payload.len())
                .ok_or_else(|| invalid_data("stripe chunk byte range overflow"))?;
            if end > payload.len() {
                return Err(invalid_data("stripe chunk byte range exceeds frame"));
            }
            payload[start..end].copy_from_slice(&chunk.payload);
        }
        Ok(payload)
    }

    fn validate_matching_frame(&self, chunk: &StageActivationStripeChunk) -> io::Result<()> {
        if chunk.request_id != self.request_id
            || chunk.session_id != self.session_id
            || chunk.frame_id != self.frame_id
            || chunk.chunk_count != self.chunk_count
            || chunk.total_bytes != self.total_bytes
        {
            return Err(invalid_data(
                "activation stripe chunk belongs to a different frame",
            ));
        }
        Ok(())
    }
}

fn validate_chunk(chunk: &StageActivationStripeChunk) -> io::Result<()> {
    let chunk_count = checked_u32_len(
        chunk.chunk_count,
        MAX_STAGE_ACTIVATION_STRIPE_CHUNKS,
        "stripe chunk count exceeds maximum",
    )?;
    if chunk_count == 0 {
        return Err(invalid_data("stripe chunk count must be greater than zero"));
    }
    let chunk_index = usize::try_from(chunk.chunk_index)
        .map_err(|_| invalid_data("stripe chunk index exceeds maximum"))?;
    if chunk_index >= chunk_count {
        return Err(invalid_data("stripe chunk index exceeds count"));
    }
    if chunk.payload.is_empty() {
        return Err(invalid_data("stripe chunk payload is empty"));
    }
    if chunk.payload.len() > MAX_STAGE_ACTIVATION_STRIPE_CHUNK_BYTES {
        return Err(invalid_data("stripe chunk byte count exceeds maximum"));
    }
    let total_bytes = usize::try_from(chunk.total_bytes)
        .map_err(|_| invalid_data("striped activation byte count exceeds maximum"))?;
    if total_bytes > MAX_STAGE_ACTIVATION_BYTES {
        return Err(invalid_data(
            "striped activation byte count exceeds maximum",
        ));
    }
    if chunk.end_offset()? > chunk.total_bytes {
        return Err(invalid_data("stripe chunk byte range exceeds frame"));
    }
    Ok(())
}

fn ranges_overlap(left_start: u64, left_end: u64, right_start: u64, right_end: u64) -> bool {
    left_start < right_end && right_start < left_end
}

fn checked_u32_len(value: u32, max: usize, too_large_message: &'static str) -> io::Result<usize> {
    let value = usize::try_from(value).map_err(|_| invalid_data(too_large_message))?;
    if value > max {
        return Err(invalid_data(too_large_message));
    }
    Ok(value)
}

fn write_i32(mut writer: impl Write, value: i32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u32(mut writer: impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64(mut writer: impl Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_i32(mut reader: impl Read) -> io::Result<i32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u32(mut reader: impl Read) -> io::Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(mut reader: impl Read) -> io::Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn activation_stripe_chunks_round_trip_and_reassemble_out_of_order() {
        let payload = (0..257).map(|value| value as u8).collect::<Vec<_>>();
        let chunks = stripe_activation_payload(7, 11, 13, &payload, 64).unwrap();
        assert_eq!(chunks.len(), 5);

        let mut encoded = Vec::new();
        write_activation_stripe_chunk(&mut encoded, &chunks[3]).unwrap();
        let decoded = read_activation_stripe_chunk(Cursor::new(encoded)).unwrap();
        assert_eq!(decoded, chunks[3]);

        let mut reassembler = StageActivationStripeReassembler::new(chunks[3].clone()).unwrap();
        assert!(!reassembler.is_complete());
        for index in [0, 4, 1, 2] {
            reassembler.push(chunks[index].clone()).unwrap();
        }
        assert_eq!(reassembler.finish().unwrap(), payload);
    }

    #[test]
    fn activation_stripe_rejects_duplicate_chunks() {
        let payload = vec![1_u8; 128];
        let chunks = stripe_activation_payload(7, 11, 13, &payload, 64).unwrap();
        let mut reassembler = StageActivationStripeReassembler::new(chunks[0].clone()).unwrap();

        let error = reassembler
            .push(chunks[0].clone())
            .expect_err("duplicate chunk must fail");
        assert_eq!(error.to_string(), "duplicate activation stripe chunk");
    }

    #[test]
    fn activation_stripe_rejects_overlapping_chunks() {
        let payload = vec![1_u8; 128];
        let mut chunks = stripe_activation_payload(7, 11, 13, &payload, 64).unwrap();
        chunks[1].offset = 32;
        let mut reassembler = StageActivationStripeReassembler::new(chunks[0].clone()).unwrap();

        let error = reassembler
            .push(chunks[1].clone())
            .expect_err("overlapping chunk must fail");
        assert_eq!(error.to_string(), "overlapping activation stripe chunk");
    }

    #[test]
    fn activation_stripe_rejects_mismatched_frames() {
        let payload = vec![1_u8; 128];
        let chunks = stripe_activation_payload(7, 11, 13, &payload, 64).unwrap();
        let mut wrong_frame = chunks[1].clone();
        wrong_frame.frame_id = 99;
        let mut reassembler = StageActivationStripeReassembler::new(chunks[0].clone()).unwrap();

        let error = reassembler
            .push(wrong_frame)
            .expect_err("mismatched frame must fail");
        assert_eq!(
            error.to_string(),
            "activation stripe chunk belongs to a different frame"
        );
    }

    #[test]
    fn activation_stripe_rejects_incomplete_finish() {
        let payload = vec![1_u8; 128];
        let chunks = stripe_activation_payload(7, 11, 13, &payload, 64).unwrap();
        let reassembler = StageActivationStripeReassembler::new(chunks[0].clone()).unwrap();

        let error = reassembler
            .finish()
            .expect_err("incomplete frame must fail");
        assert_eq!(error.to_string(), "activation stripe frame is incomplete");
    }
}
