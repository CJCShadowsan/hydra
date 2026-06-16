use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};

use crate::{ActivationDesc, ActivationFrame, RuntimeActivationDType, RuntimeActivationLayout};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q6_K: u32 = 14;
const MAX_GGUF_STRING_BYTES: u64 = 1_000_000;
const MAX_GGUF_ARRAY_ELEMENTS: u64 = 1_000_000;
const MAX_GGUF_ARRAY_DEPTH: u32 = 64;
const MAX_GGUF_TENSOR_DIMS: u32 = 8;
const MAX_GGUF_HEADER_COUNT: usize = 1_000_000;
const QK_K: usize = 256;
const Q6_K_BLOCK_BYTES: usize = 210;
const TOKEN_EMBD_TENSOR: &str = "token_embd.weight";

#[derive(Debug, Clone)]
pub struct GgufTokenEmbeddingTable {
    path: PathBuf,
    data_offset: u64,
    hidden_size: usize,
    vocab_size: usize,
    row_size: u64,
    tensor_type: GgmlTensorType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgmlTensorType {
    F32,
    F16,
    Q6K,
}

#[derive(Debug)]
struct GgufTensorInfo {
    name: String,
    dims: Vec<u64>,
    tensor_type: u32,
    offset: u64,
}

impl GgufTokenEmbeddingTable {
    pub fn open(
        path: impl AsRef<Path>,
        expected_hidden_size: usize,
        expected_vocab_size: usize,
    ) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path).with_context(|| format!("open GGUF {}", path.display()))?;
        let header = read_gguf_header(&mut file)?;
        let tensor = header
            .tensors
            .into_iter()
            .find(|tensor| tensor.name == TOKEN_EMBD_TENSOR)
            .context("GGUF token_embd.weight tensor not found")?;
        let hidden_size = usize::try_from(*tensor.dims.first().unwrap_or(&0))
            .context("GGUF token embedding hidden size exceeds usize")?;
        let vocab_size = usize::try_from(*tensor.dims.get(1).unwrap_or(&0))
            .context("GGUF token embedding vocab size exceeds usize")?;
        if tensor.dims.len() != 2 {
            bail!(
                "GGUF token_embd.weight has {} dims; expected 2",
                tensor.dims.len()
            );
        }
        if hidden_size != expected_hidden_size {
            bail!(
                "GGUF token_embd.weight hidden size {hidden_size} does not match expected {expected_hidden_size}"
            );
        }
        if vocab_size < expected_vocab_size {
            bail!(
                "GGUF token_embd.weight vocab size {vocab_size} is smaller than expected {expected_vocab_size}"
            );
        }
        let tensor_type = GgmlTensorType::from_ggml_type(tensor.tensor_type)?;
        let row_size = tensor_type.row_size(hidden_size)?;
        Ok(Self {
            path: path.to_path_buf(),
            data_offset: header
                .data_start
                .checked_add(tensor.offset)
                .context("GGUF token embedding tensor offset overflow")?,
            hidden_size,
            vocab_size,
            row_size,
            tensor_type,
        })
    }

    pub fn frame_for_positions(
        &self,
        context_tokens: &[i32],
        row_positions: &[i64],
    ) -> Result<ActivationFrame> {
        let positions = row_positions
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("negative GGUF h0 row position")?;
        let Some(token_count) = positions.iter().copied().max().map(|position| position + 1) else {
            bail!("GGUF h0 frame requires at least one row position");
        };
        let payload_len = token_count
            .checked_mul(self.hidden_size)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .context("GGUF h0 frame payload size overflow")?;
        let mut file =
            File::open(&self.path).with_context(|| format!("open GGUF {}", self.path.display()))?;
        let mut payload = vec![0_u8; payload_len];
        for position in positions {
            let token = *context_tokens
                .get(position)
                .with_context(|| format!("GGUF h0 row position {position} is outside context"))?;
            let row = self.read_row(&mut file, token)?;
            let offset = position
                .checked_mul(self.hidden_size)
                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                .context("GGUF h0 frame row offset overflow")?;
            write_f32_row_le(&row, &mut payload[offset..offset + self.hidden_size * 4]);
        }
        Ok(ActivationFrame {
            desc: ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F32,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: 0,
                layer_start: 0,
                layer_end: 0,
                token_count: u32::try_from(token_count)
                    .context("GGUF h0 token_count exceeds u32")?,
                sequence_count: 1,
                payload_bytes: u64::try_from(payload.len())
                    .context("GGUF h0 payload bytes exceed u64")?,
                flags: 0,
            },
            payload,
        })
    }

    fn read_row(&self, file: &mut File, token: i32) -> Result<Vec<f32>> {
        let token = usize::try_from(token).context("negative token id for GGUF h0 embedding")?;
        if token >= self.vocab_size {
            bail!(
                "token id {token} is outside GGUF token embedding vocab {}",
                self.vocab_size
            );
        }
        let offset = self
            .data_offset
            .checked_add(
                u64::try_from(token)
                    .context("token id exceeds u64")?
                    .checked_mul(self.row_size)
                    .context("GGUF token embedding row offset overflow")?,
            )
            .context("GGUF token embedding data offset overflow")?;
        file.seek(SeekFrom::Start(offset))
            .context("seek GGUF token embedding row")?;
        match self.tensor_type {
            GgmlTensorType::F32 => read_f32_row(file, self.hidden_size),
            GgmlTensorType::F16 => read_f16_row(file, self.hidden_size),
            GgmlTensorType::Q6K => read_q6_k_row(file, self.hidden_size),
        }
    }
}

impl GgmlTensorType {
    fn from_ggml_type(value: u32) -> Result<Self> {
        match value {
            GGML_TYPE_F32 => Ok(Self::F32),
            GGML_TYPE_F16 => Ok(Self::F16),
            GGML_TYPE_Q6_K => Ok(Self::Q6K),
            _ => bail!("unsupported GGUF token embedding tensor type {value}"),
        }
    }

    fn row_size(self, hidden_size: usize) -> Result<u64> {
        let bytes = match self {
            Self::F32 => hidden_size
                .checked_mul(4)
                .context("GGUF f32 row size overflow")?,
            Self::F16 => hidden_size
                .checked_mul(2)
                .context("GGUF f16 row size overflow")?,
            Self::Q6K => {
                if !hidden_size.is_multiple_of(QK_K) {
                    bail!(
                        "Q6_K token embedding hidden size {hidden_size} is not a multiple of {QK_K}"
                    );
                }
                hidden_size
                    .checked_div(QK_K)
                    .and_then(|blocks| blocks.checked_mul(Q6_K_BLOCK_BYTES))
                    .context("GGUF Q6_K row size overflow")?
            }
        };
        u64::try_from(bytes).context("GGUF row size exceeds u64")
    }
}

#[derive(Debug)]
struct GgufHeader {
    data_start: u64,
    tensors: Vec<GgufTensorInfo>,
}

fn read_gguf_header(file: &mut File) -> Result<GgufHeader> {
    let mut magic = [0_u8; 4];
    file.read_exact(&mut magic).context("read GGUF magic")?;
    if &magic != GGUF_MAGIC {
        bail!("not a GGUF file");
    }
    let version = read_u32(file).context("read GGUF version")?;
    if version < 2 {
        bail!("unsupported GGUF version {version}");
    }
    let tensor_count = read_bounded_i64_count(file, "tensor count")?;
    let kv_count = read_bounded_i64_count(file, "KV count")?;
    let mut alignment = 32_u32;
    for _ in 0..kv_count {
        let key = read_gguf_string(file)?;
        let value_type = read_u32(file).context("read GGUF KV type")?;
        if key == "general.alignment" && value_type == GGUF_TYPE_UINT32 {
            alignment = read_u32(file).context("read GGUF alignment")?.max(1);
        } else {
            skip_gguf_value(file, value_type, 0)?;
        }
    }
    let tensors = read_tensor_infos(file, tensor_count)?;
    let data_start = align_offset(
        file.stream_position()
            .context("read GGUF tensor data offset")?,
        alignment,
    );
    Ok(GgufHeader {
        data_start,
        tensors,
    })
}

fn read_tensor_infos(file: &mut File, tensor_count: usize) -> Result<Vec<GgufTensorInfo>> {
    let mut tensors = Vec::new();
    tensors
        .try_reserve(tensor_count)
        .context("reserve GGUF tensor info table")?;
    for _ in 0..tensor_count {
        let name = read_gguf_string(file)?;
        let dim_count = read_u32(file).context("read GGUF tensor dim count")?;
        if dim_count > MAX_GGUF_TENSOR_DIMS {
            bail!("GGUF tensor {name} has too many dimensions: {dim_count}");
        }
        let mut dims = Vec::new();
        dims.try_reserve(usize::try_from(dim_count).unwrap_or(0))
            .context("reserve GGUF tensor dims")?;
        for _ in 0..dim_count {
            dims.push(read_u64(file).context("read GGUF tensor dim")?);
        }
        let tensor_type = read_u32(file).context("read GGUF tensor type")?;
        let offset = read_u64(file).context("read GGUF tensor offset")?;
        tensors.push(GgufTensorInfo {
            name,
            dims,
            tensor_type,
            offset,
        });
    }
    Ok(tensors)
}

fn read_bounded_i64_count(file: &mut File, label: &str) -> Result<usize> {
    let value = read_i64(file).with_context(|| format!("read GGUF {label}"))?;
    let count = usize::try_from(value).with_context(|| format!("negative GGUF {label}"))?;
    if count > MAX_GGUF_HEADER_COUNT {
        bail!("GGUF {label} too large: {count}");
    }
    Ok(count)
}

fn read_gguf_string(file: &mut File) -> Result<String> {
    let len = read_u64(file).context("read GGUF string length")?;
    if len > MAX_GGUF_STRING_BYTES {
        bail!("GGUF string is too long: {len}");
    }
    let len = usize::try_from(len).context("GGUF string length exceeds usize")?;
    let mut bytes = vec![0_u8; len];
    file.read_exact(&mut bytes).context("read GGUF string")?;
    String::from_utf8(bytes).context("GGUF string is not UTF-8")
}

fn skip_gguf_value(file: &mut File, value_type: u32, depth: u32) -> Result<()> {
    match value_type {
        0 | 1 | 7 => skip_bytes(file, 1),
        2 | 3 => skip_bytes(file, 2),
        4 | 5 | GGUF_TYPE_FLOAT32 => skip_bytes(file, 4),
        10..=12 => skip_bytes(file, 8),
        GGUF_TYPE_STRING => read_gguf_string(file).map(|_| ()),
        GGUF_TYPE_ARRAY => {
            if depth >= MAX_GGUF_ARRAY_DEPTH {
                bail!("GGUF array nesting too deep");
            }
            let element_type = read_u32(file).context("read GGUF array element type")?;
            let count = read_u64(file).context("read GGUF array length")?;
            if count > MAX_GGUF_ARRAY_ELEMENTS {
                bail!("GGUF array has too many elements: {count}");
            }
            for _ in 0..count {
                skip_gguf_value(file, element_type, depth + 1)?;
            }
            Ok(())
        }
        _ => bail!("unsupported GGUF value type {value_type}"),
    }
}

fn skip_bytes(file: &mut File, count: u64) -> Result<()> {
    file.seek(SeekFrom::Current(
        i64::try_from(count).context("GGUF skip byte count exceeds i64")?,
    ))
    .context("skip GGUF value")?;
    Ok(())
}

fn read_f32_row(file: &mut File, hidden_size: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0_u8; hidden_size * 4];
    file.read_exact(&mut bytes)
        .context("read GGUF f32 token embedding row")?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk size checked")))
        .collect())
}

fn read_f16_row(file: &mut File, hidden_size: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0_u8; hidden_size * 2];
    file.read_exact(&mut bytes)
        .context("read GGUF f16 token embedding row")?;
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("chunk size checked")))
        .map(f16_to_f32)
        .collect())
}

fn read_q6_k_row(file: &mut File, hidden_size: usize) -> Result<Vec<f32>> {
    let block_count = hidden_size / QK_K;
    let mut row = Vec::with_capacity(hidden_size);
    for _ in 0..block_count {
        let mut block = [0_u8; Q6_K_BLOCK_BYTES];
        file.read_exact(&mut block)
            .context("read GGUF Q6_K token embedding block")?;
        dequantize_q6_k_block(&block, &mut row);
    }
    Ok(row)
}

fn dequantize_q6_k_block(block: &[u8; Q6_K_BLOCK_BYTES], output: &mut Vec<f32>) {
    let ql = &block[..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let mut values = [0_f32; QK_K];
    for half in 0..2 {
        let ql_base = half * 64;
        let qh_base = half * 32;
        let sc_base = half * 8;
        let out_base = half * 128;
        for lane in 0..32 {
            let is = lane / 16;
            let qh_lane = qh[qh_base + lane];
            let low_a = ql[ql_base + lane];
            let low_b = ql[ql_base + lane + 32];
            let q1 = q6_value((low_a & 0x0f) | ((qh_lane & 3) << 4));
            let q2 = q6_value((low_b & 0x0f) | (((qh_lane >> 2) & 3) << 4));
            let q3 = q6_value((low_a >> 4) | (((qh_lane >> 4) & 3) << 4));
            let q4 = q6_value((low_b >> 4) | (((qh_lane >> 6) & 3) << 4));
            values[out_base + lane] = d * scale(scales[sc_base + is]) * f32::from(q1);
            values[out_base + lane + 32] = d * scale(scales[sc_base + is + 2]) * f32::from(q2);
            values[out_base + lane + 64] = d * scale(scales[sc_base + is + 4]) * f32::from(q3);
            values[out_base + lane + 96] = d * scale(scales[sc_base + is + 6]) * f32::from(q4);
        }
    }
    output.extend_from_slice(&values);
}

fn q6_value(value: u8) -> i8 {
    i8::try_from(value).expect("Q6 value is <= 63") - 32
}

fn scale(value: u8) -> f32 {
    f32::from(i8::from_le_bytes([value]))
}

fn write_f32_row_le(row: &[f32], output: &mut [u8]) {
    for (value, chunk) in row.iter().zip(output.chunks_exact_mut(4)) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exp = (bits >> 10) & 0x1f;
    let frac = u32::from(bits & 0x03ff);
    let value = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut frac = frac;
                let mut exp = -14_i32;
                while (frac & 0x0400) == 0 {
                    frac <<= 1;
                    exp -= 1;
                }
                frac &= 0x03ff;
                let exp_bits = u32::try_from(exp + 127).expect("normalized f16 exponent") << 23;
                sign | exp_bits | (frac << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => {
            let exp_bits = (u32::from(exp) + 112) << 23;
            sign | exp_bits | (frac << 13)
        }
    };
    f32::from_bits(value)
}

fn align_offset(value: u64, alignment: u32) -> u64 {
    let alignment = u64::from(alignment.max(1));
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + (alignment - remainder)
    }
}

fn read_u32(file: &mut File) -> std::io::Result<u32> {
    let mut bytes = [0_u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i64(file: &mut File) -> std::io::Result<i64> {
    let mut bytes = [0_u8; 8];
    file.read_exact(&mut bytes)?;
    Ok(i64::from_le_bytes(bytes))
}

fn read_u64(file: &mut File) -> std::io::Result<u64> {
    let mut bytes = [0_u8; 8];
    file.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn reads_q6_k_token_embedding_rows() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("embeddings.gguf");
        write_minimal_q6_k_embedding_gguf(&path);

        let table = GgufTokenEmbeddingTable::open(&path, QK_K, 2).unwrap();
        let frame = table.frame_for_positions(&[1, 0], &[0, 1]).unwrap();
        assert_eq!(frame.desc.layer_start, 0);
        assert_eq!(frame.desc.layer_end, 0);
        assert_eq!(frame.desc.token_count, 2);

        let values = frame
            .payload
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert!(values[..QK_K].iter().all(|value| *value == 2.0));
        assert!(values[QK_K..].iter().all(|value| *value == 1.0));
    }

    #[test]
    fn rejects_hidden_size_mismatch() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("embeddings.gguf");
        write_minimal_q6_k_embedding_gguf(&path);

        let error = GgufTokenEmbeddingTable::open(&path, QK_K + 1, 2)
            .unwrap_err()
            .to_string();
        assert!(error.contains("hidden size"));
    }

    fn write_minimal_q6_k_embedding_gguf(path: &Path) {
        let mut file = File::create(path).unwrap();
        file.write_all(GGUF_MAGIC).unwrap();
        file.write_all(&3_u32.to_le_bytes()).unwrap();
        file.write_all(&1_i64.to_le_bytes()).unwrap();
        file.write_all(&1_i64.to_le_bytes()).unwrap();
        write_gguf_string(&mut file, "general.alignment");
        file.write_all(&GGUF_TYPE_UINT32.to_le_bytes()).unwrap();
        file.write_all(&32_u32.to_le_bytes()).unwrap();
        write_gguf_string(&mut file, TOKEN_EMBD_TENSOR);
        file.write_all(&2_u32.to_le_bytes()).unwrap();
        file.write_all(&(QK_K as u64).to_le_bytes()).unwrap();
        file.write_all(&2_u64.to_le_bytes()).unwrap();
        file.write_all(&GGML_TYPE_Q6_K.to_le_bytes()).unwrap();
        file.write_all(&0_u64.to_le_bytes()).unwrap();
        let position = file.stream_position().unwrap();
        let aligned = align_offset(position, 32);
        file.write_all(&vec![0_u8; usize::try_from(aligned - position).unwrap()])
            .unwrap();
        file.write_all(&q6_k_block(1)).unwrap();
        file.write_all(&q6_k_block(2)).unwrap();
    }

    fn write_gguf_string(file: &mut File, value: &str) {
        file.write_all(&(value.len() as u64).to_le_bytes()).unwrap();
        file.write_all(value.as_bytes()).unwrap();
    }

    fn q6_k_block(value: u8) -> [u8; Q6_K_BLOCK_BYTES] {
        let encoded = value + 32;
        let low = encoded & 0x0f;
        let high = encoded >> 4;
        let mut block = [0_u8; Q6_K_BLOCK_BYTES];
        for byte in &mut block[..128] {
            *byte = low | (low << 4);
        }
        for byte in &mut block[128..192] {
            *byte = high | (high << 2) | (high << 4) | (high << 6);
        }
        for byte in &mut block[192..208] {
            *byte = 1_u8;
        }
        block[208..210].copy_from_slice(&0x3c00_u16.to_le_bytes());
        block
    }
}
