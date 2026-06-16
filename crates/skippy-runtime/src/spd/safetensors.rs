use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpdSafetensorsIndex {
    pub tensors: BTreeMap<String, SpdSafetensorsTensor>,
    pub metadata: BTreeMap<String, String>,
    pub data_start: u64,
    pub data_len: u64,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct SpdSafetensorsTensor {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offsets: [u64; 2],
}

#[derive(Debug, Clone)]
pub struct SpdSafetensorsFile {
    path: PathBuf,
    pub index: SpdSafetensorsIndex,
}

impl SpdSafetensorsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let index = SpdSafetensorsIndex::from_path(&path)?;
        Ok(Self { path, index })
    }

    pub fn read_tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let tensor = self.index.tensor(name)?;
        let [start, end] = tensor.data_offsets;
        let len =
            usize::try_from(end - start).context("SPD safetensors tensor is too large to read")?;
        let mut bytes = vec![0_u8; len];
        let mut file = fs::File::open(&self.path)
            .with_context(|| format!("open SPD safetensors file {}", self.path.display()))?;
        file.seek(SeekFrom::Start(self.index.data_start + start))
            .with_context(|| format!("seek SPD safetensors tensor {name}"))?;
        file.read_exact(&mut bytes)
            .with_context(|| format!("read SPD safetensors tensor {name}"))?;
        Ok(bytes)
    }

    pub fn read_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self.index.tensor(name)?;
        let bytes = self.read_tensor_bytes(name)?;
        match tensor.dtype.as_str() {
            "BF16" => bf16_bytes_to_f32(&bytes),
            "F32" => f32_bytes_to_f32(&bytes),
            dtype => bail!("SPD safetensors tensor {name} has unsupported f32 dtype {dtype}"),
        }
    }

    pub fn read_tensor_i64(&self, name: &str) -> Result<Vec<i64>> {
        let tensor = self.index.tensor(name)?;
        if tensor.dtype != "I64" {
            bail!(
                "SPD safetensors tensor {name} has dtype {}; expected I64",
                tensor.dtype
            );
        }
        let bytes = self.read_tensor_bytes(name)?;
        i64_bytes_to_i64(&bytes)
    }
}

impl SpdSafetensorsIndex {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = fs::File::open(path)
            .with_context(|| format!("open SPD safetensors checkpoint {}", path.display()))?;
        let file_len = file
            .metadata()
            .with_context(|| format!("stat SPD safetensors checkpoint {}", path.display()))?
            .len();
        let mut header_len_bytes = [0_u8; 8];
        file.read_exact(&mut header_len_bytes)
            .with_context(|| format!("read SPD safetensors header length {}", path.display()))?;
        let header_len = u64::from_le_bytes(header_len_bytes);
        if header_len == 0 {
            bail!("SPD safetensors header must not be empty");
        }
        if header_len > file_len.saturating_sub(8) {
            bail!(
                "SPD safetensors header length {} exceeds file length {}",
                header_len,
                file_len
            );
        }
        let header_len_usize =
            usize::try_from(header_len).context("SPD safetensors header is too large")?;
        let mut header = vec![0_u8; header_len_usize];
        file.read_exact(&mut header)
            .with_context(|| format!("read SPD safetensors header {}", path.display()))?;
        Self::from_header_bytes(&header, 8 + header_len, file_len)
    }

    fn from_header_bytes(header: &[u8], data_start: u64, file_len: u64) -> Result<Self> {
        if data_start > file_len {
            bail!("SPD safetensors data section starts past end of file");
        }
        let data_len = file_len - data_start;
        let mut metadata = BTreeMap::new();
        let mut tensors = BTreeMap::new();
        let value: serde_json::Value =
            serde_json::from_slice(header).context("parse SPD safetensors header JSON")?;
        let serde_json::Value::Object(entries) = value else {
            bail!("SPD safetensors header must be a JSON object");
        };

        for (name, value) in entries {
            if name == "__metadata__" {
                metadata =
                    serde_json::from_value(value).context("parse SPD safetensors metadata map")?;
                continue;
            }
            if name.trim().is_empty() {
                bail!("SPD safetensors tensor names must not be empty");
            }
            let tensor: SpdSafetensorsTensor = serde_json::from_value(value)
                .with_context(|| format!("parse SPD safetensors tensor metadata {name}"))?;
            tensor.validate(&name, data_len)?;
            tensors.insert(name, tensor);
        }
        validate_safetensors_ranges(&tensors, data_len)?;
        Ok(Self {
            tensors,
            metadata,
            data_start,
            data_len,
        })
    }

    pub fn ensure_tensor_shape(&self, name: &str, expected_shape: &[u64]) -> Result<()> {
        let tensor = self.tensor(name)?;
        if tensor.shape != expected_shape {
            bail!(
                "SPD serving checkpoint tensor {name} shape mismatch: expected {:?}, got {:?}",
                expected_shape,
                tensor.shape
            );
        }
        Ok(())
    }

    pub fn tensor(&self, name: &str) -> Result<&SpdSafetensorsTensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("SPD safetensors file is missing tensor {name}"))
    }
}

impl SpdSafetensorsTensor {
    fn validate(&self, name: &str, data_len: u64) -> Result<()> {
        let [start, end] = self.data_offsets;
        if start > end || end > data_len {
            bail!(
                "SPD safetensors tensor {name} has invalid data offsets {:?} for data length {}",
                self.data_offsets,
                data_len
            );
        }
        let expected_bytes = tensor_byte_len(&self.dtype, &self.shape)
            .with_context(|| format!("validate SPD safetensors tensor {name}"))?;
        if end - start != expected_bytes {
            bail!(
                "SPD safetensors tensor {name} byte length mismatch: offsets describe {}, shape/dtype describe {}",
                end - start,
                expected_bytes
            );
        }
        Ok(())
    }
}

fn validate_safetensors_ranges(
    tensors: &BTreeMap<String, SpdSafetensorsTensor>,
    data_len: u64,
) -> Result<()> {
    let mut ranges: Vec<(&str, [u64; 2])> = tensors
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor.data_offsets))
        .collect();
    ranges.sort_by_key(|(_, [start, _])| *start);
    let mut previous_end = 0;
    for (name, [start, end]) in ranges {
        if start < previous_end {
            bail!("SPD safetensors tensor {name} overlaps a previous tensor range");
        }
        if start > previous_end {
            bail!("SPD safetensors tensor {name} leaves a gap before its data range");
        }
        previous_end = end;
    }
    if previous_end != data_len {
        bail!(
            "SPD safetensors tensor data ends at {}, but data section is {} bytes",
            previous_end,
            data_len
        );
    }
    Ok(())
}

fn tensor_byte_len(dtype: &str, shape: &[u64]) -> Result<u64> {
    let element_bytes = safetensors_dtype_size(dtype)?;
    let elements = tensor_element_count(shape)?;
    elements
        .checked_mul(element_bytes)
        .context("SPD safetensors tensor byte length overflow")
}

fn tensor_element_count(shape: &[u64]) -> Result<u64> {
    shape.iter().try_fold(1_u64, |acc, dimension| {
        acc.checked_mul(*dimension)
            .context("SPD safetensors tensor shape element count overflow")
    })
}

fn safetensors_dtype_size(dtype: &str) -> Result<u64> {
    match dtype {
        "BOOL" | "I8" | "U8" => Ok(1),
        "F16" | "BF16" | "I16" | "U16" => Ok(2),
        "F32" | "I32" | "U32" => Ok(4),
        "F64" | "I64" | "U64" => Ok(8),
        _ => bail!("unsupported SPD safetensors dtype {dtype}"),
    }
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    let chunks = exact_chunks(bytes, 2, "BF16")?;
    Ok(chunks
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
            f32::from_bits(bits << 16)
        })
        .collect())
}

fn f32_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    let chunks = exact_chunks(bytes, 4, "F32")?;
    Ok(chunks
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn i64_bytes_to_i64(bytes: &[u8]) -> Result<Vec<i64>> {
    let chunks = exact_chunks(bytes, 8, "I64")?;
    Ok(chunks
        .map(|chunk| {
            i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect())
}

fn exact_chunks<'a>(
    bytes: &'a [u8],
    chunk_len: usize,
    dtype: &'static str,
) -> Result<std::slice::ChunksExact<'a, u8>> {
    let chunks = bytes.chunks_exact(chunk_len);
    if !chunks.remainder().is_empty() {
        bail!("SPD safetensors {dtype} payload has trailing partial element");
    }
    Ok(chunks)
}
