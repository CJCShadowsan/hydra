use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const MAX_GGUF_STRING_BYTES: u64 = 1_000_000;
const MAX_GGUF_ARRAY_ELEMENTS: u64 = 1_000_000;
const MAX_GGUF_ARRAY_DEPTH: u32 = 64;
const MAX_GGUF_TENSOR_DIMS: u32 = 8;
const MAX_GGUF_HEADER_KV_COUNT: usize = 1_000_000;
const MAX_GGUF_TENSOR_COUNT: usize = 1_000_000;

/// GGUF value types (matching gguf.h enum).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }

    fn fixed_size(self) -> Option<usize> {
        match self {
            Self::Uint8 | Self::Int8 | Self::Bool => Some(1),
            Self::Uint16 | Self::Int16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Float32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Float64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

fn read_u32(f: &mut std::fs::File) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(f: &mut std::fs::File) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i32(f: &mut std::fs::File) -> std::io::Result<i32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_i64(f: &mut std::fs::File) -> std::io::Result<i64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_gguf_header_count(
    f: &mut std::fs::File,
    max: usize,
    label: &str,
) -> std::io::Result<usize> {
    let value = read_i64(f)?;
    let count = usize::try_from(value).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, format!("negative {label}"))
    })?;
    if count > max {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{label} too large"),
        ));
    }
    Ok(count)
}

fn read_bounded_len(f: &mut std::fs::File, max: u64, label: &str) -> std::io::Result<usize> {
    let len = read_u64(f)?;
    if len > max {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{label} too long"),
        ));
    }
    usize::try_from(len).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{label} too large"),
        )
    })
}

fn read_gguf_string(f: &mut std::fs::File) -> std::io::Result<String> {
    let len = read_bounded_len(f, MAX_GGUF_STRING_BYTES, "string")?;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid UTF-8 in GGUF string",
        )
    })
}

fn skip_gguf_value(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<()> {
    skip_gguf_value_with_depth(f, typ, 0)
}

fn skip_gguf_value_with_depth(
    f: &mut std::fs::File,
    typ: GgufType,
    depth: u32,
) -> std::io::Result<()> {
    match typ {
        GgufType::String => {
            let _ = read_gguf_string(f)?;
        }
        GgufType::Array => {
            if depth >= MAX_GGUF_ARRAY_DEPTH {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "GGUF nesting too deep",
                ));
            }
            let elem_type = GgufType::from_u32(read_u32(f)?).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "bad array type")
            })?;
            let count = read_bounded_len(f, MAX_GGUF_ARRAY_ELEMENTS, "array")?;
            for _ in 0..count {
                skip_gguf_value_with_depth(f, elem_type, depth + 1)?;
            }
        }
        other => {
            let size = other.fixed_size().unwrap_or(0);
            f.seek(SeekFrom::Current(size as i64))?;
        }
    }
    Ok(())
}

fn read_gguf_value_as_u32(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<Option<u32>> {
    match typ {
        GgufType::Uint32 => Ok(Some(read_u32(f)?)),
        GgufType::Int32 => {
            let value = read_i32(f)?;
            let value = u32::try_from(value).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "negative Int32 where unsigned GGUF value was expected",
                )
            })?;
            Ok(Some(value))
        }
        GgufType::Uint16 => {
            let mut buf = [0u8; 2];
            f.read_exact(&mut buf)?;
            Ok(Some(u16::from_le_bytes(buf) as u32))
        }
        GgufType::Uint8 => {
            let mut buf = [0u8; 1];
            f.read_exact(&mut buf)?;
            Ok(Some(buf[0] as u32))
        }
        _ => {
            skip_gguf_value(f, typ)?;
            Ok(None)
        }
    }
}

fn read_gguf_value_as_f32(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<Option<f32>> {
    match typ {
        GgufType::Float32 => {
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf)?;
            Ok(Some(f32::from_le_bytes(buf)))
        }
        _ => {
            skip_gguf_value(f, typ)?;
            Ok(None)
        }
    }
}

fn read_gguf_value_as_bool(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<Option<bool>> {
    match typ {
        GgufType::Bool => {
            let mut buf = [0u8; 1];
            f.read_exact(&mut buf)?;
            Ok(Some(buf[0] != 0))
        }
        _ => {
            skip_gguf_value(f, typ)?;
            Ok(None)
        }
    }
}

fn read_gguf_value_as_string_opt(
    f: &mut std::fs::File,
    typ: GgufType,
) -> std::io::Result<Option<String>> {
    match typ {
        GgufType::String => Ok(Some(read_gguf_string(f)?)),
        _ => {
            skip_gguf_value(f, typ)?;
            Ok(None)
        }
    }
}

fn read_gguf_value_as_string_array(
    f: &mut std::fs::File,
    typ: GgufType,
) -> std::io::Result<Option<Vec<String>>> {
    match typ {
        GgufType::String => Ok(Some(vec![read_gguf_string(f)?])),
        GgufType::Array => {
            let elem_type = GgufType::from_u32(read_u32(f)?).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "bad array type")
            })?;
            let count = read_bounded_len(f, MAX_GGUF_ARRAY_ELEMENTS, "array")?;
            if elem_type != GgufType::String {
                for _ in 0..count {
                    skip_gguf_value_with_depth(f, elem_type, 1)?;
                }
                return Ok(None);
            }
            let mut values = Vec::new();
            values.try_reserve(count).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "GGUF string array requires too much memory",
                )
            })?;
            for _ in 0..count {
                values.push(read_gguf_string(f)?);
            }
            Ok(Some(values))
        }
        _ => {
            skip_gguf_value(f, typ)?;
            Ok(None)
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GgufCompactMeta {
    pub architecture: String,
    pub parameter_size: Option<String>,
    pub context_length: u32,
    pub vocab_size: u32,
    pub embedding_size: u32,
    pub head_count: u32,
    pub kv_head_count: u32,
    pub layer_count: u32,
    pub feed_forward_length: u32,
    pub key_length: u32,
    pub value_length: u32,
    pub tokenizer_model_name: String,
    pub rope_scale: f32,
    pub rope_freq_base: f32,
    pub expert_count: u32,
    pub expert_used_count: u32,
}

#[derive(Clone, Debug, Default)]
pub struct GgufFitMeta {
    pub general_name: Option<String>,
    pub general_type: Option<String>,
    pub general_tags: Vec<String>,
    pub file_type: Option<u32>,
    pub chat_template: Option<String>,
    pub chat_templates: Vec<String>,
    pub fim_pre_token_id: Option<u32>,
    pub fim_suf_token_id: Option<u32>,
    pub fim_mid_token_id: Option<u32>,
    pub pooling_type: Option<u32>,
    pub classifier_output_labels: Vec<String>,
    pub rope_scaling_type: Option<String>,
    pub rope_scaling_factor: Option<f32>,
    pub rope_scaling_original_context_length: Option<u32>,
    pub rope_scaling_finetuned: Option<bool>,
    pub clip_projector_type: Option<String>,
    pub clip_has_vision_encoder: Option<bool>,
    pub clip_has_audio_encoder: Option<bool>,
}

impl GgufFitMeta {
    pub fn has_chat_template(&self) -> bool {
        self.chat_template
            .as_deref()
            .is_some_and(|template| !template.trim().is_empty())
            || self
                .chat_templates
                .iter()
                .any(|template| !template.trim().is_empty())
    }

    pub fn chat_template_text(&self) -> String {
        let mut text = String::new();
        if let Some(template) = &self.chat_template {
            text.push_str(template);
            text.push('\n');
        }
        for template in &self.chat_templates {
            text.push_str(template);
            text.push('\n');
        }
        text
    }

    pub fn has_fill_in_middle_tokens(&self) -> bool {
        self.fim_pre_token_id.is_some()
            && self.fim_suf_token_id.is_some()
            && self.fim_mid_token_id.is_some()
    }
}

impl GgufCompactMeta {
    pub fn effective_kv_head_count(&self) -> Option<u32> {
        if self.kv_head_count > 0 {
            Some(self.kv_head_count)
        } else if self.head_count > 0 {
            Some(self.head_count)
        } else {
            None
        }
    }

    pub fn k_cache_bytes_per_token_f16(&self) -> Option<u64> {
        GgufKvCacheQuant::f16().k_cache_bytes_per_token(self)
    }

    pub fn v_cache_bytes_per_token_f16(&self) -> Option<u64> {
        GgufKvCacheQuant::f16().v_cache_bytes_per_token(self)
    }

    pub fn kv_cache_bytes_per_token_f16(&self) -> Option<u64> {
        GgufKvCacheQuant::f16().kv_cache_bytes_per_token(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GgufKvCacheType {
    F16,
    Q8_0,
    Q4_0,
}

impl GgufKvCacheType {
    pub fn from_llama_arg(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "f16" => Some(Self::F16),
            "q8_0" => Some(Self::Q8_0),
            "q4_0" => Some(Self::Q4_0),
            _ => None,
        }
    }

    pub const fn as_llama_arg(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }

    fn block_shape(self) -> (u64, u64) {
        match self {
            Self::F16 => (1, 2),
            Self::Q8_0 => (32, 34),
            Self::Q4_0 => (32, 18),
        }
    }

    fn bytes_for_elements(self, elements: u64) -> Option<u64> {
        let (block_elements, block_bytes) = self.block_shape();
        let blocks = elements
            .checked_add(block_elements.checked_sub(1)?)?
            .checked_div(block_elements)?;
        blocks.checked_mul(block_bytes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GgufKvCacheQuant {
    pub k: GgufKvCacheType,
    pub v: GgufKvCacheType,
}

impl GgufKvCacheQuant {
    /// f16 K + f16 V — highest quality, largest KV cache.
    pub const F16: Self = Self {
        k: GgufKvCacheType::F16,
        v: GgufKvCacheType::F16,
    };

    /// q8_0 K + q8_0 V — moderate compression.
    pub const Q8_0: Self = Self {
        k: GgufKvCacheType::Q8_0,
        v: GgufKvCacheType::Q8_0,
    };

    /// q4_0 K + q4_0 V — most aggressive compression, smallest KV cache.
    pub const Q4_0: Self = Self {
        k: GgufKvCacheType::Q4_0,
        v: GgufKvCacheType::Q4_0,
    };

    pub const fn new(k: GgufKvCacheType, v: GgufKvCacheType) -> Self {
        Self { k, v }
    }

    pub const fn f16() -> Self {
        Self::F16
    }

    /// Returns `true` if `self` uses more aggressive (smaller) quantisation
    /// than `other`.
    pub const fn is_more_aggressive_than(self, other: Self) -> bool {
        Self::aggressiveness(self) > Self::aggressiveness(other)
    }

    const fn aggressiveness(q: Self) -> u8 {
        Self::type_aggressiveness(q.k) + Self::type_aggressiveness(q.v)
    }

    const fn type_aggressiveness(t: GgufKvCacheType) -> u8 {
        match t {
            GgufKvCacheType::F16 => 0,
            GgufKvCacheType::Q8_0 => 1,
            GgufKvCacheType::Q4_0 => 2,
        }
    }

    pub fn from_llama_args(cache_type_k: &str, cache_type_v: &str) -> Option<Self> {
        Some(Self {
            k: GgufKvCacheType::from_llama_arg(cache_type_k)?,
            v: GgufKvCacheType::from_llama_arg(cache_type_v)?,
        })
    }

    pub fn k_cache_bytes_per_token(self, meta: &GgufCompactMeta) -> Option<u64> {
        cache_bytes_per_token(meta, meta.key_length, self.k)
    }

    pub fn v_cache_bytes_per_token(self, meta: &GgufCompactMeta) -> Option<u64> {
        cache_bytes_per_token(meta, meta.value_length, self.v)
    }

    pub fn kv_cache_bytes_per_token(self, meta: &GgufCompactMeta) -> Option<u64> {
        self.k_cache_bytes_per_token(meta)?
            .checked_add(self.v_cache_bytes_per_token(meta)?)
    }
}

fn cache_bytes_per_token(
    meta: &GgufCompactMeta,
    vector_length: u32,
    cache_type: GgufKvCacheType,
) -> Option<u64> {
    let kv_heads = u64::from(meta.effective_kv_head_count()?);
    let vector_length = u64::from((vector_length > 0).then_some(vector_length)?);
    let layers = u64::from((meta.layer_count > 0).then_some(meta.layer_count)?);
    let elements_per_layer = kv_heads.checked_mul(vector_length)?;
    cache_type
        .bytes_for_elements(elements_per_layer)?
        .checked_mul(layers)
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GgufTensorByteProfile {
    pub tensor_count: u64,
    pub block_tensor_count: u64,
    pub distinct_block_count: u32,
    pub has_token_embedding_tensor: bool,
    pub has_output_tensor: bool,
    pub has_output_norm_tensor: bool,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub full_model_bytes: u64,
    pub base_resident_bytes: u64,
    pub expert_tensor_bytes: u64,
    pub group_bytes: GgufTensorGroupByteProfile,
    pub matmul: GgufTensorMatmulProfile,
    pub file_overhead_bytes: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GgufTensorGroupByteProfile {
    pub attention_bytes: u64,
    pub feed_forward_bytes: u64,
    pub expert_feed_forward_bytes: u64,
    pub embedding_bytes: u64,
    pub output_bytes: u64,
    pub normalization_bytes: u64,
    pub other_bytes: u64,
}

#[derive(Clone, Debug)]
struct GgufTensorInfo {
    name: String,
    dimensions: Vec<u64>,
    tensor_type: u32,
    offset: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GgufTensorMatmulProfile {
    pub base_bytes: u64,
    pub expert_bytes: u64,
    pub base_flops_per_token: u64,
    pub expert_flops_per_token: u64,
    pub base_type_bytes: GgufTensorTypeByteProfile,
    pub expert_type_bytes: GgufTensorTypeByteProfile,
    pub attention: GgufMatmulGroupProfile,
    pub feed_forward: GgufMatmulGroupProfile,
    pub expert_feed_forward: GgufMatmulGroupProfile,
    pub output: GgufMatmulGroupProfile,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GgufMatmulGroupProfile {
    pub bytes: u64,
    pub flops_per_token: u64,
    pub type_bytes: GgufTensorTypeByteProfile,
    pub shape: GgufMatmulShapeProfile,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GgufMatmulShapeProfile {
    pub tensor_count: u64,
    pub logical_matrix_count: u64,
    pub total_elements: u64,
    pub min_input_width: u64,
    pub max_input_width: u64,
    pub min_output_width: u64,
    pub max_output_width: u64,
    pub weighted_avg_input_width: u64,
    pub weighted_avg_output_width: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GgufTensorTypeByteProfile {
    pub f32_bytes: u64,
    pub f16_bytes: u64,
    pub bf16_bytes: u64,
    pub q4_0_bytes: u64,
    pub q4_k_bytes: u64,
    pub q5_k_bytes: u64,
    pub q6_k_bytes: u64,
    pub q8_0_bytes: u64,
    pub iq_bytes: u64,
    pub other_quantized_bytes: u64,
    pub unknown_bytes: u64,
}

/// Scan a GGUF file header and return compact structural metadata.
/// Reads only the KV section, never tensor data. Returns None on any parse failure.
pub fn scan_gguf_compact_meta(path: &Path) -> Option<GgufCompactMeta> {
    let mut f = std::fs::File::open(path).ok()?;

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let version = read_u32(&mut f).ok()?;
    if version < 2 {
        return None;
    }
    let _n_tensors = read_gguf_header_count(&mut f, MAX_GGUF_TENSOR_COUNT, "tensor count").ok()?;
    let n_kv = read_gguf_header_count(&mut f, MAX_GGUF_HEADER_KV_COUNT, "KV count").ok()?;

    let mut meta = GgufCompactMeta::default();
    for _ in 0..n_kv {
        let key = read_gguf_string(&mut f).ok()?;
        let vtype = GgufType::from_u32(read_u32(&mut f).ok()?)?;

        if key == "general.architecture" {
            meta.architecture = read_gguf_value_as_string_opt(&mut f, vtype).ok()??;
        } else if key == "general.size_label" {
            meta.parameter_size = read_gguf_value_as_string_opt(&mut f, vtype).ok()?;
        } else if key == "tokenizer.ggml.model" {
            meta.tokenizer_model_name = read_gguf_value_as_string_opt(&mut f, vtype).ok()??;
        } else if key.ends_with(".context_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.context_length = v;
            }
        } else if key.ends_with(".embedding_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.embedding_size = v;
            }
        } else if key.ends_with(".head_count") && !key.ends_with("_kv") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.head_count = v;
            }
        } else if key.ends_with(".attention.head_count_kv") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.kv_head_count = v;
            }
        } else if key.ends_with(".block_count") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.layer_count = v;
            }
        } else if key.ends_with(".feed_forward_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.feed_forward_length = v;
            }
        } else if key.ends_with(".attention.key_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.key_length = v;
            }
        } else if key.ends_with(".attention.value_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.value_length = v;
            }
        } else if key.ends_with(".rope.scale") {
            if let Ok(Some(v)) = read_gguf_value_as_f32(&mut f, vtype) {
                meta.rope_scale = v;
            }
        } else if key.ends_with(".rope.freq_base") {
            if let Ok(Some(v)) = read_gguf_value_as_f32(&mut f, vtype) {
                meta.rope_freq_base = v;
            }
        } else if key.ends_with(".vocab_size") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.vocab_size = v;
            }
        } else if key.ends_with(".expert_count") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.expert_count = v;
            }
        } else if key.ends_with(".expert_used_count") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.expert_used_count = v;
            }
        } else {
            skip_gguf_value(&mut f, vtype).ok()?;
        }
    }

    match meta.embedding_size.checked_div(meta.head_count.max(1)) {
        Some(key_length) if meta.key_length == 0 && meta.head_count > 0 => {
            meta.key_length = key_length;
        }
        _ => {}
    }
    match meta
        .effective_kv_head_count()
        .and_then(|effective_kv| meta.embedding_size.checked_div(effective_kv))
    {
        Some(value_length) if meta.value_length == 0 => {
            meta.value_length = value_length;
        }
        _ => {}
    }

    Some(meta)
}

/// Scan GGUF metadata used by model-fit capability and workload scoring.
/// Reads only scalar and small string-array header values, never tensor data.
pub fn scan_gguf_fit_meta(path: &Path) -> Option<GgufFitMeta> {
    let mut f = std::fs::File::open(path).ok()?;

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let version = read_u32(&mut f).ok()?;
    if version < 2 {
        return None;
    }
    let _n_tensors = read_gguf_header_count(&mut f, MAX_GGUF_TENSOR_COUNT, "tensor count").ok()?;
    let n_kv = read_gguf_header_count(&mut f, MAX_GGUF_HEADER_KV_COUNT, "KV count").ok()?;

    let mut meta = GgufFitMeta::default();
    for _ in 0..n_kv {
        let key = read_gguf_string(&mut f).ok()?;
        let vtype = GgufType::from_u32(read_u32(&mut f).ok()?)?;
        match key.as_str() {
            "general.name" => {
                meta.general_name = read_gguf_value_as_string_opt(&mut f, vtype).ok()?;
            }
            "general.type" => {
                meta.general_type = read_gguf_value_as_string_opt(&mut f, vtype).ok()?;
            }
            "general.tags" => {
                meta.general_tags = read_gguf_value_as_string_array(&mut f, vtype)
                    .ok()?
                    .unwrap_or_default();
            }
            "general.file_type" => {
                meta.file_type = read_gguf_value_as_u32(&mut f, vtype).ok()?;
            }
            "tokenizer.chat_template" => {
                meta.chat_template = read_gguf_value_as_string_opt(&mut f, vtype).ok()?;
            }
            "tokenizer.chat_templates" => {
                meta.chat_templates = read_gguf_value_as_string_array(&mut f, vtype)
                    .ok()?
                    .unwrap_or_default();
            }
            "tokenizer.ggml.fim_pre_token_id" | "tokenizer.ggml.prefix_token_id" => {
                meta.fim_pre_token_id = read_gguf_value_as_u32(&mut f, vtype).ok()?;
            }
            "tokenizer.ggml.fim_suf_token_id" | "tokenizer.ggml.suffix_token_id" => {
                meta.fim_suf_token_id = read_gguf_value_as_u32(&mut f, vtype).ok()?;
            }
            "tokenizer.ggml.fim_mid_token_id" | "tokenizer.ggml.middle_token_id" => {
                meta.fim_mid_token_id = read_gguf_value_as_u32(&mut f, vtype).ok()?;
            }
            "clip.projector_type" => {
                meta.clip_projector_type = read_gguf_value_as_string_opt(&mut f, vtype).ok()?;
            }
            "clip.has_vision_encoder" => {
                meta.clip_has_vision_encoder = read_gguf_value_as_bool(&mut f, vtype).ok()?;
            }
            "clip.has_audio_encoder" => {
                meta.clip_has_audio_encoder = read_gguf_value_as_bool(&mut f, vtype).ok()?;
            }
            _ if key.ends_with(".pooling_type") => {
                meta.pooling_type = read_gguf_value_as_u32(&mut f, vtype).ok()?;
            }
            _ if key.ends_with(".classifier.output_labels") => {
                meta.classifier_output_labels = read_gguf_value_as_string_array(&mut f, vtype)
                    .ok()?
                    .unwrap_or_default();
            }
            _ if key.ends_with(".rope.scaling.type") => {
                meta.rope_scaling_type = read_gguf_value_as_string_opt(&mut f, vtype).ok()?;
            }
            _ if key.ends_with(".rope.scaling.factor") => {
                meta.rope_scaling_factor = read_gguf_value_as_f32(&mut f, vtype).ok()?;
            }
            _ if key.ends_with(".rope.scaling.original_context_length") => {
                meta.rope_scaling_original_context_length =
                    read_gguf_value_as_u32(&mut f, vtype).ok()?;
            }
            _ if key.ends_with(".rope.scaling.finetuned") => {
                meta.rope_scaling_finetuned = read_gguf_value_as_bool(&mut f, vtype).ok()?;
            }
            _ => {
                skip_gguf_value(&mut f, vtype).ok()?;
            }
        }
    }

    Some(meta)
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

fn read_tensor_infos(
    f: &mut std::fs::File,
    n_tensors: usize,
) -> std::io::Result<Vec<GgufTensorInfo>> {
    let mut tensors = Vec::new();
    tensors.try_reserve(n_tensors).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "GGUF tensor count requires too much memory",
        )
    })?;
    for _ in 0..n_tensors {
        let name = read_gguf_string(f)?;
        let n_dims = read_u32(f)?;
        if n_dims > MAX_GGUF_TENSOR_DIMS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "too many GGUF tensor dimensions",
            ));
        }
        let mut dimensions = Vec::new();
        dimensions.try_reserve(n_dims as usize).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "GGUF tensor dimensions require too much memory",
            )
        })?;
        for _ in 0..n_dims {
            dimensions.push(read_u64(f)?);
        }
        let tensor_type = read_u32(f)?;
        let offset = read_u64(f)?;
        tensors.push(GgufTensorInfo {
            name,
            dimensions,
            tensor_type,
            offset,
        });
    }
    Ok(tensors)
}

fn is_expert_partitioned_tensor(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    if lower.contains("shared_expert") || lower.contains("sharedexpert") || lower.contains("shexp")
    {
        return false;
    }

    lower.contains("ffn_gate_exps")
        || lower.contains("ffn_up_exps")
        || lower.contains("ffn_down_exps")
        || lower.contains("exp_probs")
        || lower.contains(".expert")
        || lower.contains("_expert")
}

fn tensor_block_index(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let digits = rest
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

fn is_token_embedding_tensor(name: &str) -> bool {
    matches!(name, "token_embd.weight" | "token_embd")
}

fn is_output_tensor(name: &str) -> bool {
    matches!(name, "output.weight" | "output")
}

fn is_output_norm_tensor(name: &str) -> bool {
    matches!(name, "output_norm.weight" | "output_norm")
}

fn classify_tensor_group(name: &str) -> TensorGroup {
    let lower = name.to_ascii_lowercase();
    if is_token_embedding_tensor(name)
        || lower.contains("token_embd")
        || lower.contains("tok_embeddings")
    {
        return TensorGroup::Embedding;
    }
    if is_output_tensor(name) || lower.contains("lm_head") {
        return TensorGroup::Output;
    }
    if lower.contains("norm") {
        return TensorGroup::Normalization;
    }
    if is_expert_partitioned_tensor(name) {
        return TensorGroup::ExpertFeedForward;
    }
    if lower.contains("attn")
        || lower.contains(".wq")
        || lower.contains(".wk")
        || lower.contains(".wv")
        || lower.contains(".wo")
    {
        return TensorGroup::Attention;
    }
    if lower.contains("ffn")
        || lower.contains("feed_forward")
        || lower.contains("mlp")
        || lower.contains("w1")
        || lower.contains("w2")
        || lower.contains("w3")
    {
        return TensorGroup::FeedForward;
    }
    TensorGroup::Other
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TensorGroup {
    Attention,
    FeedForward,
    ExpertFeedForward,
    Embedding,
    Output,
    Normalization,
    Other,
}

fn add_tensor_group_bytes(
    group_bytes: &mut GgufTensorGroupByteProfile,
    group: TensorGroup,
    tensor_bytes: u64,
) {
    match group {
        TensorGroup::Attention => {
            group_bytes.attention_bytes = group_bytes.attention_bytes.saturating_add(tensor_bytes);
        }
        TensorGroup::FeedForward => {
            group_bytes.feed_forward_bytes =
                group_bytes.feed_forward_bytes.saturating_add(tensor_bytes);
        }
        TensorGroup::ExpertFeedForward => {
            group_bytes.expert_feed_forward_bytes = group_bytes
                .expert_feed_forward_bytes
                .saturating_add(tensor_bytes);
        }
        TensorGroup::Embedding => {
            group_bytes.embedding_bytes = group_bytes.embedding_bytes.saturating_add(tensor_bytes);
        }
        TensorGroup::Output => {
            group_bytes.output_bytes = group_bytes.output_bytes.saturating_add(tensor_bytes);
        }
        TensorGroup::Normalization => {
            group_bytes.normalization_bytes =
                group_bytes.normalization_bytes.saturating_add(tensor_bytes);
        }
        TensorGroup::Other => {
            group_bytes.other_bytes = group_bytes.other_bytes.saturating_add(tensor_bytes);
        }
    }
}

fn add_matmul_profile(
    profile: &mut GgufTensorMatmulProfile,
    tensor: &GgufTensorInfo,
    group: TensorGroup,
    tensor_bytes: u64,
) {
    if !is_decode_matmul_group(group) {
        return;
    }
    let flops = tensor_flops_per_token(tensor);
    let group_profile = match group {
        TensorGroup::Attention => &mut profile.attention,
        TensorGroup::FeedForward => &mut profile.feed_forward,
        TensorGroup::ExpertFeedForward => &mut profile.expert_feed_forward,
        TensorGroup::Output => &mut profile.output,
        _ => return,
    };
    add_matmul_group_profile(
        group_profile,
        tensor.tensor_type,
        tensor,
        tensor_bytes,
        flops,
    );
    if is_expert_partitioned_tensor(&tensor.name) {
        profile.expert_bytes = profile.expert_bytes.saturating_add(tensor_bytes);
        profile.expert_flops_per_token = profile.expert_flops_per_token.saturating_add(flops);
        add_tensor_type_bytes(
            &mut profile.expert_type_bytes,
            tensor.tensor_type,
            tensor_bytes,
        );
    } else {
        profile.base_bytes = profile.base_bytes.saturating_add(tensor_bytes);
        profile.base_flops_per_token = profile.base_flops_per_token.saturating_add(flops);
        add_tensor_type_bytes(
            &mut profile.base_type_bytes,
            tensor.tensor_type,
            tensor_bytes,
        );
    }
}

fn add_matmul_group_profile(
    profile: &mut GgufMatmulGroupProfile,
    tensor_type: u32,
    tensor: &GgufTensorInfo,
    tensor_bytes: u64,
    flops: u64,
) {
    profile.bytes = profile.bytes.saturating_add(tensor_bytes);
    profile.flops_per_token = profile.flops_per_token.saturating_add(flops);
    add_tensor_type_bytes(&mut profile.type_bytes, tensor_type, tensor_bytes);
    profile.shape.add_tensor(tensor);
}

fn is_decode_matmul_group(group: TensorGroup) -> bool {
    matches!(
        group,
        TensorGroup::Attention
            | TensorGroup::FeedForward
            | TensorGroup::ExpertFeedForward
            | TensorGroup::Output
    )
}

fn tensor_flops_per_token(tensor: &GgufTensorInfo) -> u64 {
    tensor
        .dimensions
        .iter()
        .try_fold(1u64, |acc, dim| acc.checked_mul(*dim))
        .unwrap_or(u64::MAX / 2)
        .saturating_mul(2)
}

impl GgufMatmulShapeProfile {
    fn add_tensor(&mut self, tensor: &GgufTensorInfo) {
        let Some(shape) = tensor_matrix_shape(tensor) else {
            return;
        };
        self.tensor_count = self.tensor_count.saturating_add(1);
        self.logical_matrix_count = self
            .logical_matrix_count
            .saturating_add(shape.logical_matrix_count);
        self.total_elements = self.total_elements.saturating_add(shape.elements);
        self.min_input_width = nonzero_min(self.min_input_width, shape.input_width);
        self.max_input_width = self.max_input_width.max(shape.input_width);
        self.min_output_width = nonzero_min(self.min_output_width, shape.output_width);
        self.max_output_width = self.max_output_width.max(shape.output_width);

        let previous_elements = self.total_elements.saturating_sub(shape.elements);
        self.weighted_avg_input_width = weighted_average_after_add(
            self.weighted_avg_input_width,
            previous_elements,
            shape.input_width,
            shape.elements,
        );
        self.weighted_avg_output_width = weighted_average_after_add(
            self.weighted_avg_output_width,
            previous_elements,
            shape.output_width,
            shape.elements,
        );
    }
}

#[derive(Clone, Copy, Debug)]
struct TensorMatrixShape {
    input_width: u64,
    output_width: u64,
    logical_matrix_count: u64,
    elements: u64,
}

fn tensor_matrix_shape(tensor: &GgufTensorInfo) -> Option<TensorMatrixShape> {
    // GGUF tensor dimensions follow GGML's `ne[]` order. For the transformer
    // weight matrices llama.cpp feeds to GGML_OP_MUL_MAT / GGML_OP_MUL_MAT_ID,
    // `ne[0]` is the input width and `ne[1]` is the output width. Additional
    // dimensions represent a stack of logical matrices, most commonly MoE
    // experts. We keep this as a compact summary rather than storing every
    // tensor because model-fit only needs portable shape facts: how large the
    // matvec/GEMM operations are and how many separately dispatched logical
    // matrices exist.
    if tensor.dimensions.len() < 2 {
        return None;
    }
    let input_width = tensor.dimensions[0];
    let output_width = tensor.dimensions[1];
    let logical_matrix_count = tensor.dimensions[2..]
        .iter()
        .try_fold(1u64, |acc, dim| acc.checked_mul(*dim))
        .unwrap_or(u64::MAX);
    let elements = tensor
        .dimensions
        .iter()
        .try_fold(1u64, |acc, dim| acc.checked_mul(*dim))
        .unwrap_or(u64::MAX);
    Some(TensorMatrixShape {
        input_width,
        output_width,
        logical_matrix_count,
        elements,
    })
}

fn nonzero_min(current: u64, candidate: u64) -> u64 {
    if current == 0 {
        candidate
    } else {
        current.min(candidate)
    }
}

fn weighted_average_after_add(
    previous_average: u64,
    previous_weight: u64,
    added_value: u64,
    added_weight: u64,
) -> u64 {
    let total_weight = previous_weight.saturating_add(added_weight);
    if total_weight == 0 {
        return 0;
    }
    let weighted_sum = u128::from(previous_average)
        .saturating_mul(u128::from(previous_weight))
        .saturating_add(u128::from(added_value).saturating_mul(u128::from(added_weight)));
    (weighted_sum / u128::from(total_weight))
        .try_into()
        .unwrap_or(u64::MAX)
}

fn add_tensor_type_bytes(profile: &mut GgufTensorTypeByteProfile, tensor_type: u32, bytes: u64) {
    match tensor_type {
        0 => profile.f32_bytes = profile.f32_bytes.saturating_add(bytes),
        1 => profile.f16_bytes = profile.f16_bytes.saturating_add(bytes),
        2 => profile.q4_0_bytes = profile.q4_0_bytes.saturating_add(bytes),
        8 => profile.q8_0_bytes = profile.q8_0_bytes.saturating_add(bytes),
        12 => profile.q4_k_bytes = profile.q4_k_bytes.saturating_add(bytes),
        13 => profile.q5_k_bytes = profile.q5_k_bytes.saturating_add(bytes),
        14 => profile.q6_k_bytes = profile.q6_k_bytes.saturating_add(bytes),
        16..=23 | 29 => profile.iq_bytes = profile.iq_bytes.saturating_add(bytes),
        30 => profile.bf16_bytes = profile.bf16_bytes.saturating_add(bytes),
        3 | 6 | 7 | 9..=11 | 15 | 34 | 35 | 39 | 40 => {
            profile.other_quantized_bytes = profile.other_quantized_bytes.saturating_add(bytes);
        }
        _ => profile.unknown_bytes = profile.unknown_bytes.saturating_add(bytes),
    }
}

/// Scan GGUF tensor metadata and estimate which bytes are always resident versus
/// expert-partitioned. Reads only the header and tensor-info tables.
pub fn scan_gguf_tensor_byte_profile(path: &Path) -> Option<GgufTensorByteProfile> {
    let mut f = std::fs::File::open(path).ok()?;
    let file_len = f.metadata().ok()?.len();

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let version = read_u32(&mut f).ok()?;
    if version < 2 {
        return None;
    }

    let n_tensors = read_gguf_header_count(&mut f, MAX_GGUF_TENSOR_COUNT, "tensor count").ok()?;
    let n_kv = read_gguf_header_count(&mut f, MAX_GGUF_HEADER_KV_COUNT, "KV count").ok()?;

    let mut expert_count = 0u32;
    let mut expert_used_count = 0u32;
    let mut alignment = 32u32;

    for _ in 0..n_kv {
        let key = read_gguf_string(&mut f).ok()?;
        let vtype = GgufType::from_u32(read_u32(&mut f).ok()?)?;

        if key == "general.alignment" {
            if let Ok(Some(value)) = read_gguf_value_as_u32(&mut f, vtype) {
                alignment = value.max(1);
            }
        } else if key.ends_with(".expert_count") {
            if let Ok(Some(value)) = read_gguf_value_as_u32(&mut f, vtype) {
                expert_count = value;
            }
        } else if key.ends_with(".expert_used_count") {
            if let Ok(Some(value)) = read_gguf_value_as_u32(&mut f, vtype) {
                expert_used_count = value;
            }
        } else {
            skip_gguf_value(&mut f, vtype).ok()?;
        }
    }

    let mut tensors = read_tensor_infos(&mut f, n_tensors).ok()?;
    if tensors.is_empty() {
        return Some(GgufTensorByteProfile {
            tensor_count: 0,
            block_tensor_count: 0,
            distinct_block_count: 0,
            has_token_embedding_tensor: false,
            has_output_tensor: false,
            has_output_norm_tensor: false,
            expert_count,
            expert_used_count,
            full_model_bytes: file_len,
            base_resident_bytes: 0,
            expert_tensor_bytes: 0,
            group_bytes: GgufTensorGroupByteProfile::default(),
            matmul: GgufTensorMatmulProfile::default(),
            file_overhead_bytes: file_len,
        });
    }

    let tensor_info_end = f.stream_position().ok()?;
    let data_start = align_offset(tensor_info_end, alignment);
    if data_start > file_len {
        return None;
    }
    let data_len = file_len - data_start;

    tensors.sort_by_key(|tensor| tensor.offset);
    if tensors.first()?.offset > data_len {
        return None;
    }

    let mut base_resident_bytes = 0u64;
    let mut expert_tensor_bytes = 0u64;
    let mut group_bytes = GgufTensorGroupByteProfile::default();
    let mut matmul = GgufTensorMatmulProfile::default();
    let mut block_indices = std::collections::BTreeSet::new();
    let mut block_tensor_count = 0u64;
    let mut has_token_embedding_tensor = false;
    let mut has_output_tensor = false;
    let mut has_output_norm_tensor = false;
    for (index, tensor) in tensors.iter().enumerate() {
        let next_offset = tensors
            .get(index + 1)
            .map(|next| next.offset)
            .unwrap_or(data_len);
        if next_offset < tensor.offset || next_offset > data_len {
            return None;
        }
        let tensor_bytes = next_offset - tensor.offset;
        let tensor_group = classify_tensor_group(&tensor.name);
        add_tensor_group_bytes(&mut group_bytes, tensor_group, tensor_bytes);
        add_matmul_profile(&mut matmul, tensor, tensor_group, tensor_bytes);
        if is_expert_partitioned_tensor(&tensor.name) {
            expert_tensor_bytes = expert_tensor_bytes.saturating_add(tensor_bytes);
        } else {
            base_resident_bytes = base_resident_bytes.saturating_add(tensor_bytes);
        }
        if let Some(block_index) = tensor_block_index(&tensor.name) {
            block_indices.insert(block_index);
            block_tensor_count = block_tensor_count.saturating_add(1);
        }
        has_token_embedding_tensor |= is_token_embedding_tensor(&tensor.name);
        has_output_tensor |= is_output_tensor(&tensor.name);
        has_output_norm_tensor |= is_output_norm_tensor(&tensor.name);
    }

    let file_overhead_bytes = file_len.saturating_sub(base_resident_bytes + expert_tensor_bytes);
    Some(GgufTensorByteProfile {
        tensor_count: n_tensors.try_into().unwrap_or(u64::MAX),
        block_tensor_count,
        distinct_block_count: block_indices.len().try_into().unwrap_or(u32::MAX),
        has_token_embedding_tensor,
        has_output_tensor,
        has_output_norm_tensor,
        expert_count,
        expert_used_count,
        full_model_bytes: file_len,
        base_resident_bytes,
        expert_tensor_bytes,
        group_bytes,
        matmul,
        file_overhead_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_file_path(prefix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{unique}.gguf"))
    }

    fn write_bytes(prefix: &str, bytes: &[u8]) -> PathBuf {
        let path = temp_file_path(prefix);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        file.flush().unwrap();
        path
    }

    fn push_array_header(bytes: &mut Vec<u8>, elem_type: GgufType, count: u64) {
        bytes.extend_from_slice(&(elem_type as u32).to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
    }

    fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_u32_kv(bytes: &mut Vec<u8>, key: &str, value: u32) {
        push_gguf_string(bytes, key);
        bytes.extend_from_slice(&(GgufType::Uint32 as u32).to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_tensor_info(bytes: &mut Vec<u8>, name: &str, offset: u64) {
        push_gguf_string(bytes, name);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&16u64.to_le_bytes());
        bytes.extend_from_slice(&(GgufType::Uint8 as u32).to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
    }

    fn push_tensor_info_2d(bytes: &mut Vec<u8>, name: &str, input: u64, output: u64, offset: u64) {
        push_gguf_string(bytes, name);
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&input.to_le_bytes());
        bytes.extend_from_slice(&output.to_le_bytes());
        bytes.extend_from_slice(&(GgufType::Uint8 as u32).to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
    }

    #[test]
    fn skip_gguf_value_rejects_excessive_array_depth() {
        let mut bytes = Vec::new();
        for _ in 0..=MAX_GGUF_ARRAY_DEPTH {
            push_array_header(&mut bytes, GgufType::Array, 1);
        }
        push_array_header(&mut bytes, GgufType::Uint8, 1);
        bytes.push(0);

        let path = write_bytes("model-artifact-gguf-depth", &bytes);
        let mut file = std::fs::File::open(&path).unwrap();
        let err = skip_gguf_value(&mut file, GgufType::Array).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("nesting too deep"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn skip_gguf_value_rejects_excessive_array_count() {
        let mut bytes = Vec::new();
        push_array_header(&mut bytes, GgufType::Uint8, MAX_GGUF_ARRAY_ELEMENTS + 1);

        let path = write_bytes("model-artifact-gguf-count", &bytes);
        let mut file = std::fs::File::open(&path).unwrap();
        let err = skip_gguf_value(&mut file, GgufType::Array).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("array too long"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_compact_meta_returns_none_on_malicious_nested_array() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&1i64.to_le_bytes());
        push_gguf_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&(GgufType::Array as u32).to_le_bytes());
        for _ in 0..=MAX_GGUF_ARRAY_DEPTH {
            push_array_header(&mut bytes, GgufType::Array, 1);
        }
        push_array_header(&mut bytes, GgufType::Uint8, 1);
        bytes.push(0);

        let path = write_bytes("model-artifact-gguf-malicious", &bytes);
        assert!(scan_gguf_compact_meta(&path).is_none());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_compact_meta_derives_value_length_from_kv_heads_without_head_count() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&2i64.to_le_bytes());
        push_u32_kv(&mut bytes, "llama.embedding_length", 4096);
        push_u32_kv(&mut bytes, "llama.attention.head_count_kv", 8);

        let path = write_bytes("model-artifact-gguf-kv-heads", &bytes);
        let meta = scan_gguf_compact_meta(&path).expect("should parse GGUF");
        assert_eq!(meta.head_count, 0);
        assert_eq!(meta.kv_head_count, 8);
        assert_eq!(meta.key_length, 0);
        assert_eq!(meta.value_length, 512);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_compact_meta_preserves_kv_head_count() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&6i64.to_le_bytes());
        push_u32_kv(&mut bytes, "llama.embedding_length", 4096);
        push_u32_kv(&mut bytes, "llama.attention.head_count", 32);
        push_u32_kv(&mut bytes, "llama.attention.head_count_kv", 8);
        push_u32_kv(&mut bytes, "llama.block_count", 24);
        push_u32_kv(&mut bytes, "llama.attention.key_length", 128);
        push_u32_kv(&mut bytes, "llama.attention.value_length", 128);

        let path = write_bytes("model-artifact-gguf-kv-head-count", &bytes);
        let meta = scan_gguf_compact_meta(&path).expect("should parse GGUF");
        assert_eq!(meta.head_count, 32);
        assert_eq!(meta.kv_head_count, 8);
        assert_eq!(meta.effective_kv_head_count(), Some(8));
        assert_eq!(meta.k_cache_bytes_per_token_f16(), Some(49_152));
        assert_eq!(meta.v_cache_bytes_per_token_f16(), Some(49_152));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn kv_cache_quant_prices_key_and_value_types_independently() {
        let meta = GgufCompactMeta {
            head_count: 32,
            kv_head_count: 8,
            layer_count: 24,
            key_length: 128,
            value_length: 128,
            ..Default::default()
        };
        let quant = GgufKvCacheQuant::new(GgufKvCacheType::Q8_0, GgufKvCacheType::Q4_0);

        assert_eq!(quant.k_cache_bytes_per_token(&meta), Some(26_112));
        assert_eq!(quant.v_cache_bytes_per_token(&meta), Some(13_824));
        assert_eq!(quant.kv_cache_bytes_per_token(&meta), Some(39_936));
    }

    #[test]
    fn kv_cache_quant_prices_key_and_value_widths_independently() {
        let meta = GgufCompactMeta {
            head_count: 32,
            kv_head_count: 8,
            layer_count: 24,
            key_length: 64,
            value_length: 256,
            ..Default::default()
        };
        let quant = GgufKvCacheQuant::new(GgufKvCacheType::Q8_0, GgufKvCacheType::Q4_0);

        assert_eq!(quant.k_cache_bytes_per_token(&meta), Some(13_056));
        assert_eq!(quant.v_cache_bytes_per_token(&meta), Some(27_648));
        assert_eq!(quant.kv_cache_bytes_per_token(&meta), Some(40_704));
    }

    #[test]
    fn kv_cache_bytes_per_token_returns_none_when_required_fields_are_missing() {
        let meta = GgufCompactMeta {
            head_count: 32,
            layer_count: 24,
            key_length: 128,
            ..Default::default()
        };

        assert_eq!(meta.k_cache_bytes_per_token_f16(), Some(196_608));
        assert_eq!(meta.v_cache_bytes_per_token_f16(), None);
        assert_eq!(
            GgufKvCacheQuant::f16().kv_cache_bytes_per_token(&meta),
            None
        );
    }

    #[test]
    fn scan_gguf_compact_meta_rejects_negative_kv_count() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&(-1i64).to_le_bytes());

        let path = write_bytes("model-artifact-gguf-negative-kv", &bytes);
        assert!(scan_gguf_compact_meta(&path).is_none());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_tensor_byte_profile_rejects_excessive_tensor_count() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&((MAX_GGUF_TENSOR_COUNT as i64) + 1).to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());

        let path = write_bytes("model-artifact-gguf-too-many-tensors", &bytes);
        assert!(scan_gguf_tensor_byte_profile(&path).is_none());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn read_gguf_value_as_u32_rejects_negative_int32() {
        let path = write_bytes("model-artifact-gguf-negative-int32", &(-1i32).to_le_bytes());
        let mut file = std::fs::File::open(&path).unwrap();
        let err = read_gguf_value_as_u32(&mut file, GgufType::Int32).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string()
                .contains("negative Int32 where unsigned GGUF value was expected")
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_tensor_byte_profile_splits_base_and_expert_bytes() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&2i64.to_le_bytes());
        bytes.extend_from_slice(&3i64.to_le_bytes());

        push_u32_kv(&mut bytes, "general.alignment", 32);
        push_u32_kv(&mut bytes, "llama.expert_count", 8);
        push_u32_kv(&mut bytes, "llama.expert_used_count", 2);

        push_tensor_info(&mut bytes, "blk.0.ffn_up_exps.weight", 0);
        push_tensor_info(&mut bytes, "blk.0.attn_q.weight", 64);

        let data_start = align_offset(bytes.len() as u64, 32) as usize;
        bytes.resize(data_start, 0);
        bytes.resize(data_start + 96, 0);

        let path = write_bytes("model-artifact-gguf-tensors", &bytes);
        let profile = scan_gguf_tensor_byte_profile(&path).unwrap();
        assert_eq!(profile.expert_count, 8);
        assert_eq!(profile.expert_used_count, 2);
        assert_eq!(profile.expert_tensor_bytes, 64);
        assert_eq!(profile.base_resident_bytes, 32);
        assert_eq!(profile.group_bytes.expert_feed_forward_bytes, 64);
        assert_eq!(profile.group_bytes.attention_bytes, 32);
        assert_eq!(profile.matmul.expert_bytes, 64);
        assert_eq!(profile.matmul.base_bytes, 32);
        assert_eq!(profile.matmul.expert_flops_per_token, 32);
        assert_eq!(profile.matmul.base_flops_per_token, 32);
        assert_eq!(profile.matmul.expert_type_bytes.f32_bytes, 64);
        assert_eq!(profile.matmul.base_type_bytes.f32_bytes, 32);
        assert_eq!(profile.matmul.expert_feed_forward.bytes, 64);
        assert_eq!(profile.matmul.expert_feed_forward.flops_per_token, 32);
        assert_eq!(profile.matmul.expert_feed_forward.type_bytes.f32_bytes, 64);
        assert_eq!(profile.matmul.attention.bytes, 32);
        assert_eq!(profile.matmul.attention.flops_per_token, 32);
        assert_eq!(profile.matmul.attention.type_bytes.f32_bytes, 32);
        assert_eq!(profile.full_model_bytes, bytes.len() as u64);
        assert_eq!(
            profile.full_model_bytes,
            profile.base_resident_bytes + profile.expert_tensor_bytes + profile.file_overhead_bytes
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_tensor_byte_profile_records_matmul_shapes() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&2i64.to_le_bytes());
        bytes.extend_from_slice(&1i64.to_le_bytes());

        push_u32_kv(&mut bytes, "general.alignment", 32);
        push_tensor_info_2d(&mut bytes, "blk.0.ffn_up.weight", 4, 16, 0);
        push_tensor_info_2d(&mut bytes, "blk.0.ffn_down.weight", 16, 4, 64);

        let data_start = align_offset(bytes.len() as u64, 32) as usize;
        bytes.resize(data_start, 0);
        bytes.resize(data_start + 96, 0);

        let path = write_bytes("model-artifact-gguf-matmul-shapes", &bytes);
        let profile = scan_gguf_tensor_byte_profile(&path).unwrap();
        let shape = profile.matmul.feed_forward.shape;

        assert_eq!(shape.tensor_count, 2);
        assert_eq!(shape.logical_matrix_count, 2);
        assert_eq!(shape.total_elements, 128);
        assert_eq!(shape.min_input_width, 4);
        assert_eq!(shape.max_input_width, 16);
        assert_eq!(shape.min_output_width, 4);
        assert_eq!(shape.max_output_width, 16);
        assert_eq!(shape.weighted_avg_input_width, 10);
        assert_eq!(shape.weighted_avg_output_width, 10);
        let _ = std::fs::remove_file(path);
    }
}
