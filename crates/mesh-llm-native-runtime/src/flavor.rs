use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativeRuntimeFlavor {
    Cpu,
    Metal,
    Cuda,
    CudaBlackwell,
    Rocm,
    Vulkan,
    Other(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeRuntimeFlavorParseError {
    value: String,
}

impl fmt::Display for NativeRuntimeFlavorParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid native runtime flavor '{}'", self.value)
    }
}

impl std::error::Error for NativeRuntimeFlavorParseError {}

impl NativeRuntimeFlavor {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::CudaBlackwell => "cuda-blackwell",
            Self::Rocm => "rocm",
            Self::Vulkan => "vulkan",
            Self::Other(value) => value.as_str(),
        }
    }

    pub fn default_rank(&self) -> i64 {
        match self {
            Self::CudaBlackwell => 700,
            Self::Cuda => 650,
            Self::Rocm => 600,
            Self::Metal => 600,
            Self::Vulkan => 350,
            Self::Cpu => 100,
            Self::Other(_) => 0,
        }
    }
}

impl fmt::Display for NativeRuntimeFlavor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for NativeRuntimeFlavor {
    type Err = NativeRuntimeFlavorParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let normalized = value.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return Err(NativeRuntimeFlavorParseError {
                value: value.to_string(),
            });
        }
        Ok(match normalized.as_str() {
            "cpu" => Self::Cpu,
            "metal" => Self::Metal,
            "cuda" => Self::Cuda,
            "cuda-blackwell" | "blackwell" => Self::CudaBlackwell,
            "rocm" | "hip" => Self::Rocm,
            "vulkan" => Self::Vulkan,
            _ => Self::Other(normalized),
        })
    }
}

impl From<&str> for NativeRuntimeFlavor {
    fn from(value: &str) -> Self {
        value
            .parse()
            .unwrap_or_else(|_| Self::Other(value.trim().to_ascii_lowercase()))
    }
}
