//! Neural-net building blocks: weight store + Linear / Embedding / RMSNorm,
//! plus the distributed sharded-linear variants used for tensor parallelism.
//!
//! These mirror the Python `mlx.nn` layers but are driven from Rust. A sharded
//! linear is exactly the Python pattern: split the weight, do the local matmul,
//! and (for the sharded-to-all case) `all_sum` across the group.

use crate::array::{Array, Stream};
use crate::distributed::Group;
use crate::ops;
use crate::{MlxError, Result};
use std::collections::HashMap;

/// A bag of named tensors (a loaded model's parameters).
#[derive(Default)]
pub struct Weights {
    map: HashMap<String, Array>,
}

impl Weights {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: impl Into<String>, value: Array) {
        self.map.insert(name.into(), value);
    }

    pub fn get(&self, name: &str) -> Result<&Array> {
        self.map
            .get(name)
            .ok_or_else(|| MlxError::MissingWeight(name.to_string()))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.map.keys()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// A dense linear layer: `y = x @ Wᵀ (+ b)`. Weight is stored `[out, in]`
/// (HF/safetensors convention), so we matmul against its transpose.
pub struct Linear<'w> {
    weight: &'w Array,
    bias: Option<&'w Array>,
}

impl<'w> Linear<'w> {
    /// Look up `"{prefix}.weight"` (and optional `"{prefix}.bias"`).
    pub fn load(w: &'w Weights, prefix: &str) -> Result<Self> {
        let weight = w.get(&format!("{prefix}.weight"))?;
        let bias = w.get(&format!("{prefix}.bias")).ok();
        Ok(Linear { weight, bias })
    }

    pub fn forward(&self, x: &Array, s: &Stream) -> Result<Array> {
        // weight is [out, in]; transpose to [in, out] for x[..,in] @ [in,out].
        let wt = ops::transpose(self.weight, &transpose_axes(self.weight.ndim()), s)?;
        let mut y = ops::matmul(x, &wt, s)?;
        if let Some(b) = self.bias {
            y = ops::add(&y, b, s)?;
        }
        Ok(y)
    }
}

/// Token embedding: gather rows of the `[vocab, dim]` table.
pub struct Embedding<'w> {
    weight: &'w Array,
}

impl<'w> Embedding<'w> {
    pub fn load(w: &'w Weights, prefix: &str) -> Result<Self> {
        Ok(Embedding {
            weight: w.get(&format!("{prefix}.weight"))?,
        })
    }

    /// `ids` is an `i32` array of token ids.
    pub fn forward(&self, ids: &Array, s: &Stream) -> Result<Array> {
        ops::take(self.weight, ids, 0, s)
    }

    /// As-linear projection (weight tying for the LM head): `x @ weightᵀ`.
    pub fn as_linear(&self, x: &Array, s: &Stream) -> Result<Array> {
        let wt = ops::transpose(self.weight, &transpose_axes(self.weight.ndim()), s)?;
        ops::matmul(x, &wt, s)
    }
}

/// RMS normalisation with a learned weight.
pub struct RmsNorm<'w> {
    weight: &'w Array,
    eps: f32,
}

impl<'w> RmsNorm<'w> {
    pub fn load(w: &'w Weights, prefix: &str, eps: f32) -> Result<Self> {
        Ok(RmsNorm {
            weight: w.get(&format!("{prefix}.weight"))?,
            eps,
        })
    }

    pub fn forward(&self, x: &Array, s: &Stream) -> Result<Array> {
        ops::rms_norm(x, self.weight, self.eps, s)
    }
}

/// How a linear is sharded for tensor parallelism.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ShardKind {
    /// Output dim split across ranks; result is sharded (no comm in forward).
    /// Used for q/k/v/gate/up projections.
    AllToSharded,
    /// Input dim split across ranks; forward does an `all_sum` so every rank
    /// holds the full result. Used for o_proj / down_proj.
    ShardedToAll,
}

/// A tensor-parallel linear. The weight is assumed to already be the local
/// shard (the loader slices it per rank). For [`ShardKind::ShardedToAll`] the
/// forward pass reduces partial results across the group with `all_sum`.
pub struct ShardedLinear<'w> {
    inner: Linear<'w>,
    kind: ShardKind,
    group: &'w Group,
}

impl<'w> ShardedLinear<'w> {
    pub fn new(inner: Linear<'w>, kind: ShardKind, group: &'w Group) -> Self {
        ShardedLinear { inner, kind, group }
    }

    pub fn forward(&self, x: &Array, s: &Stream) -> Result<Array> {
        let y = self.inner.forward(x, s)?;
        match self.kind {
            ShardKind::AllToSharded => Ok(y),
            ShardKind::ShardedToAll => self.group.all_sum(&y, s),
        }
    }
}

/// Axes that reverse the last two dims (transpose of a 2D weight), keeping any
/// leading dims in place. For a 2D weight this is `[1, 0]`.
fn transpose_axes(ndim: usize) -> Vec<i32> {
    if ndim < 2 {
        return (0..ndim as i32).collect();
    }
    let mut axes: Vec<i32> = (0..ndim as i32).collect();
    axes.swap(ndim - 1, ndim - 2);
    axes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_axes_swaps_last_two() {
        assert_eq!(transpose_axes(2), vec![1, 0]);
        assert_eq!(transpose_axes(3), vec![0, 2, 1]);
        assert_eq!(transpose_axes(1), vec![0]);
    }

    #[test]
    fn weights_missing_is_error() {
        let w = Weights::new();
        assert!(w.get("nope").is_err());
        assert!(w.is_empty());
    }
}
