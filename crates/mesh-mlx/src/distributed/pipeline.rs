//! Pipeline-parallel layer assignment + activation hand-off.
//!
//! Layers are split contiguously across ranks. Following MLX's `PipelineMixin`,
//! the split is reversed so **rank 0 owns the final layers** (and the lm_head),
//! which lets rank 0 produce the logits after receiving the pipeline output.
//!
//! Per forward pass on rank `r` (size `N`):
//!   - if `r < N-1`: `recv_like` the hidden state from rank `r+1`
//!   - run this rank's layers
//!   - if `r != 0`: `send` the hidden state to rank `r-1`
//!   - rank 0 finishes (norm + lm_head), then everyone `all_gather`s logits.
//!
//! This module computes the layer ownership; the model's forward pass calls the
//! `Group` send/recv at the boundaries.

use crate::distributed::Group;

/// The contiguous layer range a rank owns, under reverse assignment.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct LayerRange {
    /// First global layer index owned (inclusive).
    pub start: usize,
    /// One past the last global layer index owned (exclusive).
    pub end: usize,
}

impl LayerRange {
    pub fn len(&self) -> usize {
        self.end - self.start
    }
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

/// Pipeline topology derived from rank/size and total layer count.
#[derive(Clone, Debug)]
pub struct Pipeline {
    pub rank: i32,
    pub size: i32,
    pub total_layers: usize,
    pub range: LayerRange,
}

impl Pipeline {
    /// Compute this rank's layer ownership. Reverse assignment: the highest
    /// rank gets the first layers, rank 0 gets the last layers.
    pub fn plan(rank: i32, size: i32, total_layers: usize) -> Self {
        let size = size.max(1);
        let n = total_layers;
        let per = n / size as usize;
        let rem = n % size as usize;

        // Forward chunk index for this rank (rank 0 -> last chunk).
        // chunk i (i in 0..size) covers a contiguous block; reverse so rank 0
        // maps to the last chunk.
        let chunk = (size - 1 - rank.clamp(0, size - 1)) as usize;

        // Distribute remainder to the earliest chunks for balance.
        let start = chunk * per + chunk.min(rem);
        let this = per + if chunk < rem { 1 } else { 0 };
        let end = start + this;

        Pipeline {
            rank,
            size,
            total_layers: n,
            range: LayerRange { start, end },
        }
    }

    /// From a live [`Group`].
    pub fn from_group(group: &Group, total_layers: usize) -> Self {
        Self::plan(group.rank(), group.size(), total_layers)
    }

    /// Is this the last stage in the forward direction (owns the first layers)?
    pub fn is_first_forward_stage(&self) -> bool {
        self.range.start == 0
    }

    /// Does this rank produce the final output (rank 0 owns the last layers)?
    pub fn is_output_stage(&self) -> bool {
        self.rank == 0
    }

    /// Rank to receive activations from (the rank owning the next-earlier
    /// layers), or `None` if this rank owns the first layers.
    pub fn recv_from(&self) -> Option<i32> {
        if self.is_first_forward_stage() || self.size <= 1 {
            None
        } else {
            Some(self.rank + 1)
        }
    }

    /// Rank to send activations to (the rank owning the next-later layers),
    /// or `None` if this is the output stage (rank 0).
    pub fn send_to(&self) -> Option<i32> {
        if self.is_output_stage() || self.size <= 1 {
            None
        } else {
            Some(self.rank - 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_rank_owns_all() {
        let p = Pipeline::plan(0, 1, 32);
        assert_eq!(p.range, LayerRange { start: 0, end: 32 });
        assert!(p.is_first_forward_stage());
        assert!(p.is_output_stage());
        assert_eq!(p.recv_from(), None);
        assert_eq!(p.send_to(), None);
    }

    #[test]
    fn reverse_assignment_rank0_owns_last_layers() {
        // 4 ranks, 32 layers -> 8 each. rank 0 owns 24..32, rank 3 owns 0..8.
        let p0 = Pipeline::plan(0, 4, 32);
        let p3 = Pipeline::plan(3, 4, 32);
        assert_eq!(p0.range, LayerRange { start: 24, end: 32 });
        assert_eq!(p3.range, LayerRange { start: 0, end: 8 });
        assert!(p0.is_output_stage());
        assert!(p3.is_first_forward_stage());
    }

    #[test]
    fn full_partition_is_contiguous_and_covers_all() {
        let (size, layers) = (3, 11);
        let mut covered = vec![false; layers];
        for r in 0..size {
            let p = Pipeline::plan(r, size, layers);
            for slot in covered.iter_mut().take(p.range.end).skip(p.range.start) {
                assert!(!*slot, "layer double-owned");
                *slot = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "every layer owned exactly once");
    }

    #[test]
    fn neighbours_chain_correctly() {
        // rank r receives from r+1 and sends to r-1 (reverse pipeline).
        let p1 = Pipeline::plan(1, 4, 32);
        assert_eq!(p1.recv_from(), Some(2));
        assert_eq!(p1.send_to(), Some(0));
    }
}
