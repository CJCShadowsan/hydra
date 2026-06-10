//! Distributed inference primitives: process group + collectives ([`group`])
//! and pipeline layer assignment ([`pipeline`]).

mod group;
mod pipeline;

pub use group::{Backend, Group};
pub use pipeline::{LayerRange, Pipeline};
