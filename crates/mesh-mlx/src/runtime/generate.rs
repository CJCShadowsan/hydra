//! Token generation loop with single-node and pipeline-parallel execution.
//!
//! The forward pass per step, on rank `r` of a pipeline of size `N`:
//!   1. The first-forward stage (highest rank) embeds the input ids.
//!      Other stages `recv_like` the hidden state from rank `r+1`.
//!   2. Run this rank's owned layers.
//!   3. If not the output stage, `send` the hidden state to rank `r-1`.
//!   4. The output stage (rank 0) runs norm + lm_head → logits, samples the
//!      next token, then `all_gather` distributes it so every stage advances.
//!
//! For `N == 1` this collapses to a plain local forward.

use crate::Result;
use crate::array::{Array, Stream};
use crate::distributed::{Group, Pipeline};
use crate::models::{LayerCache, LlamaModel};
use crate::ops;

/// Greedy argmax sampling over the last position's logits.
fn sample_greedy(logits: &Array, s: &Stream) -> Result<i32> {
    // logits: [B, L, vocab] — flatten to [B*L, vocab] and argmax each row;
    // the last row corresponds to the most recent position.
    let shape = logits.shape();
    let (b, l, vocab) = (shape[0], shape[1], shape[2]);
    let flat = ops::reshape(logits, &[b * l, vocab], s)?;
    let arg = ops::argmax(&flat, s)?; // [b*l]
    let ids = arg.to_vec_i32()?;
    Ok(*ids.last().unwrap_or(&0))
}

/// One decode step. Returns the next token id (valid on every rank after the
/// `all_gather`). `input` is the hidden-state input for non-first stages and the
/// token-id array for the first-forward stage.
#[allow(clippy::too_many_arguments)]
pub fn decode_step(
    model: &LlamaModel<'_>,
    pipeline: &Pipeline,
    group: Option<&Group>,
    token_ids: &Array,
    cache: &mut [LayerCache],
    s: &Stream,
) -> Result<i32> {
    // Stage input: first-forward stage embeds tokens; others receive.
    let mut h = if pipeline.is_first_forward_stage() {
        model.embed(token_ids, s)?
    } else {
        // Receive from the next-earlier stage. We need a template shaped like
        // the hidden state: embed a zero-length placeholder is awkward, so for
        // the first cut single-node path this branch is exercised only when a
        // group is present.
        let src = pipeline.recv_from().expect("non-first stage has a source");
        let template = model.embed(token_ids, s)?; // same [B,L,D] shape
        group
            .expect("pipeline requires a group")
            .recv_like(&template, src, s)?
    };

    h = model.forward_layers(h, cache, s)?;

    if let Some(dst) = pipeline.send_to() {
        // Hand off to the next-later stage; this rank is done this step.
        let g = group.expect("pipeline requires a group");
        let dep = g.send(&h, dst, s)?;
        dep.eval()?;
        // Non-output stages still need the chosen token to advance their cache
        // offset; the output stage broadcasts it via all_gather below.
    }

    // Output stage computes logits + samples.
    let next = if pipeline.is_output_stage() {
        let logits = model.head(&h, s)?;
        sample_greedy(&logits, s)?
    } else {
        0
    };

    // Broadcast the chosen token to all ranks.
    let next = if let Some(g) = group {
        if pipeline.size > 1 {
            let tok = Array::from_i32(&[next], &[1])?;
            let gathered = g.all_gather(&tok, s)?;
            let all = gathered.to_vec_i32()?;
            // The output stage (rank 0) holds the real token at its slot.
            *all.first().unwrap_or(&next)
        } else {
            next
        }
    } else {
        next
    };

    Ok(next)
}

/// Generate up to `max_tokens` greedily from a prompt token sequence,
/// single-node (no group). Returns the generated token ids.
pub fn generate_local(
    model: &LlamaModel<'_>,
    pipeline: &Pipeline,
    prompt_ids: &[i32],
    max_tokens: usize,
    eos: impl Fn(i32) -> bool,
    s: &Stream,
) -> Result<Vec<i32>> {
    let mut cache = model.new_cache();

    // Prefill the prompt in one forward.
    let ids = Array::from_i32(prompt_ids, &[1, prompt_ids.len() as i32])?;
    let h = model.embed(&ids, s)?;
    let h = model.forward_layers(h, &mut cache, s)?;
    let logits = model.head(&h, s)?;
    let mut next = sample_greedy(&logits, s)?;

    let mut out = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        if eos(next) {
            break;
        }
        out.push(next);
        // Decode one token at a time.
        let step_ids = Array::from_i32(&[next], &[1, 1])?;
        next = decode_step(model, pipeline, None, &step_ids, &mut cache, s)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    // Sampling/forward require the engine; these are exercised by the
    // `link-mlx` integration test. Pure-logic coverage lives in the pipeline,
    // loader, planner, and config modules.
}
