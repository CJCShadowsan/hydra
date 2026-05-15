//! Speculative prefill: verify a draft response in one prefill pass.
//!
//! When a request includes `draft_response` (text) or `draft_tokens`
//! (pre-tokenized IDs), this module verifies the draft against the
//! target model in a single batched forward pass. Accepted prefix
//! tokens are committed without decode; only the tail after the first
//! divergence is decoded normally.
//!
//! `draft_tokens` avoids the tokenizer round-trip problem where
//! `tokenize(detokenize(tokens)) != tokens` for some subword
//! sequences. When both fields are set, `draft_tokens` takes priority.

use super::*;

/// Result of speculative prefill verification.
pub(super) struct SpecPrefillResult {
    /// Draft token IDs that were fed to verify_tokens. Callers use
    /// this to emit the accepted prefix via `on_token`.
    pub(super) draft_token_ids: Vec<i32>,
    /// Number of draft tokens that were accepted (contiguous prefix).
    pub(super) accepted_tokens: usize,
    /// Total draft tokens that were verified.
    pub(super) total_draft_tokens: usize,
    /// The target model's predicted token at the divergence point.
    /// This is the first token the target would generate that differs
    /// from the draft — it becomes `current` for the decode loop.
    /// None if the draft was fully accepted.
    pub(super) divergence_token: Option<i32>,
    /// Whether the draft was fully accepted (no decode needed).
    pub(super) fully_accepted: bool,
    /// Time spent tokenizing the draft (0 when `draft_tokens` used).
    pub(super) tokenize_ms: f64,
    /// Time spent in verify_tokens.
    pub(super) verify_ms: f64,
}

/// Verify draft token IDs (or text) against the target model.
///
/// Call this after prefilling the prompt tokens but before the decode
/// loop. The session's KV cache is warm from the prompt prefill. We
/// extend it by verifying the draft tokens in one batched forward pass.
///
/// Returns `None` if the draft is empty or produces no tokens.
pub(super) fn verify_draft(
    runtime: &Arc<Mutex<RuntimeState>>,
    session_id: &str,
    draft_tokens: Option<&[i32]>,
    draft_text: Option<&str>,
) -> OpenAiResult<Option<SpecPrefillResult>> {
    // Resolve draft token IDs: prefer pre-tokenized, fall back to text.
    let (draft_token_ids, tokenize_ms) = if let Some(tokens) = draft_tokens {
        if tokens.is_empty() {
            return Ok(None);
        }
        (tokens.to_vec(), 0.0)
    } else if let Some(text) = draft_text {
        if text.trim().is_empty() {
            return Ok(None);
        }
        let tokenize_timer = PhaseTimer::start();
        let ids = {
            let rt = runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            rt.model
                .tokenize(text, false)
                .map_err(openai_backend_error)?
        };
        let ms = tokenize_timer.elapsed_ms();
        if ids.is_empty() {
            return Ok(None);
        }
        (ids, ms)
    } else {
        return Ok(None);
    };

    // Verify: run all draft tokens through the model in one batch.
    // The model computes logits at every position and returns its
    // greedy prediction at each.
    let verify_timer = PhaseTimer::start();
    let predicted = {
        let mut rt = runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        rt.verify_tokens(session_id, &draft_token_ids)
            .map_err(openai_backend_error)?
    };
    let verify_ms = verify_timer.elapsed_ms();

    // Find the first divergence: compare predicted[i] vs draft_token_ids[i+1].
    // predicted[i] is what the target model would emit after seeing tokens 0..=i.
    // draft_token_ids[i+1] is the next draft token.
    let mut accepted = 0usize;
    let compare_len = draft_token_ids.len().saturating_sub(1);
    for i in 0..compare_len {
        if predicted[i] == draft_token_ids[i + 1] {
            accepted += 1;
        } else {
            break;
        }
    }

    // Check if the last predicted token is EOG (draft might be complete).
    let last_predicted_is_eog = if !predicted.is_empty() {
        let rt = runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        rt.model
            .token_is_eog(*predicted.last().unwrap())
            .map_err(openai_backend_error)?
    } else {
        false
    };

    let fully_accepted = accepted == compare_len && (compare_len > 0 || last_predicted_is_eog);

    let divergence_token = if fully_accepted {
        // The model agreed with all draft tokens. The predicted token
        // after the last draft token is the natural continuation.
        predicted.last().copied()
    } else {
        // The model diverged at position `accepted`. Its prediction
        // there is what it would generate instead of the draft.
        Some(predicted[accepted])
    };

    let total_draft_tokens = draft_token_ids.len();
    Ok(Some(SpecPrefillResult {
        draft_token_ids,
        accepted_tokens: accepted,
        total_draft_tokens,
        divergence_token,
        fully_accepted,
        tokenize_ms,
        verify_ms,
    }))
}
