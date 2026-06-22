use super::*;

#[derive(Default)]
pub(super) struct OpenAiSpeculativeStats {
    pub(super) windows: usize,
    pub(super) draft_tokens: usize,
    pub(super) accepted_tokens: usize,
    pub(super) rejected_tokens: usize,
    pub(super) full_accept_windows: usize,
    pub(super) accepted_stop_windows: usize,
    pub(super) rejected_windows: usize,
    pub(super) early_reject_windows: usize,
    pub(super) tail_reject_windows: usize,
    pub(super) early_reject_stop_windows: usize,
    pub(super) repair_required_windows: usize,
    pub(super) first_reject_position_sum: usize,
    pub(super) primary_verify_requests: usize,
    pub(super) primary_verify_tokens: usize,
    pub(super) primary_verify_elapsed_ms: f64,
    pub(super) primary_verify_stage0_compute_ms: f64,
    pub(super) primary_verify_runtime_lock_wait_ms: f64,
    pub(super) primary_verify_runtime_lock_hold_ms: f64,
    pub(super) primary_verify_activation_encode_ms: f64,
    pub(super) primary_verify_forward_write_ms: f64,
    pub(super) primary_verify_downstream_wait_ms: f64,
    pub(super) primary_verify_output_activation_bytes: usize,
    pub(super) primary_verify_forward_activation_bytes: usize,
    pub(super) checkpoint_ms: f64,
    pub(super) draft_reset_ms: f64,
    pub(super) draft_propose_ms: f64,
    pub(super) recovery_restores: usize,
    pub(super) recovery_ms: f64,
    pub(super) recovery_restore_local_ms: f64,
    pub(super) recovery_restore_downstream_write_ms: f64,
    pub(super) recovery_restore_downstream_wait_ms: f64,
    pub(super) adaptive_window_start: usize,
    pub(super) adaptive_window_final: usize,
    pub(super) adaptive_window_max: usize,
    pub(super) adaptive_window_min: usize,
    pub(super) adaptive_window_max_seen: usize,
    pub(super) adaptive_window_sum: usize,
    pub(super) adaptive_window_grows: usize,
    pub(super) adaptive_window_shrinks: usize,
    pub(super) adaptive_window_enabled: bool,
    pub(super) pipelined_depth: usize,
    pub(super) pipelined_sent_windows: usize,
    pub(super) pipelined_committed_windows: usize,
    pub(super) pipelined_stale_windows: usize,
    pub(super) pipelined_max_inflight_windows: usize,
    pub(super) pipelined_async_draft_windows: usize,
    pub(super) pipelined_stale_draft_windows: usize,
    pub(super) pipelined_async_draft_wait_ms: f64,
    pub(super) pipelined_fifo_return_windows: usize,
    pub(super) pipelined_fifo_return_violations: usize,
    pub(super) pipelined_identity_violations: usize,
    pub(super) tree_windows: usize,
    pub(super) tree_nodes: usize,
    pub(super) tree_gather_ms: f64,
}

impl OpenAiSpeculativeStats {
    pub(super) fn observe_verify_decision(
        &mut self,
        decision: VerifySpanDecision,
        adaptive_window: &mut usize,
        adaptive_enabled: bool,
        max_speculative_window: usize,
    ) {
        self.accepted_tokens += decision.accepted_before_reject;
        if decision.rejected() {
            self.rejected_tokens += 1;
        }
        self.adaptive_window_sum += *adaptive_window;
        self.adaptive_window_min = nonzero_min(self.adaptive_window_min, *adaptive_window);
        self.adaptive_window_max_seen = self.adaptive_window_max_seen.max(*adaptive_window);
        match decision.kind {
            VerifySpanDecisionKind::FullAccept => {
                self.full_accept_windows += 1;
                self.grow_adaptive_window(
                    adaptive_window,
                    adaptive_enabled,
                    max_speculative_window,
                );
            }
            VerifySpanDecisionKind::AcceptedStop => {
                self.accepted_stop_windows += 1;
            }
            VerifySpanDecisionKind::TailReject => {
                self.observe_reject(decision);
                self.tail_reject_windows += 1;
                self.grow_adaptive_window(
                    adaptive_window,
                    adaptive_enabled,
                    max_speculative_window,
                );
            }
            VerifySpanDecisionKind::EarlyReject => {
                self.observe_reject(decision);
                self.early_reject_windows += 1;
                self.repair_required_windows += 1;
                self.shrink_adaptive_window(adaptive_window, adaptive_enabled, decision);
            }
            VerifySpanDecisionKind::EarlyRejectStop => {
                self.observe_reject(decision);
                self.early_reject_windows += 1;
                self.early_reject_stop_windows += 1;
            }
        }
    }

    pub(super) fn observe_reject(&mut self, decision: VerifySpanDecision) {
        if let Some(repair_input_count) = decision.repair_input_count {
            self.rejected_windows += 1;
            self.first_reject_position_sum += repair_input_count;
        }
    }

    pub(super) fn grow_adaptive_window(
        &mut self,
        adaptive_window: &mut usize,
        adaptive_enabled: bool,
        max_speculative_window: usize,
    ) {
        if adaptive_enabled && *adaptive_window < max_speculative_window {
            *adaptive_window += 1;
            self.adaptive_window_grows += 1;
        }
    }

    pub(super) fn shrink_adaptive_window(
        &mut self,
        adaptive_window: &mut usize,
        adaptive_enabled: bool,
        decision: VerifySpanDecision,
    ) {
        if !adaptive_enabled {
            return;
        }
        let Some(repair_input_count) = decision.repair_input_count else {
            return;
        };
        let next_window = (*adaptive_window)
            .saturating_sub(1)
            .max(repair_input_count)
            .max(1);
        if next_window < *adaptive_window {
            *adaptive_window = next_window;
            self.adaptive_window_shrinks += 1;
        }
    }

    pub(super) fn insert_attrs(&self, attrs: &mut BTreeMap<String, Value>) {
        if self.windows == 0 {
            attrs.insert("llama_stage.spec.enabled".to_string(), json!(false));
            return;
        }
        attrs.insert("llama_stage.spec.enabled".to_string(), json!(true));
        attrs.insert("llama_stage.spec.windows".to_string(), json!(self.windows));
        attrs.insert(
            "llama_stage.spec.proposed".to_string(),
            json!(self.draft_tokens),
        );
        attrs.insert(
            "llama_stage.spec.accepted".to_string(),
            json!(self.accepted_tokens),
        );
        attrs.insert(
            "llama_stage.spec.rejected".to_string(),
            json!(self.rejected_tokens),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_requests".to_string(),
            json!(self.primary_verify_requests),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_tokens".to_string(),
            json!(self.primary_verify_tokens),
        );
        attrs.insert(
            "llama_stage.spec.accept_rate".to_string(),
            json!(if self.draft_tokens == 0 {
                0.0
            } else {
                self.accepted_tokens as f64 / self.draft_tokens as f64
            }),
        );
        attrs.insert(
            "llama_stage.spec.full_accept_windows".to_string(),
            json!(self.full_accept_windows),
        );
        attrs.insert(
            "llama_stage.spec.accepted_stop_windows".to_string(),
            json!(self.accepted_stop_windows),
        );
        attrs.insert(
            "llama_stage.spec.rejected_windows".to_string(),
            json!(self.rejected_windows),
        );
        attrs.insert(
            "llama_stage.spec.early_reject_windows".to_string(),
            json!(self.early_reject_windows),
        );
        attrs.insert(
            "llama_stage.spec.tail_reject_windows".to_string(),
            json!(self.tail_reject_windows),
        );
        attrs.insert(
            "llama_stage.spec.repair_required_windows".to_string(),
            json!(self.repair_required_windows),
        );
        attrs.insert(
            "llama_stage.spec.draft_reset_ms".to_string(),
            json!(self.draft_reset_ms),
        );
        attrs.insert(
            "llama_stage.spec.draft_propose_ms".to_string(),
            json!(self.draft_propose_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_elapsed_ms".to_string(),
            json!(self.primary_verify_elapsed_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_stage0_compute_ms".to_string(),
            json!(self.primary_verify_stage0_compute_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_runtime_lock_wait_ms".to_string(),
            json!(self.primary_verify_runtime_lock_wait_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_runtime_lock_hold_ms".to_string(),
            json!(self.primary_verify_runtime_lock_hold_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_activation_encode_ms".to_string(),
            json!(self.primary_verify_activation_encode_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_forward_write_ms".to_string(),
            json!(self.primary_verify_forward_write_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_downstream_wait_ms".to_string(),
            json!(self.primary_verify_downstream_wait_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_output_activation_bytes".to_string(),
            json!(self.primary_verify_output_activation_bytes),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_forward_activation_bytes".to_string(),
            json!(self.primary_verify_forward_activation_bytes),
        );
        attrs.insert(
            "llama_stage.spec.checkpoint_ms".to_string(),
            json!(self.checkpoint_ms),
        );
        attrs.insert(
            "llama_stage.spec.recovery_restores".to_string(),
            json!(self.recovery_restores),
        );
        attrs.insert(
            "llama_stage.spec.recovery_ms".to_string(),
            json!(self.recovery_ms),
        );
        attrs.insert(
            "llama_stage.spec.recovery_restore_local_ms".to_string(),
            json!(self.recovery_restore_local_ms),
        );
        attrs.insert(
            "llama_stage.spec.recovery_restore_downstream_write_ms".to_string(),
            json!(self.recovery_restore_downstream_write_ms),
        );
        attrs.insert(
            "llama_stage.spec.recovery_restore_downstream_wait_ms".to_string(),
            json!(self.recovery_restore_downstream_wait_ms),
        );
        attrs.insert(
            "llama_stage.spec.adaptive_enabled".to_string(),
            json!(self.adaptive_window_enabled),
        );
        attrs.insert(
            "llama_stage.spec.window_start".to_string(),
            json!(self.adaptive_window_start),
        );
        attrs.insert(
            "llama_stage.spec.window_final".to_string(),
            json!(self.adaptive_window_final),
        );
        attrs.insert(
            "llama_stage.spec.window_max".to_string(),
            json!(self.adaptive_window_max),
        );
        attrs.insert(
            "llama_stage.spec.window_min".to_string(),
            json!(self.adaptive_window_min),
        );
        attrs.insert(
            "llama_stage.spec.window_max_seen".to_string(),
            json!(self.adaptive_window_max_seen),
        );
        attrs.insert(
            "llama_stage.spec.window_grows".to_string(),
            json!(self.adaptive_window_grows),
        );
        attrs.insert(
            "llama_stage.spec.window_shrinks".to_string(),
            json!(self.adaptive_window_shrinks),
        );
        attrs.insert(
            "llama_stage.spec.pipelined_depth".to_string(),
            json!(self.pipelined_depth),
        );
        attrs.insert(
            "llama_stage.spec.pipelined_sent_windows".to_string(),
            json!(self.pipelined_sent_windows),
        );
        attrs.insert(
            "llama_stage.spec.pipelined_committed_windows".to_string(),
            json!(self.pipelined_committed_windows),
        );
        attrs.insert(
            "llama_stage.spec.pipelined_stale_windows".to_string(),
            json!(self.pipelined_stale_windows),
        );
        attrs.insert(
            "llama_stage.spec.pipelined_max_inflight_windows".to_string(),
            json!(self.pipelined_max_inflight_windows),
        );
        attrs.insert(
            "llama_stage.spec.pipelined_async_draft_windows".to_string(),
            json!(self.pipelined_async_draft_windows),
        );
        attrs.insert(
            "llama_stage.spec.pipelined_stale_draft_windows".to_string(),
            json!(self.pipelined_stale_draft_windows),
        );
        attrs.insert(
            "llama_stage.spec.pipelined_async_draft_wait_ms".to_string(),
            json!(self.pipelined_async_draft_wait_ms),
        );
        attrs.insert(
            "llama_stage.spec.pipelined_fifo_return_windows".to_string(),
            json!(self.pipelined_fifo_return_windows),
        );
        attrs.insert(
            "llama_stage.spec.pipelined_fifo_return_violations".to_string(),
            json!(self.pipelined_fifo_return_violations),
        );
        attrs.insert(
            "llama_stage.spec.pipelined_identity_violations".to_string(),
            json!(self.pipelined_identity_violations),
        );
        attrs.insert(
            "llama_stage.spec.tree_windows".to_string(),
            json!(self.tree_windows),
        );
        attrs.insert(
            "llama_stage.spec.tree_nodes".to_string(),
            json!(self.tree_nodes),
        );
        attrs.insert(
            "llama_stage.spec.tree_gather_ms".to_string(),
            json!(self.tree_gather_ms),
        );
    }
}

impl StageOpenAiBackend {
    pub(super) fn emit_speculative_commit_token_debug(
        &self,
        ids: &OpenAiGenerationIds,
        decode_step: usize,
        token: i32,
        message_kind: &'static str,
        speculative_mode: &'static str,
        window_index: Option<usize>,
    ) {
        if !self.telemetry.is_debug_enabled() {
            return;
        }
        let mut attrs = self.openai_attrs(ids);
        attrs.insert("llama_stage.decode_step".to_string(), json!(decode_step));
        attrs.insert(
            "llama_stage.decode_token_phase".to_string(),
            json!(decode_token_phase(
                u32::try_from(decode_step).unwrap_or(u32::MAX)
            )),
        );
        attrs.insert("llama_stage.predicted_token".to_string(), json!(token));
        attrs.insert("llama_stage.message_kind".to_string(), json!(message_kind));
        attrs.insert(
            "llama_stage.spec.commit_source".to_string(),
            json!(speculative_mode),
        );
        if let Some(window_index) = window_index {
            attrs.insert(
                "llama_stage.spec.pipelined_window_index".to_string(),
                json!(window_index),
            );
        }
        self.telemetry
            .emit_debug("stage.openai_decode_token", attrs);
    }
}

pub(super) fn verify_inputs_for_proposals(current: i32, proposals: &[i32]) -> Vec<i32> {
    let mut tokens = Vec::with_capacity(proposals.len().saturating_add(1));
    if proposals.is_empty() {
        return tokens;
    }
    tokens.push(current);
    tokens.extend_from_slice(proposals);
    tokens
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VerifySpanDecisionKind {
    FullAccept,
    AcceptedStop,
    TailReject,
    EarlyReject,
    EarlyRejectStop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VerifySpanDecision {
    pub(super) kind: VerifySpanDecisionKind,
    pub(super) accepted_before_reject: usize,
    pub(super) repair_input_count: Option<usize>,
    pub(super) commit_count: usize,
}

impl VerifySpanDecision {
    pub(super) fn rejected(self) -> bool {
        matches!(
            self.kind,
            VerifySpanDecisionKind::TailReject
                | VerifySpanDecisionKind::EarlyReject
                | VerifySpanDecisionKind::EarlyRejectStop
        )
    }
}

pub(super) fn shard_linear_commit_count(decision: &VerifySpanDecision, draft_len: usize) -> usize {
    match decision.kind {
        VerifySpanDecisionKind::FullAccept => draft_len,
        VerifySpanDecisionKind::AcceptedStop => decision.commit_count.min(draft_len),
        VerifySpanDecisionKind::TailReject
        | VerifySpanDecisionKind::EarlyReject
        | VerifySpanDecisionKind::EarlyRejectStop => decision.commit_count,
    }
}

pub(super) fn classify_verify_span<F>(
    draft_tokens: &[i32],
    predicted_tokens: &[i32],
    generated_len: usize,
    max_new_tokens: usize,
    mut token_is_eog: F,
) -> OpenAiResult<VerifySpanDecision>
where
    F: FnMut(i32) -> OpenAiResult<bool>,
{
    let expected_predictions = draft_tokens.len().saturating_add(1);
    if predicted_tokens.len() < expected_predictions {
        return Err(OpenAiError::backend(format!(
            "verify span returned too few tokens: got {} expected at least {}",
            predicted_tokens.len(),
            expected_predictions
        )));
    }

    let mut accepted_before_reject = 0usize;
    let mut commit_count = 0usize;
    for (draft_token, predicted) in draft_tokens.iter().zip(predicted_tokens.iter()) {
        commit_count += 1;
        let accepted = *predicted == *draft_token;
        let reached_eog = token_is_eog(*predicted)?;
        let reached_limit = generated_len + commit_count >= max_new_tokens;
        if accepted {
            accepted_before_reject += 1;
            if reached_eog || reached_limit {
                return Ok(VerifySpanDecision {
                    kind: VerifySpanDecisionKind::AcceptedStop,
                    accepted_before_reject,
                    repair_input_count: None,
                    commit_count,
                });
            }
            continue;
        }

        let repair_input_count = accepted_before_reject + 1;
        let kind = if repair_input_count == draft_tokens.len() {
            VerifySpanDecisionKind::TailReject
        } else if reached_eog || reached_limit {
            VerifySpanDecisionKind::EarlyRejectStop
        } else {
            VerifySpanDecisionKind::EarlyReject
        };
        return Ok(VerifySpanDecision {
            kind,
            accepted_before_reject,
            repair_input_count: Some(repair_input_count),
            commit_count,
        });
    }

    let correction_index = draft_tokens.len();
    let correction = predicted_tokens[correction_index];
    let can_commit_correction = generated_len + commit_count < max_new_tokens;
    if can_commit_correction {
        commit_count += 1;
        if token_is_eog(correction)? || generated_len + commit_count >= max_new_tokens {
            return Ok(VerifySpanDecision {
                kind: VerifySpanDecisionKind::AcceptedStop,
                accepted_before_reject,
                repair_input_count: None,
                commit_count,
            });
        }
    }

    Ok(VerifySpanDecision {
        kind: VerifySpanDecisionKind::FullAccept,
        accepted_before_reject,
        repair_input_count: None,
        commit_count,
    })
}

pub(super) fn nonzero_min(current: usize, candidate: usize) -> usize {
    if current == 0 {
        candidate
    } else {
        current.min(candidate)
    }
}

pub(super) fn sampling_is_tree_greedy_equivalent(sampling: &SamplingConfig) -> bool {
    if !sampling.enabled {
        return true;
    }
    let repeat_penalty = if sampling.repeat_penalty == 0.0 {
        1.0
    } else {
        sampling.repeat_penalty
    };
    sampling.temperature <= 0.0
        && sampling.presence_penalty == 0.0
        && sampling.frequency_penalty == 0.0
        && repeat_penalty == 1.0
        && sampling.logit_bias.is_empty()
}

pub(super) struct TreeAcceptDecision {
    pub(super) commit_tokens: Vec<i32>,
    pub(super) path_node_indices: Vec<usize>,
    pub(super) source_leaf_index: u32,
    pub(super) rejected: bool,
    pub(super) reached_stop: bool,
}

pub(super) fn accept_tree_path<F>(
    tree: &DraftTreeProposal,
    predicted_tokens: &[i32],
    generated_len: usize,
    max_new_tokens: usize,
    mut token_is_eog: F,
) -> OpenAiResult<TreeAcceptDecision>
where
    F: FnMut(i32) -> OpenAiResult<bool>,
{
    if tree.nodes.is_empty() {
        return Err(OpenAiError::backend("tree verify returned empty tree"));
    }
    if predicted_tokens.len() < tree.nodes.len() {
        return Err(OpenAiError::backend(format!(
            "tree verify returned too few tokens: got {} expected {}",
            predicted_tokens.len(),
            tree.nodes.len()
        )));
    }
    let mut node_index = 0usize;
    let mut path_node_indices = vec![0usize];
    let mut commit_tokens = Vec::new();
    let mut rejected = false;
    let mut reached_stop = false;
    loop {
        let predicted = predicted_tokens[node_index];
        commit_tokens.push(predicted);
        let generated_after_commit = generated_len + commit_tokens.len();
        if token_is_eog(predicted)? {
            reached_stop = true;
            break;
        }
        if generated_after_commit >= max_new_tokens {
            break;
        }
        let children = tree.children_of(node_index).collect::<Vec<_>>();
        let Some(child) = children
            .iter()
            .copied()
            .find(|child| tree.nodes[*child].token == predicted)
        else {
            rejected = !children.is_empty();
            break;
        };
        path_node_indices.push(child);
        node_index = child;
    }
    let last_node = *path_node_indices
        .last()
        .ok_or_else(|| OpenAiError::backend("tree accept path is empty"))?;
    Ok(TreeAcceptDecision {
        commit_tokens,
        path_node_indices,
        source_leaf_index: tree.nodes[last_node].first_leaf_index,
        rejected,
        reached_stop,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_inputs_for_proposals_uses_shard_k_plus_one_shape() {
        assert_eq!(
            verify_inputs_for_proposals(7, &[10, 11, 12]),
            vec![7, 10, 11, 12]
        );
    }

    #[test]
    fn classify_verify_span_commits_tail_correction_after_full_accept() {
        let decision = classify_verify_span(&[10, 11], &[10, 11, 12], 0, 8, |_| Ok(false)).unwrap();

        assert_eq!(decision.kind, VerifySpanDecisionKind::FullAccept);
        assert_eq!(decision.accepted_before_reject, 2);
        assert_eq!(decision.commit_count, 3);
        assert_eq!(shard_linear_commit_count(&decision, 2), 2);
        assert_eq!(decision.repair_input_count, None);
    }

    #[test]
    fn classify_verify_span_clips_tail_correction_at_length_budget() {
        let decision = classify_verify_span(&[10, 11], &[10, 11, 12], 0, 2, |_| Ok(false)).unwrap();

        assert_eq!(decision.kind, VerifySpanDecisionKind::AcceptedStop);
        assert_eq!(decision.accepted_before_reject, 2);
        assert_eq!(decision.commit_count, 2);
        assert_eq!(shard_linear_commit_count(&decision, 2), 2);
        assert_eq!(decision.repair_input_count, None);
    }

    #[test]
    fn classify_verify_span_requires_k_plus_one_predictions() {
        let error = classify_verify_span(&[10, 11], &[10, 11], 0, 8, |_| Ok(false))
            .expect_err("missing tail correction should fail");

        assert!(
            error.to_string().contains("expected at least 3"),
            "{error:#}"
        );
    }

    #[test]
    fn classify_verify_span_commits_early_reject_correction_without_repair() {
        let decision =
            classify_verify_span(&[10, 11, 12], &[10, 99, 12, 13], 0, 8, |_| Ok(false)).unwrap();

        assert_eq!(decision.kind, VerifySpanDecisionKind::EarlyReject);
        assert_eq!(decision.accepted_before_reject, 1);
        assert_eq!(decision.repair_input_count, Some(2));
        assert_eq!(decision.commit_count, 2);
        assert_eq!(shard_linear_commit_count(&decision, 3), 2);
    }

    fn sample_tree() -> DraftTreeProposal {
        DraftTreeProposal {
            nodes: vec![
                DraftTreeNode {
                    token: 100,
                    parent: -1,
                    depth: 0,
                    first_leaf_index: 0,
                },
                DraftTreeNode {
                    token: 10,
                    parent: 0,
                    depth: 1,
                    first_leaf_index: 0,
                },
                DraftTreeNode {
                    token: 11,
                    parent: 1,
                    depth: 2,
                    first_leaf_index: 0,
                },
                DraftTreeNode {
                    token: 20,
                    parent: 0,
                    depth: 1,
                    first_leaf_index: 1,
                },
            ],
        }
    }

    #[test]
    fn accept_tree_path_commits_leaf_direct_return_without_reject() {
        let decision =
            accept_tree_path(&sample_tree(), &[10, 11, 12, 21], 0, 8, |_| Ok(false)).unwrap();

        assert_eq!(decision.commit_tokens, vec![10, 11, 12]);
        assert_eq!(decision.path_node_indices, vec![0, 1, 2]);
        assert_eq!(decision.source_leaf_index, 0);
        assert!(!decision.rejected);
        assert!(!decision.reached_stop);
    }

    #[test]
    fn accept_tree_path_rejects_when_target_leaves_candidate_tree_early() {
        let decision =
            accept_tree_path(&sample_tree(), &[10, 99, 12, 21], 0, 8, |_| Ok(false)).unwrap();

        assert_eq!(decision.commit_tokens, vec![10, 99]);
        assert_eq!(decision.path_node_indices, vec![0, 1]);
        assert_eq!(decision.source_leaf_index, 0);
        assert!(decision.rejected);
        assert!(!decision.reached_stop);
    }

    #[test]
    fn accept_tree_path_length_budget_is_not_model_stop() {
        let decision =
            accept_tree_path(&sample_tree(), &[10, 11, 12, 21], 1, 3, |_| Ok(false)).unwrap();

        assert_eq!(decision.commit_tokens, vec![10, 11]);
        assert_eq!(decision.path_node_indices, vec![0, 1]);
        assert_eq!(decision.source_leaf_index, 0);
        assert!(!decision.rejected);
        assert!(!decision.reached_stop);
    }

    #[test]
    fn accept_tree_path_stops_without_reject_when_eog_is_predicted() {
        let decision = accept_tree_path(&sample_tree(), &[20, 11, 12, 200], 0, 8, |token| {
            Ok(token == 200)
        })
        .unwrap();

        assert_eq!(decision.commit_tokens, vec![20, 200]);
        assert_eq!(decision.path_node_indices, vec![0, 3]);
        assert_eq!(decision.source_leaf_index, 1);
        assert!(!decision.rejected);
        assert!(decision.reached_stop);
    }

    // Regression for a tree-mode early-truncation bug: when an accepted path
    // ends in a stop token, the commit loop must still emit every token in
    // `commit_tokens` (accepted prefix plus the stop correction), not just the
    // first one. Pre-seeding the loop's stop flag from `decision.reached_stop`
    // dropped all but the first committed token, so tree output was a correct
    // but truncated prefix of the target-only greedy reference.
    #[test]
    fn tree_stop_window_commits_full_accepted_path_before_stopping() {
        let decision = accept_tree_path(&sample_tree(), &[20, 11, 12, 200], 0, 8, |token| {
            Ok(token == 200)
        })
        .unwrap();

        let mut emitted = Vec::new();
        let mut reached_stop = false;
        let max_new_tokens = 8usize;
        let mut decoded_tokens = 0usize;
        // Mirror the generation commit loop: the per-token sink decides stops,
        // and `decision.reached_stop` only ends the outer loop afterwards.
        let on_token = |_token: i32| false;
        for token in &decision.commit_tokens {
            decoded_tokens += 1;
            emitted.push(*token);
            if on_token(*token) {
                reached_stop = true;
            }
            if reached_stop || decoded_tokens >= max_new_tokens {
                break;
            }
        }
        let stop = reached_stop || decision.reached_stop;

        assert_eq!(emitted, vec![20, 200]);
        assert!(stop);
    }
}
