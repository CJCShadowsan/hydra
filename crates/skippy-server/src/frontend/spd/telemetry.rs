use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SpdRollingTelemetry {
    pub(super) snapshot: SpdRollingSnapshot,
    pub(super) speculation_rows: Option<SpdRollingSpeculationRows>,
    pub(super) verified_delta: Option<SpdRollingVerifiedDelta>,
}

pub(super) fn insert_rolling_attrs(
    snapshot: &SpdRollingSnapshot,
    attrs: &mut BTreeMap<String, Value>,
) {
    attrs.insert(
        "llama_stage.spd_rolling.logical_stage_count".to_string(),
        json!(snapshot.logical_stage_count),
    );
    attrs.insert(
        "llama_stage.spd_rolling.target_position".to_string(),
        json!(snapshot.target_position),
    );
    attrs.insert(
        "llama_stage.spd_rolling.next_position".to_string(),
        json!(snapshot.next_position),
    );
    attrs.insert(
        "llama_stage.spd_rolling.inserted_drafts".to_string(),
        json!(snapshot.inserted_drafts),
    );
    attrs.insert(
        "llama_stage.spd_rolling.missing_proposals".to_string(),
        json!(snapshot.missing_proposals),
    );
    attrs.insert(
        "llama_stage.spd_rolling.first_missing_proposal_position".to_string(),
        json!(snapshot.first_missing_proposal_position),
    );
    attrs.insert(
        "llama_stage.spd_rolling.out_of_order_proposals".to_string(),
        json!(snapshot.out_of_order_proposals),
    );
    attrs.insert(
        "llama_stage.spd_rolling.first_out_of_order_proposal_position".to_string(),
        json!(snapshot.first_out_of_order_proposal_position),
    );
    attrs.insert(
        "llama_stage.spd_rolling.verified_windows".to_string(),
        json!(snapshot.verified_windows),
    );
    attrs.insert(
        "llama_stage.spd_rolling.accepted_windows".to_string(),
        json!(snapshot.accepted_windows),
    );
    attrs.insert(
        "llama_stage.spd_rolling.rejected_windows".to_string(),
        json!(snapshot.rejected_windows),
    );
    attrs.insert(
        "llama_stage.spd_rolling.first_rejected_target_position".to_string(),
        json!(snapshot.first_rejected_target_position),
    );
    attrs.insert(
        "llama_stage.spd_rolling.pipeline_len".to_string(),
        json!(snapshot.pipeline_len),
    );
    attrs.insert(
        "llama_stage.spd_rolling.verified_up_to".to_string(),
        json!(snapshot.verified_up_to),
    );
}

pub(super) fn insert_rolling_verified_delta_attrs(
    delta: &SpdRollingVerifiedDelta,
    attrs: &mut BTreeMap<String, Value>,
) {
    attrs.insert(
        "llama_stage.spd_rolling.verified_delta_start_position".to_string(),
        json!(delta.start_position),
    );
    attrs.insert(
        "llama_stage.spd_rolling.verified_delta_up_to".to_string(),
        json!(delta.verified_up_to),
    );
    attrs.insert(
        "llama_stage.spd_rolling.verified_delta_tokens".to_string(),
        json!(delta.tokens),
    );
    attrs.insert(
        "llama_stage.spd_rolling.verified_delta_token_count".to_string(),
        json!(delta.tokens.len()),
    );
}

pub(super) fn insert_rolling_speculation_rows_attrs(
    rows: &SpdRollingSpeculationRows,
    attrs: &mut BTreeMap<String, Value>,
) {
    attrs.insert(
        "llama_stage.spd_rolling.row_evicted_prefix_position".to_string(),
        json!(rows.evicted_prefix_position),
    );
    attrs.insert(
        "llama_stage.spd_rolling.row_positions".to_string(),
        json!(rows.row_positions),
    );
    attrs.insert(
        "llama_stage.spd_rolling.row_i_stages".to_string(),
        json!(rows.row_i_stages),
    );
    attrs.insert(
        "llama_stage.spd_rolling.row_newest_position".to_string(),
        json!(rows.newest_position),
    );
    attrs.insert(
        "llama_stage.spd_rolling.row_next_draft_position".to_string(),
        json!(rows.next_draft_position),
    );
}

pub(super) fn insert_proposal_stats_attrs(
    scope: &str,
    stats: &SpdProposalSourceStats,
    attrs: &mut BTreeMap<String, Value>,
) {
    let prefix = format!("llama_stage.spd_proposal.{scope}");
    attrs.insert(
        format!("{prefix}.requested_limit"),
        json!(stats.requested_limit),
    );
    attrs.insert(format!("{prefix}.attempts"), json!(stats.attempts));
    attrs.insert(format!("{prefix}.proposed"), json!(stats.proposed));
    attrs.insert(
        format!("{prefix}.inline_tap_hits"),
        json!(stats.inline_tap_hits),
    );
    attrs.insert(
        format!("{prefix}.replay_fallbacks"),
        json!(stats.replay_fallbacks),
    );
    attrs.insert(format!("{prefix}.cache_hits"), json!(stats.cache_hits));
    attrs.insert(format!("{prefix}.cache_misses"), json!(stats.cache_misses));
    attrs.insert(
        format!("{prefix}.tap_collect_ms"),
        json!(stats.tap_collect_ms),
    );
    attrs.insert(format!("{prefix}.cur_in_ms"), json!(stats.cur_in_ms));
    attrs.insert(format!("{prefix}.forward_ms"), json!(stats.forward_ms));
    attrs.insert(
        format!("{prefix}.last_cache_prefix_len"),
        json!(stats.last_cache_prefix_len),
    );
    attrs.insert(
        format!("{prefix}.max_cache_prefix_len"),
        json!(stats.max_cache_prefix_len),
    );
}
