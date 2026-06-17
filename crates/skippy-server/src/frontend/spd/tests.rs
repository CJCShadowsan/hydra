use super::cache::{
    SpdInlineTapCache, SpdInlineTapLifecycle, SpdInlineTapRecord, common_token_prefix_len,
    retained_tap_prefix_len_for_context_update,
};
use super::*;
use skippy_protocol::{
    LoadMode, StageTopologyEntry,
    binary::{StageStateHeader, WireActivationDType, WireMessageKind},
};
use skippy_runtime::{ActivationDesc, RuntimeActivationDType, RuntimeActivationLayout};

#[test]
fn stage_ranges_from_topology_are_layer_sorted() {
    let topology = StageTopology {
        topology_id: "topology".to_string(),
        model_id: "model".to_string(),
        stages: vec![
            stage_entry("stage-1", 1, 8, 16),
            stage_entry("stage-0", 0, 0, 8),
        ],
    };

    let ranges = spd_stage_ranges_from_topology(&topology).unwrap();

    assert_eq!(
        ranges,
        vec![
            SpdStageLayerRange::new(0, 0, 8),
            SpdStageLayerRange::new(1, 8, 16)
        ]
    );
}

#[test]
fn inline_proposal_carries_logit_margin_from_topk() {
    let topk = skippy_runtime::spd::SpdQwen3FixtureTopK {
        draft_indices: vec![0, 1],
        token_ids: vec![42, 17],
        logits: vec![8.25, 3.0],
    };

    let proposal = spd_inline_proposal_from_topk(&topk).unwrap();

    assert_eq!(
        proposal,
        SpdInlineProposal {
            token: 42,
            logit: Some(8.25),
            logit_margin: Some(5.25),
            cache_used: false,
            cache_prefix_len: None,
            tap_source: SpdTapCollectionSource::Inline,
            tap_collect_ms: 0.0,
            cur_in_ms: 0.0,
            forward_ms: 0.0,
            proposal_rows: SpdProposalRows::default(),
        }
    );
}

#[test]
fn inline_probe_margin_gate_controls_optimistic_decode() {
    let probe = SpdInlineProbe {
        phase: SpdInlineProbePhase::PreTargetReply,
        proposed: Some(42),
        proposed_logit: Some(8.25),
        proposed_logit_margin: Some(5.25),
        cache_used: true,
        cache_prefix_len: Some(12),
        tap_source: Some(SpdTapCollectionSource::Inline),
        tap_collect_ms: 0.2,
        cur_in_ms: 0.3,
        forward_ms: 0.4,
        elapsed_ms: 1.0,
        target_wait_after_probe_ms: 0.0,
        trigger_hf_index: Some(31),
        proposal_rows: SpdProposalRows::default(),
        proposal_miss: None,
    };

    assert!(probe.allows_optimistic_decode(None));
    assert!(probe.allows_optimistic_decode(Some(5.0)));
    assert!(!probe.allows_optimistic_decode(Some(6.0)));
}

#[test]
fn rolling_row_roles_use_deepest_available_snapshot() {
    let rows = SpdRollingSpeculationRows {
        evicted_prefix_position: None,
        row_positions: vec![20, 21, 22, 23],
        row_i_stages: vec![3, 2, 1, 0],
        newest_position: 23,
        next_draft_position: 24,
    };
    let mut partial_cache = spd_role_cache();
    record_role_cache_taps(&mut partial_cache, 20, &[8, 16, 24]);
    record_role_cache_taps(&mut partial_cache, 21, &[8, 16]);
    record_role_cache_taps(&mut partial_cache, 22, &[8]);

    assert_eq!(
        resolve_rolling_row_stage_roles(&test_spd_topology(true), &rows, &partial_cache).unwrap(),
        vec![3, 2, 1, 0]
    );

    let mut full_cache = spd_role_cache();
    record_role_cache_taps(&mut full_cache, 20, &[10, 20, 31]);
    record_role_cache_taps(&mut full_cache, 21, &[10, 20, 31]);
    record_role_cache_taps(&mut full_cache, 22, &[10, 20, 31]);

    assert_eq!(
        resolve_rolling_row_stage_roles(&test_spd_topology(true), &rows, &full_cache).unwrap(),
        vec![4, 4, 4, 0]
    );
    assert_eq!(
        resolve_rolling_row_stage_roles(&test_spd_topology(false), &rows, &full_cache).unwrap(),
        vec![3, 2, 1, 0]
    );
}

#[test]
fn rolling_row_roles_cap_fused_rows_when_evicted_prefix_is_present() {
    let rows = SpdRollingSpeculationRows {
        evicted_prefix_position: Some(20),
        row_positions: vec![20, 21, 22, 23, 24],
        row_i_stages: vec![4, 3, 2, 1, 0],
        newest_position: 24,
        next_draft_position: 25,
    };
    let mut cache = spd_role_cache();
    record_role_cache_taps(&mut cache, 20, &[10, 20, 31]);
    record_role_cache_taps(&mut cache, 21, &[8, 10, 16, 20, 24, 31]);
    record_role_cache_taps(&mut cache, 22, &[8, 10, 16, 20, 24, 31]);
    record_role_cache_taps(&mut cache, 23, &[8, 10, 16, 20, 24, 31]);

    assert_eq!(
        resolve_rolling_row_stage_roles(&test_spd_topology(true), &rows, &cache).unwrap(),
        vec![4, 3, 3, 3, 0]
    );
}

#[test]
fn rolling_state_verifies_contiguous_live_proposals() {
    let mut state = SpdRollingObserver::new(3);

    state.observe_probe(0, 10, Some(99)).unwrap();
    state.observe_probe(1, 11, Some(11)).unwrap();
    let snapshot = state.observe_probe(2, 12, Some(12)).unwrap();

    assert_eq!(snapshot.inserted_drafts, 2);
    assert_eq!(snapshot.missing_proposals, 0);
    assert_eq!(snapshot.out_of_order_proposals, 0);
    assert_eq!(snapshot.verified_windows, 1);
    assert_eq!(snapshot.accepted_windows, 1);
    assert_eq!(snapshot.rejected_windows, 0);
    assert_eq!(snapshot.pipeline_len, Some(2));
    assert_eq!(snapshot.verified_up_to, Some(2));
}

#[test]
fn rolling_state_reports_missing_and_out_of_order_live_proposals() {
    let mut state = SpdRollingObserver::new(4);

    state.observe_probe(0, 10, Some(99)).unwrap();
    state.observe_probe(1, 11, Some(11)).unwrap();
    state.observe_probe(2, 12, Some(12)).unwrap();
    state.observe_target(3, 13).unwrap();
    let snapshot = state.observe_probe(4, 14, Some(14)).unwrap();

    assert_eq!(snapshot.inserted_drafts, 2);
    assert_eq!(snapshot.missing_proposals, 1);
    assert_eq!(snapshot.first_missing_proposal_position, Some(3));
    assert_eq!(snapshot.out_of_order_proposals, 1);
    assert_eq!(snapshot.first_out_of_order_proposal_position, Some(4));
    assert_eq!(snapshot.verified_windows, 0);
    assert_eq!(snapshot.pipeline_len, Some(3));
    assert_eq!(snapshot.verified_up_to, Some(1));
}

#[test]
fn accepted_extension_retains_tap_rows_for_new_context() {
    assert_eq!(
        retained_tap_prefix_len_for_context_update(&[1, 2], &[1, 2, 3, 4], true),
        4
    );
    assert_eq!(
        retained_tap_prefix_len_for_context_update(&[1, 2], &[1, 9, 3], true),
        1
    );
    assert_eq!(
        retained_tap_prefix_len_for_context_update(&[1, 2], &[1, 2, 3, 4], false),
        2
    );
}

#[test]
fn inline_tap_cache_rebuilds_positioned_activation_frame() {
    let mut cache = SpdInlineTapCache::new(2, vec![8]);
    let config = stage_config("stage-0", 0, 0, 8);
    let message = stage_message(2, 2, Vec::new());
    let frame = activation_frame(0, 8, &[1.0, 2.0, 3.0, 4.0]);

    let record = cache
        .record_stage_output(&config, &message, &frame)
        .unwrap()
        .unwrap();
    assert_eq!(
        record,
        SpdInlineTapRecord {
            hf_index: 8,
            positions: vec![2, 3],
            rows_recorded: 2,
            cached_rows: 2,
            payload_bytes: 16,
            required: true,
        }
    );

    let rebuilt = cache
        .frame_for_positions(8, &[2, 3])
        .unwrap()
        .expect("positions should be complete");
    assert_eq!(rebuilt.desc.token_count, 4);
    assert_eq!(f32_row(&rebuilt, 2, 2), vec![1.0, 2.0]);
    assert_eq!(f32_row(&rebuilt, 3, 2), vec![3.0, 4.0]);
}

#[test]
fn inline_tap_cache_overlays_complete_required_frames() {
    let mut cache = SpdInlineTapCache::new(2, vec![8]);
    let config = stage_config("stage-0", 0, 0, 8);
    let message = stage_message(2, 2, vec![2, 3]);
    let frame = activation_frame(0, 8, &[5.0, 6.0, 7.0, 8.0]);
    cache
        .record_stage_output(&config, &message, &frame)
        .unwrap()
        .unwrap();
    let mut taps = BTreeMap::new();

    cache
        .overlay_complete_frames(&mut taps, &[2, 3], &[8], 2)
        .unwrap();

    let overlaid = taps.get(&8).expect("expected overlaid tap");
    assert_eq!(f32_row(overlaid, 2, 2), vec![5.0, 6.0]);
    assert_eq!(f32_row(overlaid, 3, 2), vec![7.0, 8.0]);
}

#[test]
fn inline_tap_cache_returns_complete_required_frames() {
    let mut cache = SpdInlineTapCache::new(2, vec![10, 20]);
    let message = stage_message(4, 2, Vec::new());
    cache
        .record_stage_output(
            &stage_config("stage-1", 1, 8, 10),
            &message,
            &activation_frame(8, 10, &[1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap()
        .unwrap();
    cache
        .record_stage_output(
            &stage_config("stage-3", 3, 16, 20),
            &message,
            &activation_frame(16, 20, &[5.0, 6.0, 7.0, 8.0]),
        )
        .unwrap()
        .unwrap();

    let complete = cache
        .complete_frames(&[4, 5], &[10, 20], 2)
        .unwrap()
        .expect("all required taps are complete");

    assert_eq!(complete.keys().copied().collect::<Vec<_>>(), vec![10, 20]);
    assert_eq!(f32_row(complete.get(&10).unwrap(), 4, 2), vec![1.0, 2.0]);
    assert_eq!(f32_row(complete.get(&20).unwrap(), 5, 2), vec![7.0, 8.0]);
}

#[test]
fn inline_tap_cache_completes_only_rows_that_need_each_hidden_state() {
    let mut cache = SpdInlineTapCache::new(2, vec![10, 20, 31]);
    let message = stage_message(23, 3, Vec::new());
    cache
        .record_stage_output(
            &stage_config("stage-1", 1, 8, 10),
            &message,
            &activation_frame(8, 10, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        .unwrap()
        .unwrap();
    cache
        .record_stage_output(
            &stage_config("stage-3", 3, 16, 20),
            &message,
            &activation_frame(16, 20, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        )
        .unwrap()
        .unwrap();
    cache
        .record_stage_output(
            &stage_config("stage-5", 5, 24, 31),
            &message,
            &activation_frame(24, 31, &[13.0, 14.0, 15.0, 16.0, 17.0, 18.0]),
        )
        .unwrap()
        .unwrap();
    let row_positions = [23, 24, 25, 26];
    let row_hf_indices = vec![
        vec![0, 10, 20, 31],
        vec![0, 10, 20, 31],
        vec![0, 10, 20, 31],
        vec![0],
    ];

    let complete = cache
        .complete_frames_for_row_hf_indices(&row_positions, &row_hf_indices, &[10, 20, 31], 2)
        .unwrap()
        .expect("stage-0 row should not require downstream taps");

    assert!(
        cache
            .complete_frames(&row_positions, &[10, 20, 31], 2)
            .unwrap()
            .is_none()
    );
    assert!(
        cache
            .missing_required_rows_for_row_hf_indices(
                &row_positions,
                &row_hf_indices,
                &[10, 20, 31],
            )
            .unwrap()
            .is_empty()
    );
    assert_eq!(f32_row(complete.get(&31).unwrap(), 25, 2), vec![17.0, 18.0]);
}

#[test]
fn inline_tap_cache_reports_missing_required_frame() {
    let mut cache = SpdInlineTapCache::new(2, vec![10, 20]);
    cache
        .record_stage_output(
            &stage_config("stage-1", 1, 8, 10),
            &stage_message(4, 1, Vec::new()),
            &activation_frame(8, 10, &[1.0, 2.0]),
        )
        .unwrap()
        .unwrap();

    assert!(cache.complete_frames(&[4], &[10, 20], 2).unwrap().is_none());
}

#[test]
fn inline_tap_cache_records_unrequired_stage_output_without_overlaying_it() {
    let mut cache = SpdInlineTapCache::new(2, vec![8]);
    let config = stage_config("stage-1", 1, 8, 10);
    let message = stage_message(0, 1, Vec::new());
    let frame = activation_frame(8, 10, &[1.0, 2.0]);

    let record = cache
        .record_stage_output(&config, &message, &frame)
        .unwrap()
        .unwrap();
    assert_eq!(
        record,
        SpdInlineTapRecord {
            hf_index: 10,
            positions: vec![0],
            rows_recorded: 1,
            cached_rows: 1,
            payload_bytes: 8,
            required: false,
        }
    );
    let mut taps = BTreeMap::new();
    cache
        .overlay_complete_frames(&mut taps, &[0], &[8], 2)
        .unwrap();
    assert!(taps.is_empty());
}

#[test]
fn inline_tap_cache_records_returned_downstream_tap() {
    let mut cache = SpdInlineTapCache::new(2, vec![10]);
    let tap = StageReplySpdTap {
        hf_index: 10,
        producer_stage_index: 1,
        layer_start: 8,
        layer_end: 10,
        token_count: 2,
        sequence_count: 1,
        dtype: RuntimeActivationDType::F32 as i32,
        layout: RuntimeActivationLayout::TokenMajor as i32,
        flags: 0,
        positions: vec![4, 5],
        payload: f32_payload(&[9.0, 10.0, 11.0, 12.0]),
    };

    let record = cache.record_returned_tap(&tap).unwrap();

    assert_eq!(
        record,
        SpdInlineTapRecord {
            hf_index: 10,
            positions: vec![4, 5],
            rows_recorded: 2,
            cached_rows: 2,
            payload_bytes: 16,
            required: true,
        }
    );
    let overlaid = cache
        .frame_for_positions(10, &[4, 5])
        .unwrap()
        .expect("returned tap rows should be complete");
    assert_eq!(f32_row(&overlaid, 4, 2), vec![9.0, 10.0]);
    assert_eq!(f32_row(&overlaid, 5, 2), vec![11.0, 12.0]);
}

#[test]
fn tap_lifecycle_filters_future_taps_without_pending_optimistic_position() {
    let mut lifecycle = SpdInlineTapLifecycle::default();
    lifecycle.accept_context_len(4);
    let prefix_tap = returned_tap_at_positions(vec![3]);
    assert_eq!(
        lifecycle.record_decision(&prefix_tap).unwrap().ignored,
        None
    );

    let future_tap = returned_tap_at_positions(vec![4]);
    let ignored = lifecycle
        .record_decision(&future_tap)
        .unwrap()
        .ignored
        .expect("future tap should be ignored");
    assert_eq!(
        ignored.reason,
        "future_position_without_pending_optimistic_context"
    );
    assert_eq!(ignored.positions, vec![4]);
    assert_eq!(ignored.accepted_context_len, 4);

    lifecycle.mark_pending_optimistic_position(4);
    assert_eq!(
        lifecycle.record_decision(&future_tap).unwrap().ignored,
        None
    );

    lifecycle.accept_context_len(6);
    assert_eq!(
        lifecycle.record_decision(&future_tap).unwrap().ignored,
        None
    );
    let stale_future_tap = returned_tap_at_positions(vec![6]);
    assert!(
        lifecycle
            .record_decision(&stale_future_tap)
            .unwrap()
            .ignored
            .is_some()
    );
}

#[test]
fn tap_lifecycle_allows_pending_verify_span_positions() {
    let mut lifecycle = SpdInlineTapLifecycle::default();
    lifecycle.accept_context_len(4);
    lifecycle.mark_pending_future_positions([4, 5, 6]);

    assert_eq!(
        lifecycle
            .record_decision(&returned_tap_at_positions(vec![4, 5]))
            .unwrap()
            .ignored,
        None
    );

    let ignored = lifecycle
        .record_decision(&returned_tap_at_positions(vec![7]))
        .unwrap()
        .ignored
        .expect("unmarked future tap should be ignored");
    assert_eq!(ignored.positions, vec![7]);
    assert_eq!(ignored.pending_positions, vec![4, 5, 6]);

    lifecycle.accept_context_len(7);
    assert_eq!(
        lifecycle
            .record_decision(&returned_tap_at_positions(vec![6]))
            .unwrap()
            .ignored,
        None
    );
    assert!(
        lifecycle
            .record_decision(&returned_tap_at_positions(vec![7]))
            .unwrap()
            .ignored
            .is_some()
    );
}

#[test]
fn inline_tap_cache_zero_prefix_drops_recorded_rows() {
    let mut cache = SpdInlineTapCache::new(2, vec![8]);
    let config = stage_config("stage-0", 0, 0, 8);
    let message = stage_message(2, 1, Vec::new());
    let frame = activation_frame(0, 8, &[1.0, 2.0]);
    cache
        .record_stage_output(&config, &message, &frame)
        .unwrap()
        .unwrap();

    cache.retain_positions_before(0);

    assert!(cache.frame_for_positions(8, &[2]).unwrap().is_none());
}

#[test]
fn inline_tap_cache_retains_only_prefix_positions() {
    let mut cache = SpdInlineTapCache::new(2, vec![8]);
    let config = stage_config("stage-0", 0, 0, 8);
    let message = stage_message(2, 3, vec![2, 3, 4]);
    let frame = activation_frame(0, 8, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    cache
        .record_stage_output(&config, &message, &frame)
        .unwrap()
        .unwrap();

    cache.retain_positions_before(4);

    assert!(cache.frame_for_positions(8, &[2, 3]).unwrap().is_some());
    assert!(cache.frame_for_positions(8, &[4]).unwrap().is_none());
}

#[test]
fn inline_tap_cache_retains_pending_future_positions() {
    let mut cache = SpdInlineTapCache::new(2, vec![8]);
    let config = stage_config("stage-0", 0, 0, 8);
    let message = stage_message(2, 3, vec![2, 3, 4]);
    let frame = activation_frame(0, 8, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    cache
        .record_stage_output(&config, &message, &frame)
        .unwrap()
        .unwrap();

    cache.retain_positions_before_or_in(4, &BTreeSet::from([4]));

    assert!(cache.frame_for_positions(8, &[2, 3, 4]).unwrap().is_some());

    cache.retain_positions_before(4);

    assert!(cache.frame_for_positions(8, &[4]).unwrap().is_none());
}

#[test]
fn token_prefix_len_stops_at_first_divergence() {
    assert_eq!(common_token_prefix_len(&[1, 2, 3], &[1, 2, 4, 5]), 2);
    assert_eq!(common_token_prefix_len(&[1, 2], &[1, 2, 3]), 2);
    assert_eq!(common_token_prefix_len(&[9], &[1]), 0);
}

fn stage_entry(
    stage_id: &str,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
) -> StageTopologyEntry {
    StageTopologyEntry {
        stage_id: stage_id.to_string(),
        stage_index,
        host: None,
        endpoint: "127.0.0.1:0".to_string(),
        layer_start,
        layer_end,
        load_mode: LoadMode::RuntimeSlice,
    }
}

fn stage_config(stage_id: &str, stage_index: u32, layer_start: u32, layer_end: u32) -> StageConfig {
    StageConfig {
        run_id: "run".to_string(),
        topology_id: "topology".to_string(),
        model_id: "model".to_string(),
        package_ref: None,
        manifest_sha256: None,
        source_model_path: None,
        source_model_sha256: None,
        source_model_bytes: None,
        materialized_path: None,
        materialized_pinned: false,
        model_path: Some("/tmp/model.gguf".to_string()),
        projector_path: None,
        stage_id: stage_id.to_string(),
        stage_index,
        layer_start,
        layer_end,
        spd_tap_return_hf_indices: Vec::new(),
        ctx_size: 128,
        lane_count: 1,
        n_batch: None,
        n_ubatch: None,
        n_gpu_layers: 0,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        flash_attn_type: skippy_protocol::FlashAttentionType::Auto,
        filter_tensors_on_load: true,
        selected_device: None,
        kv_cache: None,
        load_mode: LoadMode::RuntimeSlice,
        bind_addr: "127.0.0.1:0".to_string(),
        upstream: None,
        downstream: None,
    }
}

fn stage_message(pos_start: i32, token_count: i32, positions: Vec<i32>) -> StageWireMessage {
    StageWireMessage {
        kind: WireMessageKind::PrefillEmbd,
        pos_start,
        token_count,
        state: StageStateHeader::new(WireMessageKind::PrefillEmbd, WireActivationDType::F32),
        request_id: 1,
        session_id: 2,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: vec![1; usize::try_from(token_count).unwrap()],
        positions,
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    }
}

fn activation_frame(layer_start: i32, layer_end: i32, values: &[f32]) -> ActivationFrame {
    let payload = f32_payload(values);
    ActivationFrame {
        desc: ActivationDesc {
            version: 1,
            dtype: RuntimeActivationDType::F32,
            layout: RuntimeActivationLayout::TokenMajor,
            producer_stage_index: 0,
            layer_start,
            layer_end,
            token_count: u32::try_from(values.len() / 2).unwrap(),
            sequence_count: 1,
            payload_bytes: u64::try_from(payload.len()).unwrap(),
            flags: 0,
        },
        payload,
    }
}

fn returned_tap_at_positions(positions: Vec<i32>) -> StageReplySpdTap {
    let values = positions
        .iter()
        .enumerate()
        .flat_map(|(index, _)| {
            let base = index as f32;
            [base + 1.0, base + 2.0]
        })
        .collect::<Vec<_>>();
    StageReplySpdTap {
        hf_index: 10,
        producer_stage_index: 1,
        layer_start: 8,
        layer_end: 10,
        token_count: u32::try_from(positions.len()).unwrap(),
        sequence_count: 1,
        dtype: RuntimeActivationDType::F32 as i32,
        layout: RuntimeActivationLayout::TokenMajor as i32,
        flags: 0,
        positions,
        payload: f32_payload(&values),
    }
}

fn spd_role_cache() -> SpdInlineTapCache {
    SpdInlineTapCache::new(2, vec![8, 10, 16, 20, 24, 31])
}

fn record_role_cache_taps(cache: &mut SpdInlineTapCache, position: usize, hf_indices: &[u32]) {
    for hf_index in hf_indices {
        let tap = StageReplySpdTap {
            hf_index: *hf_index,
            producer_stage_index: 1,
            layer_start: i32::try_from(hf_index.saturating_sub(1)).unwrap(),
            layer_end: i32::try_from(*hf_index).unwrap(),
            token_count: 1,
            sequence_count: 1,
            dtype: RuntimeActivationDType::F32 as i32,
            layout: RuntimeActivationLayout::TokenMajor as i32,
            flags: 0,
            positions: vec![i32::try_from(position).unwrap()],
            payload: f32_payload(&[1.0, 2.0]),
        };
        cache.record_returned_tap(&tap).unwrap();
    }
}

fn test_spd_topology(trained_with_use_deepest: bool) -> skippy_runtime::spd::SpdHeadTopology {
    skippy_runtime::spd::SpdHeadTopology {
        hidden_size: 2,
        vocab_size: 8,
        draft_vocab_size: 8,
        num_stages: 4,
        stage_layer_boundaries: None,
        num_spec_layers: 4,
        trained_with_use_deepest,
        shallow_hidden_layer_indices: vec![
            vec![0, 10, 20, 31],
            vec![0, 8, 16, 24],
            vec![0, 8, 16],
            vec![0, 8],
        ],
        spec_init_from_base_layers: None,
        draft_token_ids: None,
    }
}

fn f32_payload(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_row(frame: &ActivationFrame, row_index: usize, width: usize) -> Vec<f32> {
    let offset = row_index * width * std::mem::size_of::<f32>();
    frame.payload[offset..offset + width * std::mem::size_of::<f32>()]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
