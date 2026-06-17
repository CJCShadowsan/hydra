use super::*;

#[test]
fn sorted_unique_nonzero_drops_h0_and_sorts() {
    assert_eq!(sorted_unique_nonzero(&[31, 0, 10, 10, 20]), [10, 20, 31]);
}

#[test]
fn prompt_from_line_reads_plain_text() {
    let prompt = prompt_from_line(2, "Write a test").unwrap();

    assert_eq!(prompt.index, 2);
    assert_eq!(prompt.label, "prompt-002");
    assert_eq!(prompt.text, "Write a test");
}

#[test]
fn prompt_from_line_reads_json_object() {
    let prompt = prompt_from_line(
        1,
        r#"{"prompt_id":"human/eval 1","prompt":"Implement add"}"#,
    )
    .unwrap();

    assert_eq!(prompt.index, 1);
    assert_eq!(prompt.label, "human-eval-1");
    assert_eq!(prompt.text, "Implement add");
}

#[test]
fn prompt_from_line_reads_chat_messages() {
    let prompt = prompt_from_line(
            3,
            r#"{"id":"messages/1","messages":[{"role":"system","content":"You are concise."},{"role":"user","content":"Patch the request parser."}]}"#,
        )
        .unwrap();

    assert_eq!(prompt.index, 3);
    assert_eq!(prompt.label, "messages-1");
    assert_eq!(prompt.messages.len(), 2);
    assert_eq!(prompt.messages[0].role, "system");
    assert_eq!(prompt.messages[1].role, "user");
    assert!(prompt.text.contains("system: You are concise."));
    assert!(prompt.text.contains("user: Patch the request parser."));
}

#[test]
fn prompt_from_line_reads_turns_as_user_message() {
    let prompt = prompt_from_line(
            4,
            r#"{"label":"turns 1","turns":["Explain the queueing risk.","Now make the report actionable."]}"#,
        )
        .unwrap();

    assert_eq!(prompt.index, 4);
    assert_eq!(prompt.label, "turns-1");
    assert_eq!(prompt.messages.len(), 1);
    assert_eq!(prompt.messages[0].role, "user");
    assert_eq!(
        prompt.messages[0].content,
        "Explain the queueing risk.\n\nNow make the report actionable."
    );
    assert_eq!(prompt.text, prompt.messages[0].content);
}

#[test]
fn stage_ranges_reject_non_ascending_splits() {
    assert!(stage_ranges(&[8, 8], 32).is_err());
    assert!(stage_ranges(&[8, 33], 32).is_err());
    assert_eq!(
        stage_ranges(&[8, 10], 12).unwrap(),
        [(0, 8), (8, 10), (10, 12)]
    );
}

#[test]
fn decode_report_reads_spec_attrs() {
    let event = json!({
        "event": "stage.openai_decode",
        "attributes": {
            "llama_stage.elapsed_ms": 123.5,
            "llama_stage.decode_token_count": 8,
            "llama_stage.spec.enabled": true,
            "llama_stage.spec.accepted": 3,
            "llama_stage.spec.rejected": 1,
            "llama_stage.spec.spd_rolling_executor_launches": 6,
            "llama_stage.spec.spd_rolling_executor_launch_misses": 2,
            "llama_stage.spec.spd_rolling_executor_margin_rejects": 1,
            "llama_stage.spec.spd_rolling_executor_max_in_flight": 4,
            "llama_stage.spec.spd_rolling_executor_accepted_oldest": 3,
            "llama_stage.spec.spd_rolling_executor_rejected_oldest": 0,
            "llama_stage.spec.spd_rolling_executor_drained_younger": 0,
            "llama_stage.spd_proposal.total.requested_limit": 8,
            "llama_stage.spd_proposal.total.attempts": 8,
            "llama_stage.spd_proposal.total.proposed": 7,
            "llama_stage.spd_proposal.total.inline_tap_hits": 2,
            "llama_stage.spd_proposal.total.replay_fallbacks": 5,
            "llama_stage.spd_proposal.total.cache_hits": 6,
            "llama_stage.spd_proposal.total.cache_misses": 1,
            "llama_stage.spd_proposal.total.tap_collect_ms": 100.0,
            "llama_stage.spd_proposal.total.cur_in_ms": 20.0,
            "llama_stage.spd_proposal.total.forward_ms": 300.0,
            "llama_stage.spd_proposal.total.cache_prefill_ms": 30.0,
            "llama_stage.spd_proposal.total.head_fixed_stage_projection_ms": 40.0,
            "llama_stage.spd_proposal.total.head_decoder_ms": 200.0,
            "llama_stage.spd_proposal.total.head_final_norm_ms": 5.0,
            "llama_stage.spd_proposal.total.head_lm_head_topk_ms": 25.0,
            "llama_stage.spd_proposal.total.head_total_ms": 270.0,
            "llama_stage.spd_proposal.total.last_cache_prefix_len": 31,
            "llama_stage.spd_proposal.total.max_cache_prefix_len": 31,
        }
    });

    let report = decode_report(&[event]).expect("decode report");

    assert_eq!(report.elapsed_ms, Some(123.5));
    assert_eq!(report.tokens, Some(8));
    assert_eq!(report.spec_enabled, Some(true));
    assert_eq!(report.spec_accepted, Some(3));
    assert_eq!(report.spec_rejected, Some(1));
    assert_eq!(report.spd_rolling_executor_launches, Some(6));
    assert_eq!(report.spd_rolling_executor_launch_misses, Some(2));
    assert_eq!(report.spd_rolling_executor_margin_rejects, Some(1));
    assert_eq!(report.spd_rolling_executor_max_in_flight, Some(4));
    assert_eq!(report.spd_rolling_executor_accepted_oldest, Some(3));
    assert_eq!(report.spd_rolling_executor_rejected_oldest, Some(0));
    assert_eq!(report.spd_rolling_executor_drained_younger, Some(0));
    assert_eq!(report.spd_proposal_total_requested_limit, Some(8));
    assert_eq!(report.spd_proposal_total_attempts, Some(8));
    assert_eq!(report.spd_proposal_total_proposed, Some(7));
    assert_eq!(report.spd_proposal_total_inline_tap_hits, Some(2));
    assert_eq!(report.spd_proposal_total_replay_fallbacks, Some(5));
    assert_eq!(report.spd_proposal_total_cache_hits, Some(6));
    assert_eq!(report.spd_proposal_total_cache_misses, Some(1));
    assert_eq!(report.spd_proposal_total_tap_collect_ms, Some(100.0));
    assert_eq!(report.spd_proposal_total_cur_in_ms, Some(20.0));
    assert_eq!(report.spd_proposal_total_forward_ms, Some(300.0));
    assert_eq!(report.spd_proposal_total_cache_prefill_ms, Some(30.0));
    assert_eq!(
        report.spd_proposal_total_head_fixed_stage_projection_ms,
        Some(40.0)
    );
    assert_eq!(report.spd_proposal_total_head_decoder_ms, Some(200.0));
    assert_eq!(report.spd_proposal_total_head_final_norm_ms, Some(5.0));
    assert_eq!(report.spd_proposal_total_head_lm_head_topk_ms, Some(25.0));
    assert_eq!(report.spd_proposal_total_head_total_ms, Some(270.0));
    assert_eq!(report.spd_proposal_total_last_cache_prefix_len, Some(31));
    assert_eq!(report.spd_proposal_total_max_cache_prefix_len, Some(31));
}

#[test]
fn decode_report_reads_live_rolling_attrs() {
    let event = json!({
        "event": "stage.openai_decode",
        "attributes": {
            "llama_stage.spd_rolling.logical_stage_count": 4,
            "llama_stage.spd_rolling.target_position": 6,
            "llama_stage.spd_rolling.next_position": 3,
            "llama_stage.spd_rolling.inserted_drafts": 2,
            "llama_stage.spd_rolling.missing_proposals": 2,
            "llama_stage.spd_rolling.first_missing_proposal_position": 3,
            "llama_stage.spd_rolling.out_of_order_proposals": 3,
            "llama_stage.spd_rolling.first_out_of_order_proposal_position": 4,
            "llama_stage.spd_rolling.verified_windows": 0,
            "llama_stage.spd_rolling.accepted_windows": 0,
            "llama_stage.spd_rolling.rejected_windows": 0,
            "llama_stage.spd_rolling.pipeline_len": 3,
            "llama_stage.spd_rolling.verified_up_to": 1,
            "llama_stage.spd_rolling.row_evicted_prefix_position": 2,
            "llama_stage.spd_rolling.row_positions": [2, 2, 3, 4, 5],
            "llama_stage.spd_rolling.row_i_stages": [4, 3, 2, 1, 0],
            "llama_stage.spd_rolling.row_newest_position": 5,
            "llama_stage.spd_rolling.row_next_draft_position": 6,
        }
    });

    let rolling = decode_report(&[event])
        .expect("decode report")
        .rolling
        .expect("rolling report");

    assert_eq!(rolling.logical_stage_count, Some(4));
    assert_eq!(rolling.target_position, Some(6));
    assert_eq!(rolling.next_position, Some(3));
    assert_eq!(rolling.inserted_drafts, Some(2));
    assert_eq!(rolling.missing_proposals, Some(2));
    assert_eq!(rolling.first_missing_proposal_position, Some(3));
    assert_eq!(rolling.out_of_order_proposals, Some(3));
    assert_eq!(rolling.first_out_of_order_proposal_position, Some(4));
    assert_eq!(rolling.verified_windows, Some(0));
    assert_eq!(rolling.pipeline_len, Some(3));
    assert_eq!(rolling.verified_up_to, Some(1));
    assert_eq!(rolling.row_evicted_prefix_position, Some(2));
    assert_eq!(rolling.row_positions, vec![2, 2, 3, 4, 5]);
    assert_eq!(rolling.row_i_stages, vec![4, 3, 2, 1, 0]);
    assert_eq!(rolling.row_newest_position, Some(5));
    assert_eq!(rolling.row_next_draft_position, Some(6));
}

#[test]
fn inline_probe_report_reads_rolling_verified_delta_attrs() {
    let event = json!({
        "event": "stage.openai_spd_inline_probe",
        "attributes": {
            "llama_stage.decode_step": 3,
            "llama_stage.spd_inline_probe_phase": "optimistic_commit",
            "llama_stage.spd_inline_probe_cache_used": true,
            "llama_stage.spd_inline_probe_cache_prefix_len": 23,
            "llama_stage.spd_inline_probe_tap_source": "inline",
            "llama_stage.spd_inline_probe_tap_collect_ms": 1.25,
            "llama_stage.spd_inline_probe_cur_in_ms": 2.5,
            "llama_stage.spd_inline_probe_forward_ms": 3.75,
            "llama_stage.spd_inline_probe_cache_prefill_ms": 0.5,
            "llama_stage.spd_inline_probe_head_fixed_stage_projection_ms": 0.75,
            "llama_stage.spd_inline_probe_head_decoder_ms": 1.5,
            "llama_stage.spd_inline_probe_head_decoder_layer_ms": [0.7, 0.8],
            "llama_stage.spd_inline_probe_head_final_norm_ms": 0.1,
            "llama_stage.spd_inline_probe_head_lm_head_topk_ms": 0.2,
            "llama_stage.spd_inline_probe_head_total_ms": 2.55,
            "llama_stage.spd_inline_probe_proposal_row_positions": [24,25,26,27],
            "llama_stage.spd_inline_probe_proposal_row_i_stages": [4,4,4,0],
            "llama_stage.spd_inline_probe_proposal_row_evicted_prefix_position": null,
            "llama_stage.spd_inline_probe_proposal_row_newest_position": 27,
            "llama_stage.spd_inline_probe_proposal_row_next_draft_position": 28,
            "llama_stage.spd_inline_probe_miss_reason": "missing_inline_taps",
            "llama_stage.spd_inline_probe_missing_taps": {"24": [29]},
            "llama_stage.spd_rolling.logical_stage_count": 4,
            "llama_stage.spd_rolling.verified_delta_start_position": 1,
            "llama_stage.spd_rolling.verified_delta_up_to": 2,
            "llama_stage.spd_rolling.verified_delta_tokens": [8340],
            "llama_stage.spd_rolling.verified_delta_token_count": 1,
        }
    });

    let probe = inline_probe_reports(&[event])
        .into_iter()
        .next()
        .expect("inline probe");
    let delta = probe
        .rolling_verified_delta
        .expect("rolling verified delta");

    assert_eq!(probe.step, Some(3));
    assert_eq!(probe.cache_used, Some(true));
    assert_eq!(probe.cache_prefix_len, Some(23));
    assert_eq!(probe.tap_source.as_deref(), Some("inline"));
    assert_eq!(probe.tap_collect_ms, Some(1.25));
    assert_eq!(probe.cur_in_ms, Some(2.5));
    assert_eq!(probe.forward_ms, Some(3.75));
    assert_eq!(probe.cache_prefill_ms, Some(0.5));
    assert_eq!(probe.head_fixed_stage_projection_ms, Some(0.75));
    assert_eq!(probe.head_decoder_ms, Some(1.5));
    assert_eq!(probe.head_decoder_layer_ms, vec![0.7, 0.8]);
    assert_eq!(probe.head_final_norm_ms, Some(0.1));
    assert_eq!(probe.head_lm_head_topk_ms, Some(0.2));
    assert_eq!(probe.head_total_ms, Some(2.55));
    assert_eq!(probe.phase.as_deref(), Some("optimistic_commit"));
    assert_eq!(probe.proposal_row_positions, vec![24, 25, 26, 27]);
    assert_eq!(probe.proposal_row_i_stages, vec![4, 4, 4, 0]);
    assert_eq!(probe.proposal_row_evicted_prefix_position, None);
    assert_eq!(probe.proposal_row_newest_position, Some(27));
    assert_eq!(probe.proposal_row_next_draft_position, Some(28));
    assert_eq!(
        probe.proposal_miss_reason.as_deref(),
        Some("missing_inline_taps")
    );
    assert_eq!(
        probe.proposal_missing_taps,
        BTreeMap::from([("24".to_string(), vec![29])])
    );
    assert_eq!(delta.start_position, Some(1));
    assert_eq!(delta.verified_up_to, Some(2));
    assert_eq!(delta.tokens, vec![8340]);
    assert_eq!(delta.token_count, Some(1));
}

#[test]
fn optimistic_decode_report_derives_hidden_wait_ms() {
    let event = json!({
        "event": "stage.openai_spd_optimistic_decode",
        "attributes": {
            "llama_stage.decode_step": 4,
            "llama_stage.spd_optimistic_chain": true,
            "llama_stage.spd_optimistic_chain_depth": 2,
            "llama_stage.spd_optimistic_proposed_token": 11,
            "llama_stage.spd_optimistic_target_token": 11,
            "llama_stage.spd_optimistic_accepted": true,
            "llama_stage.spd_optimistic_next_token": 12,
            "llama_stage.spd_optimistic_decode_elapsed_ms": 10.0,
            "llama_stage.spd_optimistic_start_elapsed_ms": 2.0,
            "llama_stage.spd_optimistic_decode_wait_ms": 3.0,
        }
    });

    let decode = optimistic_decode_reports(&[event])
        .into_iter()
        .next()
        .expect("optimistic decode");

    assert_eq!(decode.step, Some(4));
    assert_eq!(decode.chain, Some(true));
    assert_eq!(decode.chain_depth, Some(2));
    assert_eq!(decode.proposed_token, Some(11));
    assert_eq!(decode.accepted, Some(true));
    assert_eq!(decode.next_token, Some(12));
    assert_eq!(decode.hidden_wait_ms, Some(5.0));
}

#[test]
fn summary_compares_prompt_pairs_and_totals() {
    let cases = vec![
        test_case("baseline", 0, "same", 100.0, baseline_decode(50.0)),
        test_case("spd", 0, "same", 200.0, spd_decode(100.0, 4, 2, 2, 1)),
    ];

    let summary = summarize_cases(&cases, 7, 4);

    assert_eq!(summary.prompt_pairs, 1);
    assert_eq!(summary.matching_content, 1);
    assert_eq!(summary.wall_speedup_spd_vs_baseline, Some(0.5));
    assert_eq!(summary.decode_speedup_spd_vs_baseline, Some(0.5));
    assert_eq!(summary.spd_spec_proposed, 4);
    assert_eq!(summary.spd_spec_accepted, 2);
    assert_eq!(summary.spd_spec_rejected, 2);
    assert_eq!(summary.spd_accept_rate, Some(0.5));
    assert_eq!(summary.optimistic_committed, 1);
    assert_eq!(summary.tap_ignored, 1);
    assert_eq!(summary.paper_pipeline_estimate.logical_stage_count, 4);
    assert_eq!(summary.paper_pipeline_estimate.physical_stage_count, 7);
    assert_eq!(
        summary
            .paper_pipeline_estimate
            .paper_like_speedup_vs_serial_split,
        Some(2.0)
    );
    assert_eq!(
        summary
            .paper_pipeline_estimate
            .estimated_decode_ms_at_baseline_stage_cost,
        Some(25.0)
    );
    assert_eq!(
        summary
            .paper_pipeline_estimate
            .current_spd_decode_slowdown_vs_estimate,
        Some(4.0)
    );
    assert!(summary.prompt_comparisons[0].content_matches);
}

#[test]
fn summary_ignores_warmups_and_pairs_measured_repeats() {
    let mut warmup_baseline = test_case("baseline", 0, "warmup", 1000.0, baseline_decode(500.0));
    warmup_baseline.warmup = true;
    let mut warmup_spd = test_case("spd", 0, "warmup", 2000.0, spd_decode(1000.0, 10, 10, 0, 0));
    warmup_spd.warmup = true;

    let mut baseline_repeat_0 = test_case("baseline", 0, "r0", 100.0, baseline_decode(50.0));
    baseline_repeat_0.repeat_index = 0;
    let mut spd_repeat_0 = test_case("spd", 0, "r0", 200.0, spd_decode(100.0, 4, 2, 2, 1));
    spd_repeat_0.repeat_index = 0;

    let mut baseline_repeat_1 = test_case("baseline", 0, "r1", 120.0, baseline_decode(60.0));
    baseline_repeat_1.repeat_index = 1;
    let mut spd_repeat_1 = test_case("spd", 0, "mismatch", 240.0, spd_decode(120.0, 4, 2, 2, 1));
    spd_repeat_1.repeat_index = 1;

    let cases = vec![
        warmup_baseline,
        warmup_spd,
        baseline_repeat_0,
        spd_repeat_0,
        baseline_repeat_1,
        spd_repeat_1,
    ];

    let summary = summarize_cases(&cases, 7, 4);

    assert_eq!(summary.prompt_pairs, 2);
    assert_eq!(summary.matching_content, 1);
    assert_eq!(summary.baseline_wall_ms.count, 2);
    assert_eq!(summary.spd_wall_ms.count, 2);
    assert_eq!(summary.spd_spec_proposed, 8);
    assert_eq!(summary.prompt_comparisons[0].repeat_index, 0);
    assert_eq!(summary.prompt_comparisons[1].repeat_index, 1);
    assert!(summary.prompt_comparisons[0].content_matches);
    assert!(!summary.prompt_comparisons[1].content_matches);
}

#[test]
fn content_match_gate_allows_matching_pairs() {
    let comparisons = vec![prompt_comparison(0, "prompt-000", true)];

    assert!(content_mismatch_failure(&comparisons, false).is_none());
}

#[test]
fn content_match_gate_fails_mismatched_pairs() {
    let comparisons = vec![
        prompt_comparison(0, "prompt-000", true),
        prompt_comparison(1, "prompt-001", false),
    ];

    let message = content_mismatch_failure(&comparisons, false)
        .expect("mismatched content should fail by default");

    assert!(message.contains("1 paired prompt"));
    assert!(message.contains("prompt-001#1"));
    assert!(message.contains("--allow-content-mismatch"));
}

#[test]
fn content_match_gate_allows_mismatch_when_opted_out() {
    let comparisons = vec![prompt_comparison(0, "prompt-000", false)];

    assert!(content_mismatch_failure(&comparisons, true).is_none());
}

#[test]
fn summary_reports_pipeline_gap_metrics() {
    let mut spd = test_case("spd", 0, "same", 200.0, spd_decode(100.0, 4, 2, 2, 1));
    spd.inline_probes = vec![
        inline_probe("pre_target_reply", Some(11), Some(true), 10.0, 2.0),
        inline_probe("pre_target_reply", Some(12), Some(false), 14.0, 4.0),
        inline_probe("optimistic_commit", Some(13), Some(true), 4.0, 1.0),
        inline_probe("post_target_reply", None, Some(false), 0.0, 0.0),
    ];
    spd.optimistic_decodes = vec![
        optimistic_decode(true, true),
        {
            let mut decode = optimistic_decode(true, false);
            decode.chain = Some(true);
            decode.chain_depth = Some(1);
            decode.elapsed_ms = Some(8.0);
            decode.start_elapsed_ms = Some(2.0);
            decode.wait_ms = Some(3.0);
            decode.hidden_wait_ms = Some(3.0);
            decode
        },
        optimistic_decode(false, false),
    ];
    spd.token_events = vec![
        token_event("DecodeEmbd", 20.0),
        token_event("DecodeEmbd", 40.0),
        token_event("DecodeEmbdOptimistic", 5.0),
    ];
    let cases = vec![
        test_case("baseline", 0, "same", 100.0, baseline_decode(50.0)),
        spd,
    ];

    let summary = summarize_cases(&cases, 7, 4);
    let gap = summary.pipeline_gap;

    assert_eq!(gap.pre_target_probes, 2);
    assert_eq!(gap.pre_target_proposals, 2);
    assert_eq!(gap.pre_target_accepted, 1);
    assert_eq!(gap.pre_target_accept_rate, Some(0.5));
    assert_eq!(gap.optimistic_commit_probes, 1);
    assert_eq!(gap.optimistic_commit_proposals, 1);
    assert_eq!(gap.optimistic_commit_accepted, 1);
    assert_eq!(gap.optimistic_commit_accept_rate, Some(1.0));
    assert_eq!(gap.post_target_probes, 1);
    assert_eq!(gap.post_target_empty, 1);
    assert_eq!(gap.post_target_empty_rate, Some(1.0));
    assert_eq!(gap.pre_target_probe_ms.mean_ms, Some(12.0));
    assert_eq!(gap.pre_target_wait_after_probe_ms.mean_ms, Some(3.0));
    assert_eq!(gap.optimistic_commit_probe_ms.mean_ms, Some(4.0));
    assert_eq!(gap.optimistic_commit_wait_after_probe_ms.mean_ms, Some(1.0));
    assert_eq!(gap.optimistic_decode_elapsed_ms.mean_ms, Some(10.0 / 3.0));
    assert_eq!(gap.optimistic_decode_start_elapsed_ms.mean_ms, Some(1.0));
    assert_eq!(gap.optimistic_decode_wait_ms.mean_ms, Some(4.0 / 3.0));
    assert_eq!(gap.optimistic_decode_hidden_wait_ms.mean_ms, Some(1.0));
    assert_eq!(
        gap.chained_optimistic_decode_hidden_wait_ms.mean_ms,
        Some(3.0)
    );
    assert_eq!(gap.normal_token_downstream_wait_ms.mean_ms, Some(30.0));
    assert_eq!(gap.optimistic_token_downstream_wait_ms.mean_ms, Some(5.0));
    assert_eq!(gap.pre_target_proposals_without_tap_return, 0);
    assert_eq!(gap.optimistic_tap_return_requests, 2);
    assert_eq!(gap.optimistic_tap_return_accepted, 1);
    assert_eq!(gap.optimistic_tap_return_rejected, 1);
    assert_eq!(gap.optimistic_tap_return_accept_rate, Some(0.5));
}

#[test]
fn summary_replays_observed_trace_through_rolling_scheduler() {
    let mut spd = test_case("spd", 0, "same", 200.0, spd_decode(100.0, 4, 2, 2, 1));
    spd.token_events = vec![
        token_event_at_step(0, 10, "DecodeEmbd", 20.0),
        token_event_at_step(1, 11, "DecodeEmbd", 20.0),
        token_event_at_step(2, 12, "DecodeEmbd", 20.0),
        token_event_at_step(3, 13, "DecodeEmbd", 20.0),
        token_event_at_step(4, 14, "DecodeEmbd", 20.0),
    ];
    spd.inline_probes = vec![
        inline_probe_at_step(1, Some(11), Some(true), 1.0, 0.0),
        inline_probe_at_step(2, Some(99), Some(false), 1.0, 0.0),
        inline_probe_at_step(3, Some(13), Some(true), 1.0, 0.0),
        inline_probe_at_step(4, Some(14), Some(true), 1.0, 0.0),
    ];
    let cases = vec![
        test_case("baseline", 0, "same", 100.0, baseline_decode(50.0)),
        spd,
    ];

    let summary = summarize_cases(&cases, 7, 4);
    let replay = summary.rolling_trace_replay;

    assert_eq!(replay.logical_stage_count, 4);
    assert_eq!(replay.cases_replayed, 1);
    assert_eq!(replay.inserted_drafts, 4);
    assert_eq!(replay.missing_proposals, 0);
    assert_eq!(replay.first_missing_proposal_position, None);
    assert_eq!(replay.out_of_order_proposals, 0);
    assert_eq!(replay.first_out_of_order_proposal_position, None);
    assert_eq!(replay.verified_windows, 2);
    assert_eq!(replay.accepted_windows, 1);
    assert_eq!(replay.rejected_windows, 1);
    assert_eq!(replay.first_rejected_target_position, Some(2));
    assert_eq!(replay.final_pipeline_len, Some(1));
    assert_eq!(replay.final_verified_up_to, Some(5));
    assert_eq!(replay.final_verified_prefix_len, Some(5));
    assert_eq!(
        replay.final_verified_prefix_tokens,
        vec![10, 11, 12, 13, 14]
    );
    assert_eq!(replay.verified_prefix_matches_target, Some(true));
    assert_eq!(replay.first_verified_prefix_mismatch_position, None);
}

#[test]
fn summary_replay_preserves_observed_rolling_target_positions() {
    let mut spd = test_case("spd", 0, "same", 200.0, spd_decode(100.0, 4, 2, 2, 1));
    spd.token_events = vec![
        token_event_at_step(0, 10, "DecodeEmbd", 20.0),
        token_event_at_step(1, 11, "DecodeEmbd", 20.0),
        token_event_at_step(2, 12, "DecodeEmbd", 20.0),
        token_event_at_step(3, 13, "DecodeEmbd", 20.0),
        token_event_at_step(4, 14, "DecodeEmbd", 20.0),
    ];
    spd.inline_probes = vec![
        inline_probe_at_position(0, 23, Some(10), Some(true)),
        inline_probe_at_position(1, 24, Some(11), Some(true)),
        inline_probe_at_position(2, 25, Some(99), Some(false)),
        inline_probe_at_position(3, 26, Some(13), Some(true)),
        inline_probe_at_position(4, 27, Some(14), Some(true)),
    ];
    let cases = vec![
        test_case("baseline", 0, "same", 100.0, baseline_decode(50.0)),
        spd,
    ];

    let summary = summarize_cases(&cases, 7, 4);
    let replay = summary.rolling_trace_replay;

    assert_eq!(replay.inserted_drafts, 4);
    assert_eq!(replay.first_rejected_target_position, Some(25));
    assert_eq!(replay.final_verified_up_to, Some(28));
    assert_eq!(replay.final_verified_prefix_len, Some(5));
    assert_eq!(
        replay.final_verified_prefix_tokens,
        vec![10, 11, 12, 13, 14]
    );
    assert_eq!(replay.verified_prefix_matches_target, Some(true));
}

#[test]
fn summary_replay_does_not_shift_proposals_across_missing_position() {
    let mut spd = test_case("spd", 0, "same", 200.0, spd_decode(100.0, 4, 2, 2, 1));
    spd.token_events = vec![
        token_event_at_step(0, 10, "DecodeEmbd", 20.0),
        token_event_at_step(1, 11, "DecodeEmbd", 20.0),
        token_event_at_step(2, 12, "DecodeEmbd", 20.0),
        token_event_at_step(3, 13, "DecodeEmbdOptimistic", 20.0),
        token_event_at_step(4, 14, "DecodeEmbd", 20.0),
        token_event_at_step(5, 15, "DecodeEmbd", 20.0),
    ];
    spd.inline_probes = vec![
        inline_probe_at_step(1, Some(11), Some(true), 1.0, 0.0),
        inline_probe_at_step(2, Some(12), Some(true), 1.0, 0.0),
        inline_probe_at_step(4, Some(14), Some(true), 1.0, 0.0),
        inline_probe_at_step(5, Some(15), Some(true), 1.0, 0.0),
    ];
    let cases = vec![
        test_case("baseline", 0, "same", 100.0, baseline_decode(50.0)),
        spd,
    ];

    let summary = summarize_cases(&cases, 7, 4);
    let replay = summary.rolling_trace_replay;

    assert_eq!(replay.inserted_drafts, 4);
    assert_eq!(replay.missing_proposals, 1);
    assert_eq!(replay.first_missing_proposal_position, Some(3));
    assert_eq!(replay.out_of_order_proposals, 0);
    assert_eq!(replay.first_out_of_order_proposal_position, None);
    assert_eq!(replay.verified_windows, 0);
    assert_eq!(replay.final_pipeline_len, Some(3));
    assert_eq!(replay.final_verified_up_to, Some(4));
    assert_eq!(replay.final_verified_prefix_len, Some(4));
    assert_eq!(replay.final_verified_prefix_tokens, vec![10, 11, 12, 13]);
    assert_eq!(replay.verified_prefix_matches_target, Some(true));
}

#[test]
fn summary_replay_counts_optimistic_commit_proposals() {
    let mut spd = test_case("spd", 0, "same", 200.0, spd_decode(100.0, 4, 2, 2, 1));
    spd.token_events = vec![
        token_event_at_step(0, 10, "DecodeEmbd", 20.0),
        token_event_at_step(1, 11, "DecodeEmbd", 20.0),
        token_event_at_step(2, 12, "DecodeEmbd", 20.0),
        token_event_at_step(3, 13, "DecodeEmbdOptimistic", 20.0),
        token_event_at_step(4, 14, "DecodeEmbd", 20.0),
    ];
    spd.inline_probes = vec![
        inline_probe_at_step(1, Some(11), Some(true), 1.0, 0.0),
        inline_probe_at_step(2, Some(12), Some(true), 1.0, 0.0),
        inline_probe_at_step_with_phase(3, "optimistic_commit", Some(13), Some(true), 1.0, 0.0),
        inline_probe_at_step(4, Some(14), Some(true), 1.0, 0.0),
    ];
    let cases = vec![
        test_case("baseline", 0, "same", 100.0, baseline_decode(50.0)),
        spd,
    ];

    let summary = summarize_cases(&cases, 7, 4);
    let replay = summary.rolling_trace_replay;

    assert_eq!(replay.inserted_drafts, 4);
    assert_eq!(replay.missing_proposals, 0);
    assert_eq!(replay.out_of_order_proposals, 0);
    assert_eq!(replay.verified_windows, 2);
    assert_eq!(replay.accepted_windows, 2);
    assert_eq!(replay.final_pipeline_len, Some(3));
    assert_eq!(replay.final_verified_up_to, Some(3));
    assert_eq!(replay.final_verified_prefix_len, Some(3));
    assert_eq!(replay.final_verified_prefix_tokens, vec![10, 11, 12]);
    assert_eq!(replay.verified_prefix_matches_target, Some(true));
}

#[test]
fn summary_uses_live_decode_rolling_when_trace_events_are_absent() {
    let mut spd = test_case("spd", 0, "same", 200.0, spd_decode(100.0, 8, 8, 0, 0));
    spd.decode.as_mut().unwrap().rolling = Some(SpdLiveRollingReport {
        logical_stage_count: Some(4),
        target_position: Some(33),
        next_position: Some(34),
        inserted_drafts: Some(7),
        missing_proposals: Some(0),
        first_missing_proposal_position: None,
        out_of_order_proposals: Some(0),
        first_out_of_order_proposal_position: None,
        verified_windows: Some(5),
        accepted_windows: Some(5),
        rejected_windows: Some(0),
        first_rejected_target_position: None,
        pipeline_len: Some(3),
        verified_up_to: Some(32),
        row_evicted_prefix_position: None,
        row_positions: Vec::new(),
        row_i_stages: Vec::new(),
        row_newest_position: None,
        row_next_draft_position: None,
    });
    let cases = vec![
        test_case("baseline", 0, "same", 100.0, baseline_decode(50.0)),
        spd,
    ];

    let summary = summarize_cases(&cases, 7, 4);
    let replay = summary.rolling_trace_replay;

    assert_eq!(replay.cases_replayed, 0);
    assert_eq!(replay.live_cases_observed, 1);
    assert_eq!(replay.inserted_drafts, 7);
    assert_eq!(replay.missing_proposals, 0);
    assert_eq!(replay.out_of_order_proposals, 0);
    assert_eq!(replay.verified_windows, 5);
    assert_eq!(replay.accepted_windows, 5);
    assert_eq!(replay.rejected_windows, 0);
    assert_eq!(replay.final_pipeline_len, Some(3));
    assert_eq!(replay.final_verified_up_to, Some(32));
    assert_eq!(replay.final_verified_prefix_len, None);
    assert_eq!(replay.verified_prefix_matches_target, None);
}

fn test_case(
    name: &'static str,
    prompt_index: usize,
    content: &str,
    elapsed_ms: f64,
    decode: DecodeReport,
) -> CaseReport {
    CaseReport {
        name,
        prompt_index,
        prompt_label: format!("prompt-{prompt_index:03}"),
        warmup: false,
        repeat_index: 0,
        prompt: "prompt".to_string(),
        run_id: format!("{name}-{prompt_index}"),
        openai_base_url: "http://127.0.0.1:1".to_string(),
        logs_dir: "/tmp/skippy-bench-test".to_string(),
        elapsed_ms,
        content: content.to_string(),
        usage: None,
        finish_reason: Some("length".to_string()),
        decode: Some(decode),
        inline_probes: Vec::new(),
        optimistic_decodes: Vec::new(),
        token_events: Vec::new(),
        tap_returns_by_hf_index: BTreeMap::new(),
        tap_records_by_hf_index: BTreeMap::new(),
        tap_return_failures: usize::from(name == "spd"),
        tap_record_failures: usize::from(name == "spd"),
        tap_ignored: usize::from(name == "spd"),
    }
}

fn prompt_comparison(
    prompt_index: usize,
    prompt_label: &str,
    content_matches: bool,
) -> PromptComparisonReport {
    PromptComparisonReport {
        prompt_index,
        prompt_label: prompt_label.to_string(),
        repeat_index: 0,
        content_matches,
        baseline_elapsed_ms: 100.0,
        spd_elapsed_ms: 200.0,
        wall_speedup_spd_vs_baseline: Some(0.5),
        baseline_decode_ms: Some(50.0),
        spd_decode_ms: Some(100.0),
        decode_speedup_spd_vs_baseline: Some(0.5),
        spd_spec_proposed: Some(4),
        spd_spec_accepted: Some(2),
        spd_spec_rejected: Some(2),
        spd_accept_rate: Some(0.5),
        optimistic_committed: Some(1),
    }
}

fn inline_probe(
    phase: &str,
    proposed: Option<i64>,
    accepted: Option<bool>,
    elapsed_ms: f64,
    wait_after_probe_ms: f64,
) -> InlineProbeReport {
    inline_probe_at_step_with_phase(
        0,
        phase,
        proposed,
        accepted,
        elapsed_ms,
        wait_after_probe_ms,
    )
}

fn inline_probe_at_step(
    step: u64,
    proposed: Option<i64>,
    accepted: Option<bool>,
    elapsed_ms: f64,
    wait_after_probe_ms: f64,
) -> InlineProbeReport {
    inline_probe_at_step_with_phase(
        step,
        "pre_target_reply",
        proposed,
        accepted,
        elapsed_ms,
        wait_after_probe_ms,
    )
}

fn inline_probe_at_step_with_phase(
    step: u64,
    phase: &str,
    proposed: Option<i64>,
    accepted: Option<bool>,
    elapsed_ms: f64,
    wait_after_probe_ms: f64,
) -> InlineProbeReport {
    InlineProbeReport {
        step: Some(step),
        phase: Some(phase.to_string()),
        elapsed_ms: Some(elapsed_ms),
        target_wait_after_probe_ms: Some(wait_after_probe_ms),
        current_token: Some(1),
        proposed_token: proposed,
        proposed_logit: None,
        proposed_logit_margin: None,
        cache_used: None,
        cache_prefix_len: None,
        tap_source: None,
        tap_collect_ms: None,
        cur_in_ms: None,
        forward_ms: None,
        cache_prefill_ms: None,
        head_fixed_stage_projection_ms: None,
        head_decoder_ms: None,
        head_decoder_layer_ms: Vec::new(),
        head_final_norm_ms: None,
        head_lm_head_topk_ms: None,
        head_total_ms: None,
        target_token: Some(2),
        accepted,
        trigger_hf_index: Some(31),
        proposal_row_positions: Vec::new(),
        proposal_row_i_stages: Vec::new(),
        proposal_row_evicted_prefix_position: None,
        proposal_row_newest_position: None,
        proposal_row_next_draft_position: None,
        proposal_miss_reason: None,
        proposal_missing_taps: BTreeMap::new(),
        rolling: None,
        rolling_verified_delta: None,
    }
}

fn inline_probe_at_position(
    step: u64,
    target_position: u64,
    proposed: Option<i64>,
    accepted: Option<bool>,
) -> InlineProbeReport {
    let mut probe = inline_probe_at_step(step, proposed, accepted, 1.0, 0.0);
    probe.rolling = Some(rolling_report_at_position(target_position));
    probe
}

fn rolling_report_at_position(target_position: u64) -> SpdLiveRollingReport {
    SpdLiveRollingReport {
        logical_stage_count: Some(4),
        target_position: Some(target_position),
        next_position: None,
        inserted_drafts: None,
        missing_proposals: None,
        first_missing_proposal_position: None,
        out_of_order_proposals: None,
        first_out_of_order_proposal_position: None,
        verified_windows: None,
        accepted_windows: None,
        rejected_windows: None,
        first_rejected_target_position: None,
        pipeline_len: None,
        verified_up_to: None,
        row_evicted_prefix_position: None,
        row_positions: Vec::new(),
        row_i_stages: Vec::new(),
        row_newest_position: None,
        row_next_draft_position: None,
    }
}

fn optimistic_decode(requested_tap_return: bool, accepted: bool) -> OptimisticDecodeReport {
    OptimisticDecodeReport {
        step: Some(0),
        chain: None,
        chain_depth: None,
        proposed_token: Some(1),
        proposed_logit: None,
        proposed_logit_margin: None,
        requested_tap_return: Some(requested_tap_return),
        target_token: Some(1),
        accepted: Some(accepted),
        next_token: Some(2),
        checkpoint_ms: Some(0.0),
        elapsed_ms: Some(1.0),
        start_elapsed_ms: Some(0.5),
        wait_ms: Some(0.5),
        hidden_wait_ms: Some(0.0),
        stage0_compute_ms: Some(0.5),
    }
}

fn token_event(kind: &str, downstream_wait_ms: f64) -> TokenEventReport {
    token_event_at_step(0, 1, kind, downstream_wait_ms)
}

fn token_event_at_step(
    step: u64,
    predicted_token: i64,
    kind: &str,
    downstream_wait_ms: f64,
) -> TokenEventReport {
    TokenEventReport {
        step: Some(step),
        message_kind: Some(kind.to_string()),
        predicted_token: Some(predicted_token),
        chain: None,
        chain_depth: None,
        downstream_wait_ms: Some(downstream_wait_ms),
    }
}

fn baseline_decode(elapsed_ms: f64) -> DecodeReport {
    DecodeReport {
        elapsed_ms: Some(elapsed_ms),
        tokens: Some(2),
        spec_enabled: Some(false),
        spec_windows: None,
        spec_proposed: None,
        spec_accepted: None,
        spec_rejected: None,
        spec_draft_propose_ms: None,
        optimistic_requests: None,
        optimistic_accepted: None,
        optimistic_rejected: None,
        optimistic_committed: None,
        optimistic_checkpoint_ms: None,
        optimistic_decode_elapsed_ms: None,
        optimistic_decode_wait_ms: None,
        optimistic_restore_ms: None,
        chained_optimistic_requests: None,
        chained_optimistic_accepted: None,
        chained_optimistic_rejected: None,
        chained_optimistic_committed: None,
        spd_rolling_executor_launches: None,
        spd_rolling_executor_launch_misses: None,
        spd_rolling_executor_margin_rejects: None,
        spd_rolling_executor_max_in_flight: None,
        spd_rolling_executor_accepted_oldest: None,
        spd_rolling_executor_rejected_oldest: None,
        spd_rolling_executor_drained_younger: None,
        stage0_compute_ms: Some(5.0),
        downstream_wait_ms: Some(45.0),
        spd_proposal_total_requested_limit: None,
        spd_proposal_total_attempts: None,
        spd_proposal_total_proposed: None,
        spd_proposal_total_inline_tap_hits: None,
        spd_proposal_total_replay_fallbacks: None,
        spd_proposal_total_cache_hits: None,
        spd_proposal_total_cache_misses: None,
        spd_proposal_total_tap_collect_ms: None,
        spd_proposal_total_cur_in_ms: None,
        spd_proposal_total_forward_ms: None,
        spd_proposal_total_cache_prefill_ms: None,
        spd_proposal_total_head_fixed_stage_projection_ms: None,
        spd_proposal_total_head_decoder_ms: None,
        spd_proposal_total_head_final_norm_ms: None,
        spd_proposal_total_head_lm_head_topk_ms: None,
        spd_proposal_total_head_total_ms: None,
        spd_proposal_total_last_cache_prefix_len: None,
        spd_proposal_total_max_cache_prefix_len: None,
        rolling: None,
    }
}

fn spd_decode(
    elapsed_ms: f64,
    proposed: u64,
    accepted: u64,
    rejected: u64,
    optimistic_committed: u64,
) -> DecodeReport {
    DecodeReport {
        elapsed_ms: Some(elapsed_ms),
        tokens: Some(2),
        spec_enabled: Some(true),
        spec_windows: Some(proposed),
        spec_proposed: Some(proposed),
        spec_accepted: Some(accepted),
        spec_rejected: Some(rejected),
        spec_draft_propose_ms: Some(10.0),
        optimistic_requests: Some(1),
        optimistic_accepted: Some(1),
        optimistic_rejected: Some(0),
        optimistic_committed: Some(optimistic_committed),
        optimistic_checkpoint_ms: Some(1.0),
        optimistic_decode_elapsed_ms: Some(2.0),
        optimistic_decode_wait_ms: Some(3.0),
        optimistic_restore_ms: Some(0.0),
        chained_optimistic_requests: None,
        chained_optimistic_accepted: None,
        chained_optimistic_rejected: None,
        chained_optimistic_committed: None,
        spd_rolling_executor_launches: None,
        spd_rolling_executor_launch_misses: None,
        spd_rolling_executor_margin_rejects: None,
        spd_rolling_executor_max_in_flight: None,
        spd_rolling_executor_accepted_oldest: None,
        spd_rolling_executor_rejected_oldest: None,
        spd_rolling_executor_drained_younger: None,
        stage0_compute_ms: Some(5.0),
        downstream_wait_ms: Some(95.0),
        spd_proposal_total_requested_limit: None,
        spd_proposal_total_attempts: None,
        spd_proposal_total_proposed: None,
        spd_proposal_total_inline_tap_hits: None,
        spd_proposal_total_replay_fallbacks: None,
        spd_proposal_total_cache_hits: None,
        spd_proposal_total_cache_misses: None,
        spd_proposal_total_tap_collect_ms: None,
        spd_proposal_total_cur_in_ms: None,
        spd_proposal_total_forward_ms: None,
        spd_proposal_total_cache_prefill_ms: None,
        spd_proposal_total_head_fixed_stage_projection_ms: None,
        spd_proposal_total_head_decoder_ms: None,
        spd_proposal_total_head_final_norm_ms: None,
        spd_proposal_total_head_lm_head_topk_ms: None,
        spd_proposal_total_head_total_ms: None,
        spd_proposal_total_last_cache_prefix_len: None,
        spd_proposal_total_max_cache_prefix_len: None,
        rolling: None,
    }
}
