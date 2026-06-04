use super::*;

pub(super) fn prompt_anchor_positions(
    anchor_token_count: Option<usize>,
    context_token_count: usize,
) -> Vec<i32> {
    let Some(anchor_token_count) = anchor_token_count else {
        return Vec::new();
    };
    if anchor_token_count == 0 || anchor_token_count >= context_token_count {
        return Vec::new();
    }
    let Ok(anchor_token_count) = i32::try_from(anchor_token_count) else {
        return Vec::new();
    };
    vec![anchor_token_count]
}

impl StageOpenAiBackend {
    pub(super) fn try_restore_embedded_split_prompt_anchor(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        session_key: &str,
        downstream: &mut TcpStream,
    ) -> OpenAiResult<Option<ChainPrefixRestore>> {
        let Some(anchor_token_count) = request.prompt_anchor_token_count else {
            return Ok(None);
        };
        let Some(anchor_tokens) = request.prompt_token_ids.get(..anchor_token_count) else {
            return Ok(None);
        };
        let Some(restore) = self.try_restore_embedded_split_prefill(
            request,
            session_key,
            downstream,
            anchor_tokens,
        )?
        else {
            return Ok(None);
        };
        if restore.restored_tokens < anchor_tokens.len() {
            return Ok(None);
        }
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert(
            "skippy.kv.decision".to_string(),
            json!("chain_prompt_anchor_hit"),
        );
        attrs.insert(
            "skippy.kv.anchor_token_count".to_string(),
            json!(anchor_tokens.len()),
        );
        attrs.insert(
            "skippy.kv.restored_tokens".to_string(),
            json!(restore.restored_tokens),
        );
        attrs.insert(
            "skippy.kv.lookup_hits".to_string(),
            json!(restore.stats.kv_lookup_hits),
        );
        attrs.insert(
            "skippy.kv.hit_stage_mask".to_string(),
            json!(restore.stats.kv_hit_stage_mask),
        );
        super::prefix_cache::insert_chain_prefix_cache_savings_attrs(
            &mut attrs,
            super::prefix_cache::chain_prefix_cache_savings(
                &restore.stats,
                anchor_tokens.len(),
                request.wire_dtype,
                request.activation_width,
            ),
        );
        self.telemetry
            .emit("stage.openai_kv_lookup_decision", attrs);
        Ok(Some(restore))
    }

    pub(super) fn record_embedded_stage0_prompt_anchor(
        &self,
        session_id: &str,
        ids: &OpenAiGenerationIds,
        prompt_token_ids: &[i32],
        anchor_token_count: Option<usize>,
    ) -> OpenAiResult<bool> {
        let Some(anchor_token_count) = anchor_token_count else {
            return Ok(false);
        };
        let Some(anchor_tokens) = prompt_token_ids.get(..anchor_token_count) else {
            return Ok(false);
        };
        let recorded = self.record_embedded_stage0_full_prefill(session_id, ids, anchor_tokens)?;
        let mut attrs = self.openai_attrs(ids);
        attrs.insert(
            "skippy.kv.decision".to_string(),
            json!("stage0_prompt_anchor_record"),
        );
        attrs.insert(
            "skippy.kv.anchor_token_count".to_string(),
            json!(anchor_tokens.len()),
        );
        attrs.insert("skippy.kv.recorded_anchor".to_string(), json!(recorded));
        self.telemetry
            .emit("stage.openai_kv_record_decision", attrs);
        Ok(recorded)
    }
}
