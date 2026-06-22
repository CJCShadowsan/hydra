use super::*;

pub(super) struct DecodeMessageArgs {
    pub(super) request_id: u64,
    pub(super) session_id: u64,
    pub(super) prompt_token_count: usize,
    pub(super) pos_start: usize,
    pub(super) decode_step: usize,
    pub(super) current: i32,
    pub(super) sampling: Option<WireSamplingConfig>,
}

pub(super) fn embedded_decode_message(
    wire_dtype: WireActivationDType,
    args: DecodeMessageArgs,
) -> OpenAiResult<StageWireMessage> {
    let mut message = ReusableDecodeMessage::new(
        wire_dtype,
        ReusableDecodeMessageArgs {
            request_id: args.request_id,
            session_id: args.session_id,
            prompt_token_count: args.prompt_token_count,
            base_pos_start: args.pos_start,
            sampling: args.sampling,
            sideband_capacity: 1,
        },
    )?;
    message.update_at_pos(
        args.decode_step,
        args.pos_start,
        args.current,
        &[args.current],
    )?;
    Ok(message.into_message())
}

pub(super) struct ReusableDecodeMessageArgs {
    pub(super) request_id: u64,
    pub(super) session_id: u64,
    pub(super) prompt_token_count: usize,
    pub(super) base_pos_start: usize,
    pub(super) sampling: Option<WireSamplingConfig>,
    pub(super) sideband_capacity: usize,
}

pub(super) struct ReusableDecodeMessage {
    message: StageWireMessage,
    base_pos_start: usize,
}

impl ReusableDecodeMessage {
    pub(super) fn new(
        wire_dtype: WireActivationDType,
        args: ReusableDecodeMessageArgs,
    ) -> OpenAiResult<Self> {
        let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, wire_dtype);
        state.seq_id = 0;
        state.prompt_token_count = i32::try_from(args.prompt_token_count)
            .map_err(|_| OpenAiError::backend("prompt token count exceeds i32"))?;
        state.source_stage_index = -1;
        Ok(Self {
            message: StageWireMessage {
                kind: WireMessageKind::DecodeEmbd,
                pos_start: i32::try_from(args.base_pos_start)
                    .map_err(|_| OpenAiError::backend("decode position exceeds i32"))?,
                token_count: 1,
                state,
                request_id: args.request_id,
                session_id: args.session_id,
                sampling: args.sampling,
                chat_sampling_metadata: None,
                tokens: Vec::with_capacity(args.sideband_capacity.max(1)),
                positions: Vec::new(),
                activation: Vec::new(),
                raw_bytes: Vec::new(),
            },
            base_pos_start: args.base_pos_start,
        })
    }

    pub(super) fn update(
        &mut self,
        decode_step: usize,
        current: i32,
    ) -> OpenAiResult<&StageWireMessage> {
        self.update_with_tokens(decode_step, current, &[current])
    }

    pub(super) fn update_with_tokens(
        &mut self,
        decode_step: usize,
        current: i32,
        tokens: &[i32],
    ) -> OpenAiResult<&StageWireMessage> {
        let pos_start = self
            .base_pos_start
            .checked_add(decode_step)
            .ok_or_else(|| OpenAiError::backend("decode position overflow"))?;
        self.update_at_pos(decode_step, pos_start, current, tokens)
    }

    fn update_at_pos(
        &mut self,
        decode_step: usize,
        pos_start: usize,
        current: i32,
        tokens: &[i32],
    ) -> OpenAiResult<&StageWireMessage> {
        self.message.pos_start = i32::try_from(pos_start)
            .map_err(|_| OpenAiError::backend("decode position exceeds i32"))?;
        self.message.state.decode_step = i32::try_from(decode_step)
            .map_err(|_| OpenAiError::backend("decode step exceeds i32"))?;
        self.message.state.current_token = current;
        self.message.tokens.clear();
        self.message.tokens.extend_from_slice(tokens);
        Ok(&self.message)
    }

    fn into_message(self) -> StageWireMessage {
        self.message
    }
}

pub(super) struct VerifySpanMessageArgs<'a> {
    pub(super) request_id: u64,
    pub(super) session_id: u64,
    pub(super) prompt_token_count: usize,
    pub(super) pos_start: usize,
    pub(super) decode_step: usize,
    pub(super) tokens: &'a [i32],
    pub(super) sampling: Option<WireSamplingConfig>,
    pub(super) checkpoint: bool,
}

pub(super) struct TreeVerifyMessageArgs<'a> {
    pub(super) request_id: u64,
    pub(super) session_id: u64,
    pub(super) prompt_token_count: usize,
    pub(super) pos_start: usize,
    pub(super) decode_step: usize,
    pub(super) tokens: &'a [i32],
    pub(super) parents: &'a [i32],
    pub(super) depths: &'a [u32],
    pub(super) sampling: Option<WireSamplingConfig>,
    pub(super) checkpoint: bool,
    pub(super) gather: Option<TreeGatherMessageArgs<'a>>,
}

#[derive(Clone, Copy)]
pub(super) struct TreeGatherMessageArgs<'a> {
    pub(super) source_leaf_index: u32,
    pub(super) dest_start: usize,
    pub(super) source_positions: &'a [u64],
    pub(super) token_ids: &'a [i32],
}

pub(super) fn embedded_verify_message(
    wire_dtype: WireActivationDType,
    args: VerifySpanMessageArgs<'_>,
) -> OpenAiResult<StageWireMessage> {
    if args.tokens.is_empty() {
        return Err(OpenAiError::backend(
            "verify span requires at least one token",
        ));
    }
    let mut state = StageStateHeader::new(WireMessageKind::VerifySpan, wire_dtype);
    state.seq_id = 0;
    state.prompt_token_count = i32::try_from(args.prompt_token_count)
        .map_err(|_| OpenAiError::backend("prompt token count exceeds i32"))?;
    state.decode_step = i32::try_from(args.decode_step)
        .map_err(|_| OpenAiError::backend("decode step exceeds i32"))?;
    state.current_token = args.tokens[0];
    state.source_stage_index = -1;
    if !args.checkpoint {
        state.flags |= state_flags::SKIP_VERIFY_CHECKPOINT;
    }
    Ok(StageWireMessage {
        kind: WireMessageKind::VerifySpan,
        pos_start: i32::try_from(args.pos_start)
            .map_err(|_| OpenAiError::backend("verify span position exceeds i32"))?,
        token_count: i32::try_from(args.tokens.len())
            .map_err(|_| OpenAiError::backend("verify span exceeds i32"))?,
        state,
        request_id: args.request_id,
        session_id: args.session_id,
        sampling: args.sampling,
        chat_sampling_metadata: None,
        tokens: args.tokens.to_vec(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    })
}

pub(super) fn embedded_tree_verify_message(
    wire_dtype: WireActivationDType,
    args: TreeVerifyMessageArgs<'_>,
) -> OpenAiResult<StageWireMessage> {
    if args.tokens.is_empty() {
        return Err(OpenAiError::backend(
            "tree verify requires at least one token",
        ));
    }
    if args.parents.len() != args.tokens.len() || args.depths.len() != args.tokens.len() {
        return Err(OpenAiError::backend(
            "tree verify parents and depths must match token count",
        ));
    }
    let mut message = embedded_verify_message(
        wire_dtype,
        VerifySpanMessageArgs {
            request_id: args.request_id,
            session_id: args.session_id,
            prompt_token_count: args.prompt_token_count,
            pos_start: args.pos_start,
            decode_step: args.decode_step,
            tokens: args.tokens,
            sampling: args.sampling,
            checkpoint: args.checkpoint,
        },
    )?;
    message.state.flags |= state_flags::TREE_VERIFY;
    if let Some(gather) = args.gather {
        if gather.source_positions.is_empty()
            || gather.source_positions.len() != gather.token_ids.len()
        {
            return Err(OpenAiError::backend(
                "tree gather source positions must match token count",
            ));
        }
        message.state.flags |= state_flags::TREE_GATHER;
        message.tokens.extend_from_slice(gather.token_ids);
    }
    message.positions.reserve(args.tokens.len() * 2);
    message.positions.extend_from_slice(args.parents);
    for depth in args.depths {
        message.positions.push(
            i32::try_from(*depth).map_err(|_| OpenAiError::backend("tree depth exceeds i32"))?,
        );
    }
    if let Some(gather) = args.gather {
        message.positions.push(
            i32::try_from(gather.source_leaf_index)
                .map_err(|_| OpenAiError::backend("tree gather leaf index exceeds i32"))?,
        );
        message.positions.push(
            i32::try_from(gather.dest_start)
                .map_err(|_| OpenAiError::backend("tree gather destination exceeds i32"))?,
        );
        message.positions.push(
            i32::try_from(gather.source_positions.len())
                .map_err(|_| OpenAiError::backend("tree gather token count exceeds i32"))?,
        );
        for position in gather.source_positions {
            message
                .positions
                .push(i32::try_from(*position).map_err(|_| {
                    OpenAiError::backend("tree gather source position exceeds i32")
                })?);
        }
    }
    Ok(message)
}

pub(super) fn embedded_session_control_message(
    wire_dtype: WireActivationDType,
    kind: WireMessageKind,
    request_id: u64,
    session_id: u64,
) -> StageWireMessage {
    StageWireMessage {
        kind,
        pos_start: 0,
        token_count: 0,
        state: StageStateHeader::new(kind, wire_dtype),
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    }
}

pub(super) fn embedded_trim_session_message(
    wire_dtype: WireActivationDType,
    request_id: u64,
    session_id: u64,
    token_count: usize,
) -> OpenAiResult<StageWireMessage> {
    let mut message = embedded_session_control_message(
        wire_dtype,
        WireMessageKind::TrimSession,
        request_id,
        session_id,
    );
    message.token_count = i32::try_from(token_count)
        .map_err(|_| OpenAiError::backend("trim token count exceeds i32"))?;
    Ok(message)
}

pub(super) fn embedded_gather_tree_path_message(
    wire_dtype: WireActivationDType,
    request_id: u64,
    session_id: u64,
    source_leaf_index: u32,
    dest_start: usize,
    source_positions: &[u64],
    token_ids: &[i32],
) -> OpenAiResult<StageWireMessage> {
    if source_positions.is_empty() || source_positions.len() != token_ids.len() {
        return Err(OpenAiError::backend(
            "tree gather source positions must match token count",
        ));
    }
    let mut message = embedded_session_control_message(
        wire_dtype,
        WireMessageKind::GatherTreePath,
        request_id,
        session_id,
    );
    message.pos_start = i32::try_from(dest_start)
        .map_err(|_| OpenAiError::backend("tree gather destination exceeds i32"))?;
    message.token_count = i32::try_from(token_ids.len())
        .map_err(|_| OpenAiError::backend("tree gather token count exceeds i32"))?;
    message.state.current_token = i32::try_from(source_leaf_index)
        .map_err(|_| OpenAiError::backend("tree gather leaf index exceeds i32"))?;
    message.tokens = token_ids.to_vec();
    message.positions = source_positions
        .iter()
        .copied()
        .map(|position| {
            i32::try_from(position)
                .map_err(|_| OpenAiError::backend("tree gather source position exceeds i32"))
        })
        .collect::<OpenAiResult<Vec<_>>>()?;
    Ok(message)
}

pub(super) fn generation_config_message(
    wire_dtype: WireActivationDType,
    request_id: u64,
    session_id: u64,
    prompt_token_count: usize,
    sampling: Option<WireSamplingConfig>,
    chat_sampling_metadata: Option<&str>,
) -> OpenAiResult<StageWireMessage> {
    let prompt_token_count = i32::try_from(prompt_token_count)
        .map_err(|_| OpenAiError::backend("prompt token count exceeds i32"))?;
    Ok(StageWireMessage::configure_generation(
        wire_dtype,
        request_id,
        session_id,
        prompt_token_count,
        sampling,
        chat_sampling_metadata.map(str::to_string),
    ))
}

pub(super) struct OpenAiPrefillChunk<'a> {
    pub(super) seq_id: usize,
    pub(super) pos_start: usize,
    pub(super) prefill_token_count: usize,
    pub(super) tokens: &'a [i32],
    pub(super) request_id: u64,
    pub(super) session_id: u64,
}

pub(super) fn embedded_prefill_message(
    wire_dtype: WireActivationDType,
    chunk: OpenAiPrefillChunk<'_>,
) -> OpenAiResult<StageWireMessage> {
    let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd, wire_dtype);
    state.seq_id =
        i32::try_from(chunk.seq_id).map_err(|_| OpenAiError::backend("prefill seq exceeds i32"))?;
    state.prompt_token_count = i32::try_from(chunk.prefill_token_count)
        .map_err(|_| OpenAiError::backend("prefill token count exceeds i32"))?;
    state.current_token = *chunk
        .tokens
        .last()
        .ok_or_else(|| OpenAiError::backend("prefill chunk is empty"))?;
    state.source_stage_index = -1;
    Ok(StageWireMessage {
        kind: WireMessageKind::PrefillEmbd,
        pos_start: i32::try_from(chunk.pos_start)
            .map_err(|_| OpenAiError::backend("prefill chunk position exceeds i32"))?,
        token_count: i32::try_from(chunk.tokens.len())
            .map_err(|_| OpenAiError::backend("prefill token count exceeds i32"))?,
        state,
        request_id: chunk.request_id,
        session_id: chunk.session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: chunk.tokens.to_vec(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    })
}

pub(super) fn embedded_prefix_cache_message(
    kind: WireMessageKind,
    wire_dtype: WireActivationDType,
    tokens: &[i32],
    request_id: u64,
    session_id: u64,
) -> OpenAiResult<StageWireMessage> {
    let mut state = StageStateHeader::new(kind, wire_dtype);
    state.prompt_token_count = i32::try_from(tokens.len())
        .map_err(|_| OpenAiError::backend("prefix token count exceeds i32"))?;
    state.current_token = tokens.last().copied().unwrap_or(LLAMA_TOKEN_NULL);
    state.source_stage_index = -1;
    Ok(StageWireMessage {
        kind,
        pos_start: 0,
        token_count: i32::try_from(tokens.len())
            .map_err(|_| OpenAiError::backend("prefix token count exceeds i32"))?,
        state,
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: tokens.to_vec(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    })
}

pub(super) struct RestorePrefillDecodeMessageArgs<'a> {
    pub(super) request_id: u64,
    pub(super) session_id: u64,
    pub(super) prompt_token_count: usize,
    pub(super) pos_start: usize,
    pub(super) decode_step: usize,
    pub(super) prefix_tokens: &'a [i32],
    pub(super) current: i32,
    pub(super) sampling: Option<WireSamplingConfig>,
    pub(super) chat_sampling_metadata: Option<&'a str>,
}

pub(super) fn embedded_restore_prefill_decode_message(
    wire_dtype: WireActivationDType,
    args: RestorePrefillDecodeMessageArgs<'_>,
) -> OpenAiResult<StageWireMessage> {
    let mut state = StageStateHeader::new(WireMessageKind::TryRestorePrefillDecode, wire_dtype);
    state.seq_id = 0;
    state.prompt_token_count = i32::try_from(args.prompt_token_count)
        .map_err(|_| OpenAiError::backend("prompt token count exceeds i32"))?;
    state.decode_step = i32::try_from(args.decode_step)
        .map_err(|_| OpenAiError::backend("decode step exceeds i32"))?;
    state.current_token = args.current;
    state.source_stage_index = -1;
    let mut tokens = Vec::with_capacity(args.prefix_tokens.len().saturating_add(1));
    tokens.extend_from_slice(args.prefix_tokens);
    tokens.push(args.current);
    Ok(StageWireMessage {
        kind: WireMessageKind::TryRestorePrefillDecode,
        pos_start: i32::try_from(args.pos_start)
            .map_err(|_| OpenAiError::backend("decode position exceeds i32"))?,
        token_count: 1,
        state,
        request_id: args.request_id,
        session_id: args.session_id,
        sampling: args.sampling,
        chat_sampling_metadata: args.chat_sampling_metadata.map(str::to_string),
        tokens,
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    })
}

pub(super) fn openai_stage_mask(stage_index: u32) -> i64 {
    if stage_index < 63 {
        1_i64 << stage_index
    } else {
        0
    }
}

pub(super) struct MultimodalPrefillArgs {
    pub(super) request_id: u64,
    pub(super) session_id: u64,
    pub(super) prompt_token_count: usize,
    pub(super) pos_start: usize,
    pub(super) token_count: usize,
    pub(super) positions: Vec<i32>,
    pub(super) sampling: Option<WireSamplingConfig>,
    pub(super) final_chunk: bool,
}

pub(super) fn multimodal_prefill_message(
    wire_dtype: WireActivationDType,
    args: MultimodalPrefillArgs,
) -> OpenAiResult<StageWireMessage> {
    let kind = if args.final_chunk {
        WireMessageKind::PrefillFinalEmbd
    } else {
        WireMessageKind::PrefillEmbd
    };
    let mut state = StageStateHeader::new(kind, wire_dtype);
    state.seq_id = 0;
    state.prompt_token_count = i32::try_from(args.prompt_token_count)
        .map_err(|_| OpenAiError::backend("multimodal prefill token count exceeds i32"))?;
    state.current_token = LLAMA_TOKEN_NULL;
    state.source_stage_index = -1;
    Ok(StageWireMessage {
        kind,
        pos_start: i32::try_from(args.pos_start)
            .map_err(|_| OpenAiError::backend("multimodal prefill position exceeds i32"))?,
        token_count: i32::try_from(args.token_count)
            .map_err(|_| OpenAiError::backend("multimodal prefill token count exceeds i32"))?,
        state,
        request_id: args.request_id,
        session_id: args.session_id,
        sampling: args.sampling,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        positions: args.positions,
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    })
}
