use std::time::Instant;

use anyhow::{Context, Result, bail};
use skippy_protocol::{
    StageConfig,
    binary::{StageWireMessage, WireActivationDType, activation_state_flags_from_frame_flags},
};
use skippy_runtime::{ActivationFrame, RuntimeActivationDType};

pub(crate) fn forwarded_stage_message(
    config: &StageConfig,
    incoming: &StageWireMessage,
    output: &ActivationFrame,
    wire_dtype: WireActivationDType,
    activation_width: i32,
) -> Result<StageWireMessage> {
    Ok(
        forwarded_stage_message_timed(config, incoming, output, wire_dtype, activation_width)?
            .message,
    )
}

pub(crate) struct ForwardedStageMessage {
    pub message: StageWireMessage,
    pub activation_encode_ms: f64,
    pub glm_dsa_top_k_sideband_bytes: usize,
    pub glm_dsa_top_k_sideband_count: usize,
}

pub(crate) fn forwarded_stage_message_timed(
    config: &StageConfig,
    incoming: &StageWireMessage,
    output: &ActivationFrame,
    wire_dtype: WireActivationDType,
    activation_width: i32,
) -> Result<ForwardedStageMessage> {
    let mut state = incoming.state;
    state.source_stage_index = config.stage_index as i32;
    state.reserved = wire_dtype as i32;
    state.flags |= activation_state_flags_from_frame_flags(output.desc.flags);
    let encode_started = Instant::now();
    let (activation_payload, raw_bytes, sideband_stats) =
        split_activation_payload_sidebands(incoming, output, activation_width, state.flags)
            .with_context(|| {
                format!(
                    "split output activation payload; frame_dtype={:?} incoming_tokens={} output_tokens={} activation_width={} payload_bytes={} frame_payload_bytes={} state_flags={}",
                    output.desc.dtype,
                    incoming.token_count,
                    output.desc.token_count,
                    activation_width,
                    output.payload.len(),
                    output.desc.payload_bytes,
                    state.flags,
                )
            })?;
    if sideband_stats.glm_dsa_top_k_count > 0 {
        eprintln!(
            "skippy: glm_dsa_top_k_sideband_forward stage={} request={} session={} kind={:?} pos_start={} tokens={} hidden_bytes={} sideband_bytes={} sideband_i32={}",
            config.stage_id,
            incoming.request_id,
            incoming.session_id,
            incoming.kind,
            incoming.pos_start,
            incoming.token_count,
            sideband_stats.hidden_activation_bytes,
            sideband_stats.glm_dsa_top_k_bytes,
            sideband_stats.glm_dsa_top_k_count,
        );
    }
    let activation = encode_output_activation_payload(
        wire_dtype,
        incoming,
        output,
        activation_width,
        state.flags,
        activation_payload,
    )
    .with_context(|| {
        format!(
            "encode output activation payload; wire_dtype={wire_dtype:?} frame_dtype={:?} incoming_tokens={} output_tokens={} activation_width={} payload_bytes={} frame_payload_bytes={} state_flags={}",
            output.desc.dtype,
            incoming.token_count,
            output.desc.token_count,
            activation_width,
            output.payload.len(),
            output.desc.payload_bytes,
            state.flags,
        )
    })?;
    Ok(ForwardedStageMessage {
        message: StageWireMessage {
            kind: incoming.kind,
            pos_start: incoming.pos_start,
            token_count: incoming.token_count,
            state,
            request_id: incoming.request_id,
            session_id: incoming.session_id,
            sampling: incoming.sampling.clone(),
            chat_sampling_metadata: None,
            tokens: incoming.tokens.clone(),
            positions: incoming.positions.clone(),
            activation,
            raw_bytes,
        },
        activation_encode_ms: encode_started.elapsed().as_secs_f64() * 1000.0,
        glm_dsa_top_k_sideband_bytes: sideband_stats.glm_dsa_top_k_bytes,
        glm_dsa_top_k_sideband_count: sideband_stats.glm_dsa_top_k_count,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ActivationSidebandStats {
    hidden_activation_bytes: usize,
    glm_dsa_top_k_bytes: usize,
    glm_dsa_top_k_count: usize,
}

fn split_activation_payload_sidebands<'a>(
    incoming: &StageWireMessage,
    output: &'a ActivationFrame,
    activation_width: i32,
    state_flags: i32,
) -> Result<(&'a [u8], Vec<u8>, ActivationSidebandStats)> {
    if (state_flags & skippy_protocol::binary::state_flags::GLM_DSA_TOP_K_SIDEBAND) == 0 {
        return Ok((
            &output.payload,
            Vec::new(),
            ActivationSidebandStats {
                hidden_activation_bytes: output.payload.len(),
                ..ActivationSidebandStats::default()
            },
        ));
    }
    let hidden_bytes = skippy_protocol::binary::activation_wire_bytes_with_state_flags(
        WireActivationDType::F32,
        incoming.token_count,
        activation_width,
        state_flags & !skippy_protocol::binary::state_flags::GLM_DSA_TOP_K_SIDEBAND,
    )?;
    if output.payload.len() < hidden_bytes {
        bail!("GLM-DSA activation payload is smaller than hidden activation payload");
    }
    if (output.payload.len() - hidden_bytes) & 3 != 0 {
        bail!("GLM-DSA top-k sideband payload is not i32-aligned");
    }
    let sideband_bytes = output.payload.len() - hidden_bytes;
    Ok((
        &output.payload[..hidden_bytes],
        output.payload[hidden_bytes..].to_vec(),
        ActivationSidebandStats {
            hidden_activation_bytes: hidden_bytes,
            glm_dsa_top_k_bytes: sideband_bytes,
            glm_dsa_top_k_count: sideband_bytes / std::mem::size_of::<i32>(),
        },
    ))
}

fn encode_output_activation_payload(
    wire_dtype: WireActivationDType,
    incoming: &StageWireMessage,
    output: &ActivationFrame,
    activation_width: i32,
    state_flags: i32,
    activation_payload: &[u8],
) -> Result<Vec<u8>> {
    match (output.desc.dtype, wire_dtype) {
        (RuntimeActivationDType::F32, _) => Ok(
            skippy_protocol::binary::encode_f32_activation_payload_with_state_flags(
                wire_dtype,
                incoming.token_count,
                activation_width,
                activation_payload,
                state_flags,
            )?,
        ),
        (RuntimeActivationDType::F16, WireActivationDType::F16) => {
            if (state_flags & skippy_protocol::binary::state_flags::GLM_DSA_TOP_K_SIDEBAND) != 0 {
                bail!("GLM-DSA top-k sideband does not support F16 passthrough payloads");
            }
            validate_f16_passthrough_payload(incoming, output, activation_width, state_flags)?;
            Ok(activation_payload.to_vec())
        }
        (dtype, wire_dtype) => {
            bail!("unsupported activation dtype conversion: {dtype:?} to {wire_dtype:?}")
        }
    }
}

fn validate_f16_passthrough_payload(
    incoming: &StageWireMessage,
    output: &ActivationFrame,
    activation_width: i32,
    state_flags: i32,
) -> Result<()> {
    if output.payload.len() as u64 != output.desc.payload_bytes {
        bail!("F16 activation payload length does not match frame descriptor");
    }
    let expected = skippy_protocol::binary::activation_wire_bytes_with_state_flags(
        WireActivationDType::F16,
        incoming.token_count,
        activation_width,
        state_flags,
    )
    .context("compute expected F16 activation payload size")?;
    if output.payload.len() != expected {
        bail!("F16 activation payload size mismatch");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::{
        FlashAttentionType, LoadMode, PeerConfig, StageDevice, StageKvCacheConfig,
        binary::{
            StageStateHeader, WireMessageKind, activation_frame_flags_from_state_flags, state_flags,
        },
    };
    use skippy_runtime::{ActivationDesc, RuntimeActivationDType, RuntimeActivationLayout};

    fn stage_config() -> StageConfig {
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
            stage_id: "stage-1".to_string(),
            stage_index: 1,
            layer_start: 4,
            layer_end: 8,
            ctx_size: 512,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            filter_tensors_on_load: true,
            selected_device: None::<StageDevice>,
            kv_cache: None::<StageKvCacheConfig>,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: Some(PeerConfig {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                endpoint: "tcp://127.0.0.1:19000".to_string(),
            }),
            downstream: None,
        }
    }

    fn incoming_message() -> StageWireMessage {
        StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 7,
            token_count: 1,
            state: StageStateHeader::new(WireMessageKind::DecodeEmbd, WireActivationDType::F32),
            request_id: 42,
            session_id: 99,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![11],
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    fn f32_frame(flags: u64, token_count: u32, values: &[f32]) -> ActivationFrame {
        let mut payload = Vec::new();
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        ActivationFrame {
            desc: ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F32,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: 1,
                layer_start: 4,
                layer_end: 8,
                token_count,
                sequence_count: 1,
                payload_bytes: payload.len() as u64,
                flags,
            },
            payload,
        }
    }

    fn rwkv7_sideband_frame() -> ActivationFrame {
        f32_frame(
            skippy_protocol::binary::ACTIVATION_FLAG_RWKV7_V_FIRST,
            1,
            &[1.0_f32, 2.0, 3.0, 4.0],
        )
    }

    fn glm_dsa_top_k_sideband_frame() -> ActivationFrame {
        let mut frame = f32_frame(
            skippy_protocol::binary::ACTIVATION_FLAG_GLM_DSA_TOP_K,
            1,
            &[1.0_f32, 2.0],
        );
        for value in [17_i32, 23] {
            frame.payload.extend_from_slice(&value.to_le_bytes());
        }
        frame.desc.payload_bytes = frame.payload.len() as u64;
        frame
    }

    fn f16_frame() -> ActivationFrame {
        ActivationFrame {
            desc: ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F16,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: 1,
                layer_start: 4,
                layer_end: 8,
                token_count: 2,
                sequence_count: 1,
                payload_bytes: 8,
                flags: 0,
            },
            payload: vec![0, 1, 2, 3, 4, 5, 6, 7],
        }
    }

    #[test]
    fn forwarded_stage_message_preserves_rwkv7_sideband_shape() {
        let forwarded = forwarded_stage_message_timed(
            &stage_config(),
            &incoming_message(),
            &rwkv7_sideband_frame(),
            WireActivationDType::F32,
            2,
        )
        .unwrap();

        assert_eq!(forwarded.message.activation.len(), 16);
        assert_ne!(
            forwarded.message.state.flags & state_flags::RWKV7_V_FIRST_SIDEBAND,
            0
        );
        assert_eq!(
            activation_frame_flags_from_state_flags(forwarded.message.state.flags),
            skippy_protocol::binary::ACTIVATION_FLAG_RWKV7_V_FIRST
        );
    }

    #[test]
    fn forwarded_stage_message_reencodes_rwkv7_sideband_for_wire_dtype() {
        let forwarded = forwarded_stage_message_timed(
            &stage_config(),
            &incoming_message(),
            &rwkv7_sideband_frame(),
            WireActivationDType::Q8,
            2,
        )
        .unwrap();

        assert_eq!(forwarded.message.activation.len(), 12);
        assert_ne!(
            forwarded.message.state.flags & state_flags::RWKV7_V_FIRST_SIDEBAND,
            0
        );
    }

    #[test]
    fn forwarded_stage_message_carries_glm_dsa_top_k_as_raw_sideband() {
        let forwarded = forwarded_stage_message_timed(
            &stage_config(),
            &incoming_message(),
            &glm_dsa_top_k_sideband_frame(),
            WireActivationDType::F32,
            2,
        )
        .unwrap();

        assert_eq!(forwarded.message.activation.len(), 8);
        assert_eq!(forwarded.glm_dsa_top_k_sideband_bytes, 8);
        assert_eq!(forwarded.glm_dsa_top_k_sideband_count, 2);
        assert_eq!(
            forwarded.message.raw_bytes,
            [17_i32.to_le_bytes(), 23_i32.to_le_bytes()].concat()
        );
        assert_ne!(
            forwarded.message.state.flags & state_flags::GLM_DSA_TOP_K_SIDEBAND,
            0
        );
        assert_eq!(
            activation_frame_flags_from_state_flags(forwarded.message.state.flags),
            skippy_protocol::binary::ACTIVATION_FLAG_GLM_DSA_TOP_K
        );
    }

    #[test]
    fn forwarded_stage_message_passes_through_f16_activation_for_f16_wire() {
        let mut incoming = incoming_message();
        incoming.token_count = 2;

        let forwarded = forwarded_stage_message_timed(
            &stage_config(),
            &incoming,
            &f16_frame(),
            WireActivationDType::F16,
            2,
        )
        .unwrap();

        assert_eq!(forwarded.message.activation, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(
            forwarded.message.state.reserved,
            WireActivationDType::F16 as i32
        );
    }

    #[test]
    fn forwarded_stage_message_rejects_bad_f16_passthrough_size() {
        let mut incoming = incoming_message();
        incoming.token_count = 2;
        let mut output = f16_frame();
        output.payload.pop();
        output.desc.payload_bytes = output.payload.len() as u64;

        let error = match forwarded_stage_message_timed(
            &stage_config(),
            &incoming,
            &output,
            WireActivationDType::F16,
            2,
        ) {
            Ok(_) => panic!("expected bad F16 passthrough payload to fail"),
            Err(error) => error,
        };

        assert!(format!("{error:#}").contains("F16 activation payload size mismatch"));
    }
}
