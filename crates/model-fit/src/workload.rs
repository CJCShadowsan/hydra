use crate::{
    CapabilityRequirements, InteractionProfile, Requirement, ScoreWeights, WorkloadPreferences,
    WorkloadProfile, WorkloadTask,
};

impl WorkloadProfile {
    pub fn chat() -> Self {
        Self {
            task: WorkloadTask::Chat,
            interaction: InteractionProfile {
                expected_prompt_tokens: Some(4_096),
                expected_output_tokens: Some(512),
                latency_sensitive: true,
                multi_turn: true,
                agent_loop: false,
            },
            requirements: CapabilityRequirements {
                chat_template: Requirement::Preferred,
                system_messages: Requirement::Preferred,
                embeddings: Requirement::Reject,
                reranking: Requirement::Reject,
                min_context_tokens: Some(8_192),
                ..CapabilityRequirements::default()
            },
            preferences: WorkloadPreferences {
                minimum_decode_tps: Some(5.0),
                preferred_decode_tps: Some(20.0),
                ..WorkloadPreferences::default()
            },
        }
    }

    pub fn coding_agent() -> Self {
        Self {
            task: WorkloadTask::Coding,
            interaction: InteractionProfile {
                expected_prompt_tokens: Some(16_384),
                expected_output_tokens: Some(1_024),
                latency_sensitive: true,
                multi_turn: true,
                agent_loop: true,
            },
            requirements: CapabilityRequirements {
                chat_template: Requirement::Preferred,
                system_messages: Requirement::Preferred,
                tool_calling: Requirement::Preferred,
                fill_in_middle: Requirement::Preferred,
                embeddings: Requirement::Reject,
                reranking: Requirement::Reject,
                min_context_tokens: Some(32_768),
                ..CapabilityRequirements::default()
            },
            preferences: WorkloadPreferences {
                prefer_quality_over_speed: 0.55,
                prefer_context_over_speed: 0.75,
                minimum_decode_tps: Some(8.0),
                preferred_decode_tps: Some(25.0),
            },
        }
    }

    pub fn tool_calling() -> Self {
        Self {
            task: WorkloadTask::ToolCalling,
            interaction: InteractionProfile {
                expected_prompt_tokens: Some(8_192),
                expected_output_tokens: Some(512),
                latency_sensitive: true,
                multi_turn: true,
                agent_loop: true,
            },
            requirements: CapabilityRequirements {
                chat_template: Requirement::Required,
                system_messages: Requirement::Preferred,
                tool_calling: Requirement::Required,
                embeddings: Requirement::Reject,
                reranking: Requirement::Reject,
                min_context_tokens: Some(8_192),
                ..CapabilityRequirements::default()
            },
            preferences: WorkloadPreferences {
                minimum_decode_tps: Some(8.0),
                preferred_decode_tps: Some(25.0),
                ..WorkloadPreferences::default()
            },
        }
    }

    pub fn summarization() -> Self {
        Self {
            task: WorkloadTask::Summarization,
            interaction: InteractionProfile {
                expected_prompt_tokens: Some(24_576),
                expected_output_tokens: Some(768),
                latency_sensitive: false,
                multi_turn: false,
                agent_loop: false,
            },
            requirements: CapabilityRequirements {
                chat_template: Requirement::Preferred,
                embeddings: Requirement::Reject,
                reranking: Requirement::Reject,
                min_context_tokens: Some(32_768),
                ..CapabilityRequirements::default()
            },
            preferences: WorkloadPreferences {
                prefer_quality_over_speed: 0.70,
                prefer_context_over_speed: 0.90,
                minimum_decode_tps: Some(2.0),
                preferred_decode_tps: Some(12.0),
            },
        }
    }

    pub fn embedding() -> Self {
        Self {
            task: WorkloadTask::Embedding,
            interaction: InteractionProfile {
                expected_prompt_tokens: Some(512),
                expected_output_tokens: Some(0),
                latency_sensitive: true,
                multi_turn: false,
                agent_loop: false,
            },
            requirements: CapabilityRequirements {
                embeddings: Requirement::Required,
                min_context_tokens: Some(512),
                ..CapabilityRequirements::default()
            },
            preferences: WorkloadPreferences {
                minimum_decode_tps: None,
                preferred_decode_tps: None,
                ..WorkloadPreferences::default()
            },
        }
    }

    pub fn reranking() -> Self {
        Self {
            task: WorkloadTask::Reranking,
            interaction: InteractionProfile {
                expected_prompt_tokens: Some(1_024),
                expected_output_tokens: Some(0),
                latency_sensitive: true,
                multi_turn: false,
                agent_loop: false,
            },
            requirements: CapabilityRequirements {
                reranking: Requirement::Required,
                min_context_tokens: Some(512),
                ..CapabilityRequirements::default()
            },
            preferences: WorkloadPreferences {
                minimum_decode_tps: None,
                preferred_decode_tps: None,
                ..WorkloadPreferences::default()
            },
        }
    }

    pub fn vision_chat() -> Self {
        Self {
            task: WorkloadTask::MultimodalUnderstanding,
            interaction: InteractionProfile {
                expected_prompt_tokens: Some(4_096),
                expected_output_tokens: Some(512),
                latency_sensitive: true,
                multi_turn: true,
                agent_loop: false,
            },
            requirements: CapabilityRequirements {
                chat_template: Requirement::Preferred,
                vision: Requirement::Required,
                embeddings: Requirement::Reject,
                reranking: Requirement::Reject,
                min_context_tokens: Some(8_192),
                ..CapabilityRequirements::default()
            },
            preferences: WorkloadPreferences::default(),
        }
    }

    pub fn general_generation() -> Self {
        Self {
            task: WorkloadTask::GeneralGeneration,
            interaction: InteractionProfile {
                expected_prompt_tokens: Some(4_096),
                expected_output_tokens: Some(512),
                latency_sensitive: false,
                multi_turn: false,
                agent_loop: false,
            },
            requirements: CapabilityRequirements {
                embeddings: Requirement::Reject,
                reranking: Requirement::Reject,
                min_context_tokens: Some(4_096),
                ..CapabilityRequirements::default()
            },
            preferences: WorkloadPreferences::default(),
        }
    }

    pub fn default_weights(&self) -> ScoreWeights {
        match self.task {
            WorkloadTask::Embedding | WorkloadTask::Reranking | WorkloadTask::Classification => {
                ScoreWeights {
                    memory: 0.25,
                    context: 0.15,
                    decode: 0.0,
                    prefill: 0.35,
                    workload: 0.25,
                }
            }
            WorkloadTask::Summarization => ScoreWeights {
                memory: 0.20,
                context: 0.30,
                decode: 0.10,
                prefill: 0.20,
                workload: 0.20,
            },
            WorkloadTask::Coding | WorkloadTask::ToolCalling => ScoreWeights {
                memory: 0.20,
                context: 0.25,
                decode: 0.25,
                prefill: 0.10,
                workload: 0.20,
            },
            _ => ScoreWeights::default(),
        }
    }
}
