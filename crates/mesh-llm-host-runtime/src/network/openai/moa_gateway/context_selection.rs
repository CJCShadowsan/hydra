use crate::mesh;
use std::cmp::Reverse;
use std::collections::HashMap;

pub(in crate::network::openai) fn context_can_satisfy(
    required_tokens: Option<u32>,
    context_length: Option<u32>,
) -> bool {
    match (required_tokens, context_length) {
        (Some(required), Some(context)) => context >= required,
        _ => true,
    }
}

pub(in crate::network::openai) async fn select_remote_hosts_with_latencies(
    node: &mesh::Node,
    model: &str,
    required_tokens: Option<u32>,
    hosts: Vec<iroh::EndpointId>,
    latencies: &HashMap<iroh::EndpointId, u32>,
) -> Vec<iroh::EndpointId> {
    let mut adequate = Vec::new();
    let mut unknown = Vec::new();
    for host in hosts {
        match node.peer_model_context_length(host, model).await {
            Some(context) => {
                if required_tokens
                    .map(|required| context >= required)
                    .unwrap_or(true)
                {
                    adequate.push((host, context, latencies.get(&host).copied()));
                } else if let Some(required_tokens) = required_tokens {
                    tracing::info!(
                        "MoA: skipping remote worker {model} on {}; context {context} cannot fit {required_tokens} required tokens",
                        host.fmt_short()
                    );
                }
            }
            None => {
                unknown.push((host, latencies.get(&host).copied()));
            }
        }
    }
    adequate.sort_by_key(|candidate| (Reverse(candidate.1), candidate.2.unwrap_or(u32::MAX)));
    unknown.sort_by_key(|candidate| candidate.1.unwrap_or(u32::MAX));
    adequate
        .into_iter()
        .map(|(host, _, _)| host)
        .chain(unknown.into_iter().map(|(host, _)| host))
        .collect()
}

pub(in crate::network::openai) async fn best_remote_context_length(
    node: &mesh::Node,
    model: &str,
    hosts: &[iroh::EndpointId],
) -> Option<u32> {
    let mut best = None;
    for host in hosts {
        if let Some(context) = node.peer_model_context_length(*host, model).await {
            best = Some(best.map_or(context, |current: u32| current.max(context)));
        }
    }
    best
}

pub(in crate::network::openai) fn best_remote_latency_ms_from(
    latencies: &HashMap<iroh::EndpointId, u32>,
    hosts: &[iroh::EndpointId],
) -> Option<u32> {
    hosts
        .iter()
        .filter_map(|host| latencies.get(host).copied())
        .min()
}

pub(in crate::network::openai) async fn remote_latency_map(
    node: &mesh::Node,
) -> HashMap<iroh::EndpointId, u32> {
    node.peers()
        .await
        .into_iter()
        .filter_map(|peer| {
            let latency = peer.display_latency().latency_ms?;
            Some((peer.id, latency))
        })
        .collect()
}

pub(in crate::network::openai) fn virtual_mesh_context_length(
    models: &[String],
    runtimes: &[mesh::ModelRuntimeDescriptor],
) -> Option<u32> {
    let mut contexts_by_model = Vec::new();
    for model in models {
        if model == mesh_mixture_of_agents::VIRTUAL_MODEL_NAME {
            continue;
        }
        let context = runtimes
            .iter()
            .filter(|runtime| runtime.model_name == *model)
            .filter_map(mesh::ModelRuntimeDescriptor::advertised_context_length)
            .max();
        if let Some(context) = context {
            contexts_by_model.push(context);
        }
    }
    contexts_by_model.sort_unstable_by(|left, right| right.cmp(left));
    contexts_by_model.get(1).copied()
}

pub(in crate::network::openai) fn should_advertise_virtual_mesh(models: &[String]) -> bool {
    models
        .iter()
        .filter(|model| model.as_str() != mesh_mixture_of_agents::VIRTUAL_MODEL_NAME)
        .take(2)
        .count()
        >= 2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime(model_name: &str, context_length: Option<u32>) -> mesh::ModelRuntimeDescriptor {
        mesh::ModelRuntimeDescriptor {
            model_name: model_name.to_string(),
            identity_hash: None,
            context_length,
            ready: true,
        }
    }

    #[test]
    fn context_can_satisfy_keeps_unknown_as_fallback() {
        assert!(context_can_satisfy(Some(16_384), None));
        assert!(context_can_satisfy(None, Some(4096)));
        assert!(context_can_satisfy(Some(16_384), Some(32_768)));
        assert!(!context_can_satisfy(Some(16_384), Some(4096)));
    }

    #[test]
    fn virtual_mesh_context_is_minimum_when_only_two_known_contributors_fit() {
        let models = vec![
            "small".to_string(),
            "large".to_string(),
            mesh_mixture_of_agents::VIRTUAL_MODEL_NAME.to_string(),
        ];
        let runtimes = vec![runtime("small", Some(8192)), runtime("large", Some(65_536))];
        assert_eq!(virtual_mesh_context_length(&models, &runtimes), Some(8192));
    }

    #[test]
    fn virtual_mesh_context_uses_second_highest_known_model_context() {
        let models = vec![
            "small".to_string(),
            "large-a".to_string(),
            "large-b".to_string(),
        ];
        let runtimes = vec![
            runtime("small", Some(32_768)),
            runtime("large-a", Some(131_072)),
            runtime("large-b", Some(131_072)),
        ];
        assert_eq!(
            virtual_mesh_context_length(&models, &runtimes),
            Some(131_072)
        );
    }

    #[test]
    fn virtual_mesh_context_counts_each_model_once() {
        let models = vec!["small".to_string(), "large".to_string()];
        let runtimes = vec![
            runtime("large", Some(131_072)),
            runtime("large", Some(131_072)),
            runtime("small", Some(16_384)),
        ];
        assert_eq!(
            virtual_mesh_context_length(&models, &runtimes),
            Some(16_384)
        );
    }

    #[test]
    fn virtual_mesh_context_needs_two_known_contributor_contexts() {
        let models = vec!["unknown".to_string(), "known".to_string()];
        let runtimes = vec![runtime("unknown", None), runtime("known", Some(32_768))];
        assert_eq!(virtual_mesh_context_length(&models, &runtimes), None);
    }

    #[test]
    fn virtual_mesh_requires_two_concrete_models() {
        assert!(!should_advertise_virtual_mesh(&["only".to_string()]));
        assert!(should_advertise_virtual_mesh(&[
            "a".to_string(),
            "b".to_string(),
        ]));
    }
}
