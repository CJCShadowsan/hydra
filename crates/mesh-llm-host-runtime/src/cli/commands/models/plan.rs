use crate::models::remote_catalog::{self, CatalogEntry};
use crate::runtime;
use crate::system::{
    benchmark,
    hardware::{self, GpuFacts, HardwareSurvey},
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

mod disk;
mod local_hardware;
mod render;
use disk::{DiskPlan, build_disk_plan};
use local_hardware::local_gpu_capacity_fallback;
use render::{render_capacity_plan, render_recommendations};

const TIGHT_FIT_THRESHOLD: f64 = 1.15;
const UNIFIED_MEMORY_RESERVE_MIN_BYTES: u64 = 6_000_000_000;
const UNIFIED_MEMORY_RESERVE_PERCENT: u64 = 20;
const DISCRETE_VRAM_RESERVE_MIN_BYTES: u64 = 2_000_000_000;
const DISCRETE_VRAM_RESERVE_PERCENT: u64 = 10;
const UNKNOWN_MEMORY_RESERVE_MIN_BYTES: u64 = 4_000_000_000;
const UNKNOWN_MEMORY_RESERVE_PERCENT: u64 = 15;

#[derive(Clone, Debug, Eq, PartialEq)]
struct CatalogModelHint {
    catalog_id: String,
    display_name: String,
    model_ref: String,
    source_repo: String,
    source_file: String,
    model_bytes: u64,
    split_capable: bool,
    layer_package_repo: Option<String>,
    layer_count: Option<u32>,
    package_total_bytes: Option<u64>,
    is_moe: bool,
    moe_summary: Option<String>,
    active_parameter_billions: Option<OrderedF64>,
}

#[derive(Clone, Debug, Serialize)]
struct PlanNode {
    id: String,
    role: String,
    hostname: Option<String>,
    raw_capacity_bytes: u64,
    reserve_bytes: u64,
    capacity_bytes: u64,
    unified_memory: Option<bool>,
    bandwidth_gbps: Option<f64>,
    rtt_ms: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
struct CapacityPlan {
    model: ModelPlanSummary,
    local: CapacityScopePlan,
    #[serde(skip_serializing_if = "Option::is_none")]
    mesh: Option<CapacityScopePlan>,
    disk: DiskPlan,
    notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct RecommendationReport {
    source: &'static str,
    basis: &'static str,
    results: Vec<RecommendationRow>,
    notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct RecommendationRow {
    display_name: String,
    model_ref: String,
    fit: FitVerdict,
    reason: &'static str,
    model_bytes: u64,
    weight_budget_bytes: u64,
    kv_cache_budget_bytes: u64,
    required_bytes: u64,
    context_tokens: u32,
    moe: bool,
    active_parameter_billions: Option<f64>,
    best_single_node_capacity_bytes: Option<u64>,
    aggregate_capacity_bytes: u64,
    split_capable: bool,
    layer_package_repo: Option<String>,
    disk_fits_full_model: Option<bool>,
    disk_full_model_required_bytes: u64,
    disk_free_bytes: Option<u64>,
    plan_command: String,
}

#[derive(Clone, Debug, Serialize)]
struct ModelPlanSummary {
    requested: String,
    display_name: String,
    model_ref: String,
    source_repo: String,
    source_file: String,
    model_bytes: u64,
    weight_budget_bytes: u64,
    kv_cache_budget_bytes: u64,
    required_bytes: u64,
    context_tokens: u32,
    moe: bool,
    moe_summary: Option<String>,
    active_parameter_billions: Option<f64>,
    split_capable: bool,
    layer_package_repo: Option<String>,
    layer_count: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
struct CapacityScopePlan {
    label: &'static str,
    fit: FitVerdict,
    reason: &'static str,
    best_single_node_capacity_bytes: Option<u64>,
    aggregate_capacity_bytes: u64,
    eligible_node_count: usize,
    excluded_client_node_count: usize,
    missing_capacity_node_count: usize,
    nodes_needed_like_local: Option<usize>,
    split_node_count: Option<usize>,
    suggested_nodes: Vec<PlanNode>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum FitVerdict {
    ComfortableLocal,
    TightLocal,
    SplitCandidate,
    InsufficientCapacity,
    UnknownCapacity,
    NoEligibleHosts,
}

#[derive(Clone, Copy, Debug)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl PartialEq for OrderedF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0).is_eq()
    }
}

#[derive(Clone, Debug, Deserialize)]
struct StatusPayload {
    #[serde(default)]
    node_id: String,
    #[serde(default)]
    is_client: bool,
    #[serde(default)]
    my_hostname: Option<String>,
    #[serde(default)]
    my_is_soc: Option<bool>,
    #[serde(default)]
    my_vram_gb: f64,
    #[serde(default)]
    gpus: Vec<StatusGpu>,
    #[serde(default)]
    peers: Vec<StatusPeer>,
}

#[derive(Clone, Debug, Deserialize)]
struct StatusPeer {
    #[serde(default)]
    id: String,
    #[serde(default)]
    role: String,
    #[serde(default)]
    hostname: Option<String>,
    #[serde(default)]
    is_soc: Option<bool>,
    #[serde(default)]
    vram_gb: f64,
    #[serde(default)]
    rtt_ms: Option<u32>,
    #[serde(default)]
    gpus: Vec<StatusGpu>,
}

#[derive(Clone, Debug, Deserialize)]
struct StatusGpu {
    #[serde(default)]
    mem_bandwidth_gbps: Option<f64>,
    #[serde(default)]
    unified_memory: Option<bool>,
}

pub(crate) async fn run_model_plan(
    model: &str,
    api_base: Option<&str>,
    context_tokens: u32,
    json_output: bool,
) -> Result<()> {
    remote_catalog::ensure_catalog()?;
    let hint = find_catalog_model_hint(model)
        .with_context(|| format!("remote catalog model not found for {model:?}"))?;
    let local_nodes = local_capacity_nodes();
    let mesh_status = match api_base {
        Some(base) => Some(fetch_status(base).await?),
        None => None,
    };

    let plan = build_capacity_plan(model, hint, local_nodes, mesh_status, context_tokens);
    if json_output {
        println!("{}", serde_json::to_string_pretty(&plan)?);
    } else {
        render_capacity_plan(&plan);
    }
    Ok(())
}

pub(crate) async fn run_model_recommend(
    api_base: Option<&str>,
    context_tokens: u32,
    limit: usize,
    json_output: bool,
) -> Result<()> {
    remote_catalog::ensure_catalog()?;
    let local_nodes = local_capacity_nodes();
    let mesh_status = match api_base {
        Some(base) => Some(fetch_status(base).await?),
        None => None,
    };
    let basis = if mesh_status.is_some() {
        "mesh"
    } else {
        "local"
    };
    let mut rows = catalog_model_hints()
        .into_iter()
        .map(|hint| {
            recommendation_for_hint(
                hint,
                &local_nodes,
                mesh_status.clone(),
                api_base,
                context_tokens,
            )
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| recommendation_score(right).cmp(&recommendation_score(left)));
    rows.truncate(limit);

    let report = RecommendationReport {
        source: "catalog",
        basis,
        results: rows,
        notes: vec![
            "Recommendations estimate weight bytes, KV cache for the requested context, and local memory reserve; they do not download or run models."
                .to_string(),
            "Use `mesh-llm models plan <MODEL>` for the full capacity explanation.".to_string(),
        ],
    };
    if json_output {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        render_recommendations(&report);
    }
    Ok(())
}

fn recommendation_for_hint(
    hint: CatalogModelHint,
    local_nodes: &[PlanNode],
    mesh_status: Option<StatusPayload>,
    api_base: Option<&str>,
    context_tokens: u32,
) -> RecommendationRow {
    let plan = build_capacity_plan(
        &hint.catalog_id.clone(),
        hint,
        local_nodes.to_vec(),
        mesh_status,
        context_tokens,
    );
    let scope = plan.mesh.as_ref().unwrap_or(&plan.local);
    let plan_command = match api_base {
        Some(base) => format!(
            "mesh-llm models plan {} --api-base {}",
            plan.model.model_ref, base
        ),
        None => format!("mesh-llm models plan {}", plan.model.model_ref),
    };

    RecommendationRow {
        display_name: plan.model.display_name,
        model_ref: plan.model.model_ref,
        fit: scope.fit,
        reason: scope.reason,
        model_bytes: plan.model.model_bytes,
        weight_budget_bytes: plan.model.weight_budget_bytes,
        kv_cache_budget_bytes: plan.model.kv_cache_budget_bytes,
        required_bytes: plan.model.required_bytes,
        context_tokens: plan.model.context_tokens,
        moe: plan.model.moe,
        active_parameter_billions: plan.model.active_parameter_billions,
        best_single_node_capacity_bytes: scope.best_single_node_capacity_bytes,
        aggregate_capacity_bytes: scope.aggregate_capacity_bytes,
        split_capable: plan.model.split_capable,
        layer_package_repo: plan.model.layer_package_repo,
        disk_fits_full_model: plan.disk.fits_full_model,
        disk_full_model_required_bytes: plan.disk.full_model_required_bytes,
        disk_free_bytes: plan.disk.free_bytes,
        plan_command,
    }
}

fn recommendation_score(row: &RecommendationRow) -> u128 {
    let fit_score = match row.fit {
        FitVerdict::ComfortableLocal => 60,
        FitVerdict::TightLocal => 55,
        FitVerdict::SplitCandidate => 45,
        FitVerdict::UnknownCapacity => 20,
        FitVerdict::InsufficientCapacity => 5,
        FitVerdict::NoEligibleHosts => 0,
    } as u128;
    let size_signal = match row.fit {
        FitVerdict::ComfortableLocal | FitVerdict::TightLocal | FitVerdict::SplitCandidate => {
            u128::from(row.model_bytes)
        }
        FitVerdict::UnknownCapacity
        | FitVerdict::InsufficientCapacity
        | FitVerdict::NoEligibleHosts => {
            1_000_000_000_000_u128.saturating_sub(u128::from(row.model_bytes))
        }
    };
    let disk_penalty = if row.disk_fits_full_model == Some(false) {
        1_u128
    } else {
        0
    };
    fit_score.saturating_sub(disk_penalty) * 1_000_000_000_000_000 + size_signal
}

fn build_capacity_plan(
    requested: &str,
    hint: CatalogModelHint,
    local_nodes: Vec<PlanNode>,
    mesh_status: Option<StatusPayload>,
    context_tokens: u32,
) -> CapacityPlan {
    let weight_budget_bytes = runtime::runtime_model_required_bytes(hint.model_bytes);
    let kv_cache_budget_bytes = estimate_kv_cache_bytes(&hint, context_tokens);
    let required_bytes = weight_budget_bytes.saturating_add(kv_cache_budget_bytes);
    let local = plan_scope(
        "local",
        &local_nodes,
        required_bytes,
        hint.split_capable,
        true,
    );
    let mesh = mesh_status.map(|status| {
        let nodes = status_capacity_nodes(status);
        plan_scope("mesh", &nodes, required_bytes, hint.split_capable, false)
    });
    let disk = build_disk_plan(&hint, local.split_node_count);

    CapacityPlan {
        model: ModelPlanSummary {
            requested: requested.to_string(),
            display_name: hint.display_name,
            model_ref: hint.model_ref,
            source_repo: hint.source_repo,
            source_file: hint.source_file,
            model_bytes: hint.model_bytes,
            weight_budget_bytes,
            kv_cache_budget_bytes,
            required_bytes,
            context_tokens,
            moe: hint.is_moe,
            moe_summary: hint.moe_summary,
            active_parameter_billions: hint.active_parameter_billions.map(|value| value.0),
            split_capable: hint.split_capable,
            layer_package_repo: hint.layer_package_repo,
            layer_count: hint.layer_count,
        },
        local,
        mesh,
        disk,
        notes: vec![
            "This is a catalog estimate; it does not download or run the model.".to_string(),
            "Memory budget includes model-weight headroom, estimated KV cache for the requested context, and a local memory reserve for OS/apps or VRAM allocator slack.".to_string(),
            "Disk budget checks free space on the Hugging Face cache filesystem and includes download/materialization headroom.".to_string(),
            "MoE models are planned with full weight bytes resident; active experts only reduce the KV/throughput estimate, not the placement requirement.".to_string(),
            "The estimate still does not inspect tensor groups or run shaped GPU kernel probes.".to_string(),
        ],
    }
}

fn plan_scope(
    label: &'static str,
    nodes: &[PlanNode],
    required_bytes: u64,
    split_capable: bool,
    include_like_local: bool,
) -> CapacityScopePlan {
    let mut eligible = nodes
        .iter()
        .filter(|node| !node.role.eq_ignore_ascii_case("client"))
        .cloned()
        .collect::<Vec<_>>();
    eligible.sort_by(|left, right| right.capacity_bytes.cmp(&left.capacity_bytes));

    let excluded_client_node_count = nodes
        .iter()
        .filter(|node| node.role.eq_ignore_ascii_case("client"))
        .count();
    let missing_capacity_node_count = eligible
        .iter()
        .filter(|node| node.capacity_bytes == 0)
        .count();
    let eligible_with_capacity = eligible
        .iter()
        .filter(|node| node.capacity_bytes > 0)
        .cloned()
        .collect::<Vec<_>>();
    let aggregate_capacity_bytes = eligible_with_capacity
        .iter()
        .map(|node| node.capacity_bytes)
        .sum();
    let best_single_node_capacity_bytes = eligible_with_capacity
        .first()
        .map(|node| node.capacity_bytes);
    let nodes_needed_like_local = include_like_local
        .then(|| best_single_node_capacity_bytes)
        .flatten()
        .and_then(|capacity| nodes_needed_for_capacity(required_bytes, capacity));

    let (fit, reason, split_node_count, suggested_nodes) = scope_fit(
        &eligible_with_capacity,
        required_bytes,
        split_capable,
        best_single_node_capacity_bytes,
        aggregate_capacity_bytes,
        missing_capacity_node_count,
    );

    CapacityScopePlan {
        label,
        fit,
        reason,
        best_single_node_capacity_bytes,
        aggregate_capacity_bytes,
        eligible_node_count: eligible_with_capacity.len(),
        excluded_client_node_count,
        missing_capacity_node_count,
        nodes_needed_like_local,
        split_node_count,
        suggested_nodes,
    }
}

fn scope_fit(
    eligible: &[PlanNode],
    required_bytes: u64,
    split_capable: bool,
    best_single_node_capacity_bytes: Option<u64>,
    aggregate_capacity_bytes: u64,
    missing_capacity_node_count: usize,
) -> (FitVerdict, &'static str, Option<usize>, Vec<PlanNode>) {
    if missing_capacity_node_count > 0 {
        return (
            FitVerdict::UnknownCapacity,
            "eligible_nodes_missing_capacity",
            None,
            Vec::new(),
        );
    }
    if eligible.is_empty() {
        return (
            FitVerdict::NoEligibleHosts,
            "no_worker_or_host_capacity",
            None,
            Vec::new(),
        );
    }
    if let Some(best) = best_single_node_capacity_bytes {
        if best >= required_bytes {
            let verdict = if best as f64 / required_bytes as f64 <= TIGHT_FIT_THRESHOLD {
                FitVerdict::TightLocal
            } else {
                FitVerdict::ComfortableLocal
            };
            return (
                verdict,
                "single_node_capacity_available",
                Some(1),
                eligible.iter().take(1).cloned().collect(),
            );
        }
    }

    if split_capable && eligible.len() >= 2 && aggregate_capacity_bytes >= required_bytes {
        let suggested = minimal_node_set(eligible, required_bytes);
        return (
            FitVerdict::SplitCandidate,
            "aggregate_split_capacity_available",
            Some(suggested.len()),
            suggested,
        );
    }

    (
        FitVerdict::InsufficientCapacity,
        "capacity_shortfall",
        None,
        Vec::new(),
    )
}

fn minimal_node_set(nodes: &[PlanNode], required_bytes: u64) -> Vec<PlanNode> {
    let mut total = 0_u64;
    let mut selected = Vec::new();
    for node in nodes {
        total = total.saturating_add(node.capacity_bytes);
        selected.push(node.clone());
        if total >= required_bytes {
            break;
        }
    }
    selected
}

fn nodes_needed_for_capacity(required_bytes: u64, node_capacity_bytes: u64) -> Option<usize> {
    if node_capacity_bytes == 0 {
        return None;
    }
    Some(required_bytes.div_ceil(node_capacity_bytes).max(1) as usize)
}

fn local_capacity_nodes() -> Vec<PlanNode> {
    let mut survey = hardware::survey();
    attach_cached_bandwidth(&mut survey);
    if survey.gpus.is_empty() {
        return vec![plan_node(
            "local".to_string(),
            "Worker".to_string(),
            survey.hostname,
            survey.vram_bytes,
            Some(survey.is_soc),
            None,
            None,
        )];
    }

    survey.gpus.iter().map(local_gpu_node).collect()
}

fn local_gpu_node(gpu: &GpuFacts) -> PlanNode {
    let id = gpu
        .stable_id
        .as_deref()
        .filter(|stable_id| !stable_id.trim().is_empty())
        .map(ToString::to_string)
        .or_else(|| {
            gpu.backend_device
                .as_deref()
                .filter(|device| !device.trim().is_empty())
                .map(ToString::to_string)
        })
        .clone()
        .unwrap_or_else(|| format!("local-gpu-{}", gpu.index));
    let fallback_capacity = local_gpu_capacity_fallback(gpu);
    let raw_capacity_bytes = fallback_capacity.unwrap_or(gpu.vram_bytes);
    let unified_memory = Some(gpu.unified_memory || fallback_capacity.is_some());
    plan_node(
        id,
        "Worker".to_string(),
        None,
        raw_capacity_bytes,
        unified_memory,
        gpu.mem_bandwidth_gbps,
        None,
    )
}

fn plan_node(
    id: String,
    role: String,
    hostname: Option<String>,
    raw_capacity_bytes: u64,
    unified_memory: Option<bool>,
    bandwidth_gbps: Option<f64>,
    rtt_ms: Option<u32>,
) -> PlanNode {
    let reserve_bytes = local_memory_reserve_bytes(raw_capacity_bytes, unified_memory);
    PlanNode {
        id,
        role,
        hostname,
        raw_capacity_bytes,
        reserve_bytes,
        capacity_bytes: raw_capacity_bytes.saturating_sub(reserve_bytes),
        unified_memory,
        bandwidth_gbps,
        rtt_ms,
    }
}

fn local_memory_reserve_bytes(raw_capacity_bytes: u64, unified_memory: Option<bool>) -> u64 {
    if raw_capacity_bytes == 0 {
        return 0;
    }
    let (minimum, percent) = match unified_memory {
        Some(true) => (
            UNIFIED_MEMORY_RESERVE_MIN_BYTES,
            UNIFIED_MEMORY_RESERVE_PERCENT,
        ),
        Some(false) => (
            DISCRETE_VRAM_RESERVE_MIN_BYTES,
            DISCRETE_VRAM_RESERVE_PERCENT,
        ),
        None => (
            UNKNOWN_MEMORY_RESERVE_MIN_BYTES,
            UNKNOWN_MEMORY_RESERVE_PERCENT,
        ),
    };
    let percent_bytes = raw_capacity_bytes.saturating_mul(percent) / 100;
    minimum.max(percent_bytes).min(raw_capacity_bytes)
}

fn attach_cached_bandwidth(hw: &mut HardwareSurvey) {
    let path = benchmark::fingerprint_path();
    let Some(fingerprint) = benchmark::load_fingerprint(&path) else {
        return;
    };
    if benchmark::hardware_changed(&fingerprint, hw) {
        return;
    }

    for (gpu, cached) in hw.gpus.iter_mut().zip(fingerprint.gpus.iter()) {
        gpu.mem_bandwidth_gbps = Some(cached.p90_gbps);
    }
}

async fn fetch_status(api_base: &str) -> Result<StatusPayload> {
    let base = api_base.trim().trim_end_matches('/');
    if base.is_empty() {
        bail!("--api-base must not be empty");
    }
    let url = format!("{base}/api/status");
    let response = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()?
        .get(&url)
        .send()
        .await
        .with_context(|| format!("fetch {url}"))?;
    if !response.status().is_success() {
        bail!("fetch {url} failed with {}", response.status());
    }
    response.json().await.context("parse /api/status response")
}

fn status_capacity_nodes(status: StatusPayload) -> Vec<PlanNode> {
    let mut nodes = Vec::new();
    let local_unified = status
        .gpus
        .iter()
        .find_map(|gpu| gpu.unified_memory)
        .or(status.my_is_soc);
    nodes.push(plan_node(
        status.node_id,
        if status.is_client {
            "Client".to_string()
        } else {
            "Worker".to_string()
        },
        status.my_hostname,
        gb_to_bytes(status.my_vram_gb),
        local_unified,
        best_status_bandwidth(&status.gpus),
        None,
    ));
    nodes.extend(status.peers.into_iter().map(|peer| {
        let unified = peer
            .gpus
            .iter()
            .find_map(|gpu| gpu.unified_memory)
            .or(peer.is_soc);
        plan_node(
            peer.id,
            peer.role,
            peer.hostname,
            gb_to_bytes(peer.vram_gb),
            unified,
            best_status_bandwidth(&peer.gpus),
            peer.rtt_ms,
        )
    }));
    nodes
}

fn best_status_bandwidth(gpus: &[StatusGpu]) -> Option<f64> {
    gpus.iter()
        .filter_map(|gpu| gpu.mem_bandwidth_gbps)
        .max_by(f64::total_cmp)
}

fn gb_to_bytes(gb: f64) -> u64 {
    if !gb.is_finite() || gb <= 0.0 {
        return 0;
    }
    (gb * 1e9) as u64
}

fn find_catalog_model_hint(query: &str) -> Option<CatalogModelHint> {
    let hints = catalog_model_hints();
    let normalized_query = normalize_match_key(query);
    hints.into_iter().find(|hint| {
        hint.aliases()
            .iter()
            .any(|alias| normalize_match_key(alias) == normalized_query)
    })
}

fn catalog_model_hints() -> Vec<CatalogModelHint> {
    let mut hints = Vec::new();
    for entry in remote_catalog::catalog_entries().unwrap_or_default() {
        collect_entry_hints(&entry, &mut hints);
    }
    hints
}

fn collect_entry_hints(entry: &CatalogEntry, hints: &mut Vec<CatalogModelHint>) {
    for (variant_name, variant) in &entry.variants {
        let source_file = variant
            .source
            .file
            .as_deref()
            .unwrap_or(variant_name.as_str());
        let Some(model_bytes) = variant
            .curated
            .size
            .as_deref()
            .and_then(parse_size_label_bytes)
            .or_else(|| {
                variant
                    .packages
                    .iter()
                    .filter(|package| package.package_type == "layer-package")
                    .filter_map(|package| package.total_bytes)
                    .max()
            })
        else {
            continue;
        };
        let layer_package = variant
            .packages
            .iter()
            .find(|package| package.package_type == "layer-package");
        let selector = model_ref::quant_selector_from_gguf_file(source_file);
        let model_ref = model_resolver::format_huggingface_display_ref(
            &variant.source.repo,
            variant.source.revision.as_deref(),
            source_file,
        );
        let moe_summary = variant.curated.moe.as_ref().map(format_moe_value);
        let haystack = format!(
            "{} {} {} {}",
            variant_name, variant.curated.name, variant.source.repo, source_file
        );
        let is_moe = moe_summary.is_some() || looks_like_moe(&haystack);
        let active_parameter_billions =
            active_parameter_billions(moe_summary.as_deref(), &haystack)
                .or_else(|| total_parameter_billions(&haystack))
                .map(OrderedF64);
        hints.push(CatalogModelHint {
            catalog_id: variant_name.to_string(),
            display_name: variant.curated.name.clone(),
            model_ref,
            source_repo: variant.source.repo.clone(),
            source_file: source_file.to_string(),
            model_bytes,
            split_capable: layer_package.is_some(),
            layer_package_repo: layer_package.map(|package| package.repo.clone()),
            layer_count: layer_package.and_then(|package| package.layer_count),
            package_total_bytes: layer_package.and_then(|package| package.total_bytes),
            is_moe,
            moe_summary,
            active_parameter_billions,
        });
        if selector.is_none() {
            continue;
        }
    }
}

impl CatalogModelHint {
    fn aliases(&self) -> Vec<String> {
        let basename = self
            .source_file
            .rsplit('/')
            .next()
            .unwrap_or(self.source_file.as_str());
        let selector = model_ref::quant_selector_from_gguf_file(&self.source_file);
        let ref_without_revision =
            model_ref::format_model_ref(&self.source_repo, None, selector.as_deref());
        vec![
            self.catalog_id.clone(),
            self.display_name.clone(),
            self.model_ref.clone(),
            self.source_repo.clone(),
            self.source_file.clone(),
            basename.to_string(),
            basename.trim_end_matches(".gguf").to_string(),
            ref_without_revision,
        ]
        .into_iter()
        .chain(self.layer_package_repo.clone())
        .collect()
    }
}

fn estimate_kv_cache_bytes(hint: &CatalogModelHint, context_tokens: u32) -> u64 {
    if context_tokens == 0 {
        return 0;
    }
    let active_billions = hint
        .active_parameter_billions
        .map(|value| value.0)
        .unwrap_or_else(|| (hint.model_bytes as f64 / 600_000_000.0).max(0.5));
    let bytes_per_token = estimated_kv_bytes_per_token(active_billions);
    bytes_per_token.saturating_mul(u64::from(context_tokens))
}

fn estimated_kv_bytes_per_token(active_parameter_billions: f64) -> u64 {
    if active_parameter_billions <= 1.0 {
        24 * 1024
    } else if active_parameter_billions <= 4.0 {
        64 * 1024
    } else if active_parameter_billions <= 12.0 {
        128 * 1024
    } else if active_parameter_billions <= 40.0 {
        256 * 1024
    } else if active_parameter_billions <= 100.0 {
        512 * 1024
    } else {
        768 * 1024
    }
}

fn format_moe_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(value) => value.clone(),
        other => other.to_string(),
    }
}

fn looks_like_moe(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    lower.contains("moe")
        || lower.contains("mixture-of-experts")
        || lower.contains("mixture_of_experts")
}

fn active_parameter_billions(moe_summary: Option<&str>, haystack: &str) -> Option<f64> {
    moe_summary
        .and_then(parameter_after_slash_billions)
        .or_else(|| parameter_after_a_billions(haystack))
}

fn parameter_after_slash_billions(value: &str) -> Option<f64> {
    value
        .split_once('/')
        .and_then(|(_, active)| first_parameter_billions(active))
}

fn parameter_after_a_billions(value: &str) -> Option<f64> {
    let lower = value.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    for index in 0..bytes.len().saturating_sub(1) {
        if bytes[index] == b'a'
            && bytes[index + 1].is_ascii_digit()
            && let Some(value) = parameter_number_at(&lower, index + 1)
        {
            return Some(value);
        }
    }
    None
}

fn total_parameter_billions(value: &str) -> Option<f64> {
    first_parameter_billions(value)
}

fn first_parameter_billions(value: &str) -> Option<f64> {
    let lower = value.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    for index in 0..bytes.len() {
        if bytes[index].is_ascii_digit()
            && let Some(value) = parameter_number_at(&lower, index)
        {
            return Some(value);
        }
    }
    None
}

fn parameter_number_at(value: &str, start: usize) -> Option<f64> {
    let bytes = value.as_bytes();
    let mut end = start;
    while end < bytes.len() && (bytes[end].is_ascii_digit() || bytes[end] == b'.') {
        end += 1;
    }
    if end == start || bytes.get(end).copied() != Some(b'b') {
        return None;
    }
    value[start..end]
        .parse::<f64>()
        .ok()
        .filter(|value| *value > 0.0)
}

fn normalize_match_key(value: &str) -> String {
    value.trim().trim_start_matches("hf://").to_lowercase()
}

fn parse_size_label_bytes(label: &str) -> Option<u64> {
    let compact = label.trim().replace(' ', "");
    if compact.is_empty() {
        return None;
    }

    let split_at = compact
        .find(|ch: char| !(ch.is_ascii_digit() || ch == '.'))
        .unwrap_or(compact.len());
    if split_at == 0 {
        return None;
    }
    let value = compact[..split_at].parse::<f64>().ok()?;
    if !value.is_finite() || value < 0.0 {
        return None;
    }

    let unit = compact[split_at..].to_ascii_lowercase();
    let multiplier = match unit.as_str() {
        "" | "b" => 1.0,
        "kb" => 1e3,
        "mb" => 1e6,
        "gb" => 1e9,
        "tb" => 1e12,
        "kib" => 1024.0,
        "mib" => 1024.0_f64.powi(2),
        "gib" => 1024.0_f64.powi(3),
        "tib" => 1024.0_f64.powi(4),
        _ => return None,
    };

    let bytes = value * multiplier;
    if bytes > u64::MAX as f64 {
        return None;
    }
    Some(bytes as u64)
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
