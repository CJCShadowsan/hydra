use super::{CapacityPlan, CapacityScopePlan, FitVerdict, RecommendationReport};

pub(super) fn render_capacity_plan(plan: &CapacityPlan) {
    println!("📦 Model capacity plan");
    println!("  Model: {}", plan.model.display_name);
    println!("  Ref: {}", plan.model.model_ref);
    println!("  Size: {}", format_bytes(plan.model.model_bytes));
    if plan.model.moe {
        println!(
            "  Type: MoE{}",
            plan.model
                .moe_summary
                .as_deref()
                .map(|summary| format!(" ({summary})"))
                .unwrap_or_default()
        );
    }
    if let Some(active) = plan.model.active_parameter_billions {
        println!("  Active/fit parameter hint: {active:.1}B");
    }
    println!("  Planning context: {} tokens", plan.model.context_tokens);
    println!(
        "  Weight budget: {}",
        format_bytes(plan.model.weight_budget_bytes)
    );
    println!(
        "  KV budget: {}",
        format_bytes(plan.model.kv_cache_budget_bytes)
    );
    println!(
        "  Total budget: {}",
        format_bytes(plan.model.required_bytes)
    );
    if let Some(repo) = plan.model.layer_package_repo.as_deref() {
        println!("  Split package: {repo}");
    } else {
        println!("  Split package: none in catalog");
    }
    println!();
    render_scope(&plan.local);
    if let Some(mesh) = &plan.mesh {
        println!();
        render_scope(mesh);
    }
    println!();
    render_disk(&plan.disk);
    println!();
    for note in &plan.notes {
        println!("  note: {note}");
    }
}

pub(super) fn render_recommendations(report: &RecommendationReport) {
    println!("📚 Recommended catalog models ({})", report.basis);
    if report.results.is_empty() {
        println!("  No catalog models had enough metadata to plan.");
        return;
    }
    for (index, row) in report.results.iter().enumerate() {
        if index > 0 {
            println!();
        }
        println!("{}. {}", index + 1, row.display_name);
        println!("   Fit: {}", fit_label(row.fit));
        println!("   Ref: {}", row.model_ref);
        println!(
            "   Size: {}, budget: {} at {} ctx",
            format_bytes(row.model_bytes),
            format_bytes(row.required_bytes),
            row.context_tokens
        );
        if row.moe {
            println!("   Type: MoE");
        }
        println!(
            "   Weights: {}, KV: {}",
            format_bytes(row.weight_budget_bytes),
            format_bytes(row.kv_cache_budget_bytes)
        );
        if let Some(active) = row.active_parameter_billions {
            println!("   Active/fit parameter hint: {active:.1}B");
        }
        if let Some(best) = row.best_single_node_capacity_bytes {
            println!("   Best node usable capacity: {}", format_bytes(best));
        }
        if row.aggregate_capacity_bytes > 0 {
            println!(
                "   Aggregate usable capacity: {}",
                format_bytes(row.aggregate_capacity_bytes)
            );
        }
        if let Some(repo) = row.layer_package_repo.as_deref() {
            println!("   Split package: {repo}");
        }
        println!(
            "   Disk: {} needed, {} free{}",
            format_bytes(row.disk_full_model_required_bytes),
            row.disk_free_bytes
                .map(format_bytes)
                .unwrap_or_else(|| "unknown".to_string()),
            disk_fit_suffix(row.disk_fits_full_model),
        );
        println!("   Plan: {}", row.plan_command);
    }
    println!();
    for note in &report.notes {
        println!("  note: {note}");
    }
}

fn render_disk(disk: &super::DiskPlan) {
    println!("Disk:");
    println!("  Cache: {}", disk.cache_dir);
    println!(
        "  Free: {}",
        disk.free_bytes
            .map(format_bytes)
            .unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "  Full model download: {}{}",
        format_bytes(disk.full_model_required_bytes),
        disk_fit_suffix(disk.fits_full_model)
    );
    if let Some(required) = disk.layer_package_total_required_bytes {
        println!(
            "  Full layer package: {}{}",
            format_bytes(required),
            disk_fit_suffix(disk.free_bytes.map(|free| free >= required))
        );
    }
    if let Some(required) = disk.one_layer_required_bytes {
        println!(
            "  Approx one layer: {}{}",
            format_bytes(required),
            disk_fit_suffix(disk.fits_one_layer)
        );
    }
    if let Some(required) = disk.planned_split_node_required_bytes {
        println!(
            "  Approx planned split node: {}{}",
            format_bytes(required),
            disk_fit_suffix(disk.fits_planned_split_node)
        );
    }
}

fn disk_fit_suffix(fit: Option<bool>) -> &'static str {
    match fit {
        Some(true) => " fits disk",
        Some(false) => " exceeds free disk",
        None => "",
    }
}

fn render_scope(scope: &CapacityScopePlan) {
    println!("{} capacity:", title_case(scope.label));
    println!("  Fit: {}", fit_label(scope.fit));
    println!("  Reason: {}", scope.reason);
    println!(
        "  Best node usable: {}",
        scope
            .best_single_node_capacity_bytes
            .map(format_bytes)
            .unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "  Aggregate usable: {}",
        format_bytes(scope.aggregate_capacity_bytes)
    );
    println!("  Eligible nodes: {}", scope.eligible_node_count);
    if let Some(count) = scope.nodes_needed_like_local {
        println!("  Machines like best local node needed: {count}");
    }
    if let Some(count) = scope.split_node_count {
        println!("  Suggested split nodes: {count}");
    }
    for node in &scope.suggested_nodes {
        println!(
            "    - {} {} usable={} raw={} reserve={}{}{}{}",
            node.id,
            node.hostname
                .as_deref()
                .map(|host| format!("({host})"))
                .unwrap_or_default(),
            format_bytes(node.capacity_bytes),
            format_bytes(node.raw_capacity_bytes),
            format_bytes(node.reserve_bytes),
            node.unified_memory
                .map(|unified| {
                    if unified {
                        " unified-memory".to_string()
                    } else {
                        " discrete-vram".to_string()
                    }
                })
                .unwrap_or_default(),
            node.bandwidth_gbps
                .map(|bandwidth| format!(" bandwidth={bandwidth:.0}GB/s"))
                .unwrap_or_default(),
            node.rtt_ms
                .map(|rtt| format!(" rtt={rtt}ms"))
                .unwrap_or_default(),
        );
    }
}

fn title_case(value: &str) -> String {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    first.to_uppercase().chain(chars).collect()
}

fn fit_label(verdict: FitVerdict) -> &'static str {
    match verdict {
        FitVerdict::ComfortableLocal => "comfortable single-node fit",
        FitVerdict::TightLocal => "tight single-node fit",
        FitVerdict::SplitCandidate => "split candidate",
        FitVerdict::InsufficientCapacity => "insufficient capacity",
        FitVerdict::UnknownCapacity => "unknown capacity",
        FitVerdict::NoEligibleHosts => "no eligible hosts",
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000_000 {
        format!("{:.1}TB", bytes as f64 / 1e12)
    } else if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.1}MB", bytes as f64 / 1e6)
    } else {
        format!("{bytes}B")
    }
}
