use anyhow::{Context, Result};
use model_fit::{
    AcceleratorKind, BackendKind, CpuProfile, FitStatus, GpuBenchmarkAcceleratorFacts,
    GpuBenchmarkHardwareInput, GpuBenchmarkOutput, HardwareProfile, MemoryProfile,
    ModelRecommendation, SelectionConfig, WorkloadProfile, hardware_profile_from_gpu_benchmark,
    profile_gguf_path, profile_hf_cache, rank_models, score_model,
};
use std::path::{Path, PathBuf};

const GIB: u64 = 1024 * 1024 * 1024;

fn main() -> Result<()> {
    let args = Args::parse()?;
    let hardware = load_hardware_profile(&args)?;
    let paths = collect_ggufs(&args.paths)?;
    let mut profiles = Vec::new();
    for cache_root in &args.hf_cache_roots {
        for installed in profile_hf_cache(cache_root)? {
            let mut profile = installed.profile;
            profile.source.id = installed.model_ref;
            profiles.push(profile);
        }
    }
    for path in paths {
        match profile_gguf_path(&path) {
            Ok(profile) => profiles.push(profile),
            Err(err) => eprintln!("skip {}: {err:#}", path.display()),
        }
    }
    if args.workload == "all" {
        let recommendations = best_workload_recommendations(&hardware, &profiles);
        print_best_table(&recommendations);
    } else {
        let workload = args.workload_profile();
        let mut config = SelectionConfig {
            workload,
            ..SelectionConfig::default()
        };
        config.weights = config.workload.default_weights();
        let recommendations = rank_models(&hardware, &profiles, &config);
        print_table(&recommendations);
    }
    Ok(())
}

#[derive(Debug)]
struct Args {
    memory_gib: Option<u64>,
    available_gib: Option<u64>,
    workload: String,
    gpu_benchmark_json: PathBuf,
    backend: BackendKind,
    unified: bool,
    accelerator_name: Option<String>,
    hf_cache_roots: Vec<PathBuf>,
    paths: Vec<PathBuf>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            memory_gib: None,
            available_gib: None,
            workload: "chat".into(),
            gpu_benchmark_json: PathBuf::new(),
            backend: BackendKind::Metal,
            unified: true,
            accelerator_name: None,
            hf_cache_roots: Vec::new(),
            paths: Vec::new(),
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--memory-gib" => parsed.memory_gib = Some(parse_next(&mut args, "--memory-gib")?),
                "--available-gib" => {
                    parsed.available_gib = Some(parse_next(&mut args, "--available-gib")?);
                }
                "--gpu-benchmark-json" => {
                    parsed.gpu_benchmark_json =
                        PathBuf::from(next_value(&mut args, "--gpu-benchmark-json")?);
                }
                "--backend" => parsed.backend = parse_backend(&next_value(&mut args, "--backend")?),
                "--discrete" => parsed.unified = false,
                "--unified" => parsed.unified = true,
                "--accelerator-name" => {
                    parsed.accelerator_name = Some(next_value(&mut args, "--accelerator-name")?);
                }
                "--workload" => parsed.workload = next_value(&mut args, "--workload")?,
                "--hf-cache" => {
                    parsed
                        .hf_cache_roots
                        .push(PathBuf::from(next_value(&mut args, "--hf-cache")?));
                }
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                _ => parsed.paths.push(PathBuf::from(arg)),
            }
        }
        if parsed.paths.is_empty() && parsed.hf_cache_roots.is_empty() {
            anyhow::bail!("provide at least one GGUF file/directory or --hf-cache root");
        }
        if parsed.gpu_benchmark_json.as_os_str().is_empty() {
            anyhow::bail!("provide --gpu-benchmark-json from `mesh-llm gpus benchmark --json`");
        }
        Ok(parsed)
    }

    fn workload_profile(&self) -> WorkloadProfile {
        match self.workload.as_str() {
            "coding" | "coding-agent" => WorkloadProfile::coding_agent(),
            "tool" | "tool-calling" => WorkloadProfile::tool_calling(),
            "summarization" | "summary" => WorkloadProfile::summarization(),
            "embedding" | "embeddings" => WorkloadProfile::embedding(),
            "reranking" | "reranker" => WorkloadProfile::reranking(),
            "vision" | "vision-chat" => WorkloadProfile::vision_chat(),
            "general" => WorkloadProfile::general_generation(),
            _ => WorkloadProfile::chat(),
        }
    }
}

fn load_hardware_profile(args: &Args) -> Result<HardwareProfile> {
    let bytes = if args.gpu_benchmark_json == Path::new("-") {
        use std::io::Read;
        let mut bytes = Vec::new();
        std::io::stdin()
            .read_to_end(&mut bytes)
            .context("read GPU benchmark JSON from stdin")?;
        bytes
    } else {
        std::fs::read(&args.gpu_benchmark_json).with_context(|| {
            format!(
                "read GPU benchmark JSON {}",
                args.gpu_benchmark_json.display()
            )
        })?
    };
    let benchmark_outputs: Vec<GpuBenchmarkOutput> =
        serde_json::from_slice(&bytes).context("parse mesh-llm-gpu-bench output JSON")?;
    let memory = MemoryProfile {
        total_system_bytes: args.memory_gib.map(|gib| gib * GIB),
        available_system_bytes: args.available_gib.map(|gib| gib * GIB),
        total_unified_bytes: args.memory_gib.map(|gib| gib * GIB),
        available_unified_bytes: args.available_gib.map(|gib| gib * GIB),
    };
    let accelerators = benchmark_outputs
        .iter()
        .map(|_| GpuBenchmarkAcceleratorFacts {
            name: args.accelerator_name.clone(),
            kind: if args.unified {
                AcceleratorKind::IntegratedGpu
            } else {
                AcceleratorKind::DiscreteGpu
            },
            backend: Some(args.backend),
            total_memory_bytes: args.memory_gib.map(|gib| gib * GIB),
            available_memory_bytes: args.available_gib.map(|gib| gib * GIB),
            unified_memory: args.unified,
        })
        .collect();
    hardware_profile_from_gpu_benchmark(GpuBenchmarkHardwareInput {
        memory,
        cpu: CpuProfile::default(),
        default_backend: args.backend,
        accelerators,
        benchmark_outputs,
    })
}

fn collect_ggufs(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut ggufs = Vec::new();
    for path in paths {
        collect_ggufs_from_path(path, &mut ggufs)?;
    }
    ggufs.sort();
    Ok(ggufs)
}

fn collect_ggufs_from_path(path: &Path, ggufs: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_file() {
        if path.extension().and_then(|ext| ext.to_str()) == Some("gguf") {
            ggufs.push(path.to_path_buf());
        }
        return Ok(());
    }
    for entry in std::fs::read_dir(path).with_context(|| format!("read {}", path.display()))? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_ggufs_from_path(&entry.path(), ggufs)?;
        } else if file_type.is_file()
            && entry.path().extension().and_then(|ext| ext.to_str()) == Some("gguf")
        {
            ggufs.push(entry.path());
        }
    }
    Ok(())
}

fn print_table(recommendations: &[ModelRecommendation]) {
    println!(
        "{:<8} {:>5} {:>5} {:>8} {:>8} {:<14} {:<10} {:<60} path",
        "fit", "score", "ctx", "memGiB", "tok/s", "arch", "backend", "serve_target"
    );
    for rec in recommendations {
        println!(
            "{:<8} {:>5.2} {:>5.2} {:>8.1} {:>8} {:<14} {:<10} {:<60} {}",
            format!("{:?}", rec.fit_status),
            rec.total_score,
            rec.context_score,
            rec.estimated_runtime_memory_bytes as f64 / GIB as f64,
            rec.estimated_decode_tokens_per_sec
                .map(|tps| format!("{tps:.1}"))
                .unwrap_or_else(|| "-".into()),
            format!("{:?}", rec.architecture_class),
            format!("{:?}", rec.selected_backend),
            serve_target(rec),
            rec.source.id
        );
        for warning in rec.warnings.iter().take(2) {
            println!("  warning: {warning}");
        }
    }
}

fn best_workload_recommendations(
    hardware: &HardwareProfile,
    profiles: &[model_fit::ModelProfile],
) -> Vec<(String, ModelRecommendation)> {
    let workloads = [
        ("chat", WorkloadProfile::chat()),
        ("coding-agent", WorkloadProfile::coding_agent()),
        ("tool-calling", WorkloadProfile::tool_calling()),
        ("summarization", WorkloadProfile::summarization()),
        ("embedding", WorkloadProfile::embedding()),
        ("reranking", WorkloadProfile::reranking()),
        ("vision-chat", WorkloadProfile::vision_chat()),
    ];
    let mut rows = profiles
        .iter()
        .map(|profile| {
            let scored = workloads
                .iter()
                .map(|(label, workload)| {
                    let mut config = SelectionConfig {
                        workload: workload.clone(),
                        ..SelectionConfig::default()
                    };
                    config.weights = config.workload.default_weights();
                    (
                        (*label).to_string(),
                        score_model(hardware, profile, &config),
                    )
                })
                .collect::<Vec<_>>();
            scored
                .iter()
                .filter(|(_, rec)| rec.fit_status != FitStatus::Rejected)
                .max_by(|(_, left), (_, right)| compare_score(left, right))
                .cloned()
                .unwrap_or_else(|| {
                    let rec = scored
                        .into_iter()
                        .next()
                        .expect("workload list must not be empty")
                        .1;
                    ("not-standalone".into(), rec)
                })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|(_, left), (_, right)| {
        right
            .total_score
            .partial_cmp(&left.total_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.source.id.cmp(&right.source.id))
    });
    rows
}

fn compare_score(left: &ModelRecommendation, right: &ModelRecommendation) -> std::cmp::Ordering {
    left.total_score
        .partial_cmp(&right.total_score)
        .unwrap_or(std::cmp::Ordering::Equal)
}

fn print_best_table(recommendations: &[(String, ModelRecommendation)]) {
    println!(
        "{:<15} {:<8} {:>5} {:>8} {:>8} {:<14} {:<60} path",
        "workload", "fit", "score", "memGiB", "tok/s", "arch", "serve_target"
    );
    for (workload, rec) in recommendations {
        println!(
            "{:<15} {:<8} {:>5.2} {:>8.1} {:>8} {:<14} {:<60} {}",
            workload,
            format!("{:?}", rec.fit_status),
            rec.total_score,
            rec.estimated_runtime_memory_bytes as f64 / GIB as f64,
            rec.estimated_decode_tokens_per_sec
                .map(|tps| format!("{tps:.1}"))
                .unwrap_or_else(|| "-".into()),
            format!("{:?}", rec.architecture_class),
            serve_target(rec),
            rec.source.id
        );
        for warning in rec.warnings.iter().take(2) {
            println!("  warning: {warning}");
        }
    }
}

fn serve_target(rec: &ModelRecommendation) -> String {
    if model_ref::ModelRef::parse(&rec.source.id).is_ok()
        && !rec.source.id.starts_with("local-gguf/")
    {
        return format!("--model {}", shell_quote(&rec.source.id));
    }
    let Some(path) = rec.source.path.as_ref() else {
        return format!("--model {}", shell_quote(&rec.source.id));
    };
    if let Some(model_ref) = hf_model_ref_for_path(path) {
        format!("--model {}", shell_quote(&model_ref))
    } else {
        format!("--gguf {}", shell_quote(&path.display().to_string()))
    }
}

fn hf_model_ref_for_path(path: &Path) -> Option<String> {
    for revision_dir in path.ancestors() {
        let snapshots_dir = revision_dir.parent()?;
        if snapshots_dir.file_name()? != "snapshots" {
            continue;
        }
        let repo_dir = snapshots_dir.parent()?;
        let repo_folder = repo_dir.file_name()?.to_str()?;
        let repo_id = repo_folder.strip_prefix("models--")?.replace("--", "/");
        let relative_file = path
            .strip_prefix(revision_dir)
            .ok()?
            .components()
            .map(|component| component.as_os_str().to_str())
            .collect::<Option<Vec<_>>>()?
            .join("/");
        if relative_file.is_empty() {
            continue;
        }
        let selector = model_ref::quant_selector_from_gguf_file(&relative_file)
            .or_else(|| model_ref::normalize_gguf_distribution_id(&relative_file));
        return Some(model_ref::format_model_ref(
            &repo_id,
            None,
            selector.as_deref(),
        ));
    }
    None
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn parse_next<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    next_value(args, name)?
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("invalid {name}: {err}"))
}

fn next_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| anyhow::anyhow!("{name} requires a value"))
}

fn parse_backend(value: &str) -> BackendKind {
    match value {
        "metal" => BackendKind::Metal,
        "cuda" => BackendKind::Cuda,
        "rocm" | "hip" => BackendKind::Rocm,
        "vulkan" => BackendKind::Vulkan,
        "cpu" => BackendKind::Cpu,
        _ => BackendKind::Unknown,
    }
}

fn print_usage() {
    eprintln!(
        "usage: summarize_ggufs --gpu-benchmark-json bench.json --memory-gib N --available-gib N [--backend metal|cuda|rocm|cpu] [--workload all|chat|coding-agent|tool-calling|summarization|embedding|reranking|vision-chat] [--hf-cache HUB_ROOT] <path>..."
    );
}
