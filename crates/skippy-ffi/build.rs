fn main() {
    println!("cargo:rerun-if-env-changed=LLAMA_STAGE_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_STAGE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_STAGE_LINK_MODE");
    println!("cargo:rerun-if-env-changed=SKIPPY_LLAMA_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=SKIPPY_LLAMA_LIB_DIR");
    println!("cargo:rerun-if-env-changed=SKIPPY_LLAMA_LINK_MODE");
    println!("cargo:rerun-if-env-changed=SKIPPY_LLAMA_TARBALL_URL");
    println!("cargo:rerun-if-env-changed=SKIPPY_LLAMA_TARBALL_SHA256");
    println!("cargo:rerun-if-env-changed=SKIPPY_LLAMA_TARBALL_FLAVOR");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=LLVMInstallDir");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    // External-consumer path: if SKIPPY_LLAMA_TARBALL_URL is set, download
    // a prebuilt tarball containing the patched llama.cpp static archives,
    // extract it to a stable per-user cache directory, and point
    // SKIPPY_LLAMA_BUILD_DIR at the extracted root. The rest of this file
    // then proceeds as if the consumer had run `just llama-build`
    // themselves. Workspace-internal builds (no env var set) are
    // unaffected.
    if std::env::var("SKIPPY_LLAMA_BUILD_DIR").is_err()
        && std::env::var("LLAMA_STAGE_BUILD_DIR").is_err()
    {
        if let Ok(url) = std::env::var("SKIPPY_LLAMA_TARBALL_URL") {
            if !url.is_empty() {
                let build_dir = fetch_and_extract_llama_stage(&url);
                // Safety: setting env vars at the start of build.rs before any
                // thread spawn is OK; build scripts are single-threaded by
                // convention.
                unsafe { std::env::set_var("SKIPPY_LLAMA_BUILD_DIR", &build_dir) };
            }
        }
    }

    let link_mode =
        std::env::var("LLAMA_STAGE_LINK_MODE").or_else(|_| std::env::var("SKIPPY_LLAMA_LINK_MODE"));
    if link_mode.as_deref() == Ok("dynamic") {
        if let Ok(lib_dir) =
            std::env::var("LLAMA_STAGE_LIB_DIR").or_else(|_| std::env::var("SKIPPY_LLAMA_LIB_DIR"))
        {
            println!("cargo:rustc-link-search=native={lib_dir}");
        }
        println!("cargo:rustc-link-lib=dylib=mtmd");
        println!("cargo:rustc-link-lib=dylib=llama-common");
        println!("cargo:rustc-link-lib=dylib=llama");
        return;
    }

    let workspace_root = std::path::PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"),
    )
    .join("../..");
    let target = std::env::var("TARGET").unwrap_or_default();
    let build_dir = std::env::var("LLAMA_STAGE_BUILD_DIR")
        .or_else(|_| std::env::var("SKIPPY_LLAMA_BUILD_DIR"))
        .map(std::path::PathBuf::from)
        .map(|path| {
            if path.is_absolute() {
                path
            } else {
                workspace_root.join(path)
            }
        })
        .unwrap_or_else(|_| default_build_dir(&workspace_root, &target));

    let search_dirs = [
        build_dir.join("tools/mtmd"),
        build_dir.join("common"),
        build_dir.join("src"),
        build_dir.join("ggml/src"),
        build_dir.join("ggml/src/ggml-cpu"),
        build_dir.join("ggml/src/ggml-blas"),
        build_dir.join("ggml/src/ggml-cuda"),
        build_dir.join("ggml/src/ggml-hip"),
        build_dir.join("ggml/src/ggml-metal"),
        build_dir.join("ggml/src/ggml-vulkan"),
    ];

    for dir in search_dirs.iter().filter(|dir| dir.exists()) {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    let cmake_cache = build_dir.join("CMakeCache.txt");
    if cmake_cache.exists() {
        println!("cargo:rerun-if-changed={}", cmake_cache.display());
    }

    for (unix_archive, msvc_archive) in [
        ("src/libllama.a", "src/llama.lib"),
        ("tools/mtmd/libmtmd.a", "tools/mtmd/mtmd.lib"),
        ("common/libllama-common.a", "common/llama-common.lib"),
        (
            "common/libllama-common-base.a",
            "common/llama-common-base.lib",
        ),
        ("ggml/src/libggml.a", "ggml/src/ggml.lib"),
        ("ggml/src/libggml-base.a", "ggml/src/ggml-base.lib"),
        (
            "ggml/src/ggml-cpu/libggml-cpu.a",
            "ggml/src/ggml-cpu/ggml-cpu.lib",
        ),
        (
            "ggml/src/ggml-blas/libggml-blas.a",
            "ggml/src/ggml-blas/ggml-blas.lib",
        ),
        (
            "ggml/src/ggml-cuda/libggml-cuda.a",
            "ggml/src/ggml-cuda/ggml-cuda.lib",
        ),
        (
            "ggml/src/ggml-hip/libggml-hip.a",
            "ggml/src/ggml-hip/ggml-hip.lib",
        ),
        (
            "ggml/src/ggml-metal/libggml-metal.a",
            "ggml/src/ggml-metal/ggml-metal.lib",
        ),
        (
            "ggml/src/ggml-vulkan/libggml-vulkan.a",
            "ggml/src/ggml-vulkan/ggml-vulkan.lib",
        ),
    ] {
        for archive in [unix_archive, msvc_archive]
            .iter()
            .map(|path| build_dir.join(path))
            .filter(|archive| archive.exists())
        {
            println!("cargo:rerun-if-changed={}", archive.display());
        }
    }

    if static_archive_exists(&build_dir, "tools/mtmd/libmtmd.a", "tools/mtmd/mtmd.lib") {
        println!("cargo:rustc-link-lib=static=mtmd");
    }
    println!("cargo:rustc-link-lib=static=llama-common");
    println!("cargo:rustc-link-lib=static=llama-common-base");
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    let has_cuda = static_archive_exists(
        &build_dir,
        "ggml/src/ggml-cuda/libggml-cuda.a",
        "ggml/src/ggml-cuda/ggml-cuda.lib",
    );
    if has_cuda {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }
    let has_hip = static_archive_exists(
        &build_dir,
        "ggml/src/ggml-hip/libggml-hip.a",
        "ggml/src/ggml-hip/ggml-hip.lib",
    );
    if has_hip {
        println!("cargo:rustc-link-lib=static=ggml-hip");
    }
    let has_vulkan = static_archive_exists(
        &build_dir,
        "ggml/src/ggml-vulkan/libggml-vulkan.a",
        "ggml/src/ggml-vulkan/ggml-vulkan.lib",
    );
    if has_vulkan {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
    }
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    if static_archive_exists(
        &build_dir,
        "ggml/src/ggml-blas/libggml-blas.a",
        "ggml/src/ggml-blas/ggml-blas.lib",
    ) {
        println!("cargo:rustc-link-lib=static=ggml-blas");
    }
    if static_archive_exists(
        &build_dir,
        "ggml/src/ggml-metal/libggml-metal.a",
        "ggml/src/ggml-metal/ggml-metal.lib",
    ) {
        println!("cargo:rustc-link-lib=static=ggml-metal");
    }
    println!("cargo:rustc-link-lib=static=ggml-base");

    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        if static_archive_exists(
            &build_dir,
            "ggml/src/ggml-metal/libggml-metal.a",
            "ggml/src/ggml-metal/ggml-metal.lib",
        ) {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
    } else if target.contains("android") {
        println!("cargo:rustc-link-lib=static=c++_static");
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=log");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=pthread");
        for lib in linux_openmp_libs(&cmake_cache) {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
        if has_cuda {
            link_linux_cuda_libs(&cmake_cache);
        }
        if has_hip {
            link_linux_hip_libs();
        }
        if has_vulkan {
            println!("cargo:rustc-link-lib=dylib=vulkan");
        }
    } else if target.contains("windows") {
        link_windows_openmp_libs(&cmake_cache);
        if has_cuda {
            link_windows_cuda_libs(&cmake_cache);
        }
        if has_hip {
            link_windows_hip_libs();
        }
        if has_vulkan {
            link_windows_vulkan_libs();
        }
    }
}

fn fetch_and_extract_llama_stage(url: &str) -> String {
    use std::path::PathBuf;
    use std::process::Command;

    let target = std::env::var("TARGET").unwrap_or_default();
    let flavor = std::env::var("SKIPPY_LLAMA_TARBALL_FLAVOR").unwrap_or_else(|_| {
        if target.contains("apple") {
            "metal".into()
        } else {
            "cpu".into()
        }
    });
    let artifact_id = format!("llama-stage-{target}-{flavor}");
    let version = std::env::var("CARGO_PKG_VERSION").expect("CARGO_PKG_VERSION");

    let cache_root = std::env::var("SKIPPY_LLAMA_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home)
                .join(".cache")
                .join("skippy-llama-stage")
        });
    let cache_dir = cache_root.join(&version).join(&artifact_id);
    std::fs::create_dir_all(&cache_dir).expect("create skippy-llama-stage cache dir");

    let tarball_name = format!("{artifact_id}.tar.gz");
    let tarball_path = cache_dir.join(&tarball_name);
    let sha_path = cache_dir.join(format!("{tarball_name}.sha256"));

    fetch_url(url, &tarball_path);
    fetch_url(&format!("{url}.sha256"), &sha_path);

    let expected_sha = std::env::var("SKIPPY_LLAMA_TARBALL_SHA256")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            std::fs::read_to_string(&sha_path)
                .unwrap_or_else(|e| panic!("skippy-ffi: missing .sha256 sidecar at {} and SKIPPY_LLAMA_TARBALL_SHA256 not set: {e}", sha_path.display()))
                .split_whitespace()
                .next()
                .expect("sha sidecar empty")
                .to_string()
        });
    let actual_sha = sha256_of_file(&tarball_path);
    assert_eq!(
        actual_sha.to_lowercase(),
        expected_sha.to_lowercase(),
        "skippy-ffi: tarball sha256 mismatch: expected {expected_sha}, got {actual_sha}"
    );

    let extract_root = cache_dir.join("extracted");
    if extract_root.exists() {
        std::fs::remove_dir_all(&extract_root).expect("clean previous extract");
    }
    std::fs::create_dir_all(&extract_root).expect("create extract dir");
    let status = Command::new("tar")
        .arg("xzf")
        .arg(&tarball_path)
        .arg("-C")
        .arg(&extract_root)
        .status()
        .expect("invoke tar");
    assert!(status.success(), "skippy-ffi: tar extraction failed");

    // Tarball layout: <artifact_id-without-flavor-suffix>/<llama build files>.
    // We accept either `<target>-<flavor>/...` or a single top-level dir.
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&extract_root)
        .expect("read extract root")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    assert_eq!(
        entries.len(),
        1,
        "skippy-ffi: expected exactly one top-level directory in tarball, got {}",
        entries.len()
    );
    let build_dir = entries.remove(0);
    build_dir.to_string_lossy().into_owned()
}

fn fetch_url(url: &str, dest: &std::path::Path) {
    use std::process::Command;

    if let Some(local) = url.strip_prefix("file://") {
        if dest.exists() {
            std::fs::remove_file(dest).ok();
        }
        let src = std::path::PathBuf::from(local);
        if src.exists() {
            std::fs::copy(&src, dest)
                .unwrap_or_else(|e| panic!("skippy-ffi: failed to copy {}: {e}", src.display()));
        }
        return;
    }
    let status = Command::new("curl")
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--retry",
            "5",
            "-o",
        ])
        .arg(dest)
        .arg(url)
        .status()
        .expect("invoke curl");
    assert!(status.success(), "skippy-ffi: curl failed to fetch {url}");
}

fn sha256_of_file(path: &std::path::Path) -> String {
    let output = std::process::Command::new("shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()
        .expect("invoke shasum");
    assert!(output.status.success(), "shasum failed");
    String::from_utf8(output.stdout)
        .expect("shasum utf8")
        .split_whitespace()
        .next()
        .expect("shasum empty")
        .to_string()
}

fn default_build_dir(workspace_root: &std::path::Path, target: &str) -> std::path::PathBuf {
    let suffix = if target.contains("apple") {
        "metal"
    } else {
        "cpu"
    };
    workspace_root.join(format!(".deps/llama-build/build-stage-abi-{suffix}"))
}

fn static_archive_exists(
    build_dir: &std::path::Path,
    unix_archive: &str,
    msvc_archive: &str,
) -> bool {
    build_dir.join(unix_archive).exists() || build_dir.join(msvc_archive).exists()
}

fn link_linux_cuda_libs(cmake_cache: &std::path::Path) {
    for (cache_key, lib) in [
        ("CUDA_cuda_driver_LIBRARY", "cuda"),
        ("CUDA_cudart_LIBRARY", "cudart"),
        ("CUDA_cublas_LIBRARY", "cublas"),
        ("CUDA_cublasLt_LIBRARY", "cublasLt"),
    ] {
        link_linux_lib_from_cache(cmake_cache, cache_key, lib);
    }
    // NCCL is conditionally linked by CMake when found on the system.
    // Check CMakeCache for NCCL_FOUND or NCCL_LIBRARY to detect this and extract the search path.
    if let Ok(contents) = std::fs::read_to_string(cmake_cache) {
        let mut nccl_found = cmake_cache_bool(&contents, "NCCL_FOUND");
        if let Some(nccl_path) = cmake_cache_value(&contents, "NCCL_LIBRARY")
            && !nccl_path.contains("NOTFOUND")
            && !nccl_path.contains("-NOTFOUND")
        {
            nccl_found = true;
            let path = std::path::PathBuf::from(&nccl_path);
            if let Some(parent) = path.parent()
                && parent.is_dir()
            {
                println!("cargo:rustc-link-search=native={}", parent.display());
            }
        }
        if nccl_found {
            println!("cargo:rustc-link-lib=dylib=nccl");
        }
    }
}

fn link_windows_cuda_libs(cmake_cache: &std::path::Path) {
    for path in windows_cuda_search_paths(cmake_cache) {
        if path.is_dir() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }
    for lib in ["cuda", "cudart", "cublas", "cublasLt"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}

fn windows_cuda_search_paths(cmake_cache: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();
    if let Ok(cache) = std::fs::read_to_string(cmake_cache) {
        for key in [
            "CUDA_cuda_driver_LIBRARY",
            "CUDA_cudart_LIBRARY",
            "CUDA_cublas_LIBRARY",
            "CUDA_cublasLt_LIBRARY",
        ] {
            if let Some(value) = cmake_cache_value(&cache, key) {
                let path = std::path::PathBuf::from(value);
                if let Some(parent) = path.parent() {
                    push_unique_path(&mut paths, parent.to_path_buf());
                }
            }
        }
    }
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        push_unique_path(
            &mut paths,
            std::path::PathBuf::from(cuda_path).join("lib/x64"),
        );
    }
    paths
}

fn link_windows_hip_libs() {
    for env_name in ["ROCM_PATH", "HIP_PATH"] {
        if let Ok(root) = std::env::var(env_name) {
            for suffix in ["lib", "hip/lib"] {
                let path = std::path::PathBuf::from(&root).join(suffix);
                if path.is_dir() {
                    println!("cargo:rustc-link-search=native={}", path.display());
                }
            }
        }
    }
    for lib in ["amdhip64", "rocblas", "hipblas"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}

fn link_windows_vulkan_libs() {
    if let Ok(vulkan_sdk) = std::env::var("VULKAN_SDK") {
        let lib_dir = std::path::PathBuf::from(vulkan_sdk).join("Lib");
        if lib_dir.is_dir() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib=vulkan-1");
}

fn link_windows_openmp_libs(cmake_cache: &std::path::Path) {
    let libs = openmp_libs(cmake_cache, "vcomp");
    if libs.is_empty() {
        return;
    }

    for path in windows_openmp_search_paths(cmake_cache, &libs) {
        if path.is_dir() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }

    for lib in libs {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}

fn link_linux_hip_libs() {
    // Add ROCm library search paths
    for search_path in ["/opt/rocm/lib", "/opt/rocm/hip/lib"] {
        if std::path::Path::new(search_path).is_dir() {
            println!("cargo:rustc-link-search=native={search_path}");
        }
    }
    for lib in ["amdhip64", "rocblas", "hipblas"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    // RCCL (ROCm Collective Communications Library) provides the NCCL interface
    // for multi-GPU communication. Link it if available on the system.
    if std::path::Path::new("/opt/rocm/lib/librccl.so").exists() {
        println!("cargo:rustc-link-lib=dylib=rccl");
    }
}

fn push_unique_path(paths: &mut Vec<std::path::PathBuf>, path: std::path::PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

fn windows_openmp_search_paths(
    cmake_cache: &std::path::Path,
    libs: &[String],
) -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();
    if let Ok(cache) = std::fs::read_to_string(cmake_cache) {
        for lib in libs {
            for key in [
                format!("OpenMP_{lib}_LIBRARY"),
                format!("OpenMP_{lib}_LIBRARY_RELEASE"),
                format!("OpenMP_{lib}_LIBRARY_DEBUG"),
            ] {
                if let Some(value) = cmake_cache_value(&cache, &key) {
                    let path = std::path::PathBuf::from(value);
                    if let Some(parent) = path.parent() {
                        push_unique_path(&mut paths, parent.to_path_buf());
                    }
                }
            }
        }
    }

    for env_name in ["ROCM_PATH", "HIP_PATH", "LLVMInstallDir"] {
        if let Ok(root) = std::env::var(env_name) {
            for suffix in ["lib", "llvm/lib"] {
                push_unique_path(&mut paths, std::path::PathBuf::from(&root).join(suffix));
            }
        }
    }

    paths
}

fn link_linux_lib_from_cache(cmake_cache: &std::path::Path, cache_key: &str, lib: &str) {
    if let Ok(cache) = std::fs::read_to_string(cmake_cache)
        && let Some(path) = cmake_cache_value(&cache, cache_key)
    {
        let path = std::path::PathBuf::from(path);
        if path.exists()
            && let Some(parent) = path.parent()
        {
            println!("cargo:rustc-link-search=native={}", parent.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib={lib}");
}

fn linux_openmp_libs(cmake_cache: &std::path::Path) -> Vec<String> {
    openmp_libs(cmake_cache, "gomp")
}

fn openmp_libs(cmake_cache: &std::path::Path, fallback: &str) -> Vec<String> {
    let Ok(cache) = std::fs::read_to_string(cmake_cache) else {
        return Vec::new();
    };

    let mut libs = Vec::new();
    for key in ["OpenMP_C_LIB_NAMES", "OpenMP_CXX_LIB_NAMES"] {
        if let Some(value) = cmake_cache_value(&cache, key) {
            for lib in value.split(';') {
                let lib = lib.trim();
                if lib.is_empty() || lib == "NOTFOUND" || lib == "pthread" {
                    continue;
                }
                if !libs.iter().any(|existing| existing == lib) {
                    libs.push(lib.to_string());
                }
            }
        }
    }

    if libs.is_empty() && cmake_cache_bool(&cache, "GGML_OPENMP_ENABLED") {
        let fallback = if openmp_flags_reference_libomp(&cache) {
            "libomp"
        } else {
            fallback
        };
        libs.push(fallback.to_string());
    }

    libs
}

fn openmp_flags_reference_libomp(cache: &str) -> bool {
    ["OpenMP_C_FLAGS", "OpenMP_CXX_FLAGS"]
        .iter()
        .filter_map(|key| cmake_cache_value(cache, key))
        .any(|value| value.to_ascii_lowercase().contains("libomp"))
}

fn cmake_cache_value(cache: &str, key: &str) -> Option<String> {
    cache.lines().find_map(|line| {
        let (lhs, rhs) = line.split_once('=')?;
        let (name, _) = lhs.split_once(':')?;
        (name == key).then(|| rhs.to_string())
    })
}

fn cmake_cache_bool(cache: &str, key: &str) -> bool {
    cmake_cache_value(cache, key)
        .map(|value| matches!(value.as_str(), "ON" | "TRUE" | "1"))
        .unwrap_or(false)
}
