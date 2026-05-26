//! Build script for `mesh-llm-native-sdk`.
//!
//! Fetches the matching `libmeshllm_ffi.{dylib,so,dll}` tarball for the
//! consumer's target platform + selected backend, verifies sha256,
//! extracts the shared lib into `OUT_DIR`, and emits link directives so
//! the consumer's binary links against it.
//!
//! ## Source of bits
//!
//! By default, downloads from a GitHub release URL constructed from
//! `CARGO_PKG_VERSION` (the workspace version). The exact same artifact
//! `scripts/package-native-sdk.sh` produces.
//!
//! Override with environment variables:
//!
//! - `MESH_LLM_NATIVE_TARBALL_URL` — `file://` or `https://` URL for the
//!   tarball. Useful for local trials before publishing to GitHub.
//! - `MESH_LLM_NATIVE_TARBALL_SHA256` — expected sha256 (hex) of the
//!   tarball; if set, must match. Otherwise the script fetches the
//!   `.sha256` sibling from the same URL.
//! - `MESH_LLM_NATIVE_CACHE_DIR` — where to cache downloaded tarballs
//!   between builds. Defaults to a per-user cache dir.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Re-run whenever any of these env vars change. Anything else is a
    // pure function of CARGO_PKG_VERSION + target + selected feature, so
    // cargo will rerun naturally when those change.
    println!("cargo:rerun-if-env-changed=MESH_LLM_NATIVE_TARBALL_URL");
    println!("cargo:rerun-if-env-changed=MESH_LLM_NATIVE_TARBALL_SHA256");
    println!("cargo:rerun-if-env-changed=MESH_LLM_NATIVE_CACHE_DIR");
    println!("cargo:rerun-if-changed=build.rs");

    let backend = select_backend();
    let target = TargetSpec::from_env();
    let version = env::var("CARGO_PKG_VERSION").expect("CARGO_PKG_VERSION");

    let artifact_id = format!("meshllm-native-{}-{}", target.platform_slug(), backend);
    let tarball_name = format!("{artifact_id}.tar.gz");

    let cache_dir = cache_dir(&version, &artifact_id);
    fs::create_dir_all(&cache_dir).expect("create cache dir");

    let tarball_path = cache_dir.join(&tarball_name);
    let sha_path = cache_dir.join(format!("{tarball_name}.sha256"));

    let tarball_url = match env::var("MESH_LLM_NATIVE_TARBALL_URL") {
        Ok(url) if !url.is_empty() => url,
        _ => default_tarball_url(&version, &tarball_name),
    };

    fetch_to(&tarball_url, &tarball_path);
    let sha_url = format!("{tarball_url}.sha256");
    fetch_sha_to(&sha_url, &sha_path);

    let expected_sha = match env::var("MESH_LLM_NATIVE_TARBALL_SHA256") {
        Ok(s) if !s.is_empty() => s.trim().to_string(),
        _ => read_sha_from_sidecar(&sha_path),
    };

    let actual_sha = sha256_of_file(&tarball_path);
    assert_eq!(
        actual_sha.to_lowercase(),
        expected_sha.to_lowercase(),
        "tarball sha256 mismatch: expected {expected_sha}, got {actual_sha}",
    );

    let extract_root = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let extract_dir = extract_root.join("native");
    if extract_dir.exists() {
        fs::remove_dir_all(&extract_dir).expect("clean previous extract dir");
    }
    fs::create_dir_all(&extract_dir).expect("create extract dir");

    let status = Command::new("tar")
        .arg("xzf")
        .arg(&tarball_path)
        .arg("-C")
        .arg(&extract_dir)
        .status()
        .expect("invoke tar");
    assert!(status.success(), "tar extraction failed");

    let lib_dir = extract_dir.join(&artifact_id).join("lib");
    assert!(
        lib_dir.is_dir(),
        "expected lib/ directory inside tarball at {}",
        lib_dir.display()
    );

    let lib_filename = target.library_filename();
    let lib_path = lib_dir.join(lib_filename);
    assert!(
        lib_path.is_file(),
        "expected static library {} inside extracted tarball",
        lib_path.display()
    );

    // Emit link directives. Cargo will pick up the static archive from
    // OUT_DIR/native/<artifact_id>/lib/ at link time and link it into
    // the consumer's final binary, exactly like the Swift xcframework
    // does for Swift apps.
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=meshllm_ffi");

    // System frameworks / libs needed for the platform's portion of
    // patched llama.cpp and skippy. These mirror what skippy-ffi's own
    // build.rs emits when linking the static archives.
    emit_system_link_directives(&target);

    // Help dependents find the static archive and metadata.
    println!("cargo:lib_dir={}", lib_dir.display());
    println!("cargo:library={}", lib_path.display());
}

fn emit_system_link_directives(target: &TargetSpec) {
    match target.os.as_str() {
        "macos" => {
            println!("cargo:rustc-link-lib=c++");
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
            println!("cargo:rustc-link-lib=framework=Security");
            println!("cargo:rustc-link-lib=framework=SystemConfiguration");
            println!("cargo:rustc-link-lib=framework=CoreFoundation");
        }
        "linux" => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=dylib=m");
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rustc-link-lib=dylib=pthread");
        }
        "windows" => {
            println!("cargo:rustc-link-lib=user32");
            println!("cargo:rustc-link-lib=ws2_32");
            println!("cargo:rustc-link-lib=bcrypt");
            println!("cargo:rustc-link-lib=ncrypt");
        }
        other => panic!("mesh-llm-native-sdk: unsupported target OS `{other}` for system link directives"),
    }
}

fn select_backend() -> &'static str {
    let mut selected: Option<&'static str> = None;
    let mut set = |name: &'static str| {
        if selected.is_some() {
            panic!(
                "mesh-llm-native-sdk: at most one backend feature may be enabled \
                 (already have `{}`, also got `{}`)",
                selected.unwrap(),
                name,
            );
        }
        selected = Some(name);
    };

    if env::var("CARGO_FEATURE_METAL").is_ok() {
        set("metal");
    }
    if env::var("CARGO_FEATURE_CPU").is_ok() {
        set("cpu");
    }
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        set("cuda");
    }
    if env::var("CARGO_FEATURE_ROCM").is_ok() {
        set("rocm");
    }
    if env::var("CARGO_FEATURE_VULKAN").is_ok() {
        set("vulkan");
    }

    selected.unwrap_or_else(|| {
        panic!(
            "mesh-llm-native-sdk: no backend selected. Enable exactly one of \
             features `metal`, `cpu`, `cuda`, `rocm`, `vulkan` on your dependency."
        )
    })
}

struct TargetSpec {
    os: String,
    arch: String,
}

impl TargetSpec {
    fn from_env() -> Self {
        Self {
            os: env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS"),
            arch: env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH"),
        }
    }

    /// Matches the `platform` field that `scripts/package-native-sdk.sh`
    /// emits into `manifest.json`. Keep in sync if the script changes.
    fn platform_slug(&self) -> String {
        let os = match self.os.as_str() {
            "macos" => "darwin",
            other => other,
        };
        format!("{}-{}", os, self.arch)
    }

    fn library_filename(&self) -> &'static str {
        match self.os.as_str() {
            "macos" | "linux" | "android" => "libmeshllm_ffi.a",
            "windows" => "meshllm_ffi.lib",
            other => panic!("mesh-llm-native-sdk: unsupported target OS `{other}`"),
        }
    }
}

fn default_tarball_url(version: &str, tarball_name: &str) -> String {
    format!("https://github.com/Mesh-LLM/mesh-llm/releases/download/v{version}/{tarball_name}")
}

fn cache_dir(version: &str, artifact_id: &str) -> PathBuf {
    if let Ok(custom) = env::var("MESH_LLM_NATIVE_CACHE_DIR") {
        if !custom.is_empty() {
            return PathBuf::from(custom).join(version).join(artifact_id);
        }
    }
    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("mesh-llm-native-sdk")
        .join(version)
        .join(artifact_id)
}

fn fetch_to(url: &str, dest: &Path) {
    if let Some(local) = strip_file_prefix(url) {
        if dest.exists() {
            fs::remove_file(dest).ok();
        }
        fs::copy(&local, dest).unwrap_or_else(|err| {
            panic!(
                "mesh-llm-native-sdk: failed to copy {} -> {}: {err}",
                local.display(),
                dest.display()
            )
        });
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
            "--retry-delay",
            "2",
            "-o",
        ])
        .arg(dest)
        .arg(url)
        .status()
        .expect("invoke curl");
    assert!(
        status.success(),
        "mesh-llm-native-sdk: curl failed to download {url}"
    );
}

fn fetch_sha_to(url: &str, dest: &Path) {
    if let Some(local) = strip_file_prefix(url) {
        if dest.exists() {
            fs::remove_file(dest).ok();
        }
        if local.exists() {
            fs::copy(&local, dest).expect("copy sha sidecar");
        }
        return;
    }
    // Best-effort: a missing .sha256 sidecar is allowed if the
    // consumer provided MESH_LLM_NATIVE_TARBALL_SHA256 explicitly.
    let _ = Command::new("curl")
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--retry",
            "5",
            "--retry-delay",
            "2",
            "-o",
        ])
        .arg(dest)
        .arg(url)
        .status();
}

fn strip_file_prefix(url: &str) -> Option<PathBuf> {
    url.strip_prefix("file://").map(PathBuf::from)
}

fn read_sha_from_sidecar(path: &Path) -> String {
    let contents = fs::read_to_string(path).unwrap_or_else(|err| {
        panic!(
            "mesh-llm-native-sdk: missing .sha256 sidecar at {} and \
             MESH_LLM_NATIVE_TARBALL_SHA256 was not set: {err}",
            path.display()
        )
    });
    // Format from `shasum -a 256 <file>`: "<hex>  <filename>"
    contents
        .split_whitespace()
        .next()
        .expect("sha sidecar empty")
        .to_string()
}

fn sha256_of_file(path: &Path) -> String {
    let output = Command::new("shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()
        .expect("invoke shasum");
    assert!(output.status.success(), "shasum failed");
    let stdout = String::from_utf8(output.stdout).expect("shasum output utf8");
    stdout
        .split_whitespace()
        .next()
        .expect("shasum output empty")
        .to_string()
}
