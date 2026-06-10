//! Build script for `mesh-mlx-sys`.
//!
//! By default this does nothing: the FFI declarations type-check without the
//! native engine so CI can build pure-Rust logic on any platform.
//!
//! With the `link-mlx` feature (Apple Silicon only) it builds and links the MLX
//! C++/Metal engine and the `mlx-c` C API. The native sources are obtained via
//! CMake `FetchContent` driven by a tiny generated project, then linked
//! statically. This mirrors how the repo treats the patched llama.cpp ABI: the
//! native artifact is an explicit, opt-in build, not an implicit side effect.
//!
//! Environment overrides:
//!   - `MLX_C_DIR`  — path to a prebuilt/checked-out mlx-c (with CMakeLists).
//!   - `MLX_C_TAG`  — mlx-c git tag to fetch (default: a known-good pin).
//!   - `MLX_TAG`    — mlx git tag to fetch (default: matched by mlx-c).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MLX_C_DIR");

    if std::env::var("CARGO_FEATURE_LINK_MLX").is_err() {
        // Bindings-only mode: nothing to build or link.
        return;
    }

    #[cfg(not(target_os = "macos"))]
    {
        panic!("the `link-mlx` feature requires macOS (Apple Silicon / Metal)");
    }

    #[cfg(target_os = "macos")]
    native::build_and_link();
}

#[cfg(target_os = "macos")]
mod native {
    use std::path::PathBuf;
    use std::process::Command;

    const DEFAULT_MLX_C_TAG: &str = "v0.6.0";

    pub fn build_and_link() {
        let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let build_dir = out_dir.join("mlx-build");
        std::fs::create_dir_all(&build_dir).expect("create build dir");

        // A minimal CMake project that pulls mlx-c (which transitively pulls
        // mlx) and exposes the static libs. Using FetchContent keeps the heavy
        // sources out of the repo while pinning exact versions.
        let mlx_c_tag = std::env::var("MLX_C_TAG").unwrap_or_else(|_| DEFAULT_MLX_C_TAG.into());
        let cmakelists = build_dir.join("CMakeLists.txt");
        let project = format!(
            r#"
cmake_minimum_required(VERSION 3.25)
project(mesh_mlx_native LANGUAGES C CXX)
include(FetchContent)
set(MLX_C_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_METAL ON CACHE BOOL "" FORCE)
{mlx_c_source}
"#,
            mlx_c_source = mlx_c_fetch(&mlx_c_tag),
        );
        std::fs::write(&cmakelists, project).expect("write CMakeLists");

        // Configure + build the static libraries.
        run(
            Command::new("cmake")
                .arg("-G")
                .arg("Ninja")
                .arg("-DCMAKE_BUILD_TYPE=Release")
                .arg(format!("-B{}", build_dir.join("out").display()))
                .arg(format!("-S{}", build_dir.display())),
            "cmake configure",
        );
        run(
            Command::new("cmake")
                .arg("--build")
                .arg(build_dir.join("out"))
                .arg("--target")
                .arg("mlxc")
                .arg("--parallel"),
            "cmake build",
        );

        emit_link_flags(&build_dir.join("out"));
    }

    fn mlx_c_fetch(tag: &str) -> String {
        if let Ok(dir) = std::env::var("MLX_C_DIR") {
            format!("add_subdirectory({dir} mlx-c-build)")
        } else {
            format!(
                r#"
FetchContent_Declare(
  mlx-c
  GIT_REPOSITORY https://github.com/ml-explore/mlx-c.git
  GIT_TAG {tag}
)
FetchContent_MakeAvailable(mlx-c)
"#
            )
        }
    }

    fn emit_link_flags(out: &std::path::Path) {
        // CMake/FetchContent layout: libs land under `_deps/{mlx,mlx-c}-build`.
        // Search those plus a few fallbacks so layout changes don't break us.
        for sub in ["_deps/mlx-c-build", "_deps/mlx-build", "", "lib"] {
            println!("cargo:rustc-link-search=native={}", out.join(sub).display());
        }
        // Static C API first, then the engine it depends on (order matters for
        // static linking).
        println!("cargo:rustc-link-lib=static=mlxc");
        println!("cargo:rustc-link-lib=static=mlx");
        // Apple frameworks the Metal backend needs.
        for fw in ["Metal", "Foundation", "Accelerate", "QuartzCore"] {
            println!("cargo:rustc-link-lib=framework={fw}");
        }
        println!("cargo:rustc-link-lib=c++");
    }

    fn run(cmd: &mut Command, what: &str) {
        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("failed to run {what}: {e}"));
        assert!(status.success(), "{what} failed with {status}");
    }
}
