# MeshLLM SDK

The MeshLLM SDK is one embedded node contract exposed through Rust, Swift, and
Kotlin. Language packages should feel native, but they must not invent different
behavior. A customer should be able to copy an example, run a real model, and
understand failures without reading host runtime internals.

## Release Gates

Before an SDK change is review-ready, these gates must be true:

1. **One canonical SDK contract.** Rust, Swift, and Kotlin expose the same node
   lifecycle, model management, serving, inference, streaming, cancellation,
   and typed error semantics.
2. **Real end-to-end example apps.** Examples exercise the real UniFFI/native
   runtime path and either run local serving or connect to a live fixture mesh.
   They must not rely on fake controllers or no-op stubs as the final signal.
3. **SDK lifecycle polish.** The main path is deterministic:
   create node, start, discover/download/show model, load, infer, unload, stop.
   Cancellation must cancel the real request and cleanup must release loaded
   models, runtime directories, ports, callbacks, and native handles.
4. **First-class errors.** SDK failures cross the FFI boundary as typed errors
   with actionable messages. String payloads may carry detail, but callers must
   be able to distinguish invalid identity, discovery, model management,
   serving, cancellation, and unsupported-platform failures.
5. **Platform support matrix.** Every package must document what is supported,
   what is client-only, and what is planned. Unsupported local serving must fail
   with a typed unsupported error, not a placeholder implementation.

## Canonical Contract

The public SDK concept is `Node`: an embedded mesh node with namespaced APIs for
inference, model management, and local serving.

```text
Node
  start()
  stop()
  reconnect()
  status()
  inference.listModels()
  inference.chat()/chatStream()/chatFlow()
  inference.responses()/responsesStream()/responsesFlow()
  inference.cancel()
  models.recommended()
  models.search()
  models.show()
  models.installed()
  models.cacheStatus()
  models.download()
  models.delete()
  models.cleanup()
  models.pruneDerivedCache()
  serving.status()
  serving.servedModels()
  serving.load()
  serving.unload()
  serving.unloadModel()
  serving.unloadInstance()
  serving.setDevicePolicy()
```

Language names should follow the language's module conventions. Swift and
Kotlin types do not need a `Mesh` prefix because the packages already provide
that namespace.

## Lifecycle Contract

Client-only use:

```text
generate/persist owner keypair
create Node(inviteToken, ownerKeypair, servingEnabled=false)
start
inference.listModels
inference.chat or inference.responses
stop
```

Local serving use:

```text
generate/persist owner keypair
create Node(inviteToken, ownerKeypair, servingEnabled=true)
models.search or models.show
models.download unless the model is already installed
start
serving.load(modelRef, devicePolicy)
inference.listModels until the loaded model is visible
inference.chat or inference.responses
serving.unload by instance id when available, otherwise by model id
stop
```

The examples in `sdk/swift/example` and `sdk/kotlin/example` are executable
versions of this contract. CI smoke jobs should run those examples against a
real fixture mesh or a real local model.

## Error Contract

The FFI error enum is part of the SDK contract:

| Error | Meaning |
|---|---|
| `InvalidInviteToken` | The invite token is empty, malformed, or cannot be accepted. |
| `InvalidOwnerKeypair` | The caller supplied an empty or malformed owner identity. |
| `BuildFailed` | The node could not be constructed from valid inputs. |
| `JoinFailed` | The node could not join the requested mesh. |
| `DiscoveryFailed` | Public mesh discovery failed. |
| `StreamFailed` | Streaming inference setup or delivery failed. |
| `Cancelled` | A request was cancelled. |
| `ReconnectFailed` | Reconnect failed after an existing node was created. |
| `HostUnavailable` | The selected host or endpoint is unavailable. |
| `ModelManagementFailed` | Search, show, download, install, delete, cleanup, or cache inspection failed. |
| `ServingFailed` | Serving load, unload, status, or device policy control failed. |
| `ServingUnsupported` | The current platform/build cannot provide local serving. |

Swift exposes this as `MeshError`. Kotlin exposes this as `MeshException`.
Generated UniFFI names may still exist, but wrapper docs and examples should use
the SDK-level aliases.

## Platform Support Matrix

| Platform/package | Mesh inference | Model management | Local serving | Backend status |
|---|---:|---:|---:|---|
| Rust SDK on macOS | yes | yes | yes | Metal and CPU builds are supported by the native runtime. |
| Rust SDK on Linux | yes | yes | yes | CPU is supported; CUDA, ROCm/HIP, and Vulkan depend on the selected native runtime build. |
| Swift macOS | yes | yes | yes | Uses `MeshLLMFFI.xcframework`; local serving is currently validated on host macOS with Metal. |
| Swift Mac Catalyst | yes | yes | planned | Package builds through the Apple XCFramework path; local serving must be validated per target before it is advertised. |
| Swift iOS | yes | model catalog/cache APIs only where filesystem policy allows | no | Client/mesh participation only until embedded serving is validated for iOS. |
| Kotlin JVM macOS | yes | yes | yes | Requires a matching `libmeshllm_ffi.dylib`; CI validates fixture-backed inference. |
| Kotlin JVM Linux | yes | yes | yes | Requires a matching `libmeshllm_ffi.so`; CPU/CUDA/Vulkan support is selected by the native runtime artifact. |
| Kotlin Android | yes | yes | planned | AAR packaging builds CPU native libraries; local serving remains platform-gated until Android runtime smoke passes. |

Any row marked `planned` must fail with `ServingUnsupported` for local serving
until CI proves the real path works.

## Native Runtime Artifacts

Swift and Kotlin packages should load MeshLLM through `libmeshllm_ffi`, not
through a public `libllama` contract. Backend-specific llama.cpp builds are an
implementation detail of the native SDK runtime artifact.

Native SDK runtime artifacts use this layout:

```text
meshllm-native-<platform>-<flavor>/
  manifest.json
  README.md
  lib/
    libmeshllm_ffi.{dylib|so}
    libuniffi_mesh_ffi.{dylib|so}
```

The duplicate `libuniffi_mesh_ffi` file exists because generated UniFFI loaders
look up the component library name. Both files contain the same native runtime.

The artifact manifest records the SDK version, target triple, backend flavor,
library checksum, llama.cpp upstream SHA, patched SHA, and patch digest. SDK
loaders must verify `library_sha256` before loading a downloaded artifact.

Baseline artifact names:

| Artifact | Target | Backend |
|---|---|---|
| `meshllm-native-darwin-aarch64-metal` | `aarch64-apple-darwin` | Metal |
| `meshllm-native-darwin-aarch64-cpu` | `aarch64-apple-darwin` | CPU |
| `meshllm-native-linux-x86_64-cpu` | `x86_64-unknown-linux-gnu` | CPU |
| `meshllm-native-linux-x86_64-cuda` | `x86_64-unknown-linux-gnu` | CUDA |
| `meshllm-native-linux-x86_64-vulkan` | `x86_64-unknown-linux-gnu` | Vulkan |
| `meshllm-native-linux-x86_64-rocm` | `x86_64-unknown-linux-gnu` | ROCm/HIP |

CUDA and ROCm artifacts may include hardware-specific suffixes such as
`cuda-sm80` or `rocm-gfx1100` when `LLAMA_STAGE_CUDA_ARCHITECTURES` or
`LLAMA_STAGE_AMDGPU_TARGETS` is set.

Build and package one flavor:

```bash
scripts/package-native-sdk.sh \
  --build \
  --backend metal \
  --target aarch64-apple-darwin \
  --out dist/native-sdk
```

Package an already-built `mesh-llm-ffi` library:

```bash
scripts/package-native-sdk.sh \
  --backend cpu \
  --target x86_64-unknown-linux-gnu
```

Verify produced artifacts:

```bash
scripts/verify-native-sdk-package.sh dist/native-sdk/*.tar.gz
```

Generate a crates.io-ready native runtime crate from a verified artifact:

```bash
scripts/package-native-sdk-crate.sh dist/native-sdk/meshllm-native-darwin-aarch64-metal.tar.gz
```

The generated crate contains the native runtime under `native/`, copies it into
Cargo's `OUT_DIR/native` during the crate build, and uses
`links = "meshllm_native_runtime"` so dependent Rust build scripts can discover
the build-output paths:

```text
DEP_MESHLLM_NATIVE_RUNTIME_ARTIFACT_ID
DEP_MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR
DEP_MESHLLM_NATIVE_RUNTIME_MANIFEST
DEP_MESHLLM_NATIVE_RUNTIME_LIB_DIR
DEP_MESHLLM_NATIVE_RUNTIME_LIBRARY
DEP_MESHLLM_NATIVE_RUNTIME_UNIFFI_LIBRARY
```

Rust apps should choose exactly one native runtime crate for the target they are
building. Use target-specific dependencies so Cargo selects the matching
platform/backend package:

```toml
[target.'cfg(all(target_os = "macos", target_arch = "aarch64"))'.dependencies]
meshllm-native-darwin-aarch64-metal = "0.66.0"

[target.'cfg(all(target_os = "linux", target_arch = "x86_64"))'.dependencies]
meshllm-native-linux-x86-64-cpu = "0.66.0"
```

Because the runtime crates share `links = "meshllm_native_runtime"`, Cargo will
reject selecting more than one of them for the same build. Apps that need to
ship multiple backend choices in one installer should package multiple verified
tarball artifacts explicitly and choose between those packaged directories at
runtime.

An app build script can then copy the selected runtime into the final app
bundle, installer, container image, or package resource directory:

```rust
use std::{env, fs, path::PathBuf};

fn main() {
    let artifact_dir = PathBuf::from(
        env::var("DEP_MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR")
            .expect("meshllm native runtime dependency"),
    );
    let package_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("meshllm-native");
    fs::create_dir_all(&package_dir).expect("create package native dir");
    // Copy artifact_dir recursively into package_dir with the app's packaging helper.
}
```

At runtime, the app loads `libmeshllm_ffi` from the packaged artifact directory
and verifies the `manifest.json` checksum before opening the dynamic library.
Swift and Kotlin packaging can consume the same verified tarball layout
directly when Cargo is not involved.

## Validation

Minimum validation for SDK work:

```bash
scripts/check-sdk-contract.sh
scripts/verify-native-sdk-package.sh dist/native-sdk/*.tar.gz
cargo test -p mesh-llm-ffi
swift build --package-path sdk/swift/example/MeshExampleApp
./gradlew --no-daemon compileKotlin -p sdk/kotlin/example/example-jvm
```

For serving, model management, or inference behavior, also run the live fixture
smokes through `scripts/ci-sdk-fixture.sh` or the CI smoke wrappers:

```bash
scripts/ci-swift-sdk-smoke.sh <mesh-llm> <bin-dir> <model.gguf>
scripts/ci-kotlin-sdk-smoke.sh <mesh-llm> <bin-dir> <model.gguf>
```
