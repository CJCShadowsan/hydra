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

## Validation

Minimum validation for SDK work:

```bash
scripts/check-sdk-contract.sh
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
