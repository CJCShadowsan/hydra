# Rust Native SDK: in-process mesh node from cargo

## Status: Design proposal

## Goal

A Rust application adds mesh-llm to `Cargo.toml`, runs `cargo build`, and
gets a real in-process mesh node — same shape as the Swift and Kotlin SDKs:

```toml
[dependencies]
mesh-llm-api-server = "0.66"
```

```rust
let node = MeshNode::builder()
    .identity(owner)
    .join(invite)
    .build()?;
node.start().await?;
// real iroh peer in this process, optional local serving
```

No CMake on the consumer's machine. No source build of patched llama.cpp.
No separate `mesh-llm` daemon. The native bits arrive with the crate, the
same way Swift and Kotlin consumers get them today.

## How Swift and Kotlin do it (the model we're matching)

Both ship a prebuilt native library — `libmeshllm_ffi` — that contains
patched llama.cpp linked statically and exposes a UniFFI-generated C ABI.
The language SDK calls into it. The consumer of the SDK runs a real mesh
node *in their own process*.

- **Swift:** `MeshLLMFFI.xcframework.zip` (~168 MB zipped) on each GitHub
  release. SwiftPM `.binaryTarget(url:, checksum:)` in `Package.swift`
  downloads and links it. Swift `Node` class wraps the FFI, exposes
  `init(servingEnabled: true)`, `.serving.load`, `.serving.unload`.
- **Kotlin:** the same native lib shipped as an AAR via Maven (GitHub
  Packages). Loaded by the JVM. Kotlin `Node` class wraps the same FFI.
- **Node.js:** the same native lib shipped as a prebuilt N-API `.node`
  addon. Node `Node` class wraps the same FFI.

All three SDKs:
1. Pull a published prebuilt artifact through their language's native
   package channel.
2. Link/load it at consumer build/install time.
3. Expose a `Node` API that runs a full in-process mesh node, with
   local serving, through that prebuilt lib.

**There is no separate daemon, no child process, no out-of-process IPC.**
The native runtime lives inside the consumer's own binary.

## What Rust gets today

Looking at `main`:

- 11 pure-Rust crates listed in `scripts/publish-crates.sh`, no native
  code. `mesh-llm-api-server` is the SDK entrypoint — currently feature-
  less, no `build.rs`, no link path to anything native.
- The publish chain reaches crates.io but breaks on a new-crate-name rate
  limit (HTTP 429) part-way through, so even the pure-Rust SDK entrypoint
  has never landed on crates.io yet.
- The release pipeline already produces `libmeshllm_ffi.{dylib,so,dll}`
  via `scripts/package-native-sdk.sh` (`build_native_sdk_runtime` matrix
  job, currently only `macos-aarch64-metal` and `linux-x86_64-cpu`).
- The release pipeline already wraps that into a cargo crate via
  `scripts/package-native-sdk-crate.sh` — output is a real `.crate` file
  with `links = "meshllm_native_runtime"` and `build.rs` exporting
  `DEP_MESHLLM_NATIVE_RUNTIME_*` paths.
- That `.crate` file is uploaded as a **GitHub release asset, not
  published to crates.io.** No `cargo publish` step exists for it.

So we already build the right kind of artifact (a published cargo crate
carrying prebuilt `libmeshllm_ffi`). We just don't push it to crates.io
and `mesh-llm-api-server` doesn't depend on it.

That's the gap. Closing it is this plan.

## Plan

A Rust consumer wants the symmetric Swift/Kotlin experience. The cargo-native
way to deliver "a published prebuilt native lib that gets pulled at build
time" is **a published cargo crate carrying the prebuilt lib**, with the SDK
crate depending on it through a target/feature selector.

### Crate layout

The same conceptual split Swift uses (`MeshLLMFFI.xcframework` separate
from `MeshLLM` Swift code) maps cleanly to two cargo crates:

- `mesh-llm-native-sdk-<os>-<arch>-<backend>` — wrapper crate, one per
  platform/backend cell. Carries the prebuilt `libmeshllm_ffi.{dylib,so,dll}`
  inside the crate, `links = "meshllm_native_runtime"`, `build.rs` extracts
  and emits link directives. Already produced by
  `package-native-sdk-crate.sh`.
- `mesh-llm-api-server` — the SDK entrypoint. Depends on the matching
  wrapper crate via target-conditional dependency. Already published
  (or will be once the publish chain is fixed).

The naming and layout already exist — they just don't reach crates.io.

### Backend selection

Backend is selected at compile time via cargo features on
`mesh-llm-api-server`, mirroring how `install.sh` picks a flavor:

```toml
mesh-llm-api-server = { version = "0.66", features = ["native-metal"] }
# or "native-cpu", "native-cuda", "native-rocm", "native-vulkan"
```

Each feature pulls in exactly one matching wrapper crate, gated by
`cfg(target_os, target_arch)`. Mutually exclusive — exactly one
`native-*` feature may be enabled.

### Phased execution

#### Phase 1 — Unbreak the publish chain

Without this, nothing else on the plan can land on crates.io.

The v0.66.0 release publish failed at `model-artifact` with a crates.io
HTTP 429 "too many new crates in a short period." The publish script
publishes serially with a 30s sleep between crates; that wasn't enough
once the chain hit consecutive never-before-published crate names.

Two fixes, either is enough:

1. Add retry-with-exponential-backoff to `scripts/publish-crates.sh`
   when `cargo publish` exits with a 429. Cheap and self-contained.
2. Request a new-crate-publish rate limit increase from the crates.io
   team for the publishing account. Standard request, takes days.

Recommend doing both — script change lands fast, registry-side limit
prevents recurrence as we add more crates.

Deliverable: a release run completes the publish chain. `mesh-llm-api-server`
lands on crates.io for the first time.

#### Phase 2 — Expand the native artifact matrix to match the release matrix

Today `build_native_sdk_runtime` builds `libmeshllm_ffi` for only two
cells (`macos-aarch64-metal`, `linux-x86_64-cpu`). The standalone-binary
release matrix is much wider — Linux CPU/CUDA/CUDA-Blackwell/ROCm/Vulkan/ARM,
Windows CPU/Vulkan/CUDA/ROCm, macOS arm64.

For Rust consumers to have parity with the binary matrix, the native SDK
matrix must grow. Two ways to do this:

1. **Fold `libmeshllm_ffi` production into the existing per-cell
   `build` matrix job.** The cmake build of patched llama.cpp dominates
   each cell. Adding a second `cargo build -p mesh-llm-ffi
   --no-default-features --features host,embedded-runtime` after the
   existing `cargo build -p mesh-llm` reuses the cmake outputs and most
   of cargo's incremental dep cache. Net cost per cell: single-digit
   minutes. Removes the duplicated `build_native_sdk_runtime` matrix job
   entirely — overall saves CMake compute.
2. Keep `build_native_sdk_runtime` as a separate matrix job but expand
   it to all cells. Simpler diff but does redundant cmake work.

Recommend option 1.

Deliverable: every release produces `libmeshllm_ffi` for every supported
(platform, backend) cell.

#### Phase 3 — Publish the wrapper crates to crates.io

`scripts/package-native-sdk-crate.sh` already produces real
crates.io-ready `.crate` files. Currently they upload as GitHub release
assets only. To reach crates.io:

1. Reserve crate names on crates.io. One-time `cargo publish` of an
   empty `0.0.0` stub per name, claiming ownership.
2. Request a per-crate publish-size limit increase from crates.io.
   Default is 10 MiB; the smallest backend (`metal`) is ~140 MB
   unstripped, ~80–100 MB stripped, ~60 MB compressed in the `.crate`.
   CUDA is larger. Without this, the crates simply won't accept upload.
   Standard request, takes days; **start it as soon as Phase 2 begins
   producing the artifacts at all cells**.
3. Extend `scripts/publish-crates.sh` to publish each wrapper crate
   after the existing chain. Gate on real-release only (no dry-run can
   verify wrappers that depend on prebuilt artifacts).
4. Wire the release workflow's `publish_crates` job to feed the
   wrapper-crate `.crate` files in from the build matrix's
   upload-artifact step.

Deliverable: every (os, arch, backend) wrapper crate is on crates.io at
each release version. A consumer can `cargo add
mesh-llm-native-sdk-macos-aarch64-metal` directly and see it resolve.

#### Phase 4 — Wire `mesh-llm-api-server` to consume them

This is the consumer-facing change.

1. Add `native-cpu`, `native-metal`, `native-cuda`, `native-rocm`,
   `native-vulkan` features on `mesh-llm-api-server`.
2. Each feature pulls the matching wrapper crate as a target-conditional
   optional dependency:

   ```toml
   [target.'cfg(all(target_os = "macos", target_arch = "aarch64"))'.dependencies]
   mesh-llm-native-sdk-macos-aarch64-metal = { version = "0.66", optional = true }

   [features]
   native-metal = ["dep:mesh-llm-native-sdk-macos-aarch64-metal"]
   # ...
   ```

3. Add `mesh-llm-api-server/build.rs` that reads
   `DEP_MESHLLM_NATIVE_RUNTIME_LIB_DIR` and
   `DEP_MESHLLM_NATIVE_RUNTIME_LIBRARY` from the wrapper crate and emits
   `cargo:rustc-link-search` + `cargo:rustc-link-lib=dylib=meshllm_ffi`.
4. Implement the Rust glue that calls into the UniFFI C ABI exposed by
   the prebuilt `libmeshllm_ffi`. This is the same C ABI Swift, Kotlin,
   and Node already consume — we ship one set of FFI bindings and reuse.
5. Surface the existing `MeshNode::builder()` / `MeshClient` API through
   the prebuilt path, identical to what consumers see today on the
   workspace `host-runtime` development path.
6. Add a `compile_error!` guard preventing multiple `native-*` features.
7. Document in `docs/SDK.md`: pick exactly one `native-*` feature for
   your target platform. Mirror the Swift "one line in Package.swift"
   ergonomics.
8. Add `examples/cargo-consumer-native/` outside the workspace —
   resolves `mesh-llm-api-server` from crates.io, exercises
   `MeshNode::builder().build()?.start()`, asserts the in-process node
   actually joins a mesh.
9. CI: build the example on each supported platform after each release
   publish.

Deliverable: a Rust app does

```toml
mesh-llm-api-server = { version = "0.66", features = ["native-metal"] }
```

…and gets the same in-process mesh node Swift/Kotlin consumers get today.

## Risks and what we're explicitly accepting

- **Wrapper-crate size on crates.io.** Each backend wrapper is tens to
  hundreds of MiB. crates.io will need a size limit increase per crate.
  This is the same conversation Swift and Kotlin sidestep because their
  registries (SwiftPM via `.binaryTarget` URLs, Maven) have no equivalent
  size limit. It's the one real friction point Rust introduces; it is
  tractable, not blocking.
- **Network at `build.rs` time?** No. The wrapper crates carry the
  prebuilt bytes *inside the crate payload*, not via download in
  `build.rs`. A consumer's `cargo build` works offline once cargo has
  cached the wrapper crate, same as any other crate. This was an option
  earlier in the design discussion; we are not taking it. Cargo-native
  means cargo-native: the bits are in the crate.
- **Per-release maintenance.** Adding new backends or platforms now
  means adding a new wrapper crate to the publish chain. We accept that;
  it mirrors what every other language SDK already does (each platform's
  prebuilt is its own thing).

## Out of scope

- Out-of-process consumption (launching the standalone `mesh-llm`
  binary from a Rust app and talking HTTP). That's already trivially
  possible — sprout or anyone else can install `mesh-llm` via
  `install.sh` and use `mesh-llm-api-client` against the local HTTP
  port. We don't need a plan for it; if someone wants that, they can do
  it today.
- Source-build from crates.io (the `-sys` idiom — publish
  `mesh-llm-host-runtime`, `skippy-ffi`, etc., and build llama.cpp on
  consumer machines). Not the model Swift/Kotlin use; not the model
  we're matching.
- Distro-packager-friendly source-only builds. Not the model
  Swift/Kotlin offer; not the model we're matching.

## Today's state, for reference

What's actually on crates.io (verified by searching the registry):

```
mesh-llm-client      0.65.1   (older release; chain hit 429 on 0.66.0)
mesh-llm-identity    0.66.0
mesh-llm-protocol    0.66.0
mesh-llm-routing     0.66.0
mesh-llm-types       0.66.0
model-ref            0.66.0
mesh-api             0.65.1   (stale; legacy name)
```

What's in `scripts/publish-crates.sh` but missing from crates.io:

```
model-artifact, model-hf, mesh-llm-client@0.66.0,
mesh-llm-api-client, mesh-llm-node, mesh-llm-api-server
```

These are the crates the v0.66.0 publish run never reached, due to the
HTTP 429 on `model-artifact`. They have no published version. Phase 1
fixes this.

What's on GitHub releases (v0.66.0):

```
mesh-llm-{aarch64-apple-darwin,
          aarch64-unknown-linux-gnu,
          x86_64-unknown-linux-gnu,
          x86_64-unknown-linux-gnu-vulkan,
          x86_64-unknown-linux-gnu-rocm,
          x86_64-unknown-linux-gnu-cuda,
          x86_64-unknown-linux-gnu-cuda-blackwell,
          x86_64-pc-windows-msvc,
          x86_64-pc-windows-msvc-vulkan,
          x86_64-pc-windows-msvc-rocm,
          x86_64-pc-windows-msvc-cuda}.tar.gz / .zip
```

Each contains exactly one file: the standalone `mesh-llm` executable.
Nothing for cargo consumption. The xcframework that v0.65.x shipped
(`MeshLLMFFI.xcframework.zip`, ~168 MB) regressed on v0.66.x — likely
related to the v0.66.0 workflow shape predating PR #634, which
restructured SDK artifact production. Phase 2 should produce a parallel
artifact for Rust (`libmeshllm_ffi` per cell) alongside restoring the
Swift one.
