# SDK

This directory contains the language-specific Mesh SDK packages built on top of
the shared native node SDK.

Current SDKs:

- `swift/` for Apple platforms
- `kotlin/` for Android and JVM consumers

These SDK packages should stay thin. Shared node behavior belongs in the Rust
SDK crates:

- `crates/mesh-client/` for the low-level client implementation
- `crates/mesh-llm-api/` for the public Rust SDK API while the node SDK is being
  reworked
- `crates/mesh-llm-node/` for embeddable model management and serving
  orchestration. Serving SDK calls should bind to in-process node
  controllers, not the local REST management API.
- `crates/mesh-llm-ffi/` for the UniFFI/native bridge used by language SDKs

The SDK's long-term public surface is `MeshNode`: one embedded node that can
consume inference from the mesh, manage local models, serve local models, or
combine those roles. See `docs/design/EMBEDDED_CLIENT_ADR.md` for the current
SDK direction.

The customer-facing SDK release gates live in `docs/SDK.md`. SDK changes should
keep Rust, Swift, and Kotlin aligned to that canonical contract: real examples,
polished lifecycle, typed errors, and an honest platform support matrix.

Generated UniFFI bindings and Apple binary artifacts are build outputs, not
source. Do not check in `sdk/*/Generated`, generated `uniffi/mesh_ffi` Kotlin
sources, or `MeshLLMFFI.xcframework`; regenerate them from
`crates/mesh-llm-ffi/src/mesh_ffi.udl` in local builds and CI.

If you add another top-level SDK here, include a `README.md` in that SDK
directory explaining its packaging and public surface.
