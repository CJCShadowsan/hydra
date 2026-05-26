# mesh-llm-api-server

`mesh-llm-api-server` is the public Rust SDK crate for applications that embed a Mesh
node with model management and local serving. Client-only applications should
use `mesh-llm-api-client`.

The target public concept is `MeshNode`: one embedded node that can:

- join a mesh and consume inference
- search, inspect, download, install, delete, and clean up local models
- load and unload local models for serving
- observe connection, model, serving, and request lifecycle events

SDK layering:

- `crates/mesh-client/` implements the low-level client behavior
- `crates/mesh-llm-api-client/` exposes the client-only Rust SDK surface
- `crates/mesh-llm-api-server/` exposes the node Rust SDK surface and re-exports the
  client types for compatibility
- `crates/mesh-llm-node/` owns embeddable model management and serving
  orchestration as it is extracted from the host runtime. Serving is an
  in-process SDK boundary; REST management remains an external-daemon adapter.
- `crates/mesh-llm-host-runtime/` provides the reference `ServingController`
  implementation by attaching `MeshApi` to `MeshNodeBuilder` and forwarding
  load/unload to the runtime-control loop.
- `crates/mesh-llm-ffi/` wraps `crates/mesh-llm-api-server/` for Swift, Kotlin, and other native
  bindings

Serving APIs use model refs for load, explicit model-or-instance unload
targets, drain/force unload options, rich served-model status, and typed
high-level serving errors.

If an API is meant for client-only app integration, it belongs in
`mesh-llm-api-client`. If it requires model management or local serving, it
belongs in `mesh-llm-api-server`.

## Running the full mesh-llm runtime in-process (`host-runtime` feature)

For applications that want to run **exactly what `mesh-llm serve` /
`mesh-llm client` does** — not just consume mesh inference, but be the
running node — enable the `host-runtime` feature:

```toml
mesh-llm-api-server = { version = "0.66.0", features = ["host-runtime"] }
```

Then call `run_serve(MeshServeSpec { ... })`:

```rust
use mesh_llm_api_server::{run_serve, MeshServeSpec};

run_serve(MeshServeSpec {
    client: true,
    auto: true,
    relays: vec!["https://public.example/".into()],
    port: Some(9337),
    console_port: Some(3131),
    headless: true,
    max_vram_gb: Some(0.0),
    ..Default::default()
})
.await?;
```

(Gated iroh-relay support — per-relay bearer tokens via `--relay-auth
URL=TOKEN` and a `relay_auths` field on `MeshServeSpec` — lives on the
separate gated-relay PR. Once that lands, this snippet will gain the
`HashMap` shape used by `mesh-llm`'s CLI today.)

This drives the same `runtime::run_with_args` entry point the binary
uses. You get auto-discovery, election, tunnel manager, OpenAI HTTP
proxy on `--port`, management console on `--console`, local model
serving (when configured), plugin host — the entire mesh-llm runtime
inside your process.

`MeshNode::builder()` (`host-runtime` feature also required for the
fine-grained options like `.relay(...)`) is the
composable alternative for apps that want to wire pieces themselves
rather than running the whole orchestration. See `docs/SDK.md` for the
full comparison.
