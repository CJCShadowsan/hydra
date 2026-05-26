# CLI on top of the Rust SDK

## Status: design exploration, follow-on to `RUST_NATIVE_SDK.md`

## Goal

Make `mesh-llm` (the shipped binary) a normal consumer of
`mesh-llm-api-server` — the same Rust SDK an external app like sprout
consumes — rather than reaching into host-runtime internals.

Reason: today there are *two* parallel surfaces for the same domain
behaviour (one inside `mesh-llm-host-runtime::cli::commands::*`, the
other on `mesh-llm-api-server`). They drift. The SDK is the contract
external consumers are starting to depend on. If the CLI rides on the
same contract, the SDK stays first-class and the contract stays honest.

This is also the natural next step after the
`mesh-llm-api-server`-driven trial in `examples/rust-sdk-trial/`. That
example proved an external app can do the work; this proposal asks the
*shipped binary* to do the same.

## What's actually shared today vs duplicated

### Already lined up

`mesh-llm-api-server` exposes a sane surface for the user-facing commands:

| CLI command                  | SDK equivalent (already exists)                                       |
| ---------------------------- | --------------------------------------------------------------------- |
| `discover --auto`            | `mesh_llm_api_server::create_auto_node(owner, PublicMeshQuery)`       |
| `discover` (list only)       | `mesh_llm_api_server::discover_public_meshes(query)`                  |
| `download <name>`            | `MeshNode::models().download(model_ref)`                              |
| `models search/list/details` | `MeshNode::models().search() / .list() / .details()`                  |
| `models delete`              | `MeshNode::models().delete(...)`                                      |
| `load <name>` (in-proc)      | `MeshNode::serving().load(model_ref, opts)`                           |
| `unload <name>` (in-proc)    | `MeshNode::serving().unload(target, opts)`                            |
| `status` (in-proc)           | `MeshNode::serving().status()` / `MeshNode::status()`                 |
| `auth init/status/...`       | (still bespoke — see below)                                           |
| `serve` / `client`           | `run_serve(MeshServeSpec)` on `micn/relay-auth-fix` / PR #641         |
| `goose` / `claude` / `pi` / `opencode` | Mostly orchestration glue around an existing mesh-llm HTTP    |
|                              | endpoint — could call into `MeshNode::inference().list_models()` etc. |

### Stuff the CLI does that the SDK doesn't expose yet

- **HTTP-shaped CLI commands.** `mesh-llm load <name> --port 3131` and
  `mesh-llm unload` and `mesh-llm status` already speak to a *running*
  mesh-llm via the management API on `:3131`. They don't drive an
  in-process node. The SDK doesn't need to absorb this — these stay as
  thin reqwest clients hitting a localhost API. They're fine as-is.
- **`auth`.** Owner identity / keystore / node certificates. SDK has
  `OwnerKeypair` and identity types, but no `init` / `sign-node` /
  `verify-node` orchestration. Either: (a) extend the SDK with an
  `Auth` module that exposes these operations, or (b) keep auth as a
  bespoke binary-side subcommand because nobody else needs it.
- **`gpus`.** Local hardware enumeration / benchmark cache. Belongs in
  the `mesh-llm-system` crate already. SDK could re-export.
- **`update`.** Self-update of the shipped binary. Stays bespoke; no
  external consumer wants this.
- **`model-prepare`.** HF Jobs orchestration. Stays bespoke; it's a
  developer-tools-for-the-mesh-llm-team thing.
- **`blackboard`.** Cross-mesh shared notes. Spans CLI client mode,
  MCP server mode, and post/search. Probably belongs on the SDK as
  `MeshNode::blackboard()` so external Rust agents can post to it.
- **`benchmark`.** Internal-only; stays bespoke.
- **`stop`.** Talks to all local mesh-llm instances via runtime
  metadata. Bespoke binary concern, not consumer-facing.
- **Plugin install/list, integrations (`goose`/`claude`/...).**
  Wrapper-binary launchers. Could call SDK to fetch
  models/endpoint info, then exec the agent harness. Mostly fine.

### The 7,200-line CLI module

`crates/mesh-llm-host-runtime/src/cli/` is ~7,200 lines today:

```
   1593  src/cli/mod.rs                 — clap types, top-level dispatch
   1694  src/cli/commands/integrations.rs — goose/claude/pi/opencode
    923  src/cli/commands/auth.rs        — owner identity, signing
    642  src/cli/commands/model_package.rs — HF Jobs (model-prepare)
    594  src/cli/commands/runtime.rs     — runtime control via :3131 HTTP
    363  src/cli/commands/gpus.rs        — local GPU enumeration
    277  src/cli/commands/mod.rs         — dispatcher
    251  src/cli/commands/discover.rs    — nostr discovery
    204  src/cli/models.rs               — output helpers for models
    166  src/cli/commands/blackboard.rs  — blackboard post/search/MCP
    109  src/cli/terminal_progress.rs    — progress bars
     94  src/cli/runtime.rs              — runtime command enum
     94  src/cli/pager.rs                — output paging
     47  src/cli/commands/benchmark.rs   — bench dispatch
     44  src/cli/commands/plugin.rs      — plugin list/install
     42  src/cli/benchmark.rs            — bench types
     38  src/cli/commands/download.rs    — download dispatch
     18  src/cli/commands/update.rs      — update dispatch
     14  src/cli/shell.rs                — shell-detection helpers
```

Of that:

- **~3,500 lines** are *clap parsing + dispatcher + output formatting*.
  Stays as-is — that's the CLI shell's job, not the SDK's.
- **~1,700 lines** (`integrations.rs`) are launcher logic for
  goose/claude/pi/opencode. Could thin if SDK exposed `auto-join +
  list-models + get-endpoint` cleanly; not a big domain win.
- **~900 lines** (`auth.rs`) are the owner-identity surface. Real
  candidate to move into the SDK (or a sibling SDK crate) so external
  consumers can also do `OwnerKeypair::init`, `sign_node`, etc.
- **~650 lines** (`runtime.rs`) are HTTP-to-management-API plumbing.
  Stays as-is — it's a remote-control client.
- **~400 lines** across `discover.rs`, `download.rs`, models,
  `blackboard.rs` are direct calls into `crate::network::*`,
  `crate::models::*`, `crate::mesh::*` host-runtime internals. **These
  are the duplication-with-SDK candidates.**

So the practical win from "CLI on top of SDK" is concentrated in a few
hundred lines of host-runtime internals being replaced by SDK API
calls — not a 7K-line rewrite.

## Proposed shape

Three changes, each tractable on its own:

### Change 1 — Domain commands route through the SDK

**Status: started on this branch.** `discover` is done; `download` and
the `models` dispatcher are blocked on SDK extensions (see below).

#### Done: `discover`

`cli::commands::discover::run_nostr_discover` previously called
`nostr::discover(&relays, &filter, None)` directly. Now it calls
`mesh_llm_api_server::discover_public_meshes(PublicMeshQuery { ... })`
and lifts each `PublicMesh` back into the host-runtime
`DiscoveredMesh` shape via a small local helper so the existing CLI
display + scoring + `Display` impl stay untouched.

Net effect: one fewer copy of "how to talk to Nostr." Same UX. The
`mesh-llm discover [--name|--model|--min-vram|--region]` commands
verified against the live public mesh, same output.

Diff: ~40 lines in one file (`cli/commands/discover.rs`), plus a
standing `lift_public_mesh` helper. No core type changes, no Cargo.toml
changes — `mesh-llm-host-runtime` already depended on
`mesh-llm-api-server`.

#### Blocked: `download`

The CLI's `download` command uses
`crate::models::download_model_ref_with_progress_details` which prints
a live progress bar to the terminal. The SDK's
`MeshNode::models().download(...)` exists but has no progress callback,
so routing through it would silently lose the progress bar — a real
UX regression.

Unblocker: extend `MeshModels::download` with an optional progress
callback (e.g. `download_with_progress(model_ref, options, |progress|
{ ... })`) before re-pointing the CLI. Separate small SDK PR.

#### Blocked: `models` dispatcher

The CLI's `models` subcommand (`search`, `installed`, `show`, `cleanup`,
`prune`, `delete`, `certify`) reaches deep into host-runtime internals
that have no SDK equivalent today: `installed_model_display_name`,
`layered_package_layer_count_for_path`,
`huggingface_identity_for_path`, `installed_model_capabilities`,
`load_model_usage_record_for_path`, `find_model_path`,
layer-package introspection, certification gate state, etc.

Unblocker: substantially expand `MeshModels` to surface these (or accept
that the CLI `models` subcommand stays bespoke because no external
consumer wants this level of detail). Probably best left until an
external consumer actually asks for it; the SDK shouldn't grow on
speculation.

### Change 2 — `serve` and `client` route through `run_serve(MeshServeSpec)`

Currently the binary's `serve` / `client` flows enter
`runtime::run_with_args(argv)` directly. PR #641 introduced
`run_serve(MeshServeSpec) -> Result<()>` on the SDK as a typed
equivalent. The binary's `main.rs` can construct a `MeshServeSpec` from
the parsed clap struct (instead of forwarding `argv` to
`run_with_args`) and call `run_serve`.

This is a small reshuffle. It doesn't unlock anything new for the SDK
(the spec already wraps the same code path), but it makes the binary a
"first user" of the SDK's main entry point. Visible signal that the
contract is real.

### Change 3 — `auth` moves to the SDK (or stays bespoke, but with a sharper line)

Two paths:

- **(a)** Extract `cli/commands/auth.rs`'s domain logic into a
  `mesh-llm-api-server::auth` module. CLI becomes parse + format only.
  External Rust apps gain `mesh_llm_api_server::auth::init(...)` etc.
- **(b)** Decide `auth` is binary-only (CLI is the only intended
  consumer), and keep it where it is — but mark it clearly as
  "binary-only, intentionally not in the SDK."

Honest assessment: most external consumers (sprout, agent harnesses)
will need *some* form of identity bootstrap. (a) is probably right,
but it's the biggest single chunk and worth doing last.

## What's explicitly out of scope

- **Web UI / TUI.** Stays in the binary. SDK doesn't need to know
  about ratatui or the embedded React app.
- **`update`, `model-prepare`, `benchmark`, `stop`, `gpu enumerate`.**
  Binary-only concerns; no SDK consumer wants these. Keep where they
  are.
- **HTTP-to-management-API CLI commands.** `mesh-llm load --port 3131`
  is a thin HTTP client; it doesn't need to go through the SDK.

## Sequencing

Recommend doing this in the same order as the underlying SDK work
matures:

1. **First, land #690** (the static-archive trial + design) and the
   gated-relay split + the publish-chain fix (#691). This is the
   prerequisite — without `mesh-llm-api-server` actually reaching
   crates.io, the "CLI uses the SDK" narrative is internal-only.
2. Then: change 1 (domain commands). One or two PRs covering
   `discover`, `download`, `models`, `blackboard`. Each one is a small
   diff with no behavioural change.
3. Then: change 2 (binary `serve`/`client` via `run_serve`). One PR.
4. Last: change 3 (auth). Possibly broken into "extract auth into the
   SDK" + "update CLI to use it" as two PRs.

After all three: `mesh-llm` the binary is a thin wrapper around
clap-parsing + `mesh-llm-api-server` calls + TUI/UI/log glue. The
~400 lines of duplicated domain logic in `cli/commands/` go away.

## Risks

- **Premature constraint on the SDK.** Today's SDK was shaped by
  external consumers (Swift/Kotlin/Node bindings, the
  `examples/rust-sdk-trial/` consumer). Making the binary depend on
  the same surface means every CLI-driven change has to round-trip
  through SDK design. Tradeoff: that's the *point* — keeps the SDK
  honest. But it slows quick CLI iteration.
- **Loss of host-runtime-internal access.** Some commands today reach
  into host-runtime internals for things the SDK doesn't expose
  (e.g. mesh ID persistence in `discover.rs`'s `mesh::load_last_mesh_id`).
  These would need SDK accessors. Small expansion, but real.
- **Test coverage churn.** CLI commands have their own dispatch tests.
  Migrating to SDK calls means some of those tests collapse into SDK
  unit tests. Net win, but transitional disruption.

## Why this is worth doing at all

- **The SDK becomes the actual contract.** Today the SDK is a stated
  contract that the binary doesn't itself rely on. Whatever the binary
  needs gets added to host-runtime internals, and the SDK lags.
  Making the binary an SDK consumer flips that asymmetry.
- **External consumers gain parity with the binary.** Whatever
  `mesh-llm discover --auto` does, sprout can do the same way. No
  "well, the binary does X but we never wired it into the SDK."
- **Smaller ongoing maintenance.** One copy of "how to discover a
  public mesh" instead of two. One copy of "how to download a model."
- **Honest cost.** Not a rewrite — concentrated in maybe 400-1000
  lines being deleted and re-pointed at SDK calls, plus
  `auth` if we take that on. Sequenced across three or four PRs.
