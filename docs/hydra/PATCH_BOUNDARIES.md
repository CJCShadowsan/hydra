# Patch Boundaries

Hydra features should stay additive and easy to review against upstream.

## Hydra-Owned Areas

- `crates/hydra/`: network cost snapshots, SLO scheduler scoring,
  artifact placement providers, exact cache identity checks, and VAST trigger
  payloads.
- `docs/hydra/`: Hydra operation notes and sync process.
- `scripts/hydra/`: upstream drift and sync helpers.
- `.github/workflows/hydra-upstream-drift.yml`: Hydra drift reporting.

## Narrow Integration Points

- Host runtime status adds Hydra snapshots to `/api/status`.
- OpenAI routing calls the Hydra scheduler only after existing health,
  capability, media, context, and affinity filters have produced candidates.
- Attempt recording mirrors existing route metrics into the Hydra network-cost
  collector.
- Placement API routes are local management endpoints under `/api/placement/*`.
- CLI placement commands call those local management endpoints.
- Config accepts `[scheduler]` and `[placement]` sections while keeping defaults
  non-invasive.
- VAST-specific behavior stays behind explicit placement trigger config and
  sends webhook/DataEngine payloads only after namespace manifests commit.

## Merge Rules

- Prefer adding code to `hydra` over expanding central routing files.
- Keep active routing behavior feature-gated. Shadow mode is the safe default.
- Never require VAST credentials for normal CI. POSIX temp-directory tests and
  mock/object-store tests must cover provider semantics.
- Keep remote KV/cache reuse exact. If identity fields do not match, recompute.
- If upstream changes the routing, status, config, or CLI surfaces touched here,
  resolve conflicts by preserving upstream behavior first, then re-applying the
  Hydra adapter call.
