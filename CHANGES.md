# Hydra Changes

Hydra is a downstream fork of `Mesh-LLM/mesh-llm`. It tracks upstream Mesh LLM
while allowing divergence for network-aware low-latency inference.

Initial Hydra divergence:

- Adds a Hydra logo to the README and stores the source SVG in
  `docs/hydra/hydra-logo.svg`.
- Repoints the website source and publishing notes at `hydra-llm.cloud`.
- Adds `hydra` as the primary CLI binary for this fork.
- Namespaces Hydra discovery so `hydra --auto` finds Hydra Nostr and mDNS
  records instead of ordinary upstream Mesh LLM records.
- Adds Hydra-owned passive network cost tracking and compact advisory hints.
- Adds shadow/active SLO-aware scheduler scoring for target selection.
- Adds POSIX/S3 artifact placement with manifests, checksums, pin, evict, and
  status surfaces.
- Adds VAST DataSpace/DataEngine webhook trigger support after manifest commit.
- Adds exact KV/recurrent/activation cache compatibility checks.
- Adds local management API and CLI placement commands.
- Adds Hydra upstream-sync docs, scripts, and drift workflow.
- Documents Hydra performance tuning, including benchmark tuning, scheduler,
  network telemetry, runtime, Skippy, speculative, placement, and VAST knobs.

Upstream tracking:

- `upstream` should point at `https://github.com/Mesh-LLM/mesh-llm.git`.
- `origin` should point at the Hydra repository.
- Syncs should merge upstream into Hydra rather than rebasing shared Hydra
  history.
