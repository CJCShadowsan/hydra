# Hydra

Hydra is the fork programme for network-aware scheduling and artifact
placement. Keep Hydra changes here whenever possible so the fork can continue
to merge upstream `Mesh-LLM/mesh-llm` with a small conflict surface.

Current modules:

- `network_cost`: local passive metrics, bounded snapshots, and compact
  advisory hints for future gossip/control-plane exchange.
- `scheduler`: shadow/active SLO-aware scoring for target selection.
- `placement`: POSIX/S3 namespace publishing, manifests, checksum validation,
  pin/evict/status tracking, and exact cache identity checks.
- `vast`: configurable VAST DataEngine/webhook trigger payloads fired after
  artifact manifests commit.
