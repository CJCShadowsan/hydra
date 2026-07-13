# VAST-Backed Placement

Hydra integrates VAST through documented namespace surfaces:

- POSIX-mounted DataSpace: use `mesh-llm placement prefetch ... --posix-root /vast/...`.
- S3-compatible namespace: use the `S3NamespaceProvider` in `hydra` with
  an endpoint, bucket, and optional prefix.

It also includes an optional VAST-specific trigger adapter. The trigger is
deliberately configured as a webhook/DataEngine endpoint because Hydra should
not hard-code undocumented VAST-only APIs. Mesh LLM publishes the artifact into
the POSIX/S3 namespace first, commits a manifest with checksum and compatibility
metadata, then POSTs a trigger payload that a VAST DataEngine workflow or site
automation can consume to move or materialize the artifact at distant locations.

## CLI

```bash
mesh-llm placement prefetch layer_package qwen3-stage-0 /local/cache/stage-0 \
  --posix-root /vast/global/mesh-llm \
  --vast-trigger-endpoint https://vast-dataengine.example.internal/mesh-llm/ship \
  --vast-tenant acme-ai \
  --vast-dataspace prod-dataspace \
  --vast-source-namespace /vast/global/mesh-llm \
  --vast-destination-namespace /vast/site-b/mesh-llm \
  --vast-target-site site-b
```

## Local API

```bash
curl -X POST http://127.0.0.1:3131/api/placement/prefetch \
  -H 'content-type: application/json' \
  -d '{
    "kind": "layer_package",
    "artifact_id": "qwen3/stage-0@sha256:...",
    "source_path": "/local/cache/stage-0",
    "provider": { "type": "posix", "root": "/vast/global/mesh-llm" },
    "vast_trigger": {
      "mode": "data_engine_webhook",
      "endpoint": "https://vast-dataengine.example.internal/mesh-llm/ship",
      "tenant": "acme-ai",
      "dataspace": "prod-dataspace",
      "source_namespace": "/vast/global/mesh-llm",
      "destination_namespace": "/vast/site-b/mesh-llm",
      "target_sites": ["site-b"]
    }
  }'
```

The trigger request includes:

- artifact kind and id
- committed manifest, checksum, byte size, TTL, and compatibility identity
- provider location without secrets
- source path and VAST tenant/DataSpace/site hints
- operator metadata from the placement request

If the namespace publish succeeds but the trigger endpoint fails, Mesh LLM records
the placement operation as failed with the manifest attached. Operators can still
recover from the committed artifact, but the failed status makes the missed
shipment visible.

## Config

```toml
[placement]
provider = "posix"
posix_root = "/vast/global/mesh-llm"
vast_trigger_endpoint = "https://vast-dataengine.example.internal/mesh-llm/ship"
vast_tenant = "acme-ai"
vast_dataspace = "prod-dataspace"
vast_source_namespace = "/vast/global/mesh-llm"
vast_destination_namespace = "/vast/site-b/mesh-llm"
vast_target_sites = ["site-b"]
vast_trigger_timeout_secs = 30
```

Every publish writes through a staging location, validates a BLAKE3 checksum,
commits a manifest, and publishes into the final namespace path. Cache imports
must pass exact compatibility checks before KV/recurrent/activation state can be
used.
