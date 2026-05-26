# mesh-llm-native-sdk

Prebuilt native runtime for the Rust mesh-llm SDK. Fetches the matching
`libmeshllm_ffi.{dylib,so,dll}` for the consumer's target platform + selected
backend from the mesh-llm GitHub release, verifies its sha256, and links it
into the consumer's binary.

## Consumer use

Consumers should not depend on this crate directly. Depend on
`mesh-llm-api-server` with the appropriate `native-*` feature:

```toml
mesh-llm-api-server = { version = "0.66", features = ["native-metal"] }
```

The native runtime arrives transparently. No CMake, no GPU SDK, no patched
llama.cpp build on the consumer's machine.

## Override env vars (for local trials and offline builds)

- `MESH_LLM_NATIVE_TARBALL_URL` — `file://` or `https://` URL to a tarball
  produced by `scripts/package-native-sdk.sh`. Useful for testing this
  crate against a locally-built native lib before a GitHub release exists.
- `MESH_LLM_NATIVE_TARBALL_SHA256` — expected hex sha256 of the tarball.
  When set, overrides the `.sha256` sidecar.
- `MESH_LLM_NATIVE_CACHE_DIR` — where to cache downloaded tarballs.
  Defaults to `~/.cache/mesh-llm-native-sdk/<version>/<artifact_id>/`.
