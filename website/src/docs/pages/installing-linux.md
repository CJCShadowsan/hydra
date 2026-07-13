---
title: Installing on Linux
---

# Installing on Linux

Install Hydra on every Linux machine that should serve a model or call into a mesh.

## Quick install

```sh
cargo install --git https://github.com/CJCShadowsan/hydra.git --package mesh-llm --bin hydra
```

Hydra source installs currently require Rust, `cmake`, and the native build
dependencies for your selected runtime backend.

Check the install:

```sh
hydra --version
```

## Release installers

Hydra release installers are not published yet. The inherited shell installer
will be reworked once Hydra publishes its own signed release archives.

## Next step

Run `hydra setup` to finish machine setup. See the [CLI guide](/docs/pages/CLI/) for the setup flags.

## Uninstall

```sh
cargo uninstall mesh-llm
```

On Linux, uninstall disables the per-user systemd unit when present, removes
setup-owned service files, removes the native-runtime cache, and removes the
The inherited runtime configuration remains under `~/.mesh-llm` for now.

## See also

- [Installing on macOS](/docs/pages/installing-macos/)
- [Installing on Windows](/docs/pages/installing-windows/)
- [Hardware support](/docs/pages/hardware-support/)
- [Updating Mesh](/docs/pages/updating-mesh/)
