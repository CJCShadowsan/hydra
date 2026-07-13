---
title: Installing on Windows
---

# Installing on Windows

Install Hydra on every Windows machine that should serve a model or call into a mesh.

## Quick install

Hydra packaged Windows installers are not published yet. Install from source
with Rust/Cargo:

```powershell
cargo install --git https://github.com/CJCShadowsan/hydra.git --package mesh-llm --bin hydra
```

Check the install:

```powershell
hydra.exe --version
```

## Release installers

The inherited PowerShell installer will be reworked once Hydra publishes its
own signed Windows release archives.

## Next step

Run `hydra.exe setup` to finish machine setup. See the [CLI guide](/docs/pages/CLI/) for the setup flags.

## Uninstall

```powershell
cargo uninstall mesh-llm
```

The inherited runtime configuration remains under `%USERPROFILE%\.mesh-llm` for now.

## See also

- [Installing on macOS](/docs/pages/installing-macos/)
- [Installing on Linux](/docs/pages/installing-linux/)
- [Hardware support](/docs/pages/hardware-support/)
- [Updating Mesh](/docs/pages/updating-mesh/)
