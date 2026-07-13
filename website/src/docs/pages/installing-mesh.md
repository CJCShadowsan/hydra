---
title: Installing Hydra
---

# Installing Hydra

Hydra runs on macOS, Linux, and Windows. Choose your platform for detailed install instructions.

## Choose your platform

- [Installing on macOS](/docs/pages/installing-macos/) (Apple Silicon, Homebrew)
- [Installing on Linux](/docs/pages/installing-linux/) (platform details)
- [Installing on Windows](/docs/pages/installing-windows/) (platform details)

## Current install path

Hydra packaged installers are not published yet. Install from source:

```sh
cargo install --git https://github.com/CJCShadowsan/hydra.git --package mesh-llm --bin hydra
```

After install, run `hydra setup` to finish runtime configuration and service
setup.

## Verify the install

```sh
hydra --version
```

## Next step

Run `hydra setup` to finish machine setup, then follow the [Quickstart](/docs/pages/quickstart/) to start a private node and open the console.

## Uninstall

```sh
cargo uninstall mesh-llm
```

The inherited runtime configuration remains under `~/.mesh-llm` for now.

## See also

- [Hardware support](/docs/pages/hardware-support/)
- [Updating Mesh](/docs/pages/updating-mesh/)
