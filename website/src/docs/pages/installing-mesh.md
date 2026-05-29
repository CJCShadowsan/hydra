# Install

Install Mesh on every machine that should serve a model or call into a mesh.

## Recommended install

macOS or Linux:

```sh
curl -fsSL https://mesh-llm.cloud/install.sh | bash
```

Windows PowerShell:

```powershell
irm https://mesh-llm.cloud/install.ps1 | iex
```

Open a new terminal after install if the installer added Mesh to your `PATH`.

Check the install:

```sh
mesh-llm --version
```

## What the installer does

The installer:

- detects the best release bundle for this machine
- downloads the matching Mesh release
- installs the `mesh-llm` binary
- adds the install directory to your user `PATH` when needed

Default install locations:

| Platform | Default location |
|---|---|
| macOS/Linux | `~/.local/bin` |
| Windows | `%LOCALAPPDATA%\mesh-llm\bin` |

## Force a flavor

Most users should let the installer auto-detect. Force a flavor when auto-detection is wrong, when you are preparing a machine image, or when you intentionally want CPU/Vulkan instead of a vendor GPU backend.

macOS/Linux:

```sh
curl -fsSL https://mesh-llm.cloud/install.sh | MESH_LLM_INSTALL_FLAVOR=vulkan bash
```

Windows PowerShell:

```powershell
$env:MESH_LLM_INSTALL_FLAVOR = "vulkan"
irm https://mesh-llm.cloud/install.ps1 | iex
```

Supported release flavors:

| Platform | Flavors |
|---|---|
| macOS Apple Silicon | `metal` |
| Linux x86_64 | `cuda-blackwell`, `cuda`, `rocm`, `vulkan`, `cpu` |
| Linux ARM64 | `cpu` |
| Windows x86_64 | `cuda-blackwell`, `cuda`, `rocm`, `vulkan`, `cpu` |

## Advanced install

Install the latest prerelease:

```sh
curl -fsSL https://mesh-llm.cloud/install.sh | bash -s -- --pre-release
```

Windows PowerShell:

```powershell
$env:MESH_LLM_INSTALL_PRERELEASE = "1"
irm https://mesh-llm.cloud/install.ps1 | iex
```

Install somewhere else:

```sh
curl -fsSL https://mesh-llm.cloud/install.sh | MESH_LLM_INSTALL_DIR="$HOME/bin" bash
```

Windows PowerShell:

```powershell
$env:MESH_LLM_INSTALL_DIR = "$HOME\bin"
irm https://mesh-llm.cloud/install.ps1 | iex
```

## Next step

Run the [Quickstart](/docs/pages/quickstart/) to start a private node and open the console.
