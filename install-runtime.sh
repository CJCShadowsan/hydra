#!/usr/bin/env bash

set -euo pipefail

source_runtime_installer() {
    local source_path="${BASH_SOURCE[0]-}"
    local script_dir

    if [[ -n "$source_path" && "$source_path" == */* ]]; then
        script_dir="$(cd "$(dirname "$source_path")" && pwd)"
        if [[ -f "$script_dir/install.sh" ]]; then
            # shellcheck source=install.sh
            . "$script_dir/install.sh"
            return 0
        fi
    fi

    local repo="${MESH_LLM_INSTALL_REPO:-Mesh-LLM/mesh-llm}"
    local ref="${MESH_LLM_INSTALL_REF:-main}"
    local installer_file

    if ! command -v curl >/dev/null 2>&1; then
        echo "error: required command not found: curl" >&2
        exit 1
    fi
    if ! command -v mktemp >/dev/null 2>&1; then
        echo "error: required command not found: mktemp" >&2
        exit 1
    fi

    installer_file="$(mktemp)"
    curl -fsSL "https://raw.githubusercontent.com/${repo}/${ref}/install.sh" -o "$installer_file"
    # shellcheck source=/dev/null
    . "$installer_file"
    rm -f "$installer_file"
}

source_runtime_installer

if [[ "${BASH_SOURCE[0]-}" == "$0" || ( -z "${BASH_SOURCE[0]-}" && "$0" == "bash" ) ]]; then
    main "$@"
fi
