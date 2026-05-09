#!/usr/bin/env bash

set -euo pipefail

UI_DIR="${1:?usage: build-ui.sh /path/to/ui}"

# Build the preview console app and copy its dist/ to the crate root so
# build.rs finds it at $CARGO_MANIFEST_DIR/dist unchanged.
PREVIEW_DIR="$UI_DIR/preview"
DIST_DIR="$UI_DIR/dist"
NODE_MODULES_DIR="$PREVIEW_DIR/node_modules"

ui_build_inputs=(
    "$PREVIEW_DIR/package.json"
    "$PREVIEW_DIR/package-lock.json"
    "$PREVIEW_DIR/vite.config.ts"
    "$PREVIEW_DIR/tsconfig.json"
    "$PREVIEW_DIR/tsconfig.app.json"
    "$PREVIEW_DIR/index.html"
    "$PREVIEW_DIR/src"
    "$PREVIEW_DIR/public"
)

dist_has_output() {
    [[ -d "$DIST_DIR" ]] && find "$DIST_DIR" -type f -print -quit | grep -q .
}

ui_build_is_stale() {
    if ! dist_has_output; then
        return 0
    fi

    for path in "${ui_build_inputs[@]}"; do
        [[ -e "$path" ]] || continue
        if find "$path" -type f -newer "$DIST_DIR" -print -quit | grep -q .; then
            return 0
        fi
    done

    return 1
}

npm_install_is_stale() {
    if [[ ! -d "$NODE_MODULES_DIR" ]]; then
        return 0
    fi

    local manifest
    for manifest in "$PREVIEW_DIR/package.json" "$PREVIEW_DIR/package-lock.json"; do
        [[ -e "$manifest" ]] || continue
        if [[ "$manifest" -nt "$NODE_MODULES_DIR" ]]; then
            return 0
        fi
    done

    return 1
}

if ui_build_is_stale; then
    echo "Building mesh-llm UI (preview console)..."
    cd "$PREVIEW_DIR"

    if npm_install_is_stale; then
        export ONNXRUNTIME_NODE_INSTALL_CUDA="${ONNXRUNTIME_NODE_INSTALL_CUDA:-skip}"
        npm ci
    fi

    npm run build

    # Copy dist/ to the crate root where build.rs expects it.
    rm -rf "$DIST_DIR"
    cp -r "$PREVIEW_DIR/dist" "$DIST_DIR"
else
    echo "Skipping mesh-llm UI build; dist is up to date."
fi
