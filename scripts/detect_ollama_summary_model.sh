#!/bin/bash
set -euo pipefail

if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama is not installed or not on PATH." >&2
    exit 1
fi

preferred="${1:-}"
models="$(ollama list | awk 'NR > 1 {print $1}')"

if [ -n "$preferred" ]; then
    if echo "$models" | grep -qx "$preferred"; then
        echo "$preferred"
        exit 0
    fi
    echo "Requested model '$preferred' is not installed in Ollama." >&2
    exit 1
fi

selected="$(echo "$models" | grep -E '^gemma' | head -n 1 || true)"
if [ -z "$selected" ]; then
    echo "No compatible local Gemma-family Ollama model was found." >&2
    exit 1
fi

echo "$selected"
