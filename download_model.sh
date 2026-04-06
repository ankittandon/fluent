#!/bin/bash
set -euo pipefail

MODELS_DIR="models"
SUMMARY_MODELS_DIR="$MODELS_DIR/summary"
CURL_BIN="${CURL_BIN:-/usr/bin/curl}"
mkdir -p "$MODELS_DIR"
mkdir -p "$SUMMARY_MODELS_DIR"

MODEL="${1:-base}"

BASE_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
SUMMARY_BASE_URL="https://huggingface.co/jc-builds/Gemma-3-1B-Q4_K_M-GGUF/resolve/main"

case "$MODEL" in
    tiny)
        URL="$BASE_URL/ggml-tiny.en.bin"
        FILE="ggml-tiny.en.bin"
        SIZE="~75 MB"
        ;;
    base)
        URL="$BASE_URL/ggml-base.en.bin"
        FILE="ggml-base.en.bin"
        SIZE="~142 MB"
        ;;
    small)
        URL="$BASE_URL/ggml-small.en.bin"
        FILE="ggml-small.en.bin"
        SIZE="~466 MB"
        ;;
    large)
        URL="$BASE_URL/ggml-large-v3.bin"
        FILE="ggml-large.en.bin"
        SIZE="~3.1 GB"
        ;;
    bundled)
        echo "Downloading bundled app models..."
        "$0" tiny
        "$0" base
        "$0" small
        "$0" summary
        echo ""
        echo "Bundled app models downloaded!"
        ls -lh "$MODELS_DIR"/ggml-tiny.en.bin "$MODELS_DIR"/ggml-base.en.bin "$MODELS_DIR"/ggml-small.en.bin "$SUMMARY_MODELS_DIR"/gemma-3-1b-it-q4_k_m.gguf
        exit 0
        ;;
    summary)
        URL="$SUMMARY_BASE_URL/Gemma-3-1B-Q4_K_M.gguf?download=1"
        FILE="summary/gemma-3-1b-it-q4_k_m.gguf"
        SIZE="~806 MB"
        ;;
    all)
        echo "Downloading all models..."
        "$0" tiny
        "$0" base
        "$0" small
        "$0" large
        "$0" summary
        echo ""
        echo "All models downloaded!"
        ls -lh "$MODELS_DIR"/*.bin "$SUMMARY_MODELS_DIR"/gemma-3-1b-it-q4_k_m.gguf
        exit 0
        ;;
    *)
        echo "Usage: $0 [tiny|base|small|large|summary|bundled|all]"
        echo ""
        echo "Models:"
        echo "  tiny   ~75 MB   Fastest, good for simple phrases"
        echo "  base   ~142 MB  Fast, great for most use cases (default)"
        echo "  small  ~466 MB  Moderate speed, better accuracy"
        echo "  large  ~3.1 GB  Slowest, highest accuracy"
        echo "  summary ~806 MB Gemma 3 1B Q4_K_M GGUF for bundled local summaries"
        echo "  bundled          Download the app bundle set (tiny, base, small, summary)"
        echo "  all              Download all models"
        exit 1
        ;;
esac

DEST="$MODELS_DIR/$FILE"
mkdir -p "$(dirname "$DEST")"

if [ -f "$DEST" ]; then
    echo "Model already exists: $DEST ($(du -h "$DEST" | cut -f1))"
    exit 0
fi

if [ ! -x "$CURL_BIN" ]; then
    echo "Error: curl not found at $CURL_BIN"
    exit 1
fi

TMP_DEST="$(mktemp "${DEST}.partial.XXXXXX")"
cleanup() {
    rm -f "$TMP_DEST"
}
trap cleanup EXIT

echo "Downloading $FILE ($SIZE)..."
"$CURL_BIN" \
    --fail \
    --show-error \
    --location \
    --proto '=https' \
    --tlsv1.2 \
    --progress-bar \
    -o "$TMP_DEST" \
    "$URL"

mv "$TMP_DEST" "$DEST"
trap - EXIT

echo "Done: $DEST ($(du -h "$DEST" | cut -f1))"
