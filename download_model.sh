#!/bin/bash
set -euo pipefail

MODELS_DIR="models"
SUMMARY_MODELS_DIR="$MODELS_DIR/summary"
TTS_MODELS_DIR="$MODELS_DIR/tts"
TTS_RUNTIME_DIR="$TTS_MODELS_DIR/onnxruntime"
CURL_BIN="${CURL_BIN:-/usr/bin/curl}"
mkdir -p "$MODELS_DIR"
mkdir -p "$SUMMARY_MODELS_DIR"
mkdir -p "$TTS_MODELS_DIR"
mkdir -p "$TTS_RUNTIME_DIR"

MODEL="${1:-base}"

BASE_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
SUMMARY_BASE_URL="https://huggingface.co/jc-builds/Gemma-3-1B-Q4_K_M-GGUF/resolve/main"
TTS_BASE_URL="https://github.com/8b-is/kokoro-tiny/raw/main/models"
TTS_RUNTIME_VERSION="${TTS_RUNTIME_VERSION:-1.23.2}"
TTS_RUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${TTS_RUNTIME_VERSION}/onnxruntime-osx-universal2-${TTS_RUNTIME_VERSION}.tgz"
TTS_RUNTIME_DYLIB="$TTS_RUNTIME_DIR/libonnxruntime.dylib"

download_tts_runtime() {
    if [ -s "$TTS_RUNTIME_DYLIB" ]; then
        echo "ONNX Runtime already exists: $TTS_RUNTIME_DYLIB ($(du -h "$TTS_RUNTIME_DYLIB" | cut -f1))"
        return
    fi

    if [ ! -x "$CURL_BIN" ]; then
        echo "Error: curl not found at $CURL_BIN"
        exit 1
    fi

    local tmp_archive
    local tmp_extract
    tmp_archive="$(mktemp "$TTS_RUNTIME_DIR/onnxruntime.tgz.partial.XXXXXX")"
    tmp_extract="$(mktemp -d "$TTS_RUNTIME_DIR/onnxruntime.extract.XXXXXX")"
    cleanup_runtime() {
        rm -f "$tmp_archive"
        rm -rf "$tmp_extract"
    }
    trap cleanup_runtime EXIT

    echo "Downloading ONNX Runtime macOS universal2 $TTS_RUNTIME_VERSION (~41 MB compressed)..."
    "$CURL_BIN" \
        --fail \
        --show-error \
        --location \
        --proto '=https' \
        --tlsv1.2 \
        --progress-bar \
        -o "$tmp_archive" \
        "$TTS_RUNTIME_URL"

    tar -xzf "$tmp_archive" -C "$tmp_extract"

    local extracted_dylib
    extracted_dylib="$(find -L "$tmp_extract" -type f \( -name 'libonnxruntime.dylib' -o -name 'libonnxruntime.*.dylib' \) | sort | head -n 1)"
    if [ -z "$extracted_dylib" ]; then
        echo "Error: libonnxruntime.dylib not found in downloaded ONNX Runtime archive."
        exit 1
    fi

    mkdir -p "$TTS_RUNTIME_DIR"
    cp "$extracted_dylib" "$TTS_RUNTIME_DYLIB"
    trap - EXIT
    cleanup_runtime

    echo "Done: $TTS_RUNTIME_DYLIB ($(du -h "$TTS_RUNTIME_DYLIB" | cut -f1))"
}

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
        "$0" tts
        echo ""
        echo "Bundled app models downloaded!"
        ls -lh "$MODELS_DIR"/ggml-tiny.en.bin "$MODELS_DIR"/ggml-base.en.bin "$MODELS_DIR"/ggml-small.en.bin "$SUMMARY_MODELS_DIR"/gemma-3-1b-it-q4_k_m.gguf "$TTS_MODELS_DIR"/0.onnx "$TTS_MODELS_DIR"/0.bin "$TTS_RUNTIME_DYLIB"
        exit 0
        ;;
    summary)
        URL="$SUMMARY_BASE_URL/Gemma-3-1B-Q4_K_M.gguf?download=1"
        FILE="summary/gemma-3-1b-it-q4_k_m.gguf"
        SIZE="~806 MB"
        ;;
    tts)
        echo "Downloading Kokoro TTS model bundle..."
        "$0" tts-model
        "$0" tts-voices
        "$0" tts-runtime
        echo ""
        echo "Kokoro TTS models downloaded!"
        ls -lh "$TTS_MODELS_DIR"/0.onnx "$TTS_MODELS_DIR"/0.bin "$TTS_RUNTIME_DYLIB"
        exit 0
        ;;
    tts-model)
        URL="$TTS_BASE_URL/0.onnx"
        FILE="tts/0.onnx"
        SIZE="~310 MB"
        ;;
    tts-voices)
        URL="$TTS_BASE_URL/0.bin"
        FILE="tts/0.bin"
        SIZE="~27 MB"
        ;;
    tts-runtime)
        download_tts_runtime
        exit 0
        ;;
    all)
        echo "Downloading all models..."
        "$0" tiny
        "$0" base
        "$0" small
        "$0" large
        "$0" summary
        "$0" tts
        echo ""
        echo "All models downloaded!"
        ls -lh "$MODELS_DIR"/*.bin "$SUMMARY_MODELS_DIR"/gemma-3-1b-it-q4_k_m.gguf "$TTS_MODELS_DIR"/0.onnx "$TTS_MODELS_DIR"/0.bin "$TTS_RUNTIME_DYLIB"
        exit 0
        ;;
    *)
        echo "Usage: $0 [tiny|base|small|large|summary|tts|tts-runtime|bundled|all]"
        echo ""
        echo "Models:"
        echo "  tiny   ~75 MB   Fastest, good for simple phrases"
        echo "  base   ~142 MB  Fast, great for most use cases (default)"
        echo "  small  ~466 MB  Moderate speed, better accuracy"
        echo "  large  ~3.1 GB  Slowest, highest accuracy"
        echo "  summary ~806 MB Gemma 3 1B Q4_K_M GGUF for bundled local summaries"
        echo "  tts     ~378 MB Kokoro TTS ONNX model, voices, and ONNX Runtime for spoken screen help"
        echo "  tts-runtime     ONNX Runtime macOS universal2 dylib used by Kokoro TTS"
        echo "  bundled          Download the app bundle set (tiny, base, small, summary, tts)"
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
