#!/bin/bash
set -euo pipefail

APP="${APP:-Screamer-smoke.app}"
BIN_PATH="${BIN_PATH:-target/release/screamer}"
PLUTIL_BIN="${PLUTIL_BIN:-/usr/bin/plutil}"
CODESIGN_BIN="${CODESIGN_BIN:-/usr/bin/codesign}"

if [ ! -f "$BIN_PATH" ]; then
    echo "Error: binary not found at $BIN_PATH"
    exit 1
fi

TMP_MODELS_DIR="$(mktemp -d "${TMPDIR:-/tmp}/screamer-models.XXXXXX")"
TMP_SUMMARY_MODELS_DIR="$(mktemp -d "${TMPDIR:-/tmp}/screamer-summary-models.XXXXXX")"
TMP_TTS_MODELS_DIR="$(mktemp -d "${TMPDIR:-/tmp}/screamer-tts-models.XXXXXX")"
cleanup() {
    rm -rf "$TMP_MODELS_DIR"
    rm -rf "$TMP_SUMMARY_MODELS_DIR"
    rm -rf "$TMP_TTS_MODELS_DIR"
}
trap cleanup EXIT

for model in ggml-tiny.en.bin ggml-base.en.bin ggml-small.en.bin; do
    printf 'placeholder model\n' > "$TMP_MODELS_DIR/$model"
done
printf 'placeholder summary model\n' > "$TMP_SUMMARY_MODELS_DIR/gemma-3-1b-it-q4_k_m.gguf"
printf 'placeholder vision model\n' > "$TMP_SUMMARY_MODELS_DIR/gemma-3-4b-it-q4.gguf"
printf 'placeholder tts model\n' > "$TMP_TTS_MODELS_DIR/0.onnx"
printf 'placeholder tts voices\n' > "$TMP_TTS_MODELS_DIR/0.bin"
mkdir -p "$TMP_TTS_MODELS_DIR/onnxruntime"

TTS_RUNTIME_DYLIB_SOURCE="${TTS_RUNTIME_DYLIB:-models/tts/onnxruntime/libonnxruntime.dylib}"
if [ -f "$TTS_RUNTIME_DYLIB_SOURCE" ]; then
    cp "$TTS_RUNTIME_DYLIB_SOURCE" "$TMP_TTS_MODELS_DIR/onnxruntime/libonnxruntime.dylib"
else
    cc -dynamiclib -x c - -o "$TMP_TTS_MODELS_DIR/onnxruntime/libonnxruntime.dylib" <<'EOF'
int screamer_smoke_onnxruntime_placeholder(void) {
    return 0;
}
EOF
fi

SUMMARY_HELPER_PATH="${SUMMARY_HELPER_PATH:-target/release/screamer_summary_helper}"
VISION_HELPER_PATH="${VISION_HELPER_PATH:-target/release/screamer_vision_helper}"
TTS_HELPER_PATH="${TTS_HELPER_PATH:-target/release/screamer_tts_helper}"

APP="$APP" \
  MODELS_DIR="$TMP_MODELS_DIR" \
  SUMMARY_MODELS_DIR="$TMP_SUMMARY_MODELS_DIR" \
  TTS_MODELS_DIR="$TMP_TTS_MODELS_DIR" \
  SKIP_BUILD=1 \
  BIN_PATH="$BIN_PATH" \
  SUMMARY_HELPER_PATH="$SUMMARY_HELPER_PATH" \
  VISION_HELPER_PATH="$VISION_HELPER_PATH" \
  TTS_HELPER_PATH="$TTS_HELPER_PATH" \
  ./bundle.sh

test -x "$APP/Contents/MacOS/Screamer"
test -x "$APP/Contents/MacOS/screamer_tts_helper"
test -f "$APP/Contents/Info.plist"
test -f "$APP/Contents/Resources/icon.icns"
test -f "$APP/Contents/Resources/image.png"
test -f "$APP/Contents/Resources/models/ggml-tiny.en.bin"
test -f "$APP/Contents/Resources/models/ggml-base.en.bin"
test -f "$APP/Contents/Resources/models/ggml-small.en.bin"
test -f "$APP/Contents/Resources/models/summary/gemma-3-1b-it-q4_k_m.gguf"
test -f "$APP/Contents/Resources/models/summary/gemma-3-4b-it-q4.gguf"
test -f "$APP/Contents/Resources/models/tts/0.onnx"
test -f "$APP/Contents/Resources/models/tts/0.bin"
test -f "$APP/Contents/MacOS/libonnxruntime.dylib"

bundle_id="$("$PLUTIL_BIN" -extract CFBundleIdentifier raw "$APP/Contents/Info.plist")"
bundle_exec="$("$PLUTIL_BIN" -extract CFBundleExecutable raw "$APP/Contents/Info.plist")"

if [ "$bundle_id" != "com.screamer.app" ]; then
    echo "Error: unexpected bundle identifier: $bundle_id"
    exit 1
fi

if [ "$bundle_exec" != "Screamer" ]; then
    echo "Error: unexpected bundle executable: $bundle_exec"
    exit 1
fi

"$CODESIGN_BIN" --verify --deep --strict --verbose=2 "$APP"

echo "macOS bundle smoke test passed."
