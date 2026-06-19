#!/usr/bin/env bash
# NVENC dual-engine (split-encode) sweep for vsr-cli.
#
# The 4090 has two NVENC engines; a single hevc_nvenc session uses one and caps
# 4K 10-bit HEVC at ~55-60 fps regardless of preset. -split_encode_mode slices
# one stream across both engines. This sweeps the modes and reports fps; on
# failure it prints ffmpeg's error (so we see whether the GPU/args reject it).
#
# Usage:   bash benchmarks/bench_split.sh [INPUT]
# Env overrides: MODEL PD FRAMES NUM_STREAMS TRTEXEC NVENC_FIX PIX_FMT
#                PRESET MODES VSPIPE FFMPEG
set -uo pipefail

INPUT="${1:-/root/sample.mkv}"
MODEL="${MODEL:-2x_AnimeJaNai_HD_V3.1Sharp1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx}"
PD="${PD:-/root/autodl-tmp/vsr-runtime/vs-plugins}"
FRAMES="${FRAMES:-2000}"
NUM_STREAMS="${NUM_STREAMS:-4}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
NVENC_FIX="${NVENC_FIX:-/opt/libnvenc_fix.so}"
PIX_FMT="${PIX_FMT:-p010le}"
PRESET="${PRESET:-p6}"
MODES="${MODES:-disabled auto forced 2 3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VPY="$REPO_ROOT/pipeline.vpy"
if [[ -z "${VSPIPE:-}" ]]; then
  if [[ -x "$REPO_ROOT/venv/bin/vspipe" ]]; then VSPIPE="$REPO_ROOT/venv/bin/vspipe"; else VSPIPE="vspipe"; fi
fi
FFMPEG="${FFMPEG:-/usr/local/bin/ffmpeg}"
command -v "$FFMPEG" >/dev/null 2>&1 || FFMPEG="ffmpeg"
[[ -f "$NVENC_FIX" ]] && export LD_PRELOAD="$NVENC_FIX${LD_PRELOAD:+ $LD_PRELOAD}"

echo "== config =="
echo "input=$INPUT  preset=$PRESET  frames=$FRAMES  num_streams=$NUM_STREAMS"
echo "modes: $MODES"
echo "vspipe=$VSPIPE  ffmpeg=$FFMPEG  LD_PRELOAD=${LD_PRELOAD:-<unset>}"
echo

full_args=(
  --arg "video_path=$INPUT" --arg upscale=1 --arg "model=$MODEL"
  --arg "num_streams=$NUM_STREAMS" --arg "plugins_dir=$PD"
  --arg "models_dir=$PD/models" --arg "trtexec_path=$TRTEXEC"
)

ERR="$(mktemp)"
trap 'rm -f "$ERR"' EXIT

for M in $MODES; do
  echo "== split_encode_mode=$M (preset $PRESET) =="
  "$VSPIPE" -c y4m --end "$FRAMES" "${full_args[@]}" "$VPY" - 2>/dev/null \
    | "$FFMPEG" -y -hide_banner -i pipe: \
        -c:v hevc_nvenc -preset "$PRESET" -profile:v main10 -tier high \
        -pix_fmt "$PIX_FMT" -rc:v vbr -cq:v 20 -b:v 0 -bf 3 \
        -split_encode_mode "$M" -f null - 2>"$ERR"
  fps_line="$(grep -E '^frame=' "$ERR" | tail -1)"
  if [[ -n "$fps_line" ]]; then
    echo "  $fps_line"
  else
    echo "  FAILED — ffmpeg error:"
    tail -4 "$ERR" | sed 's/^/    /'
  fi
  echo
done

echo "done. Higher fps than 'disabled' => the second NVENC engine is helping."
