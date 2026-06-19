#!/usr/bin/env bash
# Pipeline performance breakdown for vsr-cli.
#
# Pinpoints the bottleneck stage by timing each part in isolation:
#   1. decode only                     (lsmas)
#   2. decode + 4K colour conversion   (no model, no GPU) -> exposes zimg cost
#   3. full graph -> /dev/null         (decode+RGB+model+YUV, no ffmpeg)
#   4. NVENC preset sweep p7..p4       (real graph piped into ffmpeg -> null)
#
# Stages 1-3 write y4m to /dev/null (no ffmpeg at all), so they measure pure
# VapourSynth throughput. Stage 4 pipes the real graph into ffmpeg and encodes
# to the null muxer, varying only the NVENC preset, so it measures the encoder
# ceiling. Compare the numbers: the lowest fps is your wall.
#
# Usage:   bash bench.sh [INPUT]
# Env overrides: MODEL PD FRAMES NUM_STREAMS THREADS TRTEXEC NVENC_FIX
#                PIX_FMT VSPIPE FFMPEG
set -uo pipefail

INPUT="${1:-/root/sample.mkv}"
MODEL="${MODEL:-2x_AnimeJaNai_HD_V3.1Sharp1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx}"
PD="${PD:-/root/autodl-tmp/vsr-runtime/vs-plugins}"
FRAMES="${FRAMES:-2000}"
NUM_STREAMS="${NUM_STREAMS:-4}"
THREADS="${THREADS:-$(nproc)}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
NVENC_FIX="${NVENC_FIX:-/opt/libnvenc_fix.so}"
PIX_FMT="${PIX_FMT:-p010le}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPY="$HERE/pipeline.vpy"
if [[ -z "${VSPIPE:-}" ]]; then
  if [[ -x "$HERE/venv/bin/vspipe" ]]; then VSPIPE="$HERE/venv/bin/vspipe"; else VSPIPE="vspipe"; fi
fi
FFMPEG="${FFMPEG:-/usr/local/bin/ffmpeg}"
command -v "$FFMPEG" >/dev/null 2>&1 || FFMPEG="ffmpeg"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== config =="
echo "input=$INPUT"
echo "model=$MODEL"
echo "frames=$FRAMES  num_streams=$NUM_STREAMS  threads=$THREADS"
echo "vspipe=$VSPIPE  ffmpeg=$FFMPEG"
echo

cat > "$TMP/dec.vpy" <<EOF
import vapoursynth as vs
from vapoursynth import core
core.num_threads = $THREADS
core.lsmas.LWLibavSource(source=r"$INPUT").set_output()
EOF

cat > "$TMP/color.vpy" <<EOF
import vapoursynth as vs
from vapoursynth import core
core.num_threads = $THREADS
c = core.lsmas.LWLibavSource(source=r"$INPUT")
c = core.resize.Bicubic(c, format=vs.RGBH, matrix_in=1, range_in=0)
c = core.resize.Point(c, width=c.width * 2, height=c.height * 2)  # cheap fake-4K
c = core.resize.Bicubic(c, format=vs.YUV420P10, matrix=1, range=0)
c.set_output()
EOF

full_args=(
  --arg "video_path=$INPUT" --arg upscale=1 --arg "model=$MODEL"
  --arg "num_streams=$NUM_STREAMS" --arg "plugins_dir=$PD"
  --arg "models_dir=$PD/models" --arg "trtexec_path=$TRTEXEC"
)

run_vs () {  # label  scriptpath
  echo "== [$1] =="
  "$VSPIPE" -p -c y4m --end "$FRAMES" "$2" /dev/null 2>&1 | grep -i fps | tail -1
  echo
}

run_vs "1. decode only" "$TMP/dec.vpy"
run_vs "2. decode + 4K colour convert (no model)" "$TMP/color.vpy"

echo "== [3. full graph -> /dev/null (no ffmpeg)] =="
"$VSPIPE" -p -c y4m --end "$FRAMES" "${full_args[@]}" "$VPY" /dev/null 2>&1 | grep -i fps | tail -1
echo

# NVENC sweep: only the preset changes. Inject the GPU-enumeration shim for ffmpeg.
if [[ -f "$NVENC_FIX" ]]; then
  export LD_PRELOAD="$NVENC_FIX${LD_PRELOAD:+ $LD_PRELOAD}"
fi
for P in p7 p6 p5 p4; do
  echo "== [4. encode hevc_nvenc -preset $P -> null] =="
  "$VSPIPE" -c y4m --end "$FRAMES" "${full_args[@]}" "$VPY" - 2>/dev/null \
    | "$FFMPEG" -y -hide_banner -i pipe: \
        -c:v hevc_nvenc -preset "$P" -profile:v main10 -tier high \
        -pix_fmt "$PIX_FMT" -rc:v vbr -cq:v 20 -b:v 0 -bf 3 \
        -f null - 2>&1 | grep -E '^frame=' | tail -1
  echo
done

echo "done. Lowest fps across the stages = your bottleneck."
