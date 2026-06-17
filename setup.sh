#!/usr/bin/env bash
# vsr-cli runtime bootstrap for Ubuntu 22.04 GPU containers (e.g. AutoDL).
#
# Provisions a self-contained VapourSynth + vs-mlrt (TensorRT) runtime:
#   1. conda-forge env with vapoursynth + source plugins (ffms2/lsmas) + onnx
#   2. vs-mlrt Linux x64 plugins (vsort/vstrt + vsmlrt-cuda incl. trtexec)
#   3. RealESRGAN + RIFE model packs
#   4. writes vsr config.toml
#
# The whole runtime dir is relocatable: tar it up and reuse across containers.
#
# Usage:
#   bash setup.sh [RUNTIME_DIR]
# Env overrides:
#   VSR_RUNTIME       runtime dir (default: /root/autodl-tmp/vsr-runtime)
#   VSMLRT_TAG        vs-mlrt release tag (default: latest)
#   CONDA_ENV         conda env name (default: vsr)
#   MODEL_PACKS       space-separated model asset name substrings to fetch
#                     (default: "RealESRGANv2 rife")
#   SKIP_CONDA=1      skip conda env creation (use current python/vapoursynth)
set -euo pipefail

RUNTIME_DIR="${1:-${VSR_RUNTIME:-/root/autodl-tmp/vsr-runtime}}"
VSMLRT_TAG="${VSMLRT_TAG:-latest}"
CONDA_ENV="${CONDA_ENV:-vsr}"
MODEL_PACKS="${MODEL_PACKS:-RealESRGANv2 rife}"
REPO="AmusementClub/vs-mlrt"

PLUGINS_DIR="$RUNTIME_DIR/vs-plugins"
MODELS_DIR="$PLUGINS_DIR/models"
DL_DIR="$RUNTIME_DIR/downloads"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[setup:error]\033[0m %s\n' "$*" >&2; }

mkdir -p "$PLUGINS_DIR" "$MODELS_DIR" "$DL_DIR"

# --- sanity: GPU ------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    log "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n1)"
else
    err "nvidia-smi not found — TensorRT backend requires an NVIDIA GPU."
fi

# --- 1. conda env -----------------------------------------------------------
PYTHON_BIN="python"
VSPIPE_BIN=""
if [[ "${SKIP_CONDA:-0}" != "1" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        err "conda not found. Install Miniconda first, or run with SKIP_CONDA=1."
        err "  e.g. wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh"
        exit 2
    fi
    log "Creating/updating conda env '$CONDA_ENV' (vapoursynth, ffms2, lsmas, onnx)…"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda env list | grep -qE "^${CONDA_ENV}\s"; then
        conda create -y -n "$CONDA_ENV" -c conda-forge \
            python=3.11 vapoursynth ffms2 lsmash-source ffmpeg onnx numpy
    else
        conda install -y -n "$CONDA_ENV" -c conda-forge \
            vapoursynth ffms2 lsmash-source ffmpeg onnx numpy
    fi
    conda activate "$CONDA_ENV"
    PYTHON_BIN="$(command -v python)"
    VSPIPE_BIN="$(command -v vspipe || true)"
else
    log "SKIP_CONDA=1 — using current python: $(command -v python)"
    VSPIPE_BIN="$(command -v vspipe || true)"
fi

[[ -z "$VSPIPE_BIN" ]] && err "vspipe not on PATH after install — check the VapourSynth install."

# 7z for extracting release archives
if ! command -v 7z >/dev/null 2>&1 && ! command -v 7za >/dev/null 2>&1; then
    log "Installing p7zip…"
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -y && sudo apt-get install -y p7zip-full
    else
        conda install -y -n "$CONDA_ENV" -c conda-forge p7zip
    fi
fi
SEVENZ="$(command -v 7z || command -v 7za)"

# --- helper: resolve release assets via GitHub API --------------------------
api_url() {
    if [[ "$VSMLRT_TAG" == "latest" ]]; then
        echo "https://api.github.com/repos/$REPO/releases/latest"
    else
        echo "https://api.github.com/repos/$REPO/releases/tags/$VSMLRT_TAG"
    fi
}

# Print "name<TAB>url" for each asset of the chosen release.
list_assets() {
    local auth=()
    [[ -n "${GITHUB_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer $GITHUB_TOKEN")
    curl -fsSL "${auth[@]}" "$(api_url)" \
      | "$PYTHON_BIN" -c '
import json,sys
d=json.load(sys.stdin)
for a in d.get("assets",[]):
    print(a["name"]+"\t"+a["browser_download_url"])
'
}

ASSETS="$(list_assets)" || { err "Failed to query GitHub release assets."; exit 2; }
log "Release '$VSMLRT_TAG' has $(printf '%s\n' "$ASSETS" | wc -l) assets."

download_and_extract() {
    # $1 = grep -E pattern to match asset name; $2 = destination dir
    local pattern="$1" dest="$2" matched=0
    while IFS=$'\t' read -r name url; do
        [[ -z "$name" ]] && continue
        if printf '%s' "$name" | grep -qiE "$pattern"; then
            matched=1
            local out="$DL_DIR/$name"
            if [[ ! -f "$out" ]]; then
                log "Downloading $name"
                curl -fL --retry 3 -o "$out" "$url"
            else
                log "Cached $name"
            fi
            log "Extracting $name -> $dest"
            mkdir -p "$dest"
            case "$name" in
                *.7z)  "$SEVENZ" x -y -o"$dest" "$out" >/dev/null ;;
                *.zip) "$SEVENZ" x -y -o"$dest" "$out" >/dev/null ;;
                *.tar|*.tar.*|*.tgz) tar -xf "$out" -C "$dest" ;;
                *) err "Unknown archive type: $name" ;;
            esac
        fi
    done <<< "$ASSETS"
    [[ "$matched" == "1" ]] || err "No asset matched /$pattern/ in release $VSMLRT_TAG."
}

# --- 2. plugins (Linux x64): vsort + vstrt + vsmlrt-cuda --------------------
log "Fetching vs-mlrt Linux plugins…"
download_and_extract 'VSORT-Linux-x64' "$PLUGINS_DIR"
download_and_extract 'VSTRT-Linux-x64' "$PLUGINS_DIR"
# vsmlrt-cuda bundles CUDA/TensorRT libs + trtexec (no OS suffix on some releases)
download_and_extract 'vsmlrt-cuda' "$PLUGINS_DIR"

# ensure trtexec is executable
find "$PLUGINS_DIR" -name trtexec -exec chmod +x {} \; 2>/dev/null || true

# --- 3. vsmlrt.py + models --------------------------------------------------
# vsmlrt.py: prefer the local checkout, fall back to the release asset.
if [[ -f "$SCRIPT_DIR/../vs-mlrt/scripts/vsmlrt.py" ]]; then
    cp "$SCRIPT_DIR/../vs-mlrt/scripts/vsmlrt.py" "$PLUGINS_DIR/vsmlrt.py"
    log "Copied vsmlrt.py from local checkout."
else
    download_and_extract 'vsmlrt.py|scripts' "$PLUGINS_DIR" || true
fi

log "Fetching model packs: $MODEL_PACKS"
for pack in $MODEL_PACKS; do
    download_and_extract "$pack" "$MODELS_DIR"
done

# --- 4. write config.toml ---------------------------------------------------
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/vsr"
mkdir -p "$CONFIG_DIR"
CONFIG_FILE="$CONFIG_DIR/config.toml"
FFMPEG_BIN="$(command -v ffmpeg || echo ffmpeg)"
PIPELINE_VPY="$SCRIPT_DIR/pipeline.vpy"

cat > "$CONFIG_FILE" <<EOF
vspipe = "$VSPIPE_BIN"
ffmpeg = "$FFMPEG_BIN"
plugins_dir = "$PLUGINS_DIR"
models_dir = "$MODELS_DIR"
pipeline_vpy = "$PIPELINE_VPY"
encoder = "nvenc"
num_streams = 2
device_id = 0
fp16 = true
EOF

log "Wrote config: $CONFIG_FILE"
log "Runtime ready at: $RUNTIME_DIR"
log "Next: run 'vsr doctor' to verify, then 'vsr build-engines -i sample.mkv --upscale --model animejanaiV3_HD_L2'"
log "Tip: tar -C '$RUNTIME_DIR' -czf vsr-runtime.tar.gz . to snapshot for reuse."
