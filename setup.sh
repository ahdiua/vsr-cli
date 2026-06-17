#!/usr/bin/env bash
# vsr-cli runtime bootstrap for Ubuntu 22.04 GPU containers (e.g. AutoDL).
#
# Conda-free: installs VapourSynth via pip into a venv (per the official docs),
# then layers on the source plugin + vs-mlrt TensorRT plugins + models.
#
#   1. system Python 3.12+ venv  ->  pip install vapoursynth vsrepo
#   2. `vapoursynth config` (+ register-install) to set up plugin autoload
#   3. source plugin (lsmas/ffms2) via VSRepo
#   4. vs-mlrt Linux x64 plugins (vsort/vstrt + vsmlrt-cuda incl. trtexec)
#   5. RealESRGAN + RIFE model packs
#   6. install the vsr CLI itself into the venv + write config.toml
#
# The whole runtime dir is relocatable: tar it up and reuse across containers.
#
# Usage:   bash setup.sh [RUNTIME_DIR]
# Env overrides:
#   VSR_RUNTIME        runtime dir (default: /root/autodl-tmp/vsr-runtime)
#   PY_BIN             python interpreter to use (default: auto-detect 3.12+)
#   VSMLRT_TAG         vs-mlrt release tag (default: latest)
#   MODEL_PACKS        model asset name substrings (default: "RealESRGANv2 rife")
#   SOURCE_PLUGIN      VSRepo source plugin id(s) (default: "lsmas ffms2")
#   SKIP_PYTHON_INSTALL=1   do not auto-install python3.12 via apt/deadsnakes
#   SKIP_APT=1         do not use apt at all (assume deps already present)
set -euo pipefail

RUNTIME_DIR="${1:-${VSR_RUNTIME:-/root/autodl-tmp/vsr-runtime}}"
VSMLRT_TAG="${VSMLRT_TAG:-latest}"
MODEL_PACKS="${MODEL_PACKS:-RealESRGANv2 rife}"
SOURCE_PLUGIN="${SOURCE_PLUGIN:-lsmas ffms2}"
REPO="AmusementClub/vs-mlrt"

VENV_DIR="$RUNTIME_DIR/venv"
PLUGINS_DIR="$RUNTIME_DIR/vs-plugins"
MODELS_DIR="$PLUGINS_DIR/models"
DL_DIR="$RUNTIME_DIR/downloads"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[setup:error]\033[0m %s\n' "$*" >&2; }

mkdir -p "$PLUGINS_DIR" "$MODELS_DIR" "$DL_DIR"

have() { command -v "$1" >/dev/null 2>&1; }
apt_get() { if [[ "${SKIP_APT:-0}" != "1" ]] && have apt-get; then sudo apt-get "$@"; fi; }

# --- sanity: GPU ------------------------------------------------------------
if have nvidia-smi; then
    log "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n1)"
else
    err "nvidia-smi not found — the TensorRT backend requires an NVIDIA GPU."
fi

# --- system deps (curl / 7z / ffmpeg) --------------------------------------
NEED_APT=()
have curl   || NEED_APT+=(curl)
{ have 7z || have 7za; } || NEED_APT+=(p7zip-full)
have ffmpeg || NEED_APT+=(ffmpeg)
if [[ ${#NEED_APT[@]} -gt 0 ]]; then
    log "Installing system deps: ${NEED_APT[*]}"
    apt_get update -y || true
    apt_get install -y "${NEED_APT[@]}" || err "apt install failed for: ${NEED_APT[*]}"
fi
SEVENZ="$(command -v 7z || command -v 7za || true)"
[[ -z "$SEVENZ" ]] && err "7z/7za not found — needed to extract release archives."

# --- 1. pick / install Python 3.12+ ----------------------------------------
pick_python() {
    if [[ -n "${PY_BIN:-}" ]]; then echo "$PY_BIN"; return; fi
    for c in python3.13 python3.12 python3; do
        if have "$c" && "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,12) else 1)' 2>/dev/null; then
            command -v "$c"; return
        fi
    done
    echo ""
}

PY="$(pick_python)"
if [[ -z "$PY" ]]; then
    if [[ "${SKIP_PYTHON_INSTALL:-0}" != "1" ]] && have apt-get; then
        log "Python 3.12+ not found — installing via deadsnakes PPA…"
        apt_get update -y
        apt_get install -y software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        apt_get update -y
        apt_get install -y python3.12 python3.12-venv python3.12-dev
        PY="$(command -v python3.12)"
    else
        err "Need Python 3.12+. Install it (e.g. deadsnakes PPA) or set PY_BIN, then re-run."
        exit 2
    fi
fi
log "Using Python: $PY ($("$PY" --version 2>&1))"

# --- 2. venv + pip install VapourSynth -------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating venv at $VENV_DIR"
    "$PY" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install -q --upgrade pip wheel

log "pip install vapoursynth + tooling…"
python -m pip install -q vapoursynth vsrepo onnx numpy onnxconverter-common

# Configure VapourSynth (sets up plugin autoload dirs / VSSCRIPT_PATH).
log "Running 'vapoursynth config'…"
vapoursynth config || err "'vapoursynth config' returned non-zero (continuing)."
vapoursynth register-install || true   # optional: export VSSCRIPT_PATH

# Verify core loads and locate vspipe.
python - <<'PY' || { err "VapourSynth failed to import — check the pip install."; exit 2; }
from vapoursynth import core
print("[setup] VapourSynth core:", str(core).splitlines()[0])
PY

VSPIPE_BIN="$(command -v vspipe || true)"
[[ -z "$VSPIPE_BIN" ]] && VSPIPE_BIN="$VENV_DIR/bin/vspipe"
[[ -x "$VSPIPE_BIN" ]] || err "vspipe not found in venv — some VapourSynth wheels omit it; check the install."
log "vspipe: $VSPIPE_BIN"

# --- 3. source plugin via VSRepo -------------------------------------------
log "Installing source plugin(s) via VSRepo: $SOURCE_PLUGIN"
vsrepo update || err "'vsrepo update' failed (continuing)."
SRC_OK=0
for sp in $SOURCE_PLUGIN; do
    if vsrepo install "$sp"; then
        log "VSRepo installed: $sp"
        SRC_OK=1
        break
    else
        err "VSRepo could not install '$sp' on this platform — trying next."
    fi
done
if [[ "$SRC_OK" != "1" ]]; then
    err "No source plugin installed via VSRepo. Install ffms2 or L-SMASH-Works .so"
    err "into the VapourSynth autoload dir (or into $PLUGINS_DIR) before running vsr."
fi

# --- GitHub release asset helpers ------------------------------------------
api_url() {
    if [[ "$VSMLRT_TAG" == "latest" ]]; then
        echo "https://api.github.com/repos/$REPO/releases/latest"
    else
        echo "https://api.github.com/repos/$REPO/releases/tags/$VSMLRT_TAG"
    fi
}
list_assets() {
    local auth=()
    [[ -n "${GITHUB_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer $GITHUB_TOKEN")
    curl -fsSL "${auth[@]}" "$(api_url)" | python - <<'PY'
import json,sys
d=json.load(sys.stdin)
for a in d.get("assets",[]):
    print(a["name"]+"\t"+a["browser_download_url"])
PY
}
ASSETS="$(list_assets)" || { err "Failed to query GitHub release assets."; exit 2; }
log "vs-mlrt release '$VSMLRT_TAG': $(printf '%s\n' "$ASSETS" | grep -c . ) assets."

download_and_extract() {
    local pattern="$1" dest="$2" matched=0
    while IFS=$'\t' read -r name url; do
        [[ -z "$name" ]] && continue
        if printf '%s' "$name" | grep -qiE "$pattern"; then
            matched=1
            local out="$DL_DIR/$name"
            if [[ ! -f "$out" ]]; then
                log "Downloading $name"; curl -fL --retry 3 -o "$out" "$url"
            else
                log "Cached $name"
            fi
            log "Extracting $name -> $dest"; mkdir -p "$dest"
            case "$name" in
                *.7z|*.zip) "$SEVENZ" x -y -o"$dest" "$out" >/dev/null ;;
                *.tar|*.tar.*|*.tgz) tar -xf "$out" -C "$dest" ;;
                *) err "Unknown archive type: $name" ;;
            esac
        fi
    done <<< "$ASSETS"
    [[ "$matched" == "1" ]] || err "No asset matched /$pattern/ in release $VSMLRT_TAG."
}

# --- 4. vs-mlrt Linux plugins ----------------------------------------------
log "Fetching vs-mlrt Linux plugins…"
download_and_extract 'VSORT-Linux-x64' "$PLUGINS_DIR"
download_and_extract 'VSTRT-Linux-x64' "$PLUGINS_DIR"
download_and_extract 'vsmlrt-cuda'      "$PLUGINS_DIR"
find "$PLUGINS_DIR" -name trtexec -exec chmod +x {} \; 2>/dev/null || true

# vsmlrt.py: prefer the local checkout, else the release asset.
if [[ -f "$SCRIPT_DIR/../vs-mlrt/scripts/vsmlrt.py" ]]; then
    cp "$SCRIPT_DIR/../vs-mlrt/scripts/vsmlrt.py" "$PLUGINS_DIR/vsmlrt.py"
    log "Copied vsmlrt.py from local checkout."
else
    download_and_extract 'vsmlrt\.py|scripts' "$PLUGINS_DIR" || true
fi

# --- 5. model packs ---------------------------------------------------------
log "Fetching model packs: $MODEL_PACKS"
for pack in $MODEL_PACKS; do
    download_and_extract "$pack" "$MODELS_DIR"
done

# --- 6. install vsr CLI + write config -------------------------------------
log "Installing vsr CLI into the venv…"
python -m pip install -q -e "$SCRIPT_DIR"

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
log ""
log "Next steps:"
log "  source $VENV_DIR/bin/activate"
log "  vsr doctor"
log "  vsr build-engines -i sample.mkv --upscale --model animejanaiV3_HD_L2"
log "  vsr run -i in.mkv -o out.mkv --upscale --model animejanaiV3_HD_L2 --rife --rife-multi 2"
log "Snapshot for reuse:  tar -C '$RUNTIME_DIR' -czf vsr-runtime.tar.gz ."
