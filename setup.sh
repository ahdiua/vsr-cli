#!/usr/bin/env bash
# vsr-cli runtime bootstrap for Ubuntu 22.04 GPU containers (e.g. AutoDL).
#
# By DEFAULT this installs into the Python environment that is ALREADY ACTIVE
# (a venv OR a conda env you created and activated). It pip-installs VapourSynth
# here, then layers on the source plugin + vs-mlrt plugins + models. Because it
# mutates the active environment, it WARNS and asks for `y` first. Only a bare
# system Python (no venv, no conda) triggers the extra "will pollute" warning.
#
#   0. confirm the target Python environment (the active one)
#   1. pip install vapoursynth vsrepo  (into the active env)
#   2. `vapoursynth config` (+ conda libpython fallback + register-install)
#      to set up plugin autoload
#   3. source plugin (lsmas/ffms2) via VSRepo
#   4. vs-mlrt Linux x64 plugins (vsort/vstrt + vsmlrt-cuda incl. trtexec)
#   5. RealESRGAN + RIFE model packs
#   6. install the vsr CLI itself + write config.toml
#
# Usage:   bash setup.sh [RUNTIME_DIR]
# Env overrides:
#   VSR_RUNTIME        runtime dir (default: /root/autodl-tmp/vsr-runtime)
#   PY_BIN             python interpreter to use (default: active `python`)
#   ASSUME_YES=1       skip the confirmation prompt (non-interactive)
#   CREATE_VENV=1      create+use a fresh venv at <runtime>/venv instead of
#                      the active env (auto-installs python3.12 if needed)
#   VSMLRT_TAG         vs-mlrt release tag (default: latest)
#   MODEL_PACKS        model asset regexes (default: "^models\\. ^contrib-models\\.")
#   SOURCE_PLUGIN      VSRepo source plugin id(s) (default: "lsmas ffms2")
#   SEVENZ_THREADS     optional 7z extraction threads override
#   SKIP_PYTHON_INSTALL=1   (CREATE_VENV) don't auto-install python3.12
#   SKIP_APT=1         do not use apt at all (assume deps already present)
set -euo pipefail

RUNTIME_DIR="${1:-${VSR_RUNTIME:-/root/autodl-tmp/vsr-runtime}}"
VSMLRT_TAG="${VSMLRT_TAG:-latest}"
MODEL_PACKS="${MODEL_PACKS:-^models\\. ^contrib-models\\.}"
SOURCE_PLUGIN="${SOURCE_PLUGIN:-lsmas ffms2}"
SEVENZ_THREADS="${SEVENZ_THREADS:-}"
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
ARIA2C="$(command -v aria2c || true)"

# --- 0. confirm target Python environment ----------------------------------
# Optionally create a fresh venv; otherwise use the ACTIVE environment.
if [[ "${CREATE_VENV:-0}" == "1" ]]; then
    PY="${PY_BIN:-}"
    if [[ -z "$PY" ]]; then
        for c in python3.13 python3.12 python3; do
            if have "$c" && "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,12) else 1)' 2>/dev/null; then
                PY="$(command -v "$c")"; break
            fi
        done
    fi
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
            err "Need Python 3.12+. Install it or set PY_BIN, then re-run."
            exit 2
        fi
    fi
    log "Creating venv at $VENV_DIR (python: $PY)"
    [[ -d "$VENV_DIR" ]] || "$PY" -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# resolve the python we will install into
PYTHON="${PY_BIN:-$(command -v python || command -v python3 || true)}"
[[ -z "$PYTHON" ]] && { err "No python found on PATH. Activate your venv or set PY_BIN."; exit 2; }

PY_VER="$("$PYTHON" -c 'import sys; print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "?")"
PY_PREFIX="$("$PYTHON" -c 'import sys; print(sys.prefix)' 2>/dev/null || echo "?")"
# classify the active environment: conda env, venv, or bare system python.
# conda envs are NOT venvs (sys.prefix == base_prefix) so detect them separately
# via a conda-meta dir / CONDA_PREFIX; both venv and conda count as "isolated".
ENV_KIND="$("$PYTHON" - <<'PY' 2>/dev/null || echo system
import sys, os
prefix = sys.prefix
base = getattr(sys, "base_prefix", prefix)
if os.path.isdir(os.path.join(prefix, "conda-meta")) or os.environ.get("CONDA_PREFIX"):
    print("conda")
elif prefix != base or os.environ.get("VIRTUAL_ENV"):
    print("venv")
else:
    print("system")
PY
)"

case "$ENV_KIND" in
    conda) ENV_DESC="conda (${CONDA_DEFAULT_ENV:-$(basename "$PY_PREFIX")})" ;;
    venv)  ENV_DESC="venv (${VIRTUAL_ENV:-$PY_PREFIX})" ;;
    *)     ENV_DESC="NO — 这是系统/全局 Python!" ;;
esac

echo
printf '\033[1;33m========================================================\033[0m\n'
printf '\033[1;33m 即将把 VapourSynth 等依赖安装进“当前激活的环境”：\033[0m\n'
printf '   python : %s  (Python %s)\n' "$PYTHON" "$PY_VER"
printf '   prefix : %s\n' "$PY_PREFIX"
printf '   env    : %s\n' "$ENV_DESC"
printf '   runtime: %s\n' "$RUNTIME_DIR"
printf '\033[1;33m========================================================\033[0m\n'
[[ "$ENV_KIND" == "system" ]] && err "未检测到虚拟环境/conda 环境 —— 将污染系统 Python。建议先激活 venv 或 conda 环境，或用 CREATE_VENV=1。"
[[ "$PY_VER" != "3.1"[2-9] && "$PY_VER" != "3."[2-9][0-9] ]] && err "VapourSynth 需要 Python 3.12+，当前为 $PY_VER。"

if [[ "${ASSUME_YES:-0}" != "1" ]]; then
    read -r -p "确认在以上环境继续安装? 输入 y 继续: " _ans < /dev/tty || _ans=""
    if [[ "$_ans" != "y" && "$_ans" != "Y" ]]; then
        err "已取消。"; exit 1
    fi
fi

# console scripts (vapoursynth/vsrepo/vspipe) live next to this interpreter
PYBIN_DIR="$(dirname "$PYTHON")"
run_script() {  # run a console script from the interpreter's bin dir, fall back to PATH
    local name="$1"; shift
    if [[ -x "$PYBIN_DIR/$name" ]]; then "$PYBIN_DIR/$name" "$@";
    elif have "$name"; then "$name" "$@";
    else return 127; fi
}

ensure_vapoursynth_runtime_config() {
    local lines config_path vsscript_path python_symbol_path
    local -a _vs_config_lines

    if ! lines="$("$PYTHON" - <<'PY'
import ctypes
import os
import sys
import sysconfig
from pathlib import Path

from vapoursynth import get_vsscript


def mangle_vsscript_key(path):
    if sys.platform == "win32":
        return path.lower()
    return path.replace("/lib64/", "/lib/")


def toml_string(value):
    return '"' + str(value).replace("\\", "\\\\") + '"'


def first_toml_string(line):
    if not line or line[0] == "#":
        return None
    start = line.find('"')
    if start < 0:
        return None
    end = line.find('"', start + 1)
    if end < 0:
        return None
    return line[start + 1:end].replace("\\", "")


def can_load_python(path):
    try:
        lib = ctypes.CDLL(str(path))
        return bool(lib and lib.Py_GetVersion)
    except Exception:
        return False


def candidate_paths():
    seen = set()
    names = []
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"

    for name in (
        f"libpython{pyver}.so.1.0",
        f"libpython{pyver}.so",
        sysconfig.get_config_var("INSTSONAME"),
        sysconfig.get_config_var("LDLIBRARY"),
        sysconfig.get_config_var("LIBRARY"),
    ):
        if name and name not in names:
            names.append(name)

    roots = []
    for root in (
        Path(sys.prefix) / "lib",
        Path(sys.base_prefix) / "lib",
        Path(os.environ["CONDA_PREFIX"]) / "lib" if os.environ.get("CONDA_PREFIX") else None,
        sysconfig.get_config_var("LIBDIR"),
        sysconfig.get_config_var("LIBPL"),
        sysconfig.get_config_var("srcdir"),
    ):
        if not root:
            continue
        root = Path(root)
        if root not in roots:
            roots.append(root)

    for root in roots:
        for name in names:
            path = root / name
            if path in seen:
                continue
            seen.add(path)
            yield path


python_symbol_path = None
for path in candidate_paths():
    if path.is_file() and can_load_python(path):
        python_symbol_path = path
        break

if python_symbol_path is None:
    raise SystemExit("could not find a loadable libpython for this Python environment")

config_home = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
config_dir = Path(config_home) / "vapoursynth"
config_dir.mkdir(parents=True, exist_ok=True)
config_path = config_dir / "vapoursynth.toml"

vsscript_path = mangle_vsscript_key(get_vsscript())
entry = f"{toml_string(vsscript_path)} = [{toml_string(sys.executable)},{toml_string(python_symbol_path)}]"

old_lines = []
if config_path.exists():
    old_lines = config_path.read_text(encoding="utf-8").splitlines()

new_lines = []
replaced = False
for line in old_lines:
    if first_toml_string(line) == vsscript_path:
        if not replaced:
            new_lines.append(entry)
            replaced = True
        continue
    new_lines.append(line)

if not replaced:
    new_lines.append(entry)

config_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

print(config_path)
print(vsscript_path)
print(python_symbol_path)
PY
)"; then
        err "Failed to write VapourSynth runtime config for $PYTHON."
        return 1
    fi

    mapfile -t _vs_config_lines <<< "$lines"
    config_path="${_vs_config_lines[0]:-}"
    vsscript_path="${_vs_config_lines[1]:-}"
    python_symbol_path="${_vs_config_lines[2]:-}"

    if [[ -z "$config_path" || -z "$vsscript_path" || -z "$python_symbol_path" ]]; then
        err "VapourSynth runtime config helper returned incomplete data."
        return 1
    fi

    export VSSCRIPT_PATH="$vsscript_path"
    log "VapourSynth config: $config_path"
    log "VSSCRIPT_PATH: $VSSCRIPT_PATH"
    log "Python symbols: $python_symbol_path"
}

ensure_python_user_site_dir() {
    local user_site

    user_site="$("$PYTHON" - <<'PY'
import site

print(site.getusersitepackages())
PY
)"
    if [[ -n "$user_site" ]]; then
        mkdir -p "$user_site"
    fi
}

run_vsrepo_install() {
    local package="$1"
    local output

    if ! output="$(run_script vsrepo install "$package" 2>&1)"; then
        printf '%s\n' "$output"
        return 1
    fi

    printf '%s\n' "$output"
    if printf '%s\n' "$output" | grep -qiE 'No binaries available|[1-9][0-9]* packages failed'; then
        return 1
    fi

    return 0
}

# --- 1. pip install VapourSynth into the active env ------------------------
log "pip install vapoursynth + tooling…"
"$PYTHON" -m pip install --upgrade pip wheel
"$PYTHON" -m pip install vapoursynth vsrepo onnx numpy onnxconverter-common

# Configure VapourSynth (sets up plugin autoload dirs / VSSCRIPT_PATH).
log "Running 'vapoursynth config'…"
run_script vapoursynth config || err "'vapoursynth config' returned non-zero (continuing)."
ensure_vapoursynth_runtime_config || exit 2
run_script vapoursynth register-install || true   # optional: export VSSCRIPT_PATH

# Verify core loads.
"$PYTHON" - <<'PY' || { err "VapourSynth failed to import — check the pip install."; exit 2; }
from vapoursynth import core
print("[setup] VapourSynth core:", str(core).splitlines()[0])
PY

VSPIPE_BIN="$PYBIN_DIR/vspipe"
[[ -x "$VSPIPE_BIN" ]] || VSPIPE_BIN="$(command -v vspipe || true)"
[[ -n "$VSPIPE_BIN" && -x "$VSPIPE_BIN" ]] || err "vspipe not found next to $PYTHON — some VapourSynth wheels omit it; check the install."
log "vspipe: $VSPIPE_BIN"

# --- 3. source plugin via VSRepo -------------------------------------------
log "Installing source plugin(s) via VSRepo: $SOURCE_PLUGIN"
ensure_python_user_site_dir
run_script vsrepo update || err "'vsrepo update' failed (continuing)."
SRC_OK=0
for sp in $SOURCE_PLUGIN; do
    if run_vsrepo_install "$sp"; then
        log "VSRepo source plugin available: $sp"
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
    curl -fsSL "${auth[@]}" "$(api_url)" | "$PYTHON" -c '
import json
import sys

d = json.load(sys.stdin)
for a in d.get("assets", []):
    print(a["name"] + "\t" + a["browser_download_url"] + "\t" + str(a.get("size", "")))
'
}
ASSETS="$(list_assets)" || { err "Failed to query GitHub release assets."; exit 2; }
log "vs-mlrt release '$VSMLRT_TAG': $(printf '%s\n' "$ASSETS" | grep -c . ) assets."

download_asset() {
    local name="$1" url="$2" out="$3"

    log "Downloading $name"
    log "Manual download target: $out"
    log "Manual download URL: $url"

    if [[ -n "$ARIA2C" ]]; then
        "$ARIA2C" -c -x 8 -s 8 --summary-interval=30 --file-allocation=none \
            -d "$(dirname "$out")" -o "$(basename "$out")" "$url"
    else
        curl -fL --retry 3 -C - --speed-limit 1024 --speed-time 120 -o "$out" "$url"
    fi
}

download_and_extract() {
    local pattern="$1" dest="$2" matched=0 extract_file=""
    while IFS=$'\t' read -r name url size; do
        [[ -z "$name" ]] && continue
        if printf '%s' "$name" | grep -qiE "$pattern"; then
            matched=1
            local out="$DL_DIR/$name"
            local current_size=""
            if [[ -f "$out" && -n "$size" ]]; then
                current_size="$(wc -c < "$out" | tr -d ' ')"
            fi
            if [[ -f "$out" && -n "$size" && "$current_size" == "$size" ]]; then
                log "Cached $name"
            elif [[ -f "$out" && -z "$size" ]]; then
                log "Cached $name"
            else
                if [[ -f "$out" && -n "$size" && "$current_size" -gt "$size" ]]; then
                    rm -f "$out"
                fi
                download_asset "$name" "$url" "$out"
                if [[ -n "$size" ]]; then
                    current_size="$(wc -c < "$out" | tr -d ' ')"
                    if [[ "$current_size" != "$size" ]]; then
                        err "Downloaded size mismatch for $name: got $current_size, expected $size"
                        return 1
                    fi
                fi
            fi
            case "$name" in
                *.7z.001|*.7z|*.zip|*.tar|*.tar.*|*.tgz)
                    [[ -z "$extract_file" ]] && extract_file="$out"
                    ;;
                *.7z.[0-9][0-9][0-9])
                    ;;
                *) err "Unknown archive type: $name"; return 1 ;;
            esac
        fi
    done <<< "$ASSETS"
    if [[ "$matched" != "1" ]]; then
        err "No asset matched /$pattern/ in release $VSMLRT_TAG."
        return 1
    fi
    if [[ -z "$extract_file" ]]; then
        err "No extractable archive matched /$pattern/ in release $VSMLRT_TAG."
        return 1
    fi

    log "Extracting $(basename "$extract_file") -> $dest"; mkdir -p "$dest"
    local extract_log="$DL_DIR/$(basename "$extract_file").extract.log"
    case "$extract_file" in
        *.7z.001|*.7z|*.zip)
            local sevenz_args=(x -y -o"$dest")
            if [[ -n "$SEVENZ_THREADS" ]]; then
                sevenz_args+=("-mmt=$SEVENZ_THREADS")
            fi
            sevenz_args+=("$extract_file")
            if ! "$SEVENZ" "${sevenz_args[@]}" >"$extract_log" 2>&1; then
                err "Failed to extract $(basename "$extract_file"). 7z log: $extract_log"
                err "If the process was killed, check RAM/swap and disk space; set SEVENZ_THREADS=1 only when you need lower peak memory."
                return 1
            fi
            ;;
        *.tar|*.tar.*|*.tgz)
            if ! tar -xf "$extract_file" -C "$dest" >"$extract_log" 2>&1; then
                err "Failed to extract $(basename "$extract_file"). tar log: $extract_log"
                return 1
            fi
            ;;
        *) err "Unknown archive type: $(basename "$extract_file")"; return 1 ;;
    esac
}

# --- 4. vs-mlrt Linux plugins ----------------------------------------------
log "Fetching vs-mlrt Linux plugins…"
download_and_extract '^vsmlrt-cuda\.' "$PLUGINS_DIR"
find "$PLUGINS_DIR" -name trtexec -exec chmod +x {} \; 2>/dev/null || true

# vsmlrt.py: prefer the local checkout, else the release asset.
if [[ -f "$SCRIPT_DIR/../vs-mlrt/scripts/vsmlrt.py" ]]; then
    cp "$SCRIPT_DIR/../vs-mlrt/scripts/vsmlrt.py" "$PLUGINS_DIR/vsmlrt.py"
    log "Copied vsmlrt.py from local checkout."
else
    download_and_extract 'vsmlrt\.py|scripts' "$PLUGINS_DIR" || true
fi

# --- 5. model packs ---------------------------------------------------------
# vs-mlrt model archives contain a top-level `models/` directory. Extracting
# into MODELS_DIR (.../vs-plugins/models) would nest them as models/models/…,
# which `vsr doctor` / vsmlrt.models_path can't find. Extract, then flatten any
# nested models/ up one level so the layout is MODELS_DIR/<RealESRGANv2|rife|…>.
log "Fetching model packs: $MODEL_PACKS"
for pack in $MODEL_PACKS; do
    download_and_extract "$pack" "$MODELS_DIR"
done
if [[ -d "$MODELS_DIR/models" ]]; then
    log "Flattening nested models/ dir into $MODELS_DIR"
    cp -a "$MODELS_DIR/models/." "$MODELS_DIR/" && rm -rf "$MODELS_DIR/models"
fi

# --- 6. install vsr CLI + write config -------------------------------------
log "Installing vsr CLI into the active environment…"
"$PYTHON" -m pip install -e "$SCRIPT_DIR"

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
[[ "${CREATE_VENV:-0}" == "1" ]] && log "  source $VENV_DIR/bin/activate   # (the venv this script created)"
log "  vsr doctor"
log "  vsr build-engines -i sample.mkv --upscale --model animejanaiV3_HD_L2"
log "  vsr run -i in.mkv -o out.mkv --upscale --model animejanaiV3_HD_L2 --rife --rife-multi 2"
log "Snapshot for reuse:  tar -C '$RUNTIME_DIR' -czf vsr-runtime.tar.gz ."
