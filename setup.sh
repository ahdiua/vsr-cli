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
#   1. pip install vapoursynth + tooling, the source-plugin wheel, and the
#      vs-mlrt TensorRT filter wheel (vapoursynth-mlrt-trt); then run
#      `vapoursynth config` (+ conda libpython fallback + register-install)
#      to set up plugin autoload
#   2. verify a source plugin is present (pip wheel; VSRepo only as fallback)
#   3. deploy vsmlrt.py (local checkout / git ref / release archive)
#   4. RealESRGAN + RIFE model packs
#   5. install the vsr CLI itself + write config.toml
#
# Usage:   bash setup.sh [RUNTIME_DIR]
# Env overrides:
#   VSR_RUNTIME        runtime dir (default: /root/autodl-tmp/vsr-runtime)
#   PY_BIN             python interpreter to use (default: active `python`)
#   ASSUME_YES=1       skip the confirmation prompt (non-interactive)
#   CREATE_VENV=1      create+use a fresh venv at <runtime>/venv instead of
#                      the active env (auto-installs python3.12 if needed)
#   VSMLRT_TAG         vs-mlrt release tag for model packs (default: latest).
#                      Use "prerelease" to pull from the newest prerelease (the
#                      stable "latest" endpoint skips prereleases, where newer
#                      models like RIFE v4.22 may live), or pin an explicit tag.
#   VSMLRT_PY_REF      git ref to fetch vsmlrt.py from (default: master). The
#                      release-bundled script lags the plugin (e.g. v15.16 ships
#                      3.22.38 with no TensorRT-11 handling), so we pull the
#                      script from git instead. Set to a tag/commit to pin.
#   VSMLRT_PY_URL      full URL override for vsmlrt.py (default: derived from REF)
#   FORCE_VSMLRT_PY=1  re-fetch vsmlrt.py from git even if a copy already exists
#   MODEL_PACKS        model asset regexes (default: "^models\\. ^contrib-models\\.")
#   SOURCE_PLUGIN      VSRepo fallback source plugin id(s) (default: "lsmas ffms2")
#   SOURCE_PIP_PACKAGES pip packages for source filters (default: vapoursynth-lsmas)
#   MLRT_TRT_PACKAGE   pip package for the TensorRT VS filter
#   MLRT_TRT_NO_DEPS   install MLRT_TRT_PACKAGE without pip TensorRT deps
#                      (1=force --no-deps, 0=force pip deps). If unset, it is
#                      auto-detected: a system TensorRT (apt/tar/NGC) => --no-deps.
#   VSR_FFMPEG         optional explicit ffmpeg path
#   VSR_FFMPEG_STATIC_URL static ffmpeg tarball URL
#   VSR_FFMPEG_STATIC_DIR static ffmpeg install dir
#   SKIP_STATIC_FFMPEG=1 use PATH/apt ffmpeg instead of the static build
#   SKIP_VSREPO_FALLBACK=1 never use VSRepo fallback for source filters
#   VSR_TRTEXEC        optional explicit Linux trtexec path
#   TENSORRT_HOME      optional TensorRT SDK root (e.g. /usr/src/tensorrt)
#   SKIP_VSMLRT_PY=1   skip deploying vsmlrt.py; requires an existing copy
#   SKIP_MODEL_EXTRACT=1 skip model release download+extract; requires models
#   SKIP_RELEASE_EXTRACT=1 legacy alias: skip both vsmlrt.py and models
#   FORCE_RELEASE_EXTRACT=1 re-extract release archives even if files exist
#   SEVENZ_THREADS     optional 7z extraction threads override
#   SKIP_PYTHON_INSTALL=1   (CREATE_VENV) don't auto-install python3.12
#   SKIP_APT=1         do not use apt at all (assume deps already present)
set -euo pipefail

RUNTIME_DIR="${1:-${VSR_RUNTIME:-/root/autodl-tmp/vsr-runtime}}"
VSMLRT_TAG="${VSMLRT_TAG:-latest}"
MODEL_PACKS="${MODEL_PACKS:-^models\\. ^contrib-models\\.}"
SOURCE_PLUGIN="${SOURCE_PLUGIN:-lsmas ffms2}"
SOURCE_PIP_PACKAGES="${SOURCE_PIP_PACKAGES:-vapoursynth-lsmas}"
MLRT_TRT_PACKAGE="${MLRT_TRT_PACKAGE:-vapoursynth-mlrt-trt}"
SEVENZ_THREADS="${SEVENZ_THREADS:-}"
SKIP_VSMLRT_PY="${SKIP_VSMLRT_PY:-${SKIP_RELEASE_EXTRACT:-0}}"
SKIP_MODEL_EXTRACT="${SKIP_MODEL_EXTRACT:-${SKIP_RELEASE_EXTRACT:-0}}"
VSR_FFMPEG_STATIC_URL="${VSR_FFMPEG_STATIC_URL:-https://github.com/ahdiua/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-nonfree-8.1.tar.xz}"
VSR_FFMPEG_STATIC_DIR="${VSR_FFMPEG_STATIC_DIR:-$RUNTIME_DIR/ffmpeg-static}"
REPO="AmusementClub/vs-mlrt"

VENV_DIR="$RUNTIME_DIR/venv"
PLUGINS_DIR="$RUNTIME_DIR/vs-plugins"
MODELS_DIR="$PLUGINS_DIR/models"
DL_DIR="$RUNTIME_DIR/downloads"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[setup:error]\033[0m %s\n' "$*" >&2; }

mkdir -p "$PLUGINS_DIR" "$MODELS_DIR" "$DL_DIR"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$RUNTIME_DIR/pip-cache}"
export TMPDIR="${TMPDIR:-$RUNTIME_DIR/tmp}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

have() { command -v "$1" >/dev/null 2>&1; }
apt_get() { if [[ "${SKIP_APT:-0}" != "1" ]] && have apt-get; then sudo apt-get "$@"; fi; }

# Detect a system-provided TensorRT (apt/tar/NGC) so we can install the mlrt
# wheel with --no-deps and avoid pip pulling a second, possibly incompatible
# TensorRT. Prints what it found on success. Honors TENSORRT_HOME/TRT_HOME.
detect_system_tensorrt() {
    local d
    if have ldconfig && ldconfig -p 2>/dev/null | grep -q 'libnvinfer\.so'; then
        echo "ldconfig: libnvinfer.so"; return 0
    fi
    for d in "${TENSORRT_HOME:-}/lib" "${TENSORRT_HOME:-}/lib64" \
             "${TRT_HOME:-}/lib" "${TRT_HOME:-}/lib64" \
             /usr/lib/x86_64-linux-gnu /usr/src/tensorrt/lib \
             /usr/local/tensorrt/lib /usr/local/TensorRT/lib; do
        [[ -n "$d" && -d "$d" ]] || continue
        if compgen -G "$d/libnvinfer.so*" >/dev/null 2>&1; then
            echo "$d/libnvinfer.so*"; return 0
        fi
    done
    for d in "${TENSORRT_HOME:-}/bin/trtexec" "${TRT_HOME:-}/bin/trtexec" \
             /usr/src/tensorrt/bin/trtexec; do
        [[ -n "$d" && -x "$d" ]] && { echo "$d"; return 0; }
    done
    have trtexec && { echo "$(command -v trtexec)"; return 0; }
    return 1
}

# --- sanity: GPU ------------------------------------------------------------
if have nvidia-smi; then
    log "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n1)"
else
    err "nvidia-smi not found — the TensorRT backend requires an NVIDIA GPU."
fi

# --- system deps (curl / 7z / ffmpeg) --------------------------------------
NEED_APT=()
NEED_CURL=0
if [[ "$SKIP_VSMLRT_PY" != "1" || "$SKIP_MODEL_EXTRACT" != "1" ]]; then
    NEED_CURL=1
fi
if [[ -z "${VSR_FFMPEG:-}" && "${SKIP_STATIC_FFMPEG:-0}" != "1" ]]; then
    NEED_CURL=1
fi
if [[ "$NEED_CURL" == "1" ]]; then
    have curl || NEED_APT+=(curl)
fi
if [[ "$SKIP_MODEL_EXTRACT" != "1" ]]; then
    { have 7z || have 7za; } || NEED_APT+=(p7zip-full)
fi
# Default to the project's static ffmpeg build instead of apt's ffmpeg package.
# The Ubuntu package pulls extra GUI/X11 dependencies; keep apt ffmpeg only as
# an explicit fallback via SKIP_STATIC_FFMPEG=1.
if [[ -n "${VSR_FFMPEG:-}" && -x "${VSR_FFMPEG}" ]]; then
    log "Using ffmpeg from VSR_FFMPEG: $VSR_FFMPEG (skipping apt ffmpeg)."
elif [[ "${SKIP_STATIC_FFMPEG:-0}" != "1" ]]; then
    have tar || NEED_APT+=(tar)
    have xz || NEED_APT+=(xz-utils)
    log "Using static ffmpeg build by default (skipping apt ffmpeg)."
elif ! have ffmpeg; then
    NEED_APT+=(ffmpeg)
fi
if [[ ${#NEED_APT[@]} -gt 0 ]]; then
    if [[ "${SKIP_APT:-0}" == "1" ]] || ! have apt-get; then
        # apt_get is a silent no-op in this case, so installs wouldn't happen and
        # the install-failure check below would never fire — surface it here.
        err "Missing required tools: ${NEED_APT[*]} — but apt is unavailable (SKIP_APT=1 or no apt-get). Install them manually, then re-run."
        exit 2
    else
        log "Installing system deps: ${NEED_APT[*]}"
        apt_get update -y || true
        # --no-install-recommends: the Ubuntu 'ffmpeg' package otherwise pulls a large
        # GUI stack via Recommends (GTK3, mesa-vulkan, libllvm, icon themes, fonts,
        # va/vdpau driver-all). We only need the codec libs + binary, so skip the
        # recommends. (libavdevice still hard-depends on a few small X11 libs via
        # libsdl2 — unavoidable with apt's ffmpeg.) If you want zero apt bloat, drop a
        # static ffmpeg (e.g. BtbN build) on disk and point VSR_FFMPEG/--ffmpeg at it.
        apt_get install -y --no-install-recommends "${NEED_APT[@]}" \
            || { err "apt install failed for: ${NEED_APT[*]}"; exit 2; }
    fi
fi
SEVENZ="$(command -v 7z || command -v 7za || true)"
if [[ -z "$SEVENZ" && "$SKIP_MODEL_EXTRACT" != "1" ]]; then
    err "7z/7za not found — needed to extract release archives."
    exit 2
fi
ARIA2C="$(command -v aria2c || true)"

resolve_ffmpeg_bin() {
    if [[ -n "${VSR_FFMPEG:-}" ]]; then
        if [[ ! -x "$VSR_FFMPEG" ]]; then
            err "VSR_FFMPEG is set but not executable: $VSR_FFMPEG"
            exit 2
        fi
        FFMPEG_BIN="$VSR_FFMPEG"
        log "ffmpeg: $FFMPEG_BIN (from VSR_FFMPEG)"
        return 0
    fi

    if [[ "${SKIP_STATIC_FFMPEG:-0}" == "1" ]]; then
        FFMPEG_BIN="$(command -v ffmpeg || true)"
        if [[ -z "$FFMPEG_BIN" ]]; then
            err "ffmpeg not found on PATH. Unset SKIP_STATIC_FFMPEG to use the static build, or set VSR_FFMPEG."
            exit 2
        fi
        log "ffmpeg: $FFMPEG_BIN (PATH; SKIP_STATIC_FFMPEG=1)"
        return 0
    fi

    local ffmpeg_bin="$VSR_FFMPEG_STATIC_DIR/bin/ffmpeg"
    if [[ ! -x "$ffmpeg_bin" ]]; then
        local archive_name archive
        archive_name="${VSR_FFMPEG_STATIC_URL##*/}"
        archive_name="${archive_name%%\?*}"
        archive="$DL_DIR/$archive_name"

        if [[ -s "$archive" ]]; then
            log "Cached static ffmpeg archive: $archive"
        else
            log "Downloading static ffmpeg from $VSR_FFMPEG_STATIC_URL"
            curl -fL --retry 3 -C - --speed-limit 1024 --speed-time 120 \
                -o "$archive" "$VSR_FFMPEG_STATIC_URL"
        fi

        mkdir -p "$VSR_FFMPEG_STATIC_DIR"
        if ! tar -xf "$archive" -C "$VSR_FFMPEG_STATIC_DIR" --strip-components=1; then
            err "Failed to extract static ffmpeg archive: $archive"
            exit 2
        fi
        chmod +x "$ffmpeg_bin" || true
    fi

    if [[ ! -x "$ffmpeg_bin" ]]; then
        err "Static ffmpeg binary not found after extraction: $ffmpeg_bin"
        exit 2
    fi
    if ! "$ffmpeg_bin" -hide_banner -encoders >/dev/null 2>&1; then
        err "Static ffmpeg exists but failed to run: $ffmpeg_bin"
        exit 2
    fi

    FFMPEG_BIN="$ffmpeg_bin"
    log "ffmpeg: $FFMPEG_BIN (static build)"
}

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
if [[ "$PY_VER" != "3.1"[2-9] && "$PY_VER" != "3."[2-9][0-9] ]]; then
    err "VapourSynth 需要 Python 3.12+，当前为 $PY_VER。"
    exit 2
fi

if [[ "${ASSUME_YES:-0}" != "1" ]]; then
    read -r -p "确认在以上环境继续安装? 输入 y 继续: " _ans < /dev/tty || _ans=""
    if [[ "$_ans" != "y" && "$_ans" != "Y" ]]; then
        err "已取消。"; exit 1
    fi
fi

resolve_ffmpeg_bin

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

probe_source_filter() {
    "$PYTHON" - <<'PY' 2>/dev/null
from vapoursynth import core

for namespace in ("lsmas", "ffms2"):
    if hasattr(core, namespace):
        plugin = getattr(core, namespace)
        try:
            version = plugin.Version()
        except Exception:
            version = "available"
        print(f"{namespace}: {version}")
        raise SystemExit(0)

raise SystemExit(1)
PY
}

extend_ld_library_path_for_python_libs() {
    local lib_dirs dir updated=0

    if ! lib_dirs="$("$PYTHON" - <<'PY'
import os
import site
import sys
from pathlib import Path

patterns = (
    "libvstrt*.so*",
    "libnvinfer*.so*",
    "libnvinfer_plugin*.so*",
    "libnvonnxparser*.so*",
    "libcudart*.so*",
    "libcublas*.so*",
    "libcudnn*.so*",
    "libnvrtc*.so*",
    "libtensorrt*.so*",
)

roots = []


def add_root(path):
    if not path:
        return
    path = Path(path)
    if path.exists() and path not in roots:
        roots.append(path)


def add_conda_base_roots(base):
    if not base:
        return
    base = Path(base)
    add_root(base / "lib")
    for path in (base / "lib").glob("python*/site-packages/tensorrt_libs"):
        add_root(path)
    for path in (base / "lib").glob("python*/site-packages/nvidia/*/lib"):
        add_root(path)


add_root(Path(sys.prefix) / "lib")
add_root(Path(getattr(sys, "base_prefix", sys.prefix)) / "lib")
for env_name in ("CONDA_PREFIX", "VIRTUAL_ENV"):
    if os.environ.get(env_name):
        add_root(Path(os.environ[env_name]) / "lib")

if os.environ.get("CONDA_PREFIX"):
    conda_prefix = Path(os.environ["CONDA_PREFIX"])
    if conda_prefix.parent.name == "envs":
        add_conda_base_roots(conda_prefix.parent.parent)
if os.environ.get("CONDA_EXE"):
    add_conda_base_roots(Path(os.environ["CONDA_EXE"]).parent.parent)
prefix = Path(sys.prefix)
if prefix.parent.name == "envs":
    add_conda_base_roots(prefix.parent.parent)

for env_name in ("TENSORRT_HOME", "TRT_HOME", "CUDA_HOME", "CUDA_PATH"):
    root = os.environ.get(env_name)
    if root:
        add_root(Path(root) / "lib")
        add_root(Path(root) / "lib64")
        add_root(Path(root) / "targets" / "x86_64-linux-gnu" / "lib")
        add_root(Path(root) / "targets" / "x86_64-linux" / "lib")

for path in (
    "/usr/src/tensorrt/lib",
    "/usr/src/tensorrt/lib64",
    "/usr/src/tensorrt/targets/x86_64-linux-gnu/lib",
    "/usr/src/tensorrt/targets/x86_64-linux/lib",
    "/usr/local/tensorrt/lib",
    "/usr/local/tensorrt/lib64",
    "/usr/local/TensorRT/lib",
    "/usr/local/TensorRT/lib64",
    "/usr/local/cuda/lib64",
    "/usr/local/cuda/targets/x86_64-linux/lib",
    "/usr/local/cuda/targets/x86_64-linux-gnu/lib",
    "/usr/lib/x86_64-linux-gnu",
):
    add_root(path)

try:
    for path in site.getsitepackages():
        add_root(path)
except Exception:
    pass

try:
    add_root(site.getusersitepackages())
except Exception:
    pass

seen = []
for root in roots:
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_file():
                parent = str(path.parent)
                if parent not in seen:
                    seen.append(parent)

for path in seen:
    print(path)
PY
)"; then
        err "Could not inspect Python package library directories."
        return 0
    fi

    while IFS= read -r dir; do
        [[ -z "$dir" ]] && continue
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$dir:"*) ;;
            *)
                export LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                updated=1
                ;;
        esac
    done <<< "$lib_dirs"

    if [[ "$updated" == "1" ]]; then
        log "Updated LD_LIBRARY_PATH for Python package shared libraries."
    fi
}

# --- 1. pip install VapourSynth into the active env ------------------------
log "pip install vapoursynth + tooling…"
"$PYTHON" -m pip install --upgrade pip wheel
"$PYTHON" -m pip install vapoursynth onnx numpy onnxconverter-common
if [[ -n "$SOURCE_PIP_PACKAGES" ]]; then
    log "pip install source plugin wheel(s): $SOURCE_PIP_PACKAGES"
    "$PYTHON" -m pip install --upgrade $SOURCE_PIP_PACKAGES
fi
# Decide whether to let pip pull TensorRT. If MLRT_TRT_NO_DEPS is set
# explicitly, honor it. Otherwise auto-detect a system TensorRT (apt/tar/NGC)
# and, when found, install with --no-deps so pip does not add a second,
# possibly version-incompatible TensorRT on top of it.
if [[ -z "${MLRT_TRT_NO_DEPS+x}" ]]; then
    if trt_found="$(detect_system_tensorrt)"; then
        MLRT_TRT_NO_DEPS=1
        log "Detected system TensorRT ($trt_found); will skip pip TensorRT deps."
    else
        MLRT_TRT_NO_DEPS=0
        log "No system TensorRT detected; pip will resolve TensorRT for $MLRT_TRT_PACKAGE."
    fi
fi
log "pip install $MLRT_TRT_PACKAGE…"
MLRT_TRT_PIP_ARGS=(--upgrade)
if [[ "${MLRT_TRT_NO_DEPS:-0}" == "1" ]]; then
    MLRT_TRT_PIP_ARGS+=(--no-deps)
    log "Using system TensorRT libraries; installing $MLRT_TRT_PACKAGE with --no-deps."
fi
if ! "$PYTHON" -m pip install "${MLRT_TRT_PIP_ARGS[@]}" "$MLRT_TRT_PACKAGE"; then
    err "Could not install $MLRT_TRT_PACKAGE."
    err "Continuing; vsr doctor will report whether core.trt is available."
fi
extend_ld_library_path_for_python_libs

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

# --- 2. source plugin check; VSRepo only as fallback ------------------------
SRC_OK=0
if SOURCE_AVAILABLE="$(probe_source_filter)"; then
    log "Source plugin available: $SOURCE_AVAILABLE"
    SRC_OK=1
elif [[ "${SKIP_VSREPO_FALLBACK:-0}" == "1" ]]; then
    err "No source plugin detected after pip install, and SKIP_VSREPO_FALLBACK=1."
else
    log "No pip/autoloaded source plugin detected; trying VSRepo fallback: $SOURCE_PLUGIN"
    "$PYTHON" -m pip install vsrepo
    ensure_python_user_site_dir
    run_script vsrepo update || err "'vsrepo update' failed (continuing)."
    for sp in $SOURCE_PLUGIN; do
        if run_vsrepo_install "$sp"; then
            log "VSRepo source plugin available: $sp"
            SRC_OK=1
            break
        else
            err "VSRepo could not install '$sp' on this platform — trying next."
        fi
    done
fi
if [[ "$SRC_OK" != "1" ]]; then
    err "No source plugin detected. Install vapoursynth-lsmas or ffms2 before running vsr."
fi

# --- GitHub release asset helpers ------------------------------------------
release_assets_loaded=0

api_url() {
    case "$VSMLRT_TAG" in
        latest)
            # GitHub's /releases/latest endpoint skips prereleases.
            echo "https://api.github.com/repos/$REPO/releases/latest" ;;
        prerelease)
            # List endpoint (newest first) so we can pick the newest prerelease;
            # newer models (e.g. RIFE v4.22) often land in a prerelease first.
            echo "https://api.github.com/repos/$REPO/releases?per_page=30" ;;
        *)
            echo "https://api.github.com/repos/$REPO/releases/tags/$VSMLRT_TAG" ;;
    esac
}
load_release_assets() {
    if [[ "$release_assets_loaded" == "1" ]]; then
        return 0
    fi

    local auth=()
    [[ -n "${GITHUB_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer $GITHUB_TOKEN")
    ASSETS="$(curl -fsSL "${auth[@]}" "$(api_url)" | "$PYTHON" -c '
import json
import sys

d = json.load(sys.stdin)
# VSMLRT_TAG=prerelease hits the list endpoint (an array, newest first): pick the
# newest prerelease, falling back to the newest release overall. Tag endpoints
# return a single release object.
if isinstance(d, list):
    rel = next((r for r in d if r.get("prerelease")), d[0] if d else None)
    if rel is None:
        sys.exit("no releases found")
    sys.stderr.write("resolved prerelease -> tag %s\n" % rel.get("tag_name"))
    d = rel
for a in d.get("assets", []):
    print(a["name"] + "\t" + a["browser_download_url"] + "\t" + str(a.get("size", "")))
')" || { err "Failed to query GitHub release assets."; return 1; }
    release_assets_loaded=1
    log "vs-mlrt release '$VSMLRT_TAG': $(printf '%s\n' "$ASSETS" | grep -c . ) assets."
}

download_asset() {
    local name="$1" url="$2" out="$3"

    log "Downloading $name"
    log "Manual download target: $out"
    log "Manual download URL: $url"

    if [[ -n "$ARIA2C" ]]; then
        "$ARIA2C" -c -x 4 -s 4 --summary-interval=30 --file-allocation=none \
            -d "$(dirname "$out")" -o "$(basename "$out")" "$url"
    else
        curl -fL --retry 3 -C - --speed-limit 1024 --speed-time 120 -o "$out" "$url"
    fi
}

download_and_extract() {
    local pattern="$1" dest="$2" matched=0 extract_file=""
    load_release_assets || return 1
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
            if [[ -z "$SEVENZ" ]]; then
                err "7z/7za not found — needed to extract $(basename "$extract_file")."
                return 1
            fi
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

# --- 3. vsmlrt.py script (filter backend comes from pip wheel) -------------
# Do not download vsmlrt-cuda.* here: current release assets are Windows
# .dll/.exe bundles and are useless for the Linux TensorRT wheel route.
#
# The vsmlrt.py bundled in the vs-mlrt *release* .7z lags the plugin: e.g. the
# v15.16 release ships script 3.22.38, which has no TensorRT-11 handling and
# emits TRT-10-only trtexec args (--inputIOFormats=fp16:chw, --useCudaGraph,
# --noTF32) that TRT 11 rejects ("Invalid TensorFormat fp16:chw"). The pip TRT
# wheel is built against TRT 11, so the two are mismatched. We therefore fetch
# vsmlrt.py from a git ref (VSMLRT_PY_REF, default master) that tracks the
# plugin and version-gates those args, instead of the stale release archive.
VSMLRT_PY_REF="${VSMLRT_PY_REF:-master}"
VSMLRT_PY_URL="${VSMLRT_PY_URL:-https://raw.githubusercontent.com/$REPO/${VSMLRT_PY_REF}/scripts/vsmlrt.py}"

fetch_vsmlrt_py() {
    log "Fetching vsmlrt.py from $VSMLRT_PY_URL"
    if curl -fL "$VSMLRT_PY_URL" -o "$PLUGINS_DIR/vsmlrt.py.tmp" \
            && grep -q '__version__' "$PLUGINS_DIR/vsmlrt.py.tmp"; then
        mv "$PLUGINS_DIR/vsmlrt.py.tmp" "$PLUGINS_DIR/vsmlrt.py"
        log "Installed vsmlrt.py (ref: $VSMLRT_PY_REF)."
        return 0
    fi
    rm -f "$PLUGINS_DIR/vsmlrt.py.tmp"
    return 1
}

if [[ "$SKIP_VSMLRT_PY" == "1" ]]; then
    log "SKIP_VSMLRT_PY=1: skipping vsmlrt.py deployment."
elif [[ -f "$SCRIPT_DIR/../vs-mlrt/scripts/vsmlrt.py" ]]; then
    # A local version-controlled checkout is the most authoritative source for a
    # pinned setup, so it wins over both the deployed copy and git. Copying is
    # idempotent, so this self-heals on every re-run.
    cp "$SCRIPT_DIR/../vs-mlrt/scripts/vsmlrt.py" "$PLUGINS_DIR/vsmlrt.py"
    log "Copied vsmlrt.py from local checkout."
elif [[ -f "$PLUGINS_DIR/vsmlrt.py" && "${FORCE_VSMLRT_PY:-0}" != "1" && "${FORCE_RELEASE_EXTRACT:-0}" != "1" ]]; then
    log "Existing vsmlrt.py found (no local checkout); set FORCE_VSMLRT_PY=1 to refresh from $VSMLRT_PY_REF."
else
    # No local checkout: pull the (TRT-11-aware) script from git, falling back to
    # the release archive only if the network fetch fails.
    fetch_vsmlrt_py || {
        log "git fetch failed; falling back to vs-mlrt release archive for vsmlrt.py."
        download_and_extract 'vsmlrt\.py|scripts' "$PLUGINS_DIR" || true
    }
fi

# --- 4. model packs ---------------------------------------------------------
# vs-mlrt model archives contain a top-level `models/` directory. Extracting
# into MODELS_DIR (.../vs-plugins/models) would nest them as models/models/…,
# which `vsr doctor` / vsmlrt.models_path can't find. Extract, then flatten any
# nested models/ up one level so the layout is MODELS_DIR/<RealESRGANv2|rife|…>.
if [[ "$SKIP_MODEL_EXTRACT" == "1" ]]; then
    :
elif [[ -d "$MODELS_DIR/RealESRGANv2" && -d "$MODELS_DIR/rife" && "${FORCE_RELEASE_EXTRACT:-0}" != "1" ]]; then
    log "Existing model directories found; skipping model archive extraction."
else
    log "Fetching model packs: $MODEL_PACKS"
    for pack in $MODEL_PACKS; do
        download_and_extract "$pack" "$MODELS_DIR"
    done
    if [[ -d "$MODELS_DIR/models" ]]; then
        log "Flattening nested models/ dir into $MODELS_DIR"
        cp -a "$MODELS_DIR/models/." "$MODELS_DIR/" && rm -rf "$MODELS_DIR/models"
    fi
fi

# --- 5. install vsr CLI + write config -------------------------------------
if [[ ! -f "$PLUGINS_DIR/vsmlrt.py" ]]; then
    err "Missing $PLUGINS_DIR/vsmlrt.py. Re-run without SKIP_VSMLRT_PY=1 or set FORCE_VSMLRT_PY=1."
    exit 2
fi
if [[ ! -d "$MODELS_DIR/RealESRGANv2" || ! -d "$MODELS_DIR/rife" ]]; then
    err "Missing model directories under $MODELS_DIR. Re-run without SKIP_MODEL_EXTRACT=1."
    exit 2
fi

log "Installing vsr CLI into the active environment…"
"$PYTHON" -m pip install -e "$SCRIPT_DIR"

CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/vsr"
mkdir -p "$CONFIG_DIR"
CONFIG_FILE="$CONFIG_DIR/config.toml"
PIPELINE_VPY="$SCRIPT_DIR/pipeline.vpy"
# trtexec detection (candidate paths, TensorRT version probe, and engine-ABI
# matching against core.trt) lives in vsr.config — the single source of truth
# the CLI itself uses at runtime. The vsr package is installed just above, so we
# resolve through it instead of duplicating the logic in bash. It honors
# VSR_TRTEXEC and returns "" when no compatible trtexec is found.
TRTEXEC_BIN="$("$PYTHON" - <<PY 2>/dev/null || true
from vsr.config import RuntimeConfig, resolve
print(resolve(RuntimeConfig(plugins_dir="$PLUGINS_DIR", models_dir="$MODELS_DIR")).trtexec or "")
PY
)"
if [[ -n "$TRTEXEC_BIN" ]]; then
    log "trtexec: $TRTEXEC_BIN"
else
    err "Compatible trtexec not found. build-engines requires trtexec TensorRT to match core.trt."
    err "Install a matching Linux TensorRT CLI or set VSR_TRTEXEC to the matching binary. (Run 'vsr doctor' for the exact mismatch.)"
fi

{
    printf 'vspipe = "%s"\n' "$VSPIPE_BIN"
    printf 'ffmpeg = "%s"\n' "$FFMPEG_BIN"
    printf 'plugins_dir = "%s"\n' "$PLUGINS_DIR"
    printf 'models_dir = "%s"\n' "$MODELS_DIR"
    printf 'pipeline_vpy = "%s"\n' "$PIPELINE_VPY"
    if [[ -n "$TRTEXEC_BIN" ]]; then
        printf 'trtexec = "%s"\n' "$TRTEXEC_BIN"
    fi
    printf 'encoder = "nvenc"\n'
    printf 'num_streams = 2\n'
    printf 'device_id = 0\n'
    printf 'fp16 = true\n'
} > "$CONFIG_FILE"

log "Wrote config: $CONFIG_FILE"
log "Runtime ready at: $RUNTIME_DIR"
log ""
log "Next steps:"
[[ "${CREATE_VENV:-0}" == "1" ]] && log "  source $VENV_DIR/bin/activate   # (the venv this script created)"
log "  vsr doctor"
log "  vsr build-engines -i sample.mkv --upscale --model animejanaiV3_HD_L2"
log "  vsr run -i in.mkv -o out.mkv --upscale --model animejanaiV3_HD_L2 --rife --rife-multi 2"
log "Snapshot for reuse:  tar -C '$RUNTIME_DIR' -czf vsr-runtime.tar.gz ."
