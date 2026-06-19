# vsr-cli

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![TensorRT 11](https://img.shields.io/badge/TensorRT-11-green)](https://developer.nvidia.com/tensorrt)
[![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)](https://github.com/)

> English | [**中文**](README.md)

CLI tool for video **super-resolution (Real-ESRGAN)** and **frame interpolation (RIFE)**. Runs inference on TensorRT via [vs-mlrt](https://github.com/AmusementClub/vs-mlrt), pipes VapourSynth (`vspipe`) directly to ffmpeg for encoding — zero intermediate files on disk. Inspired by [VideoJaNai](https://github.com/the-database/VideoJaNai) 1.x.

## Features

- **Three processing modes** (`--upscale` / `--rife`, freely combined): upscale only, interpolate only, upscale + interpolation
- **Interactive TUI wizard** (run `vsr` with no arguments) or CLI batch processing
- Audio / subtitle / attachment tracks copied from the source — only video is re-encoded
- TensorRT engines auto-built and cached; `vsr build-engines` for warm-up
- Multiple encoder presets: nvenc / x265 / x264 / ffv1
- Container cgroup-aware thread count auto-detection to prevent oversubscription
- [akarin](https://pypi.org/project/vapoursynth-akarin/) JIT-accelerated RIFE scene-change handling

## Architecture

```
vsr (CLI / TUI)
  └─ Locate runtime (config.toml / env / auto-detect)
       └─ vspipe -c y4m --arg ... pipeline.vpy -  │  ffmpeg -i pipe: -i source ...
            (VapourSynth: RealESRGAN@TRT → RIFE@TRT)   (re-encode video only, copy audio/subs)
```

---

## Prerequisites

| Dependency | Requirement |
| --- | --- |
| OS | Linux (Ubuntu 22.04+ recommended; GPU containers like AutoDL / RunPod work out of the box) |
| Python | 3.12+ (venv or conda) |
| GPU | NVIDIA (CUDA architecture ≥ Turing) |
| Driver | ≥ 535 (CUDA 12+ support); NVENC encoding requires matching API version |
| TensorRT | **11.0.0** (see installation below) |
| curl | Used by setup.sh to download dependencies |

### Install TensorRT 11 (do this first)

The `vapoursynth-mlrt-trt` wheel's `core.trt` is built against **TensorRT 11.0.0 + CUDA 13** (loads `libnvinfer.so.11` at runtime). The system must have the matching version, otherwise you'll get `libnvinfer.so.11: cannot open shared object file`.

```bash
# 1. Register the NVIDIA CUDA apt repo (Ubuntu 24.04 example; use ubuntu2204 for 22.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update

# 2. Check available version
apt-cache madison libnvinfer11        # e.g. 11.0.0.114-1+cuda13.2

# 3. Install (pin all packages to the same version to avoid pulling 11.1)
version="11.0.0.114-1+cuda13.2"
apt install -y \
    libnvinfer11=${version} \
    libnvinfer-bin=${version} \
    libnvinfer-plugin11=${version} \
    libnvonnxparsers11=${version} \
    libnvinfer-lean11=${version} \
    libnvinfer-vc-plugin11=${version} \
    libnvinfer-dispatch11=${version}

# 4. Verify
trtexec --version
ldconfig -p | grep -E 'libnvinfer.so|libcudart.so.13|libcublas.so.13'
```

<details>
<summary>What each package provides</summary>

| Package | Provides | Why it's needed |
| --- | --- | --- |
| `libnvinfer11` | `libnvinfer.so.11` | Directly linked by `core.trt` |
| `libnvinfer-bin` | `trtexec` | `vsr build-engines` builds engines with it |
| `libnvinfer-plugin11` | `libnvinfer_plugin.so.11` | trtexec dependency |
| `libnvonnxparsers11` | `libnvonnxparser.so.11` | trtexec uses it to parse ONNX |
| `libnvinfer-lean11` / `-vc-plugin11` / `-dispatch11` | — | Hard dependencies of `libnvinfer-bin`; must be pinned to the same version |

Not needed: `tensorrt` / `tensorrt-dev` / `tensorrt-libs` meta-packages, all `*-dev` / `*-headers-*`, `libnvinfer-safe-*`, `python3-libnvinfer*`.

</details>

<details>
<summary>CUDA 13 runtime libraries</summary>

TRT 11.0.0 `+cuda13.2` depends on CUDA 13's `cudart/cublas`; apt pulls these in automatically with `libnvinfer11`. Your driver only needs to support CUDA 13.0 (`nvidia-smi` header version); the 13.2 runtime libraries are compatible within the same major version.

Only manually install if `ldconfig` shows no `libcudart.so.13`/`libcublas.so.13`: `apt install -y cuda-cudart-13-2 libcublas-13-2`.

</details>

> **Already have TensorRT?** `setup.sh` auto-detects system TensorRT (`ldconfig` / `TENSORRT_HOME` / `trtexec`) and passes `--no-deps` to the mlrt wheel to avoid pulling a second copy from pip.

---

## Installation

> VideoJaNai's `backend` is a Windows portable package that doesn't run on Linux. This project installs VapourSynth / L-SMASH / vs-mlrt TensorRT filter via PyPI wheels.

```bash
# 1. Set up a Python environment
python3.12 -m venv ~/vsr-venv && source ~/vsr-venv/bin/activate

# 2. One-command runtime install
cd vsr-cli
bash setup.sh            # Defaults to /root/autodl-tmp/vsr-runtime

# 3. Health check + warm up engines
vsr doctor
vsr build-engines -i sample.mkv --upscale --model animejanaiV3-HD-L2.onnx --rife --rife-multi 2
```

> `setup.sh` prints the target Python/environment and asks for `y` confirmation; it warns extra if running on bare system Python (neither venv nor conda).

<details>
<summary>What setup.sh does</summary>

1. Print target environment and confirm
2. `pip install vapoursynth onnx numpy onnxconverter-common`
3. Install source plugin wheel (default `vapoursynth-lsmas`) and `vapoursynth-mlrt-trt`
4. Install `vapoursynth-akarin` (JIT acceleration for RIFE scene-change detection)
5. `vapoursynth config` + `register-install` to configure plugin auto-loading
6. Verify `core.lsmas`/`core.ffms2`; fall back to VSRepo if needed
7. Deploy `vsmlrt.py`; download model packs (RealESRGAN / RIFE)
8. `pip install -e` the `vsr` package; verify runtime files and write `~/.config/vsr/config.toml`

The runtime directory (plugins + models) can be tarred and reused; Python dependencies live in your venv.

</details>

---

## Quick Start

```bash
# Upscale only (2x)
vsr run -i in.mkv -o out.mkv --upscale --model animejanaiV3-HD-L2.onnx

# Interpolate only (2x frame rate)
vsr run -i in.mkv -o out.mkv --rife --rife-multi 2

# Upscale + interpolation
vsr run -i in.mkv -o out.mkv --upscale --rife --rife-multi 2

# Batch processing (recursive)
vsr batch -i ./in_dir -o ./out_dir --recursive --upscale

# Interactive wizard
vsr
```

---

## CLI Reference

### Subcommands

| Command | Purpose |
| --- | --- |
| `vsr run` | Process a single file |
| `vsr batch` | Batch-process a directory |
| `vsr build-engines` | Pre-build TensorRT engines |
| `vsr doctor` | Check runtime health |
| `vsr setup` | Provision the Linux runtime |

### Processing Arguments

| Argument | Description |
| --- | --- |
| `--upscale` | Enable R-ESRGAN super-resolution |
| `--model` | Upscale model filename or path (default `animejanaiV3-HD-L2.onnx`; relative to `models/RealESRGANv2`) |
| `--pre-resize-factor` | Pre-upscale resize percentage (e.g. `50`) |
| `--rife` | Enable RIFE frame interpolation |
| `--rife-multi` | Interpolation multiplier (e.g. `2`, `2/1`; default `2`) |
| `--rife-model` | RIFE model filename or path (default `rife_v4.10.onnx`) |
| `--final-resize-height` | Final output height in pixels |
| `--encoder` | Encoder preset: `nvenc`(default) / `x265` / `x264` / `ffv1` |
| `--ffmpeg-args` | Custom ffmpeg video args string, overrides `--encoder` |

### Runtime Arguments

| Argument | Description |
| --- | --- |
| `--vspipe` | Path to vspipe executable |
| `--ffmpeg` | Path to ffmpeg executable |
| `--plugins-dir` | VapourSynth plugins directory |
| `--models-dir` | Models directory (containing `RealESRGANv2/` and `rife/`) |
| `--trtexec` | Path to trtexec executable |
| `--device-id` | GPU device ID |
| `--num-streams` | TensorRT concurrent streams (default 4) |
| `--num-threads` | VapourSynth worker threads (default 0 = auto-detect from container cgroup quota) |
| `--no-fp16` | Disable fp16 |
| `--nvenc-fix` | Path to NVENC GPU enumeration fix library (LD_PRELOAD injected into ffmpeg) |

Run `vsr <subcommand> -h` for full options.

<details>
<summary>Model argument rules</summary>

- `--model animejanaiV3-HD-L2.onnx` → `<models_dir>/RealESRGANv2/animejanaiV3-HD-L2.onnx`
- `--rife-model rife_v4.10.onnx` → `<models_dir>/rife/rife_v4.10.onnx`
- Absolute paths and paths relative to `models_dir` also work, e.g. `--model RealESRGANv2/custom.onnx`
- Legacy vs-mlrt enum names are still supported, e.g. `--model animejanaiV3_HD_L2` / `--rife-model v4_10`
- If a custom `.onnx` is already fp16, the pipeline builds the TensorRT engine using the model's native I/O types without running vsmlrt's fp16 conversion

</details>

---

## Encoder Presets

| Name | ffmpeg arguments |
| --- | --- |
| `nvenc` | `hevc_nvenc -preset p5 -profile:v main10 -split_encode_mode forced -b:v 50M` |
| `x265` | `libx265 -crf 16 -preset slow -x265-params "sao=0:bframes=8:psy-rd=1.5:psy-rdoq=2:aq-mode=3:ref=6"` |
| `x264` | `libx264 -crf 13 -preset slow` |
| `ffv1` | Lossless |

---

## Configuration

### Priority

```
CLI args > Environment variables > ~/.config/vsr/config.toml > Auto-detect
```

Environment variables: `VSR_VSPIPE` / `VSR_FFMPEG` / `VSR_PLUGINS` / `VSR_MODELS` / `VSR_TRTEXEC` / `VSR_PIPELINE` / `VSR_NUM_THREADS` / `VSR_NVENC_FIX`

<details>
<summary>setup.sh environment variables</summary>

| Variable | Purpose |
| --- | --- |
| `ASSUME_YES=1` | Skip confirmation prompt (non-interactive) |
| `CREATE_VENV=1` | Create a venv in `<runtime>/venv` |
| `PY_BIN=/path/python` | Specify Python interpreter |
| `VSR_RUNTIME=/path` | Runtime directory |
| `VSR_FFMPEG=/path/ffmpeg` | Explicit ffmpeg path |
| `VSR_FFMPEG_STATIC_URL=https://...tar.xz` | Static ffmpeg download URL when PATH has no ffmpeg |
| `VSR_FFMPEG_INSTALL_DIR=/usr/local/bin` | Install directory for downloaded ffmpeg |
| `SKIP_STATIC_FFMPEG=1` | Don't download static build; use PATH/apt ffmpeg only |
| `VSMLRT_TAG=v15.x` | vs-mlrt release tag (`prerelease` pulls from latest pre-release) |
| `VSMLRT_PY_REF=master` | Git ref for fetching `vsmlrt.py` |
| `FORCE_VSMLRT_PY=1` | Force re-fetch `vsmlrt.py` |
| `MODEL_PACKS="^models\\. ^contrib-models\\."` | Release asset regex for model downloads |
| `SOURCE_PIP_PACKAGES="vapoursynth-lsmas"` | pip source plugin packages |
| `SKIP_AKARIN=1` | Skip akarin installation |
| `MLRT_TRT_PACKAGE="vapoursynth-mlrt-trt"` | TRT filter pip package |
| `MLRT_TRT_NO_DEPS=1` | Install filter wheel only, don't pull TensorRT libs |
| `TENSORRT_HOME=/usr/src/tensorrt` | TensorRT SDK root directory |
| `SKIP_MODEL_EXTRACT=1` | Skip model download/extraction |
| `GITHUB_TOKEN=...` | Increase GitHub API rate limit |
| `SKIP_APT=1` | Disable apt installs |

</details>

---

## ffmpeg

`setup.sh` reuses ffmpeg from the system PATH first. If none is found, it downloads [BtbN](https://github.com/BtbN/FFmpeg-Builds)'s n8.1 static build and installs to `/usr/local/bin/ffmpeg`.

> **Driver compatibility note**: `master` builds may require nvenc SDK API 13.1+ / driver 610+. AutoDL drivers are often older (e.g. 580.x = API 13.0) — use the **n8.1** release, not master.

Set `VSR_FFMPEG` to use a specific ffmpeg path. Set `SKIP_STATIC_FFMPEG=1` with no ffmpeg in PATH to fall back to apt install.

---

## Troubleshooting

<details>
<summary><code>libnvinfer_plugin.so.11</code> not found</summary>

Check `LD_LIBRARY_PATH` and the TensorRT `.so` directory in pip/conda. Some conda images place `.so` files in the base environment's `lib/python*/site-packages/tensorrt_libs`.

</details>

<details>
<summary>trtexec version mismatch</summary>

`trtexec` must share the same TensorRT engine ABI as `core.trt`. If `core.trt` uses TRT 11.0 but `/usr/src/tensorrt/bin/trtexec` is 10.7, generated engines won't deserialize. `vsr doctor` flags mismatches. Point to the correct version with `VSR_TRTEXEC` / `--trtexec`.

</details>

<details>
<summary>Container <code>hevc_nvenc</code> reports <code>unsupported device</code></summary>

NVIDIA driver 570+ bug on multi-GPU hosts: the container's NVENC sees all host GPUs but can only access the one assigned to it. Fix with [flexgrip/nvidia-gpu-enumeration](https://github.com/flexgrip/nvidia-gpu-enumeration)'s `LD_PRELOAD` shim:

```bash
# Build the shim
gcc -shared -fPIC -O2 -o /opt/libnvenc_fix.so nvenc_fix.c -ldl

# Point vsr to it
vsr run -i in.mkv -o out.mkv --upscale --nvenc-fix /opt/libnvenc_fix.so
# Or: export VSR_NVENC_FIX=/opt/libnvenc_fix.so
```

vsr only injects the library into ffmpeg's `LD_PRELOAD`. If the environment doesn't expose NVENC (`$NVIDIA_DRIVER_CAPABILITIES` lacks `video`/`all`), the shim won't help — use `--encoder x265` instead.

</details>

<details>
<summary><code>Invalid TensorFormat fp16:chw</code> (TRT 11 build failure)</summary>

`vsmlrt.py` is too old and sends TRT-10 parameters to TRT 11. Fix:

```bash
# Manual update
curl -fL https://raw.githubusercontent.com/AmusementClub/vs-mlrt/master/scripts/vsmlrt.py \
  -o /root/autodl-tmp/vsr-runtime/vs-plugins/vsmlrt.py

# Or: auto-fix via setup.sh
FORCE_VSMLRT_PY=1 bash setup.sh
```

</details>

<details>
<summary>Stale TensorRT engines</summary>

Delete old cached engines and rebuild:

```bash
rm /root/autodl-tmp/vsr-runtime/vs-plugins/models/RealESRGANv2/*.engine*
vsr build-engines -i sample.mkv --upscale --model animejanaiV3-HD-L2.onnx
```

</details>

<details>
<summary>Newer RIFE models not in latest release</summary>

Newer RIFE models (e.g. v4.22+) may only be available in pre-releases:

```bash
VSMLRT_TAG=prerelease FORCE_RELEASE_EXTRACT=1 bash setup.sh
```

</details>

---

## Notes

- First TRT engine build for each `resolution × model` combo takes several minutes; use `build-engines` to do it ahead of time
- Source plugin defaults to `vapoursynth-lsmas`; falls back to `ffms2` if needed
- `tensorrt-cu13-libs` is several GB; if root partition is small, create your venv on a persistent disk
- **VideoJaNai relationship**: vsr-cli follows the 1.x approach (VapourSynth + vs-mlrt + `vspipe | ffmpeg` pipe). VideoJaNai 2.0 has been rewritten as a GPU-resident native engine with no Python backend

> For more details, edge cases, and debugging notes, see [NOTES.md](NOTES.md).

---

## License

[GPL-3.0](LICENSE)
