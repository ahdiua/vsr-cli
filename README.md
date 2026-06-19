# vsr-cli

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![TensorRT 11](https://img.shields.io/badge/TensorRT-11-green)](https://developer.nvidia.com/tensorrt)
[![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)](https://github.com/)

> [**English**](README_EN.md) | 中文

视频**超分辨率 (Real-ESRGAN)** 与**帧插值 (RIFE)** 命令行工具。基于 [vs-mlrt](https://github.com/AmusementClub/vs-mlrt) 在 TensorRT 上推理，VapourSynth (`vspipe`) 通过管道直接喂给 ffmpeg 编码，零中间文件落盘。灵感来自 [VideoJaNai](https://github.com/the-database/VideoJaNai) 1.x。

## 特性

- **三种处理模式**（`--upscale` / `--rife` 自由组合）：仅超分、仅插帧、超分 + 插帧
- **交互式 TUI 向导**（无参数运行 `vsr`）或 CLI 批处理
- 音轨 / 字幕 / 附件从原片复制，仅视频重编码
- TensorRT engine 自动构建并缓存，支持 `vsr build-engines` 预热
- 多种编码预设：nvenc / x265 / x264 / ffv1
- 容器 cgroup 感知线程数自动探测，避免过载
- [akarin](https://pypi.org/project/vapoursynth-akarin/) JIT 加速 RIFE 场景切换判断

## 架构

```
vsr (CLI / TUI)
  └─ 定位运行时 (config.toml / env / 自动探测)
       └─ vspipe -c y4m --arg ... pipeline.vpy -  │  ffmpeg -i pipe: -i 原片 ...
            (VapourSynth: RealESRGAN@TRT → RIFE@TRT)   (仅重编码视频, 复制音轨/字幕)
```

---

## 前置条件

| 依赖 | 要求 |
| --- | --- |
| OS | Linux（Ubuntu 22.04+ 推荐；AutoDL / RunPod 等 GPU 容器直接可用） |
| Python | 3.12+（venv 或 conda） |
| GPU | NVIDIA（CUDA 架构 ≥ Turing） |
| 驱动 | ≥ 535（支持 CUDA 12+）；NVENC 编码需驱动支持对应 API 版本 |
| TensorRT | **11.0.0**（见下方安装） |
| curl | setup.sh 下载依赖用 |

### 安装 TensorRT 11（必须先做）

`vapoursynth-mlrt-trt` wheel 的 `core.trt` 按 **TensorRT 11.0.0 + CUDA 13** 编译（加载时找 `libnvinfer.so.11`）。系统侧需要匹配版本，否则报 `libnvinfer.so.11: cannot open shared object file`。

```bash
# 1. 注册 NVIDIA CUDA apt 源（Ubuntu 24.04 示例；22.04 换 ubuntu2204）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update

# 2. 确认版本
apt-cache madison libnvinfer11        # 形如 11.0.0.114-1+cuda13.2

# 3. 安装（全部 pin 同版本，避免被拉到 11.1）
version="11.0.0.114-1+cuda13.2"
apt install -y \
    libnvinfer11=${version} \
    libnvinfer-bin=${version} \
    libnvinfer-plugin11=${version} \
    libnvonnxparsers11=${version} \
    libnvinfer-lean11=${version} \
    libnvinfer-vc-plugin11=${version} \
    libnvinfer-dispatch11=${version}

# 4. 验证
trtexec --version
ldconfig -p | grep -E 'libnvinfer.so|libcudart.so.13|libcublas.so.13'
```

<details>
<summary>各包作用说明</summary>

| 包 | 提供 | 必要原因 |
| --- | --- | --- |
| `libnvinfer11` | `libnvinfer.so.11` | `core.trt` 直接链接 |
| `libnvinfer-bin` | `trtexec` | `vsr build-engines` 构建 engine |
| `libnvinfer-plugin11` | `libnvinfer_plugin.so.11` | trtexec 依赖 |
| `libnvonnxparsers11` | `libnvonnxparser.so.11` | trtexec 解析 ONNX |
| `libnvinfer-lean11` / `-vc-plugin11` / `-dispatch11` | — | `libnvinfer-bin` 的强制依赖，必须同版本 pin |

可省略：`tensorrt` / `tensorrt-dev` / `tensorrt-libs` meta 包、所有 `*-dev` / `*-headers-*`、`libnvinfer-safe-*`、`python3-libnvinfer*`。

</details>

<details>
<summary>CUDA 13 运行库说明</summary>

TRT 11.0.0 `+cuda13.2` 构建依赖 CUDA 13 的 `cudart/cublas`，apt 会随 `libnvinfer11` 自动拉进来。驱动只需支持 CUDA 13.0（`nvidia-smi` 顶部的版本号），13.2 运行库靠同主版本兼容即可。

仅当 `ldconfig` 没有 `libcudart.so.13`/`libcublas.so.13` 时手动补：`apt install -y cuda-cudart-13-2 libcublas-13-2`。

</details>

> **已有 TensorRT？** `setup.sh` 会自动检测系统 TensorRT（`ldconfig` / `TENSORRT_HOME` / `trtexec`），检测到则对 mlrt wheel 自动加 `--no-deps`，不从 pip 拉第二份 TensorRT。

---

## 安装

> VideoJaNai 的 `backend` 是 Windows 便携包，无法直接在 Linux 运行。本项目通过 PyPI wheel 安装 VapourSynth / L-SMASH / vs-mlrt TensorRT filter。

```bash
# 1. 准备 Python 环境
python3.12 -m venv ~/vsr-venv && source ~/vsr-venv/bin/activate

# 2. 一键安装运行时
cd vsr-cli
bash setup.sh            # 默认装到 /root/autodl-tmp/vsr-runtime

# 3. 自检 + 预热 engine
vsr doctor
vsr build-engines -i sample.mkv --upscale --model animejanaiV3-HD-L2.onnx --rife --rife-multi 2
```

> `setup.sh` 启动时打印目标 Python/环境并要求输入 `y` 确认；仅当是裸的系统 Python（既非 venv 也非 conda）时额外警告。

<details>
<summary>setup.sh 执行流程</summary>

1. 打印目标环境并确认
2. `pip install vapoursynth onnx numpy onnxconverter-common`
3. 安装源插件 wheel（默认 `vapoursynth-lsmas`）和 `vapoursynth-mlrt-trt`
4. 安装 `vapoursynth-akarin`（RIFE 场景切换 JIT 加速）
5. `vapoursynth config` + `register-install` 配置插件自动加载
6. 验证 `core.lsmas`/`core.ffms2`，失败时才尝试 VSRepo fallback
7. 部署 `vsmlrt.py`，下载模型包（RealESRGAN / RIFE）
8. `pip install -e` 安装 `vsr` 本体，验证运行时文件并写 `~/.config/vsr/config.toml`

运行时目录（插件 + 模型）可 `tar` 打包复用；Python 依赖在你自己的 venv 中。

</details>

---

## 快速开始

```bash
# 仅超分 (2x)
vsr run -i in.mkv -o out.mkv --upscale --model animejanaiV3-HD-L2.onnx

# 仅插帧 (2x 帧率)
vsr run -i in.mkv -o out.mkv --rife --rife-multi 2

# 超分 + 插帧
vsr run -i in.mkv -o out.mkv --upscale --rife --rife-multi 2

# 批量处理（递归）
vsr batch -i ./in_dir -o ./out_dir --recursive --upscale

# 交互向导
vsr
```

---

## CLI 参考

### 子命令

| 命令 | 用途 |
| --- | --- |
| `vsr run` | 处理单个文件 |
| `vsr batch` | 批量处理文件夹 |
| `vsr build-engines` | 预构建 TensorRT engine |
| `vsr doctor` | 检查运行时健康状态 |
| `vsr setup` | 搭建 Linux 运行时 |

### 处理参数

| 参数 | 说明 |
| --- | --- |
| `--upscale` | 启用 R-ESRGAN 超分 |
| `--model` | 超分模型文件名或路径（默认 `animejanaiV3-HD-L2.onnx`；相对路径基于 `models/RealESRGANv2`） |
| `--pre-resize-factor` | 超分前缩放百分比（如 `50`） |
| `--rife` | 启用 RIFE 插帧 |
| `--rife-multi` | 插帧倍率（如 `2`、`2/1`，默认 `2`） |
| `--rife-model` | RIFE 模型文件名或路径（默认 `rife_v4.10.onnx`） |
| `--final-resize-height` | 最终输出高度（像素） |
| `--encoder` | 编码预设：`nvenc`(默认) / `x265` / `x264` / `ffv1` |
| `--ffmpeg-args` | 自定义 ffmpeg 视频参数串，覆盖 `--encoder` |

### 运行时参数

| 参数 | 说明 |
| --- | --- |
| `--vspipe` | vspipe 可执行文件路径 |
| `--ffmpeg` | ffmpeg 可执行文件路径 |
| `--plugins-dir` | VapourSynth plugins 目录 |
| `--models-dir` | 模型目录（含 `RealESRGANv2/` `rife/`） |
| `--trtexec` | trtexec 可执行文件路径 |
| `--device-id` | GPU 设备号 |
| `--num-streams` | TensorRT 并行流数（默认 4） |
| `--num-threads` | VapourSynth 工作线程数（默认 0 = 按容器 cgroup 配额自动探测） |
| `--no-fp16` | 禁用 fp16 |
| `--nvenc-fix` | NVENC GPU 枚举修复库路径（LD_PRELOAD 注入 ffmpeg） |

运行 `vsr <子命令> -h` 查看完整选项。

<details>
<summary>模型参数规则</summary>

- `--model animejanaiV3-HD-L2.onnx` → `<models_dir>/RealESRGANv2/animejanaiV3-HD-L2.onnx`
- `--rife-model rife_v4.10.onnx` → `<models_dir>/rife/rife_v4.10.onnx`
- 也可传绝对路径，或相对 `models_dir` 的路径，例如 `--model RealESRGANv2/custom.onnx`
- 旧的 vs-mlrt enum 名仍兼容，例如 `--model animejanaiV3_HD_L2` / `--rife-model v4_10`
- 自定义 `.onnx` 若已是 fp16 模型，pipeline 会按模型 I/O 类型直接构建 TensorRT engine，不会再次调用 vsmlrt 的 fp16 转换

</details>

---

## 编码预设

| 名称 | ffmpeg 参数 |
| --- | --- |
| `nvenc` | `hevc_nvenc -preset p5 -profile:v main10 -split_encode_mode forced -b:v 50M` |
| `x265` | `libx265 -crf 16 -preset slow -x265-params "sao=0:bframes=8:psy-rd=1.5:psy-rdoq=2:aq-mode=3:ref=6"` |
| `x264` | `libx264 -crf 13 -preset slow` |
| `ffv1` | 无损 |

---

## 配置

### 优先级

```
CLI 参数 > 环境变量 > ~/.config/vsr/config.toml > 自动探测
```

环境变量：`VSR_VSPIPE` / `VSR_FFMPEG` / `VSR_PLUGINS` / `VSR_MODELS` / `VSR_TRTEXEC` / `VSR_PIPELINE` / `VSR_NUM_THREADS` / `VSR_NVENC_FIX`

<details>
<summary>setup.sh 环境变量一览</summary>

| 变量 | 作用 |
| --- | --- |
| `ASSUME_YES=1` | 跳过确认提示（非交互） |
| `CREATE_VENV=1` | 在 `<runtime>/venv` 新建 venv |
| `PY_BIN=/path/python` | 指定 Python 解释器 |
| `VSR_RUNTIME=/path` | 运行时目录 |
| `VSR_FFMPEG=/path/ffmpeg` | 显式指定 ffmpeg |
| `VSR_FFMPEG_STATIC_URL=https://...tar.xz` | PATH 无 ffmpeg 时下载的 static ffmpeg 地址 |
| `VSR_FFMPEG_INSTALL_DIR=/usr/local/bin` | 下载后安装 ffmpeg 的系统目录 |
| `SKIP_STATIC_FFMPEG=1` | 不下载 static build，只使用 PATH/apt ffmpeg |
| `VSMLRT_TAG=v15.x` | vs-mlrt release tag（`prerelease` 从最新预发布拉模型） |
| `VSMLRT_PY_REF=master` | 拉 `vsmlrt.py` 的 git ref |
| `FORCE_VSMLRT_PY=1` | 强制重新拉 `vsmlrt.py` |
| `MODEL_PACKS="^models\\. ^contrib-models\\."` | 要下载的 release asset 正则 |
| `SOURCE_PIP_PACKAGES="vapoursynth-lsmas"` | pip 源插件包 |
| `SKIP_AKARIN=1` | 跳过 akarin 安装 |
| `MLRT_TRT_PACKAGE="vapoursynth-mlrt-trt"` | TRT filter pip 包 |
| `MLRT_TRT_NO_DEPS=1` | 只装 filter wheel，不拉 TensorRT libs |
| `TENSORRT_HOME=/usr/src/tensorrt` | TensorRT SDK 根目录 |
| `SKIP_MODEL_EXTRACT=1` | 跳过模型下载/解压 |
| `GITHUB_TOKEN=...` | 提高 GitHub API 速率限制 |
| `SKIP_APT=1` | 关闭 apt 安装 |

</details>

---

## ffmpeg

`setup.sh` 先复用系统 PATH 里的 ffmpeg；若 PATH 没有，默认下载 [BtbN](https://github.com/BtbN/FFmpeg-Builds) 的 n8.1 静态构建并安装到 `/usr/local/bin/ffmpeg`。

> **选版本注意驱动兼容**：`master` 构建的 nvenc SDK 可能要求 API 13.1+ / 驱动 610+，AutoDL 驱动通常较老（如 580.x = API 13.0），选 **n8.1** 发布版而非 master。

设置 `VSR_FFMPEG` 后 `setup.sh` 使用该路径并写入 config。设置 `SKIP_STATIC_FFMPEG=1` 且 PATH 无 ffmpeg 时回退 apt 安装。

---

## 故障排查

<details>
<summary><code>libnvinfer_plugin.so.11</code> 找不到</summary>

检查 `LD_LIBRARY_PATH` 和 pip/conda 中的 TensorRT `.so` 目录。部分 conda 镜像把 `.so` 放在 base 环境的 `lib/python*/site-packages/tensorrt_libs`。

</details>

<details>
<summary>trtexec 版本不匹配</summary>

`trtexec` 需与 `core.trt` 使用同一 TensorRT engine ABI。如 `core.trt` 用 TRT 11.0 但 `/usr/src/tensorrt/bin/trtexec` 是 10.7，生成的 engine 无法反序列化。`vsr doctor` 会标记不匹配。用 `VSR_TRTEXEC` / `--trtexec` 指向正确版本。

</details>

<details>
<summary>容器内 <code>hevc_nvenc</code> 报 <code>unsupported device</code></summary>

NVIDIA 驱动 570+ 在多卡宿主机上的 bug：容器里的 NVENC 会拿到宿主机全部 GPU 列表，但只能访问分配给本容器的那块。修复用 [flexgrip/nvidia-gpu-enumeration](https://github.com/flexgrip/nvidia-gpu-enumeration) 的 `LD_PRELOAD` 垫片：

```bash
# 编译垫片
gcc -shared -fPIC -O2 -o /opt/libnvenc_fix.so nvenc_fix.c -ldl

# 指给 vsr
vsr run -i in.mkv -o out.mkv --upscale --nvenc-fix /opt/libnvenc_fix.so
# 或: export VSR_NVENC_FIX=/opt/libnvenc_fix.so
```

vsr 仅把该库注入 ffmpeg 的 `LD_PRELOAD`。若环境未开放 NVENC（`$NVIDIA_DRIVER_CAPABILITIES` 不含 `video`/`all`），垫片无效，改用 `--encoder x265`。

</details>

<details>
<summary><code>Invalid TensorFormat fp16:chw</code>（TRT 11 构建失败）</summary>

`vsmlrt.py` 太旧，对 TRT 11 仍发 TRT-10 专用参数。修复：

```bash
# 手动更新
curl -fL https://raw.githubusercontent.com/AmusementClub/vs-mlrt/master/scripts/vsmlrt.py \
  -o /root/autodl-tmp/vsr-runtime/vs-plugins/vsmlrt.py

# 或: setup.sh 自动修复
FORCE_VSMLRT_PY=1 bash setup.sh
```

</details>

<details>
<summary>错误版本构建的 engine</summary>

删除旧缓存后重建：

```bash
rm /root/autodl-tmp/vsr-runtime/vs-plugins/models/RealESRGANv2/*.engine*
vsr build-engines -i sample.mkv --upscale --model animejanaiV3-HD-L2.onnx
```

</details>

<details>
<summary>RIFE 新模型不在 latest release</summary>

较新的 RIFE 模型（如 v4.22+）可能只在 Pre-release 中：

```bash
VSMLRT_TAG=prerelease FORCE_RELEASE_EXTRACT=1 bash setup.sh
```

</details>

---

## 备注

- 首次为每个 `分辨率 × 模型` 组合构建 TRT engine 需数分钟，`build-engines` 可提前完成
- 源插件优先 `vapoursynth-lsmas`，需要时回退 `ffms2`
- `tensorrt-cu13-libs` 体积接近数 GB；根分区小时建议把 venv 建在持久盘
- 与 **VideoJaNai** 的关系：vsr-cli 走 1.x 路线（VapourSynth + vs-mlrt + `vspipe | ffmpeg` 管道）。VideoJaNai 2.0 已重构为 GPU 驻留的 native engine，不再有 Python 后端

> 更多踩坑细节、边界情况和调试经验见 [NOTES.md](NOTES.md)。

---

## License

[GPL-3.0](LICENSE)
