# vsr-cli

视频**超分辨率 (Real-ESRGAN)** 与**帧插值 (RIFE)** 命令行工具。

基于 [vs-mlrt](https://github.com/AmusementClub/vs-mlrt) 在 TensorRT 上推理，VapourSynth (`vspipe`) 通过管道直接喂给 ffmpeg 编码，零中间文件落盘。灵感来自 [VideoJaNai](https://github.com/the-database/VideoJaNai) 1.x。

## 特性

- **三种处理模式**（`--upscale` / `--rife` 自由组合）：仅超分、仅插帧、超分 + 插帧
- **交互式 TUI 向导**（无参数运行 `vsr`）或 CLI 批处理
- 音轨 / 字幕 / 附件从原片复制，仅视频重编码
- TensorRT engine 自动构建并缓存，支持 `vsr build-engines` 预热
- 多种编码预设：nvenc / x265 / x264 / ffv1

## 架构

```
vsr (CLI / TUI)
  └─ 定位运行时 (config.toml / env / 自动探测)
       └─ vspipe -c y4m --arg ... pipeline.vpy -  │  ffmpeg -i pipe: -i 原片 ...
            (VapourSynth: RealESRGAN@TRT → RIFE@TRT)   (仅重编码视频, 复制音轨/字幕)
```

---

## 快速开始

```bash
# 仅超分 (2x)
vsr run -i in.mkv -o out.mkv --upscale --model animejanaiV3-HD-L2.onnx --encoder nvenc

# 仅插帧 (2x 帧率)
vsr run -i in.mkv -o out.mkv --rife --rife-multi 2 --rife-model rife_v4.10.onnx

# 超分 + 插帧
vsr run -i in.mkv -o out.mkv --upscale --model animejanaiV3-HD-L2.onnx --rife --rife-multi 2

# 批量处理（递归）
vsr batch -i ./in_dir -o ./out_dir --recursive --upscale --model animejanaiV3-HD-L2.onnx

# 交互向导
vsr
```

---

## 安装（AutoDL / Ubuntu 22.04+ GPU 容器）

> VideoJaNai 的 `backend` 是 Windows 便携包，无法直接在 Linux 运行。本项目通过 PyPI wheel 安装 VapourSynth / L-SMASH / vs-mlrt TensorRT filter。

### 前置条件

- Python 3.12+ 环境（venv 或 conda）
- NVIDIA GPU + 驱动
- `curl` 可用

### 步骤

```bash
# 1. 准备 Python 环境（二选一）
python3.12 -m venv ~/vsr-venv && source ~/vsr-venv/bin/activate     # venv
# conda create -n vsr python=3.12 -y && conda activate vsr          # 或 conda

# 2. 一键安装运行时
cd vsr-cli
bash setup.sh            # 默认装到 /root/autodl-tmp/vsr-runtime
                         # 启动后输入 y 确认目标环境

# 3. 自检 + 预热 engine
vsr doctor
vsr build-engines -i sample.mkv --upscale --model animejanaiV3-HD-L2.onnx --rife --rife-multi 2
```

> `setup.sh` 启动时打印目标 Python/环境并要求输入 `y` 确认；仅当是裸的系统 Python（既非 venv 也非 conda）时额外警告。

### setup.sh 执行流程

1. 打印目标环境并确认
2. `pip install vapoursynth onnx numpy onnxconverter-common`
3. 安装源插件 wheel（默认 `vapoursynth-lsmas`）和 `vapoursynth-mlrt-trt`
4. `vapoursynth config` + `register-install` 配置插件自动加载
5. 验证 `core.lsmas`/`core.ffms2`，失败时才尝试 VSRepo fallback
6. 部署 `vsmlrt.py`，下载模型包（RealESRGAN / RIFE）
7. `pip install -e` 安装 `vsr` 本体，验证运行时文件并写 `~/.config/vsr/config.toml`

运行时目录（插件 + 模型）可 `tar` 打包复用；Python 依赖在你自己的 venv 中。

---

## CLI 参数

### 处理参数

| 参数 | 说明 |
| --- | --- |
| `--upscale` | 启用 R-ESRGAN 超分 |
| `--model` | 超分模型文件名或路径（默认 `animejanaiV3-HD-L2.onnx`；相对路径基于 `models/RealESRGANv2`） |
| `--pre-resize-factor` | 超分前缩放百分比（如 `50`） |
| `--rife` | 启用 RIFE 插帧 |
| `--rife-multi` | 插帧倍率（如 `2`、`2/1`，默认 `2`） |
| `--rife-model` | RIFE 模型文件名或路径（默认 `rife_v4.10.onnx`；相对路径基于 `models/rife`，也会查 `models/rife_v2`） |
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
| `--pipeline-vpy` | pipeline.vpy 路径 |
| `--device-id` | GPU 设备号 |
| `--num-streams` | TensorRT 并行流数 |
| `--no-fp16` | 禁用 fp16 |

运行 `vsr <子命令> -h` 查看完整选项。

模型参数规则：

- `--model animejanaiV3-HD-L2.onnx` 会查 `<models_dir>/RealESRGANv2/animejanaiV3-HD-L2.onnx`
- `--rife-model rife_v4.10.onnx` 会查 `<models_dir>/rife/rife_v4.10.onnx`
- 也可传绝对路径，或相对 `models_dir` 的路径，例如 `--model RealESRGANv2/custom.onnx`
- 旧的 vs-mlrt enum 名仍兼容，例如 `--model animejanaiV3_HD_L2` / `--rife-model v4_10`
- 自定义 `.onnx` 若已是 fp16 模型，pipeline 会按模型 I/O 类型直接构建 TensorRT engine，不会再次调用 vsmlrt 的 fp16 转换

### 子命令

| 命令 | 用途 |
| --- | --- |
| `vsr run` | 处理单个文件 |
| `vsr batch` | 批量处理文件夹 |
| `vsr build-engines` | 预构建 TensorRT engine |
| `vsr doctor` | 检查运行时健康状态 |
| `vsr setup` | 搭建 Linux 运行时 |

---

## 编码预设

| 名称 | ffmpeg 参数 |
| --- | --- |
| `nvenc` | `hevc_nvenc -preset p7 -profile:v main10 -b:v 50M` |
| `x265` | `libx265 -crf 16 -preset slow -x265-params "sao=0:bframes=8:psy-rd=1.5:psy-rdoq=2:aq-mode=3:ref=6"` |
| `x264` | `libx264 -crf 13 -preset slow` |
| `ffv1` | 无损 |

---

## 配置

### 运行时定位优先级

```
CLI 参数 > 环境变量 > ~/.config/vsr/config.toml > 自动探测
```

环境变量：`VSR_VSPIPE` / `VSR_FFMPEG` / `VSR_PLUGINS` / `VSR_MODELS` / `VSR_TRTEXEC` / `VSR_PIPELINE` / `VSR_NVENC_FIX`

### setup.sh 环境变量

| 变量 | 作用 |
| --- | --- |
| `ASSUME_YES=1` | 跳过确认提示（非交互） |
| `CREATE_VENV=1` | 在 `<runtime>/venv` 新建 venv |
| `PY_BIN=/path/python` | 指定 Python 解释器 |
| `VSR_RUNTIME=/path` | 运行时目录 |
| `VSR_FFMPEG=/path/ffmpeg` | 显式指定 ffmpeg，优先级最高 |
| `VSR_FFMPEG_STATIC_URL=https://...tar.xz` | PATH 无 ffmpeg 时下载的 static ffmpeg 地址 |
| `VSR_FFMPEG_INSTALL_DIR=/usr/local/bin` | 下载后安装 ffmpeg 的系统目录 |
| `SKIP_STATIC_FFMPEG=1` | 不下载 static build，只使用 PATH/apt ffmpeg |
| `VSMLRT_TAG=v15.x` | vs-mlrt release tag（`prerelease` 从最新预发布拉模型） |
| `VSMLRT_PY_REF=master` | 拉 `vsmlrt.py` 的 git ref |
| `VSMLRT_PY_URL=https://...` | 覆盖 `vsmlrt.py` 下载地址 |
| `FORCE_VSMLRT_PY=1` | 强制重新拉 `vsmlrt.py` |
| `MODEL_PACKS="^models\\. ^contrib-models\\."` | 要下载的 release asset 正则 |
| `SOURCE_PLUGIN="lsmas ffms2"` | VSRepo fallback 源插件 |
| `SOURCE_PIP_PACKAGES="vapoursynth-lsmas"` | pip 源插件包 |
| `SKIP_VSREPO_FALLBACK=1` | 禁止 VSRepo fallback |
| `MLRT_TRT_PACKAGE="vapoursynth-mlrt-trt"` | TRT filter pip 包 |
| `MLRT_TRT_NO_DEPS=1` | 只装 filter wheel，不拉 TensorRT libs |
| `VSR_TRTEXEC=/path/to/trtexec` | 指定 trtexec 路径 |
| `TENSORRT_HOME=/usr/src/tensorrt` | TensorRT SDK 根目录 |
| `SKIP_VSMLRT_PY=1` | 跳过部署 `vsmlrt.py`（要求已有文件） |
| `SKIP_MODEL_EXTRACT=1` | 跳过模型下载/解压（要求已有模型目录） |
| `SKIP_RELEASE_EXTRACT=1` | 旧兼容变量：同时跳过 `vsmlrt.py` 和模型 |
| `FORCE_RELEASE_EXTRACT=1` | 强制重新解压 |
| `SEVENZ_THREADS=1` | 限制 7z 解压线程数 |
| `PIP_CACHE_DIR` / `TMPDIR` | 大型 wheel 缓存目录 |
| `GITHUB_TOKEN=...` | 提高 GitHub API 速率限制 |
| `SKIP_PYTHON_INSTALL=1` | `CREATE_VENV=1` 时禁止自动装 Python |
| `SKIP_APT=1` | 关闭 apt 安装 |

---

## TensorRT 版本对齐

`vapoursynth-mlrt-trt` pip wheel（含 15.16.1）的 `core.trt` 按 **TensorRT 11.0.0 + CUDA 13** 编译（加载时找 `libnvinfer.so.11`）。系统侧需要 **TensorRT 11.0.0**，否则报 `libnvinfer.so.11: cannot open shared object file`。

### 用 apt 安装（推荐）

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

装好后跑 `setup.sh`——会自动检测系统 TensorRT（`ldconfig` / `TENSORRT_HOME` / `trtexec`），检测到则对 mlrt wheel 自动加 `--no-deps`，不从 pip 拉第二份 TensorRT。

---

## ffmpeg 建议

`setup.sh` 会先复用系统 PATH 里的 ffmpeg；如果 PATH 没有 ffmpeg，默认下载 BtbN 的 n8.1 静态构建并安装到 `/usr/local/bin/ffmpeg`，不再把 ffmpeg 放进 `vsr-runtime`：

```bash
bash setup.sh
/usr/local/bin/ffmpeg -hide_banner -encoders | grep nvenc
```

> **选版本注意驱动兼容**：`master` 构建的 nvenc SDK 可能要求 API 13.1+ / 驱动 610+，AutoDL 驱动通常较老（如 580.x = API 13.0），选 **n8.1** 发布版而非 master。

默认下载地址是 `https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz`，可用 `VSR_FFMPEG_STATIC_URL` 覆盖。设置 `VSR_FFMPEG` 后 `setup.sh` 会使用该路径并写入 config。只有设置 `SKIP_STATIC_FFMPEG=1` 且 PATH 无 ffmpeg 时，脚本才会回退 apt 安装（加 `--no-install-recommends` 裁剪 GUI 依赖）。

---

## 故障排查

### `libnvinfer_plugin.so.11` 找不到

检查 `LD_LIBRARY_PATH` 和 pip/conda 中的 TensorRT `.so` 目录。部分 conda 镜像把 `.so` 放在 base 环境的 `lib/python*/site-packages/tensorrt_libs`。

### trtexec 版本不匹配

`trtexec` 需与 `core.trt` 使用同一 TensorRT engine ABI。如 `core.trt` 用 TRT 11.0 但 `/usr/src/tensorrt/bin/trtexec` 是 10.7，生成的 engine 无法反序列化。`vsr doctor` 会标记不匹配。用 `VSR_TRTEXEC` / `--trtexec` 指向正确版本。

### 容器内 `hevc_nvenc` 报 `OpenEncodeSessionEx failed: unsupported device` / `No capable devices found`

NVIDIA 驱动 **570+ 在多卡宿主机上的 bug**：容器里的 NVENC 会拿到宿主机**全部** GPU 的列表，但只能访问分配给本容器的那块，初始化即崩——即使 CUDA / TensorRT 推理一切正常（它们走的是另一条枚举路径）。常见于 AutoDL 等共享 GPU 容器。

修复用 [flexgrip/nvidia-gpu-enumeration](https://github.com/flexgrip/nvidia-gpu-enumeration) 的 `LD_PRELOAD` 垫片（拦截 `ioctl`，把 GPU 列表过滤成容器实际可见的设备）：

```bash
# 1. 在容器内编译垫片
gcc -shared -fPIC -O2 -o /opt/libnvenc_fix.so nvenc_fix.c -ldl

# 2. 指给 vsr —— 只注入到 ffmpeg 子进程，不污染全局环境
vsr run -i in.mkv -o out.mkv --upscale --model X.onnx \
  --encoder nvenc --nvenc-fix /opt/libnvenc_fix.so
# 或: export VSR_NVENC_FIX=/opt/libnvenc_fix.so
```

vsr 仅把该库注入 ffmpeg 的 `LD_PRELOAD`（不会加到 vspipe / python 上）。`vsr doctor` 在 `nvenc_fix` 配置后会显示其状态。先用 `ffmpeg -f lavfi -i testsrc2=s=1280x720 -t1 -c:v hevc_nvenc -f null -` 验证垫片是否生效；若环境根本未开放 NVENC（`echo $NVIDIA_DRIVER_CAPABILITIES` 不含 `video`/`all`），垫片无效，只能改用软件编码 `--encoder x265`。

### `Invalid TensorFormat fp16:chw`（TRT 11 构建失败）

`vsmlrt.py` 太旧，对 TRT 11 仍发 TRT-10 专用参数。修复：

```bash
# 方法 1: 手动更新
curl -fL https://raw.githubusercontent.com/AmusementClub/vs-mlrt/master/scripts/vsmlrt.py \
  -o /root/autodl-tmp/vsr-runtime/vs-plugins/vsmlrt.py

# 方法 2: setup.sh 自动修复
FORCE_VSMLRT_PY=1 bash setup.sh
```

`setup.sh` 默认从 git master 拉脚本（≥3.23.1 已有 TRT-11 判断），release 自带的脚本可能滞后。

### 错误版本构建的 engine

删除旧缓存后重建：

```bash
rm /root/autodl-tmp/vsr-runtime/vs-plugins/models/RealESRGANv2/*.engine*
vsr build-engines -i sample.mkv --upscale --model animejanaiV3-HD-L2.onnx
```

### RIFE 新模型不在 latest release

较新的 RIFE 模型（如 v4.22）可能只在 Pre-release 中。获取方式：

```bash
VSMLRT_TAG=prerelease FORCE_RELEASE_EXTRACT=1 bash setup.sh
```

---

## 备注

- 首次为每个 `分辨率 × 模型` 组合构建 TRT engine 需数分钟，`build-engines` 用于提前完成
- 源插件优先 `vapoursynth-lsmas`，需要时回退 `ffms2`
- `tensorrt-cu13-libs` 体积接近数 GB；根分区小时建议把 conda env 建在持久盘
- 与 **VideoJaNai** 的关系：vsr-cli 走 1.x 路线（VapourSynth + vs-mlrt + `vspipe | ffmpeg` 管道），参考的颜色处理、RIFE padding 等逻辑来自 1.x 本地快照。VideoJaNai 2.0 已重构为 GPU 驻留的 native engine，不再有 Python 后端

> 更多踩坑细节、边界情况和调试经验见 [NOTES.md](NOTES.md)。
