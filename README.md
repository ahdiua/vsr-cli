# vsr-cli

视频 **超分(R-ESRGAN)** 与 **插帧(RIFE)** 命令行工具。基于 [vs-mlrt](https://github.com/AmusementClub/vs-mlrt) 在 **TensorRT** 上推理，VapourSynth (`vspipe`) 通过管道直接喂给 **ffmpeg** 编码，避免落盘中间文件。灵感来自 [VideoJaNai](https://github.com/the-database/VideoJaNai)。

支持三种处理模式（由 `--upscale` / `--rife` 自由组合）：

| 模式 | 开关 |
| --- | --- |
| 仅超分 | `--upscale` |
| 仅插帧 | `--rife` |
| 超分 + 插帧 | `--upscale --rife` |

既能 **交互式 TUI 向导**（无参数运行），也能 **直接用 CLI 参数** 批处理。

---

## 架构

```
vsr (CLI / TUI)
  └─ 定位运行时(config.toml / env / 自动探测)
       └─ vspipe -c y4m --arg ... pipeline.vpy -  │  ffmpeg -i pipe: -i 原片 ...
            (VapourSynth: RealESRGAN@TRT → RIFE@TRT)   (仅重编码视频, 复制音轨/字幕)
```

TensorRT engine 由 `vsmlrt.Backend.TRT` 自动构建并缓存（首次运行较慢，可用 `vsr build-engines` 预热）。

---

## 在 AutoDL / Ubuntu 22.04 GPU 容器上安装

> VideoJaNai 的 `backend` 是 **Windows** 便携包，**无法**直接在 Linux 运行。Linux 运行时使用 PyPI wheel 安装 VapourSynth / L-SMASH / vs-mlrt TensorRT filter，vs-mlrt release 只用于下载 `vsmlrt.py` 和模型包。

用 VapourSynth 官方推荐的方式：Python 3.12+ 环境 + `pip install vapoursynth`；源插件优先 `pip install vapoursynth-lsmas`，TensorRT filter 用 `pip install vapoursynth-mlrt-trt`。

> **默认安装进“当前已激活的环境”**——可以是 **venv 或 conda 环境**。`setup.sh` 启动时打印目标 Python/环境并**要求输入 `y` 确认**才继续；仅当是裸的系统 Python（既非 venv 也非 conda）时才额外警告会污染系统。

前置：自己先建好并激活一个 Python 3.12+ 环境（venv 或 conda 均可），容器内有 NVIDIA GPU、`curl`。

```bash
# 1. 准备并激活 Python 3.12+ 环境（二选一）
python3.12 -m venv ~/vsr-venv && source ~/vsr-venv/bin/activate     # venv
# conda create -n vsr python=3.12 -y && conda activate vsr          # 或 conda

# 2. 一键搭建运行时（装进当前环境：VapourSynth/source plugin/TRT filter
#    + 下载 vsmlrt.py + 模型 + 安装 vsr 本体）
cd vsr-cli
bash setup.sh            # 运行时默认装到 /root/autodl-tmp/vsr-runtime
#   启动后输入 y 确认目标环境

# 3. 自检 + 预热 engine
vsr doctor
vsr build-engines -i sample.mkv --upscale --model animejanaiV3_HD_L2 --rife --rife-multi 2
```

可选环境变量：

| 变量 | 作用 |
| --- | --- |
| `ASSUME_YES=1` | 跳过确认提示（非交互） |
| `CREATE_VENV=1` | 不用当前环境，改为在 `<runtime>/venv` 新建并使用一个 venv（缺 Python 3.12 时经 deadsnakes 自动装） |
| `PY_BIN=/path/python` | 指定解释器（默认用当前 `python`） |
| `VSR_RUNTIME=/path` | 运行时目录 |
| `VSMLRT_TAG=v15.x` | 指定 vs-mlrt release tag（默认 latest，仅用于模型包）。设 `prerelease` 则从最新预发布拉模型（`latest` 端点会跳过预发布，新模型如 RIFE v4.22 常在预发布里） |
| `VSMLRT_PY_REF=master` | 从哪个 git ref 拉 `vsmlrt.py`（默认 master）。release 自带的脚本滞后于插件，故脚本走 git；可设 tag/commit 固定 |
| `FORCE_VSMLRT_PY=1` | 即使已存在也重新从 git 拉 `vsmlrt.py` |
| `MODEL_PACKS="RealESRGANv2 rife"` | 要下载的模型包 |
| `SOURCE_PIP_PACKAGES="vapoursynth-lsmas"` | pip 源插件包 |
| `MLRT_TRT_PACKAGE="vapoursynth-mlrt-trt==15.16.1"` | TensorRT VapourSynth filter pip 包 |
| `MLRT_TRT_NO_DEPS=1` | 只安装 filter wheel，不让 pip 拉 TensorRT libs；适合已用 apt/tar 安装 TensorRT。**留空则自动检测**：发现系统 TensorRT 就自动启用 `--no-deps`，否则让 pip 解析依赖（`=0` 可强制走 pip） |
| `SOURCE_PLUGIN="lsmas ffms2"` | VSRepo fallback 源插件（仅 pip/autoload 失败时尝试） |
| `SKIP_VSREPO_FALLBACK=1` | 禁止使用 VSRepo fallback |
| `VSR_TRTEXEC=/path/to/trtexec` | 指定 Linux `trtexec` 路径；版本必须匹配 `core.trt` |
| `TENSORRT_HOME=/usr/src/tensorrt` | 指定 TensorRT SDK 根目录；AutoDL 常见路径只有版本匹配时才会自动采用 |
| `SKIP_RELEASE_EXTRACT=1` | 调试时跳过 `vsmlrt.py` 和模型包下载/解压 |
| `FORCE_RELEASE_EXTRACT=1` | 即使已存在 `vsmlrt.py`/模型目录也重新解压 |
| `PIP_CACHE_DIR=/root/autodl-tmp/pip-cache` / `TMPDIR=/root/autodl-tmp/tmp` | 大型 TensorRT wheel 构建/缓存目录；setup 默认使用 runtime 下的目录 |
| `GITHUB_TOKEN=...` | 提高 GitHub API 速率限制 |
| `SKIP_APT=1` / `SKIP_PYTHON_INSTALL=1` | 关闭 apt / 关闭自动装 Python（仅 `CREATE_VENV`） |

`setup.sh` 步骤：
0. 打印目标环境并**确认 `y`**
1. `pip install vapoursynth onnx numpy onnxconverter-common`（装进当前/指定环境）
2. `vapoursynth config`（+ `register-install`）配置插件自动加载
3. `pip install vapoursynth-lsmas`；只有未检测到 `core.lsmas` / `core.ffms2` 时才尝试 **VSRepo fallback**
4. `pip install vapoursynth-mlrt-trt`；自动检测系统 TensorRT（或显式 `MLRT_TRT_NO_DEPS=1`）时用 `--no-deps` 安装，只使用 apt/tar 提供的 TensorRT 库；不会下载/解压 `vsmlrt-cuda.v*.7z.*`（该包当前是 Windows `.dll/.exe` 依赖包）
5. 下载 **模型包**（RealESRGAN / RIFE）
6. `pip install -e` 安装 `vsr` 本体，写 `~/.config/vsr/config.toml`

运行时目录（插件 + 模型）可 `tar` 打包复用；Python 依赖在你自己的 venv 中。

### 用 apt 安装匹配版本的 TensorRT（推荐）

**版本必须和 pip 的 `vapoursynth-mlrt-trt` wheel 对齐。** 该 wheel（含最新 `15.16.1`）的 `core.trt` 是按 **TensorRT 11.0.0 + CUDA 13** 编译的（依赖 `tensorrt-cu13-libs==11.0.0.114`，加载时找 `libnvinfer.so.11`）——PyPI 上**没有** TRT-10 的 wheel。所以系统侧要装 **TensorRT 11.0.0**，否则 `core.trt` 会报 `libnvinfer.so.11: cannot open shared object file`。

> 注：vs-mlrt **GitHub release** 的 `core.trt`（v15.16 那套）才是 TRT-10 编译的；但本项目走 pip wheel，因此对齐到 TRT 11。

比起让 pip 拉一份几 GB 的 TensorRT，更稳的做法是用 NVIDIA 的 CUDA apt 源装一个**精确匹配 11.0.0** 的版本，再让 `setup.sh` 自动识别（见下）跳过 pip 的 TensorRT。

```bash
# 1. 注册 NVIDIA CUDA apt 源（Ubuntu 24.04 为例；22.04 把 ubuntu2404 换成 ubuntu2204）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update

# 2. 确认确切的 apt 版本串（cuda13 这边 11.0.0.114 通常是 +cuda13.2），再 pin
apt-cache madison libnvinfer11        # 形如 11.0.0.114-1+cuda13.2

# 3. 装运行 + build-engines 需要的包（全部 pin 同一版本，避免被 apt 拉到 11.1）
version="11.0.0.114-1+cuda13.2"
apt install -y \
    libnvinfer11=${version} \
    libnvinfer-bin=${version} \
    libnvinfer-plugin11=${version} \
    libnvonnxparsers11=${version} \
    libnvinfer-lean11=${version} \
    libnvinfer-vc-plugin11=${version} \
    libnvinfer-dispatch11=${version}

# 4. 验证（trtexec 应为 11.0.x，ldconfig 应有 libnvinfer.so.11 + libcudart.so.13）
trtexec --version
ldconfig -p | grep -E 'libnvinfer.so|libcudart.so.13|libcublas.so.13'
```

各包作用——前 4 个是真正用到的：

| 包 | 提供 | 必要原因 |
| --- | --- | --- |
| `libnvinfer11` | `libnvinfer.so.11` 核心运行库 | `core.trt` 直接链接，推理依赖 |
| `libnvinfer-bin` | `trtexec` | `vsr build-engines` 构建 engine |
| `libnvinfer-plugin11` | `libnvinfer_plugin.so.11` | trtexec 依赖（显式 pin 更稳） |
| `libnvonnxparsers11` | `libnvonnxparser.so.11` | trtexec 解析 ONNX 模型 |

后 3 个 `libnvinfer-lean11` / `libnvinfer-vc-plugin11` / `libnvinfer-dispatch11`：vs-mlrt 本身不用，但它们是 `libnvinfer-bin` 的**强制依赖**，且要求与之**完全同版本**——不一起 pin，apt 会给它们挑最新的 `11.1.0.106` 从而报 "held broken packages"。所以必须随 trtexec 一起按同版本装。

仍可省略：`tensorrt` / `tensorrt-dev` / `tensorrt-libs`（meta 包，体积膨胀根源）、所有 `*-dev` / `*-headers-*`（仅编译链接时用，wheel 是预编译的）、`libnvinfer-safe-*`（车规 safety）、`libnvinfer-win-builder-resource*`（Windows 资源，Linux 无意义）、`python3-libnvinfer*`（Python 绑定，`vsmlrt.py` 不 import）。若 apt 仍报某个 `=${version}` 依赖未满足，把那个包名加进上面列表、同版本号即可。

> **CUDA 13 运行库**：这是 `+cuda13.2` 的构建，依赖 CUDA 13 的 `cudart/cublas`，apt 会随 `libnvinfer11` 自动拉进来。驱动只需支持 CUDA 13.0（`nvidia-smi` 顶部那个），13.2 运行库靠同主版本 minor 向前兼容即可运行。仅当 `ldconfig` 没有 `libcudart.so.13`/`libcublas.so.13`（或 `vsr doctor` 报找不到）时才手动补：`apt install -y cuda-cudart-13-2 libcublas-13-2`。若不想动系统 CUDA，也可改为**不加 `--no-deps`**（设 `MLRT_TRT_NO_DEPS=0`），让 pip 把 TRT 11 + CUDA 13 全套 wheel 自包含装进 venv，此时 trtexec 仍需单独提供一个 TRT-11 的（apt `libnvinfer-bin` 或 tar，配合 `VSR_TRTEXEC`）。

装好后跑 `setup.sh`：它会**自动检测**系统 TensorRT（`ldconfig` / `TENSORRT_HOME` / 常见目录 / `trtexec`），检测到就对 mlrt wheel 自动加 `--no-deps`，不再从 pip 拉第二份 TensorRT。若之前用错误版本生成过 engine，记得先删旧缓存（见下方备注）再 `vsr build-engines`。注意别让旧的 `/usr/src/tensorrt/bin/trtexec` 等非 11.0 的 trtexec 抢占——必要时删掉或用 `VSR_TRTEXEC` 指向 11.0 那个。

---

## 用法

```bash
# 仅超分
vsr run -i in.mkv -o out.mkv --upscale --model animejanaiV3_HD_L2 --encoder nvenc

# 仅插帧（2x 帧率）
vsr run -i in.mkv -o out.mkv --rife --rife-multi 2 --rife-model v4_22

# 超分 + 插帧
vsr run -i in.mkv -o out.mkv --upscale --model animejanaiV3_HD_L2 --rife --rife-multi 2

# 批量文件夹（递归）
vsr batch -i ./in_dir -o ./out_dir --recursive --upscale --model animejanaiV3_HD_L2

# 交互向导
vsr
```

### 主要参数

| 参数 | 说明 |
| --- | --- |
| `--upscale` / `--model` | 启用超分 / 选择 R-ESRGAN 模型 |
| `--pre-resize-factor` | 超分前缩放百分比（如 `50`） |
| `--rife` / `--rife-multi` / `--rife-model` | 启用插帧 / 倍率（`2`、`2/1`）/ 模型 |
| `--final-resize-height` | 最终输出高度 |
| `--encoder` | `nvenc`(默认) / `x265` / `x264` / `ffv1` |
| `--ffmpeg-args` | 自定义 ffmpeg 视频参数串，覆盖 `--encoder` |
| `--device-id` / `--num-streams` / `--no-fp16` | TensorRT 选项 |
| `--vspipe` / `--ffmpeg` / `--plugins-dir` / `--models-dir` / `--trtexec` / `--pipeline-vpy` | 运行时路径覆盖 |

运行 `vsr <子命令> -h` 查看完整选项。

### 编码预设（沿用 VideoJaNai）

- `nvenc`: `hevc_nvenc -preset p7 -profile:v main10 -b:v 50M`
- `x265`: `libx265 -crf 16 -preset slow -x265-params "sao=0:bframes=8:psy-rd=1.5:psy-rdoq=2:aq-mode=3:ref=6"`
- `x264`: `libx264 -crf 13 -preset slow`
- `ffv1`: 无损

---

## 运行时定位优先级

`CLI 参数 > 环境变量(VSR_VSPIPE/VSR_FFMPEG/VSR_PLUGINS/VSR_MODELS/VSR_TRTEXEC/VSR_PIPELINE) > ~/.config/vsr/config.toml > 自动探测`

---

## 备注

- 首次为每个 `分辨率 × 模型` 组合构建 TRT engine 需数分钟，属正常；`build-engines` 用于提前完成。
- `vapoursynth-mlrt-trt` 的 TensorRT 依赖需要能被动态链接器找到；若 `core.trt` 报 `libnvinfer_plugin.so.11` 找不到，检查 `LD_LIBRARY_PATH` 和 pip/conda 中的 TensorRT `.so` 目录。有些 conda 镜像会把这些普通 `.so` 放在 base 环境的 `lib/python*/site-packages/tensorrt_libs` 下，路径里的 Python 版本不代表当前运行解释器版本。
- `trtexec` 需要和 `core.trt` 使用同一 TensorRT engine ABI。比如 `vapoursynth-mlrt-trt` 加载的是 TensorRT 11.0，但 `/usr/src/tensorrt/bin/trtexec` 是 10.7，就会生成无法反序列化的 engine；setup/doctor 会把这种情况标为不可用。可用 `VSR_TRTEXEC` / `--trtexec` 指向匹配版本。
- 已用 apt/tar 安装 TensorRT 时，推荐运行 `MLRT_TRT_NO_DEPS=1 MLRT_TRT_PACKAGE="vapoursynth-mlrt-trt==15.16.1" ASSUME_YES=1 SKIP_RELEASE_EXTRACT=1 bash setup.sh`，避免 pip 再下载/安装数 GB 的 TensorRT libs。
- 如果曾经用错误版本生成过 engine，删除对应模型目录下的 `*.engine` 和 `*.engine.cache` 后再重建，例如 `rm /root/autodl-tmp/vsr-runtime/vs-plugins/models/RealESRGANv2/*.engine*`。
- **`Invalid TensorFormat fp16:chw`（TRT 11 下 build-engines 失败）**：部署的 `vsmlrt.py` 太旧、对 TRT 11 仍发 `--inputIOFormats=fp16:chw`/`--useCudaGraph`/`--noTF32` 等 TRT-10 专用参数。根因是 vs-mlrt **release 自带的脚本滞后**（如 v15.16 release 是 3.22.38，无 TRT-11 判断），而 pip wheel 是 TRT 11。`setup.sh` 现在默认从 git（`VSMLRT_PY_REF`，默认 master，≥3.23.1 起按 `core.trt` 版本 gate 参数）拉脚本。手动修复：`curl -fL https://raw.githubusercontent.com/AmusementClub/vs-mlrt/master/scripts/vsmlrt.py -o /root/autodl-tmp/vsr-runtime/vs-plugins/vsmlrt.py`，或 `FORCE_VSMLRT_PY=1 bash setup.sh`。该文件在持久盘上，重置不丢。
- `tensorrt-cu13-libs` 体积接近数 GB；根分区较小时建议把 conda env 建在 `/root/autodl-tmp`，并保持 `PIP_CACHE_DIR`/`TMPDIR` 指向大盘。
- 源插件优先 `vapoursynth-lsmas`，需要时才回退 `ffms2`。
- 音轨/字幕/附件从原片复制，仅视频重编码。
- **ffmpeg 推荐用静态构建放持久盘，不走 apt**：设置 `VSR_FFMPEG` 指向可执行文件后，`setup.sh` 会跳过 apt 安装 ffmpeg、并把该路径写进 config。在 AutoDL 这种反复重置系统盘的环境，把自带 nvenc 的静态 ffmpeg 放到 `/root/autodl-tmp` 持久盘最省事：

  ```bash
  cd ~/autodl-tmp
  wget https://github.com/ahdiua/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-nonfree-8.1.tar.xz
  tar xf ffmpeg-n8.1-latest-linux64-nonfree-8.1.tar.xz
  export VSR_FFMPEG=~/autodl-tmp/ffmpeg-n8.1-latest-linux64-nonfree-8.1/bin/ffmpeg
  "$VSR_FFMPEG" -hide_banner -encoders | grep nvenc   # 确认有 hevc_nvenc
  # 之后 bash setup.sh 就不会 apt 装 ffmpeg
  ```

  > **选构建版本要匹配驱动的 nvenc API**：`master`（最新）构建捆绑的 nvenc SDK 可能要求过新的 API（如要求 NVENC API 13.1 / 驱动 610+），在老驱动上 `hevc_nvenc` 会报 `Driver does not support the required nvenc API version`。AutoDL 驱动通常较老（如 580.105.08 = API 13.0），所以选 **n8.1** 这类发布版构建（`ffmpeg-n8.1-...`）而非 master。

  仅在未设 `VSR_FFMPEG` 且 PATH 里也没有 ffmpeg 时，`setup.sh` 才退而用 apt 装，并加 `--no-install-recommends` 砍掉 GTK/mesa/`libllvm`/图标主题等推荐包（Ubuntu 的 `ffmpeg` 仍硬依赖 libsdl2→几个 X11 小库，无法避免）。误用 apt 装了 GUI 全家桶时，`sudo apt purge -y ffmpeg && sudo apt autoremove --purge -y` 可清掉变孤儿的 X11/GTK/mesa 等包。
- **与 VideoJaNai 的关系（参考冻结在 1.x）**：vsr-cli 走的是 **VideoJaNai 1.x 的路线**——VapourSynth + Python + vs-mlrt，`vspipe | ffmpeg` 管道。本仓库参考的颜色处理、RIFE padding、`_implementation` 等逻辑都来自本地 1.x 的 `backend/animejanai/core/*.py`，**请把这份 1.x 当作冻结的参考快照保留，不要升级到 2.0**。**VideoJaNai 2.0 已彻底重构**：去掉 VapourSynth/Python/vs-mlrt，改用全程 GPU 驻留的自研 native engine（decode→infer→encode 不下显存），不再有 Python 后端可供借鉴。2.0 仍用 **TensorRT 11.0.0**，印证了本项目对齐 TRT 11 的选型正确。性能上 2.0 强调"消除 `vspipe | ffmpeg` 的 CPU 往返"——这正是 vsr-cli 当前架构的瓶颈所在，若日后要提速可朝 GPU 驻留方向考虑（短期不动）。
- **RIFE 模型 / 模型包来自 vs-mlrt release，新模型在 Pre-release**：`/releases/latest` 会跳过预发布，而较新的 RIFE（如 v4.22）和更新的 AnimeJaNai 模型可能只在预发布里。最新预发布 `v16.test1` 含 `models.v16.test1.7z`。用 `VSMLRT_TAG=prerelease FORCE_RELEASE_EXTRACT=1 bash setup.sh` 从最新预发布重下模型包（`setup.sh` 会自动解析并打印 `resolved prerelease -> tag ...`）。重下前可先 `cp -r .../models/rife .../models/rife.bak` 备份，避免新包反而缺旧模型。
