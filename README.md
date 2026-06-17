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

> VideoJaNai 的 `backend` 是 **Windows** 便携包，**无法**直接在 Linux 运行。Linux 需用 vs-mlrt 的 Linux x64 release 自建运行时——`setup.sh` 已自动化这一过程。

**不使用 conda**：按 VapourSynth 官方推荐，用系统 Python 3.12+ 的 venv + `pip install vapoursynth`，源插件用 VSRepo 安装。

前置：容器内有 NVIDIA GPU、`curl`。Python 3.12+ 若缺失，`setup.sh` 会自动经 deadsnakes PPA 安装（可用 `SKIP_PYTHON_INSTALL=1` 关闭）。

```bash
# 一键搭建运行时（venv + pip 装 VapourSynth/VSRepo/源插件
#   + 下载 vs-mlrt Linux 插件/trtexec + 模型 + 安装 vsr 本体）
cd vsr-cli
bash setup.sh            # 默认装到 /root/autodl-tmp/vsr-runtime
#   可选环境变量:
#   VSR_RUNTIME=/path     运行时目录
#   PY_BIN=/usr/bin/python3.12   指定 Python 解释器
#   VSMLRT_TAG=v15.x      指定 vs-mlrt release tag（默认 latest）
#   MODEL_PACKS="RealESRGANv2 rife"   要下载的模型包
#   SOURCE_PLUGIN="lsmas ffms2"       VSRepo 源插件（按序尝试）
#   GITHUB_TOKEN=...      提高 GitHub API 速率限制
#   SKIP_PYTHON_INSTALL=1 / SKIP_APT=1

# 激活 venv 后自检 + 预热 engine
source /root/autodl-tmp/vsr-runtime/venv/bin/activate
vsr doctor
vsr build-engines -i sample.mkv --upscale --model animejanaiV3_HD_L2 --rife --rife-multi 2
```

`setup.sh` 步骤：
1. 选/装 **Python 3.12+** → 建 venv → `pip install vapoursynth vsrepo onnx numpy onnxconverter-common`
2. `vapoursynth config`（+ `register-install`）配置插件自动加载
3. **VSRepo** 安装源插件（`lsmas` 优先，回退 `ffms2`）
4. 下载 **vs-mlrt** Linux 插件（vsort/vstrt + vsmlrt-cuda/trtexec）到 `plugins/`
5. 下载 **模型包**（RealESRGAN / RIFE）
6. `pip install -e` 安装 `vsr` 本体，写 `~/.config/vsr/config.toml`

整个 `运行时目录`（含 venv + 插件 + 模型）可 `tar` 打包，在其它容器解压复用（venv 路径变了需重建或改用系统 Python）。

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
| `--vspipe` / `--ffmpeg` / `--plugins-dir` / `--models-dir` / `--pipeline-vpy` | 运行时路径覆盖 |

运行 `vsr <子命令> -h` 查看完整选项。

### 编码预设（沿用 VideoJaNai）

- `nvenc`: `hevc_nvenc -preset p7 -profile:v main10 -b:v 50M`
- `x265`: `libx265 -crf 16 -preset slow -x265-params "sao=0:bframes=8:psy-rd=1.5:psy-rdoq=2:aq-mode=3:ref=6"`
- `x264`: `libx264 -crf 13 -preset slow`
- `ffv1`: 无损

---

## 运行时定位优先级

`CLI 参数 > 环境变量(VSR_VSPIPE/VSR_FFMPEG/VSR_PLUGINS/VSR_MODELS/VSR_PIPELINE) > ~/.config/vsr/config.toml > 自动探测`

---

## 备注

- 首次为每个 `分辨率 × 模型` 组合构建 TRT engine 需数分钟，属正常；`build-engines` 用于提前完成。
- AutoDL 容器的 CUDA/TensorRT 版本需与下载的 vs-mlrt release 匹配；不匹配时用 `VSMLRT_TAG` 指定合适版本。
- 源插件优先 `lsmas`(L-SMASH)，回退 `ffms2`。
- 音轨/字幕/附件从原片复制，仅视频重编码。
