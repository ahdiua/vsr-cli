# vsr-cli 踩坑笔记

搭建和调试过程中遇到的坑，留作参考。README 只保留操作步骤，详细原因和边界情况记在这里。

---

## TensorRT 版本陷阱

### pip wheel vs GitHub release 的 core.trt 编译版本不同

vs-mlrt **GitHub release** 附带的 `core.trt`（v15.16 那套）是按 **TRT-10** 编译的；但 **pip wheel** `vapoursynth-mlrt-trt` 是 TRT 11。本项目走 pip wheel，所以系统侧要对齐到 **TRT 11.0.0**。PyPI 上没有 TRT-10 的 wheel。

### apt pin 版本必须严格一致

`libnvinfer-lean11` / `libnvinfer-vc-plugin11` / `libnvinfer-dispatch11` 这三个包 vs-mlrt 本身不用，但它们是 `libnvinfer-bin`（提供 trtexec）的**强制依赖**，且要求**完全同版本**。如果不一起 pin，apt 会给它们挑最新的 `11.1.0.106` 然后报 "held broken packages"。必须随 trtexec 一起按同版本装。

若 apt 仍报某个 `=${version}` 依赖未满足，把那个包名加进安装列表、同版本号即可。

### 可以省略的包

`tensorrt` / `tensorrt-dev` / `tensorrt-libs`（meta 包，体积膨胀根源）、所有 `*-dev` / `*-headers-*`（仅编译链接用，wheel 是预编译的）、`libnvinfer-safe-*`（车规 safety）、`libnvinfer-win-builder-resource*`（Windows 资源，Linux 无意义）、`python3-libnvinfer*`（Python 绑定，`vsmlrt.py` 不 import）。

### 旧 trtexec 抢占

注意别让旧的 `/usr/src/tensorrt/bin/trtexec` 等非 11.0 的 trtexec 抢占 PATH——必要时删掉或用 `VSR_TRTEXEC` 指向 11.0 那个。`vsr doctor` 和 `setup.sh` 会检测版本不匹配并标记为不可用。

### 错误版本生成的 engine 必须删除

如果曾经用错误版本生成过 engine，必须先删除旧缓存再重建：

```bash
rm /root/autodl-tmp/vsr-runtime/vs-plugins/models/RealESRGANv2/*.engine*
```

---

## CUDA 13 运行库

TRT 11.0.0 `+cuda13.2` 构建依赖 CUDA 13 的 `cudart/cublas`，apt 会随 `libnvinfer11` 自动拉进来。驱动只需支持 CUDA 13.0（`nvidia-smi` 顶部那个），13.2 运行库靠同主版本 minor 向前兼容即可运行。

仅当 `ldconfig` 没有 `libcudart.so.13`/`libcublas.so.13`（或 `vsr doctor` 报找不到）时才手动补：

```bash
apt install -y cuda-cudart-13-2 libcublas-13-2
```

### 不想动系统 CUDA 的替代方案

设 `MLRT_TRT_NO_DEPS=0`，让 pip 把 TRT 11 + CUDA 13 全套 wheel 自包含装进 venv。此时 trtexec 仍需单独提供一个 TRT-11 的（apt `libnvinfer-bin` 或 tar，配合 `VSR_TRTEXEC`）。

---

## MLRT_TRT_NO_DEPS 行为

- **留空**（默认）：自动检测——发现系统 TensorRT 就自动启用 `--no-deps`，否则让 pip 解析依赖
- **`=1`**：强制只装 filter wheel，不让 pip 拉 TensorRT libs；适合已用 apt/tar 安装
- **`=0`**：强制走 pip，即使检测到系统 TensorRT 也让 pip 拉一份

已有系统 TensorRT 时推荐命令（避免 pip 再下几 GB）：

```bash
MLRT_TRT_NO_DEPS=1 MLRT_TRT_PACKAGE="vapoursynth-mlrt-trt==15.16.1" ASSUME_YES=1 SKIP_RELEASE_EXTRACT=1 bash setup.sh
```

setup.sh 安装 mlrt wheel 时不会下载/解压 `vsmlrt-cuda.v*.7z.*`——该包当前是 Windows `.dll/.exe` 依赖包，Linux 无用。

---

## vsmlrt.py 版本滞后

vs-mlrt **release 自带的脚本滞后于插件**。比如 v15.16 release 的 `vsmlrt.py` 是 3.22.38，没有 TRT-11 判断（对 TRT 11 仍发 `--inputIOFormats=fp16:chw`/`--useCudaGraph`/`--noTF32` 等 TRT-10 专用参数），而 pip wheel 编译的 `core.trt` 是 TRT 11——结果就是 `Invalid TensorFormat fp16:chw` 构建失败。

`setup.sh` 现在默认从 git（`VSMLRT_PY_REF`，默认 master，≥3.23.1 起按 `core.trt` 版本 gate 参数）拉脚本，不用 release 的。该文件在持久盘上，重置不丢。

---

## setup.sh 额外环境变量（README 未列全）

| 变量 | 作用 |
| --- | --- |
| `SOURCE_PLUGIN="lsmas ffms2"` | VSRepo fallback 源插件（仅 pip/autoload 失败时尝试） |
| `SKIP_VSREPO_FALLBACK=1` | 禁止使用 VSRepo fallback |
| `SKIP_PYTHON_INSTALL=1` | 关闭自动装 Python（仅 `CREATE_VENV` 模式） |

---

## conda 下 TensorRT .so 路径

有些 conda 镜像会把 TensorRT `.so` 放在 base 环境的 `lib/python*/site-packages/tensorrt_libs` 下，**路径里的 Python 版本不代表当前运行解释器版本**。如果 `core.trt` 报找不到 `.so`，检查 `LD_LIBRARY_PATH` 是否包含这些目录。

---

## ffmpeg 注意事项

### apt 安装的 GUI 依赖膨胀

Ubuntu 的 `ffmpeg` 包硬依赖 libsdl2 → 几个 X11 小库，`--no-install-recommends` 可以砍掉 GTK/mesa/`libllvm`/图标主题等推荐包，但 X11 小库无法避免。误用 apt 装了 GUI 全家桶时清理：

```bash
sudo apt purge -y ffmpeg && sudo apt autoremove --purge -y
```

### nvenc API 版本兼容

`master`（最新）构建捆绑的 nvenc SDK 可能要求过新的 API（如要求 NVENC API 13.1 / 驱动 610+），在老驱动上 `hevc_nvenc` 会报 `Driver does not support the required nvenc API version`。AutoDL 驱动通常较老（如 580.105.08 = API 13.0），所以选 **n8.1** 这类发布版构建（`ffmpeg-n8.1-...`）而非 master。

---

## RIFE 模型获取

`/releases/latest` 会跳过预发布，较新的 RIFE（如 v4.22）和更新的 AnimeJaNai 模型可能只在预发布里。最新预发布 `v16.test1` 含 `models.v16.test1.7z`。

```bash
VSMLRT_TAG=prerelease FORCE_RELEASE_EXTRACT=1 bash setup.sh
# setup.sh 会自动解析并打印 "resolved prerelease -> tag ..."
```

重下前建议备份旧模型（新包可能缺旧模型）：

```bash
cp -r .../models/rife .../models/rife.bak
```

---

## 与 VideoJaNai 的关系

vsr-cli 走的是 **VideoJaNai 1.x 路线**——VapourSynth + Python + vs-mlrt，`vspipe | ffmpeg` 管道。参考的颜色处理、RIFE padding、`_implementation` 等逻辑都来自本地 1.x 的 `backend/animejanai/core/*.py`，**请把这份 1.x 当作冻结的参考快照保留，不要升级到 2.0**。

**VideoJaNai 2.0** 已彻底重构：去掉 VapourSynth/Python/vs-mlrt，改用全程 GPU 驻留的自研 native engine（decode→infer→encode 不下显存），不再有 Python 后端可供借鉴。2.0 仍用 **TensorRT 11.0.0**，印证了本项目对齐 TRT 11 的选型正确。

性能上 2.0 强调"消除 `vspipe | ffmpeg` 的 CPU 往返"——这正是 vsr-cli 当前架构的瓶颈所在，若日后要提速可朝 GPU 驻留方向考虑（短期不动）。
