"""Parametrized VapourSynth pipeline for vsr-cli.

Run via vspipe, passing parameters with --arg key=value, e.g.:

    vspipe -c y4m \
        --arg video_path=/in.mkv \
        --arg upscale=1 --arg model=animejanaiV3_HD_L2 \
        --arg rife=1 --arg rife_multi=2 --arg rife_model=v4_10 \
        pipeline.vpy -

Builds the processing graph conditionally:
  * upscale=1  -> insert vsmlrt.RealESRGAN on the TensorRT backend
  * rife=1     -> insert vsmlrt.RIFE on the TensorRT backend
At least one of the two must be enabled.

vspipe injects every --arg as a module global (string). Helpers below parse them.
Color handling (matrix/range detection + RGB conversion) follows VideoJaNai's
animejanai_core.py to avoid color drift through the ML models.
"""

import fractions
import os
import shutil
import sys

import vapoursynth as vs
from vapoursynth import core

# --- argument parsing -------------------------------------------------------

_G = dict(globals())


def _arg(name, default=None):
    return _G.get(name, default)


def _arg_bool(name, default=False):
    val = _G.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


def _arg_int(name, default=None):
    val = _G.get(name)
    if val is None or str(val).strip() == "":
        return default
    return int(val)


def _arg_float(name, default=None):
    val = _G.get(name)
    if val is None or str(val).strip() == "":
        return default
    return float(val)


def _arg_fraction(name, default=2):
    """Parse '2', '2/1', '2.5' into a Fraction (RIFE multi)."""
    val = _G.get(name)
    if val is None or str(val).strip() == "":
        return fractions.Fraction(default)
    s = str(val).strip()
    if "/" in s:
        num, den = s.split("/", 1)
        return fractions.Fraction(int(num), int(den))
    return fractions.Fraction(s)


video_path = _arg("video_path")
if not video_path:
    raise ValueError("pipeline.vpy: 'video_path' arg is required")

do_upscale = _arg_bool("upscale")
do_rife = _arg_bool("rife")
if not (do_upscale or do_rife):
    raise ValueError(
        "pipeline.vpy: enable at least one of 'upscale' or 'rife' "
        "(both disabled produces no work)"
    )

# --- load optional plugins + import vsmlrt ----------------------------------
# The TensorRT filter should normally be autoloaded from the
# vapoursynth-mlrt-trt wheel. plugins_dir is still used for vsmlrt.py, models,
# and optional manually dropped Linux .so files.

plugins_dir = _arg("plugins_dir")
models_dir = _arg("models_dir")
trtexec_path = _arg("trtexec_path")

# vs-mlrt inference plugins + common source plugins (ffms2 / L-SMASH-Works).
# Source plugins are normally autoloaded (installed via pip/VSRepo); loading them
# again here is harmless (vs.Error is caught) and lets users instead just drop
# the .so into plugins_dir.
_PLUGIN_PREFIXES = ("libvstrt", "libvsort", "libvsov", "libvsncnn",
                    "vstrt", "vsort", "vsov", "vsncnn",
                    "libffms2", "ffms2",
                    "liblsmashsource", "libvslsmashsource", "lsmas",
                    "libbestsource", "bestsource", "libbs")
_PLUGIN_SUFFIXES = (".so",)
_MLRT_NAMESPACES = ("trt", "trt_rtx", "ort", "ov", "ncnn", "migx")


def _load_mlrt_plugins(pdir):
    attempts = []
    loaded = []

    for root, _dirs, files in os.walk(pdir):
        for fn in files:
            if fn.startswith(_PLUGIN_PREFIXES) and fn.endswith(_PLUGIN_SUFFIXES):
                path = os.path.join(root, fn)
                attempts.append(path)
                try:
                    core.std.LoadPlugin(path=path)
                    loaded.append(path)
                except vs.Error as exc:
                    attempts[-1] = f"{path}: {exc}"

    if not any(hasattr(core, ns) for ns in _MLRT_NAMESPACES):
        detail = "\n  ".join(attempts[:20]) if attempts else "no candidate Linux .so files found"
        raise RuntimeError(
            "pipeline.vpy: no vs-mlrt runtime plugin available. "
            "Install vapoursynth-mlrt-trt and make sure TensorRT/CUDA library "
            "directories are present in LD_LIBRARY_PATH.\n"
            f"plugins_dir={pdir}\n"
            f"attempts:\n  {detail}"
        )

    return loaded


if plugins_dir:
    if plugins_dir not in sys.path:
        sys.path.insert(0, plugins_dir)  # where setup.sh placed vsmlrt.py
    _load_mlrt_plugins(plugins_dir)

import vsmlrt  # noqa: E402  (must follow plugin loading)

if plugins_dir:
    vsmlrt.plugins_path = plugins_dir
if trtexec_path:
    vsmlrt.trtexec_path = trtexec_path
else:
    detected_trtexec = shutil.which("trtexec")
    if detected_trtexec:
        vsmlrt.trtexec_path = detected_trtexec
if models_dir:
    vsmlrt.models_path = models_dir
elif plugins_dir:
    vsmlrt.models_path = os.path.join(plugins_dir, "models")

device_id = _arg_int("device_id", 0)
num_streams = _arg_int("num_streams", 2)
use_fp16 = _arg_bool("fp16", True)


def _trt_backend():
    return vsmlrt.Backend.TRT(
        fp16=use_fp16,
        # On TensorRT >= 11 the vsmlrt path pre-converts the ONNX to fp16 and
        # requires the engine's input and output formats to be equal. The input
        # format is derived from the clip's bit depth, and we feed RGBH (fp16)
        # whenever fp16 is enabled, so the output format must be fp16 too;
        # otherwise convert_model raises "input format must be equal to output
        # format". (When fp16 is off, vsmlrt skips this path, so 0 is fine.)
        output_format=1 if use_fp16 else 0,
        num_streams=num_streams,
        use_cuda_graph=True,
        device_id=device_id,
    )


# --- source -----------------------------------------------------------------

def _open_source(path):
    """Open the source, preferring lsmas then ffms2."""
    if hasattr(core, "lsmas"):
        return core.lsmas.LWLibavSource(source=path)
    if hasattr(core, "ffms2"):
        return core.ffms2.Source(source=path)
    raise RuntimeError("pipeline.vpy: no source plugin found (need lsmas or ffms2)")


clip = _open_source(video_path)

# remember original YUV format/matrix/range to restore after RGB ML stages
src_format = clip.format

# matrix detection (VideoJaNai heuristic): unknown -> guess by resolution
matrix_in = 2  # unspecified
try:
    matrix_in = clip.get_frame(0).props._Matrix
except (AttributeError, KeyError):
    pass
if matrix_in == 2:
    matrix_in = 5 if clip.height < 720 else 1  # BT.601 (SD) / BT.709 (HD)

range_in = 0  # limited
try:
    range_in = clip.get_frame(0).props._ColorRange
except (AttributeError, KeyError):
    pass


def _to_rgb(c, fmt=vs.RGBH):
    return core.resize.Bicubic(c, format=fmt, matrix_in=matrix_in, range_in=range_in)


def _to_yuv(c):
    return core.resize.Bicubic(
        c,
        format=src_format.id,
        matrix=matrix_in,
        range=range_in,
    )


# --- optional pre-upscale resize -------------------------------------------

pre_resize_factor = _arg_float("pre_resize_factor")  # percent, e.g. 50.0
if do_upscale and pre_resize_factor and pre_resize_factor != 100.0:
    new_w = round(clip.width * pre_resize_factor / 100.0)
    new_h = round(clip.height * pre_resize_factor / 100.0)
    # keep mod2 for chroma subsampling friendliness
    new_w -= new_w % 2
    new_h -= new_h % 2
    clip = core.resize.Bicubic(clip, width=new_w, height=new_h)


# --- super-resolution (R-ESRGAN on TensorRT) -------------------------------

if do_upscale:
    model_name = _arg("model", "animejanaiV3_HD_L2")
    try:
        model = vsmlrt.RealESRGANModel[model_name]
    except KeyError:
        valid = ", ".join(vsmlrt.RealESRGANModel.__members__)
        raise ValueError(
            f"pipeline.vpy: unknown RealESRGAN model {model_name!r}; valid: {valid}"
        )
    clip = _to_rgb(clip, vs.RGBH if use_fp16 else vs.RGBS)
    clip = vsmlrt.RealESRGAN(clip, model=model, backend=_trt_backend())
    clip = _to_yuv(clip)


# --- frame interpolation (RIFE on TensorRT) --------------------------------

if do_rife:
    rife_multi = _arg_fraction("rife_multi", 2)
    rife_model_name = _arg("rife_model", "v4_10")
    try:
        rife_model = vsmlrt.RIFEModel[rife_model_name]
    except KeyError:
        valid = ", ".join(vsmlrt.RIFEModel.__members__)
        raise ValueError(
            f"pipeline.vpy: unknown RIFE model {rife_model_name!r}; valid: {valid}"
        )
    rife_clip = _to_rgb(clip, vs.RGBH)
    # vsmlrt.RIFE (v1 implementation) requires the frame dimensions to be a
    # multiple of 32 and does not pad internally — a 4K upscale yields 3840x2160
    # and 2160 % 32 != 0, which raises "tile size must be divisible by 32". Pad
    # the right/bottom edges up to the next multiple of 32, then crop back so the
    # output keeps the source resolution.
    pad_w = (32 - rife_clip.width % 32) % 32
    pad_h = (32 - rife_clip.height % 32) % 32
    if pad_w or pad_h:
        rife_clip = core.std.AddBorders(rife_clip, right=pad_w, bottom=pad_h)
    rife_clip = vsmlrt.RIFE(
        rife_clip,
        multi=rife_multi,
        model=rife_model,
        backend=_trt_backend(),
    )
    if pad_w or pad_h:
        rife_clip = core.std.Crop(rife_clip, right=pad_w, bottom=pad_h)
    clip = _to_yuv(rife_clip)


# --- optional final resize --------------------------------------------------

final_resize_height = _arg_int("final_resize_height")
if final_resize_height and final_resize_height > 0 and final_resize_height != clip.height:
    new_h = final_resize_height
    new_w = round(clip.width * new_h / clip.height)
    new_w -= new_w % 2
    new_h -= new_h % 2
    clip = core.resize.Bicubic(clip, width=new_w, height=new_h)


clip.set_output()
