"""Parametrized VapourSynth pipeline for vsr-cli.

Run via vspipe, passing parameters with --arg key=value, e.g.:

    vspipe -c y4m \
        --arg video_path=/in.mkv \
        --arg upscale=1 --arg model=animejanaiV3_HD_L2 \
        --arg rife=1 --arg rife_multi=2 --arg rife_model=v4_22 \
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

# --- load vs-mlrt plugins + import vsmlrt -----------------------------------
# The vs-mlrt plugins usually live in a non-autoload dir, and vsmlrt.py derives
# its paths from a loaded plugin. So load the .so files and add vsmlrt.py to
# sys.path BEFORE importing vsmlrt, then override its paths explicitly.

plugins_dir = _arg("plugins_dir")
models_dir = _arg("models_dir")

_PLUGIN_PREFIXES = ("libvstrt", "libvsort", "libvsov", "libvsncnn",
                    "vstrt", "vsort", "vsov", "vsncnn")
_PLUGIN_SUFFIXES = (".so", ".dll")


def _load_mlrt_plugins(pdir):
    for root, _dirs, files in os.walk(pdir):
        for fn in files:
            if fn.startswith(_PLUGIN_PREFIXES) and fn.endswith(_PLUGIN_SUFFIXES):
                try:
                    core.std.LoadPlugin(path=os.path.join(root, fn))
                except vs.Error:
                    pass  # already autoloaded / not a plugin — ignore


if plugins_dir:
    if plugins_dir not in sys.path:
        sys.path.insert(0, plugins_dir)  # where setup.sh placed vsmlrt.py
    _load_mlrt_plugins(plugins_dir)

import vsmlrt  # noqa: E402  (must follow plugin loading)

if plugins_dir:
    vsmlrt.plugins_path = plugins_dir
    vsmlrt.trtexec_path = os.path.join(plugins_dir, "vsmlrt-cuda", "trtexec")
    vsmlrt.tensorrt_rtx_path = os.path.join(plugins_dir, "vsmlrt-cuda", "tensorrt_rtx")
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
    rife_model_name = _arg("rife_model", "v4_22")
    try:
        rife_model = vsmlrt.RIFEModel[rife_model_name]
    except KeyError:
        valid = ", ".join(vsmlrt.RIFEModel.__members__)
        raise ValueError(
            f"pipeline.vpy: unknown RIFE model {rife_model_name!r}; valid: {valid}"
        )
    rife_clip = _to_rgb(clip, vs.RGBH)
    rife_clip = vsmlrt.RIFE(
        rife_clip,
        multi=rife_multi,
        model=rife_model,
        backend=_trt_backend(),
    )
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
