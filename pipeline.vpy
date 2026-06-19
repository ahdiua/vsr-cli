"""Parametrized VapourSynth pipeline for vsr-cli.

Run via vspipe, passing parameters with --arg key=value, e.g.:

    vspipe -c y4m \
        --arg video_path=/in.mkv \
        --arg upscale=1 --arg model=animejanaiV3-HD-L2.onnx \
        --arg rife=1 --arg rife_multi=2 --arg rife_model=rife_v4.10.onnx \
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
                    "libbestsource", "bestsource", "libbs",
                    "libakarin", "akarin")
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


def _effective_cpu_count():
    """CPU count this container may actually use.

    VapourSynth's default thread count is ``os.cpu_count()``, which inside a
    container returns the *host* core count (e.g. 256). When the container is
    cgroup-limited to a fraction of that (e.g. an 18-core quota), VS spawns far
    more worker threads than there are usable cores, and they thrash on the CPU
    — which starves the GPU (utilisation oscillates 10–90%) and tanks throughput.
    Honour the cgroup CPU quota (v2 ``cpu.max``, then v1 cfs quota) and the CPU
    affinity mask, falling back to ``os.cpu_count()``.
    """
    limits = []
    try:  # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()[:2]
            if quota != "max" and int(period) > 0:
                limits.append(max(1, round(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            quota = int(fh.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            period = int(fh.read())
        if quota > 0 and period > 0:
            limits.append(max(1, quota // period))
    except (OSError, ValueError):
        pass
    try:  # CPU affinity mask (cpuset)
        limits.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    limits.append(os.cpu_count() or 1)
    return min(limits)


num_threads = _arg_int("num_threads")
if not num_threads or num_threads <= 0:
    num_threads = _effective_cpu_count()
if num_threads and num_threads > 0:
    core.num_threads = num_threads


def _trt_backend(fp16=None, output_format=None):
    fp16 = use_fp16 if fp16 is None else fp16
    if output_format is None:
        output_format = 1 if fp16 else 0
    return vsmlrt.Backend.TRT(
        fp16=fp16,
        # On TensorRT >= 11 the vsmlrt path pre-converts the ONNX to fp16 and
        # requires the engine's input and output formats to be equal. The input
        # format is derived from the clip's bit depth, and we feed RGBH (fp16)
        # whenever fp16 is enabled, so the output format must be fp16 too;
        # otherwise convert_model raises "input format must be equal to output
        # format". (When fp16 is off, vsmlrt skips this path, so 0 is fine.)
        output_format=output_format,
        num_streams=num_streams,
        use_cuda_graph=True,
        builder_optimization_level=5,
        device_id=device_id,
    )


def _unique(items):
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _model_roots(*subdirs):
    base = models_dir or getattr(vsmlrt, "models_path", "")
    roots = []
    if base:
        for subdir in subdirs:
            roots.append(os.path.join(base, subdir))
        roots.append(base)
    return _unique(roots)


def _path_variants(spec, filename_prefix=""):
    spec = str(spec).strip()
    dirname = os.path.dirname(spec)
    basename = os.path.basename(spec)
    variants = [spec]
    if filename_prefix and not dirname and not basename.startswith(filename_prefix):
        variants.append(filename_prefix + spec)

    out = []
    for variant in variants:
        out.append(variant)
        if not os.path.splitext(variant)[1]:
            out.append(variant + ".onnx")
    return _unique(out)


def _resolve_model_path(spec, subdirs, filename_prefix=""):
    candidates = []
    for variant in _path_variants(spec, filename_prefix):
        expanded = os.path.expanduser(variant)
        if os.path.isabs(expanded):
            candidates.append(expanded)
            continue
        for root in _model_roots(*subdirs):
            candidates.append(os.path.join(root, expanded))

    candidates = _unique(candidates)
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    searched = "\n  ".join(candidates) if candidates else "(models_dir is unset)"
    raise FileNotFoundError(
        f"pipeline.vpy: model file {spec!r} not found. Searched:\n  {searched}"
    )


def _onnx_value_format(value):
    tensor_type = value.type.tensor_type
    elem_type = tensor_type.elem_type
    if elem_type == 10:  # TensorProto.FLOAT16
        return 1
    if elem_type == 1:  # TensorProto.FLOAT
        return 0
    return None


def _onnx_model_info(network_path):
    try:
        import onnx
    except Exception:
        return None

    try:
        model = onnx.load(network_path, load_external_data=False)
    except TypeError:
        model = onnx.load(network_path)
    initializer_names = {item.name for item in model.graph.initializer}

    input_formats = [
        _onnx_value_format(item)
        for item in model.graph.input
        if item.name not in initializer_names
    ]
    input_formats = [item for item in input_formats if item is not None]

    output_formats = [
        _onnx_value_format(item)
        for item in model.graph.output
    ]
    output_formats = [item for item in output_formats if item is not None]

    input_format = input_formats[0] if input_formats else None
    output_format = output_formats[0] if output_formats else input_format

    tensor_types = [item.data_type for item in model.graph.initializer]
    tensor_types.extend(
        item.type.tensor_type.elem_type
        for item in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
        if item.type.HasField("tensor_type")
    )

    return {
        "input_format": input_format,
        "output_format": output_format,
        "has_fp16_tensor": 10 in tensor_types,  # TensorProto.FLOAT16
    }


def _custom_model_plan(network_path):
    info = _onnx_model_info(network_path)
    if info is None:
        return vs.RGBH if use_fp16 else vs.RGBS, _trt_backend()

    input_format = info["input_format"]
    output_format = info["output_format"]
    preconverted_fp16 = (
        info["has_fp16_tensor"] or input_format == 1 or output_format == 1
    )

    if not preconverted_fp16:
        return vs.RGBH if use_fp16 else vs.RGBS, _trt_backend()

    # A preconverted fp16 ONNX should not go through vsmlrt.convert_model again.
    if input_format == 1:
        rgb_format = vs.RGBH
    elif input_format == 0:
        rgb_format = vs.RGBS
    else:
        rgb_format = vs.RGBH if use_fp16 else vs.RGBS

    if output_format is None:
        output_format = 1 if rgb_format == vs.RGBH else 0
    return rgb_format, _trt_backend(fp16=False, output_format=output_format)


def _realesrgan_file(clip, network_path, backend):
    overlap_w = overlap_h = 8
    (tile_w, tile_h), (overlap_w, overlap_h) = vsmlrt.calc_tilesize(
        tiles=None, tilesize=None,
        width=clip.width, height=clip.height,
        multiple=1,
        overlap_w=overlap_w, overlap_h=overlap_h,
    )
    backend = vsmlrt.init_backend(
        backend=backend,
        trt_opt_shapes=(tile_w, tile_h),
    )
    return vsmlrt.inference_with_fallback(
        clips=[clip],
        network_path=network_path,
        overlap=(overlap_w, overlap_h),
        tilesize=(tile_w, tile_h),
        backend=backend,
    )


def _rife_merge_file(clipa, clipb, mask, network_path, backend):
    implementation_v2 = os.path.basename(os.path.dirname(network_path)) == "rife_v2"
    multiple = 1 if implementation_v2 else 32
    clips = [clipa, clipb, mask]
    if not implementation_v2:
        clips.extend(vsmlrt.get_rife_input(clipa))

    (tile_w, tile_h), (overlap_w, overlap_h) = vsmlrt.calc_tilesize(
        tiles=None, tilesize=None,
        width=clipa.width, height=clipa.height,
        multiple=multiple,
        overlap_w=0, overlap_h=0,
    )
    backend = vsmlrt.init_backend(
        backend=backend,
        trt_opt_shapes=(tile_w, tile_h),
    )
    return vsmlrt.inference_with_fallback(
        clips=clips,
        network_path=network_path,
        overlap=(overlap_w, overlap_h),
        tilesize=(tile_w, tile_h),
        backend=backend,
    )


def _rife_file(clip, multi, network_path, backend):
    gray_format = vs.GRAYS if clip.format.bits_per_sample == 32 else vs.GRAYH

    if int(multi) == multi:
        multi = int(multi)
        if multi < 2:
            raise ValueError("pipeline.vpy: RIFE multi must be at least 2")

        initial = core.std.Interleave([clip] * (multi - 1))
        terminal = clip.std.DuplicateFrames(frames=clip.num_frames - 1).std.Trim(first=1)
        terminal = core.std.Interleave([terminal] * (multi - 1))
        timepoint = core.std.Interleave([
            clip.std.BlankClip(format=gray_format, color=i / multi, length=1)
            for i in range(1, multi)
        ]).std.Loop(clip.num_frames)

        output0 = _rife_merge_file(initial, terminal, timepoint, network_path, backend)
        clip = vsmlrt.bits_as(clip, output0)
        initial = core.std.Interleave([clip] * (multi - 1))

        if hasattr(core, "akarin") and hasattr(core.akarin, "Select"):
            output = core.akarin.Select([output0, initial], initial, "x._SceneChangeNext 1 0 ?")
        else:
            def handler(n, f):
                if f.props.get("_SceneChangeNext"):
                    return initial
                return output0
            output = core.std.FrameEval(output0, handler, initial)

        if multi == 2:
            res = core.std.Interleave([clip, output])
        else:
            res = core.std.Interleave([
                clip,
                *(output.std.SelectEvery(cycle=multi - 1, offsets=i) for i in range(multi - 1))
            ])

        if clip.fps_num != 0 and clip.fps_den != 0:
            return res.std.AssumeFPS(fpsnum=clip.fps_num * multi, fpsden=clip.fps_den)
        return res

    if clip.fps_num == 0 or clip.fps_den == 0:
        src_fps = fractions.Fraction(1)
    else:
        src_fps = clip.fps

    dst_fps = src_fps * multi
    src_frames = clip.num_frames
    dst_frames = min(int(src_frames * multi), 2 ** 31 - 1)
    dst_duration = (src_fps / dst_fps).numerator
    src_duration = (src_fps / dst_fps).denominator

    if not hasattr(core, "akarin") or \
       not hasattr(core.akarin, "PropExpr") or \
       not hasattr(core.akarin, "PickFrames"):
        raise RuntimeError("pipeline.vpy: fractional RIFE multi requires akarin")

    left_indices = []
    right_indices = []
    timepoints = []
    output_indices = []
    for i in range(dst_frames):
        current_time = dst_duration * i
        if current_time % src_duration == 0:
            output_indices.append(current_time // src_duration)
        else:
            left_index = current_time // src_duration
            if left_index + 1 >= src_frames:
                output_indices.append(src_frames - 1)
                break
            output_indices.append(src_frames + len(timepoints))
            left_indices.append(left_index)
            right_indices.append(left_index + 1)
            left_time = src_duration * left_index
            timepoints.append((current_time - left_time) / src_duration)

    left_clip = core.akarin.PickFrames(clip, left_indices)
    right_clip = core.akarin.PickFrames(clip, right_indices)
    tp_clip = core.std.BlankClip(clip, format=gray_format, length=len(timepoints))
    tp_clip = tp_clip.akarin.PropExpr(lambda: dict(_tp=timepoints)).akarin.Expr("x._tp")

    output0 = _rife_merge_file(left_clip, right_clip, tp_clip, network_path, backend)
    clip0 = vsmlrt.bits_as(clip, output0)
    left0 = vsmlrt.bits_as(left_clip, output0)
    output = core.akarin.Select([output0, left0], left0, "x._SceneChangeNext 1 0 ?")
    res = core.akarin.PickFrames(clip0 + output, output_indices)

    if clip.fps_num != 0 and clip.fps_den != 0:
        return res.std.AssumeFPS(fpsnum=dst_fps.numerator, fpsden=dst_fps.denominator)
    return res


# --- source -----------------------------------------------------------------

def _open_source(path):
    """Open the source, preferring lsmas, then ffms2, then bestsource.

    The path is resolved to an absolute path first: L-SMASH-Works writes/reuses
    its ``.lwi`` index next to the source, and a relative path makes lsmas treat
    an index that was built from an absolute path as stale and rebuild it — slow,
    and flaky on overlay/fuse mounts (e.g. AutoDL's data disk), where the rebuild
    can fail outright. Absolute paths keep the index reuse deterministic.

    lsmas is tried first, but unlike a plain missing-plugin check a *runtime*
    failure (index construction errors, unreadable container, etc.) now falls
    through to the next source plugin instead of aborting the whole pipeline.
    """
    path = os.path.abspath(path)
    errors = []
    for ns, fn, kw in (
        ("lsmas", "LWLibavSource", {"source": path}),
        ("ffms2", "Source", {"source": path}),
        ("bs", "VideoSource", {"source": path}),
    ):
        plugin = getattr(core, ns, None)
        if plugin is None:
            continue
        try:
            return getattr(plugin, fn)(**kw)
        except vs.Error as exc:
            errors.append(f"{ns}: {exc}")
    detail = "; ".join(errors) if errors else "no source plugin found (need lsmas, ffms2, or bestsource)"
    raise RuntimeError(f"pipeline.vpy: could not open source {path!r}: {detail}")


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


def _ensure_rgb(c, fmt):
    """Return `c` as RGB in `fmt`.

    If `c` is already RGB (e.g. straight out of the upscaler when upscale+rife
    are chained), only adjust the pixel format/precision — a cheap same-size
    convert — instead of a full YUV<->RGB matrix conversion. This lets the
    upscale->RIFE handoff skip the redundant 4K RGB->YUV->RGB round-trip, which
    on CPU-bound boxes is a large chunk of per-frame cost.
    """
    if c.format.color_family == vs.RGB:
        return c if c.format.id == fmt else core.resize.Bicubic(c, format=fmt)
    return _to_rgb(c, fmt)


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
    model_spec = _arg("model", "animejanaiV3-HD-L2.onnx")
    if model_spec in vsmlrt.RealESRGANModel.__members__:
        model = vsmlrt.RealESRGANModel[model_spec]
        clip = _ensure_rgb(clip, vs.RGBH if use_fp16 else vs.RGBS)
        clip = vsmlrt.RealESRGAN(clip, model=model, backend=_trt_backend())
    else:
        network_path = _resolve_model_path(model_spec, ("RealESRGANv2",))
        rgb_format, backend = _custom_model_plan(network_path)
        clip = _ensure_rgb(clip, rgb_format)
        clip = _realesrgan_file(clip, network_path, backend)
    # Leave the clip in RGB when RIFE follows so the upscale->RIFE handoff skips
    # the redundant 4K RGB->YUV->RGB round-trip; convert back only at the end.
    if not do_rife:
        clip = _to_yuv(clip)


# --- frame interpolation (RIFE on TensorRT) --------------------------------

if do_rife:
    if not (hasattr(core, "akarin") and hasattr(core.akarin, "Select")):
        sys.stderr.write(
            "pipeline.vpy: WARNING: akarin plugin not found — RIFE scene-change "
            "handling falls back to a per-frame Python callback (GIL-serialized, "
            "much slower). Install libakarin.so into plugins_dir (run setup.sh).\n"
        )
    rife_multi = _arg_fraction("rife_multi", 2)
    rife_model_spec = _arg("rife_model", "rife_v4.10.onnx")
    if rife_model_spec in vsmlrt.RIFEModel.__members__:
        rife_clip = _ensure_rgb(clip, vs.RGBH if use_fp16 else vs.RGBS)
    else:
        network_path = _resolve_model_path(
            rife_model_spec,
            ("rife", "rife_v2"),
            filename_prefix="rife_",
        )
        rgb_format, backend = _custom_model_plan(network_path)
        rife_clip = _ensure_rgb(clip, rgb_format)
    # vsmlrt.RIFE (v1 implementation) requires the frame dimensions to be a
    # multiple of 32 and does not pad internally — a 4K upscale yields 3840x2160
    # and 2160 % 32 != 0, which raises "tile size must be divisible by 32". Pad
    # the right/bottom edges up to the next multiple of 32, then crop back so the
    # output keeps the source resolution.
    pad_w = (32 - rife_clip.width % 32) % 32
    pad_h = (32 - rife_clip.height % 32) % 32
    if pad_w or pad_h:
        rife_clip = core.std.AddBorders(rife_clip, right=pad_w, bottom=pad_h)
    if rife_model_spec in vsmlrt.RIFEModel.__members__:
        rife_model = vsmlrt.RIFEModel[rife_model_spec]
        rife_clip = vsmlrt.RIFE(
            rife_clip,
            multi=rife_multi,
            model=rife_model,
            backend=_trt_backend(),
        )
    else:
        rife_clip = _rife_file(
            rife_clip,
            rife_multi,
            network_path,
            backend,
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
