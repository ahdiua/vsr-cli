"""Encoder presets and model/TRT defaults.

The ffmpeg video-encoder preset strings are adopted from VideoJaNai
(MainWindowViewModel.cs). Each is the argument list that follows ``-c:v`` so
the pipeline can splice it directly into the ffmpeg command.
``-max_interleave_delta 0`` is appended by ``build_video_codec_args``.
"""

from __future__ import annotations

import shlex

# name -> ffmpeg "-c:v ..." argument string (without -max_interleave_delta).
ENCODER_PRESETS: dict[str, str] = {
    # NVIDIA hardware HEVC (default) — fast, 10-bit.
    "nvenc": "hevc_nvenc -preset p7 -profile:v main10 -b:v 50M",
    # Software HEVC — high quality, slow.
    "x265": 'libx265 -crf 16 -preset slow -x265-params "sao=0:bframes=8:psy-rd=1.5:psy-rdoq=2:aq-mode=3:ref=6"',
    # Software H.264.
    "x264": "libx264 -crf 13 -preset slow",
    # Lossless intra codec.
    "ffv1": "ffv1",
}

DEFAULT_ENCODER = "nvenc"

# RealESRGAN model file names under <models_dir>/RealESRGANv2. The pipeline also
# accepts absolute paths, relative paths, stems without .onnx, and legacy vsmlrt
# enum names such as animejanaiV3_HD_L2.
REALESRGAN_MODELS: dict[str, str] = {
    "realesr-animevideov3.onnx": "RealESRGAN animevideo v3 (4x)",
    "RealESRGANv2-animevideo-xsx2.onnx": "RealESRGAN animevideo xs (2x)",
    "RealESRGANv2-animevideo-xsx4.onnx": "RealESRGAN animevideo xs (4x)",
    "animejanaiV2L1.onnx": "AnimeJaNai V2 L1 (2x, light)",
    "animejanaiV2L2.onnx": "AnimeJaNai V2 L2 (2x)",
    "animejanaiV2L3.onnx": "AnimeJaNai V2 L3 (2x, heavy)",
    "animejanaiV3-HD-L1.onnx": "AnimeJaNai V3 HD L1 (2x, light)",
    "animejanaiV3-HD-L2.onnx": "AnimeJaNai V3 HD L2 (2x)",
    "animejanaiV3-HD-L3.onnx": "AnimeJaNai V3 HD L3 (2x, heavy)",
    "Ani4Kv2-G6i2-Compact.onnx": "Ani4K v2 Compact (2x)",
    "Ani4Kv2-G6i2-UltraCompact.onnx": "Ani4K v2 UltraCompact (2x)",
}

DEFAULT_REALESRGAN_MODEL = "animejanaiV3-HD-L2.onnx"

# RIFE model file names under <models_dir>/rife. The pipeline also accepts
# absolute paths, relative paths, stems without .onnx, and legacy enum names
# such as v4_10.
RIFE_MODELS: dict[str, str] = {
    "rife_v4.4.onnx": "RIFE v4.4",
    "rife_v4.6.onnx": "RIFE v4.6",
    "rife_v4.10.onnx": "RIFE v4.10",
    "rife_v4.15.onnx": "RIFE v4.15",
    "rife_v4.18.onnx": "RIFE v4.18",
    "rife_v4.22.onnx": "RIFE v4.22",
    "rife_v4.25.onnx": "RIFE v4.25",
    "rife_v4.26.onnx": "RIFE v4.26",
    "rife_v4.15_lite.onnx": "RIFE v4.15 lite (fast)",
    "rife_v4.25_lite.onnx": "RIFE v4.25 lite (fast)",
}

# v4_10 is the newest RIFE model shipped in the standard vs-mlrt `models` pack
# (and even the v16.test1 prerelease `contrib-models` pack tops out here); newer
# versions like v4.22 are not bundled, so default to one that always extracts.
DEFAULT_RIFE_MODEL = "rife_v4.10.onnx"


def encoder_args(name: str) -> str:
    """Return the ``-c:v ...`` argument string for a preset name.

    Raises KeyError with the valid names if unknown.
    """
    if name not in ENCODER_PRESETS:
        raise KeyError(
            f"unknown encoder preset {name!r}; choose from {', '.join(ENCODER_PRESETS)}"
        )
    return ENCODER_PRESETS[name]


def build_video_codec_args(encoder: str | None, ffmpeg_args: str | None) -> list[str]:
    """Build the ffmpeg ``-c:v ...`` token list.

    ``ffmpeg_args`` (a raw user string) overrides the named preset when provided.
    Always appends ``-max_interleave_delta 0`` like VideoJaNai does.
    """
    if ffmpeg_args:
        base = ffmpeg_args
    else:
        base = encoder_args(encoder or DEFAULT_ENCODER)
    tokens = ["-c:v", *shlex.split(base)]
    if "-max_interleave_delta" not in tokens:
        tokens += ["-max_interleave_delta", "0"]
    return tokens
