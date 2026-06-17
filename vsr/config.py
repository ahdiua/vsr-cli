"""Runtime location + config persistence.

Resolution priority for each runtime path:
    explicit CLI/arg override > environment variable > config.toml > auto-detect.

The runtime consists of:
  - vspipe          : the VapourSynth pipe executable
  - ffmpeg          : ffmpeg executable
  - plugins dir     : VapourSynth plugins dir holding vstrt + vsmlrt-cuda + models
  - models dir      : usually <plugins>/models, holding RealESRGANv2/ and rife/
  - pipeline.vpy    : shipped next to this repo

``setup.sh`` writes an initial config.toml after provisioning the runtime.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:  # py311+
    import tomllib as _toml_read
except ModuleNotFoundError:  # pragma: no cover - py38-310
    import tomli as _toml_read  # type: ignore

import tomli_w

from . import presets

APP_NAME = "vsr"


def config_path() -> Path:
    """Return the config.toml path (XDG-ish, overridable via VSR_CONFIG)."""
    env = os.environ.get("VSR_CONFIG")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(Path.home(), ".config")
    return Path(base) / APP_NAME / "config.toml"


def repo_root() -> Path:
    """Repo root = parent of the vsr package dir (where pipeline.vpy lives)."""
    return Path(__file__).resolve().parent.parent


def default_pipeline_vpy() -> Path:
    return repo_root() / "pipeline.vpy"


@dataclass
class RuntimeConfig:
    """Resolved runtime paths and defaults."""

    vspipe: str = ""
    ffmpeg: str = "ffmpeg"
    plugins_dir: str = ""
    models_dir: str = ""
    pipeline_vpy: str = ""
    # defaults
    encoder: str = presets.DEFAULT_ENCODER
    num_streams: int = 2
    device_id: int = 0
    fp16: bool = True
    trt_args: dict = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path | None = None) -> "RuntimeConfig":
        path = path or config_path()
        data: dict = {}
        if path.is_file():
            with open(path, "rb") as f:
                data = _toml_read.load(f)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: Path | None = None) -> Path:
        path = path or config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            tomli_w.dump({k: v for k, v in asdict(self).items() if v not in ("", None)}, f)
        return path


def _which(name: str) -> str | None:
    return shutil.which(name)


def _autodetect_vspipe() -> str | None:
    # On the conda-forge install, vspipe lives next to the python interpreter.
    cand = Path(sys.executable).parent / ("vspipe.exe" if os.name == "nt" else "vspipe")
    if cand.is_file():
        return str(cand)
    return _which("vspipe")


def _autodetect_models(plugins_dir: str | None) -> str | None:
    if plugins_dir:
        m = Path(plugins_dir) / "models"
        if m.is_dir():
            return str(m)
    return None


def resolve(cfg: RuntimeConfig, **overrides: object) -> RuntimeConfig:
    """Fill blanks via env vars then auto-detection, applying CLI overrides last.

    overrides: vspipe, ffmpeg, plugins_dir, models_dir, pipeline_vpy, device_id,
    num_streams, encoder, fp16 — any non-None value wins.
    """
    # env
    cfg.vspipe = cfg.vspipe or os.environ.get("VSR_VSPIPE", "")
    cfg.ffmpeg = cfg.ffmpeg or os.environ.get("VSR_FFMPEG", "")
    cfg.plugins_dir = cfg.plugins_dir or os.environ.get("VSR_PLUGINS", "")
    cfg.models_dir = cfg.models_dir or os.environ.get("VSR_MODELS", "")
    cfg.pipeline_vpy = cfg.pipeline_vpy or os.environ.get("VSR_PIPELINE", "")

    # auto-detect
    if not cfg.vspipe:
        cfg.vspipe = _autodetect_vspipe() or ""
    if not cfg.ffmpeg:
        cfg.ffmpeg = _which("ffmpeg") or "ffmpeg"
    if not cfg.models_dir:
        cfg.models_dir = _autodetect_models(cfg.plugins_dir) or ""
    if not cfg.pipeline_vpy:
        cfg.pipeline_vpy = str(default_pipeline_vpy())

    # CLI overrides (highest priority)
    for key, val in overrides.items():
        if val is not None and hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg


def diagnose(cfg: RuntimeConfig) -> list[tuple[str, bool, str]]:
    """Return [(label, ok, detail)] runtime health checks for `vsr doctor`."""
    checks: list[tuple[str, bool, str]] = []

    vspipe_ok = bool(cfg.vspipe) and Path(cfg.vspipe).is_file()
    checks.append(("vspipe", vspipe_ok, cfg.vspipe or "not found"))

    ffmpeg_ok = bool(_which(cfg.ffmpeg)) or Path(cfg.ffmpeg).is_file()
    checks.append(("ffmpeg", ffmpeg_ok, cfg.ffmpeg))

    vpy_ok = Path(cfg.pipeline_vpy).is_file()
    checks.append(("pipeline.vpy", vpy_ok, cfg.pipeline_vpy))

    models = Path(cfg.models_dir) if cfg.models_dir else None
    re_ok = bool(models) and (models / "RealESRGANv2").is_dir()
    checks.append(("RealESRGAN models", re_ok, str(models / "RealESRGANv2") if models else "models_dir unset"))
    rife_ok = bool(models) and (models / "rife").is_dir()
    checks.append(("RIFE models", rife_ok, str(models / "rife") if models else "models_dir unset"))

    nvsmi = _which("nvidia-smi")
    checks.append(("nvidia-smi", bool(nvsmi), nvsmi or "not found"))

    return checks
