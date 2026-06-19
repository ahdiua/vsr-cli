"""Runtime location + config persistence.

Resolution priority for each runtime path:
    explicit CLI/arg override > environment variable > config.toml > auto-detect.

The runtime consists of:
  - vspipe          : the VapourSynth pipe executable
  - ffmpeg          : ffmpeg executable
  - plugins dir     : vsmlrt.py, models, and optional manually dropped VS plugins
  - models dir      : usually <plugins>/models, holding RealESRGANv2/ and rife/
  - trtexec         : optional Linux TensorRT engine builder executable
  - pipeline.vpy    : shipped next to this repo

``setup.sh`` writes an initial config.toml after provisioning the runtime.
"""

from __future__ import annotations

import os
import re
import site
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import tomllib as _toml_read
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
    ffmpeg: str = ""
    plugins_dir: str = ""
    models_dir: str = ""
    trtexec: str = ""
    pipeline_vpy: str = ""
    # defaults
    encoder: str = presets.DEFAULT_ENCODER
    num_streams: int = 2
    # VapourSynth worker threads. 0 = auto: pipeline.vpy detects the container's
    # cgroup CPU quota / affinity and sets core.num_threads accordingly, instead
    # of VS's default os.cpu_count() (the *host* core count in a container, which
    # oversubscribes and thrashes). Set a positive value to override.
    num_threads: int = 0
    device_id: int = 0
    fp16: bool = True
    # LD_PRELOAD shim that filters the GPU list NVENC sees, for containers where
    # the host exposes GPUs not assigned to this container (NVIDIA driver 570+ bug
    # → hevc_nvenc "unsupported device"). Path to libnvenc_fix.so from
    # flexgrip/nvidia-gpu-enumeration. Empty = disabled. Injected into ffmpeg only.
    nvenc_fix: str = ""

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


def _compact_trt_version(value: str | bytes | int | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("ascii", "ignore")
    text = str(value).strip()
    if not text.isdigit():
        return None
    raw = int(text)
    return raw // 10000, (raw // 100) % 100, raw % 100


def _format_trt_version(version: tuple[int, int, int] | None) -> str:
    if not version:
        return "unknown"
    return ".".join(str(part) for part in version)


def _parse_trtexec_version(text: str) -> tuple[int, int, int] | None:
    match = re.search(r"TensorRT version:\s*(\d+)\.(\d+)(?:\.(\d+))?", text)
    if match:
        return (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3) or 0),
        )
    match = re.search(r"TensorRT(?:\.trtexec)?\s*\[TensorRT v(\d+)\]", text)
    if match:
        return _compact_trt_version(match.group(1))
    match = re.search(r"\[TensorRT v(\d+)\]", text)
    if match:
        return _compact_trt_version(match.group(1))
    return None


def _probe_core_trt_version(plugins_dir: str | None) -> tuple[int, int, int] | None:
    code = """
from vapoursynth import core
if not hasattr(core, "trt"):
    raise SystemExit(1)
version = core.trt.Version()
value = version.get("tensorrt_version_build") or version.get("tensorrt_version")
if isinstance(value, bytes):
    value = value.decode("ascii", "ignore")
print(value)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            env=env_with_runtime_libs(plugins_dir),
            text=True,
            timeout=20,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return _compact_trt_version((result.stdout or "").strip())


def _probe_trtexec_version(path: str | Path) -> tuple[int, int, int] | None:
    path = str(path)
    for arg in ("--version", "--help"):
        try:
            result = subprocess.run(
                [path, arg],
                capture_output=True,
                env=env_with_runtime_libs(None),
                text=True,
                timeout=10,
            )
        except Exception:
            continue
        version = _parse_trtexec_version((result.stdout or "") + "\n" + (result.stderr or ""))
        if version:
            return version
    return None


def _same_trt_engine_abi(
    runtime: tuple[int, int, int] | None,
    builder: tuple[int, int, int] | None,
) -> bool:
    if not runtime or not builder:
        return True
    return runtime[:2] == builder[:2]


def _autodetect_vspipe() -> str | None:
    # On the conda-forge install, vspipe lives next to the python interpreter.
    cand = Path(sys.executable).parent / ("vspipe.exe" if os.name == "nt" else "vspipe")
    if cand.is_file():
        return str(cand)
    return _which("vspipe")


def _autodetect_ffmpeg() -> str | None:
    path_hit = _which("ffmpeg")
    if path_hit:
        return path_hit
    system_ffmpeg = Path("/usr/local/bin/ffmpeg")
    if system_ffmpeg.is_file() and os.access(system_ffmpeg, os.X_OK):
        return str(system_ffmpeg)
    return None


def _autodetect_models(plugins_dir: str | None) -> str | None:
    if plugins_dir:
        m = Path(plugins_dir) / "models"
        if m.is_dir():
            return str(m)
    return None


def _autodetect_trtexec(plugins_dir: str | None) -> str | None:
    runtime_version = _probe_core_trt_version(plugins_dir)
    candidates: list[Path] = [
        Path(sys.executable).parent / ("trtexec.exe" if os.name == "nt" else "trtexec"),
        Path(sys.prefix) / "bin" / "trtexec",
    ]
    for env_name in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        if os.environ.get(env_name):
            candidates.append(Path(os.environ[env_name]) / "bin" / "trtexec")
    for env_name in ("TENSORRT_HOME", "TRT_HOME"):
        if os.environ.get(env_name):
            candidates.append(Path(os.environ[env_name]) / "bin" / "trtexec")
    candidates.extend(
        [
            Path("/usr/src/tensorrt/bin/trtexec"),
            Path("/usr/local/tensorrt/bin/trtexec"),
            Path("/usr/local/TensorRT/bin/trtexec"),
        ]
    )

    for root in _python_package_roots():
        if root.is_dir():
            candidates.extend(sorted(p for p in root.rglob("trtexec") if p.is_file()))

    for candidate in candidates:
        if not candidate.is_file():
            continue
        if _same_trt_engine_abi(runtime_version, _probe_trtexec_version(candidate)):
            return str(candidate)

    path_hit = _which("trtexec")
    if path_hit and _same_trt_engine_abi(runtime_version, _probe_trtexec_version(path_hit)):
        return path_hit

    if plugins_dir:
        root = Path(plugins_dir)
        if root.is_dir():
            hits = sorted(p for p in root.rglob("trtexec") if p.is_file())
            for hit in hits:
                if _same_trt_engine_abi(runtime_version, _probe_trtexec_version(hit)):
                    return str(hit)

    return None


_SHARED_LIB_PATTERNS = (
    "libvstrt*.so*",
    "libnvinfer*.so*",
    "libnvinfer_plugin*.so*",
    "libnvonnxparser*.so*",
    "libcudart*.so*",
    "libcublas*.so*",
    "libcudnn*.so*",
    "libnvrtc*.so*",
    "libtensorrt*.so*",
)


def _python_package_roots() -> list[Path]:
    roots: list[Path] = []

    def add_root(path: str | Path | None) -> None:
        if not path:
            return
        root = Path(path)
        if root.exists() and root not in roots:
            roots.append(root)

    for path in (sys.prefix, getattr(sys, "base_prefix", sys.prefix)):
        add_root(Path(path) / "lib")
    for env_name in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        if os.environ.get(env_name):
            add_root(Path(os.environ[env_name]) / "lib")

    if os.environ.get("CONDA_PREFIX"):
        conda_prefix = Path(os.environ["CONDA_PREFIX"])
        if conda_prefix.parent.name == "envs":
            add_root(conda_prefix.parent.parent / "lib")
    if os.environ.get("CONDA_EXE"):
        add_root(Path(os.environ["CONDA_EXE"]).parent.parent / "lib")
    prefix = Path(sys.prefix)
    if prefix.parent.name == "envs":
        add_root(prefix.parent.parent / "lib")

    try:
        for path in site.getsitepackages():
            add_root(path)
    except Exception:
        pass

    try:
        add_root(site.getusersitepackages())
    except Exception:
        pass

    return roots


def _python_shared_lib_dirs() -> list[Path]:
    roots: list[Path] = []

    def add_root(path: str | Path | None) -> None:
        if not path:
            return
        root = Path(path)
        if root.exists() and root not in roots:
            roots.append(root)

    def add_conda_base_roots(base: str | Path | None) -> None:
        if not base:
            return
        base_path = Path(base)
        add_root(base_path / "lib")
        for path in (base_path / "lib").glob("python*/site-packages/tensorrt_libs"):
            add_root(path)
        for path in (base_path / "lib").glob("python*/site-packages/nvidia/*/lib"):
            add_root(path)

    add_root(Path(sys.prefix) / "lib")
    add_root(Path(getattr(sys, "base_prefix", sys.prefix)) / "lib")
    for env_name in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        if os.environ.get(env_name):
            add_root(Path(os.environ[env_name]) / "lib")

    if os.environ.get("CONDA_PREFIX"):
        conda_prefix = Path(os.environ["CONDA_PREFIX"])
        if conda_prefix.parent.name == "envs":
            add_conda_base_roots(conda_prefix.parent.parent)
    if os.environ.get("CONDA_EXE"):
        add_conda_base_roots(Path(os.environ["CONDA_EXE"]).parent.parent)
    prefix = Path(sys.prefix)
    if prefix.parent.name == "envs":
        add_conda_base_roots(prefix.parent.parent)

    for env_name in ("TENSORRT_HOME", "TRT_HOME", "CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(env_name):
            root = Path(os.environ[env_name])
            add_root(root / "lib")
            add_root(root / "lib64")
            add_root(root / "targets" / "x86_64-linux-gnu" / "lib")
            add_root(root / "targets" / "x86_64-linux" / "lib")

    for path in (
        "/usr/src/tensorrt/lib",
        "/usr/src/tensorrt/lib64",
        "/usr/src/tensorrt/targets/x86_64-linux-gnu/lib",
        "/usr/src/tensorrt/targets/x86_64-linux/lib",
        "/usr/local/tensorrt/lib",
        "/usr/local/tensorrt/lib64",
        "/usr/local/TensorRT/lib",
        "/usr/local/TensorRT/lib64",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/targets/x86_64-linux-gnu/lib",
        "/usr/lib/x86_64-linux-gnu",
    ):
        add_root(path)

    try:
        for path in site.getsitepackages():
            add_root(path)
    except Exception:
        pass

    try:
        add_root(site.getusersitepackages())
    except Exception:
        pass

    dirs: list[Path] = []
    for root in roots:
        for pattern in _SHARED_LIB_PATTERNS:
            for path in root.rglob(pattern):
                if path.is_file() and path.parent not in dirs:
                    dirs.append(path.parent)
    return dirs


def env_with_runtime_libs(plugins_dir: str | None) -> dict[str, str]:
    env = os.environ.copy()
    lib_dirs: list[Path] = []

    if plugins_dir:
        root = Path(plugins_dir)
        if root.is_dir():
            lib_dirs.append(root)
            lib_dirs.extend(
                sorted(
                    {p.parent for p in root.rglob("*.so") if p.is_file()},
                    key=lambda p: str(p),
                )
            )

    lib_dirs.extend(path for path in _python_shared_lib_dirs() if path not in lib_dirs)

    existing = env.get("LD_LIBRARY_PATH", "")
    parts = [str(path) for path in lib_dirs]
    if existing:
        parts.append(existing)
    if parts:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
    return env


def _probe_vapoursynth_namespace(cfg: RuntimeConfig, namespace: str) -> tuple[bool, str]:
    code = f"""
from vapoursynth import core
ns = {namespace!r}
if not hasattr(core, ns):
    raise SystemExit(f"missing core.{{ns}}")
plugin = getattr(core, ns)
try:
    print(plugin.Version())
except Exception:
    print("available")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            env=env_with_runtime_libs(cfg.plugins_dir),
            text=True,
            timeout=20,
        )
    except Exception as exc:
        return False, str(exc)

    detail = (result.stdout or result.stderr).strip()
    return result.returncode == 0, detail or f"exit {result.returncode}"


def trtexec_compatibility(cfg: RuntimeConfig) -> tuple[bool, str]:
    if not cfg.trtexec:
        return False, "not found; install matching Linux TensorRT CLI or set VSR_TRTEXEC"
    path = Path(cfg.trtexec)
    if not path.is_file():
        return False, f"{cfg.trtexec} does not exist"

    runtime_version = _probe_core_trt_version(cfg.plugins_dir)
    builder_version = _probe_trtexec_version(path)
    if not runtime_version:
        return True, f"{cfg.trtexec} (core.trt TensorRT version unknown)"
    if not builder_version:
        return True, f"{cfg.trtexec} (trtexec TensorRT version unknown)"
    if _same_trt_engine_abi(runtime_version, builder_version):
        return True, (
            f"{cfg.trtexec} "
            f"(trtexec TensorRT {_format_trt_version(builder_version)}, "
            f"core.trt {_format_trt_version(runtime_version)})"
        )
    return False, (
        f"{cfg.trtexec} uses TensorRT {_format_trt_version(builder_version)}, "
        f"but core.trt uses TensorRT {_format_trt_version(runtime_version)}. "
        "Use a matching trtexec or delete trtexec from config.toml."
    )


def resolve(cfg: RuntimeConfig, **overrides: object) -> RuntimeConfig:
    """Fill blanks via env vars then auto-detection, applying CLI overrides last.

    overrides: vspipe, ffmpeg, plugins_dir, models_dir, trtexec, pipeline_vpy, device_id,
    num_streams, encoder, fp16 — any non-None value wins.
    """
    # env
    cfg.vspipe = cfg.vspipe or os.environ.get("VSR_VSPIPE", "")
    cfg.ffmpeg = cfg.ffmpeg or os.environ.get("VSR_FFMPEG", "")
    cfg.plugins_dir = cfg.plugins_dir or os.environ.get("VSR_PLUGINS", "")
    cfg.models_dir = cfg.models_dir or os.environ.get("VSR_MODELS", "")
    cfg.trtexec = cfg.trtexec or os.environ.get("VSR_TRTEXEC", "")
    cfg.pipeline_vpy = cfg.pipeline_vpy or os.environ.get("VSR_PIPELINE", "")
    cfg.nvenc_fix = cfg.nvenc_fix or os.environ.get("VSR_NVENC_FIX", "")
    if not cfg.num_threads:
        env_threads = os.environ.get("VSR_NUM_THREADS")
        if env_threads:
            try:
                cfg.num_threads = int(env_threads)
            except ValueError:
                pass

    # auto-detect
    if not cfg.vspipe:
        cfg.vspipe = _autodetect_vspipe() or ""
    if not cfg.ffmpeg:
        cfg.ffmpeg = _autodetect_ffmpeg() or "ffmpeg"
    if not cfg.models_dir:
        cfg.models_dir = _autodetect_models(cfg.plugins_dir) or ""
    if not cfg.trtexec:
        cfg.trtexec = _autodetect_trtexec(cfg.plugins_dir) or ""
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

    lsmas_ok, lsmas_detail = _probe_vapoursynth_namespace(cfg, "lsmas")
    ffms2_ok, ffms2_detail = _probe_vapoursynth_namespace(cfg, "ffms2")
    source_detail = "lsmas: " + lsmas_detail if lsmas_ok else "ffms2: " + ffms2_detail
    if not (lsmas_ok or ffms2_ok):
        source_detail = f"lsmas: {lsmas_detail}; ffms2: {ffms2_detail}"
    checks.append(("source filter", lsmas_ok or ffms2_ok, source_detail))

    trt_ok, trt_detail = _probe_vapoursynth_namespace(cfg, "trt")
    checks.append(("core.trt", trt_ok, trt_detail))

    checks.append(("trtexec", *trtexec_compatibility(cfg)))

    nvsmi = _which("nvidia-smi")
    checks.append(("nvidia-smi", bool(nvsmi), nvsmi or "not found"))

    if cfg.nvenc_fix:
        fix_ok = Path(cfg.nvenc_fix).is_file()
        checks.append((
            "nvenc fix (LD_PRELOAD)",
            fix_ok,
            cfg.nvenc_fix if fix_ok else f"{cfg.nvenc_fix} (not found)",
        ))

    return checks
