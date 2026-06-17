"""Build and run the ``vspipe | ffmpeg`` pipeline.

The two processes are wired directly (vspipe.stdout -> ffmpeg.stdin) via
subprocess.Popen, without a shell, so no quoting/escaping pitfalls and it works
identically on Linux and Windows. The ffmpeg track mapping (copy audio/subs/
attachments from the original file, re-encode only video) mirrors VideoJaNai's
RunUpscaleSingle.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from . import presets
from .config import RuntimeConfig, env_with_runtime_libs

console = Console()


@dataclass
class Job:
    """A single input->output processing request."""

    input_path: str
    output_path: str
    upscale: bool = False
    model: str | None = None
    pre_resize_factor: float | None = None
    rife: bool = False
    rife_multi: str = "2"
    rife_model: str | None = None
    final_resize_height: int | None = None
    encoder: str | None = None
    ffmpeg_args: str | None = None
    # engine-warmup only: process frames [start, end)
    start: int | None = None
    end: int | None = None
    extra_args: dict = field(default_factory=dict)

    def validate(self) -> None:
        if not (self.upscale or self.rife):
            raise ValueError(
                "至少需要开启 --upscale 或 --rife 其一（仅超分 / 仅插帧 / 超分+插帧）"
            )
        if self.upscale and not self.model:
            raise ValueError("开启超分时必须指定 --model")
        if not Path(self.input_path).is_file():
            raise FileNotFoundError(self.input_path)


def _vpy_args(cfg: RuntimeConfig, job: Job) -> list[str]:
    """The --arg pairs forwarded into pipeline.vpy."""
    args: dict[str, object] = {
        "video_path": job.input_path,
        "upscale": int(job.upscale),
        "rife": int(job.rife),
        "device_id": cfg.device_id,
        "num_streams": cfg.num_streams,
        "fp16": int(cfg.fp16),
    }
    if cfg.plugins_dir:
        args["plugins_dir"] = cfg.plugins_dir
    if cfg.models_dir:
        args["models_dir"] = cfg.models_dir
    if cfg.trtexec:
        args["trtexec_path"] = cfg.trtexec
    if job.upscale:
        args["model"] = job.model
        if job.pre_resize_factor is not None:
            args["pre_resize_factor"] = job.pre_resize_factor
    if job.rife:
        args["rife_multi"] = job.rife_multi
        args["rife_model"] = job.rife_model or presets.DEFAULT_RIFE_MODEL
    if job.final_resize_height is not None:
        args["final_resize_height"] = job.final_resize_height
    args.update(job.extra_args)

    out: list[str] = []
    for key, val in args.items():
        out += ["--arg", f"{key}={val}"]
    return out


def build_vspipe_cmd(cfg: RuntimeConfig, job: Job) -> list[str]:
    cmd = [cfg.vspipe, "-c", "y4m"]
    if job.start is not None:
        cmd += ["--start", str(job.start)]
    if job.end is not None:
        cmd += ["--end", str(job.end)]
    cmd += _vpy_args(cfg, job)
    cmd += [cfg.pipeline_vpy, "-"]
    return cmd


def build_ffmpeg_cmd(cfg: RuntimeConfig, job: Job) -> list[str]:
    codec = presets.build_video_codec_args(job.encoder or cfg.encoder, job.ffmpeg_args)
    return [
        cfg.ffmpeg,
        "-y",
        "-i", "pipe:",
        "-i", job.input_path,
        "-map", "0:v",
        *codec,
        "-map", "1:t?", "-map", "1:a?", "-map", "1:s?",
        "-c:t", "copy", "-c:a", "copy", "-c:s", "copy",
        job.output_path,
    ]


def build_vspipe_env(cfg: RuntimeConfig) -> dict[str, str]:
    """Environment for vspipe with bundled vs-mlrt shared libs discoverable."""
    return env_with_runtime_libs(cfg.plugins_dir)


def run_job(cfg: RuntimeConfig, job: Job, quiet: bool = False) -> int:
    """Run vspipe | ffmpeg for one job. Returns 0 on success."""
    job.validate()
    Path(job.output_path).parent.mkdir(parents=True, exist_ok=True)

    vspipe_cmd = build_vspipe_cmd(cfg, job)
    ffmpeg_cmd = build_ffmpeg_cmd(cfg, job)
    vspipe_env = build_vspipe_env(cfg)

    if not quiet:
        console.print(f"[dim]vspipe:[/dim] {' '.join(vspipe_cmd)}")
        console.print(f"[dim]ffmpeg:[/dim] {' '.join(ffmpeg_cmd)}")

    vspipe = subprocess.Popen(vspipe_cmd, stdout=subprocess.PIPE, env=vspipe_env)
    assert vspipe.stdout is not None
    ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=vspipe.stdout)
    # allow vspipe to receive SIGPIPE if ffmpeg exits
    vspipe.stdout.close()

    ffmpeg_ret = ffmpeg.wait()
    vspipe_ret = vspipe.wait()

    if vspipe_ret != 0:
        console.print(f"[red]vspipe failed (exit {vspipe_ret})[/red]")
        return vspipe_ret
    if ffmpeg_ret != 0:
        console.print(f"[red]ffmpeg failed (exit {ffmpeg_ret})[/red]")
        return ffmpeg_ret
    if not quiet:
        console.print(f"[green]done:[/green] {job.output_path}")
    return 0


def build_engines(cfg: RuntimeConfig, job: Job) -> int:
    """Warm up: run a single frame to trigger TensorRT engine construction.

    Output is discarded (ffmpeg writes to the null muxer).
    """
    warm = Job(**{**job.__dict__})
    warm.start = 0
    warm.end = 1
    warm.output_path = "-"  # placeholder; replaced below with null sink

    job.validate()
    vspipe_cmd = build_vspipe_cmd(cfg, warm)
    vspipe_env = build_vspipe_env(cfg)
    # discard pipe output entirely — we only want the engine .engine files built
    ffmpeg_cmd = [cfg.ffmpeg, "-y", "-i", "pipe:", "-f", "null", "-"]

    console.print("[cyan]Building TensorRT engines (single-frame warmup)…[/cyan]")
    console.print(f"[dim]vspipe:[/dim] {' '.join(vspipe_cmd)}")

    vspipe = subprocess.Popen(vspipe_cmd, stdout=subprocess.PIPE, env=vspipe_env)
    assert vspipe.stdout is not None
    ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=vspipe.stdout)
    vspipe.stdout.close()
    ffmpeg_ret = ffmpeg.wait()
    vspipe_ret = vspipe.wait()
    ret = vspipe_ret or ffmpeg_ret
    if ret == 0:
        console.print("[green]engines ready[/green]")
    else:
        console.print(f"[red]engine build failed (vspipe={vspipe_ret}, ffmpeg={ffmpeg_ret})[/red]")
    return ret
