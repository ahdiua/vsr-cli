"""vsr-cli entry point.

No arguments  -> launch the interactive TUI wizard.
Subcommands   -> non-interactive operation:
    run            single file
    batch          a folder of files
    build-engines  pre-build TensorRT engines (warmup)
    setup          provision the Linux runtime (runs setup.sh)
    doctor         check the runtime is usable

The three processing modes (upscale-only / rife-only / upscale+rife) are
selected purely by the --upscale / --rife flags.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import presets
from .config import RuntimeConfig, config_path, repo_root, resolve, diagnose
from .pipeline import Job, build_engines, run_job

console = Console()

VIDEO_EXTS = {".mkv", ".mp4", ".mov", ".m2ts", ".ts", ".avi", ".webm", ".wmv", ".flv"}


# --- shared argument wiring -------------------------------------------------

def _add_processing_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--upscale", action="store_true", help="启用 R-ESRGAN 超分")
    p.add_argument("--model", help=f"超分模型 (默认 {presets.DEFAULT_REALESRGAN_MODEL})")
    p.add_argument("--pre-resize-factor", type=float, help="超分前缩放百分比 (如 50)")
    p.add_argument("--rife", action="store_true", help="启用 RIFE 插帧")
    p.add_argument("--rife-multi", default="2", help="插帧倍率, 如 2 或 2/1 (默认 2)")
    p.add_argument("--rife-model", help=f"RIFE 模型 (默认 {presets.DEFAULT_RIFE_MODEL})")
    p.add_argument("--final-resize-height", type=int, help="最终输出高度 (像素)")
    p.add_argument("--encoder", choices=list(presets.ENCODER_PRESETS),
                   help=f"编码预设 (默认 {presets.DEFAULT_ENCODER})")
    p.add_argument("--ffmpeg-args", help="自定义 ffmpeg 视频参数串, 覆盖 --encoder")


def _add_runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--vspipe", help="vspipe 可执行文件路径")
    p.add_argument("--ffmpeg", help="ffmpeg 可执行文件路径")
    p.add_argument("--plugins-dir", help="VapourSynth plugins 目录")
    p.add_argument("--models-dir", help="模型目录 (含 RealESRGANv2/ rife/)")
    p.add_argument("--pipeline-vpy", help="pipeline.vpy 路径")
    p.add_argument("--device-id", type=int, help="GPU 设备号")
    p.add_argument("--num-streams", type=int, help="TensorRT 并行流数")
    p.add_argument("--no-fp16", dest="fp16", action="store_false", default=None,
                   help="禁用 fp16")


def _load_cfg(args: argparse.Namespace) -> RuntimeConfig:
    cfg = RuntimeConfig.from_file()
    overrides = dict(
        vspipe=getattr(args, "vspipe", None),
        ffmpeg=getattr(args, "ffmpeg", None),
        plugins_dir=getattr(args, "plugins_dir", None),
        models_dir=getattr(args, "models_dir", None),
        pipeline_vpy=getattr(args, "pipeline_vpy", None),
        device_id=getattr(args, "device_id", None),
        num_streams=getattr(args, "num_streams", None),
        fp16=getattr(args, "fp16", None),
        encoder=getattr(args, "encoder", None),
    )
    return resolve(cfg, **overrides)


def _job_from_args(args: argparse.Namespace, input_path: str, output_path: str) -> Job:
    return Job(
        input_path=input_path,
        output_path=output_path,
        upscale=args.upscale,
        model=args.model or (presets.DEFAULT_REALESRGAN_MODEL if args.upscale else None),
        pre_resize_factor=args.pre_resize_factor,
        rife=args.rife,
        rife_multi=args.rife_multi,
        rife_model=args.rife_model,
        final_resize_height=args.final_resize_height,
        encoder=args.encoder,
        ffmpeg_args=args.ffmpeg_args,
    )


# --- subcommand handlers ----------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    job = _job_from_args(args, args.input, args.output)
    return run_job(cfg, job)


def cmd_batch(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    if not in_dir.is_dir():
        console.print(f"[red]输入目录不存在: {in_dir}[/red]")
        return 2

    globber = in_dir.rglob if args.recursive else in_dir.glob
    pattern = args.pattern or "*"
    files = sorted(
        p for p in globber(pattern)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if not files:
        console.print(f"[yellow]没有匹配的视频文件: {in_dir} ({pattern})[/yellow]")
        return 1

    console.print(f"[cyan]批量处理 {len(files)} 个文件[/cyan]")
    failed = 0
    for i, src in enumerate(files, 1):
        rel = src.relative_to(in_dir)
        dst = out_dir / rel.with_suffix(args.out_ext)
        console.print(f"[bold]({i}/{len(files)})[/bold] {rel}")
        job = _job_from_args(args, str(src), str(dst))
        if run_job(cfg, job) != 0:
            failed += 1
    if failed:
        console.print(f"[red]{failed} 个文件失败[/red]")
        return 1
    return 0


def cmd_build_engines(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    job = _job_from_args(args, args.input, "-")
    return build_engines(cfg, job)


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    table = Table(title="vsr doctor")
    table.add_column("检查项")
    table.add_column("状态")
    table.add_column("详情")
    all_ok = True
    for label, ok, detail in diagnose(cfg):
        all_ok = all_ok and ok
        mark = "[green]OK[/green]" if ok else "[red]缺失[/red]"
        table.add_row(label, mark, detail)
    console.print(table)
    console.print(f"配置文件: {config_path()}")
    if not all_ok:
        console.print("[yellow]运行时不完整，请先执行 `vsr setup` 或检查 config.toml[/yellow]")
    return 0 if all_ok else 1


def cmd_setup(args: argparse.Namespace) -> int:
    script = repo_root() / "setup.sh"
    if not script.is_file():
        console.print(f"[red]找不到 setup.sh: {script}[/red]")
        return 2
    console.print(f"[cyan]运行 {script}[/cyan]")
    try:
        return subprocess.call(["bash", str(script), *args.setup_args])
    except FileNotFoundError:
        console.print("[red]未找到 bash。请在 Linux/AutoDL 上运行 setup.sh[/red]")
        return 2


# --- parser -----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vsr",
        description="视频超分(R-ESRGAN)/插帧(RIFE) CLI — vs-mlrt + TensorRT -> ffmpeg",
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="处理单个文件")
    p_run.add_argument("-i", "--input", required=True, help="输入视频")
    p_run.add_argument("-o", "--output", required=True, help="输出视频")
    _add_processing_args(p_run)
    _add_runtime_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_batch = sub.add_parser("batch", help="批量处理文件夹")
    p_batch.add_argument("-i", "--input", required=True, help="输入目录")
    p_batch.add_argument("-o", "--output", required=True, help="输出目录")
    p_batch.add_argument("--recursive", action="store_true", help="递归子目录")
    p_batch.add_argument("--pattern", help="文件名 glob (默认 *)")
    p_batch.add_argument("--out-ext", default=".mkv", help="输出容器扩展名 (默认 .mkv)")
    _add_processing_args(p_batch)
    _add_runtime_args(p_batch)
    p_batch.set_defaults(func=cmd_batch)

    p_eng = sub.add_parser("build-engines", help="预构建 TensorRT engine")
    p_eng.add_argument("-i", "--input", required=True, help="样本视频")
    _add_processing_args(p_eng)
    _add_runtime_args(p_eng)
    p_eng.set_defaults(func=cmd_build_engines)

    p_doc = sub.add_parser("doctor", help="检查运行时")
    _add_runtime_args(p_doc)
    p_doc.set_defaults(func=cmd_doctor)

    p_setup = sub.add_parser("setup", help="搭建 Linux 运行时 (运行 setup.sh)")
    p_setup.add_argument("setup_args", nargs=argparse.REMAINDER, help="传给 setup.sh 的参数")
    p_setup.set_defaults(func=cmd_setup)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        # interactive mode
        from .wizard import run as wizard_run
        return wizard_run()

    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
