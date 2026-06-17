"""Interactive TUI wizard (questionary + rich).

Walks the user through: input -> processing mode -> per-mode options ->
encoder -> output -> confirm, then runs the same Job machinery as `vsr run`/
`vsr batch`. The three modes map directly to the upscale/rife flags.
"""

from __future__ import annotations

from pathlib import Path

import questionary
from rich.console import Console

from . import presets
from .cli import VIDEO_EXTS
from .config import RuntimeConfig, resolve
from .pipeline import Job, run_job

console = Console()


def _select(message: str, choices: dict[str, str], default: str) -> str:
    """questionary.select over a {value: label} mapping; returns the value."""
    opts = [questionary.Choice(title=f"{k} — {v}", value=k) for k, v in choices.items()]
    return questionary.select(message, choices=opts, default=default).ask()


def run() -> int:
    console.print("[bold cyan]vsr 交互向导[/bold cyan]  (Ctrl+C 退出)")

    cfg = resolve(RuntimeConfig.from_file())

    # 1. input
    in_path = questionary.path("输入文件或文件夹:").ask()
    if not in_path:
        return 1
    in_p = Path(in_path)
    is_batch = in_p.is_dir()

    # 2. mode
    mode = questionary.select(
        "选择处理模式:",
        choices=[
            questionary.Choice("仅超分 (R-ESRGAN)", value="upscale"),
            questionary.Choice("仅插帧 (RIFE)", value="rife"),
            questionary.Choice("超分 + 插帧", value="both"),
        ],
    ).ask()
    if not mode:
        return 1
    do_upscale = mode in ("upscale", "both")
    do_rife = mode in ("rife", "both")

    model = pre_resize = None
    if do_upscale:
        model = _select("超分模型:", presets.REALESRGAN_MODELS, presets.DEFAULT_REALESRGAN_MODEL)
        pr = questionary.text("超分前缩放百分比 (留空=100):", default="").ask()
        pre_resize = float(pr) if pr.strip() else None

    rife_multi = "2"
    rife_model = None
    if do_rife:
        rife_multi = questionary.text("插帧倍率 (如 2 或 2/1):", default="2").ask() or "2"
        rife_model = _select("RIFE 模型:", presets.RIFE_MODELS, presets.DEFAULT_RIFE_MODEL)

    # encoder
    encoder = _select(
        "编码预设:",
        {k: v for k, v in zip(presets.ENCODER_PRESETS, presets.ENCODER_PRESETS.values())},
        presets.DEFAULT_ENCODER,
    )

    # output
    if is_batch:
        out_path = questionary.path("输出目录:", only_directories=True).ask()
    else:
        default_out = str(in_p.with_name(in_p.stem + "-vsr.mkv"))
        out_path = questionary.path("输出文件:", default=default_out).ask()
    if not out_path:
        return 1

    # confirm
    console.print(
        f"\n模式=[bold]{mode}[/bold]  模型={model}  插帧={rife_multi if do_rife else '-'}"
        f"  编码={encoder}\n输入={in_path}\n输出={out_path}"
    )
    if not questionary.confirm("开始处理?", default=True).ask():
        return 1

    def make_job(src: str, dst: str) -> Job:
        return Job(
            input_path=src, output_path=dst,
            upscale=do_upscale, model=model, pre_resize_factor=pre_resize,
            rife=do_rife, rife_multi=rife_multi, rife_model=rife_model,
            encoder=encoder,
        )

    if is_batch:
        files = sorted(p for p in in_p.rglob("*")
                       if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
        if not files:
            console.print("[yellow]目录中没有视频文件[/yellow]")
            return 1
        out_dir = Path(out_path)
        failed = 0
        for i, src in enumerate(files, 1):
            dst = out_dir / src.relative_to(in_p).with_suffix(".mkv")
            console.print(f"[bold]({i}/{len(files)})[/bold] {src.name}")
            if run_job(cfg, make_job(str(src), str(dst))) != 0:
                failed += 1
        return 1 if failed else 0

    return run_job(cfg, make_job(str(in_p), out_path))


if __name__ == "__main__":
    raise SystemExit(run())
