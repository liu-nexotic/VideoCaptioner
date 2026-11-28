# coding:utf-8
"""FFmpeg 子进程的超时保护。

探测类命令（读元数据、枚举编码器、截帧）正常情况下秒级返回，但 QSV/CUDA
会话卡死时会让调用线程永久阻塞，任务永远停在"合成中"。完整合成的耗时与
视频长度成正比，用环境变量可配的大超时兜底。
"""

from __future__ import annotations

import os
import subprocess
import time

FFMPEG_PROBE_TIMEOUT = 30
FFMPEG_ENCODE_TIMEOUT_DEFAULT = 7200


def ffmpeg_encode_timeout() -> int:
    """完整合成命令的超时秒数，VIDEOCAPTIONER_FFMPEG_TIMEOUT 可覆盖。"""
    raw = os.environ.get("VIDEOCAPTIONER_FFMPEG_TIMEOUT", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return FFMPEG_ENCODE_TIMEOUT_DEFAULT


class FFmpegTimeout(RuntimeError):
    """ffmpeg 超出 allotted 时间被 kill。"""


def run_ffmpeg(args, *, probe: bool = False, **kwargs):
    """subprocess.run 带硬超时；超时 kill 进程并抛 FFmpegTimeout。"""
    timeout = FFMPEG_PROBE_TIMEOUT if probe else ffmpeg_encode_timeout()
    try:
        return subprocess.run(args, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired as e:
        raise FFmpegTimeout(
            f"ffmpeg timed out after {timeout}s: "
            f"{subprocess.list2cmdline(args)[:300]}"
        ) from e


def check_encode_deadline(process, cmd: list, started: float) -> None:
    """在编码进度读取循环里调用；超时则 kill 进程并抛 FFmpegTimeout。

    只能在读到输出行之后触发——ffmpeg 完全静默挂死时 readline 本身会阻塞，
    这种场景由探测阶段的短超时兜底。
    """
    limit = ffmpeg_encode_timeout()
    if time.monotonic() - started > limit:
        process.kill()
        try:
            process.wait(timeout=10)
        except Exception:
            pass
        raise FFmpegTimeout(
            f"ffmpeg timed out after {limit}s: {subprocess.list2cmdline(cmd)[:300]}"
        )
