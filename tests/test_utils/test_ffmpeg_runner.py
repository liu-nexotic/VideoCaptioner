import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from videocaptioner.core.subtitle.ass_renderer import _escape_filter_path
from videocaptioner.core.utils.ffmpeg_runner import (
    FFmpegTimeout,
    ffmpeg_encode_timeout,
    run_ffmpeg,
)


def test_escape_filter_path_quotes_and_colon():
    assert (
        _escape_filter_path(r"C:\Users\O'Brien\Temp\xxx.ass")
        == r"C\:/Users/O\'Brien/Temp/xxx.ass"
    )
    assert (
        _escape_filter_path(r"D:\视频\字幕.ass") == r"D\:/视频/字幕.ass"
    )


def test_probe_timeout_kills_ffmpeg():
    with pytest.raises(FFmpegTimeout):
        run_ffmpeg(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            probe=True,
            capture_output=True,
        )


def test_encode_timeout_env_override(monkeypatch):
    monkeypatch.setenv("VIDEOCAPTIONER_FFMPEG_TIMEOUT", "45")
    assert ffmpeg_encode_timeout() == 45
    monkeypatch.setenv("VIDEOCAPTIONER_FFMPEG_TIMEOUT", "not-a-number")
    assert ffmpeg_encode_timeout() == 7200
    monkeypatch.delenv("VIDEOCAPTIONER_FFMPEG_TIMEOUT")
    assert ffmpeg_encode_timeout() == 7200


def test_run_ffmpeg_normal_return():
    result = run_ffmpeg(
        [sys.executable, "-c", "print('ok')"],
        probe=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ok" in result.stdout
