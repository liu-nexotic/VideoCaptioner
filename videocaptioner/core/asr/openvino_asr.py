# coding:utf-8
"""Intel Arc / iGPU / NPU speech recognition via OpenVINO GenAI.

CTranslate2 (used by faster-whisper) only supports NVIDIA CUDA and cannot run
on Intel GPUs or NPUs. This backend uses Intel's official OpenVINO GenAI Whisper
pipeline, supporting:
- Intel Arc A-series (Alchemist) & B-series (Battlemage) discrete GPUs
- Intel Core Ultra integrated Arc GPUs (e.g. Core Ultra 7 155H 8-Xe iGPU)
- Intel AI Boost NPUs (e.g. Core Ultra 7 155H Meteor Lake NPU 3720, Lunar Lake,
  Arrow Lake, Panther Lake)

NPU-specific engineering workarounds included here:
1. Weight format: NPU automatically uses INT8 IR models (``whisper-*-int8-ov``),
   halving memory footprint and matching the NPU's native INT8 MAC array.
2. Timestamp bug (#4806): ``WhisperPipeline.generate(..., return_timestamps=True)``
   on NPU returns empty/corrupted text in OpenVINO 2026.x. On NPU we segment
   16 kHz audio at natural speech pauses (< 14s, well within NPU's 30s static
   encoder window) and run ``return_timestamps=False`` per utterance.
3. State-reset bug (openvino#37937): in OpenVINO <= 2026.4, ``reset_state()``
   on NPU does not sync ``num_stored_tokens`` for 1-token language-detection
   prompts on call #2+. When language is auto-detect, we detect on utterance #1
   (when ``num_stored_tokens == 0``) and pin that language token for #2..N so
   every call runs a full multi-token prefill.
4. Level0 OOM retry & teardown (#1460): retries with
   ``DISABLE_OPENVINO_GENAI_NPU_L0=1`` on NPU init errors, falls back to Intel
   GPU if needed, and explicitly destroys the pipeline before worker exit.

The pipeline itself is executed in a separate process (see
``openvino_worker.py`` / ``openvino_process.py``) because loading OpenVINO and
PyQt5 in the same process crashes with 0xc0000005 (conflicting MSVC runtimes).
"""

from __future__ import annotations

import gc
import os
import re
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np

from ...config import MODEL_PATH
from ..entities import SubtitleLayoutEnum
from ..utils.hardware import npu_fallback_device, resolve_openvino_device
from ..utils.logger import setup_logger
from ..utils.model_downloader import (
    OPENVINO_WHISPER_REPOS,
    ensure_openvino_whisper_model,
    openvino_model_files_complete,
    openvino_whisper_model_dir,
)
from .asr_data import ASRData, ASRDataSeg
from .base import BaseASR

logger = setup_logger("openvino_asr")

# Whisper internally works on 30s windows; on GPU/CPU we aggregate 4 minutes per
# generate call to keep progress reporting responsive without excessive memory.
CHUNK_SECONDS = 240
# The Meteor Lake NPU wedges (ZE_RESULT_ERROR_DEVICE_LOST) on a first 240s call
# with large models; 60s matches the previously stable short-clip runs. Lunar
# Lake (NPU 4000) ran 240s chunks stably across long sessions, so it gets the
# full window back; unknown generations stay conservative. Override per run
# with VIDEOCAPTIONER_NPU_CHUNK_SECONDS.
NPU_CHUNK_SECONDS = 60
NPU_FAST_CHUNK_GENERATIONS = ("lunar_lake",)


def chunk_seconds_for_device(device: str) -> int:
    """Shorter generate calls on the NPU to avoid DEVICE_LOST on long audio.

    The safe 60s window applies to Meteor Lake/Arrow Lake (NPU 3720) and to
    any generation we have no stability data for; newer NPUs with proven 240s
    stability run at full length.
    """
    if not _is_npu_device(device):
        return CHUNK_SECONDS
    override = os.environ.get("VIDEOCAPTIONER_NPU_CHUNK_SECONDS", "").strip()
    if override.isdigit() and int(override) > 0:
        return int(override)
    from ..utils.hardware import npu_generation

    if npu_generation() in NPU_FAST_CHUNK_GENERATIONS:
        return CHUNK_SECONDS
    return NPU_CHUNK_SECONDS

# On NPU, encoder input is statically reshaped to a single 30s window and
# return_timestamps=False is required (#4806), so utterances are capped at 14s.
NPU_MAX_SEGMENT_SEC = 14.0
NPU_MIN_SILENCE_SEC = 0.30


def _is_npu_device(device: str) -> bool:
    return (device or "").strip().upper().startswith("NPU")


def _openvino_device_full_name(device: str) -> str:
    """Resolve OpenVINO's FULL_DEVICE_NAME for ``device`` (best-effort)."""
    try:
        from ..utils.hardware import list_openvino_devices

        mapping = dict(list_openvino_devices())
        if device in mapping:
            return mapping[device]
    except Exception:
        pass
    try:
        import openvino as ov

        return str(ov.Core().get_property(device, "FULL_DEVICE_NAME"))
    except Exception:
        return ""


def _format_runtime_device(
    device: str,
    full_name: str = "",
    precision: str = "",
    precision_hint: str = "",
) -> str:
    """Format a human-readable runtime device label for logs and UI."""
    base = device or "AUTO"
    name = (full_name or "").strip()
    if name and name.lower() != base.lower():
        label = f"{base} · {name}"
    else:
        label = base

    tags: List[str] = []
    if precision:
        tags.append(precision)
    if precision_hint and precision_hint != precision:
        tags.append(precision_hint)
    if tags:
        label = f"{label} ({', '.join(tags)})"
    return label


def _gpu_precision_hint(device: str, explicit: str = "") -> str:
    """Choose an ``INFERENCE_PRECISION_HINT`` for the target device.

    Intel's IGC compiler aborts the process on some integrated GPUs (notably
    engineering-sample Meteor Lake units) while compiling fp16 kernels for the
    Whisper encoder; f32 kernels work. Discrete Arc GPUs are fine with fp16, so
    only integrated GPUs default to f32.

    Override with ``VIDEOCAPTIONER_OPENVINO_GPU_PRECISION=f32|fp16``.
    """
    hint = (explicit or "").strip().lower()
    if not (device or "").upper().startswith("GPU"):
        return ""
    if hint not in ("f32", "fp16"):
        hint = os.environ.get(
            "VIDEOCAPTIONER_OPENVINO_GPU_PRECISION", ""
        ).strip().lower()
    if hint in ("f32", "fp16"):
        return hint
    try:
        from ..utils.hardware import intel_gpu_devices

        name = dict(intel_gpu_devices()).get(device, "")
    except Exception:
        name = ""
    return "f32" if "(iGPU)" in name else ""


def _is_npu_failure(error: Exception) -> bool:
    """Whether an exception looks like an NPU/Level Zero failure."""
    message = str(error).lower()
    return any(
        key in message
        for key in (
            "device_lost",
            "device hung",
            "ze_result",
            "zefence",
            "npu",
            "npu driver",
        )
    )


def _read_wav_mono_16k(path: str) -> "np.ndarray":
    """Read a 16 kHz mono WAV file as float32 in [-1, 1]."""
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width * 8} bit")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if sample_rate != 16000:
        raise RuntimeError(f"Audio must be 16 kHz, got {sample_rate} Hz")
    return audio


def _clean_text(text: str) -> str:
    text = re.sub(r"<\|[^|]*\|>", "", text or "")
    return text.strip()


def segment_speech_for_npu(
    audio: "np.ndarray",
    sample_rate: int = 16000,
    max_segment_sec: float = NPU_MAX_SEGMENT_SEC,
    min_silence_sec: float = NPU_MIN_SILENCE_SEC,
) -> List[Tuple[int, int]]:
    """Split 16 kHz audio into pause-aligned speech regions ``[(start, end)]``.

    Every returned span is strictly shorter than Whisper's 30s NPU static
    encoder window (capped around ``max_segment_sec``), non-overlapping, and
    aligned to natural speech pauses so accurate SRT timestamps are preserved
    even with ``return_timestamps=False`` on NPU.
    """
    import numpy as np

    total = int(len(audio))
    if total == 0:
        return []

    frame_ms = 20
    frame_len = sample_rate * frame_ms // 1000  # 320 samples @ 16 kHz
    n_frames = (total + frame_len - 1) // frame_len
    padded = np.zeros(n_frames * frame_len, dtype=np.float32)
    padded[:total] = audio

    frames = padded.reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    if n_frames >= 3:
        kernel = np.ones(3, dtype=np.float32) / 3.0
        smooth = np.convolve(rms, kernel, mode="same")
    else:
        smooth = rms

    p20 = float(np.percentile(smooth, 20))
    p95 = float(np.percentile(smooth, 95))
    if p95 < 1e-4:
        return []

    threshold = max(0.006, min(0.035, p20 * 2.5 + (p95 - p20) * 0.08))
    active = smooth >= threshold

    # Bridge short pauses inside a phrase (< min_silence_sec).
    min_silence_frames = max(1, int(round(min_silence_sec * 1000 / frame_ms)))
    idx = 0
    while idx < n_frames:
        if active[idx]:
            idx += 1
            continue
        gap_start = idx
        while idx < n_frames and not active[idx]:
            idx += 1
        if gap_start > 0 and idx < n_frames and (idx - gap_start) < min_silence_frames:
            active[gap_start:idx] = True

    # Collect contiguous speech runs and drop tiny noise blips (< 200ms).
    min_speech_frames = max(1, int(round(200 / frame_ms)))
    raw_runs: List[Tuple[int, int]] = []
    idx = 0
    while idx < n_frames:
        if not active[idx]:
            idx += 1
            continue
        run_start = idx
        while idx < n_frames and active[idx]:
            idx += 1
        if (idx - run_start) >= min_speech_frames:
            raw_runs.append((run_start, idx))

    max_frames = max(20, int(round(max_segment_sec * 1000 / frame_ms)))
    if not raw_runs:
        # Constant-level audio fallback: fixed windows <= max_segment_sec.
        step = max_frames * frame_len
        return [(s, min(total, s + step)) for s in range(0, total, step)]

    # Split long continuous monologues at the quietest local dip in [50%, 100%] of max_frames.
    bounded_runs: List[Tuple[int, int]] = []
    half_frames = max(10, max_frames // 2)
    for run_start, run_end in raw_runs:
        cur = run_start
        while (run_end - cur) > max_frames:
            win_lo = cur + half_frames
            win_hi = cur + max_frames
            split_at = win_lo + int(np.argmin(smooth[win_lo:win_hi]))
            if split_at <= cur:
                split_at = cur + max_frames
            bounded_runs.append((cur, split_at))
            cur = split_at
        if cur < run_end:
            bounded_runs.append((cur, run_end))

    # Merge very short adjacent fragments (< 1.2s) if separated by < 0.5s and combined <= 10s.
    short_frames = int(round(1200 / frame_ms))
    merge_gap_frames = int(round(500 / frame_ms))
    merge_cap_frames = int(round(10000 / frame_ms))
    merged_runs: List[Tuple[int, int]] = []
    for run_start, run_end in bounded_runs:
        if merged_runs:
            prev_start, prev_end = merged_runs[-1]
            if (
                (prev_end - prev_start) < short_frames
                and (run_start - prev_end) <= merge_gap_frames
                and (run_end - prev_start) <= merge_cap_frames
            ):
                merged_runs[-1] = (prev_start, run_end)
                continue
        merged_runs.append((run_start, run_end))

    # Convert frames to samples with gentle pre/post padding, never overlapping neighbors.
    pad_pre = int(0.12 * sample_rate)
    pad_post = int(0.16 * sample_rate)
    raw_samples = [(f0 * frame_len, min(total, f1 * frame_len)) for f0, f1 in merged_runs]
    spans: List[Tuple[int, int]] = []
    for i, (s0, s1) in enumerate(raw_samples):
        lo_bound = 0 if i == 0 else (raw_samples[i - 1][1] + s0) // 2
        hi_bound = total if i == len(raw_samples) - 1 else (s1 + raw_samples[i + 1][0]) // 2
        padded_s0 = max(lo_bound, s0 - pad_pre)
        padded_s1 = min(hi_bound, s1 + pad_post)
        if padded_s1 > padded_s0:
            spans.append((padded_s0, padded_s1))
    return spans


def split_clauses_proportionally(
    text: str,
    start_ms: int,
    end_ms: int,
    max_clause_ms: int = 6000,
) -> List[ASRDataSeg]:
    """Split a longer multi-sentence NPU utterance proportionally across clauses."""
    cleaned = _clean_text(text)
    if not cleaned:
        return []

    duration = max(200, end_ms - start_ms)
    if duration <= max_clause_ms:
        return [ASRDataSeg(cleaned, start_ms, start_ms + duration)]

    parts = [p.strip() for p in re.split(r"(?<=[。！？；.!?;])\s*", cleaned) if p.strip()]
    if len(parts) <= 1 and duration > 9000:
        comma_parts = [p.strip() for p in re.split(r"(?<=[，,、])\s*", cleaned) if p.strip()]
        if len(comma_parts) > 1:
            parts = []
            buf = ""
            for piece in comma_parts:
                buf = f"{buf} {piece}".strip() if buf else piece
                if len(buf) >= 18:
                    parts.append(buf)
                    buf = ""
            if buf:
                if parts:
                    parts[-1] = f"{parts[-1]} {buf}".strip()
                else:
                    parts.append(buf)

    if len(parts) <= 1:
        return [ASRDataSeg(cleaned, start_ms, start_ms + duration)]

    weights = [max(1, len(p)) for p in parts]
    total_weight = sum(weights)
    segments: List[ASRDataSeg] = []
    cursor = start_ms
    for idx, (part, weight) in enumerate(zip(parts, weights)):
        if idx == len(parts) - 1:
            seg_end = start_ms + duration
        else:
            seg_end = cursor + max(200, int(round(duration * weight / total_weight)))
            seg_end = min(seg_end, start_ms + duration - 100 * (len(parts) - idx - 1))
        if seg_end <= cursor:
            seg_end = cursor + 200
        segments.append(ASRDataSeg(part, cursor, seg_end))
        cursor = seg_end
    return segments


class OpenVINOASR(BaseASR):
    """OpenVINO GenAI Whisper implementation (runs in the worker process)."""

    def __init__(
        self,
        audio_input,
        whisper_model: str = "large-v3-turbo",
        model_dir: Optional[str] = None,
        language: str = "en",
        device: str = "intel_gpu",
        need_word_time_stamp: bool = False,
        precision: str = "",
        precision_hint: str = "",
        gpu_name: str = "",
    ):
        super().__init__(audio_input, need_word_time_stamp)
        if whisper_model not in OPENVINO_WHISPER_REPOS:
            raise RuntimeError(f"Unsupported OpenVINO model: {whisper_model}")
        self.model_name = whisper_model
        self.model_dir = Path(model_dir) if model_dir else Path(MODEL_PATH)
        self.language = language
        self.device = device
        self.precision_override = precision or None
        self.precision_hint = precision_hint or ""
        self.gpu_name = gpu_name or ""
        self.need_word_time_stamp = need_word_time_stamp
        self.pipeline = None
        self.used_device = ""
        self.used_device_name = ""
        self.used_precision = "fp16"
        self.used_precision_hint = ""
        self.used_device_label = ""
        self._word_ts_supported = True
        self._npu_detected_language = ""
        self._callback: Optional[Callable[[int, str], None]] = None

    # ------------------------------------------------------------------
    def _build_config(self) -> dict:
        config = {"task": "transcribe", "return_timestamps": True}
        if self.language:
            config["language"] = f"<|{self.language}|>"
        if self.need_word_time_stamp:
            config["word_timestamps"] = True
        if _is_npu_device(self.used_device):
            # Without an explicit cap the NPU decoder keeps running past the
            # 30s window and trips the driver watchdog (DEVICE_LOST). 448 is
            # Whisper's per-window maximum and is enough for 240s chunks.
            config["max_new_tokens"] = 448
        return config

    @staticmethod
    def _construct_pipeline(ov_genai, model_path: Path, device: str, properties: dict):
        try:
            return ov_genai.WhisperPipeline(str(model_path), device, **properties)
        except TypeError:
            return ov_genai.WhisperPipeline(str(model_path), device)

    def _rebuild_pipeline(self):
        """Rebuild the pipeline after a failed generate() call.

        A word_timestamps attempt that dies inside the decoder leaves the
        pipeline's internal stream state dirty; retrying on the same instance
        produced corrupted output for the first window (missing/shifted
        segments, timestamps starting at 30s). The compiled-kernel cache makes
        the rebuild fast.
        """
        self.pipeline = None
        gc.collect()
        self._create_pipeline(
            device_override=self.used_device or None,
            precision_override=self.used_precision or None,
        )

    def _record_runtime_device(
        self, device: str, precision: str, precision_hint: str
    ) -> None:
        self.used_device = device
        self.used_device_name = _openvino_device_full_name(device)
        self.used_precision = precision
        self.used_precision_hint = precision_hint
        self.used_device_label = _format_runtime_device(
            device, self.used_device_name, precision, precision_hint
        )
        logger.info("Active OpenVINO runtime device: %s", self.used_device_label)

    def _create_pipeline(self, device_override=None, precision_override=None):
        import openvino_genai as ov_genai

        if device_override:
            device = str(device_override)
            if not device.upper().startswith(("GPU", "CPU", "NPU", "AUTO")):
                device = resolve_openvino_device(device)
        else:
            device = resolve_openvino_device(self.device, self.gpu_name)
        requested_npu = (self.device or "").lower().startswith(
            ("intel_npu", "npu")
        ) and not device_override
        if requested_npu and not _is_npu_device(device):
            logger.warning(
                "Intel NPU was requested (%s) but is unavailable in OpenVINO; "
                "using %s instead (install Intel NPU driver >= 32.0.100.3104 for NPU)",
                self.device,
                device,
            )

        effective_override = (precision_override or self.precision_override or "").strip().lower()
        if _is_npu_device(device) and effective_override and effective_override != "int8":
            logger.warning(
                "Intel NPU only supports int8 Whisper models; overriding %s -> int8",
                effective_override,
            )
            effective_override = "int8"
        precision = effective_override or (
            "int8"
            if (_is_npu_device(device) or device.upper().startswith("CPU"))
            else "fp16"
        )
        if precision == "fp16" and not effective_override:
            # Reuse an already-downloaded int8 model on GPU instead of
            # pulling a separate 1.5GB fp16 copy when fp16 is not installed.
            int8_path = openvino_whisper_model_dir(
                self.model_dir, self.model_name, precision="int8"
            )
            fp16_path = openvino_whisper_model_dir(
                self.model_dir, self.model_name, precision="fp16"
            )
            if openvino_model_files_complete(
                int8_path
            ) and not openvino_model_files_complete(fp16_path):
                logger.info(
                    "Reusing existing int8 model on %s (fp16 model not installed)",
                    device,
                )
                precision = "int8"
        model_path = ensure_openvino_whisper_model(
            self.model_name,
            self.model_dir,
            progress=self._download_progress,
            precision=precision,
        )
        # Persist compiled kernels/blobs so later runs start quickly.
        cache_dir = Path(self.model_dir) / "openvino" / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        properties = {"CACHE_DIR": str(cache_dir)}
        precision_hint = _gpu_precision_hint(device, self.precision_hint)
        if precision_hint:
            properties["INFERENCE_PRECISION_HINT"] = precision_hint

        logger.info(
            "Loading OpenVINO model: %s (device: %s, precision: %s%s)",
            model_path,
            device,
            precision,
            f", hint: {precision_hint}" if precision_hint else "",
        )
        if getattr(self, "_callback", None):
            self._callback(
                2,
                f"正在编译/加载模型: {device} ({precision})，首次运行可能需要几分钟",
            )
        if _is_npu_device(device) and self.model_name in (
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
            "large-v3-turbo",
        ):
            logger.warning(
                "Model %s is heavy for Intel NPU (Meteor Lake); "
                "small/base are recommended, GPU is preferred for large models",
                self.model_name,
            )
        try:
            pipeline = self._construct_pipeline(ov_genai, model_path, device, properties)
        except Exception as first_err:
            if _is_npu_device(device):
                # Official OpenVINO NPU troubleshooting workaround for Level0 OOM
                # on Meteor Lake / Core Ultra 155H drivers:
                if os.environ.get("DISABLE_OPENVINO_GENAI_NPU_L0") != "1":
                    logger.warning(
                        "NPU pipeline init failed (%s); retrying with "
                        "DISABLE_OPENVINO_GENAI_NPU_L0=1",
                        first_err,
                    )
                    os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
                    try:
                        pipeline = self._construct_pipeline(
                            ov_genai, model_path, device, properties
                        )
                    except Exception as second_err:
                        first_err = second_err
                    else:
                        self.pipeline = pipeline
                        self._record_runtime_device(device, precision, precision_hint)
                        return pipeline

                fallback = resolve_openvino_device("intel_gpu", self.gpu_name)
                logger.warning(
                    "NPU failed (%s), falling back to %s", first_err, fallback
                )
                device = fallback
                precision_hint = _gpu_precision_hint(device, self.precision_hint)
                if precision_hint:
                    properties["INFERENCE_PRECISION_HINT"] = precision_hint
                else:
                    properties.pop("INFERENCE_PRECISION_HINT", None)
                pipeline = self._construct_pipeline(
                    ov_genai, model_path, device, properties
                )
            else:
                raise

        self.pipeline = pipeline
        self._record_runtime_device(device, precision, precision_hint)
        return pipeline

    def _download_progress(self, percent: int, message: str):
        callback = getattr(self, "_callback", None)
        if callback:
            callback(2 + int(percent * 0.16), message)

    # ------------------------------------------------------------------
    def _transcribe_npu(
        self, audio: "np.ndarray", callback: Callable[[int, str], None]
    ) -> List[ASRDataSeg]:
        """Fallback NPU path: pause segmentation per utterance.

        Only used when the native-timestamp path returns no segments. Each
        utterance costs a full 30s encoder pass, so this is ~3x slower than
        the timestamped path.
        """
        spans = segment_speech_for_npu(audio, sample_rate=16000)
        if not spans:
            return []

        total_spans = len(spans)
        logger.info(
            "NPU speech segmentation produced %d utterance(s) across %.1fs audio",
            total_spans,
            len(audio) / 16000.0,
        )
        all_segments: List[ASRDataSeg] = []

        for idx, (s0, s1) in enumerate(spans):
            clip = audio[s0:s1]
            start_ms = int(round(s0 * 1000.0 / 16000.0))
            end_ms = int(round(s1 * 1000.0 / 16000.0))

            gen_kwargs = {
                "task": "transcribe",
                # Workaround for openvino.genai#4806: return_timestamps=True
                # corrupts/empties decoded text on NPU.
                "return_timestamps": False,
                "max_new_tokens": 448,
            }
            if self.language:
                gen_kwargs["language"] = f"<|{self.language}|>"
            elif self._npu_detected_language:
                # Workaround for openvino#37937: pin language detected on the
                # first utterance (when num_stored_tokens == 0) so call #2+
                # always runs multi-token prefill instead of 1-token detect.
                gen_kwargs["language"] = self._npu_detected_language

            result = self._generate(clip, gen_kwargs)

            if not self.language and not self._npu_detected_language:
                detected = (getattr(result, "language", "") or "").strip()
                if detected:
                    lang_code = detected.strip("<|>")
                    if lang_code:
                        self._npu_detected_language = f"<|{lang_code}|>"
                        logger.info(
                            "NPU auto-detected language on first utterance: %s",
                            self._npu_detected_language,
                        )

            texts = getattr(result, "texts", None) or []
            text = _clean_text(texts[0]) if texts else ""
            if text:
                all_segments.extend(
                    split_clauses_proportionally(text, start_ms, end_ms)
                )

            progress = 20 + int(78 * (idx + 1) / total_spans)
            callback(progress, f"NPU transcribing {idx + 1}/{total_spans}")

        return all_segments

    # ------------------------------------------------------------------
    def _generate(self, audio, config: dict):
        """Run the pipeline, passing the numpy array directly when possible.

        ``audio.tolist()`` would materialise ~1M Python floats for a 60s chunk
        before every NPU call; the binding accepts arrays, so only fall back to
        a list if a particular build rejects it.
        """
        try:
            return self.pipeline.generate(audio, **config)
        except TypeError:
            return self.pipeline.generate(audio.tolist(), **config)

    def _transcribe_chunk(
        self, chunk: "np.ndarray", offset: float, config: dict
    ) -> List[ASRDataSeg]:
        try:
            result = self._generate(chunk, config)
        except Exception as e:
            if config.get("word_timestamps") and self._word_ts_supported:
                # An NPU DEVICE_LOST must go to the NPU fallback in _run():
                # rebuilding an NPU pipeline after a device loss hangs until the
                # device is reset (PnP disable/enable or reboot).
                if _is_npu_failure(e):
                    raise
                # Intel GPU plugin does not support encoder attention decomposition
                # required by word_timestamps; fall back to segment timestamps.
                logger.warning(
                    "Word-level timestamps are not supported on this device, "
                    "falling back to segment timestamps: %s",
                    e,
                )
                self._word_ts_supported = False
                config.pop("word_timestamps", None)
                # The failed generate() call corrupted the pipeline's internal
                # state; retrying on it shifts/loses segments of the first
                # window, so start from a fresh pipeline instead.
                self._rebuild_pipeline()
                result = self._generate(chunk, config)
            else:
                raise

        segments: List[ASRDataSeg] = []
        words = getattr(result, "words", None)
        chunks = getattr(result, "chunks", None)

        if self.need_word_time_stamp and words:
            for word in words:
                text = _clean_text(word.word)
                if not text:
                    continue
                start = int((offset + word.start_ts) * 1000)
                end = int((offset + word.end_ts) * 1000)
                if end <= start:
                    end = start + 80
                segments.append(ASRDataSeg(text, start, end))
            if segments:
                return segments

        if chunks:
            limit_ms = int((offset + len(chunk) / 16000) * 1000)
            for item in chunks:
                text = _clean_text(item.text)
                if not text:
                    continue
                start = max(int(offset * 1000), int((offset + item.start_ts) * 1000))
                end = min(limit_ms, int((offset + item.end_ts) * 1000))
                if end <= start:
                    end = start + 200
                segments.append(ASRDataSeg(text, start, end))
            if segments:
                return segments

        texts = getattr(result, "texts", None) or []
        text = _clean_text(texts[0]) if texts else ""
        if text:
            start = int(offset * 1000)
            end = int((offset + len(chunk) / 16000) * 1000)
            segments.append(ASRDataSeg(text, start, end))
        return segments

    def _transcribe_gpu(
        self, audio: "np.ndarray", callback: Callable[[int, str], None]
    ) -> List[ASRDataSeg]:
        """Standard GPU/CPU path with native timestamps (240s chunks)."""
        total_samples = len(audio)
        chunk_seconds = chunk_seconds_for_device(self.used_device)
        chunk_samples = chunk_seconds * 16000
        chunk_count = max(1, (total_samples + chunk_samples - 1) // chunk_samples)
        config = self._build_config()

        all_segments: List[ASRDataSeg] = []
        started = time.perf_counter()
        for index in range(chunk_count):
            start_sample = index * chunk_samples
            end_sample = min(total_samples, start_sample + chunk_samples)
            chunk = audio[start_sample:end_sample]
            offset = start_sample / 16000.0

            logger.info(
                "Transcribing chunk %d/%d (%.1fs - %.1fs)",
                index + 1,
                chunk_count,
                offset,
                end_sample / 16000.0,
            )
            chunk_started = time.perf_counter()
            all_segments.extend(self._transcribe_chunk(chunk, offset, config))
            chunk_elapsed = time.perf_counter() - chunk_started
            audio_seconds = len(chunk) / 16000.0
            speed = audio_seconds / chunk_elapsed if chunk_elapsed > 0 else 0.0
            logger.info(
                "Chunk %d/%d done in %.1fs: %.1fs audio, %.1fx realtime",
                index + 1,
                chunk_count,
                chunk_elapsed,
                audio_seconds,
                speed,
            )

            progress = 20 + int(78 * (index + 1) / chunk_count)
            callback(
                progress,
                f"transcribing {index + 1}/{chunk_count} "
                f"({chunk_elapsed:.1f}s, {speed:.1f}x realtime)",
            )
        total_elapsed = time.perf_counter() - started
        logger.info(
            "Transcription finished: %.1fs audio in %.1fs (%.1fx realtime)",
            total_samples / 16000.0,
            total_elapsed,
            (total_samples / 16000.0) / total_elapsed if total_elapsed > 0 else 0.0,
        )
        return all_segments

    def _fallback_after_npu_failure(
        self,
        audio: "np.ndarray",
        callback: Callable[[int, str], None],
        exc: Exception,
    ) -> List[ASRDataSeg]:
        """Recover from an NPU failure without touching the display GPU.

        Only a discrete Arc is a safe GPU fallback. Integrated GPUs share
        silicon with the display, so a hang there freezes the screen; use the
        CPU in that case.
        """
        target = npu_fallback_device()
        if target == "intel_gpu":
            logger.warning(
                "NPU inference failed (%s); retrying on the discrete Intel GPU "
                "with the already downloaded int8 model",
                exc,
            )
            callback(21, "NPU failed, switching to the discrete Intel GPU")
            self.device = "intel_gpu"
            override = "intel_gpu"
        else:
            logger.warning(
                "NPU inference failed (%s); falling back to Intel CPU because "
                "the integrated GPU shares the display and can freeze the screen",
                exc,
            )
            callback(21, "NPU failed, switching to Intel CPU (safer)")
            self.device = "intel_cpu"
            override = "CPU"
        self.pipeline = None
        self.used_device = ""
        self.used_precision = ""
        gc.collect()
        self._create_pipeline(device_override=override, precision_override="int8")
        callback(21, f"设备: {self.used_device_label}")
        return self._transcribe_gpu(audio, callback)

    def _run(self, callback=None, **kwargs) -> str:
        if callback is None:

            def callback(x, y):
                return None

        self._callback = callback

        try:
            import openvino_genai  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "OpenVINO GenAI is not installed. Install the Intel extras:\n"
                '  pip install -e ".[intel]"\n'
                f"Details: {e}"
            )

        try:
            requested = (self.device or "auto").lower()
            if requested.startswith(("intel_npu", "npu")):
                target_label = "NPU"
            elif requested.startswith(("intel_gpu", "gpu")):
                target_label = "Intel GPU"
            elif requested.startswith(("intel_cpu", "intel-cpu")):
                target_label = "Intel CPU"
            else:
                target_label = "OpenVINO"
            callback(
                1,
                f"正在编译/加载 {target_label} 模型（首次运行可能需要几分钟）",
            )
            self._create_pipeline()
            callback(20, f"设备: {self.used_device_label}")

            audio = _read_wav_mono_16k(self.audio_input)
            total_samples = len(audio)
            if total_samples == 0:
                raise RuntimeError("Empty audio input")

            if _is_npu_device(self.used_device):
                # OpenVINO GenAI 2026.4 returns proper timestamps on the NPU,
                # which is ~3x faster than per-utterance segmentation (one
                # encoder pass per 30s window instead of one per utterance).
                try:
                    all_segments = self._transcribe_gpu(audio, callback)
                except Exception as exc:
                    if not _is_npu_failure(exc):
                        raise
                    all_segments = self._fallback_after_npu_failure(audio, callback, exc)
                else:
                    if not all_segments:
                        logger.warning(
                            "NPU returned no segments with native timestamps; "
                            "falling back to pause segmentation"
                        )
                        try:
                            all_segments = self._transcribe_npu(audio, callback)
                        except Exception as exc:
                            if not _is_npu_failure(exc):
                                raise
                            all_segments = self._fallback_after_npu_failure(
                                audio, callback, exc
                            )
            else:
                all_segments = self._transcribe_gpu(audio, callback)

            if not all_segments:
                raise RuntimeError("No speech recognized; check the audio or the model")

            all_segments.sort(key=lambda seg: seg.start_time)
            srt_text = ASRData(all_segments).to_srt(
                layout=SubtitleLayoutEnum.ORIGINAL_ON_TOP
            )
            callback(100, "completed")
            return srt_text
        finally:
            # Workaround for openvino.genai#1460: destroy NPU pipeline explicitly
            # before worker process teardown so Level0 cleans up cleanly.
            self.pipeline = None
            gc.collect()

    def _make_segments(self, resp_data: str) -> List[ASRDataSeg]:
        asr_data = ASRData.from_srt(resp_data)
        filtered = []
        for seg in asr_data.segments:
            text = seg.text.strip()
            if not text:
                continue
            if text.startswith(("【", "[", "(", "（")):
                continue
            filtered.append(seg)
        return filtered
