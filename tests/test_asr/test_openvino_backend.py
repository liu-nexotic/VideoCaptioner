"""Tests for the Intel Arc / OpenVINO backend (no GPU required)."""

import numpy as np

from videocaptioner.core.asr.openvino_asr import (
    segment_speech_for_npu,
    split_clauses_proportionally,
)
from videocaptioner.core.entities import FasterWhisperModelEnum
from videocaptioner.core.utils.hardware import (
    describe_hardware,
    intel_npu_adapters,
    is_intel_arc_a_series,
    is_intel_arc_b_series,
    resolve_faster_whisper_device,
)
from videocaptioner.core.utils.model_downloader import (
    FASTER_WHISPER_REPOS,
    OPENVINO_WHISPER_INT8_REPOS,
    OPENVINO_WHISPER_REPOS,
    faster_whisper_model_status,
    openvino_whisper_model_dir,
)


def test_all_ui_models_have_backend_mappings():
    """Every model exposed in the UI must have CT2, fp16 and int8 mappings."""
    for model in FasterWhisperModelEnum:
        assert model.value in FASTER_WHISPER_REPOS
        assert model.value in OPENVINO_WHISPER_REPOS
        assert model.value in OPENVINO_WHISPER_INT8_REPOS
        assert OPENVINO_WHISPER_REPOS[model.value].endswith("-fp16-ov")
        assert OPENVINO_WHISPER_INT8_REPOS[model.value].endswith("-int8-ov")


def test_arc_series_name_detection():
    assert is_intel_arc_a_series("Intel(R) Arc(TM) A770 Graphics")
    assert is_intel_arc_a_series("Intel(R) Arc(TM) A380 Graphics")
    assert is_intel_arc_a_series("Intel(R) Arc(TM) A750 Graphics")
    assert not is_intel_arc_a_series("Intel(R) Arc(TM) B580 Graphics")

    assert is_intel_arc_b_series("Intel(R) Arc(TM) B580 Graphics")
    assert is_intel_arc_b_series("Intel(R) Arc(TM) Pro B50 Graphics")
    assert not is_intel_arc_b_series("Intel(R) Arc(TM) A770 Graphics")


def test_faster_whisper_never_uses_cuda_on_intel_options():
    assert resolve_faster_whisper_device("intel_gpu") == "cpu"
    assert resolve_faster_whisper_device("intel_npu") == "cpu"
    assert resolve_faster_whisper_device("cpu") == "cpu"


def test_truncated_model_is_reported_as_not_ready(tmp_path):
    model_dir = tmp_path / "faster-whisper-tiny"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"0" * 1024)

    ready, reason = faster_whisper_model_status(tmp_path, "tiny")
    assert ready is False
    assert "size" in reason.lower()


def test_openvino_model_directory_layout(tmp_path):
    model_path = openvino_whisper_model_dir(tmp_path, "large-v3-turbo")
    assert model_path.name == "whisper-large-v3-turbo-fp16-ov"
    assert model_path.parent.name == "openvino"


def test_hardware_description_is_not_empty():
    assert describe_hardware().strip()


def test_openvino_int8_model_directory_layout(tmp_path):
    model_path = openvino_whisper_model_dir(tmp_path, "large-v3-turbo", precision="int8")
    assert model_path.name == "whisper-large-v3-turbo-int8-ov"
    assert model_path.parent.name == "openvino"


def test_intel_npu_adapters_list_without_openvino():
    # Must never raise and must not import OpenVINO in the UI process.
    assert isinstance(intel_npu_adapters(), list)


def test_npu_speech_segmentation_splits_on_pauses():
    sr = 16000
    silence = np.zeros(int(1.0 * sr), dtype=np.float32)

    def tone(seconds: float) -> np.ndarray:
        t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
        return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    audio = np.concatenate([silence, tone(2.0), silence, tone(3.0), silence])
    spans = segment_speech_for_npu(audio)

    assert 2 <= len(spans) <= 3
    for start, end in spans:
        assert 0 <= start < end <= len(audio)
        assert (end - start) / sr <= 14.5
    assert [start for start, _ in spans] == sorted(start for start, _ in spans)


def test_npu_speech_segmentation_caps_long_speech():
    sr = 16000
    t = np.arange(25 * sr) / sr
    audio = (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    spans = segment_speech_for_npu(audio)

    assert spans
    assert all((end - start) / sr <= 14.5 for start, end in spans)
    # Long monologues must still be fully covered.
    assert sum(end - start for start, end in spans) >= len(audio) * 0.9


def test_clause_splitting_keeps_total_time():
    segments = split_clauses_proportionally(
        "First sentence. Second sentence! Third sentence?",
        1000,
        13000,
        max_clause_ms=6000,
    )
    assert len(segments) == 3
    assert segments[0].start_time == 1000
    assert segments[-1].end_time == 13000
    assert [seg.text for seg in segments] == [
        "First sentence.",
        "Second sentence!",
        "Third sentence?",
    ]

def test_gpu_precision_hint_explicit_and_env(monkeypatch):
    from videocaptioner.core.asr.openvino_asr import _gpu_precision_hint

    assert _gpu_precision_hint("GPU", "f32") == "f32"
    assert _gpu_precision_hint("GPU", "fp16") == "fp16"

    monkeypatch.setenv("VIDEOCAPTIONER_OPENVINO_GPU_PRECISION", "fp16")
    assert _gpu_precision_hint("GPU") == "fp16"
    assert _gpu_precision_hint("CPU") == ""


def test_gpu_precision_hint_uses_f32_on_integrated_gpu(monkeypatch):
    from videocaptioner.core.asr.openvino_asr import _gpu_precision_hint
    from videocaptioner.core.utils import hardware

    monkeypatch.delenv("VIDEOCAPTIONER_OPENVINO_GPU_PRECISION", raising=False)

    monkeypatch.setattr(
        hardware,
        "intel_gpu_devices",
        lambda: [("GPU", "Intel(R) Arc(TM) Graphics (iGPU)")],
    )
    assert _gpu_precision_hint("GPU") == "f32"

    monkeypatch.setattr(
        hardware,
        "intel_gpu_devices",
        lambda: [("GPU.1", "Intel(R) Arc(TM) Pro B50 Graphics (dGPU)")],
    )
    assert _gpu_precision_hint("GPU.1") == ""


def test_openvino_required_files_cover_tokenizer_but_not_int8_only_config():
    """A partial download must not be reported complete.

    The tokenizer files are needed to load the pipeline, but
    ``openvino_config.json`` only exists in the int8 repos: requiring it would
    make every fp16 model look incomplete and re-download on each run.
    """
    from videocaptioner.core.utils.model_downloader import (
        OPENVINO_MODEL_REQUIRED_FILES,
    )

    required = set(OPENVINO_MODEL_REQUIRED_FILES)
    assert "openvino_tokenizer.xml" in required
    assert "openvino_tokenizer.bin" in required
    assert "preprocessor_config.json" in required
    assert "tokenizer.json" in required
    assert "vocab.json" in required
    assert "openvino_config.json" not in required


def test_format_runtime_device_includes_device_name_and_precision():
    from videocaptioner.core.asr.openvino_asr import _format_runtime_device

    assert (
        _format_runtime_device(
            "GPU.1", "Intel(R) Arc(TM) Pro B50 Graphics (dGPU)", "fp16", ""
        )
        == "GPU.1 · Intel(R) Arc(TM) Pro B50 Graphics (dGPU) (fp16)"
    )
    assert (
        _format_runtime_device(
            "GPU", "Intel(R) Arc(TM) Graphics (iGPU)", "int8", "f32"
        )
        == "GPU · Intel(R) Arc(TM) Graphics (iGPU) (int8, f32)"
    )
    assert (
        _format_runtime_device("NPU", "Intel(R) AI Boost", "int8", "")
        == "NPU · Intel(R) AI Boost (int8)"
    )
    assert _format_runtime_device("CPU", "", "int8", "") == "CPU (int8)"


def test_openvino_model_precision_enum_two_items():
    from videocaptioner.core.entities import OpenVINOModelPrecisionEnum

    items = list(OpenVINOModelPrecisionEnum)
    assert len(items) == 2
    assert OpenVINOModelPrecisionEnum.INT8.precision_key == "int8"
    assert OpenVINOModelPrecisionEnum.FP16.precision_key == "fp16"
    assert "INT8" in OpenVINOModelPrecisionEnum.INT8.value
    assert "FP16" in OpenVINOModelPrecisionEnum.FP16.value


_FAKE_GPUS = [
    ("GPU.0", "Intel(R) UHD Graphics 770 (iGPU)"),
    ("GPU.1", "Intel(R) Arc(TM) Pro B50 Graphics (dGPU)"),
]


def test_resolve_openvino_device_by_adapter_name(monkeypatch):
    """Multi-GPU machines pick a concrete GPU by name, not by index."""
    import videocaptioner.core.utils.hardware as hw

    monkeypatch.setattr(hw, "intel_gpu_devices", lambda: list(_FAKE_GPUS))

    assert (
        hw.resolve_openvino_device("intel_gpu", "Intel(R) UHD Graphics 770") == "GPU.0"
    )
    assert (
        hw.resolve_openvino_device("intel_gpu", "Intel(R) Arc(TM) Pro B50 Graphics")
        == "GPU.1"
    )
    # OpenVINO appends "(iGPU)"/"(dGPU)" to the driver name; partial names match too.
    assert hw.resolve_openvino_device("intel_gpu", "Arc(TM) Pro B50") == "GPU.1"
    # Auto selection and unknown names fall back to the discrete-first policy.
    assert hw.resolve_openvino_device("intel_gpu", "") == "GPU.1"
    assert hw.resolve_openvino_device("intel_gpu", "Not A Real GPU") == "GPU.1"


def test_resolve_openvino_device_npu_never_falls_back_to_gpu(monkeypatch):
    """An NPU job must never be silently moved onto the display GPU."""
    import pytest

    import videocaptioner.core.utils.hardware as hw

    monkeypatch.setattr(hw, "intel_npu_devices", lambda: [])
    monkeypatch.setattr(hw, "intel_gpu_devices", lambda: list(_FAKE_GPUS))

    with pytest.raises(RuntimeError, match="NPU"):
        hw.resolve_openvino_device("intel_npu", "Intel(R) UHD Graphics 770")


def test_npu_failure_uses_cpu_unless_a_discrete_gpu_exists(monkeypatch):
    import videocaptioner.core.utils.hardware as hw

    monkeypatch.setattr(hw, "has_discrete_intel_gpu", lambda: False)
    assert hw.npu_fallback_device() == "intel_cpu"
    monkeypatch.setattr(hw, "has_discrete_intel_gpu", lambda: True)
    assert hw.npu_fallback_device() == "intel_gpu"


def test_preferred_backend_prefers_npu_on_integrated_gpu(monkeypatch):
    """iGPU-only Core Ultra laptops default to the NPU (GPU shares the display)."""
    import videocaptioner.core.utils.hardware as hw

    monkeypatch.setattr(hw, "has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(hw, "openvino_packages_available", lambda: True)
    monkeypatch.setattr(hw, "has_intel_gpu", lambda: True)
    monkeypatch.setattr(hw, "has_intel_npu", lambda: True)
    monkeypatch.setattr(hw, "has_discrete_intel_gpu", lambda: False)
    assert hw.preferred_backend_device() == "intel_npu"

    monkeypatch.setattr(hw, "has_discrete_intel_gpu", lambda: True)
    assert hw.preferred_backend_device() == "intel_gpu"

    monkeypatch.setattr(hw, "has_intel_gpu", lambda: False)
    monkeypatch.setattr(hw, "has_discrete_intel_gpu", lambda: False)
    assert hw.preferred_backend_device() == "intel_npu"

def test_intel_gpu_kind_classification_and_choices(monkeypatch):
    import videocaptioner.core.utils.hardware as hw

    assert hw.classify_intel_gpu_name("Intel(R) UHD Graphics 770") == "integrated"
    assert hw.classify_intel_gpu_name("Intel(R) Iris(R) Xe Graphics") == "integrated"
    assert hw.classify_intel_gpu_name("Intel(R) Arc(TM) Pro B50 Graphics") == "discrete"
    # Meteor Lake's "Arc(TM) Graphics" has no model number: it is the iGPU.
    assert hw.classify_intel_gpu_name("Intel(R) Arc(TM) Graphics") == "integrated"
    assert hw.classify_intel_gpu_name("Intel(R) Arc(TM) 140V") == "integrated"
    assert hw.classify_intel_gpu_name("Intel(R) Arc(TM) A770 Graphics") == "discrete"
    assert hw.classify_intel_gpu_name("GameViewer Virtual Display") == "unknown"

    monkeypatch.setattr(
        hw,
        "intel_gpu_adapters",
        lambda: [
            "Intel(R) UHD Graphics 770",
            "Intel(R) Arc(TM) Pro B50 Graphics",
        ],
    )
    choices = hw.openvino_gpu_choices()
    assert [name for _label, name in choices] == [
        "Intel(R) UHD Graphics 770",
        "Intel(R) Arc(TM) Pro B50 Graphics",
    ]
    assert "核显" in choices[0][0]
    assert "独显" in choices[1][0]

def test_transcribe_device_choices_merge_gpu_names():
    """运行设备下拉框直接包含具体 GPU，选中即绑定到该卡。"""
    from videocaptioner.ui.components.TranscribeDeviceCard import (
        build_device_choices,
    )

    gpus = [
        ("Intel(R) UHD Graphics 770  ·  核显", "Intel(R) UHD Graphics 770"),
        (
            "Intel(R) Arc(TM) Pro B50 Graphics  ·  独显",
            "Intel(R) Arc(TM) Pro B50 Graphics",
        ),
    ]
    choices = build_device_choices(gpus)

    displays = [display for display, _device, _gpu in choices]
    assert displays[0].startswith("自动选择")
    assert all("Intel GPU · " in d for d in displays[2:4])
    assert choices[1] == ("Intel GPU（自动，优先独显）", "intel_gpu", "")
    assert [c[2] for c in choices[2:4]] == [name for _d, name in gpus]
    assert [c[1] for c in choices[2:4]] == ["intel_gpu", "intel_gpu"]

    # NPU / CPU / CUDA entries remain available after the GPU entries.
    tail = [c[1] for c in choices[4:]]
    assert tail == ["intel_npu", "intel_cpu", "cuda", "cpu"]

    # Without display enumeration the automatic Intel GPU entry must survive.
    no_gpu = build_device_choices([])
    assert ("Intel GPU（自动，优先独显）", "intel_gpu", "") in no_gpu

def test_worker_output_stops_at_terminal_marker():
    """DONE/ERROR end the protocol; do not block waiting for EOF."""
    from videocaptioner.core.asr.openvino_process import OpenVINOProcessASR

    class FakeStream:
        def __init__(self, lines):
            self.lines = list(lines)
            self.read_count = 0

        def readline(self):
            if self.read_count >= len(self.lines):
                raise AssertionError("read past the terminal marker")
            line = self.lines[self.read_count]
            self.read_count += 1
            return line

    events = []
    stream = FakeStream(
        [
            "PROGRESS 20|设备: GPU.1 · Arc (dGPU) (int8)\n",
            "PROGRESS 100|completed\n",
            "DONE\n",
            "never read\n",
        ]
    )
    completed, device, errors, recent = OpenVINOProcessASR._consume_worker_output(
        stream, lambda p, m: events.append((p, m))
    )
    assert completed is True
    assert device == "GPU.1 · Arc (dGPU) (int8)"
    assert errors == [] and recent == []
    assert stream.read_count == 3
    assert events[-1] == (100, "completed")

    completed, _device, errors, _recent = OpenVINOProcessASR._consume_worker_output(
        FakeStream(["ERROR boom\n", "never read\n"]), lambda p, m: None
    )
    assert completed is False
    assert errors == ["boom"]


def test_run_worker_kills_child_stuck_after_done(tmp_path):
    """A worker that prints DONE then hangs in teardown must not block the GUI."""
    import sys
    import time

    from videocaptioner.core.asr.openvino_process import OpenVINOProcessASR

    script = tmp_path / "fake_worker.py"
    script.write_text(
        "import time\n"
        "print('PROGRESS 100|completed', flush=True)\n"
        "print('DONE', flush=True)\n"
        "time.sleep(120)\n",
        encoding="utf-8",
    )
    output = tmp_path / "openvino_output.srt"
    output.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")

    worker = object.__new__(OpenVINOProcessASR)
    worker.active_device_label = ""
    started = time.perf_counter()
    ok, detail = worker._run_worker(
        [sys.executable, str(script)], output, lambda p, m: None
    )
    elapsed = time.perf_counter() - started

    assert ok, detail
    assert elapsed < 15

def test_npu_uses_shorter_chunks(monkeypatch):
    from videocaptioner.core.asr.openvino_asr import (
        CHUNK_SECONDS,
        NPU_CHUNK_SECONDS,
        chunk_seconds_for_device,
    )
    from videocaptioner.core.utils import hardware

    assert NPU_CHUNK_SECONDS < CHUNK_SECONDS

    monkeypatch.setattr(hardware, "npu_generation", lambda: "meteor_lake")
    assert chunk_seconds_for_device("NPU") == NPU_CHUNK_SECONDS
    monkeypatch.setattr(hardware, "npu_generation", lambda: "arrow_lake")
    assert chunk_seconds_for_device("NPU") == NPU_CHUNK_SECONDS
    monkeypatch.setattr(hardware, "npu_generation", lambda: "panther_lake")
    assert chunk_seconds_for_device("NPU") == NPU_CHUNK_SECONDS
    # No stability data for unknown generations: stay conservative.
    monkeypatch.setattr(hardware, "npu_generation", lambda: "")
    assert chunk_seconds_for_device("NPU") == NPU_CHUNK_SECONDS
    # Lunar Lake ran 240s chunks stably; give it the full window.
    monkeypatch.setattr(hardware, "npu_generation", lambda: "lunar_lake")
    assert chunk_seconds_for_device("NPU") == CHUNK_SECONDS
    # Explicit override wins over the generation table.
    monkeypatch.setenv("VIDEOCAPTIONER_NPU_CHUNK_SECONDS", "90")
    assert chunk_seconds_for_device("NPU") == 90
    monkeypatch.delenv("VIDEOCAPTIONER_NPU_CHUNK_SECONDS")

    assert chunk_seconds_for_device("GPU.1") == CHUNK_SECONDS
    assert chunk_seconds_for_device("CPU") == CHUNK_SECONDS


def test_npu_generation_matches_adapter_labels(monkeypatch):
    from videocaptioner.core.utils.hardware import npu_generation

    monkeypatch.setattr(
        "videocaptioner.core.utils.hardware.intel_npu_adapters",
        lambda: ["Intel(R) AI Boost (Lunar Lake NPU)"],
    )
    assert npu_generation() == "lunar_lake"
    monkeypatch.setattr(
        "videocaptioner.core.utils.hardware.intel_npu_adapters",
        lambda: ["Intel(R) AI Boost (Meteor Lake NPU)"],
    )
    assert npu_generation() == "meteor_lake"
    # FriendlyName without a generation tag must not guess.
    monkeypatch.setattr(
        "videocaptioner.core.utils.hardware.intel_npu_adapters",
        lambda: ["Intel(R) AI Boost"],
    )
    assert npu_generation() == ""
    monkeypatch.setattr(
        "videocaptioner.core.utils.hardware.intel_npu_adapters", lambda: []
    )
    assert npu_generation() == ""
