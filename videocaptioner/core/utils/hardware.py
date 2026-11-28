# coding:utf-8
"""Hardware detection helpers.

Important: the OpenVINO runtime must never be imported in the Qt (UI) process.
OpenVINO 2026 requires a newer MSVC runtime than the one bundled with PyQt5
(14.26), and loading both in one process crashes with 0xc0000005.

Therefore all UI-facing helpers below avoid importing OpenVINO; the
OpenVINO-specific helpers are only used inside ``openvino_worker.py``.
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
from typing import List, Tuple


def _run(cmd: List[str], timeout: int = 8) -> str:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return (result.stdout or "") + (result.stderr or "")
    except Exception:
        return ""


def display_adapters() -> List[str]:
    """Enumerate display adapter names without WMI or OpenVINO."""
    if os.name != "nt":
        return []
    try:
        from ctypes import wintypes

        class DISPLAY_DEVICEW(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("DeviceName", wintypes.WCHAR * 32),
                ("DeviceString", wintypes.WCHAR * 128),
                ("StateFlags", wintypes.DWORD),
                ("DeviceID", wintypes.WCHAR * 128),
                ("DeviceKey", wintypes.WCHAR * 128),
            ]

        names: List[str] = []
        user32 = ctypes.windll.user32
        index = 0
        while True:
            device = DISPLAY_DEVICEW()
            device.cb = ctypes.sizeof(DISPLAY_DEVICEW)
            if not user32.EnumDisplayDevicesW(None, index, ctypes.byref(device), 0):
                break
            name = (device.DeviceString or "").strip()
            if name and "microsoft" not in name.lower():
                names.append(name)
            index += 1
        # One physical adapter can be reported once per output; keep it unique.
        return list(dict.fromkeys(names))
    except Exception:
        return []


def intel_gpu_adapters() -> List[str]:
    return [name for name in display_adapters() if "intel" in name.lower()]


def classify_intel_gpu_name(name: str) -> str:
    """Return ``discrete`` / ``integrated`` / ``unknown`` for an adapter name."""
    lowered = (name or "").lower()
    if "arc" in lowered:
        # Arc A/B series and Pro cards are discrete. The bare
        # "Intel(R) Arc(TM) Graphics" (Meteor Lake) and the Arc 1xxV naming
        # (Lunar Lake) are integrated GPUs that share silicon with the display.
        if re.search(r"arc.*\b[ab]\d{2,3}\b", lowered) or "pro" in lowered:
            return "discrete"
        return "integrated"
    if any(tag in lowered for tag in ("uhd", "iris", "hd graphics", "graphics 7")):
        return "integrated"
    if "intel" in lowered and "graphics" in lowered:
        return "integrated"
    return "unknown"


def has_discrete_intel_gpu() -> bool:
    """True when a discrete Intel Arc card is present (no OpenVINO import)."""
    return any(
        classify_intel_gpu_name(name) == "discrete"
        for name in intel_gpu_adapters()
    )


def is_integrated_openvino_gpu(name: str) -> bool:
    """Classify an OpenVINO FULL_DEVICE_NAME (which appends iGPU/dGPU)."""
    lowered = (name or "").lower()
    if "(igpu)" in lowered:
        return True
    if "(dgpu)" in lowered:
        return False
    return classify_intel_gpu_name(name) == "integrated"


def npu_fallback_device() -> str:
    """Where to go when the NPU fails.

    Only fall back to a discrete Arc. An integrated GPU shares silicon with the
    display, so a driver hang there freezes the whole screen; use the CPU
    instead of risking a black screen.
    """
    return "intel_gpu" if has_discrete_intel_gpu() else "intel_cpu"


def openvino_gpu_choices() -> List[Tuple[str, str]]:
    """Return ``[(display_text, adapter_name)]`` for the transcription UI.

    Uses display adapters (never imports OpenVINO in the UI process); the name
    is matched to the OpenVINO device at run time.
    """
    labels = {"discrete": "独显", "integrated": "核显", "unknown": "GPU"}
    choices: List[Tuple[str, str]] = []
    for name in intel_gpu_adapters():
        kind = classify_intel_gpu_name(name)
        choices.append((f"{name}  ·  {labels[kind]}", name))
    return choices


def is_intel_arc(name: str) -> bool:
    return "arc" in (name or "").lower()


def is_intel_arc_a_series(name: str) -> bool:
    """Arc A-series (Alchemist): A310 / A380 / A580 / A750 / A770 ..."""
    return bool(re.search(r"arc.*\ba\d{3}\b", (name or "").lower()))


def is_intel_arc_b_series(name: str) -> bool:
    """Arc B-series (Battlemage): B570 / B580 / Pro B50 ..."""
    return bool(re.search(r"arc.*\bb\d{2,3}\b", (name or "").lower()))


# Known Intel AI Boost NPU PCI device IDs (VEN_8086) across Core Ultra generations:
# - 7D1D: Core Ultra Series 1 (Meteor Lake, e.g. Core Ultra 7 155H, NPU 3720)
# - AD1D: Core Ultra Series 2 (Arrow Lake-H/S, NPU 3720)
# - 643E: Core Ultra 200V (Lunar Lake, NPU 4000)
# - B03E: Core Ultra Series 3 (Panther Lake)
INTEL_NPU_PCI_DEVICES = {
    "7D1D": "Intel(R) AI Boost (Meteor Lake NPU)",
    "AD1D": "Intel(R) AI Boost (Arrow Lake NPU)",
    "643E": "Intel(R) AI Boost (Lunar Lake NPU)",
    "B03E": "Intel(R) AI Boost (Panther Lake NPU)",
}

# Generation keys returned by npu_generation(), derived from the PCI labels
# above (and the Linux accel names).
_NPU_GENERATION_KEYWORDS = {
    "meteor_lake": ("meteor lake",),
    "arrow_lake": ("arrow lake",),
    "lunar_lake": ("lunar lake",),
    "panther_lake": ("panther lake",),
}


def npu_generation() -> str:
    """Return the newest present NPU generation, e.g. ``"lunar_lake"``.

    Empty string when no known-generation NPU is found (unknown OEM naming,
    Linux accel names, or no NPU at all). Callers should treat unknown as
    the most conservative behaviour.
    """
    for name in intel_npu_adapters():
        lowered = name.lower()
        for generation, keywords in _NPU_GENERATION_KEYWORDS.items():
            if any(kw in lowered for kw in keywords):
                return generation
    return ""


def intel_npu_adapters() -> List[str]:
    """Detect active Intel AI Boost NPU devices without importing OpenVINO."""
    names: List[str] = []
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Enum\PCI"
            ) as pci:
                subkey_count = winreg.QueryInfoKey(pci)[0]
                for i in range(subkey_count):
                    dev_key_name = winreg.EnumKey(pci, i)
                    upper_name = dev_key_name.upper()
                    if "VEN_8086" not in upper_name:
                        continue
                    matched_pci_label = ""
                    for dev_id, label in INTEL_NPU_PCI_DEVICES.items():
                        if f"DEV_{dev_id}" in upper_name:
                            matched_pci_label = label
                            break
                    with winreg.OpenKey(pci, dev_key_name) as dev_key:
                        inst_count = winreg.QueryInfoKey(dev_key)[0]
                        for j in range(inst_count):
                            inst_name = winreg.EnumKey(dev_key, j)
                            with winreg.OpenKey(dev_key, inst_name) as inst:
                                # Volatile 'Control' subkey exists iff device is currently present
                                try:
                                    winreg.OpenKey(inst, "Control").Close()
                                except OSError:
                                    continue
                                desc = ""
                                for value_name in ("FriendlyName", "DeviceDesc"):
                                    try:
                                        raw_val, _ = winreg.QueryValueEx(inst, value_name)
                                        if raw_val:
                                            desc = str(raw_val).split(";")[-1].strip()
                                            break
                                    except OSError:
                                        continue
                                lowered = desc.lower()
                                if matched_pci_label or any(
                                    kw in lowered for kw in ("ai boost", "npu")
                                ):
                                    # The FriendlyName is often a bare
                                    # "Intel(R) AI Boost"; when the PCI ID
                                    # told us the generation and the name
                                    # doesn't, keep the tagged label so
                                    # callers can tell NPU generations apart.
                                    name_known = any(
                                        gen in lowered
                                        for gens in _NPU_GENERATION_KEYWORDS.values()
                                        for gen in gens
                                    )
                                    if matched_pci_label and not name_known:
                                        names.append(matched_pci_label)
                                    else:
                                        names.append(desc or matched_pci_label)
        except Exception:
            pass
    else:
        # Linux intel_vpu exposes /sys/class/accel/accel*
        try:
            from pathlib import Path

            accel_root = Path("/sys/class/accel")
            if accel_root.is_dir():
                for entry in sorted(accel_root.iterdir()):
                    vendor_file = entry / "device" / "vendor"
                    if vendor_file.exists():
                        vendor = vendor_file.read_text(encoding="utf-8", errors="ignore").strip()
                        if vendor.lower() == "0x8086":
                            names.append(f"Intel(R) AI Boost ({entry.name})")
        except Exception:
            pass
    return list(dict.fromkeys(names))


def has_nvidia_gpu() -> bool:
    if any("nvidia" in name.lower() for name in display_adapters()):
        return True
    if shutil.which("nvidia-smi"):
        return "GPU" in _run(["nvidia-smi", "-L"])
    return False


def has_intel_gpu() -> bool:
    return bool(intel_gpu_adapters())


def has_intel_npu() -> bool:
    return bool(intel_npu_adapters())


def openvino_packages_available() -> bool:
    """Check whether the Intel extras are installed without loading them."""
    import importlib.util

    try:
        return (
            importlib.util.find_spec("openvino") is not None
            and importlib.util.find_spec("openvino_genai") is not None
        )
    except Exception:
        return False


def preferred_backend_device() -> str:
    """Pick a sensible default for the ``auto`` device option.

    Prefer a discrete Arc when present. On iGPU-only Core Ultra laptops the
    integrated GPU shares silicon with the display, and OpenVINO GPU inference
    can hang the graphics driver (the whole screen freezes), so the NPU is the
    safer default there. An explicitly selected intel_gpu still works.
    """
    if has_nvidia_gpu():
        return "cuda"
    if openvino_packages_available():
        if has_discrete_intel_gpu():
            return "intel_gpu"
        if has_intel_npu():
            return "intel_npu"
        if has_intel_gpu():
            return "intel_gpu"
    return "cpu"


def describe_hardware() -> str:
    """Human readable hardware summary for the settings page (no OpenVINO)."""
    adapters = display_adapters()
    npus = intel_npu_adapters()
    parts: List[str] = []
    intel = [name for name in adapters if "intel" in name.lower()]
    nvidia = [name for name in adapters if "nvidia" in name.lower()]
    if intel:
        parts.append("Intel GPU: " + " / ".join(intel))
    if npus:
        parts.append("Intel NPU: " + " / ".join(npus))
    if nvidia:
        parts.append("NVIDIA GPU: " + " / ".join(nvidia))
    if not parts:
        if adapters:
            parts.append("GPU: " + " / ".join(adapters))
        else:
            parts.append("No acceleration hardware detected")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# The helpers below are only used inside the OpenVINO worker subprocess.
# ---------------------------------------------------------------------------
def openvino_available() -> bool:
    try:
        import openvino  # noqa: F401

        return True
    except Exception:
        return False


def list_openvino_devices() -> List[Tuple[str, str]]:
    """Return ``[(device_id, full_name)]`` such as ``[("GPU.1", "Intel(R) Arc(TM) A770")]``."""
    if not openvino_available():
        return []
    try:
        import openvino as ov

        core = ov.Core()
        devices: List[Tuple[str, str]] = []
        for device in core.available_devices:
            try:
                name = core.get_property(device, "FULL_DEVICE_NAME")
            except Exception:
                name = device
            devices.append((device, name))
        return devices
    except Exception:
        return []


def intel_gpu_devices() -> List[Tuple[str, str]]:
    return [(device, name) for device, name in list_openvino_devices() if device.startswith("GPU")]


def intel_npu_devices() -> List[Tuple[str, str]]:
    return [(device, name) for device, name in list_openvino_devices() if device.startswith("NPU")]


def intel_npu_available() -> bool:
    return bool(intel_npu_devices())


def _normalize_device_name(name: str) -> str:
    """Normalize adapter names for matching ("(iGPU)"/"(dGPU)" suffixes differ)."""
    text = (name or "").lower().strip()
    for suffix in ("(igpu)", "(dgpu)", "(gpu)", " graphics"):
        text = text.replace(suffix, " ")
    return " ".join(text.split())


def resolve_openvino_device(preference: str, device_name: str = "") -> str:
    """Map the UI device option to an OpenVINO device string.

    ``device_name`` selects a concrete adapter by name (multi-GPU machines);
    OpenVINO appends "(iGPU)"/"(dGPU)" to the driver name, so matching ignores
    those suffixes. Falls back to the default policy when the name is unknown.
    """
    preference = (preference or "").lower()
    wanted = _normalize_device_name(device_name) if device_name else ""

    def _match(devices: List[Tuple[str, str]]) -> str:
        if not wanted:
            return ""
        for device, name in devices:
            if _normalize_device_name(name) == wanted:
                return device
        for device, name in devices:
            normalized = _normalize_device_name(name)
            if wanted in normalized or normalized in wanted:
                return device
        return ""

    if preference.startswith("intel_cpu") or preference.startswith("intel-cpu"):
        return "CPU"

    if preference.startswith("intel_npu") or preference.startswith("npu"):
        npus = intel_npu_devices()
        if npus:
            return _match(npus) or npus[0][0]
        # Never silently move an NPU job onto the GPU: on laptops the Intel GPU
        # is usually the display adapter, and a driver hang there freezes the
        # screen. Let the caller retry on CPU with a clear message instead.
        raise RuntimeError(
            "Intel NPU (AI Boost) is not available right now. It may be busy "
            "with another task or the driver is not ready. Wait for the "
            "running NPU job to finish, or select Intel CPU."
        )

    if preference.startswith("intel_gpu") or preference.startswith("gpu"):
        gpus = intel_gpu_devices()
        if not gpus:
            raise RuntimeError(
                "No Intel GPU detected. Please install the Intel graphics driver "
                "(OpenVINO GPU plugin)."
            )
        if wanted and (matched := _match(gpus)):
            return matched
        arc = [(device, name) for device, name in gpus if is_intel_arc(name)]
        return (arc or gpus)[0][0]

    if preference.startswith("cpu"):
        return "CPU"

    return "AUTO"


def resolve_faster_whisper_device(preference: str) -> str:
    """CTranslate2 only supports cuda/cpu; never return cuda without an NVIDIA GPU."""
    preference = (preference or "").lower()
    if preference.startswith("intel"):
        return "cpu"
    if preference.startswith("cuda"):
        return "cuda" if has_nvidia_gpu() else "cpu"
    return "cpu"
