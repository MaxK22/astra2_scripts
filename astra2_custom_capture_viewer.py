#!/usr/bin/env python3
"""
Astra 2 Custom Capture Viewer
=============================

Direct pyorbbecsdk2 application. No Orbbec Viewer dependency.

Purpose
-------
- Connect directly to an Orbbec Astra 2.
- Force/verify the user's research capture profile:
    Depth: 1600x1200 @ 15 FPS, Y16
    Color: 1920x1080 @ 15 FPS, YUYV
    Depth work mode: High Resolution
    SDK depth post-processing: none
- Show live RGB, raw-depth visualization, and IMU status.
- Press SPACE to capture a short independent .bag burst (default 1.5 s).
- Write a sidecar JSON with device/profile/capture metadata.
- Keep RAW acquisition separate from preview visualization/processing.

Important
---------
The preview is only a visualization. It does NOT modify the frames written
by RecordDevice. No depth filtering, alignment, hole filling, colorization,
or resampling is applied to the recorded data.

This program follows the current pyorbbecsdk v2 recorder approach using
RecordDevice and separate ACCEL/GYRO streams, consistent with Orbbec's
current official recorder example.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

try:
    from pyorbbecsdk import (  # type: ignore
        Config,
        Context,
        OBFormat,
        OBError,
        OBSensorType,
        Pipeline,
        RecordDevice,
        get_version,
    )
except ImportError as exc:  # pragma: no cover - import-time diagnostic
    print("ERROR: pyorbbecsdk2 is not installed or is not importable.")
    print("Install with: pip install pyorbbecsdk2")
    raise


# ---------------------------------------------------------------------------
# Fixed research acquisition settings
# ---------------------------------------------------------------------------
DEPTH_WIDTH = 1600
DEPTH_HEIGHT = 1200
DEPTH_FPS = 30
DEPTH_FORMAT = OBFormat.Y16

COLOR_WIDTH = 1920
COLOR_HEIGHT = 1080
COLOR_FPS = 30
COLOR_FORMAT = OBFormat.YUYV

DEPTH_WORK_MODE = "High Resolution"
DEFAULT_DURATION_S = 2.0
DEFAULT_OUTPUT_DIR = "astra2_recordings"
DEFAULT_PREFIX = "stair"

WINDOW_NAME = "Astra 2 Custom Capture Viewer"
VIEW_W = 1400
VIEW_H = 820

# Preview depth limits only. RAW depth is never clipped in the .bag.
PREVIEW_MIN_M = 0.30
PREVIEW_MAX_M = 8.00


# ---------------------------------------------------------------------------
# Small compatibility helpers
# ---------------------------------------------------------------------------
def enum_text(value: Any) -> str:
    """Best-effort conversion of pybind11 enums to readable text."""
    try:
        return value.name  # some enum wrappers
    except Exception:
        pass
    return str(value)


def safe_scalar(value: Any) -> Any:
    """Convert common pybind/numpy scalar values into JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return enum_text(value)


def profile_attr(profile: Any, method: str, default: Any = None) -> Any:
    try:
        return getattr(profile, method)()
    except Exception:
        return default


def profile_dict(profile: Any) -> dict[str, Any]:
    return {
        "width": safe_scalar(profile_attr(profile, "get_width")),
        "height": safe_scalar(profile_attr(profile, "get_height")),
        "fps": safe_scalar(profile_attr(profile, "get_fps")),
        "format": enum_text(profile_attr(profile, "get_format")),
        "type": enum_text(profile_attr(profile, "get_type")),
    }


def get_fps(profile: Any) -> Optional[int]:
    value = profile_attr(profile, "get_fps")
    try:
        return int(value)
    except Exception:
        return None


def get_dims(profile: Any) -> tuple[Optional[int], Optional[int]]:
    try:
        return int(profile.get_width()), int(profile.get_height())
    except Exception:
        return None, None


def list_profiles(pipeline: Pipeline, sensor_type: Any) -> list[Any]:
    profiles = pipeline.get_stream_profile_list(sensor_type)
    result = []
    for i in range(len(profiles)):
        try:
            result.append(profiles[i])
        except Exception:
            # Some SDK versions expose get_stream_profile_by_index.
            result.append(profiles.get_stream_profile_by_index(i))
    return result


def format_matches(actual: Any, expected: Any) -> bool:
    try:
        return actual == expected
    except Exception:
        return enum_text(actual).lower() == enum_text(expected).lower()


def find_exact_video_profile(
    pipeline: Pipeline,
    sensor_type: Any,
    width: int,
    height: int,
    fps: int,
    pixel_format: Any,
    label: str,
) -> Any:
    """Find an EXACT requested profile; never silently substitute another profile."""
    profiles = list_profiles(pipeline, sensor_type)
    matches = []
    for profile in profiles:
        pw, ph = get_dims(profile)
        pfps = get_fps(profile)
        pfmt = profile_attr(profile, "get_format")
        if (
            pw == width
            and ph == height
            and pfps == fps
            and format_matches(pfmt, pixel_format)
        ):
            matches.append(profile)

    if not matches:
        available = [profile_dict(p) for p in profiles]
        raise RuntimeError(
            f"Required {label} profile is not available exactly.\n"
            f"Requested: {width}x{height}@{fps} {enum_text(pixel_format)}\n"
            f"Available profiles: {json.dumps(available, indent=2, default=str)}"
        )

    return matches[0]


def device_info_dict(device: Any) -> dict[str, Any]:
    info = device.get_device_info()
    fields = [
        ("name", "get_name"),
        ("serial_number", "get_serial_number"),
        ("firmware_version", "get_firmware_version"),
        ("hardware_version", "get_hardware_version"),
        ("asic_name", "get_asic_name"),
        ("vid", "get_vid"),
        ("pid", "get_pid"),
        ("device_type", "get_device_type"),
        ("connection_type", "get_connection_type"),
    ]
    result = {}
    for key, method in fields:
        value = profile_attr(info, method, None)
        if value is not None:
            result[key] = safe_scalar(value)
    try:
        result["sdk_version"] = safe_scalar(get_version())
    except Exception:
        pass
    return result


def get_supported_depth_modes(device: Any) -> list[str]:
    modes = []
    try:
        mode_list = device.get_depth_work_mode_list()
        for i in range(len(mode_list)):
            try:
                item = mode_list[i]
            except Exception:
                try:
                    item = mode_list.get_mode_by_index(i)
                except Exception:
                    item = None
            if item is None:
                continue
            # OBDepthWorkMode wrappers often expose get_name(); otherwise str().
            name = profile_attr(item, "get_name", None)
            modes.append(str(name if name is not None else item))
    except Exception:
        pass
    return modes


def enforce_depth_work_mode(device: Any, target_name: str) -> str:
    """Set High Resolution when possible and verify the resulting mode."""
    current = None
    try:
        current = device.get_current_depth_mode_name()
    except Exception:
        try:
            current = device.get_current_preset_name()
        except Exception:
            current = None

    if current and str(current).strip().casefold() == target_name.casefold():
        return str(current)

    supported = get_supported_depth_modes(device)
    # Prefer exact string, then case-insensitive exact.
    candidate = next((x for x in supported if x == target_name), None)
    if candidate is None:
        candidate = next((x for x in supported if x.casefold() == target_name.casefold()), None)

    # Current pyorbbecsdk exposes set_depth_work_mode(str) as a supported overload.
    if candidate is None:
        # Still attempt the documented string overload when enumeration is weak.
        candidate = target_name

    try:
        device.set_depth_work_mode(candidate)
    except Exception as exc:
        raise RuntimeError(
            f"Could not set depth work mode to '{target_name}'. "
            f"Current='{current}', supported={supported}. Error: {exc}"
        ) from exc

    try:
        verified = str(device.get_current_depth_mode_name())
    except Exception:
        try:
            verified = str(device.get_current_preset_name())
        except Exception:
            verified = str(candidate)

    if verified.strip().casefold() != target_name.casefold():
        raise RuntimeError(
            f"Depth work mode verification failed. Requested '{target_name}', got '{verified}'."
        )
    return verified


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
@dataclass
class CaptureCounters:
    framesets: int = 0
    color_frames: int = 0
    depth_frames: int = 0
    accel_frames: int = 0
    gyro_frames: int = 0


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.imu_lock = threading.Lock()
        self.latest_color: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_color_ts_us: Optional[int] = None
        self.latest_depth_ts_us: Optional[int] = None
        self.latest_accel: Optional[dict[str, Any]] = None
        self.latest_gyro: Optional[dict[str, Any]] = None
        self.counters = CaptureCounters()
        self.capture_in_progress = False
        self.status = "READY"
        self.status_until = 0.0
        self.last_bag: Optional[str] = None
        self.capture_number = 0
        self.failed_captures = 0
        self.running = True

    def set_status(self, message: str, seconds: float = 0.0) -> None:
        with self.lock:
            self.status = message
            self.status_until = time.monotonic() + seconds if seconds > 0 else 0.0


# ---------------------------------------------------------------------------
# Frame conversion / preview helpers
# ---------------------------------------------------------------------------
def color_frame_to_bgr(frame: Any) -> Optional[np.ndarray]:
    if frame is None:
        return None
    try:
        w, h = int(frame.get_width()), int(frame.get_height())
        fmt = frame.get_format()
        data = np.asanyarray(frame.get_data())
        if format_matches(fmt, OBFormat.YUYV):
            yuyv = data.reshape((h, w, 2))
            return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
        if format_matches(fmt, OBFormat.RGB):
            rgb = data.reshape((h, w, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if format_matches(fmt, OBFormat.BGR):
            return data.reshape((h, w, 3)).copy()
        # Fallback for SDK builds with a conversion-friendly format.
        if data.size:
            if data.ndim == 1:
                return None
    except Exception:
        return None
    return None


def depth_frame_to_u16(frame: Any) -> Optional[np.ndarray]:
    if frame is None:
        return None
    try:
        h, w = int(frame.get_height()), int(frame.get_width())
        raw = np.frombuffer(frame.get_data(), dtype=np.uint16)
        return raw.reshape((h, w)).copy()
    except Exception:
        return None


def depth_preview(depth_u16: Optional[np.ndarray], depth_scale: float) -> np.ndarray:
    if depth_u16 is None:
        return np.zeros((360, 640, 3), dtype=np.uint8)
    meters = depth_u16.astype(np.float32) * float(depth_scale)
    # Visualization only: invalid=black; fixed range gives consistent preview.
    m = np.clip(meters, PREVIEW_MIN_M, PREVIEW_MAX_M)
    valid = depth_u16 > 0
    norm = ((m - PREVIEW_MIN_M) / (PREVIEW_MAX_M - PREVIEW_MIN_M) * 255.0).astype(np.uint8)
    norm[~valid] = 0
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def fit_image(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x = (w - nw) // 2
    y = (h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def panel_title(panel: np.ndarray, title: str) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(panel, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def overlay_center_depth(panel: np.ndarray, raw_depth: Optional[np.ndarray], scale: float) -> None:
    if raw_depth is None:
        return
    h, w = raw_depth.shape
    cx, cy = w // 2, h // 2
    r = 10
    y1, y2 = max(0, cy - r), min(h, cy + r + 1)
    x1, x2 = max(0, cx - r), min(w, cx + r + 1)
    roi = raw_depth[y1:y2, x1:x2]
    valid = roi[roi > 0]
    if valid.size:
        z_m = float(np.median(valid)) * float(scale)
        cv2.drawMarker(panel, (panel.shape[1] // 2, panel.shape[0] // 2), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(panel, f"center median: {z_m:.3f} m", (12, panel.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


def imu_panel(state: AppState, w: int, h: int) -> np.ndarray:
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel_title(panel, "IMU / ACQUISITION STATUS")
    with state.imu_lock:
        accel = dict(state.latest_accel) if state.latest_accel else None
        gyro = dict(state.latest_gyro) if state.latest_gyro else None
    lines = []
    if accel:
        lines += [
            f"ACCEL  t={accel.get('timestamp_us', '?')} us",
            f"  X {accel.get('x', 0): .4f}   Y {accel.get('y', 0): .4f}   Z {accel.get('z', 0): .4f} m/s^2",
        ]
    else:
        lines += ["ACCEL  no data yet"]
    if gyro:
        lines += [
            f"GYRO   t={gyro.get('timestamp_us', '?')} us",
            f"  X {gyro.get('x', 0): .4f}   Y {gyro.get('y', 0): .4f}   Z {gyro.get('z', 0): .4f} rad/s",
        ]
    else:
        lines += ["GYRO   no data yet"]

    with state.lock:
        c = state.counters
        status = state.status
        n = state.capture_number
        failed = state.failed_captures
        last_bag = state.last_bag
    lines += [
        "",
        f"Capture state: {status}",
        f"Samples saved: {n}   Failed: {failed}",
        f"Framesets: {c.framesets}   RGB: {c.color_frames}   Depth: {c.depth_frames}",
        f"ACCEL: {c.accel_frames}   GYRO: {c.gyro_frames}",
    ]
    if last_bag:
        lines.append(f"Last bag: {Path(last_bag).name}")

    y = 62
    for line in lines:
        cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        y += 30
        if y > h - 10:
            break
    return panel


def status_panel(state: AppState, device: Any, depth_profile: Any, color_profile: Any, depth_mode: str, out_dir: Path,
                 w: int, h: int, duration_s: float) -> np.ndarray:
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel_title(panel, "FIXED RESEARCH CAPTURE CONFIGURATION")

    info = device_info_dict(device)
    lines = [
        f"Device: {info.get('name', 'unknown')}",
        f"Serial: {info.get('serial_number', 'unknown')}",
        f"Firmware: {info.get('firmware_version', 'unknown')}",
        "",
        f"DEPTH   required  {DEPTH_WIDTH}x{DEPTH_HEIGHT} @ {DEPTH_FPS}  Y16",
        f"        actual    {profile_dict(depth_profile)['width']}x{profile_dict(depth_profile)['height']} @ {profile_dict(depth_profile)['fps']}  {profile_dict(depth_profile)['format']}",
        f"COLOR   required  {COLOR_WIDTH}x{COLOR_HEIGHT} @ {COLOR_FPS}  YUYV",
        f"        actual    {profile_dict(color_profile)['width']}x{profile_dict(color_profile)['height']} @ {profile_dict(color_profile)['fps']}  {profile_dict(color_profile)['format']}",
        f"Depth work mode: {depth_mode}",
        "SDK depth filters: NONE (RAW acquisition path)",
        "Frame sync: SDK FrameSet aggregation; no alignment/filtering applied",
        "",
        f"Output: {out_dir.resolve()}",
        f"Burst duration: {duration_s:.2f} s",
        "",
        "SPACE  capture burst",
        "Q/ESC quit",
    ]
    y = 62
    for line in lines:
        cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (238, 238, 238), 1, cv2.LINE_AA)
        y += 26
        if y > h - 12:
            break
    return panel


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
class DepthScaleHolder:
    def __init__(self, value: float = 0.001) -> None:
        self._lock = threading.Lock()
        self._value = float(value)

    def set(self, value: float) -> None:
        with self._lock:
            self._value = float(value)

    def get(self) -> float:
        with self._lock:
            return self._value


class CallbackBundle:
    def __init__(self, state: AppState, depth_scale_getter: DepthScaleHolder) -> None:
        self.state = state
        self.depth_scale_getter = depth_scale_getter

    def camera(self, frameset: Any) -> None:
        if frameset is None:
            return
        color = None
        depth = None
        try:
            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame()
            if color_frame is not None:
                color = color_frame_to_bgr(color_frame)
            if depth_frame is not None:
                depth = depth_frame_to_u16(depth_frame)
                try:
                    scale = float(depth_frame.get_depth_scale())
                    if scale > 0:
                        self.depth_scale_getter.set(scale)
                except Exception:
                    pass
        except Exception:
            return

        with self.state.lock:
            self.state.counters.framesets += 1
            if color is not None:
                self.state.latest_color = color
                self.state.counters.color_frames += 1
            if depth is not None:
                self.state.latest_depth = depth
                self.state.counters.depth_frames += 1
            try:
                if color_frame is not None:
                    self.state.latest_color_ts_us = int(color_frame.get_timestamp_us())
            except Exception:
                pass
            try:
                if depth_frame is not None:
                    self.state.latest_depth_ts_us = int(depth_frame.get_timestamp_us())
            except Exception:
                pass

    def imu(self, frameset: Any) -> None:
        if frameset is None:
            return
        with self.state.imu_lock:
            try:
                accel = frameset.get_accel_frame()
            except Exception:
                accel = None
            try:
                gyro = frameset.get_gyro_frame()
            except Exception:
                gyro = None
            if accel is not None:
                try:
                    self.state.latest_accel = {
                        "timestamp_us": int(accel.get_timestamp_us()),
                        "x": float(accel.get_x()),
                        "y": float(accel.get_y()),
                        "z": float(accel.get_z()),
                    }
                except Exception:
                    pass
                with self.state.lock:
                    self.state.counters.accel_frames += 1
            if gyro is not None:
                try:
                    self.state.latest_gyro = {
                        "timestamp_us": int(gyro.get_timestamp_us()),
                        "x": float(gyro.get_x()),
                        "y": float(gyro.get_y()),
                        "z": float(gyro.get_z()),
                    }
                except Exception:
                    pass
                with self.state.lock:
                    self.state.counters.gyro_frames += 1


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def unique_bag_path(out_dir: Path, prefix: str, index: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return out_dir / f"{prefix}_{index:06d}_{stamp}.bag"


def write_sidecar_json(
    bag_path: Path,
    device: Any,
    depth_profile: Any,
    color_profile: Any,
    depth_mode: str,
    duration_s: float,
    before: CaptureCounters,
    after: CaptureCounters,
    started_ns: int,
    stopped_ns: int,
    reason: str,
) -> Path:
    data = {
        "schema": "astra2_custom_capture_v1",
        "recording": {
            "bag_file": bag_path.name,
            "capture_started_utc": datetime.fromtimestamp(started_ns / 1e9, tz=timezone.utc).isoformat(),
            "capture_stopped_utc": datetime.fromtimestamp(stopped_ns / 1e9, tz=timezone.utc).isoformat(),
            "requested_duration_s": duration_s,
            "actual_wall_duration_s": (stopped_ns - started_ns) / 1e9,
            "termination_reason": reason,
            "raw_recording": True,
        },
        "device": device_info_dict(device),
        "requested_profile": {
            "depth": {"width": DEPTH_WIDTH, "height": DEPTH_HEIGHT, "fps": DEPTH_FPS, "format": enum_text(DEPTH_FORMAT)},
            "color": {"width": COLOR_WIDTH, "height": COLOR_HEIGHT, "fps": COLOR_FPS, "format": enum_text(COLOR_FORMAT)},
            "depth_work_mode": DEPTH_WORK_MODE,
            "sdk_post_processing": False,
        },
        "actual_profile": {
            "depth": profile_dict(depth_profile),
            "color": profile_dict(color_profile),
        },
        "observed_pipeline_counters": {
            "before": before.__dict__.copy(),
            "after": after.__dict__.copy(),
            "delta": {
                key: after.__dict__[key] - before.__dict__[key]
                for key in after.__dict__
            },
        },
        "notes": [
            "The RGB/depth preview is display-only and is not the recorded data.",
            "No SDK depth filtering, alignment, hole filling, or resampling is performed by this application.",
            "ACCEL/GYRO are enabled through a separate SDK IMU pipeline, following Orbbec's current Python recorder example.",
        ],
    }
    json_path = bag_path.with_suffix(".json")
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return json_path


def capture_burst(
    state: AppState,
    device: Any,
    duration_s: float,
    bag_path: Path,
    depth_profile: Any,
    color_profile: Any,
    depth_mode: str,
) -> None:
    with state.lock:
        before = CaptureCounters(**state.counters.__dict__)
        state.capture_in_progress = True
        state.status = "RECORDING"

    bag_path.parent.mkdir(parents=True, exist_ok=True)
    if bag_path.exists():
        bag_path.unlink()

    started_ns = time.time_ns()
    recorder = None
    reason = "completed"
    try:
        # RecordDevice is the SDK-native .bag recorder used by Orbbec's current
        # official Python recorder example.
        recorder = RecordDevice(device, str(bag_path))
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            # Sleep in short intervals so Ctrl+C/Q remains responsive.
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
        raise
    except Exception:
        reason = "exception"
        raise
    finally:
        # Official Orbbec example closes the bag by releasing RecordDevice.
        recorder = None
        gc.collect()
        stopped_ns = time.time_ns()

        with state.lock:
            after = CaptureCounters(**state.counters.__dict__)
            state.capture_in_progress = False

        if bag_path.exists() and bag_path.stat().st_size > 0:
            try:
                write_sidecar_json(
                    bag_path, device, depth_profile, color_profile, depth_mode,
                    duration_s, before, after, started_ns, stopped_ns, reason,
                )
            except Exception as exc:
                print(f"WARNING: failed to write sidecar JSON: {exc}")
            with state.lock:
                state.capture_number += 1
                state.last_bag = str(bag_path)
                state.status = f"SAVED: {bag_path.name}"
                state.status_until = time.monotonic() + 2.5
        else:
            with state.lock:
                state.failed_captures += 1
                state.status = "CAPTURE FAILED (empty/no bag file)"
                state.status_until = time.monotonic() + 3.0


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------
def create_camera_pipeline() -> tuple[Pipeline, Config, Any, Any, Any, str]:
    pipeline = Pipeline()
    device = pipeline.get_device()

    # Set/verify work mode before creating the fixed stream configuration.
    depth_mode = enforce_depth_work_mode(device, DEPTH_WORK_MODE)

    config = Config()
    depth_profile = find_exact_video_profile(
        pipeline, OBSensorType.DEPTH_SENSOR,
        DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_FPS, DEPTH_FORMAT, "depth"
    )
    color_profile = find_exact_video_profile(
        pipeline, OBSensorType.COLOR_SENSOR,
        COLOR_WIDTH, COLOR_HEIGHT, COLOR_FPS, COLOR_FORMAT, "color"
    )

    config.enable_stream(depth_profile)
    config.enable_stream(color_profile)

    # Keep complete Color+Depth FrameSets for the preview. This is aggregation,
    # not geometric RGB-D alignment. The enum is imported dynamically for
    # compatibility with older pyorbbecsdk2 wheels.
    try:
        from pyorbbecsdk import OBFrameAggregateOutputMode  # type: ignore
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    except Exception:
        pass

    # Enable frame sync where supported (Astra 2 supports it). This is not
    # geometric alignment; it only synchronizes Color/Depth delivery.
    try:
        pipeline.enable_frame_sync()
    except Exception:
        pass

    return pipeline, config, device, depth_profile, color_profile, depth_mode


def create_imu_pipeline() -> Optional[Pipeline]:
    try:
        pipeline = Pipeline()
        config = Config()
        config.enable_accel_stream()
        config.enable_gyro_stream()
        return pipeline, config
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    state = AppState()
    print("Starting Astra 2 custom capture viewer...")
    print("No Orbbec Viewer is used by this application.")

    camera_pipeline: Optional[Pipeline] = None
    imu_pipeline: Optional[Pipeline] = None

    try:
        # Device discovery first for a clearer error message.
        ctx = Context()
        devices = ctx.query_devices()
        if devices.get_count() == 0:
            print("ERROR: No Orbbec device found.")
            return 2

        camera_pipeline, config, device, depth_profile, color_profile, depth_mode = create_camera_pipeline()

        # Verify the selected profiles one more time before opening the streams.
        d = profile_dict(depth_profile)
        c = profile_dict(color_profile)
        print("\nSelected fixed profiles:")
        print(json.dumps({"depth": d, "color": c}, indent=2, default=str))
        print(f"Depth work mode: {depth_mode}")

        # Depth scale comes from the actual DepthFrame. Start with the common
        # 1-mm/raw-unit fallback until the first depth callback supplies it.
        depth_scale_holder = DepthScaleHolder(0.001)

        # Callback path is preferred for smooth preview.
        callbacks = CallbackBundle(state, depth_scale_holder)
        camera_pipeline.start(config, callbacks.camera)

        # Wait for an actual depth callback. The callback itself reads the
        # DepthFrame scale, so this avoids mixing callback-mode streaming with
        # wait_for_frames() calls.
        warm_deadline = time.monotonic() + 5.0
        while time.monotonic() < warm_deadline and state.latest_depth is None:
            time.sleep(0.02)

        # Reaching here without depth frames is a hard error for this dataset.
        if state.latest_depth is None:
            raise RuntimeError("Camera started but no depth frame arrived during warm-up.")

        # IMU is optional at runtime; if unsupported, the viewer clearly shows it.
        imu_setup = create_imu_pipeline()
        if imu_setup is not None:
            imu_pipeline, imu_config = imu_setup
            try:
                imu_pipeline.start(imu_config, callbacks.imu)
                print("IMU: ACCEL + GYRO pipeline started.")
            except Exception as exc:
                print(f"WARNING: IMU could not be started: {exc}")
                try:
                    imu_pipeline.stop()
                except Exception:
                    pass
                imu_pipeline = None
        else:
            print("IMU: not available through this SDK/device configuration.")

        # Print device information once for reproducibility.
        print("\nDevice information:")
        print(json.dumps(device_info_dict(device), indent=2, default=str))
        print("\nControls: SPACE=capture burst, Q/ESC=quit")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, VIEW_W, VIEW_H)

        last_render = time.monotonic()
        render_frames = 0
        fps_display = 0.0

        while state.running:
            now = time.monotonic()
            render_frames += 1
            if now - last_render >= 1.0:
                fps_display = render_frames / (now - last_render)
                render_frames = 0
                last_render = now

            with state.lock:
                if state.status_until and now >= state.status_until and not state.capture_in_progress:
                    state.status = "READY"
                    state.status_until = 0.0
                color = None if state.latest_color is None else state.latest_color.copy()
                depth = None if state.latest_depth is None else state.latest_depth.copy()
                capture_busy = state.capture_in_progress
                counters = CaptureCounters(**state.counters.__dict__)
                status = state.status

            current_depth_scale = depth_scale_holder.get()
            dpreview = depth_preview(depth, current_depth_scale)
            rgb_panel = fit_image(color if color is not None else np.zeros((10, 10, 3), np.uint8), 690, 360)
            dep_panel = fit_image(dpreview, 690, 360)
            panel_title(rgb_panel, "RGB 1920×1080 YUYV")
            panel_title(dep_panel, "DEPTH 1600×1200 Y16 (display only)")
            overlay_center_depth(dep_panel, depth, current_depth_scale)

            status_img = status_panel(
                state, device, depth_profile, color_profile, depth_mode,
                out_dir, 690, 390, args.duration,
            )
            imu_img = imu_panel(state, 690, 390)

            # Add a small dynamic status/FPS overlay.
            cv2.putText(rgb_panel, f"Preview FPS: {fps_display:.1f}", (rgb_panel.shape[1] - 180, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(dep_panel, f"Depth scale: {current_depth_scale * 1000.0:.3f} mm/unit",
                        (dep_panel.shape[1] - 250, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(status_img, status, (18, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(imu_img, f"Wall preview: {counters.framesets} framesets", (18, 365),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)

            canvas = np.zeros((800, 1380, 3), dtype=np.uint8)
            canvas[0:360, 0:690] = rgb_panel
            canvas[0:360, 690:1380] = dep_panel
            canvas[390:780, 0:690] = status_img
            canvas[390:780, 690:1380] = imu_img

            if capture_busy:
                cv2.rectangle(canvas, (0, 0), (1379, 779), (255, 255, 255), 6)
                cv2.putText(canvas, "RECORDING...", (560, 395), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                state.running = False
                break

            if key == 32 and not capture_busy:
                with state.lock:
                    idx = state.capture_number + state.failed_captures + 1
                bag_path = unique_bag_path(out_dir, args.prefix, idx)
                # Capture in the GUI thread so we can safely serialize access to
                # the RecordDevice. The live SDK callbacks continue in their own
                # threads; OpenCV preview is blocked only for the short burst.
                try:
                    capture_burst(
                        state, device, args.duration, bag_path,
                        depth_profile, color_profile, depth_mode,
                    )
                    print(f"Saved: {bag_path}")
                except Exception as exc:
                    print(f"CAPTURE ERROR: {exc}")
                    with state.lock:
                        state.failed_captures += 1
                        state.capture_in_progress = False
                        state.status = f"CAPTURE ERROR: {exc}"
                        state.status_until = time.monotonic() + 4.0

        cv2.destroyWindow(WINDOW_NAME)
        return 0

    except OBError as exc:
        print(f"Orbbec SDK error: {exc}")
        return 3
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 4
    finally:
        if imu_pipeline is not None:
            try:
                imu_pipeline.stop()
            except Exception:
                pass
        if camera_pipeline is not None:
            try:
                camera_pipeline.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Astra 2 direct custom viewer and short RAW .bag recorder"
    )
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION_S,
        help=f"RAW .bag burst duration in seconds (default: {DEFAULT_DURATION_S})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--prefix", default=DEFAULT_PREFIX,
        help=f"Filename prefix (default: {DEFAULT_PREFIX})",
    )
    args = parser.parse_args()
    if args.duration <= 0 or args.duration > 30:
        parser.error("--duration must be > 0 and <= 30 seconds")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
