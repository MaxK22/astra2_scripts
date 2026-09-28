#!/usr/bin/env python3
"""
Orbbec Astra 2 - depth calibration & precision validation

NO ORBBEC VIEWER IS REQUIRED.

Purpose
=======
Validate the factory/current Astra 2 depth output using a known-distance,
large, flat, matte target. The script follows the core methodology described
by Orbbec for quantitative depth-camera evaluation:
  - flat planar target
  - independently measured distances
  - post-processing disabled
  - multiple ROIs across the depth image
  - accuracy/RMSE/standard deviation analysis
  - temporal stability testing

Reference:
https://www.orbbec.com/blog/decoding-depth-camera-performance-quantitative-evaluation-of-accuracy-and-precision/

Astra 2 specifications:
https://www.orbbec.com/products/structured-light-camera/astra-2/

YOUR FINAL DATASET CONFIGURATION
================================
DepthWorkMode:        High Resolution
Synchronization:      Standalone
Timed Sync:           ON
Depth:                1600 x 1200 @ 15 FPS, Y16
Depth engine:         Hardware
Depth post-processing: OFF

IMPORTANT
=========
1. This script does NOT modify factory calibration.
2. It measures the current device output and produces a candidate depth-scale
   correction only as a diagnostic. Do not apply it automatically.
3. Ground truth distance must be independently measured (laser distance
   meter / calibrated rail / other appropriate reference).
4. For best accuracy validation, the target plane should be perpendicular to
   the depth optical axis and fill a large, central portion of the field of
   view.
5. The script uses RAW Y16 depth frames directly and applies only the depth
   scale supplied by the SDK to convert sensor units to metres.

Outputs
=======
results/
    raw_depth_measurements.csv
    accuracy_by_distance.csv
    spatial_precision_by_distance.csv
    temporal_precision.csv                 (when temporal test is enabled)
    depth_validation_report.json

    accuracy_vs_distance.png
    rmse_vs_distance.png
    spatial_precision_vs_distance.png
    roi_bias_heatmap.png
    depth_error_histogram.png
    temporal_stability.png                  (when temporal test is enabled)
    candidate_depth_scale_fit.png

Dependencies
============
pip install --upgrade pyorbbecsdk2
pip install numpy opencv-python matplotlib

Example
=======
python astra2_depth_validation.py

Or a smaller test:
python astra2_depth_validation.py --distances 0.8 1.0 1.5 2.0 3.0 4.0 5.0

Disable temporal test:
python astra2_depth_validation.py --temporal-distance 0

Notes on the manufacturer reference
====================================
Orbbec reports Astra 2 typical spatial precision <=0.16% at 1 m and <=0.30%
at 2 m, under the stated test conditions (1600x1200 and 81% central ROI with
an >80% reflectivity planar target). Max values are stated as <=0.5% at 1 m
and <=1% at 2 m. These are manufacturer test references, not a guarantee for
staircase surfaces or your exact setup.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyorbbecsdk import Config, Context, OBError, OBFormat, OBSensorType, Pipeline


# -----------------------------------------------------------------------------
# Exact requested dataset profile
# -----------------------------------------------------------------------------
DEPTH_WIDTH = 1600
DEPTH_HEIGHT = 1200
DEPTH_FPS = 15
DEPTH_FORMAT = OBFormat.Y16

# Capture defaults. No images are permanently stored except optional samples.
DEFAULT_DISTANCES_M = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
DEFAULT_FRAMES_PER_DISTANCE = 30
DEFAULT_WARMUP_FRAMES = 30

# Orbbec's temporal example uses 1-second samples for 10 minutes. That is a
# long experiment. The default here is 60 s; use --temporal-seconds 600 for
# a 10-minute test.
DEFAULT_TEMPORAL_SECONDS = 60.0
DEFAULT_TEMPORAL_INTERVAL = 1.0
DEFAULT_TEMPORAL_DISTANCE = 2.0

# Nine ROIs similar in concept to the multi-ROI evaluation described by
# Orbbec. Each ROI occupies 20% of image width/height, centered in a 3x3 grid.
ROI_FRACTION = {
    "TL": (0.05, 0.05, 0.25, 0.25),
    "TC": (0.40, 0.05, 0.60, 0.25),
    "TR": (0.75, 0.05, 0.95, 0.25),
    "ML": (0.05, 0.40, 0.25, 0.60),
    "C":  (0.40, 0.40, 0.60, 0.60),
    "MR": (0.75, 0.40, 0.95, 0.60),
    "BL": (0.05, 0.75, 0.25, 0.95),
    "BC": (0.40, 0.75, 0.60, 0.95),
    "BR": (0.75, 0.75, 0.95, 0.95),
}


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass
class ROIMeasurement:
    distance_m: float
    frame_index: int
    timestamp_ms: float | None
    roi: str
    n_valid: int
    total_pixels: int
    valid_percent: float
    mean_m: float | None
    median_m: float | None
    std_m: float | None
    min_m: float | None
    max_m: float | None
    bias_m: float | None
    abs_error_m: float | None
    rmse_m: float | None


# -----------------------------------------------------------------------------
# JSON / SDK helpers
# -----------------------------------------------------------------------------
def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        arr = np.asarray(value)
        if arr.ndim == 0:
            item = arr.item()
            # pybind11/Orbbec enum objects (for example OBDeviceType) can
            # survive as the scalar object returned by ndarray.item().
            # Convert any non-native scalar to a string so json.dump can
            # always serialize it.
            if isinstance(item, (str, int, float, bool)) or item is None:
                return item
            return str(item)

        # For arrays, recursively convert elements in case an SDK object
        # appears inside an object-dtype array.
        return [json_safe(item) for item in arr.tolist()]
    except Exception:
        return str(value)


def safe_call(obj: Any, method_name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, method_name)()
    except Exception:
        return default


def get_timestamp_ms(frame) -> float | None:
    # Prefer device timestamp; fall back to system timestamp.
    for name in ("get_timestamp", "get_system_timestamp"):
        try:
            return float(getattr(frame, name)())
        except Exception:
            pass
    return None


def get_device_metadata(device) -> dict[str, Any]:
    info = device.get_device_info()
    return {
        "name": safe_call(info, "get_name"),
        "serial_number": safe_call(info, "get_serial_number"),
        "firmware_version": safe_call(info, "get_firmware_version"),
        "hardware_version": safe_call(info, "get_hardware_version"),
        "vid": json_safe(safe_call(info, "get_vid")),
        "pid": json_safe(safe_call(info, "get_pid")),
        "device_type": json_safe(safe_call(info, "get_device_type")),
        "connection_type": json_safe(safe_call(info, "get_connection_type")),
    }


# -----------------------------------------------------------------------------
# Factory/current calibration readout
# -----------------------------------------------------------------------------
def get_factory_depth_calibration(device) -> dict[str, Any] | None:
    """Read the device calibration list directly; no Viewer involved.

    We use the calibration list at device level so that the JSON keeps all
    calibration parameter sets exposed by the camera. Different SDK versions
    can expose the list slightly differently, so this function is deliberately
    defensive.
    """
    try:
        param_list = device.get_calibration_camera_param_list()
        count = int(param_list.get_count())
    except Exception:
        return None

    entries: list[dict[str, Any]] = []

    for idx in range(count):
        try:
            param = param_list.get_camera_param(idx)
        except Exception:
            try:
                param = param_list[idx]
            except Exception:
                continue

        def intrinsic_dict(intr):
            return {
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "cx": float(intr.cx),
                "cy": float(intr.cy),
                "width": int(intr.width),
                "height": int(intr.height),
            }

        def distortion_dict(dist):
            out: dict[str, Any] = {}
            for name in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"):
                if hasattr(dist, name):
                    try:
                        out[name] = float(getattr(dist, name))
                    except Exception:
                        out[name] = json_safe(getattr(dist, name))
            if hasattr(dist, "model"):
                try:
                    out["model"] = str(dist.model)
                except Exception:
                    pass
            return out

        ext = param.transform
        rot = np.asarray(ext.rot, dtype=np.float64).reshape(-1)
        trans = np.asarray(ext.transform, dtype=np.float64).reshape(-1)

        entry = {
            "index": idx,
            "DEPTH": {
                "intrinsic": intrinsic_dict(param.depth_intrinsic),
                "distortion": distortion_dict(param.depth_distortion),
            },
            "RGB": {
                "intrinsic": intrinsic_dict(param.rgb_intrinsic),
                "distortion": distortion_dict(param.rgb_distortion),
            },
            "DEPTH_TO_RGB": {
                "R": rot.reshape(3, 3).tolist() if rot.size == 9 else rot.tolist(),
                "t": trans.tolist(),
                "t_units": "mm",
            },
        }
        entries.append(entry)

    return {
        "count": len(entries),
        "entries": entries,
    }


# -----------------------------------------------------------------------------
# Device/profile setup
# -----------------------------------------------------------------------------
def open_astra_depth(device_index: int = 0):
    ctx = Context()
    devices = ctx.query_devices()
    count = int(devices.get_count())

    if count == 0:
        raise RuntimeError("No Orbbec device detected. Connect Astra 2.")
    if not (0 <= device_index < count):
        raise RuntimeError(
            f"Invalid device index {device_index}; {count} device(s) detected."
        )

    device = devices.get_device_by_index(device_index)
    metadata = get_device_metadata(device)

    pipeline = Pipeline(device)
    config = Config()

    # The user requested High Resolution. This mode must be selected before
    # starting the streams. On devices/SDK versions where the mode API is
    # exposed, select it explicitly; otherwise the requested 1600x1200 profile
    # below is used and the actual profile is printed/verified.
    try:
        modes = device.get_depth_work_mode_list()
        selected_mode = None
        for i in range(int(modes.get_count())):
            try:
                mode = modes.get_depth_work_mode_by_index(i)
            except Exception:
                mode = modes[i]
            name = str(getattr(mode, "name", mode))
            if "high" in name.lower() and "resolution" in name.lower():
                selected_mode = name
                break
        if selected_mode:
            device.set_depth_work_mode(selected_mode)
    except Exception:
        # Do not fail solely because this SDK build does not expose the mode
        # switching API. The exact stream profile is still verified below.
        selected_mode = None

    profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

    try:
        depth_profile = profiles.get_video_stream_profile(
            DEPTH_WIDTH,
            DEPTH_HEIGHT,
            DEPTH_FORMAT,
            DEPTH_FPS,
        )
    except Exception as exc:
        raise RuntimeError(
            "The exact requested depth profile was not available:\n"
            f"  {DEPTH_WIDTH}x{DEPTH_HEIGHT} @ {DEPTH_FPS} {DEPTH_FORMAT}\n"
            f"SDK error: {exc}\n\n"
            "This is important: do not silently substitute another profile for "
            "your validation experiment."
        ) from exc

    config.enable_stream(depth_profile)

    return ctx, device, pipeline, config, depth_profile, metadata, selected_mode


def depth_profile_dict(profile) -> dict[str, Any]:
    return {
        "width": int(profile.get_width()),
        "height": int(profile.get_height()),
        "fps": int(profile.get_fps()),
        "format": str(profile.get_format()),
    }


# -----------------------------------------------------------------------------
# Depth frame handling
# -----------------------------------------------------------------------------
def depth_frame_to_meters(depth_frame) -> tuple[np.ndarray, float]:
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())

    raw = np.frombuffer(
        depth_frame.get_data(),
        dtype=np.uint16,
    ).reshape(height, width)

    scale = float(depth_frame.get_depth_scale())
    depth_m = raw.astype(np.float64) * scale

    # Zero is conventionally used for invalid depth by the SDK/file formats.
    depth_m[(raw == 0)] = np.nan

    return depth_m, scale


def roi_bounds(depth_m: np.ndarray, roi_name: str) -> tuple[int, int, int, int]:
    h, w = depth_m.shape
    x0f, y0f, x1f, y1f = ROI_FRACTION[roi_name]
    x0 = int(round(x0f * w))
    y0 = int(round(y0f * h))
    x1 = int(round(x1f * w))
    y1 = int(round(y1f * h))
    return x0, y0, x1, y1


def roi_values(depth_m: np.ndarray, roi_name: str) -> tuple[np.ndarray, int]:
    x0, y0, x1, y1 = roi_bounds(depth_m, roi_name)
    region = depth_m[y0:y1, x0:x1]
    values = region[np.isfinite(region) & (region > 0)]
    return values, int(region.size)


def roi_stats(
    depth_m: np.ndarray,
    roi_name: str,
    reference_m: float,
) -> dict[str, Any]:
    values, total = roi_values(depth_m, roi_name)

    if len(values) == 0:
        return {
            "n_valid": 0,
            "total_pixels": total,
            "valid_percent": 0.0,
            "mean_m": None,
            "median_m": None,
            "std_m": None,
            "min_m": None,
            "max_m": None,
            "bias_m": None,
            "abs_error_m": None,
            "rmse_m": None,
        }

    errors = values - reference_m

    return {
        "n_valid": int(len(values)),
        "total_pixels": total,
        "valid_percent": 100.0 * len(values) / total,
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "std_m": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min_m": float(np.min(values)),
        "max_m": float(np.max(values)),
        "bias_m": float(np.mean(errors)),
        "abs_error_m": float(np.mean(np.abs(errors))),
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
    }


# -----------------------------------------------------------------------------
# Accuracy test
# -----------------------------------------------------------------------------
def run_accuracy_test(
    pipeline: Pipeline,
    distances: list[float],
    frames_per_distance: int,
    warmup_frames: int,
    output_dir: Path,
    save_samples: bool,
):
    all_measurements: list[ROIMeasurement] = []
    depth_scale_used: float | None = None

    for distance in distances:
        print()
        print("=" * 78)
        print(f"REFERENCE DISTANCE = {distance:.3f} m")
        print("=" * 78)
        print(
            "Position a LARGE, FLAT, MATTE target at this independently "
            "measured distance."
        )
        print(
            "Best practice: target plane perpendicular to the depth optical "
            "axis; target fills a substantial part of the image."
        )

        input(
            "Press ENTER when the target is positioned and the camera/target "
            "are stationary..."
        )

        # Allow projector/exposure and setup vibrations to settle.
        for _ in range(warmup_frames):
            pipeline.wait_for_frames(1000)

        saved_sample = False

        for frame_index in range(frames_per_distance):
            frames = pipeline.wait_for_frames(2000)
            if frames is None:
                continue

            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue

            depth_m, scale = depth_frame_to_meters(depth_frame)
            depth_scale_used = scale
            timestamp_ms = get_timestamp_ms(depth_frame)

            if save_samples and not saved_sample:
                raw = np.frombuffer(
                    depth_frame.get_data(),
                    dtype=np.uint16,
                ).reshape(
                    int(depth_frame.get_height()),
                    int(depth_frame.get_width()),
                )
                sample_path = (
                    output_dir
                    / f"sample_depth_{distance:.3f}m.png"
                )
                # Lossless 16-bit storage; raw values preserved.
                if not cv2.imwrite(str(sample_path), raw):
                    print(f"WARNING: failed to save {sample_path}")
                saved_sample = True

            for roi_name in ROI_FRACTION:
                stats = roi_stats(
                    depth_m,
                    roi_name,
                    distance,
                )

                all_measurements.append(
                    ROIMeasurement(
                        distance_m=distance,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        roi=roi_name,
                        n_valid=stats["n_valid"],
                        total_pixels=stats["total_pixels"],
                        valid_percent=stats["valid_percent"],
                        mean_m=stats["mean_m"],
                        median_m=stats["median_m"],
                        std_m=stats["std_m"],
                        min_m=stats["min_m"],
                        max_m=stats["max_m"],
                        bias_m=stats["bias_m"],
                        abs_error_m=stats["abs_error_m"],
                        rmse_m=stats["rmse_m"],
                    )
                )

        captured = [
            m for m in all_measurements
            if math.isclose(m.distance_m, distance, abs_tol=1e-12)
        ]
        print(
            f"Collected {len(captured)} ROI/frame measurements "
            f"({frames_per_distance} requested frames x {len(ROI_FRACTION)} ROIs)."
        )

    return all_measurements, depth_scale_used


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------
def aggregate_accuracy(
    measurements: list[ROIMeasurement],
) -> list[dict[str, Any]]:
    rows = []

    for distance in sorted({m.distance_m for m in measurements}):
        for roi_name in ROI_FRACTION:
            items = [
                m for m in measurements
                if math.isclose(m.distance_m, distance, abs_tol=1e-12)
                and m.roi == roi_name
                and m.mean_m is not None
            ]

            if not items:
                continue

            depth_means = np.asarray(
                [m.mean_m for m in items],
                dtype=np.float64,
            )
            errors = depth_means - distance
            within_roi_std = np.asarray(
                [m.std_m for m in items],
                dtype=np.float64,
            )
            valid = np.asarray(
                [m.valid_percent for m in items],
                dtype=np.float64,
            )

            rows.append(
                {
                    "distance_m": distance,
                    "roi": roi_name,
                    "frames": int(len(items)),
                    "mean_depth_m": float(np.mean(depth_means)),
                    "median_depth_m": float(np.median(depth_means)),
                    "bias_m": float(np.mean(errors)),
                    "absolute_bias_m": float(abs(np.mean(errors))),
                    "mean_absolute_error_m": float(np.mean(np.abs(errors))),
                    "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
                    "temporal_std_of_roi_mean_m": float(
                        np.std(depth_means, ddof=1)
                    ) if len(depth_means) > 1 else 0.0,
                    "mean_spatial_pixel_std_m": float(np.mean(within_roi_std)),
                    "mean_valid_percent": float(np.mean(valid)),
                }
            )

    return rows


def aggregate_spatial_precision(
    accuracy_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Orbbec describes spatial precision using mean STD within selected ROIs
    plotted against distance. Here we compute exactly that concept:

        mean of per-ROI pixel STD values

    We also report the spatial spread of the ROI mean depths, which is a useful
    second indicator of plane non-uniformity/lens/depth-field effects.
    """
    rows = []

    for distance in sorted({r["distance_m"] for r in accuracy_rows}):
        items = [
            r for r in accuracy_rows
            if math.isclose(r["distance_m"], distance, abs_tol=1e-12)
        ]

        spatial_stds = np.asarray(
            [r["mean_spatial_pixel_std_m"] for r in items],
            dtype=np.float64,
        )
        roi_means = np.asarray(
            [r["mean_depth_m"] for r in items],
            dtype=np.float64,
        )

        rows.append(
            {
                "distance_m": distance,
                "mean_roi_pixel_std_m": float(np.mean(spatial_stds)),
                "roi_pixel_std_std_m": float(
                    np.std(spatial_stds, ddof=1)
                ) if len(spatial_stds) > 1 else 0.0,
                "roi_mean_depth_spread_std_m": float(
                    np.std(roi_means, ddof=1)
                ) if len(roi_means) > 1 else 0.0,
                "roi_mean_depth_range_m": float(np.ptp(roi_means)),
            }
        )

    return rows


# -----------------------------------------------------------------------------
# Candidate depth-scale calibration
# -----------------------------------------------------------------------------
def fit_depth_scale_correction(
    accuracy_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Fit a simple affine model using central ROI distance measurements:

        reference_distance = a * measured_depth + b

    This is ONLY a diagnostic candidate correction. It must be validated on
    independent distances before use.
    """
    subset = [
        r for r in accuracy_rows
        if r["roi"] == "C" and r["mean_depth_m"] is not None
    ]

    if len(subset) < 2:
        return {
            "available": False,
            "reason": "At least two central-ROI distances are required."
        }

    x = np.asarray(
        [r["mean_depth_m"] for r in subset],
        dtype=np.float64,
    )
    y = np.asarray(
        [r["distance_m"] for r in subset],
        dtype=np.float64,
    )

    A = np.column_stack([x, np.ones_like(x)])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a = float(coeffs[0])
    b = float(coeffs[1])

    y_hat = a * x + b
    residuals = y_hat - y

    return {
        "available": True,
        "model": "reference_m = scale_factor * measured_m + offset_m",
        "scale_factor": a,
        "offset_m": b,
        "offset_mm": b * 1000.0,
        "residual_rmse_m": float(np.sqrt(np.mean(residuals ** 2))),
        "residual_rmse_mm": float(np.sqrt(np.mean(residuals ** 2)) * 1000.0),
        "points": [
            {
                "measured_m": float(measured),
                "reference_m": float(reference),
                "corrected_m": float(corrected),
                "residual_m": float(residual),
            }
            for measured, reference, corrected, residual in zip(
                x, y, y_hat, residuals
            )
        ],
        "warning": (
            "Diagnostic only. Do not overwrite the factory calibration or "
            "depth scale unless the correction is confirmed on independent "
            "held-out distances and repeated measurement sessions."
        ),
    }


# -----------------------------------------------------------------------------
# Temporal precision
# -----------------------------------------------------------------------------
def run_temporal_test(
    pipeline: Pipeline,
    distance_m: float,
    duration_s: float,
    interval_s: float,
    warmup_frames: int,
) -> dict[str, Any]:
    print()
    print("=" * 78)
    print("TEMPORAL PRECISION TEST")
    print("=" * 78)
    print(
        f"Place the same flat target at {distance_m:.3f} m and keep camera and "
        "target completely stationary."
    )
    input("Press ENTER to start the temporal test...")

    for _ in range(warmup_frames):
        pipeline.wait_for_frames(1000)

    start = time.monotonic()
    next_sample = start

    values = []
    times = []
    depth_scale = None

    while time.monotonic() - start < duration_s:
        remaining = next_sample - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

        frames = pipeline.wait_for_frames(2000)
        next_sample += interval_s

        if frames is None:
            continue

        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            continue

        depth_m, scale = depth_frame_to_meters(depth_frame)
        depth_scale = scale
        vals, _ = roi_values(depth_m, "C")

        if len(vals) == 0:
            continue

        values.append(float(np.mean(vals)))
        ts = get_timestamp_ms(depth_frame)
        times.append(
            (ts / 1000.0) if ts is not None else (time.monotonic() - start)
        )

    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {
            "enabled": True,
            "distance_m": distance_m,
            "samples": 0,
            "error": "No valid central-ROI samples were obtained.",
            "depth_scale": depth_scale,
        }

    errors = arr - distance_m

    result = {
        "enabled": True,
        "distance_m": distance_m,
        "roi": "C",
        "duration_requested_s": duration_s,
        "interval_requested_s": interval_s,
        "samples": int(len(arr)),
        "mean_m": float(np.mean(arr)),
        "bias_m": float(np.mean(errors)),
        "std_m": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "peak_to_peak_m": float(np.ptp(arr)),
        "depth_scale": depth_scale,
        "values_m": arr.tolist(),
        "times_s": (
            (np.asarray(times) - times[0]).tolist()
            if times else []
        ),
    }

    return result


# -----------------------------------------------------------------------------
# CSV + plots
# -----------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    keys = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: (
                    json.dumps(json_safe(v), ensure_ascii=False)
                    if isinstance(v, (dict, list))
                    else v
                )
                for k, v in row.items()
            })


def plot_accuracy(
    accuracy_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    subset = sorted(
        [r for r in accuracy_rows if r["roi"] == "C"],
        key=lambda r: r["distance_m"],
    )

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    if subset:
        x = np.asarray([r["distance_m"] for r in subset])
        y = np.asarray([r["bias_m"] * 1000.0 for r in subset])
        e = np.asarray([
            r["temporal_std_of_roi_mean_m"] * 1000.0
            for r in subset
        ])

        ax.errorbar(
            x,
            y,
            yerr=e,
            marker="o",
            capsize=3,
            label="Central ROI: mean signed error ± temporal SD",
        )

    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Reference distance (m)")
    ax.set_ylabel("Depth error (mm)")
    ax.set_title("Astra 2 depth accuracy vs distance")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_relative_accuracy(
    accuracy_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot central-ROI mean absolute error as percent of reference distance."""
    subset = sorted(
        [r for r in accuracy_rows if r["roi"] == "C"],
        key=lambda r: r["distance_m"],
    )

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    if subset:
        x = np.asarray([r["distance_m"] for r in subset])
        y = np.asarray([
            100.0 * r["mean_absolute_error_m"] / r["distance_m"]
            for r in subset
        ])
        ax.plot(x, y, marker="o", label="Measured central ROI")

        # Manufacturer reference points are plotted only as reference points,
        # not as guarantees for the user's exact target/material/setup.
        ax.scatter([1.0], [0.16], marker="x", label="Orbbec typical @ 1 m")
        ax.scatter([2.0], [0.30], marker="x", label="Orbbec typical @ 2 m")
        ax.scatter([1.0], [0.50], marker="+", label="Orbbec max @ 1 m")
        ax.scatter([2.0], [1.00], marker="+", label="Orbbec max @ 2 m")

    ax.set_xlabel("Reference distance (m)")
    ax.set_ylabel("Mean absolute error (% of distance)")
    ax.set_title("Astra 2 relative depth accuracy vs distance")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rmse(
    accuracy_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    subset = sorted(
        [r for r in accuracy_rows if r["roi"] == "C"],
        key=lambda r: r["distance_m"],
    )

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    if subset:
        x = [r["distance_m"] for r in subset]
        y = [r["rmse_m"] * 1000.0 for r in subset]
        ax.plot(x, y, marker="o")

    ax.set_xlabel("Reference distance (m)")
    ax.set_ylabel("RMSE (mm)")
    ax.set_title("Astra 2 depth RMSE vs distance")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_precision(
    spatial_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    rows = sorted(spatial_rows, key=lambda r: r["distance_m"])

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    if rows:
        x = [r["distance_m"] for r in rows]
        y = [r["mean_roi_pixel_std_m"] * 1000.0 for r in rows]
        ax.plot(x, y, marker="o")

    ax.set_xlabel("Reference distance (m)")
    ax.set_ylabel("Mean within-ROI depth STD (mm)")
    ax.set_title("Astra 2 spatial precision vs distance")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roi_heatmap(
    accuracy_rows: list[dict[str, Any]],
    distance_m: float,
    output_path: Path,
) -> None:
    distances = sorted({r["distance_m"] for r in accuracy_rows})
    if not distances:
        return

    selected = min(distances, key=lambda x: abs(x - distance_m))
    selected_rows = {
        r["roi"]: r
        for r in accuracy_rows
        if math.isclose(r["distance_m"], selected, abs_tol=1e-12)
    }

    order = [
        ["TL", "TC", "TR"],
        ["ML", "C", "MR"],
        ["BL", "BC", "BR"],
    ]

    values = np.full((3, 3), np.nan, dtype=np.float64)

    for iy, row in enumerate(order):
        for ix, roi in enumerate(row):
            if roi in selected_rows:
                values[iy, ix] = selected_rows[roi]["bias_m"] * 1000.0

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    im = ax.imshow(
        values,
        interpolation="nearest",
        aspect="equal",
        cmap="coolwarm",
    )

    ax.set_xticks([0, 1, 2], ["Left", "Centre", "Right"])
    ax.set_yticks([0, 1, 2], ["Top", "Middle", "Bottom"])

    finite = values[np.isfinite(values)]
    vmax = float(np.max(np.abs(finite))) if len(finite) else 1.0
    im.set_clim(-vmax, vmax if vmax > 0 else 1.0)

    for iy in range(3):
        for ix in range(3):
            if np.isfinite(values[iy, ix]):
                ax.text(
                    ix,
                    iy,
                    f"{values[iy, ix]:+.1f}",
                    ha="center",
                    va="center",
                )

    ax.set_title(
        f"Spatial depth-error map at {selected:.2f} m (mm)"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean depth error (mm)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_error_histogram(
    measurements: list[ROIMeasurement],
    distance_m: float,
    output_path: Path,
) -> None:
    """True frame-level central-ROI depth-error histogram."""
    candidates = [
        m for m in measurements
        if m.roi == "C"
        and math.isclose(m.distance_m, distance_m, abs_tol=1e-12)
        and m.mean_m is not None
    ]

    if not candidates:
        distances = sorted({m.distance_m for m in measurements})
        if not distances:
            return
        selected = min(distances, key=lambda d: abs(d - distance_m))
        candidates = [
            m for m in measurements
            if m.roi == "C"
            and math.isclose(m.distance_m, selected, abs_tol=1e-12)
            and m.mean_m is not None
        ]
        distance_m = selected

    errors_mm = np.asarray(
        [(m.mean_m - distance_m) * 1000.0 for m in candidates],
        dtype=np.float64,
    )

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    if len(errors_mm):
        bins = min(20, max(8, int(np.sqrt(len(errors_mm)))))
        ax.hist(errors_mm, bins=bins)
        ax.axvline(0.0, linewidth=1, label="Zero error")
        ax.axvline(
            float(np.mean(errors_mm)),
            linewidth=1,
            label=f"Mean = {np.mean(errors_mm):+.2f} mm",
        )
        ax.legend()

    ax.set_xlabel("Central-ROI depth error (mm)")
    ax.set_ylabel("Number of frames")
    ax.set_title(
        f"Astra 2 central-ROI depth-error histogram at {distance_m:.2f} m"
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_temporal(
    temporal: dict[str, Any],
    output_path: Path,
) -> None:
    values = np.asarray(temporal.get("values_m", []), dtype=np.float64)
    times = np.asarray(temporal.get("times_s", []), dtype=np.float64)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    if len(values):
        t = times if len(times) == len(values) else np.arange(len(values))
        ax.plot(t, values, marker=".", linewidth=1)
        ax.axhline(
            temporal["distance_m"],
            linewidth=1,
            label="Reference distance",
        )
        ax.legend()

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Measured central-ROI depth (m)")
    ax.set_title("Astra 2 temporal depth stability")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scale_fit(
    fit: dict[str, Any],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    if fit.get("available"):
        measured = np.asarray(
            [p["measured_m"] for p in fit["points"]],
            dtype=np.float64,
        )
        reference = np.asarray(
            [p["reference_m"] for p in fit["points"]],
            dtype=np.float64,
        )

        xline = np.linspace(
            float(np.min(measured)),
            float(np.max(measured)),
            100,
        )

        ax.scatter(
            measured,
            reference,
            label="Measured central-ROI means",
        )

        ax.plot(
            xline,
            xline,
            linewidth=1,
            label="Ideal y=x",
        )

        ax.plot(
            xline,
            fit["scale_factor"] * xline + fit["offset_m"],
            linewidth=1,
            label=(
                f"Fit: y={fit['scale_factor']:.7f}x"
                f"{fit['offset_m']:+.4f}"
            ),
        )

    ax.set_xlabel("Measured depth (m)")
    ax.set_ylabel("Reference distance (m)")
    ax.set_title("Diagnostic Astra 2 depth-scale fit")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Astra 2 depth accuracy and precision."
    )

    parser.add_argument(
        "--output-dir",
        default="depth_validation_results",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        type=float,
        default=DEFAULT_DISTANCES_M,
        help="Known target-plane distances in metres.",
    )
    parser.add_argument(
        "--frames-per-distance",
        type=int,
        default=DEFAULT_FRAMES_PER_DISTANCE,
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=DEFAULT_WARMUP_FRAMES,
    )
    parser.add_argument(
        "--temporal-distance",
        type=float,
        default=DEFAULT_TEMPORAL_DISTANCE,
        help="Distance for temporal test. Use 0 to disable.",
    )
    parser.add_argument(
        "--temporal-seconds",
        type=float,
        default=DEFAULT_TEMPORAL_SECONDS,
    )
    parser.add_argument(
        "--temporal-interval",
        type=float,
        default=DEFAULT_TEMPORAL_INTERVAL,
    )
    parser.add_argument(
        "--heatmap-distance",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Save one raw 16-bit depth image per tested distance.",
    )

    args = parser.parse_args()

    if args.frames_per_distance < 2:
        print("ERROR: --frames-per-distance must be >= 2.")
        return 1

    if any(d <= 0 for d in args.distances):
        print("ERROR: all --distances must be > 0.")
        return 1

    if args.temporal_seconds <= 0 and args.temporal_distance != 0:
        print("ERROR: --temporal-seconds must be > 0 unless temporal test is disabled.")
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = None
    pipeline = None

    try:
        (
            ctx,
            device,
            pipeline,
            config,
            depth_profile,
            device_meta,
            selected_mode,
        ) = open_astra_depth(args.device_index)

        print()
        print("=" * 78)
        print("ORBBEC ASTRA 2 DEPTH CALIBRATION / PRECISION VALIDATION")
        print("=" * 78)
        print("Viewer used: NO")
        print()
        print("Camera:")
        print(json.dumps(device_meta, indent=2))
        print()
        print("Requested depth configuration:")
        print("  Work mode:        High Resolution")
        print("  Synchronization:  Standalone")
        print("  Timed Sync:       ON")
        print("  Depth:             1600 x 1200 @ 15 FPS Y16")
        print("  Depth engine:      Hardware")
        print("  Post-processing:   OFF")
        print()
        print("Actual selected depth profile:")
        print(json.dumps(depth_profile_dict(depth_profile), indent=2))
        if selected_mode:
            print(f"Depth work mode selected by SDK: {selected_mode}")
        else:
            print(
                "Depth work mode was not changed by this script; the exact "
                "requested 1600x1200@15 Y16 profile is still enforced."
            )

        calibration = get_factory_depth_calibration(device)

        if calibration is not None:
            print()
            print(
                f"Factory/current calibration sets exposed by device: "
                f"{calibration['count']}"
            )
        else:
            print()
            print("WARNING: Could not retrieve the device calibration list.")

        pipeline.start(config)

        measurements, depth_scale = run_accuracy_test(
            pipeline,
            args.distances,
            args.frames_per_distance,
            args.warmup_frames,
            output_dir,
            args.save_samples,
        )

        raw_rows = [asdict(m) for m in measurements]
        write_csv(
            output_dir / "raw_depth_measurements.csv",
            raw_rows,
        )

        accuracy_rows = aggregate_accuracy(measurements)
        write_csv(
            output_dir / "accuracy_by_distance.csv",
            accuracy_rows,
        )

        spatial_rows = aggregate_spatial_precision(accuracy_rows)
        write_csv(
            output_dir / "spatial_precision_by_distance.csv",
            spatial_rows,
        )

        # -------------------------------------------------------------
        # Candidate depth scale model.
        # -------------------------------------------------------------
        scale_fit = fit_depth_scale_correction(accuracy_rows)
        plot_scale_fit(
            scale_fit,
            output_dir / "candidate_depth_scale_fit.png",
        )

        # -------------------------------------------------------------
        # Temporal test.
        # -------------------------------------------------------------
        temporal = None
        if args.temporal_distance != 0:
            temporal = run_temporal_test(
                pipeline,
                args.temporal_distance,
                args.temporal_seconds,
                args.temporal_interval,
                args.warmup_frames,
            )

            temporal_csv = {
                k: v for k, v in temporal.items()
                if k not in {"values_m", "times_s"}
            }
            write_csv(
                output_dir / "temporal_precision.csv",
                [temporal_csv],
            )
            plot_temporal(
                temporal,
                output_dir / "temporal_stability.png",
            )

        # -------------------------------------------------------------
        # Graphs.
        # -------------------------------------------------------------
        plot_accuracy(
            accuracy_rows,
            output_dir / "accuracy_vs_distance.png",
        )

        plot_relative_accuracy(
            accuracy_rows,
            output_dir / "relative_accuracy_vs_distance.png",
        )

        plot_rmse(
            accuracy_rows,
            output_dir / "rmse_vs_distance.png",
        )

        plot_spatial_precision(
            spatial_rows,
            output_dir / "spatial_precision_vs_distance.png",
        )

        plot_roi_heatmap(
            accuracy_rows,
            args.heatmap_distance,
            output_dir / "roi_bias_heatmap.png",
        )

        plot_error_histogram(
            measurements,
            args.heatmap_distance,
            output_dir / "depth_error_histogram.png",
        )

        # -------------------------------------------------------------
        # Final JSON report.
        # -------------------------------------------------------------
        now_local = datetime.now().astimezone()
        now_utc = datetime.now(timezone.utc)

        report = {
            "schema_version": "1.0",
            "viewer_used": False,
            "capture": {
                "date": now_local.strftime("%Y-%m-%d"),
                "time": now_local.strftime("%H:%M:%S.%f"),
                "datetime_local": now_local.isoformat(),
                "datetime_utc": now_utc.isoformat(),
            },
            "camera": device_meta,
            "configuration": {
                "depth_work_mode": "High Resolution",
                "synchronization_mode": "Standalone",
                "timed_sync": True,
                "depth": depth_profile_dict(depth_profile),
                "depth_engine": "hardware",
                "depth_post_processing": False,
                "recording_mode": "RAW (not used by this live validation script)",
                "actual_depth_scale": depth_scale,
            },
            "target_protocol": {
                "target_description": (
                    "large flat rigid matte planar target, preferably perpendicular "
                    "to the depth optical axis"
                ),
                "independent_distance_reference_required": True,
                "tested_distances_m": args.distances,
                "frames_per_distance": args.frames_per_distance,
                "warmup_frames": args.warmup_frames,
                "rois": ROI_FRACTION,
            },
            "manufacturer_reference": {
                "typical_precision_percent_at_1m": 0.16,
                "typical_precision_percent_at_2m": 0.30,
                "max_precision_percent_at_1m": 0.50,
                "max_precision_percent_at_2m": 1.00,
                "conditions": (
                    "1600x1200, 81% central ROI, high-reflectivity planar target"
                ),
                "note": (
                    "These manufacturer figures are reference values under their "
                    "test conditions; do not treat them as guaranteed performance "
                    "for staircase materials or your exact setup."
                ),
            },
            "factory_or_sdk_calibration": calibration,
            "accuracy_by_distance": accuracy_rows,
            "spatial_precision_by_distance": spatial_rows,
            "candidate_depth_scale_fit": scale_fit,
            "temporal_precision": temporal,
            "interpretation_guidance": {
                "accuracy": (
                    "Use signed bias to detect systematic over/under-estimation and "
                    "RMSE/absolute error to quantify magnitude."
                ),
                "spatial_precision": (
                    "Low within-ROI STD indicates uniform/stable depth on the planar "
                    "target. Large differences among ROI means can indicate spatial "
                    "non-uniformity, optical/depth-model effects, or target geometry "
                    "issues."
                ),
                "temporal_precision": (
                    "Low temporal STD and small peak-to-peak variation indicate stable "
                    "measurements when camera and target are stationary."
                ),
                "factory_calibration_change_rule": (
                    "Do not replace factory calibration based on a single run. Repeat "
                    "the test on independent sessions and held-out distances; only "
                    "consider a custom correction when the systematic error is stable "
                    "and reproducible."
                ),
            },
        }

        with (output_dir / "depth_validation_report.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(json_safe(report), f, indent=2, ensure_ascii=False)
            f.write("\n")

        # -------------------------------------------------------------
        # Console summary.
        # -------------------------------------------------------------
        print()
        print("=" * 78)
        print("VALIDATION COMPLETE")
        print("=" * 78)
        print()
        print("Central ROI results:")
        print(
            f"{'Distance':>10} {'Bias mm':>12} {'MAE mm':>12} "
            f"{'RMSE mm':>12} {'Valid %':>10}"
        )

        for row in sorted(
            [r for r in accuracy_rows if r["roi"] == "C"],
            key=lambda r: r["distance_m"],
        ):
            print(
                f"{row['distance_m']:>10.2f} "
                f"{row['bias_m'] * 1000.0:>12.3f} "
                f"{row['mean_absolute_error_m'] * 1000.0:>12.3f} "
                f"{row['rmse_m'] * 1000.0:>12.3f} "
                f"{row['mean_valid_percent']:>10.2f}"
            )

        print()
        print("Depth-scale diagnostic:")
        if scale_fit.get("available"):
            print(
                f"  scale_factor = {scale_fit['scale_factor']:.9f}"
            )
            print(
                f"  offset        = {scale_fit['offset_mm']:.3f} mm"
            )
            print(
                f"  fit RMSE      = {scale_fit['residual_rmse_mm']:.3f} mm"
            )
        else:
            print("  unavailable")

        if temporal is not None and temporal.get("samples", 0) > 0:
            print()
            print("Temporal test:")
            print(
                f"  distance      = {temporal['distance_m']:.3f} m"
            )
            print(
                f"  samples       = {temporal['samples']}"
            )
            print(
                f"  bias          = {temporal['bias_m'] * 1000.0:.3f} mm"
            )
            print(
                f"  STD           = {temporal['std_m'] * 1000.0:.3f} mm"
            )
            print(
                f"  RMSE          = {temporal['rmse_m'] * 1000.0:.3f} mm"
            )
            print(
                f"  peak-to-peak   = {temporal['peak_to_peak_m'] * 1000.0:.3f} mm"
            )

        print()
        print("Results:")
        for name in [
            "depth_validation_report.json",
            "raw_depth_measurements.csv",
            "accuracy_by_distance.csv",
            "spatial_precision_by_distance.csv",
            "accuracy_vs_distance.png",
            "relative_accuracy_vs_distance.png",
            "rmse_vs_distance.png",
            "spatial_precision_vs_distance.png",
            "roi_bias_heatmap.png",
            "depth_error_histogram.png",
            "candidate_depth_scale_fit.png",
        ]:
            print(f"  {output_dir / name}")

        if temporal is not None:
            print(f"  {output_dir / 'temporal_precision.csv'}")
            print(f"  {output_dir / 'temporal_stability.png'}")

        print()
        print(
            "IMPORTANT: no factory calibration was changed. The scale fit is only "
            "a diagnostic candidate for a future, independently validated correction."
        )

        return 0

    except OBError as exc:
        print(f"ORBBEC SDK ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
