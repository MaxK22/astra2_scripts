#!/usr/bin/env python3
"""
Astra 2 RGB factory-calibration verification
=============================================

NO Orbbec Viewer is used.

Purpose
-------
Verify whether the current Astra 2 factory RGB calibration is suitable for
this research dataset. The program does NOT modify the camera's factory
calibration. If the factory calibration does not pass the verification, the
program saves the independent OpenCV calibration as a CANDIDATE calibration
for downstream use, so it can be reviewed before adoption.

Dataset acquisition settings supplied by the user
--------------------------------------------------
DepthWorkMode:              High Resolution
Synchronization mode:       Standalone
Timed Sync:                 ON
Record Playback:            RAW
Depth resolution:           1600 x 1200
Depth FPS:                  15
Depth Format:               Y16
Depth engine:               Hardware
Depth post processing:      OFF
Color resolution:           1920 x 1080
Color FPS:                  15
Color Format:               YUYV

Checkerboard
------------
8 x 6 INNER corners
Square size = 24.0 mm

The calibration verification is RGB-intrinsic calibration. Depth calibration
and depth<->RGB extrinsic verification are separate experiments.

Install
-------
    pip install --upgrade pyorbbecsdk2
    pip install opencv-python numpy

Usage
-----
Capture checkerboards directly from Astra 2:
    python astra2_rgb_calibration_verify.py capture

Calibrate + factory-vs-OpenCV held-out verification:
    python astra2_rgb_calibration_verify.py verify

Recommended:
    30-40 valid checkerboard images, with ~10 held out for validation.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pyorbbecsdk import Config, Context, OBError, OBFormat, OBSensorType, Pipeline


# ---------------------------------------------------------------------------
# EXACT user settings
# ---------------------------------------------------------------------------
DEFAULT_RGB_WIDTH = 1920
DEFAULT_RGB_HEIGHT = 1080
DEFAULT_RGB_FPS = 15
DEFAULT_RGB_FORMAT = "YUYV"

DEFAULT_DEPTH_WIDTH = 1600
DEFAULT_DEPTH_HEIGHT = 1200
DEFAULT_DEPTH_FPS = 15
DEFAULT_DEPTH_FORMAT = "Y16"

DEFAULT_CHECKER_COLS = 8     # INNER corners
DEFAULT_CHECKER_ROWS = 6     # INNER corners
DEFAULT_SQUARE_MM = 24.0

DEFAULT_CAPTURE_DIR = "rgb_calibration_images"
DEFAULT_OUTPUT_DIR = "rgb_calibration_results"
DEFAULT_MAX_IMAGES = 40
DEFAULT_HOLDOUT_IMAGES = 10

# Screening parameters. These are deliberately configurable and are NOT
# universal camera-calibration pass/fail standards.
DEFAULT_MAX_OPENCV_MEAN_REPROJECTION_PX = 0.75
DEFAULT_MINIMUM_IMPROVEMENT = 0.10  # 10% held-out improvement
DEFAULT_MIN_IMAGES = 20


# ---------------------------------------------------------------------------
# JSON-safe conversion -- never turn an array into a scalar
# ---------------------------------------------------------------------------
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
            return arr.item()
        return arr.tolist()
    except Exception:
        return str(value)


def safe_call(obj: Any, method_name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, method_name)()
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Checkerboard
# ---------------------------------------------------------------------------
def make_object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    obj = np.zeros((rows * cols, 3), dtype=np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_mm)
    return obj


def find_checkerboard(gray: np.ndarray, cols: int, rows: int):
    pattern = (cols, rows)

    # Prefer the newer robust detector.
    if hasattr(cv2, "findChessboardCornersSB"):
        try:
            flags = (
                cv2.CALIB_CB_EXHAUSTIVE
                | cv2.CALIB_CB_ACCURACY
                | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
            if ok:
                return True, corners.astype(np.float32)
        except Exception:
            pass

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-5,
    )
    refined = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), criteria
    )
    return True, refined.astype(np.float32)


# ---------------------------------------------------------------------------
# Astra 2 stream handling
# ---------------------------------------------------------------------------
def get_exact_color_profile(pipeline: Pipeline):
    profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    return profiles.get_video_stream_profile(
        DEFAULT_RGB_WIDTH,
        DEFAULT_RGB_HEIGHT,
        OBFormat.YUYV,
        DEFAULT_RGB_FPS,
    )


def get_exact_depth_profile(pipeline: Pipeline):
    profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    return profiles.get_video_stream_profile(
        DEFAULT_DEPTH_WIDTH,
        DEFAULT_DEPTH_HEIGHT,
        OBFormat.Y16,
        DEFAULT_DEPTH_FPS,
    )


def color_frame_to_bgr(frame) -> np.ndarray:
    width = int(frame.get_width())
    height = int(frame.get_height())
    fmt = str(frame.get_format()).upper()
    raw = frame.get_data()

    if "YUYV" in fmt or "YUY2" in fmt:
        arr = np.frombuffer(raw, dtype=np.uint8)
        expected = width * height * 2
        if arr.size != expected:
            raise RuntimeError(
                f"YUYV buffer size {arr.size}; expected {expected}"
            )
        arr = arr.reshape(height, width, 2)
        return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_YUY2)

    if "MJPG" in fmt or "MJPEG" in fmt:
        arr = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Could not decode MJPG frame")
        return image

    if "RGB" in fmt and "BGR" not in fmt:
        arr = np.frombuffer(raw, dtype=np.uint8)
        expected = width * height * 3
        if arr.size != expected:
            raise RuntimeError(
                f"RGB buffer size {arr.size}; expected {expected}"
            )
        arr = arr.reshape(height, width, 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    if "BGR" in fmt:
        arr = np.frombuffer(raw, dtype=np.uint8)
        expected = width * height * 3
        if arr.size != expected:
            raise RuntimeError(
                f"BGR buffer size {arr.size}; expected {expected}"
            )
        return arr.reshape(height, width, 3).copy()

    raise RuntimeError(f"Unsupported color format: {fmt}")


def get_device_info(device) -> dict[str, Any]:
    info = device.get_device_info()
    return json_safe({
        "name": safe_call(info, "get_name"),
        "serial_number": safe_call(info, "get_serial_number"),
        "firmware_version": safe_call(info, "get_firmware_version"),
        "hardware_version": safe_call(info, "get_hardware_version"),
        "vid": safe_call(info, "get_vid"),
        "pid": safe_call(info, "get_pid"),
        "uid": safe_call(info, "get_uid"),
    })


def get_sdk_version() -> str:
    try:
        return importlib.metadata.version("pyorbbecsdk2")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Factory RGB calibration
# ---------------------------------------------------------------------------
def get_factory_rgb_calibration(pipeline: Pipeline) -> dict[str, Any]:
    cam = pipeline.get_camera_param()
    c = cam.rgb_intrinsic
    d = cam.rgb_distortion

    K = np.array(
        [
            [float(c.fx), 0.0, float(c.cx)],
            [0.0, float(c.fy), float(c.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    named: dict[str, float] = {}
    for name in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"):
        if hasattr(d, name):
            try:
                named[name] = float(getattr(d, name))
            except Exception:
                pass

    # OpenCV rational-model ordering.
    D = np.array(
        [
            named.get("k1", 0.0),
            named.get("k2", 0.0),
            named.get("p1", 0.0),
            named.get("p2", 0.0),
            named.get("k3", 0.0),
            named.get("k4", 0.0),
            named.get("k5", 0.0),
            named.get("k6", 0.0),
        ],
        dtype=np.float64,
    ).reshape(-1, 1)

    return {
        "resolution": {
            "width": int(c.width),
            "height": int(c.height),
        },
        "K": K,
        "distortion": D,
        "distortion_named": named,
        "distortion_order": "k1, k2, p1, p2, k3, k4, k5, k6",
    }


# ---------------------------------------------------------------------------
# Capture checkerboards directly from Astra 2
# ---------------------------------------------------------------------------
def capture(args) -> int:
    out = Path(args.capture_dir)
    out.mkdir(parents=True, exist_ok=True)

    ctx = Context()
    devices = ctx.query_devices()
    if devices.get_count() == 0:
        print("ERROR: No Orbbec device found.")
        return 1

    if not 0 <= args.device_index < devices.get_count():
        print("ERROR: Invalid device index.")
        return 1

    device = devices.get_device_by_index(args.device_index)
    pipeline = Pipeline(device)
    config = Config()

    color_profile = get_exact_color_profile(pipeline)
    depth_profile = get_exact_depth_profile(pipeline)

    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)

    print()
    print("ASTRA 2 RGB CALIBRATION IMAGE CAPTURE")
    print("=====================================")
    print("NO Orbbec Viewer")
    print(f"RGB   : {DEFAULT_RGB_WIDTH}x{DEFAULT_RGB_HEIGHT}@{DEFAULT_RGB_FPS} YUYV")
    print(f"Depth : {DEFAULT_DEPTH_WIDTH}x{DEFAULT_DEPTH_HEIGHT}@{DEFAULT_DEPTH_FPS} Y16")
    print(f"Board : {DEFAULT_CHECKER_COLS}x{DEFAULT_CHECKER_ROWS} INNER corners")
    print(f"Square: {DEFAULT_SQUARE_MM} mm")
    print()
    print("Keep Astra 2 fixed. Move the checkerboard.")
    print("Use different distances, positions and tilts.")
    print("SPACE = save if checkerboard is detected")
    print("Q/ESC = quit")
    print()

    pipeline.start(config)
    saved = 0

    try:
        while saved < args.max_images:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue

            color = frames.get_color_frame()
            if color is None:
                continue

            image = color_frame_to_bgr(color)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            found, corners = find_checkerboard(
                gray,
                DEFAULT_CHECKER_COLS,
                DEFAULT_CHECKER_ROWS,
            )

            display = image.copy()
            if found:
                cv2.drawChessboardCorners(
                    display,
                    (DEFAULT_CHECKER_COLS, DEFAULT_CHECKER_ROWS),
                    corners,
                    True,
                )
                message = "FOUND - SPACE to save"
                message_color = (0, 255, 0)
            else:
                message = "Checkerboard not found"
                message_color = (0, 0, 255)

            cv2.putText(
                display,
                message,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                message_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Saved: {saved}/{args.max_images}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Astra 2 RGB Calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            if key == 32 and found:
                filename = out / f"calib_{saved:03d}.png"
                if cv2.imwrite(str(filename), image):
                    saved += 1
                    print(f"Saved: {filename.resolve()}")

    finally:
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass

    print(f"\nSaved {saved} calibration images to {out.resolve()}")
    return 0


# ---------------------------------------------------------------------------
# OpenCV calibration
# ---------------------------------------------------------------------------
def calibrate_opencv(image_paths: list[Path], args):
    obj_template = make_object_points(
        DEFAULT_CHECKER_COLS,
        DEFAULT_CHECKER_ROWS,
        DEFAULT_SQUARE_MM,
    )

    objects: list[np.ndarray] = []
    images: list[np.ndarray] = []
    names: list[str] = []
    rejected: list[str] = []
    image_size = None

    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rejected.append(f"{path.name}: cannot read")
            continue

        h, w = image.shape[:2]
        current_size = (w, h)
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            rejected.append(
                f"{path.name}: resolution {current_size} differs from {image_size}"
            )
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = find_checkerboard(
            gray,
            DEFAULT_CHECKER_COLS,
            DEFAULT_CHECKER_ROWS,
        )
        if not found:
            rejected.append(f"{path.name}: checkerboard not detected")
            continue

        objects.append(obj_template.copy())
        images.append(corners.astype(np.float32))
        names.append(path.name)

    if image_size is None:
        raise RuntimeError("No valid images found.")

    if len(objects) < DEFAULT_MIN_IMAGES:
        raise RuntimeError(
            f"Only {len(objects)} usable images. "
            f"At least {DEFAULT_MIN_IMAGES} are required by this verification procedure."
        )

    flags = 0
    # Match the Astra distortion structure containing k1..k6, p1,p2.
    if hasattr(cv2, "CALIB_RATIONAL_MODEL"):
        flags |= cv2.CALIB_RATIONAL_MODEL

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-9,
    )

    (
        rms,
        K,
        D,
        rvecs,
        tvecs,
        std_int,
        std_ext,
        per_view,
    ) = cv2.calibrateCameraExtended(
        objects,
        images,
        image_size,
        None,
        None,
        flags=flags,
        criteria=criteria,
    )

    per_image = []
    all_error = []

    for name, obj, img, rvec, tvec in zip(
        names, objects, images, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        projected = projected.reshape(-1, 2)
        observed = img.reshape(-1, 2)
        err = np.linalg.norm(observed - projected, axis=1)
        all_error.extend(err.tolist())

        per_image.append({
            "image": name,
            "mean_px": float(np.mean(err)),
            "median_px": float(np.median(err)),
            "max_px": float(np.max(err)),
        })

    return {
        "image_size": {
            "width": image_size[0],
            "height": image_size[1],
        },
        "images_used": len(names),
        "images_rejected": rejected,
        "used_images": names,
        "K": K,
        "distortion": D,
        "distortion_order": "k1, k2, p1, p2, k3, k4, k5, k6",
        "calibration_flags": int(flags),
        "rms_px": float(rms),
        "mean_reprojection_px": float(np.mean(all_error)),
        "median_reprojection_px": float(np.median(all_error)),
        "max_reprojection_px": float(np.max(all_error)),
        "per_image": per_image,
        "std_intrinsics": json_safe(np.asarray(std_int).reshape(-1)),
        "opencv_per_view_error": json_safe(np.asarray(per_view).reshape(-1)),
    }, objects, images, names, obj_template


# ---------------------------------------------------------------------------
# Hold-out evaluation
# ---------------------------------------------------------------------------
def choose_holdout(paths: list[Path], holdout_count: int) -> tuple[list[Path], list[Path]]:
    """Select approximately evenly spaced images for holdout evaluation."""
    n = len(paths)
    if holdout_count <= 0 or holdout_count >= n:
        return paths, []

    indices = set(np.rint(np.linspace(0, n - 1, holdout_count)).astype(int).tolist())

    while len(indices) < holdout_count:
        for i in range(n):
            if i not in indices:
                indices.add(i)
                if len(indices) >= holdout_count:
                    break

    holdout = [p for i, p in enumerate(paths) if i in indices]
    train = [p for i, p in enumerate(paths) if i not in indices]
    return train, holdout


def reprojection_with_model(
    image_path: Path,
    obj_template: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    cols: int,
    rows: int,
) -> dict[str, float]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("cannot read image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = find_checkerboard(gray, cols, rows)
    if not found:
        raise RuntimeError("checkerboard not detected")

    image_points = corners.reshape(-1, 2).astype(np.float64)
    object_points = obj_template.astype(np.float64)

    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K,
            D,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error as exc:
        raise RuntimeError(f"solvePnP failed: {exc}") from exc

    if not ok:
        raise RuntimeError("solvePnP returned False")

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K,
        D,
    )
    projected = projected.reshape(-1, 2)
    error = np.linalg.norm(image_points - projected, axis=1)

    return {
        "mean_px": float(np.mean(error)),
        "median_px": float(np.median(error)),
        "max_px": float(np.max(error)),
    }


def evaluate_holdout(
    holdout_paths: list[Path],
    obj_template: np.ndarray,
    factory: dict[str, Any],
    opencv: dict[str, Any],
) -> dict[str, Any]:
    FK = np.asarray(factory["K"], dtype=np.float64)
    FD = np.asarray(factory["distortion"], dtype=np.float64)
    OK = np.asarray(opencv["K"], dtype=np.float64)
    OD = np.asarray(opencv["distortion"], dtype=np.float64)

    results = []
    for path in holdout_paths:
        row = {"image": path.name}

        try:
            row["factory"] = reprojection_with_model(
                path, obj_template, FK, FD,
                DEFAULT_CHECKER_COLS, DEFAULT_CHECKER_ROWS,
            )
        except Exception as exc:
            row["factory"] = None
            row["factory_error"] = str(exc)

        try:
            row["opencv"] = reprojection_with_model(
                path, obj_template, OK, OD,
                DEFAULT_CHECKER_COLS, DEFAULT_CHECKER_ROWS,
            )
        except Exception as exc:
            row["opencv"] = None
            row["opencv_error"] = str(exc)

        results.append(row)

    def means(label: str, metric: str) -> list[float]:
        out = []
        for row in results:
            v = row.get(label)
            if isinstance(v, dict) and metric in v:
                out.append(float(v[metric]))
        return out

    aggregate = {}
    for label in ("factory", "opencv"):
        values_mean = means(label, "mean_px")
        values_median = means(label, "median_px")
        values_max = means(label, "max_px")
        aggregate[label] = {
            "valid_images": len(values_mean),
            "mean_of_image_mean_px": float(np.mean(values_mean)) if values_mean else None,
            "mean_of_image_median_px": float(np.mean(values_median)) if values_median else None,
            "mean_of_image_max_px": float(np.mean(values_max)) if values_max else None,
        }

    return {
        "holdout_images": [p.name for p in holdout_paths],
        "num_holdout_images": len(holdout_paths),
        "aggregate": aggregate,
        "per_image": results,
    }


# ---------------------------------------------------------------------------
# Factory calibration decision
# ---------------------------------------------------------------------------
def assess_factory(
    opencv: dict[str, Any],
    holdout: dict[str, Any],
    args,
) -> dict[str, Any]:
    """
    Conservative decision helper.

    It never writes anything to the camera.
    "FACTORY_CALIBRATION_ACCEPTABLE" means that the independent calibration
    did not demonstrate a material held-out improvement.
    "CUSTOM_CALIBRATION_WORTH_INVESTIGATING" means OpenCV performed materially
    better on hold-out images and therefore a custom calibration deserves a
    second independent run before adoption.
    "MORE_DATA_REQUIRED" is returned for insufficient image count.
    """
    n = int(opencv["images_used"])
    if n < args.min_images:
        return {
            "status": "MORE_DATA_REQUIRED",
            "recommendation": "Retain factory calibration; collect more calibration images.",
            "reason": f"Only {n} usable images; minimum configured is {args.min_images}.",
        }

    if float(opencv["mean_reprojection_px"]) > args.max_opencv_mean_px:
        return {
            "status": "CALIBRATION_CAPTURE_PROBLEM",
            "recommendation": "Do not replace factory calibration. Improve checkerboard capture first.",
            "reason": (
                f"OpenCV mean reprojection error is {opencv['mean_reprojection_px']:.4f} px, "
                f"above the configured screening limit {args.max_opencv_mean_px:.4f} px."
            ),
        }

    agg = holdout.get("aggregate", {})
    f = agg.get("factory", {}).get("mean_of_image_mean_px")
    o = agg.get("opencv", {}).get("mean_of_image_mean_px")

    if f is None or o is None or f <= 0:
        return {
            "status": "VALIDATION_INCOMPLETE",
            "recommendation": "Retain factory calibration; repeat the hold-out verification.",
            "reason": "A valid factory/OpenCV hold-out comparison was not obtained.",
        }

    improvement = (f - o) / f

    if improvement >= args.minimum_improvement:
        return {
            "status": "CUSTOM_CALIBRATION_WORTH_INVESTIGATING",
            "recommendation": (
                "Do NOT change the camera yet. Save and inspect the OpenCV calibration, "
                "then repeat with an independent checkerboard set. If the result repeats, "
                "use the OpenCV model as a downstream custom calibration."
            ),
            "reason": (
                f"OpenCV mean hold-out error {o:.4f} px is {improvement * 100:.1f}% lower "
                f"than factory {f:.4f} px."
            ),
        }

    return {
        "status": "FACTORY_CALIBRATION_CURRENTLY_ACCEPTABLE",
        "recommendation": "Keep the Astra 2 factory calibration for the dataset.",
        "reason": (
            f"OpenCV improvement on held-out images is only {improvement * 100:.1f}%, "
            f"below the configured {args.minimum_improvement * 100:.1f}% threshold."
        ),
    }


# ---------------------------------------------------------------------------
# Verification command
# ---------------------------------------------------------------------------
def verify(args) -> int:
    capture_dir = Path(args.capture_dir)
    paths = sorted(
        p for p in capture_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    if len(paths) < args.min_images:
        raise RuntimeError(
            f"Found only {len(paths)} candidate images. "
            f"At least {args.min_images} are required."
        )

    train_paths, holdout_paths = choose_holdout(
        paths,
        min(args.holdout_images, max(1, len(paths) - args.min_images)),
    )

    # Calibration is fitted ONLY on the training/calibration images.
    opencv, _, _, _, obj_template = calibrate_opencv(train_paths, args)

    # Read factory/current Astra calibration directly from the camera.
    ctx = Context()
    devices = ctx.query_devices()
    if devices.get_count() == 0:
        raise RuntimeError("No Astra 2 connected; factory comparison is impossible.")
    if not 0 <= args.device_index < devices.get_count():
        raise RuntimeError("Invalid device index.")

    device = devices.get_device_by_index(args.device_index)
    pipeline = Pipeline(device)
    config = Config()

    # EXACT DATASET PROFILES
    config.enable_stream(get_exact_color_profile(pipeline))
    config.enable_stream(get_exact_depth_profile(pipeline))

    pipeline.start(config)
    try:
        frames = None
        for _ in range(30):
            frames = pipeline.wait_for_frames(1000)
            if frames is not None:
                break
        if frames is None:
            raise RuntimeError("No valid Astra 2 frames received.")

        factory = json_safe(get_factory_rgb_calibration(pipeline))
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass

    holdout = evaluate_holdout(
        holdout_paths,
        obj_template,
        factory,
        opencv,
    )

    decision = assess_factory(opencv, holdout, args)

    now = datetime.now().astimezone()
    utc = datetime.now(timezone.utc)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_name = (
        "astra2_rgb_calibration_verification_"
        + now.strftime("%Y%m%d_%H%M%S_%f")
        + ".json"
    )
    report_path = output_dir / report_name

    # Save candidate OpenCV calibration independently. It is NOT applied to
    # the camera and is not declared to be the final calibration.
    candidate_name = (
        "astra2_rgb_candidate_opencv_calibration_"
        + now.strftime("%Y%m%d_%H%M%S_%f")
        + ".json"
    )
    candidate_path = output_dir / candidate_name

    device_meta = get_device_info(device)
    sdk_version = get_sdk_version()

    report = {
        "schema_version": "2.0",
        "viewer_used": False,
        "purpose": (
            "Independent verification of Astra 2 factory RGB calibration "
            "for the exact dataset RGB profile, with held-out comparison."
        ),

        "capture": {
            "datetime_local": now.isoformat(),
            "datetime_utc": utc.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S.%f"),
        },

        "device": device_meta,

        "software": {
            "sdk_package": "pyorbbecsdk2",
            "sdk_version": sdk_version,
            "viewer_used": False,
            "viewer_version": None,
            "opencv_version": cv2.__version__,
        },

        "dataset_capture_settings": {
            "depth_work_mode": "High Resolution",
            "synchronization_configuration_mode": "Standalone",
            "timed_sync": True,
            "record_playback_format": "RAW",
            "depth": {
                "width": DEFAULT_DEPTH_WIDTH,
                "height": DEFAULT_DEPTH_HEIGHT,
                "fps": DEFAULT_DEPTH_FPS,
                "format": DEFAULT_DEPTH_FORMAT,
                "engine": "hardware",
                "post_processing": False,
            },
            "rgb": {
                "width": DEFAULT_RGB_WIDTH,
                "height": DEFAULT_RGB_HEIGHT,
                "fps": DEFAULT_RGB_FPS,
                "format": DEFAULT_RGB_FORMAT,
            },
        },

        "checkerboard": {
            "inner_corners_columns": DEFAULT_CHECKER_COLS,
            "inner_corners_rows": DEFAULT_CHECKER_ROWS,
            "square_size_mm": DEFAULT_SQUARE_MM,
            "world_coordinate_units": "mm",
        },

        "data_split": {
            "all_candidate_images": [p.name for p in paths],
            "calibration_images": [p.name for p in train_paths],
            "holdout_images": [p.name for p in holdout_paths],
            "calibration_image_count": len(train_paths),
            "holdout_image_count": len(holdout_paths),
        },

        "astra2_factory_or_sdk_calibration": factory,
        "opencv_independent_calibration": opencv,
        "heldout_factory_vs_opencv": holdout,
        "factory_calibration_assessment": decision,

        "important_notes": [
            "Factory calibration was read directly from the Astra 2 through pyorbbecsdk; Orbbec Viewer was not used.",
            "The OpenCV calibration was fitted only on the calibration subset; holdout images were not used to fit K/D.",
            "This program never writes calibration parameters back to Astra 2.",
            "A custom calibration should only be adopted after repeating the experiment with an independent checkerboard set.",
            "This program verifies RGB intrinsics/distortion. It does not yet verify depth metric accuracy or depth-to-RGB extrinsics.",
        ],
    }

    candidate = {
        "schema_version": "1.0",
        "status": "CANDIDATE_ONLY_NOT_APPLIED",
        "camera": device_meta,
        "rgb_profile": {
            "width": DEFAULT_RGB_WIDTH,
            "height": DEFAULT_RGB_HEIGHT,
            "fps": DEFAULT_RGB_FPS,
            "format": DEFAULT_RGB_FORMAT,
        },
        "K": opencv["K"],
        "distortion": opencv["distortion"],
        "distortion_order": opencv["distortion_order"],
        "checkerboard": {
            "inner_corners": [DEFAULT_CHECKER_COLS, DEFAULT_CHECKER_ROWS],
            "square_size_mm": DEFAULT_SQUARE_MM,
        },
        "source_report": report_name,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2, ensure_ascii=False)
        f.write("\n")

    with candidate_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(candidate), f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Console summary
    f_agg = holdout["aggregate"].get("factory", {})
    o_agg = holdout["aggregate"].get("opencv", {})

    print()
    print("=" * 82)
    print("ASTRA 2 RGB FACTORY CALIBRATION VERIFICATION")
    print("=" * 82)
    print(f"Camera:              {device_meta.get('name')}")
    print(f"Serial:              {device_meta.get('serial_number')}")
    print(f"Firmware:            {device_meta.get('firmware_version')}")
    print(f"SDK:                 {sdk_version}")
    print("Viewer used:         NO")
    print()
    print("RGB profile:         1920x1080 @ 15 FPS YUYV")
    print("Depth profile:       1600x1200 @ 15 FPS Y16")
    print(f"Checkerboard:        8x6 inner corners, {DEFAULT_SQUARE_MM:.1f} mm")
    print()
    print(f"Calibration images:  {len(train_paths)}")
    print(f"Holdout images:      {len(holdout_paths)}")
    print(f"OpenCV RMS:          {opencv['rms_px']:.6f} px")
    print(f"OpenCV mean:         {opencv['mean_reprojection_px']:.6f} px")
    print(f"OpenCV median:       {opencv['median_reprojection_px']:.6f} px")
    print(f"OpenCV max:          {opencv['max_reprojection_px']:.6f} px")
    print()
    print("HELD-OUT MEAN REPROJECTION")
    print(f"Factory:             {f_agg.get('mean_of_image_mean_px')}")
    print(f"OpenCV:              {o_agg.get('mean_of_image_mean_px')}")
    print()
    print(f"ASSESSMENT:          {decision['status']}")
    print(f"Recommendation:      {decision['recommendation']}")
    print(f"Reason:              {decision['reason']}")
    print()
    print(f"Verification report: {report_path.resolve()}")
    print(f"Candidate K/D:       {candidate_path.resolve()}")
    print()
    print("IMPORTANT: Astra 2 factory calibration was NOT modified.")
    print("=" * 82)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify Astra 2 factory RGB calibration without Orbbec Viewer."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cap = sub.add_parser("capture", help="Capture checkerboard images")
    p_cap.add_argument("--device-index", type=int, default=0)
    p_cap.add_argument("--capture-dir", default=DEFAULT_CAPTURE_DIR)
    p_cap.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES)

    p_ver = sub.add_parser("verify", help="Calibrate OpenCV and compare with factory")
    p_ver.add_argument("--device-index", type=int, default=0)
    p_ver.add_argument("--capture-dir", default=DEFAULT_CAPTURE_DIR)
    p_ver.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p_ver.add_argument("--holdout-images", type=int, default=DEFAULT_HOLDOUT_IMAGES)
    p_ver.add_argument(
        "--min-images",
        type=int,
        default=DEFAULT_MIN_IMAGES,
        help="Minimum usable checkerboard images.",
    )
    p_ver.add_argument(
        "--max-opencv-mean-px",
        type=float,
        default=DEFAULT_MAX_OPENCV_MEAN_REPROJECTION_PX,
        help="Screening threshold for OpenCV mean reprojection error.",
    )
    p_ver.add_argument(
        "--minimum-improvement",
        type=float,
        default=DEFAULT_MINIMUM_IMPROVEMENT,
        help="Required held-out improvement fraction to flag custom calibration.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "capture":
            return capture(args)
        return verify(args)
    except OBError as exc:
        print(f"Orbbec SDK error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
