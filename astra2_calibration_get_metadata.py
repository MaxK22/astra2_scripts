import json
from datetime import datetime
import numpy as np
import pyorbbecsdk
from pyorbbecsdk import Context, Pipeline, Config, OBSensorType


def extract_distortion(dist_obj):
    """Extract available distortion parameters from the OBCameraDistortion object."""
    distortion = {}
    for attr in ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']:
        if hasattr(dist_obj, attr):
            distortion[attr] = float(getattr(dist_obj, attr))
    return distortion


def get_sdk_version():
    """Retrieve SDK version dynamically."""
    if hasattr(pyorbbecsdk, "get_version"):
        return pyorbbecsdk.get_version()
    return getattr(pyorbbecsdk, "__version__", "Unknown")


def main():
    ctx = Context()
    dev_list = ctx.query_devices()

    if dev_list.get_count() == 0:
        raise RuntimeError("No Orbbec Astra 2 device found. Ensure the camera is connected.")

    device = dev_list.get_device_by_index(0)
    info = device.get_device_info()

    # Device Metadata
    camera_name = info.get_name()
    serial_number = info.get_serial_number()
    firmware = info.get_firmware_version()
    sdk_version = str(get_sdk_version())

    # Setup Pipeline
    pipeline = Pipeline()
    config = Config()

    try:
        depth_profile = pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        ).get_default_video_stream_profile()
        config.enable_stream(depth_profile)
    except Exception as e:
        print(f"Warning: Unable to enable depth stream profile: {e}")

    try:
        color_profile = pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        ).get_default_video_stream_profile()
        config.enable_stream(color_profile)
    except Exception as e:
        print(f"Warning: Unable to enable color stream profile: {e}")

    pipeline.start(config)

    try:
        # Wait for frames to ensure the pipeline is fully initialized 
        # and calibration data is populated from the device
        pipeline.wait_for_frames(1000)
        
        # Retrieve the calibration parameters
        camera_param = pipeline.get_camera_param()
    finally:
        pipeline.stop()

    # Depth Parameters
    depth_int = camera_param.depth_intrinsic
    depth_dist = camera_param.depth_distortion
    depth_data = {
        "fx": float(depth_int.fx),
        "fy": float(depth_int.fy),
        "cx": float(depth_int.cx),
        "cy": float(depth_int.cy),
        "distortion": extract_distortion(depth_dist)
    }

    # RGB Parameters
    rgb_int = camera_param.rgb_intrinsic
    rgb_dist = camera_param.rgb_distortion
    rgb_data = {
        "fx": float(rgb_int.fx),
        "fy": float(rgb_int.fy),
        "cx": float(rgb_int.cx),
        "cy": float(rgb_int.cy),
        "distortion": extract_distortion(rgb_dist)
    }

    # Extrinsic Transform (DEPTH -> RGB)
    transform = camera_param.transform
    
    # Safely flatten multi-dimensional arrays to standard Python float lists
    rot_flat = np.array(transform.rot).flatten().tolist()
    trans_flat = np.array(transform.transform).flatten().tolist()

    # Format 1D rotation array (9 floats) into a 3x3 matrix format for JSON
    if len(rot_flat) == 9:
        rot_matrix = [rot_flat[i:i + 3] for i in range(0, 9, 3)]
    else:
        rot_matrix = rot_flat

    extrinsics_data = {
        "R": rot_matrix,
        "t": trans_flat
    }

    # Timestamp & Output Path
    now = datetime.now()
    timestamp_filename = now.strftime("%Y%m%d_%H%M%S")
    capture_date_str = now.strftime("%Y-%m-%d %H:%M:%S")
    output_filename = f"calibration_verification_{timestamp_filename}.json"

    # Save Structure
    payload = {
        "camera_model": camera_name,
        "serial_number": serial_number,
        "firmware": firmware,
        "sdk_version": sdk_version,
        "capture_date": capture_date_str,
        "calibration_data": {
            "DEPTH": depth_data,
            "RGB": rgb_data,
            "DEPTH_TO_RGB": extrinsics_data
        }
    }

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    print(f"Successfully saved calibration parameters to: {output_filename}")


if __name__ == "__main__":
    main()