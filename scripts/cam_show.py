"""Display an Orbbec camera's color stream with OpenCV."""

import cv2
import numpy as np
import pyorbbecsdk as ob
from lerobot_camera_orbbec.orbbec_utils import set_device_color_settings


WIDTH = 640
HEIGHT = 480
FPS = 60
FRAME_TIMEOUT_MS = 1000

# gemini 305
# # Edit these values and restart the script to compare the captured image.
# # Gemini 305 and 336 commonly support exposure 1-1990/1665, gain 16-248,
# # white balance 2800-6500, and backlight compensation 0-6.
# AUTO_EXPOSURE = False
# EXPOSURE = 155
# GAIN = 35
# AUTO_WHITE_BALANCE = False
# WHITE_BALANCE = 3700
# AUTO_EXPOSURE_PRIORITY = 0
# ANTI_FLICKER = False
# # 0 disables power-line compensation, 1 selects 50 Hz, and 2 selects 60 Hz.
# POWER_LINE_FREQUENCY = 1
# BACKLIGHT_COMPENSATION = 0

# Edit these values and restart the script to compare the captured image.
# Gemini 305 and 336 commonly support exposure 1-1990/1665, gain 16-248,
# white balance 2800-6500, and backlight compensation 0-6.
AUTO_EXPOSURE = False
EXPOSURE = 160
GAIN = 19
AUTO_WHITE_BALANCE = False
WHITE_BALANCE = 4200
AUTO_EXPOSURE_PRIORITY = 0
ANTI_FLICKER = False
# 0 disables power-line compensation, 1 selects 50 Hz, and 2 selects 60 Hz.
POWER_LINE_FREQUENCY = 1
BACKLIGHT_COMPENSATION = 0


def select_color_profile(pipeline):
    """Prefer an RGB profile, then MJPG, and finally the camera default."""
    profiles = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
    preferred_formats = (ob.OBFormat.RGB, ob.OBFormat.MJPG)

    for frame_format in preferred_formats:
        for index in range(profiles.get_count()):
            profile = profiles.get_stream_profile_by_index(index).as_video_stream_profile()
            if (
                profile.get_width() == WIDTH
                and profile.get_height() == HEIGHT
                and profile.get_fps() == FPS
                and profile.get_format() == frame_format
            ):
                return profile

    print(f"Warning: {WIDTH}x{HEIGHT}@{FPS} RGB/MJPG is unavailable, using the default profile")
    return profiles.get_default_video_stream_profile()


def color_frame_to_bgr(frame):
    """Convert an SDK color frame to the BGR layout expected by OpenCV."""
    width = frame.get_width()
    height = frame.get_height()
    frame_format = frame.get_format()
    data = np.asarray(frame.get_data(), dtype=np.uint8)

    if frame_format == ob.OBFormat.RGB:
        return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if frame_format == ob.OBFormat.BGR:
        return data.reshape(height, width, 3)
    if frame_format == ob.OBFormat.RGBA:
        return cv2.cvtColor(data.reshape(height, width, 4), cv2.COLOR_RGBA2BGR)
    if frame_format == ob.OBFormat.BGRA:
        return cv2.cvtColor(data.reshape(height, width, 4), cv2.COLOR_BGRA2BGR)
    if frame_format == ob.OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("OpenCV failed to decode the MJPG color frame")
        return image
    if frame_format in (ob.OBFormat.YUYV, ob.OBFormat.YUY2):
        return cv2.cvtColor(
            data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2
        )
    if frame_format == ob.OBFormat.UYVY:
        return cv2.cvtColor(
            data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_UYVY
        )

    raise RuntimeError(f"Unsupported color frame format: {frame_format}")


def main():
    context = ob.Context()
    device_list = context.query_devices()
    if device_list.get_count() == 0:
        raise RuntimeError("No Orbbec camera detected, check the USB connection and permissions")

    device = device_list.get_device_by_index(1)
    device_info = device.get_device_info()
    pipeline = ob.Pipeline(device)
    config = ob.Config()
    color_profile = select_color_profile(pipeline)
    config.enable_stream(color_profile)

    pipeline.start(config)
    try:
        color_settings = {
            "auto_exposure": AUTO_EXPOSURE,
            "exposure": EXPOSURE,
            "gain": GAIN,
            "auto_white_balance": AUTO_WHITE_BALANCE,
            "white_balance": WHITE_BALANCE,
            "auto_exposure_priority": AUTO_EXPOSURE_PRIORITY,
            "anti_flicker": ANTI_FLICKER,
            "power_line_frequency": POWER_LINE_FREQUENCY,
            "backlight_compensation": BACKLIGHT_COMPENSATION,
        }
        applied_settings = set_device_color_settings(device, color_settings)

        print(
            f"Opened {device_info.get_name()} ({device_info.get_serial_number()}): "
            f"{color_profile.get_width()}x{color_profile.get_height()}@"
            f"{color_profile.get_fps()}, format={color_profile.get_format()}"
        )
        print(f"Applied color settings: {applied_settings}")
        print("Press q or Esc to exit")

        while True:
            frames = pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
            if frames is None:
                print("Warning: timed out waiting for a color frame")
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            image = color_frame_to_bgr(color_frame)
            cv2.imshow("Orbbec RGB Camera", image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
