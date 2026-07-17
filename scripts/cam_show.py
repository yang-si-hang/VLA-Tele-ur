"""Display an Orbbec camera's color stream with OpenCV."""

import cv2
import numpy as np
import pyorbbecsdk as ob


WIDTH = 1280
HEIGHT = 720
FPS = 30
FRAME_TIMEOUT_MS = 1000


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

    print(f"警告：相机不支持 {WIDTH}x{HEIGHT}@{FPS} RGB/MJPG，使用默认彩色配置")
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
            raise RuntimeError("OpenCV 无法解码 MJPG 彩色帧")
        return image
    if frame_format in (ob.OBFormat.YUYV, ob.OBFormat.YUY2):
        return cv2.cvtColor(
            data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2
        )
    if frame_format == ob.OBFormat.UYVY:
        return cv2.cvtColor(
            data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_UYVY
        )

    raise RuntimeError(f"不支持的彩色图像格式：{frame_format}")


def main():
    context = ob.Context()
    device_list = context.query_devices()
    if device_list.get_count() == 0:
        raise RuntimeError("未检测到 Orbbec 相机，请检查 USB 连接和设备权限")

    device = device_list.get_device_by_index(0)
    device_info = device.get_device_info()
    pipeline = ob.Pipeline(device)
    config = ob.Config()
    color_profile = select_color_profile(pipeline)
    config.enable_stream(color_profile)

    pipeline.start(config)
    print(
        f"已打开 {device_info.get_name()} ({device_info.get_serial_number()})："
        f"{color_profile.get_width()}x{color_profile.get_height()}@"
        f"{color_profile.get_fps()}，格式 {color_profile.get_format()}"
    )
    print("按 q 或 Esc 键退出")

    try:
        while True:
            frames = pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
            if frames is None:
                print("警告：等待图像帧超时")
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
