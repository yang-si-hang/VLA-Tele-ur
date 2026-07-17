"""Small pyorbbecSDK adapter used by the LeRobot-facing camera class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    import pyorbbecsdk as ob
except ImportError as exc:  # pragma: no cover - exercised only in an incomplete installation
    raise ImportError(
        "pyorbbecSDK is required by lerobot_camera_orbbec. Install the `pyorbbecsdk2` package."
    ) from exc


@dataclass(frozen=True)
class ColorProfile:
    width: int
    height: int
    fps: int
    format_name: str


def _format_name(value: Any) -> str:
    return str(value).removeprefix("OBFormat.")


def _device_records(context: Any) -> list[tuple[Any, dict[str, Any]]]:
    """ Return a list of (device, record) for all Orbbec devices in the given context. """
    device_list = context.query_devices()
    records: list[tuple[Any, dict[str, Any]]] = []
    for index in range(device_list.get_count()):
        device = device_list.get_device_by_index(index)
        info = device.get_device_info()
        records.append(
            (
                device,
                {
                    "name": info.get_name(),
                    "serial_number": info.get_serial_number(),
                    "firmware_version": info.get_firmware_version(),
                    "connection_type": info.get_connection_type(),
                    "uid": info.get_uid(),
                    "vendor_id": info.get_vid(),
                    "product_id": info.get_pid(),
                },
            )
        )
    return records


def _select_device(context: Any, serial_number_or_name: str) -> tuple[Any, dict[str, Any]]:
    records = _device_records(context)
    serial_matches = [record for record in records if record[1]["serial_number"] == serial_number_or_name]
    if len(serial_matches) == 1:
        return serial_matches[0]

    name_matches = [record for record in records if record[1]["name"] == serial_number_or_name]
    if len(name_matches) == 1:
        return name_matches[0]
    if len(name_matches) > 1:
        serials = ", ".join(record[1]["serial_number"] for record in name_matches)
        raise ValueError(
            f"Multiple Orbbec cameras are named {serial_number_or_name!r}; use a serial number: {serials}."
        )

    available = ", ".join(
        f"{record[1]['name']} ({record[1]['serial_number']})" for record in records
    ) or "none"
    raise ConnectionError(
        f"Orbbec camera {serial_number_or_name!r} was not found. Connected Orbbec cameras: {available}."
    )


def _video_profiles(profile_list: Any) -> list[Any]:
    return [
        profile_list.get_stream_profile_by_index(index).as_video_stream_profile()
        for index in range(profile_list.get_count())
    ]


def _profile_dict(profile: Any) -> dict[str, Any]:
    return {
        "format": _format_name(profile.get_format()),
        "width": profile.get_width(),
        "height": profile.get_height(),
        "fps": profile.get_fps(),
    }


def find_orbbec_cameras() -> list[dict[str, Any]]:
    """Enumerate Orbbec devices without starting any streams."""

    context = ob.Context()
    cameras: list[dict[str, Any]] = []
    for device, record in _device_records(context):
        pipeline = ob.Pipeline(device)
        profiles = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        default_profile = profiles.get_default_video_stream_profile()
        cameras.append(
            {
                "name": record["name"],
                "type": "Orbbec",
                "id": record["serial_number"],
                **record,
                "default_stream_profile": _profile_dict(default_profile),
            }
        )
    return cameras


class OrbbecColorStream:
    """Own an Orbbec color-only pipeline and return canonical RGB arrays."""

    # MJPG is preferred for USB bandwidth and is reliable on both Gemini 305 and 336.
    _FORMAT_PREFERENCE = (
        ob.OBFormat.MJPG,
        ob.OBFormat.RGB,
        ob.OBFormat.BGR,
        ob.OBFormat.RGBA,
        ob.OBFormat.BGRA,
        ob.OBFormat.YUYV,
        ob.OBFormat.YUY2,
        ob.OBFormat.UYVY,
    )

    def __init__(self, serial_number_or_name: str):
        self.serial_number_or_name = serial_number_or_name
        self.serial_number: str | None = None
        self._context: Any | None = None
        self._device: Any | None = None
        self._pipeline: Any | None = None
        self._config: Any | None = None
        self._profile: Any | None = None

    @property
    def is_started(self) -> bool:
        return self._pipeline is not None and self._profile is not None

    def start(self, *, width: int, height: int, fps: int) -> ColorProfile:
        if self.is_started:
            raise RuntimeError("Orbbec color stream is already started.")

        self._context = ob.Context()
        self._device, record = _select_device(self._context, self.serial_number_or_name)
        self.serial_number = record["serial_number"]
        self._pipeline = ob.Pipeline(self._device)
        self._config = ob.Config()

        profile_list = self._pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        profiles = _video_profiles(profile_list)
        matches = [
            profile
            for profile in profiles
            if profile.get_width() == width
            and profile.get_height() == height
            and profile.get_fps() == fps
        ]

        # Filter the matched color profile
        selected = None
        for preferred_format in self._FORMAT_PREFERENCE:
            selected = next(
                (profile for profile in matches if profile.get_format() == preferred_format),
                None,
            )
            if selected is not None:
                break

        if selected is None:
            serial_number = self.serial_number
            supported = sorted(
                {
                    (profile.get_width(), profile.get_height(), profile.get_fps())
                    for profile in profiles
                    if profile.get_format() in self._FORMAT_PREFERENCE
                }
            )
            preview = ", ".join(f"{w}x{h}@{rate}" for w, h, rate in supported[:20])
            if len(supported) > 20:
                preview += ", ..."
            self.stop()
            raise ValueError(
                f"Orbbec camera {serial_number!r} does not support color profile "
                f"{width}x{height}@{fps}. Supported profiles include: {preview}."
            )

        self._profile = selected
        self._config.enable_stream(selected)
        try:
            self._pipeline.start(self._config)
        except Exception as exc:
            serial_number = self.serial_number
            self.stop()
            raise ConnectionError(
                f"Failed to start Orbbec camera {serial_number!r} at {width}x{height}@{fps}: {exc}"
            ) from exc

        return ColorProfile(width, height, fps, _format_name(selected.get_format()))

    def read_rgb(self, *, timeout_ms: int) -> NDArray[np.uint8] | None:
        if not self.is_started or self._pipeline is None:
            raise RuntimeError("Orbbec color stream is not started.")

        frames = self._pipeline.wait_for_frames(int(timeout_ms))
        if frames is None:
            return None
        frame = frames.get_color_frame()
        if frame is None:
            return None
        return self._frame_to_rgb(frame)

    @staticmethod
    def _frame_to_rgb(frame: Any) -> NDArray[np.uint8]:
        width = frame.get_width()
        height = frame.get_height()
        frame_format = frame.get_format()
        data = np.asarray(frame.get_data(), dtype=np.uint8)

        if frame_format == ob.OBFormat.MJPG:
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("OpenCV failed to decode an Orbbec MJPG color frame.")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif frame_format == ob.OBFormat.RGB:
            rgb = data.reshape(height, width, 3)
        elif frame_format == ob.OBFormat.BGR:
            bgr = data.reshape(height, width, 3)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif frame_format == ob.OBFormat.RGBA:
            rgba = data.reshape(height, width, 4)
            rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        elif frame_format == ob.OBFormat.BGRA:
            bgra = data.reshape(height, width, 4)
            rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        elif frame_format in (ob.OBFormat.YUYV, ob.OBFormat.YUY2):
            yuyv = data.reshape(height, width, 2)
            rgb = cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)
        elif frame_format == ob.OBFormat.UYVY:
            uyvy = data.reshape(height, width, 2)
            rgb = cv2.cvtColor(uyvy, cv2.COLOR_YUV2RGB_UYVY)
        else:  # pragma: no cover - selection prevents this branch
            raise RuntimeError(f"Unsupported Orbbec color format: {_format_name(frame_format)}.")

        if rgb.shape != (height, width, 3):
            raise RuntimeError(
                f"Decoded Orbbec frame has shape {rgb.shape}; expected ({height}, {width}, 3)."
            )
        # The SDK owns ``frame.get_data()``. Always detach the returned image so
        # it remains valid after the FrameSet is released or its buffer is reused.
        return np.array(rgb, dtype=np.uint8, order="C", copy=True)

    def stop(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        self._profile = None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                # A physically disconnected device may reject stop; local ownership
                # still has to be released so the wrapper can reconnect later.
                pass

        self._config = None
        self._device = None
        self._context = None
        self.serial_number = None
