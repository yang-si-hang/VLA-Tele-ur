"""Expose Orbbec Gemini RGB capture through the LeRobot camera API.

``OrbbecCamera`` selects and configures an exact SDK color profile, then uses a
single background thread for blocking frame acquisition. Synchronous and
asynchronous LeRobot reads consume the newest buffered frame, while
``read_latest`` returns a fresh snapshot without consuming it. Captured images
are converted to the configured color mode and rotation before publication.
When explicitly enabled, the capture thread also monitors Camera timestamps,
host receive rate, and frame-index gaps without performing an additional SDK
read.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from threading import Condition, Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
from lerobot.cameras import Camera
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.utils import get_cv2_rotation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from numpy.typing import NDArray

from .config_orbbec import OrbbecCameraConfig
from .orbbec_utils import OrbbecColorStream, find_orbbec_cameras

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OrbbecFrame:
    """One processed frame with Camera and host capture metadata."""

    image: NDArray[np.uint8]
    frame_index: int
    # global_timestamp_us: int
    device_timestamp_us: int
    system_timestamp_us: int
    received_at: float = 0.0


@dataclass(frozen=True, slots=True)
class OrbbecFrameRateStats:
    """Snapshot of Camera capture and host receive frequency statistics."""

    target_fps: float
    capture_fps: float | None       # based on device_timestamp_us
    receive_fps: float | None       # based on host time (received_at)
    received_frames: int
    dropped_frames: int
    latest_frame_age_s: float | None


class OrbbecCamera(Camera):
    """Capture RGB images from one Orbbec camera through pyorbbecSDK v2.

    A single background thread owns all blocking SDK reads. ``read`` and
    ``async_read`` consume the newest unconsumed frame, while ``read_latest``
    peeks at the most recent buffered frame without waiting.
    """

    def __init__(self, config: OrbbecCameraConfig, *, sdk_context: Any | None = None):
        super().__init__(config)
        self.config = config
        self.serial_number_or_name = config.serial_number_or_name
        self.serial_number: str | None = None
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.color_settings = config.color_settings
        self.applied_color_settings: dict[str, bool | int] = {}
        self.frame_rate_warning_ratio = config.frame_rate_warning_ratio
        self.frame_rate_check_interval_s = config.frame_rate_check_interval_s
        self._sdk_context = sdk_context
        self.rotation: int | None = get_cv2_rotation(config.rotation)

        if self.width is None or self.height is None or self.fps is None:
            raise ValueError("OrbbecCamera requires `width`, `height`, and `fps`.")

        self.capture_width = self.width
        self.capture_height = self.height
        if self.rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            self.capture_width, self.capture_height = self.height, self.width

        self._stream: OrbbecColorStream | None = None
        self.thread: Thread | None = None
        self.stop_event: Event | None = None

        self.frame_lock = Lock()
        self._frame_condition = Condition(self.frame_lock)
        self.latest_frame: OrbbecFrame | None = None
        self._frame_sequence = 0
        self._consumed_sequence = 0
        self._read_error: BaseException | None = None
        self._frame_rate_monitor_enabled = False
        self._monitor_window_started_at = 0.0
        self._monitor_first_device_timestamp_us: int | None = None
        self._monitor_last_device_timestamp_us: int | None = None
        self._monitor_device_timestamps_valid = True
        self._monitor_received_frames = 0
        self._monitor_dropped_frames = 0
        self._received_frames = 0
        self._dropped_frames = 0
        self._last_frame_index: int | None = None
        self._capture_fps: float | None = None
        self._receive_fps: float | None = None

    def __str__(self) -> str:
        identifier = self.serial_number or self.serial_number_or_name
        return f"{self.__class__.__name__}({identifier})"

    @property
    def is_connected(self) -> bool:
        return self._stream is not None and self._stream.is_started

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Return connected Orbbec devices and their default color profiles."""

        return find_orbbec_cameras()

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        stream = OrbbecColorStream(self.serial_number_or_name, context=self._sdk_context)
        try:
            profile = stream.start(
                width=self.capture_width,
                height=self.capture_height,
                fps=self.fps,
            )
            self._stream = stream
            self.serial_number = stream.serial_number
            if self.color_settings:
                self.applied_color_settings = stream.set_color_settings(self.color_settings)

            if (
                profile.width != self.capture_width
                or profile.height != self.capture_height
                or profile.fps != self.fps
            ):
                raise RuntimeError(
                    f"{self} started an unexpected color profile: {profile.width}x{profile.height} "
                    f"at {profile.fps} FPS."
                )

            stream.synchronize_clock_with_host()
            self._start_read_thread()
            if warmup:
                self._warmup()
                if self.color_settings:
                    self.applied_color_settings = stream.verify_color_settings(self.color_settings)
            self._reset_frame_rate_monitor()
        except Exception as exc:
            self._stop_read_thread()
            stream.stop()
            self._stream = None
            self.serial_number = None
            self.applied_color_settings = {}
            if isinstance(exc, (ValueError, RuntimeError, ConnectionError)):
                raise
            raise ConnectionError(f"Failed to connect {self}: {exc}") from exc

        logger.info(
            "%s connected with %dx%d @ %d FPS (%s transport)",
            self,
            self.width,
            self.height,
            self.fps,
            profile.format_name,
        )
        if self.applied_color_settings:
            logger.info("%s applied color settings: %s", self, self.applied_color_settings)

    @check_if_not_connected
    def synchronize_clock_with_host(self) -> None:
        """Synchronize the connected camera's device clock with the host."""

        stream = self._stream
        if stream is None:
            raise RuntimeError(f"{self} has no active Orbbec stream.")
        stream.synchronize_clock_with_host()

    def _warmup(self) -> None:
        """Wait for a first frame, then keep consuming during the warmup period."""

        deadline = time.perf_counter() + max(5.0, self.warmup_s)
        first_frame = False
        while time.perf_counter() < deadline:
            remaining_ms = max(1.0, (deadline - time.perf_counter()) * 1000.0)
            try:
                self.async_read(timeout_ms=min(500.0, remaining_ms))
                first_frame = True
                break
            except TimeoutError:
                continue

        if not first_frame:
            raise ConnectionError(f"{self} did not produce a color frame within 5 seconds.")

        end = time.perf_counter() + self.warmup_s
        while time.perf_counter() < end:
            remaining_ms = max(1.0, (end - time.perf_counter()) * 1000.0)
            try:
                self.async_read(timeout_ms=min(500.0, remaining_ms))
            except TimeoutError:
                continue

    def _postprocess_image(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if image.shape != (self.capture_height, self.capture_width, 3):
            raise RuntimeError(
                f"{self} returned frame shape {image.shape}; expected "
                f"({self.capture_height}, {self.capture_width}, 3)."
            )

        processed = image
        if self.color_mode == ColorMode.BGR:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        if self.rotation is not None:
            processed = cv2.rotate(processed, self.rotation)

        if processed.shape != (self.height, self.width, 3):
            raise RuntimeError(
                f"{self} produced frame shape {processed.shape}; expected "
                f"({self.height}, {self.width}, 3)."
            )
        return np.ascontiguousarray(processed, dtype=np.uint8)

    def _reset_monitor_window_locked(self, now: float) -> None:
        self._monitor_window_started_at = now
        self._monitor_first_device_timestamp_us = None
        self._monitor_last_device_timestamp_us = None
        self._monitor_device_timestamps_valid = True
        self._monitor_received_frames = 0
        self._monitor_dropped_frames = 0

    def _reset_frame_rate_monitor_locked(self, now: float) -> None:
        self._reset_monitor_window_locked(now)
        self._received_frames = 0
        self._dropped_frames = 0
        self._last_frame_index = (
            None if self.latest_frame is None else self.latest_frame.frame_index
        )
        self._capture_fps = None
        self._receive_fps = None

    def _reset_frame_rate_monitor(self) -> None:
        with self._frame_condition:
            self._reset_frame_rate_monitor_locked(time.perf_counter())

    @check_if_not_connected
    def start_frame_rate_monitoring(self) -> None:
        """Start a fresh frame-rate monitoring window without reading another frame."""

        with self._frame_condition:
            if self._frame_rate_monitor_enabled:
                return
            self._reset_frame_rate_monitor_locked(time.perf_counter())
            self._frame_rate_monitor_enabled = True
        logger.info("%s frame-rate monitoring started.", self)

    def stop_frame_rate_monitoring(self) -> None:
        """Stop monitoring and discard the incomplete statistics window."""

        with self._frame_condition:
            if not self._frame_rate_monitor_enabled:
                return
            self._frame_rate_monitor_enabled = False
            self._reset_frame_rate_monitor_locked(time.perf_counter())
        logger.info("%s frame-rate monitoring stopped.", self)

    def _record_frame_rate_locked(self, frame: OrbbecFrame) -> None:
        previous_index = self._last_frame_index
        if previous_index is not None and frame.frame_index > previous_index:
            dropped = max(frame.frame_index - previous_index - 1, 0)
            self._dropped_frames += dropped
            self._monitor_dropped_frames += dropped
        self._last_frame_index = frame.frame_index
        self._received_frames += 1

        if self._monitor_received_frames == 0:
            self._monitor_first_device_timestamp_us = frame.device_timestamp_us
        elif (
            self._monitor_last_device_timestamp_us is not None
            and frame.device_timestamp_us <= self._monitor_last_device_timestamp_us
        ):
            self._monitor_device_timestamps_valid = False
        self._monitor_last_device_timestamp_us = frame.device_timestamp_us
        self._monitor_received_frames += 1

    def _frame_rate_stats_locked(self, now: float) -> OrbbecFrameRateStats:
        latest_age_s = (
            None
            if self.latest_frame is None
            else max(now - self.latest_frame.received_at, 0.0)
        )
        return OrbbecFrameRateStats(
            target_fps=float(self.fps),
            capture_fps=self._capture_fps,
            receive_fps=self._receive_fps,
            received_frames=self._received_frames,
            dropped_frames=self._dropped_frames,
            latest_frame_age_s=latest_age_s,
        )

    def _finish_monitor_window_locked(
        self,
        now: float,
    ) -> tuple[OrbbecFrameRateStats, int, bool] | None:
        if (
            not self._frame_rate_monitor_enabled
            or now - self._monitor_window_started_at < self.frame_rate_check_interval_s
        ):
            return None

        timestamp_invalid = False
        host_elapsed_s = now - self._monitor_window_started_at
        self._receive_fps = (
            self._monitor_received_frames / host_elapsed_s
            if host_elapsed_s > 0
            else None
        )

        if (
            self._monitor_received_frames >= 2
            and self._monitor_device_timestamps_valid
            and self._monitor_first_device_timestamp_us is not None
            and self._monitor_last_device_timestamp_us is not None
        ):
            device_elapsed_s = (
                self._monitor_last_device_timestamp_us
                - self._monitor_first_device_timestamp_us
            ) / 1_000_000.0
            self._capture_fps = (
                (self._monitor_received_frames - 1) / device_elapsed_s
                if device_elapsed_s > 0
                else None
            )
        else:
            timestamp_invalid = self._monitor_received_frames >= 2
            self._capture_fps = None

        dropped_frames = self._monitor_dropped_frames
        stats = self._frame_rate_stats_locked(now)
        self._reset_monitor_window_locked(now)
        return stats, dropped_frames, timestamp_invalid

    def _log_frame_rate_report(
        self,
        report: tuple[OrbbecFrameRateStats, int, bool] | None,
    ) -> None:
        if report is None:
            return
        with self._frame_condition:
            if not self._frame_rate_monitor_enabled:
                return
            stats, dropped_frames, timestamp_invalid = report
            warning_fps = stats.target_fps * self.frame_rate_warning_ratio
            if stats.capture_fps is not None and stats.capture_fps < warning_fps:
                logger.warning(
                    "%s capture frequency is %.1f Hz, below %.1f Hz with target %.1f Hz",
                    self,
                    stats.capture_fps,
                    warning_fps,
                    stats.target_fps,
                )
            if stats.receive_fps == 0.0:
                logger.warning(
                    "%s received no Camera frames in the latest monitor window; "
                    "capture frequency cannot be determined",
                    self,
                )
            elif stats.receive_fps is not None and stats.receive_fps < warning_fps:
                logger.warning(
                    "%s host receive frequency is %.1f Hz, below %.1f Hz with target %.1f Hz",
                    self,
                    stats.receive_fps,
                    warning_fps,
                    stats.target_fps,
                )
            if timestamp_invalid:
                logger.warning(
                    "%s device timestamps were not monotonic in the latest monitor window",
                    self,
                )
            if dropped_frames > 0:
                logger.warning(
                    "%s skipped %d Camera frame indices in the latest monitor window",
                    self,
                    dropped_frames,
                )

    @check_if_not_connected
    def get_frame_rate_stats(self) -> OrbbecFrameRateStats:
        """Return a snapshot without reading another frame from the Camera."""

        with self._frame_condition:
            return self._frame_rate_stats_locked(time.perf_counter())

    def _read_loop(self) -> None:
        stop_event = self.stop_event
        stream = self._stream
        if stop_event is None or stream is None:
            return

        consecutive_failures = 0
        while not stop_event.is_set():
            try:
                captured = stream.read_frame(timeout_ms=200)
                received_at = time.perf_counter()
                if stop_event.is_set():
                    break
                if captured is None:
                    with self._frame_condition:
                        report = self._finish_monitor_window_locked(received_at)
                    self._log_frame_rate_report(report)
                    continue
                processed = self._postprocess_image(captured.image)
                published = OrbbecFrame(
                    image=processed,
                    frame_index=captured.frame_index,
                    # global_timestamp_us=captured.global_timestamp_us,
                    device_timestamp_us=captured.device_timestamp_us,
                    system_timestamp_us=captured.system_timestamp_us,
                    received_at=received_at,
                )
                with self._frame_condition:
                    self.latest_frame = published
                    self._frame_sequence += 1
                    if self._frame_rate_monitor_enabled:
                        self._record_frame_rate_locked(published)
                    report = self._finish_monitor_window_locked(received_at)
                    self._frame_condition.notify_all()
                self._log_frame_rate_report(report)
                consecutive_failures = 0
            except Exception as exc:
                if stop_event.is_set():
                    break
                consecutive_failures += 1
                logger.warning(
                    "Error reading frame from %s (%d/10): %s",
                    self,
                    consecutive_failures,
                    exc,
                )
                if consecutive_failures >= 10:
                    with self._frame_condition:
                        self._read_error = exc
                        self._frame_condition.notify_all()
                    break

    def _start_read_thread(self) -> None:   # 开启后台相机读取线程
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self.thread.start()

    def _stop_read_thread(self) -> None:
        self.stop_frame_rate_monitoring()
        if self.stop_event is not None:
            self.stop_event.set()
        with self._frame_condition:
            self._frame_condition.notify_all()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("%s read thread did not stop within 2 seconds.", self)

        self.thread = None
        self.stop_event = None
        with self._frame_condition:
            self.latest_frame = None
            self._frame_sequence = 0
            self._consumed_sequence = 0
            self._read_error = None
            self._reset_frame_rate_monitor_locked(time.perf_counter())

    def _raise_if_reader_failed(self) -> None:
        if self._read_error is not None:
            raise RuntimeError(f"{self} background read thread failed.") from self._read_error
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

    @check_if_not_connected
    def read(self) -> NDArray[Any]:
        """Block until a new RGB/BGR frame is available."""

        return self.async_read(timeout_ms=10_000)

    @staticmethod
    def _copy_frame(frame: OrbbecFrame) -> OrbbecFrame:
        return replace(frame, image=frame.image.copy())

    @check_if_not_connected
    def async_read_with_metadata(self, timeout_ms: float = 200) -> OrbbecFrame:
        """Return the latest unconsumed frame and its capture metadata."""

        if timeout_ms < 0:
            raise ValueError("`timeout_ms` must be non-negative.")

        deadline = time.perf_counter() + timeout_ms / 1000.0
        with self._frame_condition:
            while self._frame_sequence <= self._consumed_sequence:
                self._raise_if_reader_failed()
                remaining = deadline - time.perf_counter()
                if remaining <= 0 or not self._frame_condition.wait(timeout=remaining):
                    self._raise_if_reader_failed()
                    raise TimeoutError(f"Timed out waiting for a new frame from {self} after {timeout_ms} ms.")

            frame = self.latest_frame
            self._consumed_sequence = self._frame_sequence

        if frame is None:
            raise RuntimeError(f"{self} signaled a frame without buffered image data.")
        return self._copy_frame(frame)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Return the latest unconsumed frame, waiting up to ``timeout_ms``."""

        return self.async_read_with_metadata(timeout_ms=timeout_ms).image

    @check_if_not_connected
    def read_latest_with_metadata(self, max_age_ms: float = 500) -> OrbbecFrame:
        """Return the newest buffered frame and metadata without consuming it."""

        if max_age_ms < 0:
            raise ValueError("`max_age_ms` must be non-negative.")
        self._raise_if_reader_failed()

        with self._frame_condition:
            frame = self.latest_frame

        if frame is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - frame.received_at) * 1000.0
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is {age_ms:.1f} ms old (maximum allowed: {max_age_ms} ms)."
            )
        return self._copy_frame(frame)

    @check_if_not_connected
    def read_latest(self, max_age_ms: float = 500) -> NDArray[Any]:
        """Immediately return the newest buffered frame without consuming it."""

        return self.read_latest_with_metadata(max_age_ms=max_age_ms).image

    def disconnect(self) -> None:
        """Stop capture and release the pyorbbecSDK pipeline."""

        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._stop_read_thread()
        stream = self._stream
        self._stream = None
        self.serial_number = None
        self.applied_color_settings = {}
        if stream is not None:
            stream.stop()
        logger.info("%s disconnected.", self)
