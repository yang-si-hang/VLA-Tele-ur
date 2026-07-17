"""LeRobot 0.6 camera implementation for Orbbec Gemini cameras."""

from __future__ import annotations

import logging
import time
from threading import Condition, Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.cameras import Camera
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.utils import get_cv2_rotation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from .config_orbbec import OrbbecCameraConfig
from .orbbec_utils import OrbbecColorStream, find_orbbec_cameras

logger = logging.getLogger(__name__)


class OrbbecCamera(Camera):
    """Capture RGB images from one Orbbec camera through pyorbbecSDK v2.

    A single background thread owns all blocking SDK reads. ``read`` and
    ``async_read`` consume the newest unconsumed frame, while ``read_latest``
    peeks at the most recent buffered frame without waiting.
    """

    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config)
        self.config = config
        self.serial_number_or_name = config.serial_number_or_name
        self.serial_number: str | None = None
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
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
        self.latest_frame: NDArray[np.uint8] | None = None
        self.latest_timestamp: float | None = None
        self._frame_sequence = 0
        self._consumed_sequence = 0
        self._read_error: BaseException | None = None

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
        stream = OrbbecColorStream(self.serial_number_or_name)
        try:
            profile = stream.start(
                width=self.capture_width,
                height=self.capture_height,
                fps=self.fps,
            )
            self._stream = stream
            self.serial_number = stream.serial_number

            if (
                profile.width != self.capture_width
                or profile.height != self.capture_height
                or profile.fps != self.fps
            ):
                raise RuntimeError(
                    f"{self} started an unexpected color profile: {profile.width}x{profile.height} "
                    f"at {profile.fps} FPS."
                )

            self._start_read_thread()
            if warmup:
                self._warmup()
        except Exception as exc:
            self._stop_read_thread()
            stream.stop()
            self._stream = None
            self.serial_number = None
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

    def _read_loop(self) -> None:
        stop_event = self.stop_event
        stream = self._stream
        if stop_event is None or stream is None:
            return

        consecutive_failures = 0
        while not stop_event.is_set():
            try:
                image = stream.read_rgb(timeout_ms=200)
                if image is None:
                    continue
                processed = self._postprocess_image(image)
                capture_time = time.perf_counter()
                with self._frame_condition:
                    self.latest_frame = processed
                    self.latest_timestamp = capture_time
                    self._frame_sequence += 1
                    self._frame_condition.notify_all()
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
            self.latest_timestamp = None
            self._frame_sequence = 0
            self._consumed_sequence = 0
            self._read_error = None

    def _raise_if_reader_failed(self) -> None:
        if self._read_error is not None:
            raise RuntimeError(f"{self} background read thread failed.") from self._read_error
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

    @check_if_not_connected
    def read(self) -> NDArray[Any]:
        """Block until a new RGB/BGR frame is available."""

        start = time.perf_counter()
        frame = self.async_read(timeout_ms=10_000)
        logger.debug("%s read took %.1f ms", self, (time.perf_counter() - start) * 1000.0)
        return frame

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Return the latest unconsumed frame, waiting up to ``timeout_ms``."""

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
        return frame.copy()

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Immediately return the newest buffered frame without consuming it."""

        if max_age_ms < 0:
            raise ValueError("`max_age_ms` must be non-negative.")
        self._raise_if_reader_failed()

        with self._frame_condition:
            frame = self.latest_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1000.0
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is {age_ms:.1f} ms old (maximum allowed: {max_age_ms} ms)."
            )
        return frame.copy()

    def disconnect(self) -> None:
        """Stop capture and release the pyorbbecSDK pipeline."""

        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._stop_read_thread()
        stream = self._stream
        self._stream = None
        self.serial_number = None
        if stream is not None:
            stream.stop()
        logger.info("%s disconnected.", self)
