"""Implement low-level Force Dimension Sigma access with ``forcedimension_core``.

``SigmaDevice`` loads the configured SDK, selects and validates one device,
owns its DHD connection, and provides thread-safe pose, gripper, status, mode,
and gravity-compensation operations. A dedicated refresh thread repeatedly
sends a zero force/torque/gripper wrench required by force mode. The module
never sends non-zero haptic feedback or robot position commands.
"""

from __future__ import annotations

import ctypes
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SigmaPose:
    """One Sigma end-effector pose expressed in the SDK device base frame."""

    position_m: np.ndarray
    rotation_matrix: np.ndarray


class SigmaDevice:
    """Own one connection to a Force Dimension Sigma device."""

    def __init__(
        self,
        *,
        sdk_path: str | Path = "/opt/forcedimension/sdk",
        device_index: int | None = None,
        serial_number: int | None = None,
        require_sigma_device: bool = True,
        force_refresh_frequency_hz: float = 200.0,
    ) -> None:
        if device_index is not None and serial_number is not None:
            raise ValueError("device_index and serial_number are mutually exclusive")
        if (
            not math.isfinite(force_refresh_frequency_hz)
            or force_refresh_frequency_hz <= 0
        ):
            raise ValueError("force_refresh_frequency_hz must be finite and positive")

        self.sdk_path = Path(sdk_path).expanduser()
        self.device_index = device_index
        self.serial_number = serial_number
        self.require_sigma_device = require_sigma_device
        self.force_refresh_frequency_hz = float(force_refresh_frequency_hz)

        self._dhd: Any | None = None
        self._constants: Any | None = None
        self._containers: Any | None = None
        self._device_id: int | None = None
        self._device_name: str | None = None
        self._status: Any | None = None
        self._lock = threading.RLock()
        self._force_refresh_stop = threading.Event()
        self._force_refresh_thread: threading.Thread | None = None
        self._force_refresh_error: BaseException | None = None

    @property
    def is_connected(self) -> bool:
        return self._device_id is not None

    @property
    def device_id(self) -> int | None:
        return self._device_id

    @property
    def device_name(self) -> str | None:
        return self._device_name

    @property
    def is_force_refresh_running(self) -> bool:
        with self._lock:
            thread = self._force_refresh_thread
            return thread is not None and thread.is_alive()

    @property
    def status(self) -> Any | None:
        """Read the current ``forcedimension_core.containers.Status`` snapshot."""

        with self._lock:
            if not self.is_connected:
                return None
            if self._dhd is None or self._status is None:
                raise ConnectionError("SigmaDevice connection state is incomplete")

            result = self._dhd.getStatus(self._status, self._device_id)
            if result < 0:
                raise RuntimeError(
                    f"Failed to read Sigma status: {self._last_error(self._dhd)}"
                )

            return self._status

    def connect(self) -> None:
        with self._lock:
            if self.is_connected:
                raise RuntimeError("SigmaDevice is already connected")

            dhd, constants, containers = self._load_forcedimension()
            device_id = self._open_device(dhd)
            if device_id < 0:
                raise ConnectionError(
                    f"Failed to open Force Dimension device: {self._last_error(dhd)}"
                )

            self._dhd = dhd
            self._constants = constants
            self._containers = containers
            self._device_id = int(device_id)
            self._status = containers.Status()

            try:
                self._validate_open_device()
                # Fail during connect, rather than on the first LeRobot loop,
                # if the SDK cannot return a complete 6-DoF pose.
                self.read_pose()
            except BaseException:
                self._close_unchecked()
                raise

            logger.info(
                "Connected to %s (Force Dimension device ID %d)",
                self._device_name,
                self._device_id,
            )

    def read_pose(self) -> SigmaPose:
        with self._lock:
            if not self.is_connected or self._dhd is None:
                raise ConnectionError("SigmaDevice is not connected")

            position = [0.0, 0.0, 0.0]
            rotation = [[0.0, 0.0, 0.0] for _ in range(3)]
            result = self._dhd.getPositionAndOrientationFrame(
                position,
                rotation,
                self._device_id,
            )
            if result < 0:
                raise RuntimeError(
                    f"Failed to read Sigma pose: {self._last_error(self._dhd)}"
                )

            position_array = np.asarray(position, dtype=np.float64)
            rotation_array = np.asarray(rotation, dtype=np.float64)
            if position_array.shape != (3,) or rotation_array.shape != (3, 3):
                raise RuntimeError(
                    "Force Dimension SDK returned an invalid pose shape: "
                    f"{position_array.shape}, {rotation_array.shape}"
                )
            if not np.all(np.isfinite(position_array)) or not np.all(
                np.isfinite(rotation_array)
            ):
                raise RuntimeError("Force Dimension SDK returned a non-finite pose")

            # A valid SDK rotation should already be orthonormal. Reject gross
            # corruption while allowing normal floating-point measurement noise.
            determinant = float(np.linalg.det(rotation_array))
            orthogonality_error = float(
                np.linalg.norm(rotation_array.T @ rotation_array - np.eye(3), ord="fro")
            )
            if (
                not math.isfinite(determinant)
                or abs(determinant - 1.0) > 0.05
                or orthogonality_error > 0.05
            ):
                raise RuntimeError(
                    "Force Dimension SDK returned an invalid orientation matrix "
                    f"(determinant={determinant:.6f}, "
                    f"orthogonality_error={orthogonality_error:.6f})"
                )

            return SigmaPose(position_array, rotation_array)

    def read_gripper_angle_rad(self) -> float:
        """Read the signed Sigma gripper opening angle in radians."""

        with self._lock:
            if not self.is_connected or self._dhd is None:
                raise ConnectionError("SigmaDevice is not connected")

            angle = ctypes.c_double()
            result = self._dhd.getGripperAngleRad(
                ctypes.byref(angle),
                self._device_id,
            )
            if result < 0:
                raise RuntimeError(
                    "Failed to read Sigma gripper angle: "
                    f"{self._last_error(self._dhd)}"
                )
            if not math.isfinite(angle.value):
                raise RuntimeError("Force Dimension SDK returned a non-finite gripper angle")

            return float(angle.value)

    def set_device_mode(self, mode: str) -> Any:
        """Set IDLE, FORCE, or BRAKE mode and return the resulting status."""

        requested_mode = str(mode).strip().lower()
        if requested_mode not in {"idle", "force", "brake"}:
            raise ValueError(
                f"Unsupported Sigma device mode {mode!r}; "
                "expected one of: idle, force, brake"
            )

        with self._lock:
            if not self.is_connected or self._dhd is None:
                raise ConnectionError("SigmaDevice is not connected")

            if requested_mode == "idle":
                result = self._dhd.setBrakes(False, self._device_id)
            elif requested_mode == "force":
                result = self._dhd.enableForce(True, self._device_id)
            else:
                result = self._dhd.setBrakes(True, self._device_id)

            if result < 0:
                raise RuntimeError(
                    f"Failed to set Sigma device mode to {requested_mode}: "
                    f"{self._last_error(self._dhd)}"
                )

            logger.info("Sigma device mode set to %s", requested_mode)
            return self.status

    def set_gravity_compensation(self, enable: bool) -> None:
        """Enable or disable the DHD SDK's built-in gravity compensation."""

        with self._lock:
            if not self.is_connected or self._dhd is None:
                raise ConnectionError("SigmaDevice is not connected")

            result = self._dhd.setGravityCompensation(
                bool(enable),
                self._device_id,
            )
            if result < 0:
                raise RuntimeError(
                    "Failed to configure Sigma gravity compensation: "
                    f"{self._last_error(self._dhd)}"
                )

            logger.info(
                "Sigma gravity compensation %s",
                "enabled" if enable else "disabled",
            )

    def set_zero_force_and_torque(self) -> None:
        """Send a zero wrench so DHD can apply gravity compensation."""

        with self._lock:
            if not self.is_connected or self._dhd is None:
                raise ConnectionError("SigmaDevice is not connected")

            result = self._dhd.setForceAndTorque(
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                self._device_id,
            )
            if result < 0:
                raise RuntimeError(
                    "Failed to send zero force and torque to Sigma: "
                    f"{self._last_error(self._dhd)}"
                )

    def start_force_refresh(self) -> None:
        """Start the fixed-rate background zero-wrench refresh loop."""

        with self._lock:
            if not self.is_connected:
                raise ConnectionError("SigmaDevice is not connected")
            if self._force_refresh_thread is not None and self._force_refresh_thread.is_alive():
                raise RuntimeError("Sigma force refresh loop is already running")

            # Surface SDK errors synchronously before launching the worker.
            self.set_zero_force_and_torque()
            self._force_refresh_error = None
            self._force_refresh_stop.clear()
            thread = threading.Thread(
                target=self._force_refresh_loop,
                name=f"sigma-force-refresh-{self._device_id}",
                daemon=True,
            )
            self._force_refresh_thread = thread
            thread.start()

    def stop_force_refresh(self) -> None:
        """Stop and join the background zero-wrench refresh loop."""

        with self._lock:
            thread = self._force_refresh_thread
            if thread is None:
                return
            self._force_refresh_stop.set()

        if thread is threading.current_thread():
            raise RuntimeError("Sigma force refresh loop cannot join itself")

        thread.join(timeout=1.0)
        if thread.is_alive():
            raise RuntimeError("Timed out while stopping Sigma force refresh loop")

        with self._lock:
            if self._force_refresh_thread is thread:
                self._force_refresh_thread = None

        logger.info("Sigma force refresh loop stopped")

    def check_force_refresh(self) -> None:
        """Raise an error if the background force refresh loop has failed."""

        with self._lock:
            error = self._force_refresh_error
        if error is not None:
            raise RuntimeError("Sigma force refresh loop failed") from error

    def _force_refresh_loop(self) -> None:
        """set zero force and torque to make free movement."""
        period_s = 1.0 / self.force_refresh_frequency_hz
        next_cycle = time.perf_counter()

        while not self._force_refresh_stop.is_set():
            next_cycle += period_s
            remaining = next_cycle - time.perf_counter()
            if remaining > 0 and self._force_refresh_stop.wait(remaining):
                return
            if remaining <= 0:
                next_cycle = time.perf_counter()

            try:
                self.set_zero_force_and_torque()
            except BaseException as exc:
                with self._lock:
                    self._force_refresh_error = exc
                self._force_refresh_stop.set()
                return

    def disconnect(self) -> None:
        with self._lock:
            if not self.is_connected:
                return

        self.stop_force_refresh()

        with self._lock:
            device_id = self._device_id
            dhd = self._dhd
            result = self._close_unchecked()
            if result < 0 and dhd is not None:
                raise RuntimeError(
                    f"Failed to close Force Dimension device {device_id}: "
                    f"{self._last_error(dhd)}"
                )
            logger.info("Disconnected Force Dimension device ID %s", device_id)

    def _load_forcedimension(self) -> tuple[Any, Any, Any]:
        sdk_path = self.sdk_path.resolve()
        if not sdk_path.is_dir():
            raise FileNotFoundError(f"Force Dimension SDK path does not exist: {sdk_path}")

        try:
            from forcedimension_core import constants, containers, dhd
        except ImportError as exc:
            raise ImportError(
                "Unable to load forcedimension_core/Force Dimension SDK. "
                f"Expected the Force Dimension SDK under {sdk_path}. Set FDSDK "
                "before Python starts if forcedimension_core was already imported."
            ) from exc
        return dhd, constants, containers

    def _open_device(self, dhd: Any) -> int:
        if self.serial_number is not None:
            return int(dhd.openSerial(self.serial_number))
        if self.device_index is not None:
            return int(dhd.openID(self.device_index))
        return int(dhd.open())

    def _validate_open_device(self) -> None:
        if self._dhd is None or self._constants is None or self._device_id is None:
            raise RuntimeError("Force Dimension SDK connection state is incomplete")

        self._device_name = self._dhd.getSystemName(self._device_id)
        if not self._device_name:
            raise RuntimeError(
                f"Failed to identify Force Dimension device: {self._last_error(self._dhd)}"
            )
        if not self._dhd.hasWrist(self._device_id):
            raise RuntimeError(
                f"{self._device_name} does not expose the wrist orientation required for 6-DoF pose"
            )

        if self.require_sigma_device:
            sigma_types = {
                int(self._constants.DeviceType.SIGMA7_RIGHT),
                int(self._constants.DeviceType.SIGMA7_LEFT),
            }
            device_type = int(self._dhd.getSystemType(self._device_id))
            if device_type not in sigma_types:
                raise RuntimeError(
                    f"Expected a Sigma device, but connected to {self._device_name} "
                    f"(SDK device type {device_type})"
                )

    def _close_unchecked(self) -> int:
        dhd = self._dhd
        device_id = self._device_id
        self._device_id = None
        self._device_name = None
        self._dhd = None
        self._constants = None
        self._containers = None
        self._status = None
        self._force_refresh_thread = None
        self._force_refresh_error = None
        self._force_refresh_stop.set()
        if dhd is None or device_id is None:
            return 0
        return int(dhd.close(device_id))

    @staticmethod
    def _last_error(dhd: Any) -> str:
        try:
            return str(dhd.errorGetLastStr())
        except BaseException:
            return "unknown SDK error"
