"""Expose Universal Robots TCP pose control through the LeRobot robot API.

``URRobot`` translates LeRobot observation and action dictionaries between
Cartesian position plus 6D rotation features and the UR axis-angle TCP pose
used by ``ur_rtde``. It delegates real-time arm streaming to ``URControl``,
drives an optional Robotiq gripper through host USB-RS485, validates and limits
pose steps, and manages both devices as one LeRobot connection.

``rtde_check_pose_safety`` is disabled by default to reduce blocking time and
help maintain the ``servoL`` control frequency.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from functools import cached_property
from typing import Any

import numpy as np
from lerobot.robots import Robot
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from scipy.spatial.transform import Rotation

from .config_ur import URRobotConfig
from .robotiq_usb_high_freq import RobotiqGripperUSBHighFrequency
from .ur_control import URControl, rot6d_to_rotation, rotation_to_rot6d

logger = logging.getLogger(__name__)


TCP_POSITION_FEATURES = ("ee.x", "ee.y", "ee.z")
TCP_ROTATION_6D_FEATURES = ("ee.rot6d_0", "ee.rot6d_1", "ee.rot6d_2", "ee.rot6d_3", "ee.rot6d_4", "ee.rot6d_5")
TCP_FEATURES = (*TCP_POSITION_FEATURES, *TCP_ROTATION_6D_FEATURES)
GRIPPER_FEATURE = "ee.gripper_pos"


class URRobot(Robot):
    config_class = URRobotConfig
    name = "ur"

    def __init__(self, config: URRobotConfig):
        super().__init__(config)
        self.config = config
        self.ur_control = URControl(
            config.robot_ip,
            rtde_frequency_hz=config.rtde_frequency_hz,
            tcp_pose=config.tcp_pose,
            servo_lookahead_time=config.servo_lookahead_time,
            servo_gain=config.servo_gain,
            action_timeout_s=config.action_timeout_s,
            command_timeout_s=config.command_timeout_s,
            check_pose_safety=config.rtde_check_pose_safety,
        )
        self.gripper: RobotiqGripperUSBHighFrequency | None = None
        self._gripper_connected = False
        self._last_gripper_command: int | None = None

    @cached_property
    def observation_features(self) -> dict[str, type]:
        features = {name: float for name in TCP_FEATURES}
        if self.config.use_gripper:
            features[GRIPPER_FEATURE] = float
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self.observation_features.copy()

    @property
    def is_connected(self) -> bool:
        if not self.ur_control.is_connected:
            return False
        if not self.config.use_gripper:
            return True
        if not self._gripper_connected or self.gripper is None:
            return False
        return self.gripper.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate  # UR TCP pose control has no LeRobot-side calibration.
        try:
            self.ur_control.connect()
            if self.config.use_gripper:
                self.gripper = RobotiqGripperUSBHighFrequency(
                    self.config.gripper_usb_port,
                    slave_id=self.config.gripper_slave_id,
                    baudrate=self.config.gripper_baudrate,
                    control_frequency_hz=self.config.gripper_control_frequency_hz,
                    serial_timeout_s=self.config.gripper_serial_timeout_s,
                    max_consecutive_errors=self.config.gripper_max_consecutive_errors,
                )
                self.gripper.connect(auto_activate=True)
                self._gripper_connected = True
            self.configure()
        except BaseException:
            self._disconnect_resources(raise_errors=False)
            raise
        logger.info("%s connected", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """No LeRobot-side calibration is required for UR Cartesian poses."""

    @check_if_not_connected
    def configure(self) -> None:
        """No additional controller configuration is currently required."""

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        tcp_pose = self.ur_control.get_tcp_pose()
        observation = self._tcp_pose_to_features(tcp_pose)
        if self.config.use_gripper:
            observation[GRIPPER_FEATURE] = self._read_gripper_position()
        return observation

    @check_if_not_connected
    def send_action(
        self,
        action: RobotAction,
        current_observation: RobotObservation,
        step_limit_check: bool = True,
    ) -> RobotAction:
        requested_pose = self._extract_tcp_pose(action)

        requested_gripper: float | None = None
        if self.config.use_gripper:
            if GRIPPER_FEATURE not in action:
                raise ValueError(f"Action is missing required key: {GRIPPER_FEATURE}")
            requested_gripper = self._finite_float(action[GRIPPER_FEATURE], GRIPPER_FEATURE)
            requested_gripper = float(np.clip(requested_gripper, 0.0, 1.0))

        current_pose = self._extract_tcp_pose(current_observation)
        if step_limit_check:
            safe_pose = self._limit_tcp_step(current_pose, requested_pose)
        else:
            safe_pose = requested_pose
        sent_pose = self.ur_control.set_tcp_pose(safe_pose)
        sent_action = self._tcp_pose_to_features(sent_pose)

        if requested_gripper is not None:
            sent_action[GRIPPER_FEATURE] = self._send_gripper_position(requested_gripper)
        return sent_action

    @check_if_not_connected
    def move_to_action(self, action: RobotAction, *, speed: float = 0.25, acceleration: float = 0.5) -> RobotAction:
        """Move linearly to one absolute action pose before streamed control."""

        requested_pose = self._extract_tcp_pose(action)
        moved_pose = self.ur_control.move_tcp_pose(requested_pose, speed=speed, acceleration=acceleration)
        moved_action = self._tcp_pose_to_features(moved_pose)
        if self.config.use_gripper:
            if GRIPPER_FEATURE not in action:
                raise ValueError(f"Action is missing required key: {GRIPPER_FEATURE}")
            requested_gripper = self._finite_float(action[GRIPPER_FEATURE], GRIPPER_FEATURE)
            moved_action[GRIPPER_FEATURE] = self._send_gripper_position(float(np.clip(requested_gripper, 0.0, 1.0)))
        return moved_action

    def disconnect(self) -> None:
        self._disconnect_resources(raise_errors=True)
        logger.info("%s disconnected", self)

    def _extract_tcp_pose(self, action: RobotAction) -> list[float]:
        missing = [name for name in TCP_FEATURES if name not in action]
        if missing:
            raise ValueError(f"Action is missing required TCP pose keys: {missing}")

        position = [
            self._finite_float(action[name], name) for name in TCP_POSITION_FEATURES
        ]
        rot6d = [
            self._finite_float(action[name], name) for name in TCP_ROTATION_6D_FEATURES
        ]
        try:
            rotation_vector = rot6d_to_rotation(rot6d).as_rotvec()
        except ValueError as exc:
            raise ValueError("Action contains an invalid Rot6D orientation") from exc
        return [*position, *rotation_vector.tolist()]

    @staticmethod
    def _tcp_pose_to_features(tcp_pose: Sequence[float]) -> RobotAction:
        values = np.asarray(tcp_pose, dtype=np.float64)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("TCP pose must contain 6 finite values")

        rot6d = rotation_to_rot6d(Rotation.from_rotvec(values[3:]))
        feature_values = np.concatenate((values[:3], rot6d))
        return {
            name: float(value)
            for name, value in zip(TCP_FEATURES, feature_values, strict=True)
        }

    def _limit_tcp_step(self, current_pose: list[float], target_pose: list[float]) -> list[float]:
        current = np.asarray(current_pose, dtype=np.float64)
        target = np.asarray(target_pose, dtype=np.float64)
        limited = target.copy()
        clipped = False

        max_translation = self.config.max_tcp_translation_delta_m
        translation_delta = target[:3] - current[:3]
        translation_distance = float(np.linalg.norm(translation_delta))
        if max_translation is not None and translation_distance > max_translation:
            limited[:3] = current[:3] + translation_delta * (max_translation / translation_distance)
            clipped = True

        max_rotation = self.config.max_tcp_rotation_delta_rad
        current_rotation = Rotation.from_rotvec(current[3:])
        target_rotation = Rotation.from_rotvec(target[3:])
        rotation_delta = target_rotation * current_rotation.inv()
        rotation_delta_vector = rotation_delta.as_rotvec()
        rotation_distance = float(np.linalg.norm(rotation_delta_vector))
        if max_rotation is not None and rotation_distance > max_rotation:
            clipped_delta = Rotation.from_rotvec(
                rotation_delta_vector * (max_rotation / rotation_distance)
            )
            limited[3:] = (clipped_delta * current_rotation).as_rotvec()
            clipped = True

        if clipped:
            logger.warning(
                "TCP target was limited to %.3f m translation and %.3f rad rotation per action",
                self.config.max_tcp_translation_delta_m or math.inf,
                self.config.max_tcp_rotation_delta_rad or math.inf,
            )
        return limited.tolist()

    def _read_gripper_position(self) -> float:
        if self.gripper is None:
            raise RuntimeError("Robotiq gripper is not available")
        worker_error = self.gripper.get_last_error()
        if worker_error is not None:
            raise RuntimeError("Robotiq gripper worker failed") from worker_error
        state = self.gripper.get_state()
        if state.sequence <= 0 or state.response_received_ns <= 0:
            raise RuntimeError("Robotiq gripper position cache is unavailable")
        if (time.monotonic_ns() - state.response_received_ns) / 1e9 > self.config.gripper_cache_max_age_s:
            raise RuntimeError(
                "Robotiq gripper position cache is unavailable or stale"
            )
        if state.fault != 0:
            raise RuntimeError(f"Robotiq gripper fault: 0x{state.fault:02X}")
        return self._raw_gripper_to_normalized(state.position)

    def _send_gripper_position(self, normalized_position: float) -> float:
        if self.gripper is None:
            raise RuntimeError("Robotiq gripper is not available")
        raw_position = self._normalized_gripper_to_raw(normalized_position)
        if raw_position != self._last_gripper_command:
            self.gripper.set_target(
                raw_position,
                self.config.gripper_speed,
                self.config.gripper_force,
            )
            self._last_gripper_command = raw_position
        return self._raw_gripper_to_normalized(raw_position)

    def _normalized_gripper_to_raw(self, normalized_position: float) -> int:
        minimum, maximum = self._gripper_range()
        return round(minimum + normalized_position * (maximum - minimum))

    def _raw_gripper_to_normalized(self, raw_position: int) -> float:
        minimum, maximum = self._gripper_range()
        return float(np.clip((raw_position - minimum) / (maximum - minimum), 0.0, 1.0))

    def _gripper_range(self) -> tuple[int, int]:
        minimum = 0
        maximum = 255
        if maximum <= minimum:
            raise RuntimeError(f"Invalid Robotiq gripper range: [{minimum}, {maximum}]")
        return minimum, maximum

    def _disconnect_resources(self, *, raise_errors: bool) -> None:
        errors: list[BaseException] = []
        try:
            self.ur_control.disconnect()
        except BaseException as exc:  # noqa: BLE001 - collect all disconnect failures.
            errors.append(exc)

        if self.gripper is not None:
            try:
                self.gripper.disconnect()
            except BaseException as exc:  # noqa: BLE001 - collect all disconnect failures.
                errors.append(exc)
        self._gripper_connected = False
        self._last_gripper_command = None
        self.gripper = None

        if errors and raise_errors:
            raise RuntimeError("Errors occurred while disconnecting URRobot") from errors[0]

    @staticmethod
    def _finite_float(value: Any, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a real number") from exc
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result
