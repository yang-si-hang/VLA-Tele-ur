"""LeRobot 0.6 integration for Universal Robots TCP pose control."""

from __future__ import annotations

import logging
import math
from functools import cached_property
from typing import Any

import numpy as np
from lerobot.robots import Robot
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from scipy.spatial.transform import Rotation

from .config_ur import URRobotConfig
from .robotiq_gripper import RobotiqGripper
from .ur_control import URControl


logger = logging.getLogger(__name__)


TCP_FEATURES = ("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz")  # rot vector: r = [wx, wy, wz], theta=||r||
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
            servo_lookahead_time=config.servo_lookahead_time,
            servo_gain=config.servo_gain,
            action_timeout_s=config.action_timeout_s,
            command_timeout_s=config.command_timeout_s,
        )
        self.gripper: RobotiqGripper | None = None
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
        socket = getattr(self.gripper, "socket", None)
        try:
            return socket is not None and socket.fileno() != -1
        except OSError:
            return False

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate  # UR TCP pose control has no LeRobot-side calibration.
        try:
            self.ur_control.connect()
            if self.config.use_gripper:
                self.gripper = RobotiqGripper()
                self.gripper.connect(self.config.robot_ip, self.config.gripper_port)
                self._gripper_connected = True
                self.gripper.activate(auto_calibrate=self.config.gripper_auto_calibrate)
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
        observation: RobotObservation = {
            name: float(value) for name, value in zip(TCP_FEATURES, tcp_pose, strict=True)
        }
        if self.config.use_gripper:
            observation[GRIPPER_FEATURE] = self._read_gripper_position()
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        requested_pose = self._extract_tcp_pose(action)

        requested_gripper: float | None = None
        if self.config.use_gripper:
            if GRIPPER_FEATURE not in action:
                raise ValueError(f"Action is missing required key: {GRIPPER_FEATURE}")
            requested_gripper = self._finite_float(action[GRIPPER_FEATURE], GRIPPER_FEATURE)
            requested_gripper = float(np.clip(requested_gripper, 0.0, 1.0))

        current_pose = self.ur_control.get_tcp_pose()
        safe_pose = self._limit_tcp_step(current_pose, requested_pose)
        sent_pose = self.ur_control.set_tcp_pose(safe_pose)
        sent_action: RobotAction = {
            name: float(value) for name, value in zip(TCP_FEATURES, sent_pose, strict=True)
        }

        if requested_gripper is not None:
            sent_action[GRIPPER_FEATURE] = self._send_gripper_position(requested_gripper)
        return sent_action

    def disconnect(self) -> None:
        self._disconnect_resources(raise_errors=True)
        logger.info("%s disconnected", self)

    def _extract_tcp_pose(self, action: RobotAction) -> list[float]:
        missing = [name for name in TCP_FEATURES if name not in action]
        if missing:
            raise ValueError(f"Action is missing required TCP pose keys: {missing}")
        return [self._finite_float(action[name], name) for name in TCP_FEATURES]

    def _limit_tcp_step(self, current_pose: list[float], target_pose: list[float]) -> list[float]:  # TODO: 计算逻辑
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
        return self._raw_gripper_to_normalized(self.gripper.get_current_position())

    def _send_gripper_position(self, normalized_position: float) -> float:
        if self.gripper is None:
            raise RuntimeError("Robotiq gripper is not available")
        raw_position = self._normalized_gripper_to_raw(normalized_position)
        actual_raw_position = raw_position
        if raw_position != self._last_gripper_command:
            acknowledged, actual_raw_position = self.gripper.move(
                raw_position,
                self.config.gripper_speed,
                self.config.gripper_force,
            )
            if not acknowledged:
                raise RuntimeError("Robotiq gripper did not acknowledge the move command")
            self._last_gripper_command = actual_raw_position
        return self._raw_gripper_to_normalized(actual_raw_position)

    def _normalized_gripper_to_raw(self, normalized_position: float) -> int:
        minimum, maximum = self._gripper_range()
        return round(minimum + normalized_position * (maximum - minimum))

    def _raw_gripper_to_normalized(self, raw_position: int) -> float:
        minimum, maximum = self._gripper_range()
        return float(np.clip((raw_position - minimum) / (maximum - minimum), 0.0, 1.0))

    def _gripper_range(self) -> tuple[int, int]:
        if self.gripper is None:
            raise RuntimeError("Robotiq gripper is not available")
        minimum = self.gripper.get_min_position()
        maximum = self.gripper.get_max_position()
        if maximum <= minimum:
            raise RuntimeError(f"Invalid Robotiq gripper range: [{minimum}, {maximum}]")
        return minimum, maximum

    def _disconnect_resources(self, *, raise_errors: bool) -> None:
        errors: list[BaseException] = []
        try:
            self.ur_control.disconnect()
        except BaseException as exc:
            errors.append(exc)

        if self.gripper is not None and self._gripper_connected:
            try:
                self.gripper.disconnect()
            except BaseException as exc:
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
