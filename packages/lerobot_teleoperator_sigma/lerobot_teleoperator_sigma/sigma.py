"""LeRobot 0.6 teleoperator integration for Force Dimension Sigma devices."""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any

import numpy as np
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from scipy.spatial.transform import Rotation

from .config_sigma import SigmaConfig
from .sigma_control import SigmaDevice


logger = logging.getLogger(__name__)


POSE_FEATURES = ("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz")
GRIPPER_ANGLE_FEATURE = "ee.gripper_angle_rad"


class Sigma(Teleoperator):
    """Expose the Sigma end-effector pose as a LeRobot Cartesian action."""

    config_class = SigmaConfig
    name = "sigma"

    def __init__(self, config: SigmaConfig):
        super().__init__(config)
        self.config = config
        self.sigma_device = SigmaDevice(
            sdk_path=config.sdk_path,
            device_index=config.device_index,
            serial_number=config.serial_number,
            require_sigma_device=config.require_sigma_device,
            force_refresh_frequency_hz=config.force_refresh_frequency_hz,
        )
        self._position_offset = np.asarray(config.position_offset_m, dtype=np.float64)
        self._base_rotation = Rotation.from_rotvec(config.base_rotation_offset_rad)
        self._tool_rotation = Rotation.from_rotvec(config.tool_rotation_offset_rad)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {name: float for name in (*POSE_FEATURES, GRIPPER_ANGLE_FEATURE)}

    @property
    def feedback_features(self) -> dict[str, type]:
        # Force/torque/gripper feedback is intentionally outside the current scope.
        return {}

    @property
    def is_connected(self) -> bool:
        return self.sigma_device.is_connected

    @property
    def device_status(self) -> Any | None:
        """Most recently read Force Dimension device status snapshot."""

        return self.sigma_device.status

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate  # The Sigma performs its own hardware-level initialization.
        try:
            self.sigma_device.connect()
            self.set_device_mode("force")
            self.configure()
            self.sigma_device.start_force_refresh()
            # print(f"Device status: {self.sigma_device.status}")
        except BaseException:
            self.sigma_device.disconnect()
            raise
        logger.info("%s connected", self)

    @property
    def is_calibrated(self) -> bool:
        # No LeRobot-side encoder calibration is required for Cartesian SDK poses.
        return True

    def calibrate(self) -> None:
        """No LeRobot-side calibration is required."""

    @check_if_not_connected
    def configure(self) -> None:
        """Apply Sigma hardware options after the device is connected."""

        self.sigma_device.set_gravity_compensation(
            self.config.enable_gravity_compensation
        )

    @check_if_not_connected
    def get_device_status(self) -> Any:
        """Read and return the current Force Dimension device status."""

        return self.sigma_device.status

    @check_if_not_connected
    def set_device_mode(self, mode: str) -> Any:
        """Set the Force Dimension device mode and return its new status."""

        return self.sigma_device.set_device_mode(mode)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self.sigma_device.check_force_refresh()
        pose = self.sigma_device.read_pose()

        position = (
            self._position_offset
            + self.config.position_scale * self._base_rotation.apply(pose.position_m)
        )
        orientation = (
            self._base_rotation
            * Rotation.from_matrix(pose.rotation_matrix)
            * self._tool_rotation
        )
        rotation_vector = orientation.as_rotvec()
        gripper_angle_rad = self.sigma_device.read_gripper_angle_rad()

        values = np.concatenate((position, rotation_vector))
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Mapped Sigma pose contains non-finite values")
        action = {
            name: float(value) for name, value in zip(POSE_FEATURES, values, strict=True)
        }
        action[GRIPPER_ANGLE_FEATURE] = gripper_angle_rad
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Accept only empty feedback until haptic feedback is implemented."""

        if feedback:
            raise NotImplementedError(
                "Sigma force/torque/gripper feedback is not implemented; expected an empty mapping"
            )

    def disconnect(self) -> None:
        self.sigma_device.disconnect()
        logger.info("%s disconnected", self)
