"""Define and validate configuration for the LeRobot Universal Robots driver.

``URRobotConfig`` is registered under the ``ur`` robot type and groups the
network, RTDE servo, pose-safety, and optional Robotiq gripper settings used by
``URRobot``. Its post-initialization checks reject invalid values before any
hardware connection or control thread is started.
"""

import math
from dataclasses import dataclass

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("ur")
@dataclass
class URRobotConfig(RobotConfig):
    """Configuration for a Universal Robots arm controlled through ``ur_rtde``."""

    robot_ip: str

    use_gripper: bool = True
    gripper_port: int = 63352
    gripper_speed: int = 255
    gripper_force: int = 150
    gripper_auto_calibrate: bool = False
    gripper_position_poll_frequency_hz: float = 60.0
    gripper_status_poll_frequency_hz: float = 1.0
    gripper_cache_max_age_s: float = 0.1

    # ``None`` lets ur_rtde select the controller's native frequency
    # (typically 500 Hz for e-Series and 125 Hz for CB-Series).
    rtde_frequency_hz: float | None = None
    # Transformation from the output flange to the active TCP: [x, y, z, rx, ry, rz].
    tcp_pose: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    servo_lookahead_time: float = 0.1
    servo_gain: float = 800.0
    action_timeout_s: float = 0.25      # 动作指令有效时间, 超过则停止伺服
    command_timeout_s: float = 1.0      # 命令等待时间, 超过报错
    check_pose_safety: bool = True

    # Maximum change accepted from one LeRobot action to the next measured pose.
    # Set either value to ``None`` to disable that limit.
    max_tcp_translation_delta_m: float | None = 0.02
    max_tcp_rotation_delta_rad: float | None = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()

        if not self.robot_ip.strip():
            raise ValueError("robot_ip must not be empty")
        if not 1 <= self.gripper_port <= 65535:
            raise ValueError("gripper_port must be in [1, 65535]")
        if not 0 <= self.gripper_speed <= 255:
            raise ValueError("gripper_speed must be in [0, 255]")
        if not 0 <= self.gripper_force <= 255:
            raise ValueError("gripper_force must be in [0, 255]")
        if (
            not math.isfinite(self.gripper_position_poll_frequency_hz)
            or self.gripper_position_poll_frequency_hz <= 0
        ):
            raise ValueError("gripper_position_poll_frequency_hz must be positive")
        if (
            not math.isfinite(self.gripper_status_poll_frequency_hz)
            or self.gripper_status_poll_frequency_hz <= 0
        ):
            raise ValueError("gripper_status_poll_frequency_hz must be positive")
        if (
            not math.isfinite(self.gripper_cache_max_age_s)
            or self.gripper_cache_max_age_s <= 0
        ):
            raise ValueError("gripper_cache_max_age_s must be positive")
        if self.rtde_frequency_hz is not None and self.rtde_frequency_hz <= 0:
            raise ValueError("rtde_frequency_hz must be positive or None")
        if len(self.tcp_pose) != 6:
            raise ValueError("tcp_pose must contain 6 values")
        if not all(math.isfinite(value) for value in self.tcp_pose):
            raise ValueError("tcp_pose must contain only finite values")
        if not 0.03 <= self.servo_lookahead_time <= 0.2:
            raise ValueError("servo_lookahead_time must be in [0.03, 0.2]")
        if not 100 <= self.servo_gain <= 2000:
            raise ValueError("servo_gain must be in [100, 2000]")
        if self.action_timeout_s <= 0:
            raise ValueError("action_timeout_s must be positive")
        if self.command_timeout_s <= 0:
            raise ValueError("command_timeout_s must be positive")
        if (
            self.max_tcp_translation_delta_m is not None
            and self.max_tcp_translation_delta_m <= 0
        ):
            raise ValueError("max_tcp_translation_delta_m must be positive or None")
        if self.max_tcp_rotation_delta_rad is not None and self.max_tcp_rotation_delta_rad <= 0:
            raise ValueError("max_tcp_rotation_delta_rad must be positive or None")
