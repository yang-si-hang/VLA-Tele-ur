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

    # ``None`` lets ur_rtde select the controller's native frequency
    # (typically 500 Hz for e-Series and 125 Hz for CB-Series).
    rtde_frequency_hz: float | None = None
    servo_lookahead_time: float = 0.1
    servo_gain: float = 800.0
    action_timeout_s: float = 0.25
    command_timeout_s: float = 1.0

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
        if self.rtde_frequency_hz is not None and self.rtde_frequency_hz <= 0:
            raise ValueError("rtde_frequency_hz must be positive or None")
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
