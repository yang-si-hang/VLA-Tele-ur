"""Configuration for the Force Dimension Sigma teleoperator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("sigma")
@dataclass
class SigmaConfig(TeleoperatorConfig):
    """Configuration for reading a Sigma end-effector pose.

    The Force Dimension SDK reports position in metres and orientation as a
    rotation matrix in the device base frame. ``Sigma`` converts the matrix to
    the rotation-vector representation used by ``lerobot_robot_ur``.

    The configured rigid transforms are applied as::

        p_out = position_offset_m + position_scale * R_base @ p_sigma
        R_out = R_base @ R_sigma @ R_tool

    All rotation offsets are rotation vectors in radians.
    """

    sdk_path: Path = Path("/opt/forcedimension/sdk")

    # Select at most one. When both are None, the first available device opens.
    device_index: int | None = None
    serial_number: int | None = None
    require_sigma_device: bool = True
    enable_gravity_compensation: bool = True
    force_refresh_frequency_hz: float = 200.0

    # setting from teleoperator slave config
    position_scale: float = 1.0
    position_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_rotation_offset_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tool_rotation_offset_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.device_index is not None and self.serial_number is not None:
            raise ValueError("device_index and serial_number are mutually exclusive")
        if self.device_index is not None and self.device_index < 0:
            raise ValueError("device_index must be non-negative or None")
        if self.serial_number is not None and self.serial_number < 0:
            raise ValueError("serial_number must be non-negative or None")
        if not str(self.sdk_path):
            raise ValueError("sdk_path must not be empty")
        if not math.isfinite(self.position_scale) or self.position_scale <= 0:
            raise ValueError("position_scale must be finite and positive")
        if (
            not math.isfinite(self.force_refresh_frequency_hz)
            or self.force_refresh_frequency_hz <= 0
        ):
            raise ValueError("force_refresh_frequency_hz must be finite and positive")
        self._validate_vector(self.position_offset_m, "position_offset_m")
        self._validate_vector(self.base_rotation_offset_rad, "base_rotation_offset_rad")
        self._validate_vector(self.tool_rotation_offset_rad, "tool_rotation_offset_rad")

    @staticmethod
    def _validate_vector(values: tuple[float, float, float], name: str) -> None:
        if len(values) != 3:
            raise ValueError(f"{name} must contain exactly 3 values")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"{name} must contain only finite values")
