"""Public API for the Force Dimension Sigma LeRobot teleoperator integration.

The package re-exports the registered configuration, LeRobot teleoperator,
low-level device wrapper, pose container, and shared action feature names used
to connect Sigma input to robot-control pipelines.
"""

from .config_sigma import SigmaConfig
from .sigma import GRIPPER_ANGLE_FEATURE, POSE_FEATURES, Sigma
from .sigma_control import SigmaDevice, SigmaPose

__all__ = [
    "GRIPPER_ANGLE_FEATURE",
    "POSE_FEATURES",
    "Sigma",
    "SigmaConfig",
    "SigmaDevice",
    "SigmaPose",
]
