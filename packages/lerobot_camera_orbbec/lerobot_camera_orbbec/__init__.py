"""Public API for the Orbbec Gemini LeRobot camera integration.

The package exports the registered ``OrbbecCameraConfig`` type together with
the ``OrbbecCamera`` adapter and its metadata-bearing ``OrbbecFrame`` result
while keeping SDK-specific stream helpers internal.
"""

from .config_orbbec import OrbbecCameraConfig
from .orbbec import OrbbecCamera, OrbbecFrame, OrbbecFrameRateStats

__all__ = [
    "OrbbecCamera",
    "OrbbecCameraConfig",
    "OrbbecFrame",
    "OrbbecFrameRateStats",
]
