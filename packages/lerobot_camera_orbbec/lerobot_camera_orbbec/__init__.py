"""Public API for the Orbbec Gemini LeRobot camera integration.

The package exports the registered ``OrbbecCameraConfig`` type together with
the ``OrbbecCamera`` adapter while keeping SDK-specific stream helpers
internal.
"""

from .config_orbbec import OrbbecCameraConfig
from .orbbec import OrbbecCamera

__all__ = ["OrbbecCamera", "OrbbecCameraConfig"]
