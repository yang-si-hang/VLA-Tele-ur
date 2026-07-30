"""Public API for the Universal Robots LeRobot integration.

Importing this package exposes the registered ``URRobotConfig`` configuration
and the ``URRobot`` runtime adapter, so applications can construct the robot
without depending on the package's internal control modules.
"""

from .config_ur import URRobotConfig
from .ur import URRobot

__all__ = ["URRobot", "URRobotConfig"]
