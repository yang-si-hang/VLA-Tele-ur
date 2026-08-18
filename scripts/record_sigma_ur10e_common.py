"""Shared UR10e and Sigma dataset-recording helpers."""

from __future__ import annotations

import contextlib
import logging
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.datasets import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor import RobotActionProcessorStep, RobotProcessorPipeline
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.feature_utils import combine_feature_dicts
from scipy.spatial.transform import Rotation

from lerobot_robot_ur import URRobot, URRobotConfig
from lerobot_robot_ur.ur import GRIPPER_FEATURE, TCP_FEATURES
from lerobot_teleoperator_sigma import GRIPPER_ANGLE_FEATURE, Sigma

try:
    from scripts.sigma_ur10e_relative_teleop import (
        CartesianPose,
        action_to_pose,
        capture_sigma_reference,
        compute_relative_target,
        pose_to_action,
        sigma_action_to_pose,
        sigma_gripper_angle_to_ur_position,
    )
except ModuleNotFoundError:  # Support direct execution from the scripts directory.
    from sigma_ur10e_relative_teleop import (  # type: ignore[no-redef]
        CartesianPose,
        action_to_pose,
        capture_sigma_reference,
        compute_relative_target,
        pose_to_action,
        sigma_action_to_pose,
        sigma_gripper_angle_to_ur_position,
    )


class CameraAugmentedURRobot(URRobot):
    """A URRobot variant that includes configured RGB cameras."""

    def __init__(
        self,
        config: URRobotConfig,
        camera_configs: dict[str, CameraConfig],
    ) -> None:
        super().__init__(config)
        self.cameras = make_cameras_from_configs(camera_configs)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features: dict[str, type | tuple[int, int, int]] = {
            name: float for name in TCP_FEATURES
        }
        if self.config.use_gripper:
            features[GRIPPER_FEATURE] = float
        for name, camera in self.cameras.items():
            features[name] = (camera.height, camera.width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        features = {name: float for name in TCP_FEATURES}
        if self.config.use_gripper:
            features[GRIPPER_FEATURE] = float
        return features

    @property
    def is_connected(self) -> bool:
        return super().is_connected and all(
            camera.is_connected for camera in self.cameras.values()
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        connected_cameras = []
        try:
            # Cameras are connected first because URRobot.connect calls configure,
            # whose connection guard resolves this subclass's is_connected property.
            for camera in self.cameras.values():
                camera.connect()
                connected_cameras.append(camera)
            super().connect(calibrate=calibrate)
            for camera in connected_cameras:
                start_monitoring = getattr(camera, "start_frame_rate_monitoring", None)
                if callable(start_monitoring):
                    start_monitoring()
        except BaseException:
            for camera in connected_cameras:
                stop_monitoring = getattr(camera, "stop_frame_rate_monitoring", None)
                if callable(stop_monitoring):
                    with contextlib.suppress(BaseException):
                        stop_monitoring()
            if super().is_connected:
                with contextlib.suppress(BaseException):
                    super().disconnect()
            for camera in reversed(connected_cameras):
                if camera.is_connected:
                    with contextlib.suppress(BaseException):
                        camera.disconnect()
            raise
        logging.info("UR robot and %d cameras connected", len(self.cameras))

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        return self.add_camera_observations(self.get_control_observation())

    def get_control_observation(self) -> RobotObservation:
        """Read robot state without copying camera frames."""

        return super().get_observation()

    def add_camera_observations(
        self,
        observation: RobotObservation,
    ) -> RobotObservation:
        """Return a snapshot augmented with the latest camera frames."""

        augmented_observation = dict(observation)
        for name, camera in self.cameras.items():
            augmented_observation[name] = camera.read_latest()
        return augmented_observation

    def disconnect(self) -> None:
        errors: list[BaseException] = []
        for camera in self.cameras.values():
            stop_monitoring = getattr(camera, "stop_frame_rate_monitoring", None)
            if callable(stop_monitoring):
                try:
                    stop_monitoring()
                except BaseException as exc:
                    errors.append(exc)
        if super().is_connected:
            try:
                super().disconnect()
            except BaseException as exc:
                errors.append(exc)
        for camera in reversed(list(self.cameras.values())):
            if camera.is_connected:
                try:
                    camera.disconnect()
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            raise RuntimeError(
                "Errors occurred while disconnecting the recording robot"
            ) from errors[0]
        logging.info("UR robot and cameras disconnected")


class SigmaRelativeURActionStep(RobotActionProcessorStep):
    """Map Sigma motion relative to a reference into a UR target pose."""

    def __init__(
        self,
        *,
        position_scale: float,
        base_rotation_offset_rad: tuple[float, float, float],
        gripper_closed_angle_rad: float,
        gripper_open_angle_rad: float,
    ) -> None:
        self.position_scale = position_scale
        self.base_mapping = Rotation.from_rotvec(base_rotation_offset_rad)
        self.gripper_closed_angle_rad = gripper_closed_angle_rad
        self.gripper_open_angle_rad = gripper_open_angle_rad
        self.sigma_reference: CartesianPose | None = None
        self.ur_reference: CartesianPose | None = None

    def set_reference(
        self,
        *,
        sigma_reference: CartesianPose,
        ur_reference: CartesianPose,
    ) -> None:
        self.sigma_reference = sigma_reference
        self.ur_reference = ur_reference

    def clear_reference(self) -> None:
        self.sigma_reference = None
        self.ur_reference = None

    def action(self, action: RobotAction) -> RobotAction:
        if self.sigma_reference is None or self.ur_reference is None:
            raise RuntimeError("Sigma and UR episode references have not been captured")
        if GRIPPER_ANGLE_FEATURE not in action:
            raise ValueError(
                f"Sigma action is missing required key: {GRIPPER_ANGLE_FEATURE}"
            )

        sigma_current = sigma_action_to_pose(action)
        target = compute_relative_target(
            self.sigma_reference,
            sigma_current,
            self.ur_reference,
            position_scale=self.position_scale,
            base_mapping=self.base_mapping,
        )
        mapped_action = pose_to_action(target)
        mapped_action[GRIPPER_FEATURE] = sigma_gripper_angle_to_ur_position(
            action[GRIPPER_ANGLE_FEATURE],
            closed_angle_rad=self.gripper_closed_angle_rad,
            open_angle_rad=self.gripper_open_angle_rad,
        )
        return mapped_action

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = deepcopy(features)
        transformed[PipelineFeatureType.ACTION] = {
            **{name: float for name in TCP_FEATURES},
            GRIPPER_FEATURE: float,
        }  # type: ignore[dict-item]
        return transformed


@dataclass(frozen=True)
class RecordSample:
    observation: RobotObservation
    sent_action: RobotAction


@dataclass
class FrequencyMonitor:
    name: str
    target_hz: float
    warning_ratio: float
    check_interval_s: float = 1.0
    clock: Callable[[], float] = time.perf_counter

    def __post_init__(self) -> None:
        if self.target_hz <= 0:
            raise ValueError("Frequency monitor target must be positive")
        if not 0 < self.warning_ratio <= 1:
            raise ValueError("Frequency warning ratio must be in (0, 1]")
        if self.check_interval_s <= 0:
            raise ValueError("Frequency check interval must be positive")
        self._window_start = self.clock()
        self._completed_cycles = 0

    def tick(self) -> None:
        """Record one completed cycle and periodically report a low rate."""

        self._completed_cycles += 1
        now = self.clock()
        elapsed_s = now - self._window_start
        if elapsed_s < self.check_interval_s:
            return

        actual_hz = self._completed_cycles / elapsed_s
        warning_threshold_hz = self.target_hz * self.warning_ratio
        if actual_hz < warning_threshold_hz:
            logging.warning(
                "%s frequency is %.1f Hz, below the warning threshold %.1f Hz "
                "with a target of %.1f Hz",
                self.name,
                actual_hz,
                warning_threshold_hz,
                self.target_hz,
            )
        else:
            logging.debug(
                "%s frequency is %.1f Hz with a target of %.1f Hz",
                self.name,
                actual_hz,
                self.target_hz,
            )

        self._window_start = now
        self._completed_cycles = 0


def validate_frequency_ratio(control_fps: int, dataset_fps: int) -> int:
    """Validate synchronous downsampling and return its control-step stride."""

    if control_fps <= 0:
        raise ValueError("Control FPS must be positive")
    if dataset_fps <= 0:
        raise ValueError("Dataset FPS must be positive")
    if control_fps % dataset_fps != 0:
        raise ValueError("Control FPS must be an integer multiple of dataset FPS")
    return control_fps // dataset_fps


def build_dataset_features(
    *,
    robot: CameraAugmentedURRobot,
    teleop: Sigma,
    teleop_action_processor: RobotProcessorPipeline,
    robot_observation_processor: RobotProcessorPipeline,
) -> dict[str, dict[str, Any]]:
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=teleop.action_features),
            use_videos=False,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(
                observation=robot.observation_features
            ),
            use_videos=False,
        ),
    )


def capture_episode_references(
    *,
    robot: CameraAugmentedURRobot,
    teleop: Sigma,
    mapper: SigmaRelativeURActionStep,
    sample_count: int,
    control_fps: int,
) -> None:
    logging.info("Hold Sigma still while episode references are captured")
    sigma_reference = capture_sigma_reference(
        teleop,
        sample_count=sample_count,
        sample_interval_s=min(0.01, 1.0 / control_fps),
    )
    ur_reference = action_to_pose(robot.get_control_observation())
    mapper.set_reference(
        sigma_reference=sigma_reference,
        ur_reference=ur_reference,
    )
    logging.info("Episode references captured")
