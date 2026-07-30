#!/usr/bin/env python
"""Record UR10e demonstrations with session-wide Sigma teleoperation.

The robot, cameras, and Sigma remain physically connected for the whole
session. Dataset recording and Sigma-to-UR coupling are independent states:

* c couples Sigma to UR after capturing fresh relative-pose references.
* d decouples Sigma from UR without disconnecting either device.
* s starts an episode while teleoperation is coupled.
* n saves the current episode and leaves teleoperation coupled for reset.
* r discards the current episode and leaves teleoperation coupled for reset.
* q saves a non-empty active episode and shuts down the session.

After the requested number of episodes has been saved, teleoperation remains
available so the operator can reset the robot and object before pressing q.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot.cameras import CameraConfig
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset, safe_stop_image_writer
from lerobot.processor import (
    RobotProcessorPipeline,
    make_default_processors,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.keyboard_input import create_key_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

from lerobot_camera_orbbec import OrbbecCameraConfig
from lerobot_robot_ur import URRobotConfig
from lerobot_teleoperator_sigma import Sigma, SigmaConfig

try:
    from scripts.record_sigma_ur10e_dataset import (
        DEFAULT_FREQUENCY_WARNING_RATIO,
        CameraAugmentedURRobot,
        FrequencyMonitor,
        RecordSample,
        SigmaRelativeURActionStep,
        build_dataset_features,
        capture_episode_references,
        parse_args,
        validate_frequency_ratio,
    )
except ModuleNotFoundError:
    from record_sigma_ur10e_dataset import (  # type: ignore[no-redef]
        DEFAULT_FREQUENCY_WARNING_RATIO,
        CameraAugmentedURRobot,
        FrequencyMonitor,
        RecordSample,
        SigmaRelativeURActionStep,
        build_dataset_features,
        capture_episode_references,
        parse_args,
        validate_frequency_ratio,
    )


class TeleoperationState(Enum):
    DISABLED = auto()
    ENABLED = auto()


class EpisodeState(Enum):
    IDLE = auto()
    RECORDING = auto()
    SAVING = auto()
    FINISHED = auto()


class SessionCommand(Enum):
    ENABLE_TELEOP = auto()
    DISABLE_TELEOP = auto()
    START_EPISODE = auto()
    FINISH_EPISODE = auto()
    RERECORD_EPISODE = auto()
    QUIT = auto()


@dataclass(frozen=True, slots=True)
class EpisodeBoundary:
    save: bool
    frame_count: int


@dataclass(frozen=True, slots=True)
class EpisodeWriteResult:
    save_requested: bool
    saved: bool
    frame_count: int


_WRITER_SENTINEL = object()


def handle_session_key(
    name: str,
    command_queue: queue.SimpleQueue[SessionCommand],
) -> None:
    """Translate one keyboard event into a session command."""

    key = name.lower()
    if key == "c":
        command_queue.put(SessionCommand.ENABLE_TELEOP)
    elif key == "d":
        command_queue.put(SessionCommand.DISABLE_TELEOP)
    elif key == "s":
        command_queue.put(SessionCommand.START_EPISODE)
    elif key in {"n", "right"}:
        command_queue.put(SessionCommand.FINISH_EPISODE)
    elif key in {"r", "left"}:
        command_queue.put(SessionCommand.RERECORD_EPISODE)
    elif key in {"q", "esc"}:
        command_queue.put(SessionCommand.QUIT)


def _get_pending_commands(
    command_queue: queue.SimpleQueue[SessionCommand],
) -> list[SessionCommand]:
    commands: list[SessionCommand] = []
    while True:
        try:
            commands.append(command_queue.get_nowait())
        except queue.Empty:
            return commands


@safe_stop_image_writer
def run_continuous_session(
    *,
    robot: CameraAugmentedURRobot,
    teleop: Sigma,
    mapper: SigmaRelativeURActionStep,
    dataset: LeRobotDataset,
    command_queue: queue.SimpleQueue[SessionCommand],
    num_episodes: int,
    control_fps: int,
    dataset_fps: int,
    episode_time_s: float,
    task: str,
    reference_samples: int,
    teleop_action_processor: RobotProcessorPipeline,
    robot_action_processor: RobotProcessorPipeline,
    robot_observation_processor: RobotProcessorPipeline,
    frequency_warning_ratio: float = DEFAULT_FREQUENCY_WARNING_RATIO,
    capture_references: Callable[..., None] = capture_episode_references,
) -> int:
    """Run teleoperation continuously and gate dataset writes by episode state."""

    record_stride = validate_frequency_ratio(control_fps, dataset_fps)
    if num_episodes <= 0:
        raise ValueError("Number of episodes must be positive")
    if not math.isfinite(episode_time_s) or episode_time_s <= 0:
        raise ValueError("Episode time must be finite and positive")
    if (
        not math.isfinite(frequency_warning_ratio)
        or not 0 < frequency_warning_ratio <= 1
    ):
        raise ValueError("Frequency warning ratio must be finite and in (0, 1]")
    if dataset.fps != dataset_fps:
        raise ValueError(
            "Dataset FPS does not match requested dataset FPS: "
            f"{dataset.fps} != {dataset_fps}"
        )

    writer_queue_capacity = max(2 * dataset_fps + 2, 4)
    writer_queue: queue.Queue[RecordSample | EpisodeBoundary | object] = queue.Queue(
        maxsize=writer_queue_capacity
    )
    writer_results: queue.SimpleQueue[EpisodeWriteResult] = queue.SimpleQueue()
    writer_errors: list[BaseException] = []

    def recording_worker() -> None:
        frequency_monitor: FrequencyMonitor | None = None
        try:
            while True:
                item = writer_queue.get()
                if item is _WRITER_SENTINEL:
                    return

                if isinstance(item, EpisodeBoundary):
                    if item.save and item.frame_count > 0:
                        dataset.save_episode()
                        saved = True
                    else:
                        if dataset.has_pending_frames():
                            dataset.clear_episode_buffer()
                        saved = False
                    writer_results.put(
                        EpisodeWriteResult(
                            save_requested=item.save,
                            saved=saved,
                            frame_count=item.frame_count,
                        )
                    )
                    frequency_monitor = None
                    continue

                if not isinstance(item, RecordSample):
                    raise TypeError(f"Unexpected recording queue item: {type(item)}")
                if frequency_monitor is None:
                    frequency_monitor = FrequencyMonitor(
                        name="Dataset recording",
                        target_hz=dataset_fps,
                        warning_ratio=frequency_warning_ratio,
                    )

                processed_observation = robot_observation_processor(item.observation)
                dataset_observation = build_dataset_frame(
                    dataset.features,
                    processed_observation,
                    prefix=OBS_STR,
                )
                dataset_action = build_dataset_frame(
                    dataset.features,
                    item.sent_action,
                    prefix=ACTION,
                )
                dataset.add_frame(
                    {**dataset_observation, **dataset_action, "task": task}
                )
                frequency_monitor.tick()
        except BaseException as exc:
            writer_errors.append(exc)

    writer_thread = threading.Thread(
        target=recording_worker,
        name="ur10e-continuous-dataset-recording",
    )
    writer_thread.start()

    teleop_state = TeleoperationState.DISABLED
    episode_state = EpisodeState.IDLE
    recorded_episodes = 0
    discarded_episodes = 0
    episode_frame_count = 0
    episode_control_step = 0
    episode_start_t: float | None = None
    stop_requested = False
    control_frequency_monitor: FrequencyMonitor | None = None
    control_interval_s = 1.0 / control_fps
    next_control_t = time.perf_counter()

    def capture_current_references(message: str) -> None:
        nonlocal next_control_t
        log_say(message)
        capture_references(
            robot=robot,
            teleop=teleop,
            mapper=mapper,
            sample_count=reference_samples,
            control_fps=control_fps,
        )
        next_control_t = time.perf_counter()

    def enqueue_boundary(*, save: bool) -> None:
        nonlocal episode_state
        boundary = EpisodeBoundary(save=save, frame_count=episode_frame_count)
        try:
            writer_queue.put_nowait(boundary)
        except queue.Full as exc:
            raise RuntimeError(
                "Recording queue is full while closing the current episode"
            ) from exc
        episode_state = EpisodeState.SAVING

    def process_writer_results() -> None:
        nonlocal episode_state, recorded_episodes, discarded_episodes
        while True:
            try:
                result = writer_results.get_nowait()
            except queue.Empty:
                return

            if result.saved:
                recorded_episodes += 1
                log_say(
                    f"Episode {recorded_episodes} saved with "
                    f"{result.frame_count} frames"
                )
            elif result.save_requested:
                logging.warning("Empty episode was not saved")
                log_say("Empty episode was not saved")
            else:
                discarded_episodes += 1
                log_say(
                    f"Episode discarded with {result.frame_count} buffered frames"
                )

            if recorded_episodes >= num_episodes:
                episode_state = EpisodeState.FINISHED
                log_say(
                    "All requested episodes have been saved. "
                    "Teleoperation remains available for reset. "
                    "Press q after reset"
                )
            else:
                episode_state = EpisodeState.IDLE
                log_say(
                    f"Episode {recorded_episodes + 1} of {num_episodes} is ready. "
                    "Press s to start recording"
                )

    def stop_writer() -> None:
        while writer_thread.is_alive():
            try:
                writer_queue.put(_WRITER_SENTINEL, timeout=0.1)
                break
            except queue.Full:
                if writer_errors:
                    break
        writer_thread.join()

    log_say(
        "Teleoperation is disabled. Press c to capture references and enable it"
    )

    session_error: BaseException | None = None
    try:
        while not stop_requested:
            process_writer_results()
            if writer_errors:
                raise RuntimeError("Dataset recording thread failed") from writer_errors[0]

            for command in _get_pending_commands(command_queue):
                if command is SessionCommand.QUIT:
                    if episode_state is EpisodeState.RECORDING:
                        log_say("Quit requested. Saving the active episode")
                        enqueue_boundary(save=True)
                    stop_requested = True
                    break

                if command is SessionCommand.ENABLE_TELEOP:
                    if episode_state is EpisodeState.RECORDING:
                        logging.warning(
                            "Teleoperation cannot be reconnected while recording"
                        )
                    elif teleop_state is TeleoperationState.ENABLED:
                        logging.info("Teleoperation is already enabled")
                    else:
                        capture_current_references(
                            "Hold Sigma still while teleoperation references are captured"
                        )
                        teleop_state = TeleoperationState.ENABLED
                        control_frequency_monitor = FrequencyMonitor(
                            name="Teleoperation control",
                            target_hz=control_fps,
                            warning_ratio=frequency_warning_ratio,
                        )
                        log_say("Teleoperation enabled")
                    continue

                if command is SessionCommand.DISABLE_TELEOP:
                    if episode_state is EpisodeState.RECORDING:
                        log_say(
                            "Teleoperation disabled during recording. "
                            "Discarding the active episode"
                        )
                        enqueue_boundary(save=False)
                    if teleop_state is TeleoperationState.ENABLED:
                        teleop_state = TeleoperationState.DISABLED
                        mapper.clear_reference()
                        control_frequency_monitor = None
                        log_say(
                            "Teleoperation disabled. Move Sigma to a comfortable "
                            "position and press c to reconnect"
                        )
                    else:
                        logging.info("Teleoperation is already disabled")
                    continue

                if command is SessionCommand.START_EPISODE:
                    if episode_state is EpisodeState.FINISHED:
                        logging.warning(
                            "All requested episodes have already been saved"
                        )
                    elif episode_state is EpisodeState.SAVING:
                        logging.warning(
                            "The previous episode is still being saved"
                        )
                    elif episode_state is EpisodeState.RECORDING:
                        logging.info("An episode is already being recorded")
                    elif teleop_state is TeleoperationState.DISABLED:
                        logging.warning(
                            "Teleoperation is disabled. Press c before starting"
                        )
                    else:
                        capture_current_references(
                            "Hold Sigma still while episode references are captured"
                        )
                        control_frequency_monitor = FrequencyMonitor(
                            name="Teleoperation control",
                            target_hz=control_fps,
                            warning_ratio=frequency_warning_ratio,
                        )
                        episode_frame_count = 0
                        episode_control_step = 0
                        episode_start_t = time.perf_counter()
                        episode_state = EpisodeState.RECORDING
                        log_say(
                            f"Recording episode {recorded_episodes + 1} of "
                            f"{num_episodes}. Press n to save, r to discard, "
                            "d to decouple and discard, or q to quit"
                        )
                    continue

                if command is SessionCommand.FINISH_EPISODE:
                    if episode_state is EpisodeState.RECORDING:
                        log_say("Episode stop requested")
                        enqueue_boundary(save=True)
                    else:
                        logging.info("No episode is currently recording")
                    continue

                if command is SessionCommand.RERECORD_EPISODE:
                    if episode_state is EpisodeState.RECORDING:
                        log_say("Episode rerecord requested")
                        enqueue_boundary(save=False)
                    else:
                        logging.info("No episode is currently recording")

            if stop_requested:
                break

            if (
                episode_state is EpisodeState.RECORDING
                and episode_start_t is not None
                and time.perf_counter() - episode_start_t >= episode_time_s
            ):
                log_say("Episode time limit reached. Saving the episode")
                enqueue_boundary(save=True)

            if teleop_state is TeleoperationState.ENABLED:
                control_observation = robot.get_control_observation()
                should_record = (
                    episode_state is EpisodeState.RECORDING
                    and episode_control_step % record_stride == 0
                )
                if should_record:
                    recorded_observation = robot.add_camera_observations(
                        control_observation
                    )

                raw_sigma_action = teleop.get_action()
                requested_action = teleop_action_processor(
                    (raw_sigma_action, control_observation)
                )
                robot_action = robot_action_processor(
                    (requested_action, control_observation)
                )
                sent_action = robot.send_action(robot_action, control_observation)

                if control_frequency_monitor is not None:
                    control_frequency_monitor.tick()

                if should_record:
                    if writer_queue.qsize() >= writer_queue_capacity - 2:
                        raise RuntimeError(
                            "Recording queue is full because dataset writing "
                            "cannot keep up"
                        )
                    writer_queue.put_nowait(
                        RecordSample(recorded_observation, sent_action)
                    )
                    episode_frame_count += 1

                if episode_state is EpisodeState.RECORDING:
                    episode_control_step += 1

            next_control_t += control_interval_s
            precise_sleep(max(next_control_t - time.perf_counter(), 0.0))
            if next_control_t < time.perf_counter() - control_interval_s:
                next_control_t = time.perf_counter()
    except BaseException as exc:
        session_error = exc
        if episode_state is EpisodeState.RECORDING:
            with contextlib.suppress(BaseException):
                enqueue_boundary(save=False)
    finally:
        stop_writer()
        process_writer_results()

    if writer_errors:
        raise RuntimeError("Dataset recording thread failed") from writer_errors[0]
    if session_error is not None:
        raise session_error

    logging.info(
        "Continuous session completed with %d saved and %d discarded episodes",
        recorded_episodes,
        discarded_episodes,
    )
    return recorded_episodes


def run(args: argparse.Namespace) -> LeRobotDataset:
    """Create devices and dataset, then run the continuous session."""

    validate_frequency_ratio(args.control_fps, args.dataset_fps)

    gemini_305_width, gemini_305_height = args.gemini_305_resolution
    gemini_336_width, gemini_336_height = args.gemini_336_resolution
    camera_configs: dict[str, CameraConfig] = {
        "left_wrist_0_rgb": OrbbecCameraConfig(
            serial_number_or_name=args.gemini_305_serial,
            width=gemini_305_width,
            height=gemini_305_height,
            fps=args.dataset_fps,
            warmup_s=args.camera_warmup_s,
            exposure=155,
            gain=35,
            white_balance=3700,
        ),
        "base_0_rgb": OrbbecCameraConfig(
            serial_number_or_name=args.gemini_336_serial,
            width=gemini_336_width,
            height=gemini_336_height,
            fps=args.dataset_fps,
            warmup_s=args.camera_warmup_s,
            exposure=160,
            gain=19,
            white_balance=4200,
        ),
    }
    robot = CameraAugmentedURRobot(
        URRobotConfig(
            id="ur10e-continuous-dataset-recorder",
            robot_ip=args.robot_ip,
            use_gripper=True,
            gripper_position_poll_frequency_hz=args.control_fps,
            check_pose_safety=False,
            max_tcp_translation_delta_m=args.max_command_translation_m,
            max_tcp_rotation_delta_rad=args.max_command_rotation_rad,
        ),
        camera_configs,
    )
    teleop = Sigma(
        SigmaConfig(
            id="sigma-continuous-dataset-recorder",
            sdk_path=args.sdk_path,
            device_index=args.sigma_device_index,
            serial_number=args.sigma_serial_number,
            enable_gravity_compensation=args.gravity_compensation,
            position_scale=1.0,
            position_offset_m=(0.0, 0.0, 0.0),
            base_rotation_offset_rad=(0.0, 0.0, 0.0),
            tool_rotation_offset_rad=(0.0, 0.0, 0.0),
        )
    )
    mapper = SigmaRelativeURActionStep(
        position_scale=args.position_scale,
        base_rotation_offset_rad=tuple(args.base_rotation_offset_rad),
        gripper_closed_angle_rad=args.sigma_gripper_closed_angle_rad,
        gripper_open_angle_rad=args.sigma_gripper_open_angle_rad,
    )
    teleop_action_processor = RobotProcessorPipeline(
        steps=[mapper],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    _, robot_action_processor, robot_observation_processor = make_default_processors()
    features = build_dataset_features(
        robot=robot,
        teleop=teleop,
        teleop_action_processor=teleop_action_processor,
        robot_observation_processor=robot_observation_processor,
    )
    action_names = features[ACTION].get("names")
    expected_action_names = list(robot.action_features)
    if action_names != expected_action_names:
        raise RuntimeError(
            "Dataset action features do not match URRobot action features: "
            f"{action_names} != {expected_action_names}"
        )
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.dataset_fps,
        root=args.root,
        robot_type=robot.name,
        features=features,
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=args.image_writer_threads_per_camera
        * len(robot.cameras),
        rgb_encoder=RGBEncoderConfig(vcodec="h264"),
    )

    command_queue: queue.SimpleQueue[SessionCommand] = queue.SimpleQueue()
    listener = create_key_listener(
        lambda key: handle_session_key(key, command_queue),
        controls_help=(
            "c=connect teleop, d=disconnect teleop, s=start, "
            "n=save, r=rerecord, q=quit"
        ),
    )
    if listener is None:
        dataset.finalize()
        raise RuntimeError(
            "Interactive keyboard input is required for manual episode recording"
        )

    failed = False
    try:
        log_say("Connecting the robot, cameras, and Sigma")
        robot.connect()
        teleop.connect()
        log_say("All devices connected")
        log_say(f"Dataset output is {dataset.root}")
        log_say(
            "Controls are c to connect teleoperation, d to disconnect "
            "teleoperation, s to start, n to save, r to rerecord, and q to quit"
        )
        run_continuous_session(
            robot=robot,
            teleop=teleop,
            mapper=mapper,
            dataset=dataset,
            command_queue=command_queue,
            num_episodes=args.num_episodes,
            control_fps=args.control_fps,
            dataset_fps=args.dataset_fps,
            episode_time_s=args.episode_time_s,
            task=args.task,
            reference_samples=args.reference_samples,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            frequency_warning_ratio=args.frequency_warning_ratio,
        )
    except BaseException:
        failed = True
        log_say("Recording stopped because of an error")
        if dataset.has_pending_frames():
            with contextlib.suppress(BaseException):
                dataset.clear_episode_buffer()
        raise
    finally:
        log_say("Stopping recording and disconnecting devices", blocking=True)
        listener.stop()
        if teleop.is_connected:
            with contextlib.suppress(BaseException):
                teleop.disconnect()
        if robot.is_connected or super(CameraAugmentedURRobot, robot).is_connected:
            with contextlib.suppress(BaseException):
                robot.disconnect()
        dataset.finalize()
        if failed:
            logging.error("Recording stopped because of an error")
            log_say(
                f"Saved episodes are available at {dataset.root}",
                blocking=True,
            )
        else:
            log_say(
                f"Recording finished. Data saved to {dataset.root}",
                blocking=True,
            )
    return dataset


def main() -> None:
    init_logging()
    run(parse_args())


if __name__ == "__main__":
    main()
