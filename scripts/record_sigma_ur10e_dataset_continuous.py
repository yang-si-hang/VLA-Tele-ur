#!/usr/bin/env python
"""Record UR10e demonstrations with session-wide Sigma teleoperation.

Camera acquisition, teleoperation control, and dataset sampling use independent
frequencies. UR RTDE and Robotiq use their driver defaults. Each dataset frame
pairs one control observation with the action successfully sent from that same
control step. A fixed-rate sampling thread adds the latest camera frames, and a
separate writer thread commits queued samples to the dataset.

The robot, cameras, and Sigma remain physically connected for the whole
session. Dataset recording and Sigma-to-UR coupling are independent states:

* c couples Sigma to UR after capturing fresh relative-pose references.
* d decouples Sigma from UR without disconnecting either device.
* s opens candidate-task selection; enter its number and press Enter to start.
* n saves the current episode and leaves teleoperation coupled for reset.
* r discards the current episode and leaves teleoperation coupled for reset.
* q saves a non-empty active episode and shuts down the session.

After the requested number of episodes has been saved, teleoperation remains
available so the operator can reset the robot and object before pressing q.

Example:
    python scripts/record_sigma_ur10e_dataset_continuous.py --camera-fps 60 --control-fps 60 --dataset-fps 20
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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

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

from utils.const import DATA_PATH

try:
    from scripts.record_sigma_ur10e_common import (
        CameraAugmentedURRobot,
        FrequencyMonitor,
        RecordSample,
        SigmaRelativeURActionStep,
        build_dataset_features,
        capture_episode_references,
    )
except ModuleNotFoundError:
    from record_sigma_ur10e_common import (  # type: ignore[no-redef]
        CameraAugmentedURRobot,
        FrequencyMonitor,
        RecordSample,
        SigmaRelativeURActionStep,
        build_dataset_features,
        capture_episode_references,
    )


DEFAULT_ROBOT_IP = "192.168.253.102"
DEFAULT_GEMINI_305_SERIAL = "CV2L360000C7"
DEFAULT_GEMINI_336_SERIAL = "CP9JA530008V"
DEFAULT_CAMERA_FPS = 60
DEFAULT_CONTROL_FPS = 500
DEFAULT_DATASET_FPS = 20
DEFAULT_FREQUENCY_WARNING_RATIO = 0.95
DEFAULT_NUM_EPISODES = 10
DEFAULT_EPISODE_TIME_S = 90.0
DEFAULT_REFERENCE_SAMPLES = 20      # number of Sigma-UR teleoperation 0 reference point.
DEFAULT_POSITION_SCALE = 3.5
DEFAULT_BASE_ROTATION_RAD = (0.0, 0.0, math.pi)
DEFAULT_SIGMA_GRIPPER_CLOSED_RAD = 0.0
DEFAULT_SIGMA_GRIPPER_OPEN_RAD = 0.5315
DEFAULT_CANDIDATE_TASKS = (
    "Pick up the blue can and place it upright on the red tape marker.",
    "Pick up the purple can and place it upright on the red tape marker.",
    "Pick up the yellow can and place it upright on the red tape marker.",
    "Pick up the white box and place it upright on the red tape marker.",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse continuous-session recorder arguments."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Record a local LeRobot v3 dataset from UR10e and Sigma.7 with continuous teleoperation")

    parser.add_argument("--task", help="Fallback task when --candidate-task is not provided")
    parser.add_argument("--candidate-tasks", action="append", default=None, metavar="TASK", help="Candidate task shown before each episode. Repeat this option to build the candidate task set")

    parser.add_argument("--repo-id", default=f"local/ur10e_sigma_{timestamp}")
    parser.add_argument("--root", type=Path, default=DATA_PATH / f"pick_{timestamp}")
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--episode-time-s", type=float, default=DEFAULT_EPISODE_TIME_S)
    parser.add_argument("--camera-fps", type=int, default=DEFAULT_CAMERA_FPS, help="Camera acquisition frequency")
    parser.add_argument("--control-fps", type=int, default=DEFAULT_CONTROL_FPS, help="Sigma teleoperation processing and send_action frequency")
    parser.add_argument("--dataset-fps", type=int, default=DEFAULT_DATASET_FPS, help="Fixed-frequency dataset sampling rate")
    parser.add_argument("--frequency-warning-ratio", type=float, default=DEFAULT_FREQUENCY_WARNING_RATIO, help="Warn when an actual frequency is below this fraction of its target")
    parser.add_argument("--image-writer-threads-per-camera", type=int, default=4)

    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--max-command-translation-m", type=float, default=0.1)
    parser.add_argument("--max-command-rotation-rad", type=float, default=0.2)

    parser.add_argument("--gemini-305-serial", default=DEFAULT_GEMINI_305_SERIAL)
    parser.add_argument("--gemini-336-serial", default=DEFAULT_GEMINI_336_SERIAL)
    parser.add_argument("--gemini-305-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(640, 480))
    parser.add_argument("--gemini-336-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(320, 240))
    parser.add_argument("--camera-warmup-s", type=float, default=1.0)

    parser.add_argument("--sdk-path", type=Path, default=Path("/opt/forcedimension/sdk"))
    sigma_selector = parser.add_mutually_exclusive_group()
    sigma_selector.add_argument("--sigma-device-index", type=int, default=None)
    sigma_selector.add_argument("--sigma-serial-number", type=int, default=None)
    parser.add_argument("--gravity-compensation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-samples", type=int, default=DEFAULT_REFERENCE_SAMPLES)
    parser.add_argument("--position-scale", type=float, default=DEFAULT_POSITION_SCALE)
    parser.add_argument("--base-rotation-offset-rad", type=float, nargs=3, metavar=("RX", "RY", "RZ"), default=DEFAULT_BASE_ROTATION_RAD)
    parser.add_argument("--sigma-gripper-closed-angle-rad", type=float, default=DEFAULT_SIGMA_GRIPPER_CLOSED_RAD)
    parser.add_argument("--sigma-gripper-open-angle-rad", type=float, default=DEFAULT_SIGMA_GRIPPER_OPEN_RAD)

    args = parser.parse_args(argv)
    if args.task is not None:
        args.task = args.task.strip()
        if not args.task:
            parser.error("--task must not be empty")
    if args.candidate_tasks is not None:
        args.candidate_tasks = [task.strip() for task in args.candidate_tasks]
        if any(not task for task in args.candidate_tasks):
            parser.error("--candidate-tasks must not be empty")
    elif args.task is not None:
        args.candidate_tasks = [args.task]
    else:
        args.candidate_tasks = list(DEFAULT_CANDIDATE_TASKS)
    for name in (
        "num_episodes",
        "camera_fps",
        "control_fps",
        "dataset_fps",
        "image_writer_threads_per_camera",
        "reference_samples",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.dataset_fps > args.control_fps:
        parser.error("--dataset-fps must not exceed --control-fps")
    if (
        not math.isfinite(args.frequency_warning_ratio)
        or not 0 < args.frequency_warning_ratio <= 1
    ):
        parser.error("--frequency-warning-ratio must be finite and in (0, 1]")
    for name in (
        "episode_time_s",
        "max_command_translation_m",
        "max_command_rotation_rad",
        "position_scale",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for camera_name in ("gemini_305", "gemini_336"):
        if any(value <= 0 for value in getattr(args, f"{camera_name}_resolution")):
            parser.error(f"{camera_name.replace('_', '-')} dimensions must be positive")
    if not math.isfinite(args.camera_warmup_s) or args.camera_warmup_s < 0:
        parser.error("--camera-warmup-s must be finite and non-negative")
    if args.sigma_gripper_closed_angle_rad < 0:
        parser.error("--sigma-gripper-closed-angle-rad must be non-negative")
    if args.sigma_gripper_open_angle_rad <= args.sigma_gripper_closed_angle_rad:
        parser.error("Sigma gripper open angle must be greater than the closed angle")
    return args


class TeleoperationState(Enum):
    DISABLED = auto()
    ENABLED = auto()


class EpisodeState(Enum):
    IDLE = auto()
    SELECTING_TASK = auto()
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


KEY_COMMANDS = {
    "c": SessionCommand.ENABLE_TELEOP,
    "d": SessionCommand.DISABLE_TELEOP,
    "s": SessionCommand.START_EPISODE,
    "n": SessionCommand.FINISH_EPISODE,
    "right": SessionCommand.FINISH_EPISODE,
    "r": SessionCommand.RERECORD_EPISODE,
    "left": SessionCommand.RERECORD_EPISODE,
    "q": SessionCommand.QUIT,
    "esc": SessionCommand.QUIT,
}


@dataclass(frozen=True, slots=True)
class TaskSelectionKey:
    name: str
    captured_at: float = field(default_factory=time.perf_counter)


@dataclass(frozen=True, slots=True)
class EpisodeBoundary:
    save: bool
    frame_count: int


@dataclass(frozen=True, slots=True)
class EpisodeWriteResult:
    save_requested: bool
    saved: bool
    frame_count: int


@dataclass(frozen=True, slots=True)
class EpisodeRecordSample:
    sample: RecordSample
    task: str


@dataclass(frozen=True, slots=True)
class LatestControlSample:
    sample: RecordSample
    captured_at: float
    sequence: int
    episode_generation: int


@dataclass(frozen=True, slots=True)
class SamplingEpisodeStart:
    task: str
    generation: int
    started_at: float


@dataclass(frozen=True, slots=True)
class SamplingEpisodeBoundary:
    save: bool
    generation: int


_WRITER_SENTINEL = object()
_SAMPLER_SENTINEL = object()


def handle_session_key(
    name: str,
    command_queue: queue.SimpleQueue[SessionCommand | TaskSelectionKey],
) -> None:
    """Translate one keyboard event into a session command."""

    command = KEY_COMMANDS.get(name.lower())
    if command is not None:
        command_queue.put(command)
    elif name.isdigit() or name in {"enter", "backspace"}:
        command_queue.put(TaskSelectionKey(name))


def _get_pending_commands(
    command_queue: queue.SimpleQueue[SessionCommand | TaskSelectionKey],
) -> list[SessionCommand | TaskSelectionKey]:
    commands: list[SessionCommand | TaskSelectionKey] = []
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
    command_queue: queue.SimpleQueue[SessionCommand | TaskSelectionKey],
    num_episodes: int,
    control_fps: int,
    dataset_fps: int,
    episode_time_s: float,
    task: str | None = None,
    candidate_tasks: Sequence[str] | None = None,
    reference_samples: int,
    teleop_action_processor: RobotProcessorPipeline,
    robot_action_processor: RobotProcessorPipeline,
    robot_observation_processor: RobotProcessorPipeline,
    frequency_warning_ratio: float = DEFAULT_FREQUENCY_WARNING_RATIO,
    capture_references: Callable[..., None] = capture_episode_references,
    task_selector: Callable[[Sequence[str]], str] | None = None,
) -> int:
    """Run teleoperation continuously and gate dataset writes by episode state."""

    if num_episodes <= 0:
        raise ValueError("Number of episodes must be positive")
    if control_fps <= 0:
        raise ValueError("Control FPS must be positive")
    if dataset_fps <= 0:
        raise ValueError("Dataset FPS must be positive")
    if dataset_fps > control_fps:
        raise ValueError("Dataset FPS must not exceed control FPS")
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
    resolved_candidate_tasks = tuple(
        candidate_tasks or (() if task is None else (task,))
    )
    if not resolved_candidate_tasks or any(
        not candidate_task.strip() for candidate_task in resolved_candidate_tasks
    ):
        raise ValueError("Candidate task set must contain non-empty tasks")

    writer_queue_capacity = max(2 * dataset_fps + 2, 4)
    writer_queue: queue.Queue[EpisodeRecordSample | EpisodeBoundary | object] = (
        queue.Queue(maxsize=writer_queue_capacity)
    )
    writer_results: queue.SimpleQueue[EpisodeWriteResult] = queue.SimpleQueue()
    writer_errors: list[BaseException] = []
    task_selection_results: queue.SimpleQueue[str | BaseException] = queue.SimpleQueue()
    sampling_commands: queue.Queue[
        SamplingEpisodeStart | SamplingEpisodeBoundary | object
    ] = queue.Queue()
    sampling_errors: list[BaseException] = []
    latest_control_lock = threading.Lock()
    latest_control_sample: LatestControlSample | None = None
    dataset_interval_s = 1.0 / dataset_fps

    def recording_worker() -> None:
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
                    continue

                if not isinstance(item, EpisodeRecordSample):
                    raise TypeError(f"Unexpected recording queue item: {type(item)}")

                processed_observation = robot_observation_processor(
                    item.sample.observation
                )
                dataset_observation = build_dataset_frame(
                    dataset.features,
                    processed_observation,
                    prefix=OBS_STR,
                )
                dataset_action = build_dataset_frame(
                    dataset.features,
                    item.sample.sent_action,
                    prefix=ACTION,
                )
                dataset.add_frame(
                    {**dataset_observation, **dataset_action, "task": item.task}
                )
        except BaseException as exc:
            writer_errors.append(exc)

    def sampling_worker() -> None:
        active_episode: SamplingEpisodeStart | None = None
        episode_frame_count = 0
        next_dataset_t: float | None = None
        frequency_monitor: FrequencyMonitor | None = None

        try:
            while True:
                if next_dataset_t is None:
                    timeout = None
                else:
                    timeout = max(next_dataset_t - time.perf_counter(), 0.0)

                try:
                    command = sampling_commands.get(timeout=timeout)
                except queue.Empty:
                    command = None

                if command is _SAMPLER_SENTINEL:
                    return

                if isinstance(command, SamplingEpisodeStart):
                    if active_episode is not None:
                        raise RuntimeError(
                            "Cannot start dataset sampling while an episode is active"
                        )
                    active_episode = command
                    episode_frame_count = 0
                    next_dataset_t = command.started_at
                    frequency_monitor = FrequencyMonitor(
                        name="Dataset sampling",
                        target_hz=dataset_fps,
                        warning_ratio=frequency_warning_ratio,
                    )
                elif isinstance(command, SamplingEpisodeBoundary):
                    if (
                        active_episode is None
                        or command.generation != active_episode.generation
                    ):
                        raise RuntimeError(
                            "Dataset sampling boundary does not match the active episode"
                        )
                    try:
                        writer_queue.put_nowait(
                            EpisodeBoundary(
                                save=command.save,
                                frame_count=episode_frame_count,
                            )
                        )
                    except queue.Full as exc:
                        raise RuntimeError(
                            "Recording queue is full while closing the current episode"
                        ) from exc
                    active_episode = None
                    episode_frame_count = 0
                    next_dataset_t = None
                    frequency_monitor = None
                    continue
                elif command is not None:
                    raise TypeError(
                        f"Unexpected sampling command: {type(command)}"
                    )

                if active_episode is None or next_dataset_t is None:
                    continue

                sample_t = time.perf_counter()
                if sample_t < next_dataset_t:
                    continue

                with latest_control_lock:
                    control_sample = latest_control_sample
                if (
                    control_sample is not None
                    and control_sample.episode_generation
                    == active_episode.generation
                ):
                    recorded_observation = robot.add_camera_observations(
                        control_sample.sample.observation
                    )
                    if writer_queue.qsize() >= writer_queue_capacity - 2:
                        raise RuntimeError(
                            "Recording queue is full because dataset writing "
                            "cannot keep up"
                        )
                    writer_queue.put_nowait(
                        EpisodeRecordSample(
                            sample=RecordSample(
                                recorded_observation,
                                dict(control_sample.sample.sent_action),
                            ),
                            task=active_episode.task,
                        )
                    )
                    episode_frame_count += 1
                    if frequency_monitor is not None:
                        frequency_monitor.tick()

                elapsed_intervals = math.floor(
                    (sample_t - next_dataset_t) / dataset_interval_s
                ) + 1
                next_dataset_t += elapsed_intervals * dataset_interval_s
        except BaseException as exc:  # noqa: BLE001 - propagate worker failures.
            sampling_errors.append(exc)

    writer_thread = threading.Thread(
        target=recording_worker,
        name="ur10e-continuous-dataset-recording",
    )
    writer_thread.start()
    sampling_thread = threading.Thread(
        target=sampling_worker,
        name="ur10e-continuous-dataset-sampling",
    )
    sampling_thread.start()

    teleop_state = TeleoperationState.DISABLED
    episode_state = EpisodeState.IDLE
    recorded_episodes = 0
    discarded_episodes = 0
    episode_start_t: float | None = None
    episode_task: str | None = None
    task_selection_buffer = ""
    task_selection_started_at = 0.0
    episode_generation = 0
    sampling_started_generation: int | None = None
    control_sequence = 0
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

    def start_task_selection() -> None:
        nonlocal episode_state, task_selection_buffer, task_selection_started_at

        task_selection_buffer = ""
        task_selection_started_at = time.perf_counter()
        episode_state = EpisodeState.SELECTING_TASK
        print("Candidate tasks:")
        for index, candidate_task in enumerate(resolved_candidate_tasks, start=1):
            print(f"  {index}. {candidate_task}")
        print(
            f"Select task for the next episode [1-{len(resolved_candidate_tasks)}], "
            "then press Enter:"
        )

        if task_selector is None:
            return

        def select_task_worker() -> None:
            try:
                task_selection_results.put(task_selector(resolved_candidate_tasks))
            except BaseException as exc:
                task_selection_results.put(exc)

        task_selection_thread = threading.Thread(
            target=select_task_worker,
            name="ur10e-episode-task-selection",
            daemon=True,
        )
        task_selection_thread.start()

    def confirm_task_selection(selected_index: int) -> None:
        nonlocal episode_state, episode_task
        nonlocal episode_start_t, episode_generation
        nonlocal sampling_started_generation, control_frequency_monitor

        episode_task = resolved_candidate_tasks[selected_index - 1]
        log_say(f"Selected task {selected_index}: {episode_task}")
        capture_current_references(
            "Hold Sigma still while episode references are captured"
        )
        control_frequency_monitor = FrequencyMonitor(
            name="Teleoperation control",
            target_hz=control_fps,
            warning_ratio=frequency_warning_ratio,
        )
        episode_generation += 1
        sampling_started_generation = None
        episode_start_t = time.perf_counter()
        episode_state = EpisodeState.RECORDING
        log_say(
            f"Recording episode {recorded_episodes + 1} of "
            f"{num_episodes}. Press n to save, r to discard, "
            "d to decouple and discard, or q to quit"
        )

    def process_task_selection_key(name: str) -> None:
        nonlocal task_selection_buffer

        if episode_state is not EpisodeState.SELECTING_TASK:
            return
        if name.isdigit():
            task_selection_buffer += name
            print(f"Task selection: {task_selection_buffer}")
            return
        if name == "backspace":
            task_selection_buffer = task_selection_buffer[:-1]
            print(f"Task selection: {task_selection_buffer or '(empty)'}")
            return
        if name != "enter":
            return
        if not task_selection_buffer:
            print("Task selection is empty. Enter a task number, then press Enter.")
            return
        selected_index = int(task_selection_buffer)
        task_selection_buffer = ""
        if not 1 <= selected_index <= len(resolved_candidate_tasks):
            print("Task index is out of range. Enter a number from the list.")
            return
        confirm_task_selection(selected_index)

    def process_task_selection() -> None:
        nonlocal episode_state, episode_task
        nonlocal episode_start_t, episode_generation
        nonlocal sampling_started_generation, control_frequency_monitor

        if episode_state is not EpisodeState.SELECTING_TASK:
            return
        try:
            result = task_selection_results.get_nowait()
        except queue.Empty:
            return
        if isinstance(result, BaseException):
            raise RuntimeError("Task selection failed") from result
        if result not in resolved_candidate_tasks:
            raise ValueError("Task selector returned a task outside the candidate set")
        if teleop_state is TeleoperationState.DISABLED:
            episode_state = EpisodeState.IDLE
            logging.warning(
                "Task was selected after teleoperation was disabled. "
                "Press s to select again"
            )
            return

        selected_task_index = resolved_candidate_tasks.index(result) + 1
        confirm_task_selection(selected_task_index)

    def start_sampling_if_needed() -> None:
        nonlocal sampling_started_generation
        if sampling_started_generation == episode_generation:
            return
        if episode_task is None or episode_start_t is None:
            raise RuntimeError("Recording episode is missing sampling metadata")
        sampling_commands.put_nowait(
            SamplingEpisodeStart(
                task=episode_task,
                generation=episode_generation,
                started_at=episode_start_t,
            )
        )
        sampling_started_generation = episode_generation

    def enqueue_boundary(*, save: bool) -> None:
        nonlocal episode_state
        start_sampling_if_needed()
        sampling_commands.put_nowait(
            SamplingEpisodeBoundary(
                save=save,
                generation=episode_generation,
            )
        )
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
                log_say(f"Episode discarded with {result.frame_count} buffered frames")

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

    def stop_sampler() -> None:
        sampling_commands.put(_SAMPLER_SENTINEL)
        sampling_thread.join()

    log_say("Teleoperation is disabled. Press c to capture references and enable it")

    session_error: BaseException | None = None
    try:
        while not stop_requested:
            process_writer_results()
            if writer_errors:
                raise RuntimeError(
                    "Dataset recording thread failed"
                ) from writer_errors[0]
            if sampling_errors:
                raise RuntimeError(
                    "Dataset sampling thread failed"
                ) from sampling_errors[0]

            for command in _get_pending_commands(command_queue):
                if isinstance(command, TaskSelectionKey):
                    if command.captured_at >= task_selection_started_at:
                        process_task_selection_key(command.name)
                    continue

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
                            "Hold Sigma still while teleoperation references are "
                            "captured"
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
                    elif episode_state is EpisodeState.SELECTING_TASK:
                        logging.info(
                            "Teleoperation disabled while task selection is pending"
                        )
                        episode_state = EpisodeState.IDLE
                        task_selection_buffer = ""
                        log_say("Task selection cancelled")
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
                        logging.warning("The previous episode is still being saved")
                    elif episode_state is EpisodeState.SELECTING_TASK:
                        logging.info("Task selection is already in progress")
                    elif episode_state is EpisodeState.RECORDING:
                        logging.info("An episode is already being recorded")
                    elif teleop_state is TeleoperationState.DISABLED:
                        logging.warning(
                            "Teleoperation is disabled. Press c before starting"
                        )
                    else:
                        start_task_selection()
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

            process_task_selection()

            if (
                episode_state is EpisodeState.RECORDING
                and episode_start_t is not None
                and time.perf_counter() - episode_start_t >= episode_time_s
            ):
                log_say("Episode time limit reached. Saving the episode")
                enqueue_boundary(save=True)

            if teleop_state is TeleoperationState.ENABLED:
                control_observation = robot.get_control_observation()
                raw_sigma_action = teleop.get_action()
                requested_action = teleop_action_processor(
                    (raw_sigma_action, control_observation)
                )
                robot_action = robot_action_processor(
                    (requested_action, control_observation)
                )
                sent_action = robot.send_action(
                    robot_action,
                    control_observation,
                    step_limit_check=True,
                )
                control_sequence += 1
                with latest_control_lock:
                    latest_control_sample = LatestControlSample(
                        sample=RecordSample(
                            dict(control_observation),
                            dict(sent_action),
                        ),
                        captured_at=time.perf_counter(),
                        sequence=control_sequence,
                        episode_generation=episode_generation,
                    )

                if control_frequency_monitor is not None:
                    control_frequency_monitor.tick()

                if episode_state is EpisodeState.RECORDING:
                    start_sampling_if_needed()

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
        stop_sampler()
        stop_writer()
        process_writer_results()

    if sampling_errors:
        raise RuntimeError("Dataset sampling thread failed") from sampling_errors[0]
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

    gemini_305_width, gemini_305_height = args.gemini_305_resolution
    gemini_336_width, gemini_336_height = args.gemini_336_resolution
    camera_configs: dict[str, CameraConfig] = {
        "left_wrist_0_rgb": OrbbecCameraConfig(
            serial_number_or_name=args.gemini_305_serial,
            width=gemini_305_width,
            height=gemini_305_height,
            fps=args.camera_fps,
            warmup_s=args.camera_warmup_s,
            frame_rate_warning_ratio=args.frequency_warning_ratio,
            exposure=155,
            gain=35,
            white_balance=3700,
            anti_flicker=True,
        ),
        "base_0_rgb": OrbbecCameraConfig(
            serial_number_or_name=args.gemini_336_serial,
            width=gemini_336_width,
            height=gemini_336_height,
            fps=args.camera_fps,
            warmup_s=args.camera_warmup_s,
            frame_rate_warning_ratio=args.frequency_warning_ratio,
            exposure=150,
            gain=19,
            white_balance=4200,
            anti_flicker=True,
        ),
    }
    robot = CameraAugmentedURRobot(
        URRobotConfig(
            id="ur10e-continuous-dataset-recorder",
            robot_ip=args.robot_ip,
            use_gripper=True,
            rtde_check_pose_safety=False,
            tcp_pose=(0.0, 0.0, 0.174, 0.0, 0.0, 0.0),
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
        image_writer_threads=args.image_writer_threads_per_camera * len(robot.cameras),
        rgb_encoder=RGBEncoderConfig(vcodec="h264"),
    )

    command_queue: queue.SimpleQueue[SessionCommand | TaskSelectionKey] = queue.SimpleQueue()
    listener = create_key_listener(
        lambda key: handle_session_key(key, command_queue),
        controls_help=(
            "c=connect teleop, d=disconnect teleop, s=start, n=save, r=rerecord, q=quit"
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
            candidate_tasks=args.candidate_tasks,
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
    init_logging(console_level="INFO")
    run(parse_args())


if __name__ == "__main__":
    main()
