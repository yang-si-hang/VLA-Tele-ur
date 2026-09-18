#!/usr/bin/env python3
"""Run a four-mode OpenPI controller for a UR robot and two Orbbec cameras.

Requires a reachable compatible policy server, UR robot, and two Gemini cameras.
After connecting, it keeps the current TCP position, moves to the user-provided
RPY orientation, and fully opens the gripper before policy warmup.
The controller supports RTC off, test-time non-VJP or VJP guidance, and a
train-time RTC checkpoint. Policy targets are produced at the configured
control frequency and interpolated independently by the native RTDE servo
loop. It can also record one LeRobot dataset episode,
optionally including policy capture timing and base-frame delta actions, and
timestamped UR TCP pose and speed at 500 Hz through RTDE. Recorded datasets
store the effective RTC configuration in ``meta/rtc.json``. Dataset creation
and RTDE state recording begin only after all hardware is connected. Before
control starts, the script waits for fresh gripper feedback after setup.
Example: python scripts/inference/pi05_inference_rtc.py --initial-rpy 3.14 0 0 --rtc-mode non-vjp --prefix-len 2 --record-robot-state
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rtde_receive
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.io_utils import write_json
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot_camera_orbbec import OrbbecCamera, OrbbecCameraConfig
from lerobot_robot_ur import URRobot, URRobotConfig
from lerobot_robot_ur.ur import (
    GRIPPER_FEATURE,
    TCP_FEATURES,
    TCP_POSITION_FEATURES,
    TCP_ROTATION_6D_FEATURES,
)
from openpi_client import base_policy, image_tools, websocket_client_policy
from scipy.spatial.transform import Rotation

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.openpi.ur_action_adapter import (
    DELTA_ACTIONS_KEY,
    UR_ACTION_DIM,
    RTCActionChunkBroker,
    create_absolute_action_broker,
    create_rtc_action_broker,
)
from utils.const import DATA_PATH, OUTPUT_PATH

POLICY_HOST = "127.0.0.1"
POLICY_PORT = 8000
DEFAULT_ROBOT_IP = "192.168.253.102"
DEFAULT_GEMINI_305_SERIAL = "CP9JA5300068"
DEFAULT_GEMINI_336_SERIAL = "CP9JA530008V"
DEFAULT_PROMPT = "Pick up the yellow can and place it upright on the red tape marker."
DEFAULT_CONTROL_FPS = 20
DEFAULT_EXECUTION_HORIZON = 10
DEFAULT_DATASET_DIR = DATA_PATH / "deploy"
DEFAULT_ROBOT_STATE_RECORD_DIR = OUTPUT_PATH / "pi05_inference_rtc"
RTDE_CONTROL_FREQUENCY_HZ = 500.0
RTDE_RECORD_FREQUENCY_HZ = 500.0
RTDE_RECORD_VARIABLES = ["timestamp", "actual_TCP_pose", "actual_TCP_speed"]

ACTION_FEATURES = (*TCP_FEATURES, GRIPPER_FEATURE)

CAMERA_CONFIG = {
    "observation.images.left_wrist_0_rgb": {"crop": [48, 0, 272, 224], "size": None},
    "observation.images.base_0_rgb": {"crop": [48, 16, 272, 240], "size": None},
}

RTC_MODES = ("off", "non-vjp", "vjp", "train-rtc")

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Control a UR robot with a four-mode OpenPI policy server")

    parser.add_argument("--policy-host", default=POLICY_HOST)
    parser.add_argument("--policy-port", type=int, default=POLICY_PORT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--execution-horizon", type=int, default=DEFAULT_EXECUTION_HORIZON)
    parser.add_argument("--warmup-inferences", type=int, default=2)
    parser.add_argument("--control-fps", type=float, default=DEFAULT_CONTROL_FPS)
    parser.add_argument("--start-immediately", action="store_true", help="Skip the interactive safety confirmation before sending actions")

    parser.add_argument("--rtc-mode", choices=RTC_MODES, default="non-vjp")
    parser.add_argument("--prefix-len", type=int, default=2)
    parser.add_argument("--decay-end", type=int, default=None)
    parser.add_argument("--rtc-schedule", choices=("exp", "linear", "ones", "zeros"), default=None)
    parser.add_argument("--max-guidance-weight", type=float, default=None)

    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-policy-infer-time", default=True, action=argparse.BooleanOptionalAction, help="Record the elapsed time between consecutive policy observation captures")
    parser.add_argument("--record-policy-delta-action", default=True, action=argparse.BooleanOptionalAction, help="Record the policy's base-frame delta action")
    parser.add_argument("--repo-id", default=f"pi05_pick_{timestamp}")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--image-writer-threads-per-camera", type=int, default=4)

    parser.add_argument("--record-robot-state", default=False, action=argparse.BooleanOptionalAction, help="Record timestamped actual UR TCP pose and speed through RTDE at 500 Hz")
    parser.add_argument("--robot-state-record-dir", type=Path, default=DEFAULT_ROBOT_STATE_RECORD_DIR, help="Directory for timestamped RTDE robot-state CSV files")

    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--tcp-pose", type=float, nargs=6, default=(0.0, 0.0, 0.174, 0.0, 0.0, 0.0), metavar=("X", "Y", "Z", "RX", "RY", "RZ"))
    parser.add_argument("--initial-rpy", type=float, nargs=3, default=(-np.pi, 0.0, -np.pi/2), metavar=("ROLL", "PITCH", "YAW"), help="Initial TCP orientation as fixed-axis xyz roll, pitch, yaw angles in radians")
    parser.add_argument("--max-command-translation-m", type=float, default=0.1)
    parser.add_argument("--max-command-rotation-rad", type=float, default=0.2)

    parser.add_argument("--gemini-305-serial", default=DEFAULT_GEMINI_305_SERIAL)
    parser.add_argument("--gemini-336-serial", default=DEFAULT_GEMINI_336_SERIAL)
    parser.add_argument("--gemini-305-resolution", type=int, nargs=2, default=(320, 240))
    parser.add_argument("--gemini-336-resolution", type=int, nargs=2, default=(320, 240))
    parser.add_argument("--camera-fps", type=int, default=60)
    parser.add_argument("--camera-warmup-s", type=float, default=1.0)
    parser.add_argument("--camera-max-age-ms", type=int, default=500)

    args = parser.parse_args(argv)
    if not args.policy_host.strip():
        parser.error("--policy-host must not be empty")
    if not 1 <= args.policy_port <= 65535:
        parser.error("--policy-port must be in [1, 65535]")
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.execution_horizon <= 0:
        parser.error("--execution-horizon must be positive")
    if args.warmup_inferences < 0:
        parser.error("--warmup-inferences must be non-negative")
    if not math.isfinite(args.control_fps) or args.control_fps <= 0:
        parser.error("--control-fps must be finite and positive")
    if args.record and not float(args.control_fps).is_integer():
        parser.error("--control-fps must be an integer when recording")
    if not args.repo_id.strip():
        parser.error("--repo-id must not be empty")
    if args.image_writer_threads_per_camera <= 0:
        parser.error("--image-writer-threads-per-camera must be positive")
    if args.camera_fps <= 0 or args.camera_max_age_ms < 0:
        parser.error("camera FPS must be positive and max age must be non-negative")
    if not math.isfinite(args.camera_warmup_s) or args.camera_warmup_s < 0:
        parser.error("--camera-warmup-s must be finite and non-negative")
    for camera_name in ("gemini_305", "gemini_336"):
        if any(value <= 0 for value in getattr(args, f"{camera_name}_resolution")):
            parser.error(f"--{camera_name.replace('_', '-')}-resolution values must be positive")
    for name in ("max_command_translation_m", "max_command_rotation_rad"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not all(math.isfinite(value) for value in args.initial_rpy):
        parser.error("--initial-rpy values must be finite")
    return args


def _wire_mode(cli_mode: str) -> str:
    return cli_mode.replace("-", "_")


def validate_rtc_arguments(args: argparse.Namespace, metadata: Mapping[str, Any]) -> None:
    """Validate client-known timing constraints; checkpoint checks remain server-side."""
    prediction_horizon = int(metadata.get("prediction_horizon", 0))
    execution_horizon = int(args.execution_horizon)
    if prediction_horizon <= 0:
        raise ValueError("Server metadata must contain a positive prediction_horizon")

    if args.rtc_mode == "off":
        if not 0 < execution_horizon <= prediction_horizon:
            raise ValueError("execution_horizon must be between 1 and prediction_horizon")
        rtc_argument_names = ("prefix_len", "decay_end", "rtc_schedule", "max_guidance_weight")
        ignored_arguments = [name for name in rtc_argument_names if getattr(args, name) is not None]
        if ignored_arguments:
            logger.warning(
                "Ignoring RTC arguments because --rtc-mode=off: %s",
                ", ".join(f"--{name.replace('_', '-')}" for name in ignored_arguments),
            )
            for name in rtc_argument_names:
                setattr(args, name, None)
        return

    if not 0 < execution_horizon < prediction_horizon:
        raise ValueError("execution_horizon must be between 1 and prediction_horizon - 1 for RTC")
    if args.prefix_len is None or args.prefix_len <= 0:
        raise ValueError("--prefix-len must be explicitly set to a positive integer when RTC is enabled")
    remaining_horizon = prediction_horizon - execution_horizon
    if args.prefix_len > remaining_horizon:
        raise ValueError(f"--prefix-len must not exceed prediction_horizon - execution_horizon ({remaining_horizon})")
    if args.rtc_mode == "train-rtc" and any(
        value is not None for value in (args.decay_end, args.rtc_schedule, args.max_guidance_weight)
    ):
        raise ValueError("guidance arguments are not supported by --rtc-mode=train-rtc")
    if args.decay_end is not None and args.decay_end < args.prefix_len:
        raise ValueError("--decay-end must be greater than or equal to --prefix-len")

    control_frequency_hz = metadata.get("control_frequency_hz")
    if control_frequency_hz is not None and float(args.control_fps) != float(control_frequency_hz):
        raise ValueError(
            f"Client control_fps={args.control_fps} does not match server control_frequency_hz={control_frequency_hz}"
        )


def create_execution_policy(
    client: base_policy.BasePolicy,
    metadata: Mapping[str, Any],
    args: argparse.Namespace,
) -> base_policy.BasePolicy:
    """Create the synchronous or explicitly selected asynchronous broker."""
    validate_rtc_arguments(args, metadata)
    if args.rtc_mode == "off":
        return create_absolute_action_broker(
            client,
            metadata,
            execution_horizon=args.execution_horizon,
        )

    return create_rtc_action_broker(
        client,
        metadata,
        rtc_mode=_wire_mode(args.rtc_mode),
        prefix_len=args.prefix_len,
        replan_interval=args.execution_horizon,
        decay_end=args.decay_end,
        schedule=args.rtc_schedule or "exp",
        max_guidance_weight=5.0 if args.max_guidance_weight is None else args.max_guidance_weight,
    )


def warmup_execution_policy(policy: base_policy.BasePolicy, observation: dict) -> None:
    """Warm the selected path before robot motion without consuming a chunk."""
    if isinstance(policy, RTCActionChunkBroker):
        policy.warmup(observation)


def close_execution_policy(policy: base_policy.BasePolicy) -> None:
    """Drain RTC work and release its executor; reset synchronous brokers."""
    if isinstance(policy, RTCActionChunkBroker):
        policy.close()
    else:
        policy.reset()


def prepare_camera_image(image: np.ndarray, camera_key: str) -> np.ndarray:
    """Validate, crop, and optionally resize one RGB image for OpenPI."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 HWC RGB image, got {image.shape}@{image.dtype}")
    if camera_key not in CAMERA_CONFIG:
        raise ValueError(f"Unknown camera key: {camera_key}")

    config = CAMERA_CONFIG[camera_key]
    left, top, right, bottom = config["crop"]
    height, width = image.shape[:2]
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"Crop {config['crop']} for {camera_key} is outside image shape {image.shape}")
    processed = image[top:bottom, left:right]
    if (size := config["size"]) is not None:
        processed = image_tools.resize_with_pad(processed, size[0], size[1])
    return np.ascontiguousarray(processed, dtype=np.uint8)


def robot_observation_to_state(observation: Mapping[str, object]) -> np.ndarray:
    """Convert a UR observation dictionary to OpenPI's fixed 10-D state."""
    missing = [name for name in ACTION_FEATURES if name not in observation]
    if missing:
        raise ValueError(f"Robot observation is missing required keys: {missing}")
    state = np.asarray([observation[name] for name in ACTION_FEATURES], dtype=np.float32)
    if state.shape != (UR_ACTION_DIM,) or not np.all(np.isfinite(state)):
        raise ValueError(f"Expected finite UR state shape ({UR_ACTION_DIM},), got {state.shape}")
    return state


def build_observation(
    state: np.ndarray,
    base_image: np.ndarray,
    wrist_image: np.ndarray,
    prompt: str,
) -> dict[str, object]:
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (UR_ACTION_DIM,) or not np.all(np.isfinite(state)):
        raise ValueError(f"Expected finite UR state shape ({UR_ACTION_DIM},), got {state.shape}")
    return {
        "observation.state": state,
        "observation.images.base_0_rgb": prepare_camera_image(base_image, "observation.images.base_0_rgb"),
        "observation.images.left_wrist_0_rgb": prepare_camera_image(wrist_image, "observation.images.left_wrist_0_rgb"),
        "prompt": prompt,
    }


def policy_action_to_robot_action(action: np.ndarray) -> dict[str, float]:
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (UR_ACTION_DIM,) or not np.all(np.isfinite(values)):
        raise ValueError(f"Expected one finite action with shape ({UR_ACTION_DIM},)")
    return {name: float(value) for name, value in zip(ACTION_FEATURES, values, strict=True)}


def initialize_robot(robot: URRobot, initial_rpy: Sequence[float]) -> dict[str, float]:
    """Keep the current TCP position, set an RPY orientation, and open the gripper."""
    current_observation = robot.get_observation()
    rotation_matrix = Rotation.from_euler("xyz", initial_rpy).as_matrix()
    rot6d = rotation_matrix[:, :2].T.reshape(-1)
    initial_action = {name: float(current_observation[name]) for name in TCP_POSITION_FEATURES}
    initial_action.update(
        {
            name: float(value)
            for name, value in zip(TCP_ROTATION_6D_FEATURES, rot6d, strict=True)
        }
    )
    # initial_action[GRIPPER_FEATURE] = 0.0
    initial_action[GRIPPER_FEATURE] = float(current_observation[GRIPPER_FEATURE])
    return robot.move_to_action(initial_action)


def create_recording_dataset(
    *,
    repo_id: str,
    root: Path,
    fps: int,
    image_writer_threads_per_camera: int,
    record_policy_infer_time: bool,
    record_policy_delta_action: bool,
    rtc_config: Mapping[str, Any],
) -> LeRobotDataset:
    vector_feature = {
        "dtype": "float32",
        "shape": (UR_ACTION_DIM,),
        "names": list(ACTION_FEATURES),
    }
    image_feature = {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channels"]}
    features = {
        "action": vector_feature,
        "observation.state": vector_feature.copy(),
        "observation.images.left_wrist_0_rgb": image_feature,
        "observation.images.base_0_rgb": image_feature.copy(),
    }
    if record_policy_infer_time:
        features["debug.capture_time"] = {"dtype": "float32", "shape": (1,), "names": ["seconds_since_previous_frame"]}
    if record_policy_delta_action:
        features["debug.delta_action"] = vector_feature.copy()
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        robot_type="ur",
        features=features,
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=image_writer_threads_per_camera * 2,
        rgb_encoder=RGBEncoderConfig(vcodec="h264"),
    )
    write_json(dict(rtc_config), dataset.root / "meta" / "rtc.json")
    return dataset


def build_rtc_recording_config(args: argparse.Namespace, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return the effective RTC settings used for this recording."""
    guidance_enabled = args.rtc_mode in {"non-vjp", "vjp"}
    return {
        "rtc_mode": args.rtc_mode,
        "prediction_horizon": int(metadata["prediction_horizon"]),
        "execution_horizon": args.execution_horizon,
        "prefix_len": args.prefix_len,
        "decay_end": args.decay_end,
        "rtc_schedule": args.rtc_schedule,
        "max_guidance_weight": (5.0 if args.max_guidance_weight is None else args.max_guidance_weight) if guidance_enabled else None,
    }


def add_recording_frame(
    dataset: LeRobotDataset,
    observation: Mapping[str, object],
    sent_action: Mapping[str, object],
    capture_time_s: float | None = None,
    policy_delta_action: np.ndarray | None = None,
) -> None:
    frame = {
        "observation.state": np.asarray(observation["observation.state"], dtype=np.float32).copy(),
        "observation.images.base_0_rgb": np.asarray(
            observation["observation.images.base_0_rgb"], dtype=np.uint8
        ).copy(),
        "observation.images.left_wrist_0_rgb": np.asarray(
            observation["observation.images.left_wrist_0_rgb"], dtype=np.uint8
        ).copy(),
        "action": robot_observation_to_state(sent_action),
        "task": str(observation["prompt"]),
    }
    if capture_time_s is not None:
        frame["debug.capture_time"] = np.asarray([capture_time_s], dtype=np.float32)
    if policy_delta_action is not None:
        frame["debug.delta_action"] = np.asarray(policy_delta_action, dtype=np.float32).copy()

    dataset.add_frame(frame)


def capture_policy_observation(
    *,
    robot: URRobot,
    base_camera: OrbbecCamera,
    wrist_camera: OrbbecCamera,
    prompt: str,
    camera_max_age_ms: int,
) -> tuple[dict[str, object], dict[str, object]]:
    robot_observation = robot.get_observation()
    base_image = base_camera.read_latest(max_age_ms=camera_max_age_ms)
    wrist_image = wrist_camera.read_latest(max_age_ms=camera_max_age_ms)
    policy_observation = build_observation(
        robot_observation_to_state(robot_observation),
        base_image,
        wrist_image,
        prompt,
    )
    return policy_observation, robot_observation


def run_control_loop(
    *,
    robot: URRobot,
    base_camera: OrbbecCamera,
    wrist_camera: OrbbecCamera,
    policy: base_policy.BasePolicy,
    prompt: str,
    control_fps: float,
    camera_max_age_ms: int,
    dataset: LeRobotDataset | None,
    rtc_mode: str = "off",
    record_policy_infer_time: bool = False,
    record_policy_delta_action: bool = False,
) -> None:
    """Observe at policy frequency and publish targets to the RTDE interpolator."""
    control_interval_s = 1.0 / control_fps
    next_control_t = time.perf_counter()
    previous_capture_t: float | None = None
    step = 0

    while True:
        policy_observation, robot_observation = capture_policy_observation(
            robot=robot,
            base_camera=base_camera,
            wrist_camera=wrist_camera,
            prompt=prompt,
            camera_max_age_ms=camera_max_age_ms,
        )
        capture_time_s = None
        if record_policy_infer_time:
            capture_t = time.perf_counter()
            capture_time_s = 0.0 if previous_capture_t is None else capture_t - previous_capture_t
            previous_capture_t = capture_t

        infer_start = time.perf_counter()
        result = policy.infer(policy_observation)
        infer_elapsed_s = time.perf_counter() - infer_start
        if "actions" not in result:
            raise RuntimeError("Policy result does not contain actions")
        if record_policy_delta_action and DELTA_ACTIONS_KEY not in result:
            raise RuntimeError(f"Policy result does not contain {DELTA_ACTIONS_KEY}")
        target_action = np.asarray(result["actions"], dtype=np.float32)
        if target_action.shape != (UR_ACTION_DIM,) or not np.all(np.isfinite(target_action)):
            raise RuntimeError(
                f"Expected one finite absolute action with shape ({UR_ACTION_DIM},), "
                f"got {target_action.shape}"
            )
        delta_action = (
            np.asarray(result[DELTA_ACTIONS_KEY], dtype=np.float32)
            if record_policy_delta_action
            else None
        )
        if delta_action is not None and (delta_action.shape != (UR_ACTION_DIM,) or not np.all(np.isfinite(delta_action))):
            raise RuntimeError(f"Expected one finite delta policy action with shape ({UR_ACTION_DIM},), got {delta_action.shape}")
        robot_action = policy_action_to_robot_action(target_action)
        sent_action = robot.send_trajectory_action(
            robot_action,
            robot_observation,
            transition_duration_s=control_interval_s,
            step_limit_check=False,
        )

        if dataset is not None:
            add_recording_frame(
                dataset,
                policy_observation,
                sent_action,
                capture_time_s,
                policy_delta_action=delta_action,
            )

        step += 1
        if infer_elapsed_s > control_interval_s and rtc_mode != "off":
            logger.warning(
                "Step %d inference took %.1f ms, exceeding the %.1f ms control period",
                step,
                infer_elapsed_s * 1000,
                control_interval_s * 1000,
            )
        if step % max(round(control_fps), 1) == 0:
            xyz_keys = ACTION_FEATURES[:3]
            xyz = tuple(float(sent_action[key]) for key in xyz_keys)
            print(f"Step {step}: published TCP target ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})")

        next_control_t += control_interval_s
        precise_sleep(max(next_control_t - time.perf_counter(), 0.0))
        if next_control_t < time.perf_counter() - control_interval_s:
            next_control_t = time.perf_counter()


def _validate_warmup_result(result: Mapping[str, object]) -> None:
    actions = np.asarray(result.get("actions"))
    if actions.ndim != 2 or actions.shape[1] != UR_ACTION_DIM:
        raise RuntimeError(
            f"Expected warmup actions shaped (prediction_horizon, {UR_ACTION_DIM}), "
            f"got {actions.shape}"
        )
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("Policy warmup returned NaN or Inf actions")


def main(argv: Sequence[str] | None = None) -> None:
    init_logging()
    args = parse_args(argv)

    client = websocket_client_policy.WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
    metadata = client.get_server_metadata()
    validate_rtc_arguments(args, metadata)
    print(f"Server metadata: {metadata}")
    print(f"Selected RTC mode: {args.rtc_mode}")

    execution_policy = create_execution_policy(client, metadata, args)
    wrist_width, wrist_height = args.gemini_305_resolution
    base_width, base_height = args.gemini_336_resolution

    wrist_camera = OrbbecCamera(
        OrbbecCameraConfig(
            serial_number_or_name=args.gemini_305_serial,
            width=wrist_width,
            height=wrist_height,
            fps=args.camera_fps,
            warmup_s=args.camera_warmup_s,
            exposure=150,
            gain=19,
            white_balance=4200,
        )
    )
    base_camera = OrbbecCamera(
        OrbbecCameraConfig(
            serial_number_or_name=args.gemini_336_serial,
            width=base_width,
            height=base_height,
            fps=args.camera_fps,
            warmup_s=args.camera_warmup_s,
            exposure=150,
            gain=19,
            white_balance=4200,
        )
    )

    robot = URRobot(
        URRobotConfig(
            id="openpi-ur-controller",
            robot_ip=args.robot_ip,
            use_gripper=True,
            max_tcp_translation_delta_m=args.max_command_translation_m,
            max_tcp_rotation_delta_rad=args.max_command_rotation_rad,
            tcp_pose=args.tcp_pose,
            rtde_frequency_hz=RTDE_CONTROL_FREQUENCY_HZ,
            rtde_check_pose_safety=False,
        )
    )

    dataset = None
    rtde_recorder = None
    rtde_recording = False
    robot_state_path = None
    try:
        logger.info("Connecting cameras")
        wrist_camera.connect()
        base_camera.connect()
        logger.info("Connecting robot")
        robot.connect()
        # set inital robot state
        initialized_action = initialize_robot(robot, args.initial_rpy)
        logger.info(
            "Robot initialized at current TCP position with "
            f"RPY={tuple(args.initial_rpy)} rad and gripper fully open: {initialized_action}"
        )
        logger.info("All devices connected")

        initial_observation, _ = capture_policy_observation(
            robot=robot,
            base_camera=base_camera,
            wrist_camera=wrist_camera,
            prompt=args.prompt,
            camera_max_age_ms=args.camera_max_age_ms,
        )
        for index in range(args.warmup_inferences):
            start = time.perf_counter()
            result = client.infer(initial_observation)
            _validate_warmup_result(result)
            logger.info(f"Plain warmup {index + 1}: {(time.perf_counter() - start) * 1000:.1f} ms")

        if args.rtc_mode != "off":
            start = time.perf_counter()
            warmup_execution_policy(execution_policy, initial_observation)
            logger.info(f"{args.rtc_mode} warmup: {(time.perf_counter() - start) * 1000:.1f} ms")

        if not args.start_immediately:
            confirmation = input("Type 's' to begin robot motion, or press Enter to exit: ")
            if confirmation.strip().lower() != "s":
                print("Start cancelled")
                return

        if args.record:
            dataset = create_recording_dataset(
                repo_id=args.repo_id,
                root=args.dataset_dir / args.repo_id,
                fps=int(args.control_fps),
                image_writer_threads_per_camera=args.image_writer_threads_per_camera,
                record_policy_infer_time=args.record_policy_infer_time,
                record_policy_delta_action=args.record_policy_delta_action,
                rtc_config=build_rtc_recording_config(args, metadata),
            )
            logger.info(f"Recording dataset to {dataset.root}")

        if args.record_robot_state:
            args.robot_state_record_dir.mkdir(parents=True, exist_ok=True)
            run_timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            robot_state_path = args.robot_state_record_dir / f"pi05_rtc_{run_timestamp}_robot_state.csv"
            rtde_recorder = rtde_receive.RTDEReceiveInterface(args.robot_ip, RTDE_RECORD_FREQUENCY_HZ, RTDE_RECORD_VARIABLES)
            if not rtde_recorder.startFileRecording(str(robot_state_path), RTDE_RECORD_VARIABLES):
                raise RuntimeError(f"Failed to start RTDE recording: {robot_state_path}")
            rtde_recording = True
            logger.info(f"Recording RTDE robot state to {robot_state_path}")

        if robot.gripper is not None:
            previous_sequence = robot.gripper.get_state().sequence
            robot.gripper.wait_for_state(previous_sequence, timeout_s=1.0)
            logger.info("Received fresh gripper state before starting robot control")

        print("Robot control started. Press Ctrl+C to stop")
        try:
            run_control_loop(
                robot=robot,
                base_camera=base_camera,
                wrist_camera=wrist_camera,
                policy=execution_policy,
                prompt=args.prompt,
                control_fps=args.control_fps,
                camera_max_age_ms=args.camera_max_age_ms,
                dataset=dataset,
                record_policy_infer_time=args.record_policy_infer_time,
                record_policy_delta_action=args.record_policy_delta_action,
                rtc_mode=args.rtc_mode,
            )
        except KeyboardInterrupt:
            print("Stop requested")
    finally:
        # RTC close drains a pending WebSocket call before releasing its thread.
        with contextlib.suppress(Exception):
            close_execution_policy(execution_policy)
        if rtde_recorder is not None:
            try:
                if rtde_recording and not rtde_recorder.stopFileRecording():
                    logger.warning("Failed to stop RTDE recording cleanly")
            except Exception:
                logger.exception("Failed to stop RTDE recording cleanly")
            with contextlib.suppress(Exception):
                rtde_recorder.disconnect()
            if robot_state_path is not None:
                print(f"RTDE robot state recording: {robot_state_path}")
        print("Disconnecting devices")
        if getattr(robot, "is_connected", False):
            with contextlib.suppress(Exception):
                robot.disconnect()
        if getattr(base_camera, "is_connected", False):
            with contextlib.suppress(Exception):
                base_camera.disconnect()
        if getattr(wrist_camera, "is_connected", False):
            with contextlib.suppress(Exception):
                wrist_camera.disconnect()
        if dataset is not None:
            try:
                if dataset.has_pending_frames():
                    dataset.save_episode()
                    print(f"Recorded episode with {dataset.num_frames} frames")
            finally:
                dataset.finalize()
                print(f"Dataset finalized at {dataset.root}")
        print("Devices disconnected")


if __name__ == "__main__":
    main()
