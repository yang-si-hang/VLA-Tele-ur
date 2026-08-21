#!/usr/bin/env python3
"""Run an OpenPI policy server as a closed-loop controller for a UR robot.

Requires a reachable policy server, UR robot, and two Gemini cameras. The script
commands the robot and can record a LeRobot dataset episode, optionally including
the single base-frame policy delta action used at each control timestep.
Set --action-sampling-factor above 1 to interpolate and send actions faster than
policy inference. Example: python scripts/inference/pi05_inference.py --record-policy-delta-action
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.utils.robot_utils import precise_sleep
from lerobot_camera_orbbec import OrbbecCamera, OrbbecCameraConfig
from lerobot_robot_ur import URRobot, URRobotConfig
from lerobot_robot_ur.ur import GRIPPER_FEATURE, TCP_FEATURES
from openpi_client import image_tools, websocket_client_policy

from scripts.openpi.ur_action_adapter import (
    DELTA_ACTIONS_KEY,
    UR_ACTION_DIM,
    create_absolute_action_broker,
    create_rtc_action_broker,
)
from utils.const import DATA_PATH
from utils.ur_action_utils import interpolate_actions

# state 慢于 action 两个step，理论上只慢一个step，原因需要查找

POLICY_HOST = "127.0.0.1"
POLICY_PORT = 8000

DEFAULT_ROBOT_IP = "192.168.253.102"
DEFAULT_GEMINI_305_SERIAL = "CV2L360000C7"
DEFAULT_GEMINI_336_SERIAL = "CP9JA530008V"
DEFAULT_PROMPT = "Pick up the yellow can and place it upright on the red tape marker."
DEFAULT_CONTROL_FPS = 20
DEFAULT_EXECUTION_HORIZON = 10

ACTION_FEATURES = (*TCP_FEATURES, GRIPPER_FEATURE)
CAMERA_CONFIG = {
    "observation.images.left_wrist_0_rgb": {
        "crop": [124, 16, 572, 464],
        "size": [224, 224],
    },
    "observation.images.base_0_rgb": {
        "crop": [48, 8, 272, 232],
        "size": None,
    },
}

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)          # 设置根记录器的等级为 INFO

console_handler = logging.StreamHandler()   # 创建一个控制台输出处理器
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

if not root_logger.handlers:
    root_logger.addHandler(console_handler)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Control a UR robot with actions from an OpenPI policy server")

    parser.add_argument("--policy-host", default=POLICY_HOST)
    parser.add_argument("--policy-port", type=int, default=POLICY_PORT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--execution-horizon", type=int, default=DEFAULT_EXECUTION_HORIZON)
    parser.add_argument("--warmup-inferences", type=int, default=2)
    parser.add_argument("--control-fps", type=float, default=DEFAULT_CONTROL_FPS)
    parser.add_argument("--action-sampling-factor", type=int, default=25, help="Number of evenly spaced robot actions sent per policy control period")
    parser.add_argument("--use-rtc", default=False, action=argparse.BooleanOptionalAction, help="Use RTC action broker instead of absolute action broker")
    parser.add_argument("--prefix-len", type=int, default=2)
    parser.add_argument("--decay-end", type=int, default=4)
    parser.add_argument("--use-vjp", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--start-immediately", action="store_true", help="Skip the interactive safety confirmation before sending actions")

    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True, help="Record one LeRobot Dataset episode during policy execution")
    parser.add_argument("--record-policy-delta-action", default=True, action=argparse.BooleanOptionalAction, help="Record the single policy delta action in the robot base frame at each control timestep")
    parser.add_argument("--repo-id", default=f"pi05_pick_{timestamp}")
    parser.add_argument("--dataset-dir", type=Path, default=DATA_PATH / "deploy")
    parser.add_argument("--image-writer-threads-per-camera", type=int, default=4)

    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--tcp-pose", type=float, nargs=6, default=(0.0, 0.0, 0.174, 0.0, 0.0, 0.0), metavar=("X", "Y", "Z", "RX", "RY", "RZ"))
    parser.add_argument("--max-command-translation-m", type=float, default=0.1)
    parser.add_argument("--max-command-rotation-rad", type=float, default=0.2)

    parser.add_argument("--gemini-305-serial", default=DEFAULT_GEMINI_305_SERIAL)
    parser.add_argument("--gemini-336-serial", default=DEFAULT_GEMINI_336_SERIAL)
    parser.add_argument("--gemini-305-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(640, 480))
    parser.add_argument("--gemini-336-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(320, 240))
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
    if args.prefix_len <= 0:
        parser.error("--prefix-len must be positive")
    if args.decay_end < args.prefix_len:
        parser.error("--decay-end must be greater than or equal to --prefix-len")
    if not math.isfinite(args.control_fps) or args.control_fps <= 0:
        parser.error("--control-fps must be finite and positive")
    if args.action_sampling_factor <= 0:
        parser.error("--action-sampling-factor must be positive")
    if args.record and not float(args.control_fps).is_integer():
        parser.error("--control-fps must be an integer when --record is enabled")
    if not args.repo_id.strip():
        parser.error("--repo-id must not be empty")
    if args.image_writer_threads_per_camera <= 0:
        parser.error("--image-writer-threads-per-camera must be positive")
    if args.camera_fps <= 0:
        parser.error("--camera-fps must be positive")
    if args.camera_max_age_ms < 0:
        parser.error("--camera-max-age-ms must be non-negative")
    if not math.isfinite(args.camera_warmup_s) or args.camera_warmup_s < 0:
        parser.error("--camera-warmup-s must be finite and non-negative")
    for camera_name in ("gemini_305", "gemini_336"):
        resolution = getattr(args, f"{camera_name}_resolution")
        if any(value <= 0 for value in resolution):
            parser.error(
                f"--{camera_name.replace('_', '-')}-resolution values must be positive"
            )
    for name in ("max_command_translation_m", "max_command_rotation_rad"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    return args


def prepare_camera_image(image: np.ndarray, camera_key: str) -> np.ndarray:
    """Validate, crop, and optionally resize one RGB image for OpenPI."""

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")
    if camera_key not in CAMERA_CONFIG:
        raise ValueError(f"Unknown camera key: {camera_key}")

    config = CAMERA_CONFIG[camera_key]
    left, top, right, bottom = config["crop"]
    height, width = image.shape[:2]
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(
            f"Crop {config['crop']} for {camera_key} is outside image shape "
            f"{image.shape}"
        )

    processed = image[top:bottom, left:right]
    size = config["size"]
    if size is not None:
        target_height, target_width = size
        processed = image_tools.resize_with_pad(
            processed,
            target_height,
            target_width,
        )

    processed = np.ascontiguousarray(processed, dtype=np.uint8)
    expected_height = size[0] if size is not None else bottom - top
    expected_width = size[1] if size is not None else right - left
    if processed.shape != (expected_height, expected_width, 3):
        raise RuntimeError(
            f"Processed {camera_key} image has shape {processed.shape}, expected "
            f"({expected_height}, {expected_width}, 3)"
        )
    return processed


def robot_observation_to_state(observation: dict[str, object]) -> np.ndarray:
    """Convert a LeRobot UR observation dictionary to the policy state vector."""

    missing = [name for name in ACTION_FEATURES if name not in observation]
    if missing:
        raise ValueError(f"Robot observation is missing required keys: {missing}")

    state = np.asarray(
        [observation[name] for name in ACTION_FEATURES],
        dtype=np.float32,
    )
    if state.shape != (UR_ACTION_DIM,):
        raise ValueError(
            f"Expected UR state shape ({UR_ACTION_DIM},), got {state.shape}"
        )
    if not np.all(np.isfinite(state)):
        raise ValueError("Robot observation contains NaN or Inf values")
    return state


def build_observation(
    state: np.ndarray,
    base_image: np.ndarray,
    wrist_image: np.ndarray,
    prompt: str,
) -> dict[str, object]:
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (UR_ACTION_DIM,):
        raise ValueError(
            f"Expected UR state shape ({UR_ACTION_DIM},), got {state.shape}"
        )
    if not np.all(np.isfinite(state)):
        raise ValueError("UR state contains NaN or Inf values")

    return {
        "observation.state": state,
        "observation.images.base_0_rgb": prepare_camera_image(
            base_image,
            "observation.images.base_0_rgb",
        ),
        "observation.images.left_wrist_0_rgb": prepare_camera_image(
            wrist_image,
            "observation.images.left_wrist_0_rgb",
        ),
        "prompt": prompt,
    }


def policy_action_to_robot_action(action: np.ndarray) -> dict[str, float]:
    """Convert one broker action into the dictionary accepted by URRobot."""

    values = np.asarray(action, dtype=np.float32)
    if values.shape != (UR_ACTION_DIM,):
        raise ValueError(
            f"Expected policy action shape ({UR_ACTION_DIM},), got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Policy action contains NaN or Inf values")

    return {
        name: float(value)
        for name, value in zip(ACTION_FEATURES, values, strict=True)
    }


def create_recording_dataset(
    *,
    repo_id: str,
    root: Path,
    fps: int,
    image_writer_threads_per_camera: int,
    record_policy_delta_action: bool = False,
) -> LeRobotDataset:
    """Create an image-based LeRobot Dataset for one policy-control session."""

    vector_feature = {
        "dtype": "float32",
        "shape": (UR_ACTION_DIM,),
        "names": list(ACTION_FEATURES),
    }
    image_feature = {
        "dtype": "image",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channels"],
    }
    features = {
        "action": vector_feature,
        "debug.capture_time": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["seconds_since_previous_frame"],
        },
        "observation.state": vector_feature.copy(),
        "observation.images.left_wrist_0_rgb": image_feature,
        "observation.images.base_0_rgb": image_feature.copy(),
    }
    if record_policy_delta_action:
        features["debug.delta_action"] = vector_feature.copy()
    return LeRobotDataset.create(
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


def add_recording_frame(
    dataset: LeRobotDataset,
    policy_observation: dict[str, object],
    sent_action: dict[str, object],
    capture_time_s: float,
    policy_delta_action: np.ndarray | None = None,
) -> None:
    """Record the pre-action observation and the action actually sent to the UR."""

    frame = {
        "observation.state": np.asarray(
            policy_observation["observation.state"],
            dtype=np.float32,
        ).copy(),
        "observation.images.base_0_rgb": np.asarray(
            policy_observation["observation.images.base_0_rgb"],
            dtype=np.uint8,
        ).copy(),
        "observation.images.left_wrist_0_rgb": np.asarray(
            policy_observation["observation.images.left_wrist_0_rgb"],
            dtype=np.uint8,
        ).copy(),
        "action": robot_observation_to_state(sent_action),
        "debug.capture_time": np.asarray([capture_time_s], dtype=np.float32),
        "task": str(policy_observation["prompt"]),
    }
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
    """Capture synchronized-enough latest sensor values for one control step."""

    robot_observation = robot.get_observation()
    base_image = base_camera.read_latest(max_age_ms=camera_max_age_ms)
    wrist_image = wrist_camera.read_latest(max_age_ms=camera_max_age_ms)
    policy_observation = build_observation(
        state=robot_observation_to_state(robot_observation),
        base_image=base_image,
        wrist_image=wrist_image,
        prompt=prompt,
    )
    return policy_observation, robot_observation


def run_control_loop(
    *,
    robot: URRobot,
    base_camera: OrbbecCamera,
    wrist_camera: OrbbecCamera,
    policy: object,
    prompt: str,
    control_fps: float,
    action_sampling_factor: int,
    camera_max_age_ms: int,
    dataset: LeRobotDataset | None = None,
    record_policy_delta_action: bool = False,
) -> None:
    """Observe and infer at control FPS, sending interpolated actions faster."""

    control_interval_s = 1.0 / control_fps
    action_interval_s = control_interval_s / action_sampling_factor
    next_control_t = time.perf_counter()
    previous_capture_t: float | None = None
    previous_policy_action: np.ndarray | None = None
    step = 0

    while True:
        policy_observation, robot_observation = capture_policy_observation(
            robot=robot,
            base_camera=base_camera,
            wrist_camera=wrist_camera,
            prompt=prompt,
            camera_max_age_ms=camera_max_age_ms,
        )
        capture_t = time.perf_counter()
        capture_time_s = (
            0.0 if previous_capture_t is None else capture_t - previous_capture_t
        )
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
            raise RuntimeError(f"Expected one finite absolute policy action with shape ({UR_ACTION_DIM},), got {target_action.shape}")
        delta_action = (
            np.asarray(result[DELTA_ACTIONS_KEY], dtype=np.float32)
            if record_policy_delta_action
            else None
        )
        if delta_action is not None and (delta_action.shape != (UR_ACTION_DIM,) or not np.all(np.isfinite(delta_action))):
            raise RuntimeError(f"Expected one finite delta policy action with shape ({UR_ACTION_DIM},), got {delta_action.shape}")
        start_action = (
            robot_observation_to_state(robot_observation)
            if previous_policy_action is None
            else previous_policy_action
        )
        sampled_actions = interpolate_actions(start_action, target_action, action_sampling_factor)
        previous_policy_action = target_action.copy()
        sent_action = robot_observation
        for sample_index, sampled_action in enumerate(sampled_actions, start=1):
            send_t = next_control_t + sample_index * action_interval_s
            precise_sleep(max(send_t - time.perf_counter(), 0.0))
            robot_action = policy_action_to_robot_action(sampled_action)
            sent_action = robot.send_action(robot_action, sent_action, False)
        if dataset is not None:
            add_recording_frame(
                dataset,
                policy_observation,
                sent_action,
                capture_time_s,
                policy_delta_action=delta_action,
            )

        step += 1
        # if infer_elapsed_s > control_interval_s:
        #     logging.warning(
        #         f"Step {step}: inference took {infer_elapsed_s * 1000:.1f} ms, "
        #         f"longer than the {control_interval_s * 1000:.1f} ms control period"
        #     )
        if step % max(round(control_fps), 1) == 0:
            print(
                f"Step {step}: sent TCP target "
                f"({sent_action['ee.x']:.3f}, "
                f"{sent_action['ee.y']:.3f}, "
                f"{sent_action['ee.z']:.3f})"
            )

        next_control_t += control_interval_s
        precise_sleep(max(next_control_t - time.perf_counter(), 0.0))
        if next_control_t < time.perf_counter() - control_interval_s:
            next_control_t = time.perf_counter()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    wrist_width, wrist_height = args.gemini_305_resolution
    base_width, base_height = args.gemini_336_resolution
    wrist_camera = OrbbecCamera(
        OrbbecCameraConfig(
            serial_number_or_name=args.gemini_305_serial,
            width=wrist_width,
            height=wrist_height,
            fps=args.camera_fps,
            warmup_s=args.camera_warmup_s,
            exposure=155,
            gain=35,
            white_balance=3700,
        )
    )
    base_camera = OrbbecCamera(
        OrbbecCameraConfig(
            serial_number_or_name=args.gemini_336_serial,
            width=base_width,
            height=base_height,
            fps=args.camera_fps,
            warmup_s=args.camera_warmup_s,
            exposure=160,
            gain=19,
            white_balance=4200,
        )
    )
    robot = URRobot(
        URRobotConfig(
            id="openpi-ur-controller",
            robot_ip=args.robot_ip,
            use_gripper=True,
            gripper_control_frequency_hz=args.control_fps * args.action_sampling_factor,
            max_tcp_translation_delta_m=args.max_command_translation_m,
            max_tcp_rotation_delta_rad=args.max_command_rotation_rad,
            tcp_pose=args.tcp_pose,
            rtde_check_pose_safety=False,
        )
    )

    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.policy_host,
        port=args.policy_port,
    )
    metadata = client.get_server_metadata()
    print("Server metadata:")
    print(metadata)

    dataset = None
    if args.record:
        dataset = create_recording_dataset(
            repo_id=args.repo_id,
            root=args.dataset_dir / f"{args.repo_id}",
            fps=int(args.control_fps),
            image_writer_threads_per_camera=args.image_writer_threads_per_camera,
            record_policy_delta_action=args.record_policy_delta_action,
        )
        print(f"Recording dataset to {dataset.root}")

    try:
        print("Connecting cameras")
        wrist_camera.connect()
        base_camera.connect()
        print("Connecting robot")
        robot.connect()
        print("All devices connected")

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
            elapsed_s = time.perf_counter() - start
            actions = np.asarray(result.get("actions"))
            if actions.ndim != 2 or actions.shape[1] != UR_ACTION_DIM:
                raise RuntimeError(
                    "Expected warmup actions with shape "
                    f"(prediction_horizon, {UR_ACTION_DIM}), got {actions.shape}"
                )
            if not np.all(np.isfinite(actions)):
                raise RuntimeError("Policy warmup returned NaN or Inf actions")
            print(f"Warmup {index + 1}: {elapsed_s * 1000:.1f} ms")

        if args.use_rtc:
            print("Using RTC action broker")
            policy = create_rtc_action_broker(
                client,
                metadata,
                replan_interval=args.execution_horizon,     # execution chunk
                prefix_len=args.prefix_len,
                decay_end=args.decay_end,
                use_vjp=args.use_vjp,
            )
        else:
            print("Using absolute action broker")
            policy = create_absolute_action_broker(
                client,
                metadata,
                execution_horizon=args.execution_horizon,
            )

        if not args.start_immediately:
            confirmation = input(
                "Type 'start' to begin robot motion, or press Enter to exit: "
            )
            if confirmation.strip().lower() != "start":
                print("Start cancelled")
                return

        print("Robot control started. Press Ctrl+C to stop")
        try:
            run_control_loop(
                robot=robot,
                base_camera=base_camera,
                wrist_camera=wrist_camera,
                policy=policy,
                prompt=args.prompt,
                control_fps=args.control_fps,
                action_sampling_factor=args.action_sampling_factor,
                camera_max_age_ms=args.camera_max_age_ms,
                dataset=dataset,
                record_policy_delta_action=args.record_policy_delta_action,
            )
        except KeyboardInterrupt:
            print("Stop requested")
        finally:
            policy.reset()
    finally:
        print("Disconnecting devices")
        if robot.is_connected:
            with contextlib.suppress(Exception):
                robot.disconnect()
        if base_camera.is_connected:
            with contextlib.suppress(Exception):
                base_camera.disconnect()
        if wrist_camera.is_connected:
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
