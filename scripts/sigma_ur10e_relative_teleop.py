#!/usr/bin/env python
"""Control a UR10e from a Force Dimension Sigma using relative Cartesian poses.

The script captures the Sigma and UR TCP poses when the user presses ENTER.
Subsequent Sigma motion is applied relative to those reference poses:

    p_ur_target = p_ur_0 + position_scale * R_map * (p_sigma - p_sigma_0)

    R_sigma_delta = R_sigma * inverse(R_sigma_0)
    R_ur_delta = R_map * R_sigma_delta * inverse(R_map)
    R_ur_target = R_ur_delta * R_ur_0

Sigma gravity compensation is enabled by default. Force feedback is not
implemented.
"""

from __future__ import annotations

import argparse
import math
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot_robot_ur import URRobot, URRobotConfig
from lerobot_robot_ur.ur import GRIPPER_FEATURE, TCP_FEATURES
from lerobot_teleoperator_sigma import GRIPPER_ANGLE_FEATURE, Sigma, SigmaConfig


root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)          # 设置根记录器的等级为 INFO

console_handler = logging.StreamHandler()   # 创建一个控制台输出处理器
console_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

if not root_logger.handlers:
    root_logger.addHandler(console_handler)


POSITION_SCALE = 1.0
SIGMA_TO_UR_BASE_ROTATION_RAD = (0.0, 0.0, np.pi / 2) # sigma in UR base frame: Sigma X->UR Y, Sigma Y->UR -X
CONTROL_FREQUENCY_HZ = 50.0
REFERENCE_SAMPLE_COUNT = 20
# The connected right-hand sigma.7 reports an expected gripper joint range of
# approximately [-0.00194, 0.53154] rad. The SDK uses the opposite sign for a
# left-hand device, so the mapping below operates on angle magnitude.
SIGMA_GRIPPER_CLOSED_ANGLE_RAD = 0.0
SIGMA_GRIPPER_OPEN_ANGLE_RAD = 0.5315

# Total motion limits relative to the captured UR reference pose. These are in
# addition to URRobot's per-command translation and rotation step limits.
MAX_RELATIVE_TRANSLATION_M = 0.25
MAX_RELATIVE_ROTATION_RAD = math.pi / 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control a UR10e from a Sigma using relative Cartesian poses.")

    # UR10e connection.
    parser.add_argument("--robot-ip", default="192.168.253.102", help="UR10e controller IP address")

    # Sigma-to-UR motion mapping.
    parser.add_argument("--position-scale", type=float, default=POSITION_SCALE, help="Scale applied to Sigma translation relative to its captured start pose")
    parser.add_argument("--base-rotation-offset-rad", type=float, nargs=3, metavar=("RX", "RY", "RZ"), default=SIGMA_TO_UR_BASE_ROTATION_RAD, help="Rotation vector mapping Sigma base axes into UR base axes")

    # Sigma connection and reference capture.
    parser.add_argument("--sdk-path", type=Path, default=Path("/opt/forcedimension/sdk"), help="Force Dimension SDK root directory")
    sigma_selector = parser.add_mutually_exclusive_group()
    sigma_selector.add_argument("--sigma-device-index", type=int, default=None)
    sigma_selector.add_argument("--sigma-serial-number", type=int, default=None)
    parser.add_argument("--reference-samples", type=int, default=REFERENCE_SAMPLE_COUNT, help="Number of stationary Sigma samples averaged for the start pose")
    parser.add_argument("--gravity-compensation", action=argparse.BooleanOptionalAction, default=True, help="Enable Sigma gravity compensation (default: enabled; use --no-gravity-compensation to disable)")
    parser.add_argument("--sigma-gripper-closed-angle-rad", type=float, default=SIGMA_GRIPPER_CLOSED_ANGLE_RAD, help="Magnitude of the Sigma gripper angle at the closed limit")
    parser.add_argument("--sigma-gripper-open-angle-rad", type=float, default=SIGMA_GRIPPER_OPEN_ANGLE_RAD, help="Magnitude of the Sigma gripper angle at the open limit")

    # Control-loop timing.
    parser.add_argument("--fps", type=float, default=CONTROL_FREQUENCY_HZ, help="Outer teleoperation command frequency")

    # Workspace and per-command safety limits.
    parser.add_argument("--max-relative-translation-m", type=float, default=MAX_RELATIVE_TRANSLATION_M, help="Maximum TCP displacement from the captured UR start position")
    parser.add_argument("--max-relative-rotation-rad", type=float, default=MAX_RELATIVE_ROTATION_RAD, help="Maximum TCP orientation change from the captured UR start orientation")
    parser.add_argument("--max-command-translation-m", type=float, default=0.05, help="URRobot maximum translation step for each outer-loop command")
    parser.add_argument("--max-command-rotation-rad", type=float, default=0.2, help="URRobot maximum rotation step for each outer-loop command")

    # Runtime mode.
    parser.add_argument("--dry-run", action="store_true", help="Read both devices and print targets without sending UR motion commands")
    parser.add_argument("--duration-s", type=float, default=None, help="Optional run duration; by default run until Ctrl+C")

    args = parser.parse_args()

    if not math.isfinite(args.position_scale) or args.position_scale <= 0:
        parser.error("--position-scale must be finite and positive")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    if args.reference_samples <= 0:
        parser.error("--reference-samples must be positive")
    if (
        not math.isfinite(args.sigma_gripper_closed_angle_rad)
        or args.sigma_gripper_closed_angle_rad < 0
    ):
        parser.error("--sigma-gripper-closed-angle-rad must be finite and non-negative")
    if (
        not math.isfinite(args.sigma_gripper_open_angle_rad)
        or args.sigma_gripper_open_angle_rad <= args.sigma_gripper_closed_angle_rad
    ):
        parser.error(
            "--sigma-gripper-open-angle-rad must be finite and greater than "
            "--sigma-gripper-closed-angle-rad"
        )
    for name in (
        "max_relative_translation_m",
        "max_relative_rotation_rad",
        "max_command_translation_m",
        "max_command_rotation_rad",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.duration_s is not None and (
        not math.isfinite(args.duration_s) or args.duration_s <= 0
    ):
        parser.error("--duration-s must be finite and positive")
    return args


@dataclass(frozen=True, slots=True)
class CartesianPose:
    position: np.ndarray
    orientation: Rotation


def action_to_pose(action: dict[str, Any]) -> CartesianPose:
    """Convert LeRobot's UR-compatible action dictionary to a Cartesian pose."""

    missing = [name for name in TCP_FEATURES if name not in action]
    if missing:
        raise ValueError(f"Pose action is missing required keys: {missing}")

    values = np.asarray([action[name] for name in TCP_FEATURES], dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("Pose action must contain six finite values")
    return CartesianPose(values[:3], Rotation.from_rotvec(values[3:]))


def pose_to_action(pose: CartesianPose) -> dict[str, float]:
    """Convert a Cartesian pose to the action format expected by URRobot."""

    values = np.concatenate((pose.position, pose.orientation.as_rotvec()))
    return {
        name: float(value) for name, value in zip(TCP_FEATURES, values, strict=True)
    }


def sigma_gripper_angle_to_ur_position(
    angle_rad: float,
    *,
    closed_angle_rad: float,
    open_angle_rad: float,
) -> float:
    """Map signed Sigma opening angle to UR gripper position, 0=open and 1=closed."""

    angle_magnitude = abs(float(angle_rad))
    if not math.isfinite(angle_magnitude):
        raise ValueError("Sigma gripper angle must be finite")
    if (
        not math.isfinite(closed_angle_rad)
        or not math.isfinite(open_angle_rad)
        or closed_angle_rad < 0
        or open_angle_rad <= closed_angle_rad
    ):
        raise ValueError("Invalid Sigma gripper angle range")

    opening_fraction = (angle_magnitude - closed_angle_rad) / (
        open_angle_rad - closed_angle_rad
    )
    return float(1.0 - np.clip(opening_fraction, 0.0, 1.0))


def capture_sigma_reference(
    teleop: Sigma,
    *,
    sample_count: int,
    sample_interval_s: float,
) -> CartesianPose:
    """Average several stationary Sigma readings to reduce reference noise."""

    positions: list[np.ndarray] = []
    rotation_vectors: list[np.ndarray] = []
    for sample_index in range(sample_count):
        pose = action_to_pose(teleop.get_action())
        positions.append(pose.position)
        rotation_vectors.append(pose.orientation.as_rotvec())
        if sample_index + 1 < sample_count:
            time.sleep(sample_interval_s)

    mean_position = np.mean(np.stack(positions), axis=0)
    mean_orientation = Rotation.from_rotvec(np.stack(rotation_vectors)).mean()
    return CartesianPose(mean_position, mean_orientation)


def compute_relative_target(
    sigma_reference: CartesianPose,
    sigma_current: CartesianPose,
    ur_reference: CartesianPose,
    *,
    position_scale: float,
    base_mapping: Rotation,
    max_relative_translation_m: float | None,
    max_relative_rotation_rad: float | None,
) -> CartesianPose:
    """Map Sigma motion relative to its reference onto the UR reference pose."""

    translation_delta = position_scale * base_mapping.apply(
        sigma_current.position - sigma_reference.position
    )
    translation_delta = _limit_vector_norm(
        translation_delta,
        max_relative_translation_m,
    )

    sigma_rotation_delta = sigma_current.orientation * sigma_reference.orientation.inv()
    ur_rotation_delta = base_mapping * sigma_rotation_delta * base_mapping.inv()    # 共轭变换
    rotation_delta_vector = _limit_vector_norm(
        ur_rotation_delta.as_rotvec(),
        max_relative_rotation_rad,
    )
    limited_ur_rotation_delta = Rotation.from_rotvec(rotation_delta_vector)

    return CartesianPose(
        position=ur_reference.position + translation_delta,
        orientation=limited_ur_rotation_delta * ur_reference.orientation,
    )


def _limit_vector_norm(vector: np.ndarray, maximum: float | None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    magnitude = float(np.linalg.norm(vector))
    if maximum is None or magnitude <= maximum or magnitude == 0.0:
        return vector
    return vector * (maximum / magnitude)


def main() -> None:
    args = parse_args()
    base_mapping = Rotation.from_rotvec(args.base_rotation_offset_rad)

    # Keep Sigma's own mapping neutral. This script owns the relative mapping.
    teleop = Sigma(
        SigmaConfig(
            id="sigma-ur10e-relative",
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
    robot = URRobot(
        URRobotConfig(
            id="ur10e-relative",
            robot_ip=args.robot_ip,
            use_gripper=True,
            max_tcp_translation_delta_m=args.max_command_translation_m,
            max_tcp_rotation_delta_rad=args.max_command_rotation_rad,
        )
    )

    try:
        teleop.connect()
        robot.connect()

        print(f"已连接 Sigma: {teleop.sigma_device.device_name}")
        print(f"已连接 UR10e: {args.robot_ip}")
        if args.dry_run:
            print("当前为 dry-run 模式，不会向 UR10e 发送运动指令。")
        print("将 Sigma 放在舒适的遥操作零位并保持静止。")
        input("确认机器人周围安全后按 ENTER 捕获主从初始位姿并开始; Ctrl+C 退出: ")

        sample_interval_s = min(0.01, 1.0 / args.fps)
        sigma_reference = capture_sigma_reference(
            teleop,
            sample_count=args.reference_samples,
            sample_interval_s=sample_interval_s,
        )
        ur_reference = action_to_pose(robot.get_observation())

        print("初始位姿已捕获，开始相对位姿遥操作。")
        print(f"position_scale={args.position_scale:.3f}, fps={args.fps:.1f}")

        control_period_s = 1.0 / args.fps
        start_time = time.perf_counter()
        next_cycle = start_time
        last_display_time = 0.0

        while args.duration_s is None or time.perf_counter() - start_time < args.duration_s:
            sigma_action = teleop.get_action()
            sigma_current = action_to_pose(sigma_action)
            target = compute_relative_target(
                sigma_reference,
                sigma_current,
                ur_reference,
                position_scale=args.position_scale,
                base_mapping=base_mapping,
                max_relative_translation_m=args.max_relative_translation_m,
                max_relative_rotation_rad=args.max_relative_rotation_rad,
            )
            target_action = pose_to_action(target)
            target_action[GRIPPER_FEATURE] = sigma_gripper_angle_to_ur_position(
                sigma_action[GRIPPER_ANGLE_FEATURE],
                closed_angle_rad=args.sigma_gripper_closed_angle_rad,
                open_angle_rad=args.sigma_gripper_open_angle_rad,
            )

            if args.dry_run:
                sent_action = target_action
            else:
                sent_action = robot.send_action(target_action)

            now = time.perf_counter()
            if now - last_display_time >= 0.2:
                values = " ".join(f"{sent_action[name]:+.4f}" for name in TCP_FEATURES)
                gripper = sent_action[GRIPPER_FEATURE]
                print(
                    f"\rUR target [x y z wx wy wz gripper]: {values} {gripper:.3f}",
                    end="",
                    flush=True,
                )
                last_display_time = now

            next_cycle += control_period_s
            remaining = next_cycle - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_cycle = time.perf_counter()

    except KeyboardInterrupt:
        print("\n收到退出请求, 正在停止 UR 控制并断开设备。")
    finally:
        disconnect_errors: list[BaseException] = []
        if robot.is_connected:
            try:
                robot.disconnect()
            except BaseException as exc:
                disconnect_errors.append(exc)
        if teleop.is_connected:
            try:
                teleop.disconnect()
            except BaseException as exc:
                disconnect_errors.append(exc)
        if disconnect_errors:
            raise RuntimeError("断开遥操作设备时发生错误") from disconnect_errors[0]
        print("\n设备已断开。")


if __name__ == "__main__":
    main()
