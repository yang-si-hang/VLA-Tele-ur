"""Move or recalibrate a host-USB Robotiq gripper.

Position mode accepts a target from 0 to 255. Recalibration mode resets the
gripper, clears faults, and activates it without starting position control.
Both modes use the configured USB-RS485 serial port, may cause physical gripper
motion, and do not create output files.

Commands:
    python -m lerobot_robot_ur.robotiq_pos_control 128 --port /dev/ttyUSB0
    python -m lerobot_robot_ur.robotiq_pos_control --recalibrate --port /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
import time

from .robotiq_usb_high_freq import (
    GripperState,
    ObjectStatus,
    RobotiqGripperUSBHighFrequency,
)


def move_and_wait(driver: RobotiqGripperUSBHighFrequency, position: int, speed: int, force: int, timeout_s: float) -> GripperState:
    driver.set_target(position, speed, force)
    deadline = time.monotonic() + timeout_s
    sequence = driver.get_state().sequence

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for Robotiq target {position}")
        state = driver.wait_for_state(sequence, timeout_s=min(remaining, 1.0))
        sequence = state.sequence
        if state.fault != 0:
            raise RuntimeError(f"Robotiq gripper fault: 0x{state.fault:02X}")
        if state.command_position == position and state.requested_position == position and state.object_status != ObjectStatus.MOVING:
            return state


def run(args: argparse.Namespace) -> int:
    driver = RobotiqGripperUSBHighFrequency(
        args.port,
        slave_id=args.slave_id,
        baudrate=args.baudrate,
        control_frequency_hz=args.frequency,
        serial_timeout_s=args.serial_timeout,
        activation_timeout_s=args.activation_timeout,
    )
    if args.recalibrate:
        state = driver.recalibrate()
        print(f"Activated gACT: {int(state.activated)}")
        print(f"Gripper status gSTA: {state.gripper_status.name}")
        print(f"Fault gFLT: 0x{state.fault:02X}")
        print(f"Actual position gPO: {state.position}")
        return 0

    try:
        driver.connect(auto_activate=True)
        state = move_and_wait(driver, args.position, args.speed, args.force, args.motion_timeout)
    finally:
        driver.disconnect()

    print(f"Command position: {args.position}")
    print(f"Requested position gPR: {state.requested_position}")
    print(f"Actual position gPO: {state.position}")
    print(f"Object status gOBJ: {state.object_status.name}")
    print(f"Fault gFLT: 0x{state.fault:02X}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Move or recalibrate a Robotiq gripper")

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("position", nargs="?", type=int, choices=range(256), metavar="POSITION", help="Target position in [0, 255]")
    action.add_argument("--recalibrate", action="store_true", help="Reset the gripper, clear faults, activate it, and exit")

    parser.add_argument("--port", default="/dev/ttyUSB0", help="USB-RS485 serial port")
    parser.add_argument("--speed", type=int, choices=range(256), default=100, metavar="SPEED", help="Motion speed in [0, 255]")
    parser.add_argument("--force", type=int, choices=range(256), default=30, metavar="FORCE", help="Motion force in [0, 255]")

    parser.add_argument("--frequency", type=float, default=100.0, help="FC23 control frequency in Hz")
    parser.add_argument("--motion-timeout", type=float, default=10.0, help="Motion completion timeout in seconds")
    parser.add_argument("--serial-timeout", type=float, default=0.03, help="Serial transaction timeout in seconds")
    parser.add_argument("--activation-timeout", type=float, default=10.0, help="Timeout for each reset or activation phase")
    parser.add_argument("--slave-id", type=int, default=9, help="Modbus slave ID")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baud rate")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
