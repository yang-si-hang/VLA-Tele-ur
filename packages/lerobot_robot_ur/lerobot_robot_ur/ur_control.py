"""Implement low-level Universal Robots TCP pose streaming through ``ur_rtde``.

``URControl`` owns the RTDE receive and control interfaces and confines all
runtime control calls to a dedicated servo thread. Public pose commands are
validated, placed in a bounded FIFO command queue, checked with the controller's
pose safety API, and streamed with ``servoL`` until stale or disconnected.
Rotation helpers convert between UR axis-angle values and LeRobot's continuous
6D representation.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation, Slerp

logger = logging.getLogger(__name__)


def rotation_to_rot6d(rotation: Rotation) -> np.ndarray:
    """Encode a single rotation using the first two columns of its matrix."""

    matrix = rotation.as_matrix()
    if matrix.shape != (3, 3):
        raise ValueError("Expected a single rotation")
    return np.concatenate((matrix[:, 0], matrix[:, 1]))


def rot6d_to_rotation(rot6d: Sequence[float]) -> Rotation:
    """Decode a 6D rotation with Gram-Schmidt orthonormalization."""

    values = np.asarray(rot6d, dtype=np.float64)
    if values.shape != (6,):
        raise ValueError(f"Rot6D must contain 6 values, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Rot6D values must be finite")

    first = values[:3]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-8:
        raise ValueError("Rot6D first direction must be non-zero")
    first = first / first_norm

    second = values[3:] - np.dot(first, values[3:]) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-8:
        raise ValueError("Rot6D directions must not be parallel")
    second = second / second_norm

    third = np.cross(first, second)
    return Rotation.from_matrix(np.column_stack((first, second, third)))


@dataclass(slots=True)
class _PoseCommand:
    pose: list[float]
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


@dataclass(slots=True)
class _MoveLCommand(_PoseCommand):
    speed: float = 0.25
    acceleration: float = 0.5


@dataclass(slots=True)
class _TrajectoryCommand:
    start_pose: list[float]
    target_pose: list[float]
    duration_s: float


@dataclass(slots=True)
class _PoseTrajectory:
    start_position: np.ndarray
    target_position: np.ndarray
    rotation_slerp: Slerp
    start_time: float
    duration_s: float

    @classmethod
    def create(
        cls,
        start_pose: Sequence[float],
        target_pose: Sequence[float],
        *,
        start_time: float,
        duration_s: float,
    ) -> _PoseTrajectory:
        start = np.asarray(start_pose, dtype=np.float64)
        target = np.asarray(target_pose, dtype=np.float64)
        rotations = Rotation.from_rotvec(np.stack((start[3:], target[3:])))
        return cls(
            start_position=start[:3].copy(),
            target_position=target[:3].copy(),
            rotation_slerp=Slerp([0.0, 1.0], rotations),
            start_time=start_time,
            duration_s=duration_s,
        )

    def evaluate(self, current_time: float) -> list[float]:
        alpha = float(np.clip((current_time - self.start_time) / self.duration_s, 0.0, 1.0))
        position = self.start_position + alpha * (self.target_position - self.start_position)
        rotation_vector = self.rotation_slerp([alpha])[0].as_rotvec()
        return [*position.tolist(), *rotation_vector.tolist()]


class URControl:
    """Own RTDE control/receive interfaces and stream TCP targets.

    ``RTDEControlInterface`` is not thread-safe. All of its runtime control
    methods are therefore confined to the servo thread. Immediate commands use
    a capacity-one FIFO queue, while interpolated policy targets use a
    non-blocking latest-target slot.
    """

    def __init__(
        self,
        robot_ip: str,
        *,
        rtde_frequency_hz: float | None = None,
        tcp_pose: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        servo_lookahead_time: float = 0.1,
        servo_gain: float = 600.0,
        action_timeout_s: float = 0.25,
        command_timeout_s: float = 1.0,
        check_pose_safety: bool = True,
    ) -> None:
        self.robot_ip = robot_ip
        self.rtde_frequency_hz = rtde_frequency_hz
        self.tcp_pose = self._validate_pose(tcp_pose, name="TCP offset pose")
        self.servo_lookahead_time = servo_lookahead_time
        self.servo_gain = servo_gain
        self.action_timeout_s = action_timeout_s
        self.command_timeout_s = command_timeout_s
        self.check_pose_safety = check_pose_safety

        self._rtde_c = None
        self._rtde_r = None
        self._command_queue: queue.Queue[_PoseCommand | _MoveLCommand] = queue.Queue(maxsize=1)
        self._trajectory_lock = threading.Lock()
        self._pending_trajectory: _TrajectoryCommand | None = None
        self._stop_event = threading.Event()
        self._thread_started = threading.Event()
        self._control_thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._control_connected = False
        self._receive_connected = False
        self._thread_error: BaseException | None = None

    @property
    def is_connected(self) -> bool:
        thread = self._control_thread
        with self._state_lock:
            healthy = (
                self._control_connected
                and self._receive_connected
                and self._thread_error is None
            )
        return healthy and thread is not None and thread.is_alive()

    def connect(self) -> None:
        if self.is_connected:
            raise RuntimeError("URControl is already connected")

        self._reset_runtime_state()
        try:
            if self.rtde_frequency_hz is None:
                self._rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
                self._rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            else:
                self._rtde_r = rtde_receive.RTDEReceiveInterface(
                    self.robot_ip, self.rtde_frequency_hz
                )
                self._rtde_c = rtde_control.RTDEControlInterface(
                    self.robot_ip, self.rtde_frequency_hz
                )

            if not self._rtde_r.isConnected() or not self._rtde_c.isConnected():
                raise ConnectionError(f"Failed to connect RTDE interfaces to {self.robot_ip}")
            if not self._rtde_c.setTcp(self.tcp_pose):
                raise RuntimeError("Failed to set UR TCP offset pose")

            with self._state_lock:
                self._receive_connected = True
                self._control_connected = True

            self._control_thread = threading.Thread(
                target=self._servo_loop,
                name=f"ur-servo-{self.robot_ip}",
                daemon=True,
            )
            self._control_thread.start()
            if not self._thread_started.wait(timeout=self.command_timeout_s):
                raise TimeoutError("UR servo thread did not start in time")
            self._raise_if_thread_failed()
        except BaseException:
            try:
                self.disconnect()
            except BaseException:
                logger.exception("Failed to clean up a partial URControl connection")
            raise

    def get_tcp_pose(self) -> list[float]:
        self._require_connected()
        pose = self._rtde_r.getActualTCPPose()
        return self._validate_pose(pose, name="actual TCP pose")

    def set_tcp_pose(self, pose: Sequence[float]) -> list[float]:
        self._require_connected()
        target = self._validate_pose(pose, name="target TCP pose")
        command = _PoseCommand(target)

        try:
            self._command_queue.put(command, timeout=self.command_timeout_s)
        except queue.Full as exc:
            self._raise_if_thread_failed()
            raise TimeoutError("Timed out while queueing a UR TCP pose command") from exc

        if not command.done.wait(timeout=self.command_timeout_s):
            command.cancelled.set()
            self._raise_if_thread_failed()
            raise TimeoutError("Timed out waiting for UR TCP pose command acknowledgement")
        if command.error is not None:
            raise RuntimeError("Failed to send UR TCP pose command") from command.error
        self._raise_if_thread_failed()
        return target

    def submit_tcp_pose_trajectory(
        self,
        start_pose: Sequence[float],
        target_pose: Sequence[float],
        *,
        duration_s: float,
    ) -> list[float]:
        """Publish the latest time-interpolated pose target without blocking."""

        self._require_connected()
        start = self._validate_pose(start_pose, name="trajectory start TCP pose")
        target = self._validate_pose(target_pose, name="trajectory target TCP pose")
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("Trajectory duration must be finite and positive")
        with self._trajectory_lock:
            self._pending_trajectory = _TrajectoryCommand(start, target, float(duration_s))
        return target

    def move_tcp_pose(self, pose: Sequence[float], *, speed: float = 0.25, acceleration: float = 0.5) -> list[float]:
        """Move linearly to one TCP pose with RTDE moveL."""

        self._require_connected()
        target = self._validate_pose(pose, name="moveL target TCP pose")
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("moveL speed must be finite and positive")
        if not math.isfinite(acceleration) or acceleration <= 0:
            raise ValueError("moveL acceleration must be finite and positive")
        command = _MoveLCommand(target, speed=float(speed), acceleration=float(acceleration))

        try:
            self._command_queue.put(command, timeout=self.command_timeout_s)
        except queue.Full as exc:
            self._raise_if_thread_failed()
            raise TimeoutError("Timed out while queueing a UR moveL command") from exc

        while not command.done.wait(timeout=self.command_timeout_s):
            self._raise_if_thread_failed()
        if command.error is not None:
            raise RuntimeError("Failed to execute UR moveL command") from command.error
        self._raise_if_thread_failed()
        return target

    def disconnect(self) -> None:
        errors: list[BaseException] = []
        self._stop_event.set()

        thread = self._control_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(2.0, self.command_timeout_s))
            if thread.is_alive():
                errors.append(RuntimeError("UR servo thread did not stop in time"))

        # The control thread has exclusive access while running. Only clean up
        # the control interface here after it has stopped.
        if thread is None or not thread.is_alive():
            if self._rtde_c is not None:
                try:
                    self._rtde_c.stopScript()
                except BaseException as exc:
                    errors.append(exc)
                try:
                    self._rtde_c.disconnect()
                except BaseException as exc:
                    errors.append(exc)

        if self._rtde_r is not None:
            try:
                self._rtde_r.disconnect()
            except BaseException as exc:
                errors.append(exc)

        with self._state_lock:
            self._control_connected = False
            self._receive_connected = False
        self._control_thread = None
        self._rtde_c = None
        self._rtde_r = None
        self._drain_pending(RuntimeError("URControl disconnected"))

        if errors:
            raise RuntimeError("Errors occurred while disconnecting URControl") from errors[0]

    def _servo_loop(self) -> None:
        active_pose: list[float] | None = None
        active_trajectory: _PoseTrajectory | None = None
        last_command_time: float | None = None
        servo_active = False
        current_command: _PoseCommand | None = None

        try:
            step_time = self._wait_for_step_time()      # control frequency loop time
            self._thread_started.set()

            while not self._stop_event.is_set():
                try:
                    current_command = self._command_queue.get_nowait()
                except queue.Empty:
                    current_command = None

                if isinstance(current_command, _MoveLCommand):
                    if servo_active:
                        self._rtde_c.servoStop()
                    servo_active = False
                    active_pose = None
                    active_trajectory = None
                    last_command_time = None
                    if (
                        self.check_pose_safety
                        and not self._rtde_c.isPoseWithinSafetyLimits(current_command.pose)
                    ):
                        current_command.error = ValueError(
                            "moveL target TCP pose is unreachable or outside UR safety limits"
                        )
                    elif not self._rtde_c.moveL(
                        current_command.pose,
                        current_command.speed,
                        current_command.acceleration,
                    ):
                        current_command.error = RuntimeError("moveL returned False")
                    current_command.done.set()
                    current_command = None
                    continue

                trajectory_command = None
                if current_command is None:
                    with self._trajectory_lock:
                        trajectory_command = self._pending_trajectory
                        self._pending_trajectory = None

                cycle_start = self._rtde_c.initPeriod()
                current_time = time.monotonic()
                command_accepted = False
                if current_command is not None:
                    if current_command.cancelled.is_set():
                        current_command.error = TimeoutError("TCP pose command was cancelled")
                        current_command.done.set()
                    elif (
                        self.check_pose_safety
                        and not self._rtde_c.isPoseWithinSafetyLimits(
                            current_command.pose
                        )
                    ):
                        current_command.error = ValueError(
                            "Target TCP pose is unreachable or outside UR safety limits"
                        )
                        current_command.done.set()
                    else:
                        active_pose = current_command.pose
                        active_trajectory = None
                        last_command_time = current_time
                        command_accepted = True

                if trajectory_command is not None:
                    if (
                        self.check_pose_safety
                        and not self._rtde_c.isPoseWithinSafetyLimits(
                            trajectory_command.target_pose
                        )
                    ):
                        raise ValueError(
                            "Trajectory target TCP pose is unreachable or outside UR safety limits"
                        )
                    if active_trajectory is not None:
                        trajectory_start_pose = active_trajectory.evaluate(current_time)
                    elif active_pose is not None:
                        trajectory_start_pose = active_pose
                    else:
                        trajectory_start_pose = trajectory_command.start_pose
                    active_trajectory = _PoseTrajectory.create(
                        trajectory_start_pose,
                        trajectory_command.target_pose,
                        start_time=current_time,
                        duration_s=trajectory_command.duration_s,
                    )
                    active_pose = trajectory_start_pose
                    last_command_time = current_time

                if active_pose is not None and last_command_time is not None:
                    if time.monotonic() - last_command_time > self.action_timeout_s:
                        if servo_active:
                            self._rtde_c.servoStop()
                        servo_active = False
                        active_pose = None
                        active_trajectory = None
                        last_command_time = None
                    else:
                        if active_trajectory is not None:
                            active_pose = active_trajectory.evaluate(current_time)
                        ok = self._rtde_c.servoL(
                            active_pose,
                            0.5,  # speed and acceleration are unused by servoL
                            0.5,
                            step_time,
                            self.servo_lookahead_time,
                            self.servo_gain,
                        )
                        if not ok:
                            raise RuntimeError("servoL returned False")
                        servo_active = True
                        if command_accepted and current_command is not None:
                            current_command.done.set()

                self._rtde_c.waitPeriod(cycle_start)
        except BaseException as exc:
            if current_command is not None and not current_command.done.is_set():
                current_command.error = exc
                current_command.done.set()
            with self._state_lock:
                self._thread_error = exc
                self._control_connected = False
            self._drain_pending(exc)
            logger.exception("UR servo loop failed")
        finally:
            self._thread_started.set()
            if servo_active:
                try:
                    self._rtde_c.servoStop()
                except BaseException:
                    logger.exception("Failed to stop UR servo mode")

    def _wait_for_step_time(self) -> float:
        """Wait until the uploaded control script reports its native cycle time.

        Immediately after ``RTDEControlInterface`` uploads its script,
        ``getStepTime()`` can transiently return zero while the script is still
        starting. Keep querying within the normal command timeout instead of
        hard-coding the e-Series/CB-Series controller frequency.
        """

        deadline = time.monotonic() + self.command_timeout_s
        last_step_time = 0.0
        while not self._stop_event.is_set():
            last_step_time = float(self._rtde_c.getStepTime())
            if math.isfinite(last_step_time) and last_step_time > 0:
                return last_step_time
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        raise RuntimeError(f"Invalid RTDE control step time: {last_step_time}")

    def _require_connected(self) -> None:
        self._raise_if_thread_failed()
        if not self.is_connected:
            raise ConnectionError("URControl is not connected")

    def _raise_if_thread_failed(self) -> None:
        with self._state_lock:
            error = self._thread_error
        if error is not None:
            raise RuntimeError("UR servo thread is not healthy") from error

    def _reset_runtime_state(self) -> None:
        self._stop_event.clear()
        self._thread_started.clear()
        with self._trajectory_lock:
            self._pending_trajectory = None
        with self._state_lock:
            self._control_connected = False
            self._receive_connected = False
            self._thread_error = None
        self._drain_pending(RuntimeError("Discarding stale UR TCP pose command"))

    def _drain_pending(self, error: BaseException) -> None:
        while True:
            try:
                command = self._command_queue.get_nowait()
            except queue.Empty:
                break
            command.error = error
            command.done.set()

    @staticmethod
    def _validate_pose(pose: Sequence[float], *, name: str) -> list[float]:
        if len(pose) != 6:
            raise ValueError(f"{name} must contain 6 values, got {len(pose)}")
        values = [float(value) for value in pose]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{name} must contain only finite values")
        return values
