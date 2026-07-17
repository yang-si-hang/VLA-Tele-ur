"""Low-level TCP pose control for Universal Robots through ``ur_rtde``."""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Sequence

import rtde_control
import rtde_receive


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _PoseCommand:
    pose: list[float]
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


class URControl:
    """Own RTDE control/receive interfaces and stream the latest TCP target.

    ``RTDEControlInterface`` is not thread-safe. All of its runtime control
    methods are therefore confined to the servo thread. Public callers submit
    commands through a queue and wait until the first ``servoL`` call has been
    acknowledged.
    """

    def __init__(
        self,
        robot_ip: str,
        *,
        rtde_frequency_hz: float | None = None,
        servo_lookahead_time: float = 0.1,
        servo_gain: float = 600.0,
        action_timeout_s: float = 0.25,
        command_timeout_s: float = 1.0,
    ) -> None:
        self.robot_ip = robot_ip
        self.rtde_frequency_hz = rtde_frequency_hz
        self.servo_lookahead_time = servo_lookahead_time
        self.servo_gain = servo_gain
        self.action_timeout_s = action_timeout_s
        self.command_timeout_s = command_timeout_s

        self._rtde_c = None
        self._rtde_r = None
        self._command_queue: queue.Queue[_PoseCommand] = queue.Queue(maxsize=1)
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
        last_command_time: float | None = None
        servo_active = False
        current_command: _PoseCommand | None = None

        try:
            step_time = self._wait_for_step_time()
            self._thread_started.set()

            while not self._stop_event.is_set():
                cycle_start = self._rtde_c.initPeriod()
                try:
                    current_command = self._command_queue.get_nowait()
                except queue.Empty:
                    current_command = None

                command_accepted = False
                if current_command is not None:
                    if current_command.cancelled.is_set():
                        current_command.error = TimeoutError("TCP pose command was cancelled")
                        current_command.done.set()
                    elif not self._rtde_c.isPoseWithinSafetyLimits(current_command.pose):
                        current_command.error = ValueError(
                            "Target TCP pose is unreachable or outside UR safety limits"
                        )
                        current_command.done.set()
                    else:
                        active_pose = current_command.pose
                        last_command_time = time.monotonic()
                        command_accepted = True

                if active_pose is not None and last_command_time is not None:
                    if time.monotonic() - last_command_time > self.action_timeout_s:
                        if servo_active:
                            self._rtde_c.servoStop()
                        servo_active = False
                        active_pose = None
                        last_command_time = None
                    else:
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
