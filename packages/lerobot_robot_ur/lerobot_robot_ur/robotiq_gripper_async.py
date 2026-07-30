"""Provide threaded, non-blocking socket control for a Robotiq gripper.

The implementation keeps every Robotiq ASCII protocol exchange in one worker
thread because the request-response protocol has no request identifiers.
Runtime motion calls return ``Future`` objects, pending targets are coalesced,
and periodic polling maintains a lock-protected state cache for fast LeRobot
observations. Activation and optional automatic calibration remain blocking
setup operations; physical position changes are observed at roughly 6 Hz.
"""


from __future__ import annotations

import socket
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, TypeVar, cast


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class GripperState:
    """Last state observed by the socket worker."""

    position: int | None = None
    requested_position: int | None = None
    object_status: RobotiqGripper.ObjectStatus | None = None
    gripper_status: RobotiqGripper.GripperStatus | None = None
    fault: int | None = None
    updated_at: float | None = None
    position_updated_at: float | None = None
    status_updated_at: float | None = None


@dataclass(slots=True)
class _Command:
    action: Callable[[], object]
    future: Future[object]


class RobotiqGripper:
    """Control a Robotiq gripper through one dedicated socket worker.

    The Robotiq TCP protocol is request-response based and does not include a
    request identifier. Consequently, the worker is the only thread allowed to
    send to or receive from the socket.

    ``move_async`` coalesces commands that have not started yet: when several
    targets arrive faster than the socket can accept them, only the latest
    pending target is retained. Management commands and commands that wait for
    motion completion are never coalesced.
    """

    ACT = "ACT"
    GTO = "GTO"
    ATR = "ATR"
    ADR = "ADR"
    FOR = "FOR"
    SPE = "SPE"
    POS = "POS"

    STA = "STA"
    PRE = "PRE"
    OBJ = "OBJ"
    FLT = "FLT"

    ENCODING = "UTF-8"

    class GripperStatus(Enum):
        RESET = 0
        ACTIVATING = 1
        ACTIVE = 3

    class ObjectStatus(Enum):
        MOVING = 0
        STOPPED_OUTER_OBJECT = 1
        STOPPED_INNER_OBJECT = 2
        AT_DEST = 3

    def __init__(
        self,
        *,
        position_poll_frequency_hz: float = 60.0,
        status_poll_frequency_hz: float = 60.0,      # gripper state: 是否解除物体等
        activation_timeout_s: float = 10.0,
        motion_timeout_s: float = 10.0,
        sync_timeout_s: float = 60.0,
    ) -> None:
        self._validate_positive(
            position_poll_frequency_hz, "position_poll_frequency_hz"
        )
        self._validate_positive(status_poll_frequency_hz, "status_poll_frequency_hz")
        self._validate_positive(activation_timeout_s, "activation_timeout_s")
        self._validate_positive(motion_timeout_s, "motion_timeout_s")
        self._validate_positive(sync_timeout_s, "sync_timeout_s")

        self.position_poll_frequency_hz = float(position_poll_frequency_hz)
        self.status_poll_frequency_hz = float(status_poll_frequency_hz)
        self.activation_timeout_s = float(activation_timeout_s)
        self.motion_timeout_s = float(motion_timeout_s)
        self.sync_timeout_s = float(sync_timeout_s)

        self.socket: socket.socket | None = None
        self._socket_timeout_s = 2.0
        self._worker: threading.Thread | None = None
        self._worker_started = threading.Event()
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._commands: deque[_Command] = deque()
        self._pending_move: _Command | None = None

        self._state_lock = threading.Lock()
        self._state = GripperState()
        self._thread_error: BaseException | None = None

        self._range_lock = threading.Lock()
        self._min_position = 0
        self._max_position = 255
        self._min_speed = 0
        self._max_speed = 255
        self._min_force = 0
        self._max_force = 255

    @property
    def is_connected(self) -> bool:
        worker = self._worker
        sock = self.socket
        with self._state_lock:
            healthy = self._thread_error is None
        if worker is None or not worker.is_alive() or sock is None or not healthy:
            return False
        try:
            return sock.fileno() != -1
        except OSError:
            return False

    def connect(
        self, hostname: str, port: int, socket_timeout: float = 2.0
    ) -> None:
        """Connect the socket and start the I/O worker.

        Connection establishment is intentionally synchronous. Runtime gripper
        operations are available through the non-blocking ``*_async`` methods.
        """

        if self.is_connected:
            raise RuntimeError("Robotiq gripper is already connected")
        self._validate_positive(socket_timeout, "socket_timeout")

        self._reset_runtime_state()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(float(socket_timeout))
            sock.connect((hostname, port))
            self.socket = sock
            self._socket_timeout_s = float(socket_timeout)
            self._worker = threading.Thread(
                target=self._worker_loop,
                name=f"robotiq-gripper-{hostname}:{port}",
                daemon=True,
            )
            self._worker.start()
            if not self._worker_started.wait(timeout=socket_timeout):
                raise TimeoutError("Robotiq gripper worker did not start in time")
            self._raise_if_worker_failed()
        except BaseException:
            self._stop_event.set()
            with self._condition:
                self._condition.notify_all()
            try:
                sock.close()
            finally:
                self.socket = None
            raise

    def disconnect(self) -> None:
        """Stop the worker, fail pending work, and close the socket."""

        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()

        worker = self._worker
        sock = self.socket
        if worker is not None and worker.is_alive():
            worker.join(timeout=self._socket_timeout_s + 1.0)
            if worker.is_alive() and sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                worker.join(timeout=self._socket_timeout_s + 1.0)

        worker_alive = worker is not None and worker.is_alive()
        close_error: OSError | None = None
        if sock is not None:
            try:
                sock.close()
            except OSError as exc:
                close_error = exc

        self.socket = None
        self._worker = None
        self._fail_pending(RuntimeError("Robotiq gripper disconnected"))

        if worker_alive:
            raise RuntimeError("Robotiq gripper worker did not stop in time")
        if close_error is not None:
            raise RuntimeError(
                "Failed to close Robotiq gripper socket"
            ) from close_error

    def move_async(
        self, position: int, speed: int, force: int
    ) -> Future[tuple[bool, int]]:
        """Queue a move command and return immediately.

        The result contains the command acknowledgement and clipped requested
        position. It is not the measured physical position.
        """

        return self._submit(
            lambda: self._move_io(position, speed, force),
            coalesce_move=True,
        )

    def move_and_wait_for_pos_async(
        self, position: int, speed: int, force: int
    ) -> Future[tuple[int, RobotiqGripper.ObjectStatus]]:
        """Queue a move that completes after the gripper stops."""

        return self._submit(
            lambda: self._move_and_wait_for_pos_io(position, speed, force)
        )

    def get_current_position_async(self) -> Future[int]:
        """Queue a fresh ``GET POS`` request."""

        return self._submit(lambda: self._get_var_io(self.POS))

    def is_active_async(self) -> Future[bool]:
        """Queue a fresh activation-state query."""

        return self._submit(
            lambda: self.GripperStatus(self._get_var_io(self.STA))
            == self.GripperStatus.ACTIVE
        )

    def get_state(self) -> GripperState:
        """Return a snapshot of the cached state without socket I/O."""

        with self._state_lock:
            return self._state

    def get_cached_position(self, max_age_s: float | None = None) -> int | None:
        """Return the last polled position without blocking.

        ``None`` is returned before the first successful poll, or when
        ``max_age_s`` is supplied and the cached position is older than it.
        """

        if max_age_s is not None and max_age_s < 0:
            raise ValueError("max_age_s must be non-negative")
        state = self.get_state()
        if state.position is None or state.position_updated_at is None:
            return None
        if (
            max_age_s is not None
            and time.monotonic() - state.position_updated_at > max_age_s
        ):
            return None
        return state.position

    def get_last_error(self) -> BaseException | None:
        with self._state_lock:
            return self._thread_error

    # Activation and calibration deliberately block because control must not
    # continue before these prerequisites have completed.

    def activate(self, auto_calibrate: bool = True) -> None:
        self._submit(lambda: self._activate_io(auto_calibrate)).result(
            timeout=self.sync_timeout_s
        )

    def is_active(self) -> bool:
        return self.is_active_async().result(timeout=self.sync_timeout_s)

    def get_min_position(self) -> int:
        with self._range_lock:
            return self._min_position

    def get_max_position(self) -> int:
        with self._range_lock:
            return self._max_position

    def get_open_position(self) -> int:
        return self.get_min_position()

    def get_closed_position(self) -> int:
        return self.get_max_position()

    def is_open(self) -> bool:
        return self.get_current_position() <= self.get_open_position()

    def is_closed(self) -> bool:
        return self.get_current_position() >= self.get_closed_position()

    def get_current_position(self) -> int:
        return self.get_current_position_async().result(timeout=self.sync_timeout_s)

    def auto_calibrate(self, log: bool = True) -> None:
        self._submit(lambda: self._auto_calibrate_io(log)).result(
            timeout=self.sync_timeout_s
        )

    def move(self, position: int, speed: int, force: int) -> tuple[bool, int]:
        return self.move_async(position, speed, force).result(
            timeout=self.sync_timeout_s
        )

    def move_and_wait_for_pos(
        self, position: int, speed: int, force: int
    ) -> tuple[int, RobotiqGripper.ObjectStatus]:
        return self.move_and_wait_for_pos_async(position, speed, force).result(
            timeout=self.sync_timeout_s
        )

    def _worker_loop(self) -> None:
        position_interval = 1.0 / self.position_poll_frequency_hz
        status_interval = 1.0 / self.status_poll_frequency_hz
        now = time.monotonic()
        next_position_poll = now
        next_status_poll = now

        self._worker_started.set()
        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                if now >= next_position_poll:
                    self._get_var_io(self.POS)
                    next_position_poll = self._next_deadline(
                        next_position_poll, position_interval
                    )

                now = time.monotonic()
                if now >= next_status_poll:
                    self._get_var_io(self.OBJ)
                    self._get_var_io(self.STA)
                    self._get_var_io(self.FLT)
                    next_status_poll = self._next_deadline(
                        next_status_poll, status_interval
                    )

                timeout = max(
                    0.0,
                    min(next_position_poll, next_status_poll) - time.monotonic(),
                )
                command = self._take_command(timeout)
                if command is not None:
                    self._run_command(command)
        except BaseException as exc:
            with self._state_lock:
                self._thread_error = exc
        finally:
            self._worker_started.set()
            self._fail_pending(
                RuntimeError("Robotiq gripper worker stopped")
            )

    def _take_command(self, timeout: float) -> _Command | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._stop_event.is_set():
                if self._commands:
                    return self._commands.popleft()
                if self._pending_move is not None:
                    command = self._pending_move
                    self._pending_move = None
                    return command

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
        return None

    def _run_command(self, command: _Command) -> None:
        if not command.future.set_running_or_notify_cancel():
            return
        try:
            result = command.action()
        except BaseException as exc:
            command.future.set_exception(exc)
            if isinstance(exc, (OSError, ValueError)):
                raise
        else:
            command.future.set_result(result)

    def _submit(
        self,
        action: Callable[[], T],
        *,
        coalesce_move: bool = False,
    ) -> Future[T]:
        self._require_connected()
        future: Future[object] = Future()
        command = _Command(cast(Callable[[], object], action), future)

        with self._condition:
            self._raise_if_worker_failed()
            if self._stop_event.is_set():
                raise RuntimeError("Robotiq gripper is disconnecting")
            if coalesce_move:
                previous = self._pending_move
                self._pending_move = command
                if previous is not None:
                    previous.future.cancel()
            else:
                self._commands.append(command)
            self._condition.notify()
        return cast(Future[T], future)

    def _move_io(
        self, position: int, speed: int, force: int
    ) -> tuple[bool, int]:
        with self._range_lock:
            clip_pos = self._clip(self._min_position, position, self._max_position)
            clip_spe = self._clip(self._min_speed, speed, self._max_speed)
            clip_for = self._clip(self._min_force, force, self._max_force)

        values = OrderedDict(
            (
                (self.POS, clip_pos),
                (self.SPE, clip_spe),
                (self.FOR, clip_for),
                (self.GTO, 1),
            )
        )
        acknowledged = self._set_vars_io(values)
        if acknowledged:
            self._update_state(requested_position=clip_pos)
        return acknowledged, clip_pos

    def _move_and_wait_for_pos_io(
        self, position: int, speed: int, force: int
    ) -> tuple[int, RobotiqGripper.ObjectStatus]:
        set_ok, command_position = self._move_io(position, speed, force)
        if not set_ok:
            raise RuntimeError("Failed to set variables for move")

        self._wait_until_io(
            lambda: self._get_var_io(self.PRE) == command_position,
            timeout_s=self.motion_timeout_s,
            message="Timed out waiting for the requested gripper position",
        )

        object_status = self.ObjectStatus(self._get_var_io(self.OBJ))
        deadline = time.monotonic() + self.motion_timeout_s
        position_interval = 1.0 / self.position_poll_frequency_hz
        next_position_poll = time.monotonic()
        while object_status == self.ObjectStatus.MOVING:
            self._raise_if_stopping()
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError("Timed out waiting for gripper motion to finish")
            if now >= next_position_poll:
                self._get_var_io(self.POS)
                next_position_poll = self._next_deadline(
                    next_position_poll, position_interval
                )
            self._stop_event.wait(
                min(0.01, max(0.0, next_position_poll - time.monotonic()))
            )
            object_status = self.ObjectStatus(self._get_var_io(self.OBJ))

        final_position = self._get_var_io(self.POS)
        return final_position, object_status

    def _reset_io(self) -> None:
        self._set_var_io(self.ACT, 0)
        self._set_var_io(self.ATR, 0)

        deadline = time.monotonic() + self.activation_timeout_s
        while True:
            self._raise_if_stopping()
            act = self._get_var_io(self.ACT)
            status = self._get_var_io(self.STA)
            if act == 0 and status == self.GripperStatus.RESET.value:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out resetting Robotiq gripper")
            self._set_var_io(self.ACT, 0)
            self._set_var_io(self.ATR, 0)
            self._stop_event.wait(0.01)
        self._stop_event.wait(0.5)
        self._raise_if_stopping()

    def _activate_io(self, auto_calibrate: bool) -> None:
        if self.GripperStatus(self._get_var_io(self.STA)) != self.GripperStatus.ACTIVE:
            self._reset_io()
            self._set_var_io(self.ACT, 1)
            self._wait_until_io(
                lambda: (
                    self._get_var_io(self.ACT) == 1
                    and self._get_var_io(self.STA)
                    == self.GripperStatus.ACTIVE.value
                ),
                timeout_s=self.activation_timeout_s,
                message="Timed out activating Robotiq gripper",
            )

        if auto_calibrate:
            self._auto_calibrate_io(log=True)

    def _auto_calibrate_io(self, log: bool) -> None:
        position, status = self._move_and_wait_for_pos_io(
            self.get_open_position(), 64, 1
        )
        if status != self.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed opening to start: {status}")

        position, status = self._move_and_wait_for_pos_io(
            self.get_closed_position(), 64, 1
        )
        if status != self.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed because of an object: {status}")
        with self._range_lock:
            if position > self._max_position:
                raise RuntimeError("Calibrated maximum exceeds configured range")
            self._max_position = position

        position, status = self._move_and_wait_for_pos_io(
            self.get_open_position(), 64, 1
        )
        if status != self.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed because of an object: {status}")
        with self._range_lock:
            if position < self._min_position:
                raise RuntimeError("Calibrated minimum is below configured range")
            self._min_position = position

        if log:
            print(
                "Gripper auto-calibrated to "
                f"[{self.get_min_position()}, {self.get_max_position()}]"
            )

    def _set_vars_io(self, values: OrderedDict[str, int | float]) -> bool:
        sock = self._require_socket()
        command = "SET"
        for variable, value in values.items():
            command += f" {variable} {value}"
        command += "\n"
        sock.sendall(command.encode(self.ENCODING))
        return self._is_ack(sock.recv(1024))

    def _set_var_io(self, variable: str, value: int | float) -> bool:
        return self._set_vars_io(OrderedDict(((variable, value),)))

    def _get_var_io(self, variable: str) -> int:
        sock = self._require_socket()
        command = f"GET {variable}\n"
        sock.sendall(command.encode(self.ENCODING))
        data = sock.recv(1024)

        try:
            variable_name, value_text = data.decode(self.ENCODING).split()
        except ValueError as exc:
            raise ValueError(
                f"Malformed response from Robotiq gripper: {data!r}"
            ) from exc
        if variable_name != variable:
            raise ValueError(
                f"Unexpected response {data!r}: expected variable {variable!r}"
            )
        value = int(value_text)
        self._cache_variable(variable, value)
        return value

    def _cache_variable(self, variable: str, value: int) -> None:
        now = time.monotonic()
        changes: dict[str, object] = {"updated_at": now}
        if variable == self.POS:
            changes["position"] = value
            changes["position_updated_at"] = now
        elif variable == self.PRE:
            changes["requested_position"] = value
        elif variable == self.OBJ:
            changes["object_status"] = self.ObjectStatus(value)
        elif variable == self.STA:
            changes["gripper_status"] = self.GripperStatus(value)
        elif variable == self.FLT:
            changes["fault"] = value
            # FLT is the final query in each OBJ/STA/FLT status poll cycle.
            changes["status_updated_at"] = now
        self._update_state(**changes)

    def _update_state(self, **changes: object) -> None:
        with self._state_lock:
            self._state = replace(self._state, **changes)

    def _wait_until_io(
        self,
        predicate: Callable[[], bool],
        *,
        timeout_s: float,
        message: str,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            self._raise_if_stopping()
            if predicate():
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(message)
            self._stop_event.wait(0.01)

    def _require_connected(self) -> None:
        if not self.is_connected:
            self._raise_if_worker_failed()
            raise RuntimeError("Robotiq gripper is not connected")

    def _require_socket(self) -> socket.socket:
        sock = self.socket
        if sock is None:
            raise RuntimeError("Robotiq gripper socket is not connected")
        return sock

    def _raise_if_stopping(self) -> None:
        if self._stop_event.is_set():
            raise RuntimeError("Robotiq gripper is disconnecting")

    def _raise_if_worker_failed(self) -> None:
        with self._state_lock:
            error = self._thread_error
        if error is not None:
            raise RuntimeError("Robotiq gripper worker failed") from error

    def _reset_runtime_state(self) -> None:
        self._stop_event.clear()
        self._worker_started.clear()
        with self._condition:
            self._commands.clear()
            self._pending_move = None
        with self._state_lock:
            self._state = GripperState()
            self._thread_error = None

    def _fail_pending(self, error: BaseException) -> None:
        with self._condition:
            pending = list(self._commands)
            self._commands.clear()
            if self._pending_move is not None:
                pending.append(self._pending_move)
                self._pending_move = None
        for command in pending:
            if not command.future.done():
                command.future.set_exception(error)

    @staticmethod
    def _is_ack(data: bytes) -> bool:
        return data == b"ack"

    @staticmethod
    def _clip(minimum: int, value: int, maximum: int) -> int:
        return max(minimum, min(value, maximum))

    @staticmethod
    def _next_deadline(deadline: float, interval: float) -> float:
        now = time.monotonic()
        while deadline <= now:
            deadline += interval
        return deadline

    @staticmethod
    def _validate_positive(value: float, name: str) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be positive")
