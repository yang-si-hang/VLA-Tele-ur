"""High-frequency Robotiq 2F control over host USB-RS485 and Modbus RTU.

Initialization uses FC03 to read state and FC16 to reset or activate the
gripper. Runtime control keeps rACT=1 and rGTO=1, then uses one FC23 transaction
per cycle to perform both operations below:

* Write ``00, position, speed, force`` to two registers from 0x03E9.
* Read three registers from 0x07D0 to obtain gACT, gGTO, gSTA, gOBJ, gFLT,
  gPR, gPO, and gCU.

A dedicated fixed-frequency worker is the only thread that accesses the serial
port. ``set_target()`` only replaces the latest target in memory, so commands
never accumulate in a queue. The worker uses absolute monotonic deadlines and
publishes a cached state containing the command, gripper feedback, timestamps,
and RTT. Responses are checked for exact length, CRC, malformed data, timeout,
and Modbus exceptions; occasional errors are resynchronized and retried.

The tested FTDI adapter originally used a 16 ms Linux ``latency_timer``, which
limited measured FC23 throughput to about 62.5 Hz. Testing 100 Hz requires
setting this timer to 1 ms and rerunning the hardware benchmark.
"""

from __future__ import annotations

import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from enum import IntEnum

import serial


class GripperStatus(IntEnum):
    RESET = 0
    ACTIVATING = 1
    ACTIVE = 3


class ObjectStatus(IntEnum):
    MOVING = 0
    STOPPED_OUTER_OBJECT = 1
    STOPPED_INNER_OBJECT = 2
    AT_DEST = 3


class RobotiqCommunicationError(RuntimeError):
    """Base class for recoverable Modbus communication errors."""


class ResponseTimeoutError(RobotiqCommunicationError):
    pass


class CRCError(RobotiqCommunicationError):
    pass


class MalformedResponseError(RobotiqCommunicationError):
    pass


class ModbusExceptionError(RobotiqCommunicationError):
    def __init__(self, exception_code: int) -> None:
        self.exception_code = exception_code
        super().__init__(f"Robotiq Modbus exception code 0x{exception_code:02X}")


class SerialDisconnectedError(RobotiqCommunicationError):
    pass


@dataclass(frozen=True, slots=True)
class TargetCommand:
    position: int
    speed: int
    force: int


@dataclass(frozen=True, slots=True)
class GripperState:
    activated: bool = False
    go_to: bool = False
    gripper_status: GripperStatus = GripperStatus.RESET
    object_status: ObjectStatus = ObjectStatus.MOVING
    fault: int = 0
    requested_position: int = 0
    position: int = 0
    current: int = 0
    command_position: int = 0
    request_started_ns: int = 0
    response_received_ns: int = 0
    rtt_ns: int = 0
    sequence: int = 0


@dataclass(frozen=True, slots=True)
class WorkerStats:
    cycles_attempted: int = 0
    successful_cycles: int = 0
    timeout_errors: int = 0
    crc_errors: int = 0
    malformed_frame_errors: int = 0
    modbus_exception_errors: int = 0
    serial_errors: int = 0
    deadline_misses: int = 0
    consecutive_errors: int = 0

    @property
    def communication_errors(self) -> int:
        return self.timeout_errors + self.crc_errors + self.malformed_frame_errors + self.modbus_exception_errors + self.serial_errors


class RobotiqGripperUSBHighFrequency:
    """Fixed-rate FC23 controller with latest-command-wins semantics."""

    INPUT_REGISTER = 0x07D0
    OUTPUT_REGISTER = 0x03E8
    RUNTIME_OUTPUT_REGISTER = 0x03E9

    FC_READ_HOLDING = 0x03
    FC_WRITE_MULTIPLE = 0x10
    FC_READ_WRITE_MULTIPLE = 0x17

    def __init__(
        self,
        port: str,
        *,
        slave_id: int = 9,
        baudrate: int = 115200,
        control_frequency_hz: float = 100.0,
        serial_timeout_s: float = 0.03,
        activation_timeout_s: float = 10.0,
        min_request_interval_s: float = 0.005,
        max_consecutive_errors: int = 3,
        history_size: int = 20_000,
    ) -> None:
        if not port:
            raise ValueError("port must not be empty")
        if not 1 <= slave_id <= 247:
            raise ValueError("slave_id must be in [1, 247]")
        for value, name in (
            (baudrate, "baudrate"),
            (control_frequency_hz, "control_frequency_hz"),
            (serial_timeout_s, "serial_timeout_s"),
            (activation_timeout_s, "activation_timeout_s"),
            (min_request_interval_s, "min_request_interval_s"),
            (max_consecutive_errors, "max_consecutive_errors"),
            (history_size, "history_size"),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        self.port = port
        self.slave_id = int(slave_id)
        self.baudrate = int(baudrate)
        self.control_frequency_hz = float(control_frequency_hz)
        self.serial_timeout_s = float(serial_timeout_s)
        self.activation_timeout_s = float(activation_timeout_s)
        self.min_request_interval_ns = int(float(min_request_interval_s) * 1e9)
        self.max_consecutive_errors = int(max_consecutive_errors)

        self._serial: serial.Serial | None = None
        self._worker: threading.Thread | None = None
        self._worker_ready = threading.Event()
        self._stop_event = threading.Event()
        self._target_lock = threading.Lock()
        self._target = TargetCommand(0, 255, 100)
        self._target_was_set = False
        self._state_condition = threading.Condition()
        self._state = GripperState()
        self._history: deque[GripperState] = deque(maxlen=int(history_size))
        self._stats_lock = threading.Lock()
        self._stats = WorkerStats()
        self._worker_error: BaseException | None = None
        self._last_request_started_ns: int | None = None

    @property
    def is_connected(self) -> bool:
        ser = self._serial
        worker = self._worker
        return bool(ser is not None and ser.is_open and worker is not None and worker.is_alive() and self._worker_ready.is_set() and self._worker_error is None)

    def connect(self, *, auto_activate: bool = True) -> None:
        """Open the port and let the dedicated worker initialize all I/O."""

        if self._serial is not None:
            raise RuntimeError("Robotiq gripper is already connected")
        self._reset_runtime_state()
        ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.serial_timeout_s,
            write_timeout=self.serial_timeout_s,
        )
        self._serial = ser
        self._worker = threading.Thread(target=self._worker_loop, args=(auto_activate,), name=f"robotiq-fc23-{self.port}", daemon=True)
        self._worker.start()
        ready_timeout = self.activation_timeout_s + 2 * self.serial_timeout_s + 1.0
        if not self._worker_ready.wait(timeout=ready_timeout):
            self.disconnect()
            raise TimeoutError("Robotiq FC23 worker did not initialize in time")
        self._raise_if_failed()

    def disconnect(self) -> None:
        """Stop the worker and close the serial port."""

        self._stop_event.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=self.serial_timeout_s + 1.0)
        ser = self._serial
        if worker is not None and worker.is_alive() and ser is not None:
            ser.close()
            worker.join(timeout=self.serial_timeout_s + 1.0)
        worker_alive = worker is not None and worker.is_alive()
        if ser is not None and ser.is_open:
            ser.close()
        self._worker = None
        self._serial = None
        if worker_alive:
            raise RuntimeError("Robotiq FC23 worker did not stop in time")

    def set_target(self, position: int, speed: int = 255, force: int = 100) -> None:
        """Atomically replace the latest target without touching the serial port."""

        target = TargetCommand(self._clip_u8(position), self._clip_u8(speed), self._clip_u8(force))
        with self._target_lock:
            self._target = target
            self._target_was_set = True

    def get_target(self) -> TargetCommand:
        with self._target_lock:
            return self._target

    def get_state(self) -> GripperState:
        with self._state_condition:
            return self._state

    def wait_for_state(self, after_sequence: int, timeout_s: float = 1.0) -> GripperState:
        deadline = time.monotonic() + timeout_s
        with self._state_condition:
            while self._state.sequence <= after_sequence:
                self._raise_if_failed()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for a fresh Robotiq state")
                self._state_condition.wait(timeout=remaining)
            return self._state

    def get_history(self, after_sequence: int = 0) -> list[GripperState]:
        with self._state_condition:
            return [state for state in self._history if state.sequence > after_sequence]

    def get_stats(self) -> WorkerStats:
        with self._stats_lock:
            return self._stats

    def get_last_error(self) -> BaseException | None:
        return self._worker_error

    def _worker_loop(self, auto_activate: bool) -> None:
        try:
            self._initialize_io(auto_activate)
            self._run_fixed_rate_loop()
        except Exception as exc:  # noqa: BLE001 - worker failures must remain observable.
            if not self._stop_event.is_set():
                self._worker_error = exc
        finally:
            self._worker_ready.set()
            with self._state_condition:
                self._state_condition.notify_all()

    def _initialize_io(self, auto_activate: bool) -> None:
        ser = self._require_serial()
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        state = self._read_state_io()
        self._publish_state(replace(state, sequence=0))
        if auto_activate:
            self._activate_io()
            state = self._read_state_io()
            self._publish_state(replace(state, sequence=0))
        if state.gripper_status != GripperStatus.ACTIVE:
            raise RuntimeError("Robotiq gripper is not active")

        with self._target_lock:
            if not self._target_was_set:
                self._target = TargetCommand(state.position, 255, 100)
            target = self._target
        self._write_control_io(0x09, target.position, target.speed, target.force)

    def _run_fixed_rate_loop(self) -> None:
        period_ns = int(1e9 / self.control_frequency_hz)
        next_deadline_ns = time.monotonic_ns()
        consecutive_errors = 0

        while not self._stop_event.is_set():
            with self._target_lock:
                target = self._target
            self._increment_stats(cycles_attempted=1)
            try:
                state = self._fc23_exchange_io(target.position, target.speed, target.force)
            except RobotiqCommunicationError as exc:
                consecutive_errors += 1
                self._record_communication_error(exc, consecutive_errors)
                self._resynchronize_io()
                if consecutive_errors >= self.max_consecutive_errors:
                    raise RuntimeError(f"Robotiq FC23 failed {consecutive_errors} consecutive times") from exc
            else:
                consecutive_errors = 0
                with self._stats_lock:
                    sequence = self._stats.successful_cycles + 1
                    self._stats = replace(self._stats, successful_cycles=sequence, consecutive_errors=0)
                self._publish_state(replace(state, sequence=sequence))
                self._worker_ready.set()

            next_deadline_ns += period_ns
            now_ns = time.monotonic_ns()
            if now_ns > next_deadline_ns:
                self._increment_stats(deadline_misses=1)
            remaining_ns = next_deadline_ns - time.monotonic_ns()
            if remaining_ns > 0:
                self._stop_event.wait(remaining_ns / 1e9)

    def _fc23_exchange_io(self, position: int, speed: int, force: int) -> GripperState:
        """Write 0x03E9..0x03EA and read 0x07D0..0x07D2 atomically."""

        position = self._clip_u8(position)
        speed = self._clip_u8(speed)
        force = self._clip_u8(force)
        write_data = bytes((0, position, speed, force))
        request = struct.pack(
            ">BBHHHHB",
            self.slave_id,
            self.FC_READ_WRITE_MULTIPLE,
            self.INPUT_REGISTER,
            3,
            self.RUNTIME_OUTPUT_REGISTER,
            2,
            len(write_data),
        ) + write_data
        response, request_started_ns, response_received_ns = self._transaction(request, expected_length=11)
        if response[:3] != bytes((self.slave_id, self.FC_READ_WRITE_MULTIPLE, 6)):
            raise MalformedResponseError(f"Unexpected FC23 response: {response.hex(' ')}")
        return self._parse_state(response[3:9], position, request_started_ns, response_received_ns)

    def _read_state_io(self) -> GripperState:
        request = struct.pack(">BBHH", self.slave_id, self.FC_READ_HOLDING, self.INPUT_REGISTER, 3)
        response, request_started_ns, response_received_ns = self._transaction(request, expected_length=11)
        if response[:3] != bytes((self.slave_id, self.FC_READ_HOLDING, 6)):
            raise MalformedResponseError(f"Unexpected FC03 response: {response.hex(' ')}")
        return self._parse_state(response[3:9], response[6], request_started_ns, response_received_ns)

    def _parse_state(self, data: bytes, command_position: int, request_started_ns: int, response_received_ns: int) -> GripperState:
        if len(data) != 6:
            raise MalformedResponseError(f"Expected 6 state bytes, received {len(data)}")
        status, fault, requested_position, position, current = data[0], data[2], data[3], data[4], data[5]
        try:
            gripper_status = GripperStatus((status >> 4) & 0x03)
            object_status = ObjectStatus((status >> 6) & 0x03)
        except ValueError as exc:
            raise MalformedResponseError(f"Invalid Robotiq status byte 0x{status:02X}") from exc
        return GripperState(
            activated=bool(status & 0x01),
            go_to=bool(status & 0x08),
            gripper_status=gripper_status,
            object_status=object_status,
            fault=fault,
            requested_position=requested_position,
            position=position,
            current=current,
            command_position=command_position,
            request_started_ns=request_started_ns,
            response_received_ns=response_received_ns,
            rtt_ns=response_received_ns - request_started_ns,
        )

    def _activate_io(self) -> None:
        state = self._read_state_io()
        if state.gripper_status == GripperStatus.ACTIVE and state.fault == 0:
            return
        self._write_control_io(0x00, 0, 0, 0)
        self._wait_for_activation_state(False)
        self._write_control_io(0x01, 0, 0, 0)
        self._wait_for_activation_state(True)

    def _wait_for_activation_state(self, active: bool) -> None:
        deadline = time.monotonic() + self.activation_timeout_s
        while True:
            state = self._read_state_io()
            self._publish_state(replace(state, sequence=0))
            if active and state.fault not in (0x00, 0x09):
                raise RuntimeError(f"Robotiq activation fault 0x{state.fault:02X}")
            if active and state.activated and state.gripper_status == GripperStatus.ACTIVE:
                return
            if not active and not state.activated and state.gripper_status == GripperStatus.RESET:
                return
            if time.monotonic() >= deadline:
                action = "activating" if active else "resetting"
                raise TimeoutError(f"Timed out {action} Robotiq gripper")

    def _write_control_io(self, action: int, position: int, speed: int, force: int) -> None:
        payload = bytes((action, 0, 0, position, speed, force))
        request = struct.pack(">BBHHB", self.slave_id, self.FC_WRITE_MULTIPLE, self.OUTPUT_REGISTER, 3, len(payload)) + payload
        response, _, _ = self._transaction(request, expected_length=8)
        expected = struct.pack(">BBHH", self.slave_id, self.FC_WRITE_MULTIPLE, self.OUTPUT_REGISTER, 3)
        if response[:6] != expected:
            raise MalformedResponseError(f"Unexpected FC16 response: {response.hex(' ')}")

    def _transaction(self, payload: bytes, *, expected_length: int) -> tuple[bytes, int, int]:
        ser = self._require_serial()
        self._wait_for_bus_slot()
        frame = self._append_crc(payload)
        request_started_ns = time.monotonic_ns()
        self._last_request_started_ns = request_started_ns
        try:
            written = ser.write(frame)
            if written != len(frame):
                raise ResponseTimeoutError(f"Incomplete request write: expected {len(frame)} bytes, wrote {written}")
            ser.flush()
            prefix = self._read_exact(2)
            if prefix[1] == (payload[1] | 0x80):
                response = prefix + self._read_exact(3)
                response_received_ns = time.monotonic_ns()
                if not self._check_crc(response):
                    raise CRCError(f"Invalid Modbus exception CRC: {response.hex(' ')}")
                if response[0] != self.slave_id:
                    raise MalformedResponseError(f"Unexpected slave ID in response: {response.hex(' ')}")
                raise ModbusExceptionError(response[2])
            response = prefix + self._read_exact(expected_length - 2)
        except serial.SerialException as exc:
            raise SerialDisconnectedError(f"Robotiq serial I/O failed: {exc}") from exc
        response_received_ns = time.monotonic_ns()
        if not self._check_crc(response):
            raise CRCError(f"Invalid Modbus CRC: {response.hex(' ')}")
        if response[0] != self.slave_id or response[1] != payload[1]:
            raise MalformedResponseError(f"Unexpected Modbus response: {response.hex(' ')}")
        return response, request_started_ns, response_received_ns

    def _read_exact(self, size: int) -> bytes:
        ser = self._require_serial()
        chunks = bytearray()
        while len(chunks) < size:
            try:
                chunk = ser.read(size - len(chunks))
            except serial.SerialException as exc:
                raise SerialDisconnectedError(f"Robotiq serial read failed: {exc}") from exc
            if not chunk:
                raise ResponseTimeoutError(f"Robotiq response timeout: expected {size} bytes, received {len(chunks)}")
            chunks.extend(chunk)
        return bytes(chunks)

    def _wait_for_bus_slot(self) -> None:
        previous = self._last_request_started_ns
        if previous is None:
            return
        remaining_ns = previous + self.min_request_interval_ns - time.monotonic_ns()
        if remaining_ns > 0:
            self._stop_event.wait(remaining_ns / 1e9)

    def _resynchronize_io(self) -> None:
        try:
            self._require_serial().reset_input_buffer()
        except serial.SerialException as exc:
            raise SerialDisconnectedError(f"Failed to resynchronize Robotiq serial input: {exc}") from exc

    def _record_communication_error(self, error: RobotiqCommunicationError, consecutive_errors: int) -> None:
        field = "malformed_frame_errors"
        if isinstance(error, ResponseTimeoutError):
            field = "timeout_errors"
        elif isinstance(error, CRCError):
            field = "crc_errors"
        elif isinstance(error, ModbusExceptionError):
            field = "modbus_exception_errors"
        elif isinstance(error, SerialDisconnectedError):
            field = "serial_errors"
        with self._stats_lock:
            self._stats = replace(self._stats, **{field: getattr(self._stats, field) + 1, "consecutive_errors": consecutive_errors})

    def _increment_stats(self, **increments: int) -> None:
        with self._stats_lock:
            changes = {name: getattr(self._stats, name) + increment for name, increment in increments.items()}
            self._stats = replace(self._stats, **changes)

    def _publish_state(self, state: GripperState) -> None:
        with self._state_condition:
            self._state = state
            if state.sequence > 0:
                self._history.append(state)
            self._state_condition.notify_all()

    def _require_serial(self) -> serial.Serial:
        ser = self._serial
        if ser is None or not ser.is_open:
            raise SerialDisconnectedError("Robotiq serial port is not connected")
        return ser

    def _raise_if_failed(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("Robotiq FC23 worker failed") from self._worker_error

    def _reset_runtime_state(self) -> None:
        self._stop_event.clear()
        self._worker_ready.clear()
        self._worker_error = None
        self._last_request_started_ns = None
        with self._state_condition:
            self._state = GripperState()
            self._history.clear()
        with self._stats_lock:
            self._stats = WorkerStats()

    @staticmethod
    def _clip_u8(value: int) -> int:
        return max(0, min(int(value), 255))

    @staticmethod
    def _crc16(data: bytes) -> int:
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
        return crc

    @classmethod
    def _append_crc(cls, data: bytes) -> bytes:
        crc = cls._crc16(data)
        return data + bytes((crc & 0xFF, crc >> 8))

    @classmethod
    def _check_crc(cls, frame: bytes) -> bool:
        return len(frame) >= 3 and cls._append_crc(frame[:-2]) == frame
