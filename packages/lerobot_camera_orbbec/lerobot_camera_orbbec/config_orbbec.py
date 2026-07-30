"""Define and validate configuration for the LeRobot Orbbec camera driver.

``OrbbecCameraConfig`` is registered under the ``orbbec`` camera type and
describes device selection, the exact RGB stream profile, output rotation,
warmup time, and optional color controls. Validation normalizes enum values and
rejects inconsistent manual and automatic camera settings before the SDK stream
is opened.
"""

from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("orbbec")
@dataclass(kw_only=True)
class OrbbecCameraConfig(CameraConfig):
    """Configuration for an Orbbec RGB camera.

    ``width`` and ``height`` describe the image returned to LeRobot. For a 90°
    or 270° rotation, the SDK capture dimensions are swapped automatically.

    Color controls are optional so existing configurations keep using the
    device defaults. Manual exposure and white balance values must be paired
    with their corresponding automatic control explicitly disabled.

    The default profile (1280x720 at 30 FPS) is supported by both Gemini 305
    and Gemini 336. Stream profiles are matched exactly; unsupported settings
    fail during :meth:`OrbbecCamera.connect` instead of silently falling back.
    """

    serial_number_or_name: str
    fps: int = 30
    width: int = 1280
    height: int = 720
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: float = 1.0

    auto_exposure: bool | None = False
    exposure: int | None = None
    gain: int | None = None
    auto_white_balance: bool | None = False
    white_balance: int | None = None
    auto_exposure_priority: int | None = 0
    anti_flicker: bool | None = False
    power_line_frequency: int | None = 1
    backlight_compensation: int | None = 0

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        if not isinstance(self.serial_number_or_name, str) or not self.serial_number_or_name.strip():
            raise ValueError("`serial_number_or_name` must be a non-empty string.")

        if isinstance(self.fps, bool) or not isinstance(self.fps, int) or self.fps <= 0:
            raise ValueError("`fps` must be a positive integer.")
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("`width` must be a positive integer.")
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height <= 0:
            raise ValueError("`height` must be a positive integer.")
        if isinstance(self.warmup_s, bool) or not isinstance(self.warmup_s, (int, float)):
            raise ValueError("`warmup_s` must be a non-negative number.")
        if self.warmup_s < 0:
            raise ValueError("`warmup_s` must be non-negative.")

        for field_name in ("auto_exposure", "auto_white_balance", "anti_flicker"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"`{field_name}` must be a boolean or None.")

        for field_name in ("exposure", "gain", "white_balance", "backlight_compensation"):
            value = getattr(self, field_name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"`{field_name}` must be an integer or None.")

        if self.auto_exposure_priority not in (None, 0, 1):
            raise ValueError("`auto_exposure_priority` must be 0, 1, or None.")
        if self.power_line_frequency not in (None, 0, 1, 2):
            raise ValueError(
                "`power_line_frequency` must be 0 (disabled), 1 (50 Hz), 2 (60 Hz), or None."
            )

        if (self.exposure is not None or self.gain is not None) and self.auto_exposure is not False:
            raise ValueError(
                "`auto_exposure` must be False when `exposure` or `gain` is configured."
            )
        if self.white_balance is not None and self.auto_white_balance is not False:
            raise ValueError(
                "`auto_white_balance` must be False when `white_balance` is configured."
            )

        self.serial_number_or_name = self.serial_number_or_name.strip()
        self.warmup_s = float(self.warmup_s)

    @property
    def color_settings(self) -> dict[str, bool | int]:
        """Return only color controls explicitly configured by the caller."""

        names = (
            "auto_exposure",
            "exposure",
            "gain",
            "auto_white_balance",
            "white_balance",
            "auto_exposure_priority",
            "anti_flicker",
            "power_line_frequency",
            "backlight_compensation",
        )
        return {
            name: value
            for name in names
            if (value := getattr(self, name)) is not None
        }
