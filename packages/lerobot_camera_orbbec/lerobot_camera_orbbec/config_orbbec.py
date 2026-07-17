from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("orbbec")
@dataclass(kw_only=True)
class OrbbecCameraConfig(CameraConfig):
    """Configuration for an Orbbec RGB camera.

    ``width`` and ``height`` describe the image returned to LeRobot. For a 90°
    or 270° rotation, the SDK capture dimensions are swapped automatically.

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

        self.serial_number_or_name = self.serial_number_or_name.strip()
        self.warmup_s = float(self.warmup_s)
