#!/usr/bin/env python
"""Crop, resize, and convert a local LeRobot 0.6 image dataset to video.

LeRobot 0.6 provides ``convert_image_to_video_dataset`` but rebuilds the
episode metadata with only the fields needed for locating data and videos.
This wrapper restores source task information, non-image statistics, and
custom episode fields after the official conversion finishes. RGB crop and
resize filters run on the source PNG sequence immediately before the first
video encode, so the output is never transcoded from another lossy video.

Example:
    /home/ubuntu/miniconda3/envs/lerobot-0.6/bin/python \
        scripts/datasets/dataset_image_crop_resize_to_video.py \
        --source-root data/pick_20260725_174915 \
        --output-root data/pick_20260725_174915_crop_vid
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
import torch
from lerobot.configs.video import DepthEncoderConfig, RGBEncoderConfig, VideoEncoderConfig
from lerobot.datasets import LeRobotDataset, convert_image_to_video_dataset
from lerobot.datasets.compute_stats import (
    aggregate_stats,
    auto_downsample_height_width,
    get_feature_stats,
    sample_indices,
)
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.video_utils import get_pix_fmt_channels
from tqdm import tqdm

from utils.const import DATA_PATH

ROOT_PATH = DATA_PATH / "pick_20260725_174915_20fps"

LOGGER = logging.getLogger(__name__)

EPISODE_INDEX = "episode_index"
EPISODE_METADATA_CHUNK_INDEX = "meta/episodes/chunk_index"
EPISODE_METADATA_FILE_INDEX = "meta/episodes/file_index"
NVENC_CODECS = frozenset({"h264_nvenc", "hevc_nvenc"})

# crop: [left, top, right, bottom]
# size: [height, width], or None to keep the cropped resolution
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


@dataclass(frozen=True)
class CameraTransform:
    """Crop and optional resize geometry for one RGB camera."""

    left: int
    top: int
    right: int
    bottom: int
    resize_height: int | None = None
    resize_width: int | None = None

    @classmethod
    def from_dict(
        cls,
        camera_key: str,
        raw_config: Mapping[str, Any],
    ) -> CameraTransform:
        if not isinstance(raw_config, Mapping):
            raise TypeError(f"Configuration for {camera_key!r} must be a mapping")
        if set(raw_config) != {"crop", "size"}:
            raise ValueError(
                f"Configuration for {camera_key!r} must contain exactly 'crop' and 'size'"
            )

        left, top, right, bottom = integer_sequence(
            raw_config["crop"],
            length=4,
            label=f"{camera_key}.crop",
        )
        raw_size = raw_config["size"]
        if raw_size is None:
            resize_height = None
            resize_width = None
        else:
            resize_height, resize_width = integer_sequence(
                raw_size,
                length=2,
                label=f"{camera_key}.size",
            )

        transform = cls(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            resize_height=resize_height,
            resize_width=resize_width,
        )
        transform.validate(camera_key)
        return transform

    def validate(self, camera_key: str) -> None:
        if self.left < 0 or self.top < 0:
            raise ValueError(f"Crop origin for {camera_key!r} must be non-negative")
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError(f"Crop for {camera_key!r} must have positive width and height")
        if (self.resize_height is None) != (self.resize_width is None):
            raise ValueError(f"Resize height and width for {camera_key!r} must both be set or None")
        if self.resize_height is not None and (
            self.resize_height <= 0 or self.resize_width <= 0
        ):
            raise ValueError(f"Resize dimensions for {camera_key!r} must be positive")

    @property
    def crop_width(self) -> int:
        return self.right - self.left

    @property
    def crop_height(self) -> int:
        return self.bottom - self.top

    @property
    def output_height(self) -> int:
        return self.resize_height if self.resize_height is not None else self.crop_height

    @property
    def output_width(self) -> int:
        return self.resize_width if self.resize_width is not None else self.crop_width

    @property
    def needs_resize(self) -> bool:
        return self.resize_height is not None

    def ffmpeg_filter(self, scale_flags: str) -> str:
        filters = [f"crop={self.crop_width}:{self.crop_height}:{self.left}:{self.top}"]
        if self.needs_resize:
            filters.append(
                f"scale={self.output_width}:{self.output_height}:flags={scale_flags}"
            )
        filters.append("setsar=1")
        return ",".join(filters)


@dataclass(frozen=True)
class FFmpegNVENCConfig:
    """Settings for RGB encoding through the system FFmpeg executable."""

    vcodec: str
    pix_fmt: str = "yuv420p"
    g: int = 2
    qp: float = 30
    preset: str = "p5"
    tune: str = "hq"
    gpu: int = 0
    scale_flags: str = "lanczos"
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"

    def __post_init__(self) -> None:
        if self.vcodec not in NVENC_CODECS:
            raise ValueError(f"FFmpeg NVENC codec must be one of: {sorted(NVENC_CODECS)}")
        if self.g < 1:
            raise ValueError("NVENC GOP must be greater than zero")
        if not 0 <= self.qp <= 51 or not float(self.qp).is_integer():
            raise ValueError("NVENC QP must be an integer between 0 and 51")
        if self.gpu < 0:
            raise ValueError("NVENC GPU index must be zero or greater")
        if shutil.which(self.ffprobe_path) is None:
            raise FileNotFoundError(f"FFprobe executable not found: {self.ffprobe_path}")
        if self.vcodec not in ffmpeg_video_encoders(self.ffmpeg_path):
            raise ValueError(
                f"System FFmpeg does not expose encoder {self.vcodec!r}: {self.ffmpeg_path}"
            )

    @property
    def ffmpeg_gop(self) -> int:
        # Some FFmpeg/driver combinations crash with NVENC g=0 and reject g=1.
        # Use the minimum accepted GOP and force every frame to IDR instead.
        return 0 if self.g == 1 else self.g

    @property
    def b_frames(self) -> int:
        return 0 if self.g == 1 else min(3, self.g - 2)


@dataclass(frozen=True)
class VideoProbe:
    """Encoded video properties that must match the dataset metadata."""

    width: int
    height: int
    frame_count: int
    fps: Fraction


def load_info(dataset_root: Path) -> dict[str, Any]:
    """Load and validate the dataset info file."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset info not found: {info_path}")
    with info_path.open("r", encoding="utf-8") as info_file:
        return json.load(info_file)


def episode_metadata_paths(dataset_root: Path) -> list[Path]:
    """Return episode metadata shards in chunk and file order."""
    paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata found under: {dataset_root}")
    return paths


def load_episode_metadata(dataset_root: Path) -> pd.DataFrame:
    """Read all episode metadata shards."""
    return pd.concat(
        [pd.read_parquet(path) for path in episode_metadata_paths(dataset_root)],
        ignore_index=True,
    )


def parse_chunk_file_indices(parquet_path: Path) -> tuple[int, int]:
    """Parse chunk and file indices from an episode metadata path."""
    try:
        chunk_index = int(parquet_path.parent.name.removeprefix("chunk-"))
        file_index = int(parquet_path.stem.removeprefix("file-"))
    except ValueError as error:
        raise ValueError(f"Invalid episode metadata path: {parquet_path}") from error
    return chunk_index, file_index


def image_feature_names(info: dict[str, Any]) -> set[str]:
    """Return image feature names from info.json."""
    return {
        name
        for name, feature in info.get("features", {}).items()
        if feature.get("dtype") == "image"
    }


def episode_stat_feature(column: str) -> str | None:
    """Return the feature name represented by an episode stats column."""
    if not column.startswith("stats/"):
        return None
    feature, separator, _statistic = column.removeprefix("stats/").rpartition("/")
    return feature if separator else None


def source_columns_to_restore(
    source_columns: Sequence[str],
    image_features: set[str],
) -> list[str]:
    """Select source columns that remain valid after video encoding."""
    columns = []
    for column in source_columns:
        stat_feature = episode_stat_feature(column)
        is_image_stat = stat_feature in image_features
        is_output_locator = column in {
            EPISODE_METADATA_CHUNK_INDEX,
            EPISODE_METADATA_FILE_INDEX,
        }
        if column != EPISODE_INDEX and not is_image_stat and not is_output_locator:
            columns.append(column)
    return columns


def restore_episode_metadata(source_root: Path, output_root: Path) -> None:
    """Restore metadata fields omitted by LeRobot's official conversion."""
    source_info = load_info(source_root)
    source_episodes = load_episode_metadata(source_root)
    if EPISODE_INDEX not in source_episodes:
        raise KeyError(f"Source episode metadata is missing: {EPISODE_INDEX}")
    if source_episodes[EPISODE_INDEX].duplicated().any():
        raise ValueError("Source episode metadata contains duplicate episode indices")

    restore_columns = source_columns_to_restore(
        list(source_episodes.columns),
        image_feature_names(source_info),
    )
    source_by_episode = source_episodes.set_index(EPISODE_INDEX)

    for output_path in episode_metadata_paths(output_root):
        output_episodes = pd.read_parquet(output_path)
        if EPISODE_INDEX not in output_episodes:
            raise KeyError(f"Output episode metadata is missing: {EPISODE_INDEX}")

        missing_episode_indices = set(output_episodes[EPISODE_INDEX]) - set(source_by_episode.index)
        if missing_episode_indices:
            raise ValueError(
                "Output contains episode indices absent from the source: "
                f"{sorted(missing_episode_indices)}"
            )

        for column in restore_columns:
            source_values = source_by_episode[column]
            if column in output_episodes:
                output_episodes[column] = output_episodes[column].where(
                    output_episodes[column].notna(),
                    output_episodes[EPISODE_INDEX].map(source_values),
                )
            else:
                output_episodes[column] = output_episodes[EPISODE_INDEX].map(source_values)

        chunk_index, file_index = parse_chunk_file_indices(output_path)
        output_episodes[EPISODE_METADATA_CHUNK_INDEX] = chunk_index
        output_episodes[EPISODE_METADATA_FILE_INDEX] = file_index
        output_episodes.to_parquet(output_path, index=False)


def validate_lerobot_version() -> str:
    """Require the LeRobot API used by this script."""
    try:
        installed_version = version("lerobot")
    except PackageNotFoundError as error:
        raise RuntimeError("LeRobot is not installed in the active Python environment") from error

    major_minor = installed_version.split(".")[:2]
    if major_minor != ["0", "6"]:
        raise RuntimeError(
            f"This script requires LeRobot 0.6.x, found {installed_version}. "
            "Run it with the lerobot-0.6 Python environment."
        )
    return installed_version


def integer_sequence(value: Any, length: int, label: str) -> tuple[int, ...]:
    """Validate and normalize a fixed-length integer sequence."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence of {length} integers")
    if len(value) != length:
        raise ValueError(f"{label} must contain {length} integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError(f"{label} must contain only integers")
    return tuple(value)


def parse_camera_config(
    raw_config: Mapping[str, Mapping[str, Any]],
) -> dict[str, CameraTransform]:
    """Parse the module-level camera transformation mapping."""
    if not isinstance(raw_config, Mapping) or not raw_config:
        raise ValueError("CAMERA_CONFIG must be a non-empty mapping")
    return {
        camera_key: CameraTransform.from_dict(camera_key, config)
        for camera_key, config in raw_config.items()
    }


def validate_dataset_and_transforms(
    dataset: LeRobotDataset,
    transforms: Mapping[str, CameraTransform],
    encoder: FFmpegNVENCConfig,
) -> None:
    """Validate RGB camera coverage, crop bounds, and encoder geometry."""
    if not dataset.meta.image_keys:
        raise ValueError(f"Source dataset has no image features: {dataset.root}")
    if dataset.meta.video_keys:
        raise ValueError(f"Source dataset already contains video features: {dataset.root}")
    if dataset.meta.total_episodes <= 0:
        raise ValueError(f"Source dataset has no episodes: {dataset.root}")

    rgb_keys = set(dataset.meta.image_keys) - set(dataset.meta.depth_keys)
    configured_keys = set(transforms)
    missing = sorted(rgb_keys - configured_keys)
    unknown = sorted(configured_keys - rgb_keys)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing camera configuration for {missing}")
        if unknown:
            details.append(f"unknown RGB camera keys {unknown}")
        raise ValueError("Invalid CAMERA_CONFIG: " + "; ".join(details))

    for camera_key in sorted(rgb_keys):
        shape = tuple(dataset.meta.features[camera_key]["shape"])
        if len(shape) != 3 or shape[-1] != 3:
            raise ValueError(
                f"Camera {camera_key!r} must use HWC shape [H, W, 3], found {shape}"
            )
        source_height, source_width, _ = shape
        transform = transforms[camera_key]
        if transform.right > source_width or transform.bottom > source_height:
            raise ValueError(
                f"Crop for {camera_key!r} exceeds source size "
                f"{source_width}x{source_height}: "
                f"{[transform.left, transform.top, transform.right, transform.bottom]}"
            )
        if encoder.pix_fmt.startswith("yuv420") and (
            transform.output_width % 2 or transform.output_height % 2
        ):
            raise ValueError(
                f"Output size for {camera_key!r} must be even for {encoder.pix_fmt}: "
                f"{transform.output_width}x{transform.output_height}"
            )
        if encoder.pix_fmt.startswith("yuv422") and transform.output_width % 2:
            raise ValueError(
                f"Output width for {camera_key!r} must be even for {encoder.pix_fmt}"
            )


def remove_existing_output(output_root: Path, source_root: Path, overwrite: bool) -> None:
    """Remove an existing output only when explicitly requested."""
    if (
        output_root == source_root
        or output_root in source_root.parents
        or source_root in output_root.parents
    ):
        raise ValueError("Source and output roots must be separate, non-nested directories")
    if not output_root.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite to replace it.")
    if output_root.is_symlink() or output_root.is_file():
        output_root.unlink()
    elif output_root.is_dir():
        shutil.rmtree(output_root)
    else:
        output_root.unlink()


@cache
def ffmpeg_video_encoders(ffmpeg_path: str = "ffmpeg") -> frozenset[str]:
    """Return video encoders exposed by the selected system FFmpeg."""
    resolved_path = shutil.which(ffmpeg_path)
    if resolved_path is None:
        raise FileNotFoundError(f"FFmpeg executable not found: {ffmpeg_path}")
    result = subprocess.run(
        [resolved_path, "-hide_banner", "-encoders"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    encoders = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and len(fields[0]) == 6 and fields[0].startswith("V"):
            encoders.add(fields[1])
    return frozenset(encoders)


def encode_rgb_video_frames_ffmpeg(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    config: FFmpegNVENCConfig,
    transform: CameraTransform,
    *,
    overwrite: bool,
) -> None:
    """Crop, resize, and encode LeRobot RGB PNG frames in one FFmpeg pass."""
    imgs_dir = Path(imgs_dir)
    video_path = Path(video_path)
    first_frame = imgs_dir / "frame-000000.png"
    if not first_frame.is_file():
        raise FileNotFoundError(f"No RGB frames found under: {imgs_dir}")
    if video_path.exists() and not overwrite:
        LOGGER.warning("Video already exists, skipping: %s", video_path)
        return

    resolved_ffmpeg = shutil.which(config.ffmpeg_path)
    if resolved_ffmpeg is None:
        raise FileNotFoundError(f"FFmpeg executable not found: {config.ffmpeg_path}")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        resolved_ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        str(imgs_dir / "frame-%06d.png"),
        "-an",
        "-vf",
        transform.ffmpeg_filter(config.scale_flags),
        "-c:v",
        config.vcodec,
        "-gpu",
        str(config.gpu),
        "-preset",
        config.preset,
        "-tune",
        config.tune,
        "-rc",
        "constqp",
        "-qp",
        str(int(config.qp)),
        "-g",
        str(config.ffmpeg_gop),
        "-bf",
        str(config.b_frames),
    ]
    if config.g == 1:
        command.extend(
            [
                "-forced-idr",
                "1",
                "-force_key_frames",
                "expr:gte(n,n_forced*1)",
            ]
        )
    command.extend(
        [
            "-pix_fmt",
            config.pix_fmt,
            "-movflags",
            "+faststart",
            str(video_path),
        ]
    )
    LOGGER.info(
        "Encoding with system FFmpeg: codec=%s preset=%s qp=%s g=%d ffmpeg_g=%d "
        "b_frames=%d gpu=%d filter=%s",
        config.vcodec,
        config.preset,
        config.qp,
        config.g,
        config.ffmpeg_gop,
        config.b_frames,
        config.gpu,
        transform.ffmpeg_filter(config.scale_flags),
    )
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        video_path.unlink(missing_ok=True)
        error_message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"FFmpeg {config.vcodec} encoding failed with exit code {result.returncode}: "
            f"{error_message}"
        )
    if not video_path.is_file():
        raise OSError(f"FFmpeg did not create the expected video: {video_path}")


@contextmanager
def ffmpeg_rgb_encoding_patch(
    config: FFmpegNVENCConfig,
    transforms: Mapping[str, CameraTransform],
):
    """Route LeRobot RGB encoding through the crop/resize FFmpeg pipeline."""
    import lerobot.datasets.dataset_tools as dataset_tools

    original_encode_video_frames = dataset_tools.encode_video_frames

    def encode_video_frames_dispatch(
        imgs_dir: Path | str,
        video_path: Path | str,
        fps: int,
        video_encoder: VideoEncoderConfig | None = None,
        encoder_threads: int | None = None,
        *,
        log_level: int | None = av.logging.WARNING,
        overwrite: bool = False,
    ) -> None:
        if isinstance(video_encoder, DepthEncoderConfig):
            original_encode_video_frames(
                imgs_dir=imgs_dir,
                video_path=video_path,
                fps=fps,
                video_encoder=video_encoder,
                encoder_threads=encoder_threads,
                log_level=log_level,
                overwrite=overwrite,
            )
            return
        camera_key = Path(imgs_dir).name
        if camera_key not in transforms:
            raise KeyError(
                f"No camera transform found for RGB frame directory {camera_key!r}"
            )
        encode_rgb_video_frames_ffmpeg(
            imgs_dir=imgs_dir,
            video_path=video_path,
            fps=fps,
            config=config,
            transform=transforms[camera_key],
            overwrite=overwrite,
        )

    LOGGER.info(
        "Using system FFmpeg RGB backend with %s on GPU %d",
        config.vcodec,
        config.gpu,
    )
    dataset_tools.encode_video_frames = encode_video_frames_dispatch
    try:
        yield
    finally:
        dataset_tools.encode_video_frames = original_encode_video_frames


def update_output_video_metadata(
    output_root: Path,
    config: FFmpegNVENCConfig,
    transforms: Mapping[str, CameraTransform],
) -> None:
    """Record transformed shapes and external FFmpeg encoder details."""
    info = load_info(output_root)
    features = info.get("features", {})
    for camera_key, transform in transforms.items():
        if camera_key not in features:
            raise KeyError(f"Output metadata is missing camera feature {camera_key!r}")
        feature = features[camera_key]
        if feature.get("dtype") != "video":
            raise ValueError(f"Output feature {camera_key!r} is not a video")
        feature_info = feature.get("info") or {}
        actual_width = feature_info.get("video.width")
        actual_height = feature_info.get("video.height")
        expected_size = (transform.output_width, transform.output_height)
        actual_size = (actual_width, actual_height)
        if actual_size != expected_size:
            raise ValueError(
                f"Encoded size mismatch for {camera_key!r}: "
                f"expected {expected_size}, found {actual_size}"
            )
        feature["shape"] = [transform.output_height, transform.output_width, 3]
        feature_info.update(
            {
                "video.ffmpeg_encoder": config.vcodec,
                "video.ffmpeg_preset": config.preset,
                "video.ffmpeg_tune": config.tune,
                "video.ffmpeg_rate_control": "constqp",
                "video.ffmpeg_qp": config.qp,
                "video.ffmpeg_gop": config.ffmpeg_gop,
                "video.ffmpeg_b_frames": config.b_frames,
                "video.ffmpeg_gpu": config.gpu,
                "video.ffmpeg_scale_flags": config.scale_flags,
                "video.ffmpeg_filter": transform.ffmpeg_filter(config.scale_flags),
            }
        )
        feature["info"] = feature_info
    info_path = output_root / "meta" / "info.json"
    with info_path.open("w", encoding="utf-8") as info_file:
        json.dump(info, info_file, indent=4)
        info_file.write("\n")


def codec_name(codec: Any) -> str:
    """Return a codec name across supported PyAV versions."""
    return getattr(codec, "canonical_name", None) or codec.name


def get_video_info_compat(
    video_path: Path | str,
    video_encoder: VideoEncoderConfig | None = None,
) -> dict[str, Any]:
    """LeRobot get_video_info implementation compatible with PyAV before 15."""
    video_info: dict[str, Any] = {}
    with av.open(str(video_path), "r") as video_file:
        if not video_file.streams.video:
            return video_info
        video_stream = video_file.streams.video[0]
        video_info.update(
            {
                "video.height": video_stream.height,
                "video.width": video_stream.width,
                "video.codec": codec_name(video_stream.codec),
                "video.pix_fmt": video_stream.pix_fmt,
                "video.fps": int(video_stream.base_rate),
                "video.channels": get_pix_fmt_channels(video_stream.pix_fmt),
            }
        )

        if video_file.streams.audio:
            audio_stream = video_file.streams.audio[0]
            video_info.update(
                {
                    "audio.channels": audio_stream.channels,
                    "audio.codec": codec_name(audio_stream.codec),
                    "audio.bit_rate": audio_stream.bit_rate,
                    "audio.sample_rate": audio_stream.sample_rate,
                    "audio.bit_depth": audio_stream.format.bits,
                    "audio.channel_layout": audio_stream.layout.name,
                    "has_audio": True,
                }
            )
        else:
            video_info["has_audio"] = False

    if video_encoder is not None:
        for field_name, field_value in asdict(video_encoder).items():
            if field_name != "vcodec":
                video_info.setdefault(f"video.{field_name}", field_value)
    video_info["is_depth_map"] = isinstance(video_encoder, DepthEncoderConfig)
    return video_info


@contextmanager
def pyav_compatibility_patch():
    """Patch LeRobot 0.6 video probing only when PyAV lacks canonical_name."""
    probe_codec = av.Codec("h264", "r")
    if hasattr(probe_codec, "canonical_name"):
        yield
        return

    import lerobot.datasets.dataset_metadata as dataset_metadata

    original_get_video_info = dataset_metadata.get_video_info
    LOGGER.warning(
        "PyAV %s lacks Codec.canonical_name, enabling LeRobot 0.6 compatibility mode",
        av.__version__,
    )
    dataset_metadata.get_video_info = get_video_info_compat
    try:
        yield
    finally:
        dataset_metadata.get_video_info = original_get_video_info


def frame_to_uint8_chw(frame: torch.Tensor, camera_key: str) -> np.ndarray:
    """Convert one decoded LeRobot RGB frame to a sampled CHW uint8 array."""
    array = frame.detach().cpu().numpy()
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(
            f"Decoded frame for {camera_key!r} must be CHW RGB, found {array.shape}"
        )
    if array.dtype != np.uint8:
        raise TypeError(
            f"Decoded frame for {camera_key!r} must be uint8, found {array.dtype}"
        )
    return auto_downsample_height_width(array)


def compute_episode_camera_stats(
    dataset: LeRobotDataset,
    camera_keys: Sequence[str],
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    """Recompute sampled image statistics from the final decoded videos."""
    episode_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    with tqdm(
        total=dataset.meta.total_episodes,
        desc="Recomputing image stats",
        unit="episode",
    ) as progress:
        for episode_index in range(dataset.meta.total_episodes):
            episode = dataset.meta.episodes[episode_index]
            start_index = episode["dataset_from_index"]
            episode_length = episode["length"]
            absolute_indices = [
                start_index + relative_index
                for relative_index in sample_indices(episode_length)
            ]
            sampled_frames = {camera_key: [] for camera_key in camera_keys}
            for absolute_index in absolute_indices:
                item = dataset[absolute_index]
                for camera_key in camera_keys:
                    sampled_frames[camera_key].append(
                        frame_to_uint8_chw(item[camera_key], camera_key)
                    )

            camera_stats = {}
            for camera_key in camera_keys:
                batch = np.stack(sampled_frames[camera_key])
                stats = get_feature_stats(
                    batch,
                    axis=(0, 2, 3),
                    keepdims=True,
                )
                camera_stats[camera_key] = {
                    stat_name: (
                        stat_value
                        if stat_name == "count"
                        else np.squeeze(stat_value / 255.0, axis=0)
                    )
                    for stat_name, stat_value in stats.items()
                }
            episode_stats[episode_index] = camera_stats
            progress.update()
    return episode_stats


def update_episode_camera_stats(
    output_root: Path,
    episode_stats: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
) -> None:
    """Write transformed camera statistics into every episode metadata shard."""
    episode_paths = episode_metadata_paths(output_root)
    first_episode_stats = next(iter(episode_stats.values()))
    stat_names = {
        camera_key: tuple(camera_stats)
        for camera_key, camera_stats in first_episode_stats.items()
    }
    for episode_path in episode_paths:
        episode_df = pd.read_parquet(episode_path)
        for camera_key, camera_stat_names in stat_names.items():
            for stat_name in camera_stat_names:
                column = f"stats/{camera_key}/{stat_name}"
                episode_df[column] = episode_df[EPISODE_INDEX].map(
                    lambda episode_index: episode_stats[int(episode_index)][camera_key][
                        stat_name
                    ].tolist()
                )
        episode_df.to_parquet(episode_path, index=False)


def update_global_camera_stats(
    output_dataset: LeRobotDataset,
    output_root: Path,
    episode_stats: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
) -> None:
    """Merge transformed camera statistics with unchanged output statistics."""
    camera_stats = aggregate_stats(list(episode_stats.values()))
    output_stats = deepcopy(output_dataset.meta.stats) if output_dataset.meta.stats else {}
    output_stats.update(camera_stats)
    write_stats(output_stats, output_root)


def probe_video(video_path: Path, ffprobe_path: str) -> VideoProbe:
    """Read encoded dimensions, frame count, and FPS with FFprobe."""
    resolved_ffprobe = shutil.which(ffprobe_path)
    if resolved_ffprobe is None:
        raise FileNotFoundError(f"FFprobe executable not found: {ffprobe_path}")
    result = subprocess.run(
        [
            resolved_ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found: {video_path}")
    stream = streams[0]
    frame_count = stream.get("nb_frames")
    if frame_count in (None, "N/A"):
        count_result = subprocess.run(
            [
                resolved_ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        count_streams = json.loads(count_result.stdout).get("streams", [])
        frame_count = count_streams[0].get("nb_read_frames") if count_streams else None
        if frame_count in (None, "N/A"):
            raise ValueError(f"Unable to count video frames: {video_path}")
    return VideoProbe(
        width=int(stream["width"]),
        height=int(stream["height"]),
        frame_count=int(frame_count),
        fps=Fraction(stream["avg_frame_rate"]),
    )


def validate_output_video_shards(
    output_dataset: LeRobotDataset,
    transforms: Mapping[str, CameraTransform],
    encoder: FFmpegNVENCConfig,
) -> None:
    """Validate every output shard against episode lengths and dataset FPS."""
    for camera_key, transform in transforms.items():
        expected_frames_by_path: dict[Path, int] = {}
        for episode_index in range(output_dataset.meta.total_episodes):
            relative_path = output_dataset.meta.get_video_file_path(
                episode_index,
                camera_key,
            )
            video_path = output_dataset.root / relative_path
            expected_frames_by_path[video_path] = (
                expected_frames_by_path.get(video_path, 0)
                + output_dataset.meta.episodes[episode_index]["length"]
            )

        for video_path, expected_frames in expected_frames_by_path.items():
            video_probe = probe_video(video_path, encoder.ffprobe_path)
            expected_size = (transform.output_width, transform.output_height)
            actual_size = (video_probe.width, video_probe.height)
            if actual_size != expected_size:
                raise ValueError(
                    f"Video size mismatch for {video_path}: "
                    f"expected {expected_size}, found {actual_size}"
                )
            if video_probe.frame_count != expected_frames:
                raise ValueError(
                    f"Video frame count mismatch for {video_path}: "
                    f"expected {expected_frames}, found {video_probe.frame_count}"
                )
            if video_probe.fps != Fraction(output_dataset.meta.fps):
                raise ValueError(
                    f"Video FPS mismatch for {video_path}: "
                    f"expected {output_dataset.meta.fps}, found {video_probe.fps}"
                )


def validate_output_dataset(
    source_dataset: LeRobotDataset,
    output_dataset: LeRobotDataset,
    transforms: Mapping[str, CameraTransform],
    encoder: FFmpegNVENCConfig,
) -> None:
    """Validate full-dataset counts, transformed shapes, and decoded frames."""
    if output_dataset.meta.total_episodes != source_dataset.meta.total_episodes:
        raise ValueError("Output episode count does not match the source")
    if len(output_dataset) != len(source_dataset):
        raise ValueError("Output frame count does not match the source")
    for camera_key, transform in transforms.items():
        expected_hwc = (transform.output_height, transform.output_width, 3)
        actual_hwc = tuple(output_dataset.meta.features[camera_key]["shape"])
        if actual_hwc != expected_hwc:
            raise ValueError(
                f"Metadata shape mismatch for {camera_key!r}: "
                f"expected {expected_hwc}, found {actual_hwc}"
            )
        decoded_shape = tuple(output_dataset[0][camera_key].shape)
        expected_chw = (3, transform.output_height, transform.output_width)
        if decoded_shape != expected_chw:
            raise ValueError(
                f"Decoded shape mismatch for {camera_key!r}: "
                f"expected {expected_chw}, found {decoded_shape}"
            )
    validate_output_video_shards(output_dataset, transforms, encoder)


def convert_dataset(
    source_root: Path,
    output_root: Path,
    camera_config: Mapping[str, Mapping[str, Any]],
    source_repo_id: str | None = None,
    output_repo_id: str | None = None,
    *,
    rgb_encoder: RGBEncoderConfig | None = None,
    ffmpeg_rgb_encoder: FFmpegNVENCConfig | None = None,
    depth_encoder: DepthEncoderConfig | None = None,
    num_workers: int = 4,
    max_episodes_per_batch: int | None = None,
    max_frames_per_batch: int | None = None,
    overwrite: bool = False,
) -> LeRobotDataset:
    """Transform every image episode, encode once, and restore metadata."""
    validate_lerobot_version()
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    load_info(source_root)

    source_repo_id = source_repo_id or source_root.name
    output_repo_id = output_repo_id or f"{source_repo_id}_crop_vid"
    source_dataset = LeRobotDataset(repo_id=source_repo_id, root=source_root)
    transforms = parse_camera_config(camera_config)
    if ffmpeg_rgb_encoder is None:
        raise ValueError("Crop and resize conversion requires a system FFmpeg NVENC encoder")
    validate_dataset_and_transforms(source_dataset, transforms, ffmpeg_rgb_encoder)

    if rgb_encoder is None:
        rgb_encoder = RGBEncoderConfig(
            vcodec=ffmpeg_rgb_encoder.vcodec.removesuffix("_nvenc"),
            pix_fmt=ffmpeg_rgb_encoder.pix_fmt,
            g=ffmpeg_rgb_encoder.g,
            crf=ffmpeg_rgb_encoder.qp,
            preset=None,
        )
    if depth_encoder is None:
        depth_encoder = DepthEncoderConfig()

    remove_existing_output(output_root, source_root, overwrite)
    LOGGER.info(
        "Converting dataset %s with %d episodes and %d image features",
        source_repo_id,
        source_dataset.meta.total_episodes,
        len(source_dataset.meta.image_keys),
    )

    with pyav_compatibility_patch(), ffmpeg_rgb_encoding_patch(
        ffmpeg_rgb_encoder,
        transforms,
    ):
        convert_image_to_video_dataset(
            dataset=source_dataset,
            output_dir=output_root,
            repo_id=output_repo_id,
            rgb_encoder=rgb_encoder,
            depth_encoder=depth_encoder,
            episode_indices=None,
            num_workers=num_workers,
            max_episodes_per_batch=max_episodes_per_batch,
            max_frames_per_batch=max_frames_per_batch,
        )
    restore_episode_metadata(source_root, output_root)
    update_output_video_metadata(
        output_root=output_root,
        config=ffmpeg_rgb_encoder,
        transforms=transforms,
    )

    converted_dataset = LeRobotDataset(
        repo_id=output_repo_id,
        root=output_root,
        return_uint8=True,
        video_backend="pyav",
    )
    episode_stats = compute_episode_camera_stats(
        dataset=converted_dataset,
        camera_keys=list(transforms),
    )
    update_episode_camera_stats(output_root, episode_stats)
    update_global_camera_stats(converted_dataset, output_root, episode_stats)

    converted_dataset = LeRobotDataset(
        repo_id=output_repo_id,
        root=output_root,
        return_uint8=True,
        video_backend="pyav",
    )
    validate_output_dataset(
        source_dataset,
        converted_dataset,
        transforms,
        ffmpeg_rgb_encoder,
    )
    LOGGER.info(
        "Converted dataset saved to %s with %d episodes and %d frames",
        output_root,
        converted_dataset.meta.total_episodes,
        converted_dataset.meta.total_frames,
    )
    return converted_dataset


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def optional_preset(value: str) -> int | str | None:
    """Parse a numeric, named, or disabled encoder preset."""
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def build_rgb_encoder_configs(
    args: argparse.Namespace,
) -> tuple[RGBEncoderConfig, FFmpegNVENCConfig | None]:
    """Build the LeRobot metadata config and optional external FFmpeg config."""
    preset = args.preset
    if preset == "auto":
        preset = "p5" if args.vcodec in NVENC_CODECS else "veryfast"

    if args.vcodec in NVENC_CODECS:
        if not isinstance(preset, str):
            raise ValueError("NVENC preset must be a name such as p4 or p5")
        ffmpeg_encoder = FFmpegNVENCConfig(
            vcodec=args.vcodec,
            pix_fmt=args.pix_fmt,
            g=args.gop,
            qp=args.qp,
            preset=preset,
            tune=args.nvenc_tune,
            gpu=args.nvenc_gpu,
            scale_flags=args.scale_flags,
            ffmpeg_path=args.ffmpeg_path,
            ffprobe_path=args.ffprobe_path,
        )
        stream_codec = args.vcodec.removesuffix("_nvenc")
        metadata_encoder = RGBEncoderConfig(
            vcodec=stream_codec,
            pix_fmt=args.pix_fmt,
            g=args.gop,
            crf=args.qp,
            preset=None,
            fast_decode=args.fast_decode,
        )
        return metadata_encoder, ffmpeg_encoder

    metadata_encoder = RGBEncoderConfig(
        vcodec=args.vcodec,
        pix_fmt=args.pix_fmt,
        g=args.gop,
        crf=args.qp,
        preset=preset,
        fast_decode=args.fast_decode,
    )
    return metadata_encoder, None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop, resize, and encode every episode in a local LeRobot 0.6 image dataset"
        ),
    )
    parser.add_argument("--source-root", type=Path, default=ROOT_PATH, help="Source dataset directory")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output dataset directory. Defaults to SOURCE_ROOT_crop_vid.",
    )
    parser.add_argument("--repo-id", help="Source repo id. Defaults to the source directory name.")
    parser.add_argument(
        "--output-repo-id",
        help="Output repo id. Defaults to REPO_ID_crop_vid.",
    )
    parser.add_argument("--num-workers", type=positive_integer, default=12)
    parser.add_argument("--max-episodes-per-batch", type=positive_integer)
    parser.add_argument("--max-frames-per-batch", type=positive_integer)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show the conversion configuration without writing output.",
    )

    rgb_group = parser.add_argument_group("RGB encoder")
    rgb_group.add_argument(
        "--vcodec",
        choices=sorted(NVENC_CODECS),
        default="hevc_nvenc",
    )
    rgb_group.add_argument("--pix-fmt", default="yuv420p")
    rgb_group.add_argument("--gop", type=positive_integer, default=1)
    rgb_group.add_argument(
        "--qp",
        "--crf",
        dest="qp",
        type=float,
        default=21,
        help="NVENC constant QP. Lower values preserve more detail.",
    )
    rgb_group.add_argument(
        "--preset",
        type=optional_preset,
        default="p6",
        help="NVENC preset name such as p4, p5, or p6. Auto selects p5.",
    )
    rgb_group.add_argument("--fast-decode", type=int, choices=range(0, 3), default=0)
    rgb_group.add_argument(
        "--ffmpeg-path",
        default="ffmpeg",
        help="System FFmpeg executable used by NVENC codecs.",
    )
    rgb_group.add_argument("--ffprobe-path", default="ffprobe")
    rgb_group.add_argument("--nvenc-gpu", type=int, default=0)
    rgb_group.add_argument("--scale-flags", default="lanczos")
    rgb_group.add_argument(
        "--nvenc-tune",
        choices=("hq", "ll", "ull", "lossless"),
        default="hq",
    )

    depth_group = parser.add_argument_group("Depth encoder")
    depth_group.add_argument("--depth-vcodec", default="hevc")
    depth_group.add_argument("--depth-pix-fmt", default="gray12le")
    depth_group.add_argument("--depth-gop", type=positive_integer, default=1)
    depth_group.add_argument("--depth-crf", type=float, default=30)
    depth_group.add_argument("--depth-preset", type=optional_preset, default=None)
    depth_group.add_argument("--depth-fast-decode", type=int, choices=range(0, 3), default=0)
    depth_group.add_argument("--depth-min", type=float, default=0.01)
    depth_group.add_argument("--depth-max", type=float, default=10.0)
    depth_group.add_argument("--depth-shift", type=float, default=3.5)
    depth_group.add_argument(
        "--depth-use-log",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

    source_root = args.source_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else source_root.with_name(f"{source_root.name}_crop_vid")
    )
    load_info(source_root)
    rgb_encoder, ffmpeg_rgb_encoder = build_rgb_encoder_configs(args)
    if ffmpeg_rgb_encoder is None:
        raise ValueError("Crop and resize conversion requires an NVENC codec")

    if args.dry_run:
        installed_version = validate_lerobot_version()
        source_repo_id = args.repo_id or source_root.name
        source_dataset = LeRobotDataset(repo_id=source_repo_id, root=source_root)
        transforms = parse_camera_config(CAMERA_CONFIG)
        validate_dataset_and_transforms(source_dataset, transforms, ffmpeg_rgb_encoder)
        LOGGER.info("LeRobot version: %s", installed_version)
        LOGGER.info("Source root: %s", source_root)
        LOGGER.info("Output root: %s", output_root)
        LOGGER.info("Episodes: all (%d)", source_dataset.meta.total_episodes)
        LOGGER.info(
            "RGB encoder: backend=system_ffmpeg codec=%s pix_fmt=%s g=%d qp=%s "
            "preset=%s tune=%s gpu=%d scale_flags=%s",
            ffmpeg_rgb_encoder.vcodec,
            ffmpeg_rgb_encoder.pix_fmt,
            ffmpeg_rgb_encoder.g,
            ffmpeg_rgb_encoder.qp,
            ffmpeg_rgb_encoder.preset,
            ffmpeg_rgb_encoder.tune,
            ffmpeg_rgb_encoder.gpu,
            ffmpeg_rgb_encoder.scale_flags,
        )
        for camera_key, transform in transforms.items():
            LOGGER.info(
                "Camera transform: key=%s filter=%s output=%dx%d",
                camera_key,
                transform.ffmpeg_filter(ffmpeg_rgb_encoder.scale_flags),
                transform.output_width,
                transform.output_height,
            )
        return 0

    depth_encoder = DepthEncoderConfig(
        vcodec=args.depth_vcodec,
        pix_fmt=args.depth_pix_fmt,
        g=args.depth_gop,
        crf=args.depth_crf,
        preset=args.depth_preset,
        fast_decode=args.depth_fast_decode,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        shift=args.depth_shift,
        use_log=args.depth_use_log,
    )
    if depth_encoder.depth_min >= depth_encoder.depth_max:
        raise ValueError("--depth-min must be less than --depth-max")

    convert_dataset(
        source_root=source_root,
        output_root=output_root,
        camera_config=CAMERA_CONFIG,
        source_repo_id=args.repo_id,
        output_repo_id=args.output_repo_id,
        rgb_encoder=rgb_encoder,
        ffmpeg_rgb_encoder=ffmpeg_rgb_encoder,
        depth_encoder=depth_encoder,
        num_workers=args.num_workers,
        max_episodes_per_batch=args.max_episodes_per_batch,
        max_frames_per_batch=args.max_frames_per_batch,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
