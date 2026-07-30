#!/usr/bin/env python
"""Convert a local LeRobot 0.6 image dataset to video format.

LeRobot 0.6 provides ``convert_image_to_video_dataset`` but rebuilds the
episode metadata with only the fields needed for locating data and videos.
This wrapper restores source task information, non-image statistics, and
custom episode fields after the official conversion finishes.

Example:
    /home/ubuntu/miniconda3/envs/lerobot-0.6/bin/python \
        scripts/datasets/dataset_image_to_video.py \
        data/pick_20260725_172423 \
        --output-root data/pick_20260725_172423_video \
        --vcodec hevc_nvenc --preset p5 --crf 22 --gop 10
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import av
import pandas as pd
from lerobot.configs.video import DepthEncoderConfig, RGBEncoderConfig, VideoEncoderConfig
from lerobot.datasets import LeRobotDataset, convert_image_to_video_dataset
from lerobot.datasets.video_utils import get_pix_fmt_channels

from utils.const import DATA_PATH

ROOT_PATH = DATA_PATH / "pick_20260725_174915"

LOGGER = logging.getLogger(__name__)

EPISODE_INDEX = "episode_index"
EPISODE_METADATA_CHUNK_INDEX = "meta/episodes/chunk_index"
EPISODE_METADATA_FILE_INDEX = "meta/episodes/file_index"
NVENC_CODECS = frozenset({"h264_nvenc", "hevc_nvenc"})


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
    ffmpeg_path: str = "ffmpeg"

    def __post_init__(self) -> None:
        if self.vcodec not in NVENC_CODECS:
            raise ValueError(f"FFmpeg NVENC codec must be one of: {sorted(NVENC_CODECS)}")
        if self.g < 1:
            raise ValueError("NVENC GOP must be greater than zero")
        if not 0 <= self.qp <= 51 or not float(self.qp).is_integer():
            raise ValueError("NVENC QP must be an integer between 0 and 51")
        if self.gpu < 0:
            raise ValueError("NVENC GPU index must be zero or greater")
        if self.vcodec not in ffmpeg_video_encoders(self.ffmpeg_path):
            raise ValueError(
                f"System FFmpeg does not expose encoder {self.vcodec!r}: {self.ffmpeg_path}"
            )


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


def validate_episode_indices(
    episode_indices: Sequence[int] | None,
    total_episodes: int,
) -> list[int] | None:
    """Validate and normalize the requested episode indices."""
    if episode_indices is None:
        return None
    normalized = list(dict.fromkeys(episode_indices))
    invalid = [index for index in normalized if index < 0 or index >= total_episodes]
    if invalid:
        raise ValueError(
            f"Episode indices out of range: {invalid}. "
            f"Expected indices from 0 to {total_episodes - 1}."
        )
    if not normalized:
        raise ValueError("At least one episode index is required")
    return normalized


def remove_existing_output(output_root: Path, source_root: Path, overwrite: bool) -> None:
    """Remove an existing output only when explicitly requested."""
    if output_root == source_root or output_root in source_root.parents:
        raise ValueError("Output root must not be the source root or one of its parent directories")
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
    *,
    overwrite: bool,
) -> None:
    """Encode LeRobot RGB frame files with the system FFmpeg NVENC encoder."""
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
    # FFmpeg 8 NVENC uses g=0 for all-intra encoding. g=1 is rejected by
    # NVENC because the GOP must be longer than the B-frame interval.
    ffmpeg_gop = 0 if config.g == 1 else config.g
    b_frames = 0 if config.g == 1 else min(3, config.g - 2)
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
        str(ffmpeg_gop),
        "-bf",
        str(b_frames),
        "-pix_fmt",
        config.pix_fmt,
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    LOGGER.info(
        "Encoding with system FFmpeg: codec=%s preset=%s qp=%s g=%d ffmpeg_g=%d "
        "b_frames=%d gpu=%d",
        config.vcodec,
        config.preset,
        config.qp,
        config.g,
        ffmpeg_gop,
        b_frames,
        config.gpu,
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
def ffmpeg_rgb_encoding_patch(config: FFmpegNVENCConfig | None):
    """Route LeRobot RGB frame encoding through system FFmpeg when requested."""
    if config is None:
        yield
        return

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
        encode_rgb_video_frames_ffmpeg(
            imgs_dir=imgs_dir,
            video_path=video_path,
            fps=fps,
            config=config,
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


def annotate_ffmpeg_encoder(output_root: Path, config: FFmpegNVENCConfig) -> None:
    """Record external FFmpeg encoder details without affecting LeRobot decoding."""
    info = load_info(output_root)
    for feature in info.get("features", {}).values():
        feature_info = feature.get("info") or {}
        if feature.get("dtype") != "video" or feature_info.get("is_depth_map"):
            continue
        feature_info.update(
            {
                "video.ffmpeg_encoder": config.vcodec,
                "video.ffmpeg_preset": config.preset,
                "video.ffmpeg_tune": config.tune,
                "video.ffmpeg_rate_control": "constqp",
                "video.ffmpeg_qp": config.qp,
                "video.ffmpeg_gop": 0 if config.g == 1 else config.g,
                "video.ffmpeg_b_frames": 0 if config.g == 1 else min(3, config.g - 2),
                "video.ffmpeg_gpu": config.gpu,
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


def convert_dataset(
    source_root: Path,
    output_root: Path,
    source_repo_id: str | None = None,
    output_repo_id: str | None = None,
    *,
    episode_indices: Sequence[int] | None = None,
    rgb_encoder: RGBEncoderConfig | None = None,
    ffmpeg_rgb_encoder: FFmpegNVENCConfig | None = None,
    depth_encoder: DepthEncoderConfig | None = None,
    num_workers: int = 4,
    max_episodes_per_batch: int | None = None,
    max_frames_per_batch: int | None = None,
    overwrite: bool = False,
) -> LeRobotDataset:
    """Convert a local image dataset and restore its episode metadata."""
    validate_lerobot_version()
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    load_info(source_root)

    source_repo_id = source_repo_id or source_root.name
    output_repo_id = output_repo_id or f"{source_repo_id}_video"
    source_dataset = LeRobotDataset(repo_id=source_repo_id, root=source_root)
    selected_episodes = validate_episode_indices(
        episode_indices,
        source_dataset.meta.total_episodes,
    )

    if not source_dataset.meta.image_keys:
        raise ValueError(f"Source dataset has no image features: {source_root}")
    if source_dataset.meta.video_keys:
        raise ValueError(f"Source dataset already contains video features: {source_root}")

    if rgb_encoder is None:
        if ffmpeg_rgb_encoder is None:
            rgb_encoder = RGBEncoderConfig(vcodec="h264", preset="veryfast")
        else:
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
        len(selected_episodes) if selected_episodes is not None else source_dataset.meta.total_episodes,
        len(source_dataset.meta.image_keys),
    )

    with pyav_compatibility_patch(), ffmpeg_rgb_encoding_patch(ffmpeg_rgb_encoder):
        convert_image_to_video_dataset(
            dataset=source_dataset,
            output_dir=output_root,
            repo_id=output_repo_id,
            rgb_encoder=rgb_encoder,
            depth_encoder=depth_encoder,
            episode_indices=selected_episodes,
            num_workers=num_workers,
            max_episodes_per_batch=max_episodes_per_batch,
            max_frames_per_batch=max_frames_per_batch,
        )
    restore_episode_metadata(source_root, output_root)
    if ffmpeg_rgb_encoder is not None:
        annotate_ffmpeg_encoder(output_root, ffmpeg_rgb_encoder)

    converted_dataset = LeRobotDataset(repo_id=output_repo_id, root=output_root)
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
            qp=args.crf,
            preset=preset,
            tune=args.nvenc_tune,
            gpu=args.nvenc_gpu,
            ffmpeg_path=args.ffmpeg_path,
        )
        stream_codec = args.vcodec.removesuffix("_nvenc")
        metadata_encoder = RGBEncoderConfig(
            vcodec=stream_codec,
            pix_fmt=args.pix_fmt,
            g=args.gop,
            crf=args.crf,
            preset=None,
            fast_decode=args.fast_decode,
        )
        return metadata_encoder, ffmpeg_encoder

    metadata_encoder = RGBEncoderConfig(
        vcodec=args.vcodec,
        pix_fmt=args.pix_fmt,
        g=args.gop,
        crf=args.crf,
        preset=preset,
        fast_decode=args.fast_decode,
    )
    return metadata_encoder, None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a local LeRobot 0.6 image dataset to video format",
    )
    parser.add_argument("--source-root", type=Path, default=ROOT_PATH, help="Source dataset directory")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output dataset directory. Defaults to SOURCE_ROOT_video.",
    )
    parser.add_argument("--repo-id", help="Source repo id. Defaults to the source directory name.")
    parser.add_argument(
        "--output-repo-id",
        help="Output repo id. Defaults to REPO_ID_vid.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        help="Episode indices to convert. Defaults to all episodes.",
    )
    parser.add_argument("--num-workers", type=positive_integer, default=12)
    parser.add_argument("--max-episodes-per-batch", type=positive_integer)
    parser.add_argument("--max-frames-per-batch", type=positive_integer)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show the conversion configuration without writing output.",
    )

    rgb_group = parser.add_argument_group("RGB encoder")
    rgb_group.add_argument("--vcodec", default="hevc_nvenc")
    rgb_group.add_argument("--pix-fmt", default="yuv420p")
    rgb_group.add_argument("--gop", type=positive_integer, default=1)
    rgb_group.add_argument("--crf", type=float, default=22)
    rgb_group.add_argument(
        "--preset",
        type=optional_preset,
        default="p6",
        help="Encoder preset. Auto selects veryfast for CPU or p5 for NVENC.",
    )
    rgb_group.add_argument("--fast-decode", type=int, choices=range(0, 3), default=0)
    rgb_group.add_argument(
        "--ffmpeg-path",
        default="ffmpeg",
        help="System FFmpeg executable used by NVENC codecs.",
    )
    rgb_group.add_argument("--nvenc-gpu", type=int, default=0)
    rgb_group.add_argument(
        "--nvenc-tune",
        choices=("hq", "ll", "ull", "lossless"),
        default="hq",
    )

    depth_group = parser.add_argument_group("Depth encoder")
    depth_group.add_argument("--depth-vcodec", default="hevc")
    depth_group.add_argument("--depth-pix-fmt", default="gray12le")
    depth_group.add_argument("--depth-gop", type=positive_integer, default=2)
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
        else source_root.with_name(f"{source_root.name}_vid")
    )
    source_info = load_info(source_root)
    selected_episodes = validate_episode_indices(
        args.episodes,
        int(source_info["total_episodes"]),
    )
    rgb_encoder, ffmpeg_rgb_encoder = build_rgb_encoder_configs(args)

    if args.dry_run:
        installed_version = validate_lerobot_version()
        LOGGER.info("LeRobot version: %s", installed_version)
        LOGGER.info("Source root: %s", source_root)
        LOGGER.info("Output root: %s", output_root)
        LOGGER.info("Episodes: %s", selected_episodes if selected_episodes is not None else "all")
        if ffmpeg_rgb_encoder is not None:
            LOGGER.info(
                "RGB encoder: backend=system_ffmpeg codec=%s pix_fmt=%s g=%d qp=%s "
                "preset=%s tune=%s gpu=%d",
                ffmpeg_rgb_encoder.vcodec,
                ffmpeg_rgb_encoder.pix_fmt,
                ffmpeg_rgb_encoder.g,
                ffmpeg_rgb_encoder.qp,
                ffmpeg_rgb_encoder.preset,
                ffmpeg_rgb_encoder.tune,
                ffmpeg_rgb_encoder.gpu,
            )
        else:
            LOGGER.info("RGB encoder: backend=pyav config=%s", rgb_encoder)
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
        source_repo_id=args.repo_id,
        output_repo_id=args.output_repo_id,
        episode_indices=selected_episodes,
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
