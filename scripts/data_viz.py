# 直接调用 lerobot 内置的 visualize_dataset 函数进行可视化

from pathlib import Path

import numpy as np
import pandas as pd
import rerun as rr
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import visualize_dataset

from utils.const import *
from utils.dataset_utils import validate_local_dataset_root_and_repo_id


# ====== 在这里直接指定你的配置 ======
REPO_ID:str = "pick_20260725_174915_vid"             # local不需要, 但需要有值
EPISODE_INDICES = list(range(0, 5, 1))
ROOT = DATA_PATH / REPO_ID
# ROOT = OUTPUT_DIR / "inference" / "lapar" / "act_20260310_192829"
OUTPUT_DIR = OUTPUT_PATH / "viz_output"      # 仅当 save=True 时使用
SAVE = False                                # 设为 True 会保存 .rrd 文件而不弹窗
MODE = "local"                              # 可选: "local" 或 "distant"
BATCH_SIZE = 32
NUM_WORKERS = 12
TOLERANCE_S = 1e-4
DISPLAY_COMPRESSED_IMAGES = True
RERUN_TIMELINE = "frame_index"              # timestamp or frame_index

# 将 observation.environment_state 写入 dataframe, 需要手动从时间轴中选择
ENABLE_ENVIRONMENT_STATE_DATAFRAME = False
ENVIRONMENT_STATE_FEATURE = "observation.environment_state"
ENVIRONMENT_STATE_ENTITY_PATH = "environment_state"

# 额外写入 gripper 相关 time series
ENABLE_GRIPPER_SERIES = False
GRIPPER_SERIES_FEATURE = "observation.state"
GRIPPER_SERIES_DIM_NAMES = ["gripper_tool3.pos"]
GRIPPER_SERIES_ENTITY_PREFIX = "gripper"

# 额外写入 pose_tool3.pos.x/y/z 到同一张 time series 图
ENABLE_POSE_TOOL3_POS_SERIES = False
POSE_TOOL3_POS_FEATURE = "observation.state"
POSE_TOOL3_POS_DIM_NAMES = [
    "pose_tool3.pos.x",
    "pose_tool3.pos.y",
    "pose_tool3.pos.z",
]
POSE_TOOL3_POS_ENTITY_PREFIX = "pose_tool3_pos"


def set_sample_time(sample: dict, first_index: int) -> None:
    sample_index = int(sample["index"])
    rr.reset_time()
    if RERUN_TIMELINE == "timestamp":
        rr.set_time(RERUN_TIMELINE, timestamp=float(sample["timestamp"]))
    else:
        rr.set_time(RERUN_TIMELINE, sequence=sample_index - first_index)


def get_feature_dim_names(dataset: LeRobotDataset, feature: str) -> list[str]:
    feature_info = dataset.meta.info["features"].get(feature)
    if feature_info is None:
        raise KeyError(f"Feature {feature!r} not found in dataset metadata.")

    names = list(feature_info.get("names") or [])
    if names:
        return names
    return [f"dim_{i}" for i in range(int(feature_info["shape"][0]))]


def resolve_feature_dim(dataset: LeRobotDataset, feature: str, dim_name: str) -> tuple[int, str]:
    names = get_feature_dim_names(dataset, feature)
    if dim_name not in names:
        raise ValueError(f"Dimension {dim_name!r} not found in {feature}. Available names: {names}")
    return names.index(dim_name), dim_name


def resolve_feature_dims(dataset: LeRobotDataset, feature: str, dim_names: list[str]) -> list[tuple[int, str]]:
    return [resolve_feature_dim(dataset, feature, dim_name) for dim_name in dim_names]


def feature_value_vector(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def log_scalar_series_to_rerun(
    dataset: LeRobotDataset,
    feature: str,
    dim_names: list[str],
    entity_prefix: str,
) -> None:
    dim_specs = resolve_feature_dims(dataset, feature, dim_names)
    first_index = None

    for sample in dataset.hf_dataset:
        sample_index = int(sample["index"])
        if first_index is None:
            first_index = sample_index
        set_sample_time(sample, first_index)

        values = feature_value_vector(sample[feature])
        for dim_index, dim_label in dim_specs:
            rr.log(f"{entity_prefix}/{dim_label}", rr.Scalars(float(values[dim_index])))


def build_feature_dataframe(dataset: LeRobotDataset, feature: str) -> pd.DataFrame:
    dim_names = get_feature_dim_names(dataset, feature)
    rows = []
    first_index = None

    for sample in dataset.hf_dataset:
        sample_index = int(sample["index"])
        if first_index is None:
            first_index = sample_index

        row = {
            "rerun_index": sample_index - first_index,
            "index": sample_index,
            "episode_index": int(sample["episode_index"]),
            "frame_index": int(sample["frame_index"]),
            "timestamp": float(sample["timestamp"]),
        }
        values = feature_value_vector(sample[feature])
        for dim_index, dim_name in enumerate(dim_names):
            row[dim_name] = float(values[dim_index])
        rows.append(row)

    return pd.DataFrame(rows)


def log_dataframe_to_rerun(df: pd.DataFrame, entity_path: str) -> None:
    if df.empty:
        return

    if RERUN_TIMELINE == "timestamp":
        indexes = [rr.TimeColumn(RERUN_TIMELINE, timestamp=df["timestamp"].tolist())]
    else:
        indexes = [rr.TimeColumn(RERUN_TIMELINE, sequence=df["rerun_index"].astype(int).tolist())]

    component_df = df.drop(columns=["rerun_index"])
    rr.send_columns(
        entity_path,
        indexes=indexes,
        columns=rr.AnyValues.columns(**component_df.to_dict(orient="list")),
    )


def log_environment_state_dataframe_to_rerun(dataset: LeRobotDataset) -> pd.DataFrame:
    df = build_feature_dataframe(dataset, ENVIRONMENT_STATE_FEATURE)
    log_dataframe_to_rerun(df, ENVIRONMENT_STATE_ENTITY_PATH)
    return df


def load_episode_dataset(episode_index: int) -> LeRobotDataset:
    return LeRobotDataset(
        repo_id=REPO_ID,
        episodes=[episode_index],
        root=ROOT,
        video_backend="pyav"
    )


if __name__ == "__main__":
    if MODE == "distant" and (
        ENABLE_GRIPPER_SERIES
        or ENABLE_ENVIRONMENT_STATE_DATAFRAME
        or ENABLE_POSE_TOOL3_POS_SERIES
    ):
        raise ValueError("MODE='distant' blocks inside visualize_dataset, so extra rerun logging cannot run after it.")

    validate_local_dataset_root_and_repo_id(ROOT, REPO_ID)

    for episode_index in EPISODE_INDICES:
        print(f"Loading episode {episode_index} from {ROOT}...")
        dataset = load_episode_dataset(episode_index)
        print(f"Episode {episode_index} loaded with {len(dataset)} samples.")
        print(f"Checking first sample: {dataset[0].keys()}")

        rrd_path = visualize_dataset(
            dataset=dataset,
            episode_index=episode_index,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            mode=MODE,
            save=SAVE,
            output_dir=OUTPUT_DIR if SAVE else None,
            display_compressed_images=DISPLAY_COMPRESSED_IMAGES,
        )

        if ENABLE_GRIPPER_SERIES:
            log_scalar_series_to_rerun(
                dataset=dataset,
                feature=GRIPPER_SERIES_FEATURE,
                dim_names=GRIPPER_SERIES_DIM_NAMES,
                entity_prefix=GRIPPER_SERIES_ENTITY_PREFIX,
            )
            print(f"Logged gripper series for episode {episode_index}: {GRIPPER_SERIES_DIM_NAMES}")

        if ENABLE_POSE_TOOL3_POS_SERIES:
            log_scalar_series_to_rerun(
                dataset=dataset,
                feature=POSE_TOOL3_POS_FEATURE,
                dim_names=POSE_TOOL3_POS_DIM_NAMES,
                entity_prefix=POSE_TOOL3_POS_ENTITY_PREFIX,
            )
            print(f"Logged pose_tool3.pos series for episode {episode_index}: {POSE_TOOL3_POS_FEATURE}")

        if ENABLE_ENVIRONMENT_STATE_DATAFRAME:
            environment_state_df = log_environment_state_dataframe_to_rerun(dataset)
            print(
                f"Logged environment state dataframe for episode {episode_index}: "
                f"{ENVIRONMENT_STATE_FEATURE}, rows={len(environment_state_df)}"
            )

        if (
            ENABLE_GRIPPER_SERIES
            or ENABLE_ENVIRONMENT_STATE_DATAFRAME
            or ENABLE_POSE_TOOL3_POS_SERIES
        ):
            if SAVE and rrd_path is not None:
                rr.save(rrd_path)
                print(f"Updated saved rerun file with extra series: {rrd_path}")


# rerun --connect rerun+http://127.0.0.1:9876/proxy     # lerobot ssh调用数据显示

# Note:
