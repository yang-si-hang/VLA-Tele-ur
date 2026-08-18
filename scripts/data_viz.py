# 直接调用 lerobot 内置的 visualize_dataset 函数进行可视化

import gc
import logging
import time
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import tqdm
from lerobot.configs import DEPTH_MILLIMETER_UNIT
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import to_hwc_float32_numpy, to_hwc_uint8_numpy
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD, SUCCESS

from utils.const import *
from utils.dataset_utils import validate_local_dataset_root_and_repo_id


# ====== 在这里直接指定你的配置 ======
REPO_ID:str = "pick_20260817_220807"             # local不需要, 但需要有值
EPISODE_INDICES = list(range(0, 10, 1))
# ROOT = DATA_PATH / "deploy" / REPO_ID
ROOT = DATA_PATH / REPO_ID
OUTPUT_DIR = OUTPUT_PATH / "viz_output"      # 仅当 save=True 时使用
SAVE = False                                # 设为 True 会保存 .rrd 文件而不弹窗
MODE = "local"                              # 可选: "local" 或 "distant"
BATCH_SIZE = 32
NUM_WORKERS = 12
TOLERANCE_S = 1e-4
DISPLAY_COMPRESSED_IMAGES = True
DISPLAY_DEBUG_FEATURES = True


def get_feature_names(dataset: LeRobotDataset, key: str) -> list[str]:
    feature = dataset.features[key]
    dimension = feature["shape"][-1]
    names = feature.get("names")
    if isinstance(names, list) and len(names) == dimension:
        return [str(name) for name in names]
    return [f"dim_{index}" for index in range(dimension)]


def log_vector(root: str, values, names: list[str] | None = None) -> None:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.asarray(values).reshape(-1)
    if names is None or len(names) != len(values):
        names = [f"dim_{index}" for index in range(len(values))]

    for name, value in zip(names, values, strict=True):
        rr.log(f"{root}/{name}", rr.Scalars(float(value)))


def get_debug_feature_keys(dataset: LeRobotDataset) -> list[str]:
    return [key for key in dataset.features if key.startswith("debug.")]


def debug_feature_entity_path(key: str) -> str:
    return key.replace(".", "/")


def build_blueprint_from_dataset(
    dataset: LeRobotDataset,
    display_debug_features: bool = False,
) -> rrb.Blueprint:
    views = [rrb.Spatial2DView(origin=key, name=key) for key in dataset.meta.camera_keys]
    for root, key in ((ACTION, ACTION), ("state", OBS_STATE)):
        if key in dataset.features:
            views.append(rrb.TimeSeriesView(origin=root, name=root))
    for key in (DONE, REWARD, SUCCESS):
        if key in dataset.features:
            views.append(rrb.TimeSeriesView(origin=key, name=key))
    if display_debug_features:
        for key in get_debug_feature_keys(dataset):
            entity_path = debug_feature_entity_path(key)
            views.append(rrb.TimeSeriesView(origin=entity_path, name=key))
    return rrb.Blueprint(rrb.Grid(*views))


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int | None = None,
    grpc_port: int = 9876,
    save: bool = False,
    output_dir: Path | None = None,
    display_compressed_images: bool = False,
    display_debug_features: bool = False,
) -> Path | None:
    if save and output_dir is None:
        raise ValueError("output_dir is required when save=True.")
    if mode not in ("local", "distant"):
        raise ValueError(f"Unsupported mode: {mode}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )
    rr.init(
        f"{dataset.repo_id}/episode_{episode_index}",
        spawn=mode == "local" and not save,
        default_blueprint=build_blueprint_from_dataset(dataset, display_debug_features),
    )
    gc.collect()

    if mode == "distant":
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        logging.info(f"Connect to a Rerun Server: rerun rerun+http://IP:{grpc_port}/proxy")
        rr.serve_web_viewer(
            open_browser=False,
            web_port=web_port or 9090,
            connect_to=server_uri,
        )

    action_names = get_feature_names(dataset, ACTION) if ACTION in dataset.features else None
    state_names = get_feature_names(dataset, OBS_STATE) if OBS_STATE in dataset.features else None
    debug_feature_keys = get_debug_feature_keys(dataset) if display_debug_features else []
    debug_feature_names = {key: get_feature_names(dataset, key) for key in debug_feature_keys}
    depth_meter = 1000.0 if dataset.depth_output_unit == DEPTH_MILLIMETER_UNIT else 1.0
    depth_ranges = {}
    for key in dataset.meta.depth_keys:
        stats = (dataset.meta.stats or {}).get(key)
        if stats:
            lower = stats["q01"] if "q01" in stats else stats["min"]
            upper = stats["q99"] if "q99" in stats else stats["max"]
            depth_ranges[key] = (float(np.asarray(lower).item()), float(np.asarray(upper).item()))

    first_index = None
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        if first_index is None:
            first_index = batch["index"][0].item()

        for index in range(len(batch["index"])):
            rr.set_time("frame_index", sequence=batch["index"][index].item() - first_index)
            rr.set_time("timestamp", timestamp=batch["timestamp"][index].item())

            for key in dataset.meta.camera_keys:
                if key in dataset.meta.depth_keys:
                    depth = to_hwc_float32_numpy(batch[key][index])
                    rr.log(
                        key,
                        rr.DepthImage(
                            depth,
                            meter=depth_meter,
                            colormap=rr.components.Colormap.Viridis,
                            depth_range=depth_ranges.get(key),
                        ),
                    )
                else:
                    image = rr.Image(to_hwc_uint8_numpy(batch[key][index]))
                    rr.log(key, image.compress() if display_compressed_images else image)

            if ACTION in batch:
                log_vector(ACTION, batch[ACTION][index], action_names)
            if OBS_STATE in batch:
                log_vector("state", batch[OBS_STATE][index], state_names)
            for key in (DONE, REWARD, SUCCESS):
                if key in batch:
                    rr.log(key, rr.Scalars(batch[key][index].item()))
            for key in debug_feature_keys:
                if key in batch:
                    log_vector(
                        debug_feature_entity_path(key),
                        batch[key][index],
                        debug_feature_names[key],
                    )

    if mode == "local" and save:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{dataset.repo_id.replace('/', '_')}_episode_{episode_index}.rrd"
        rr.save(output_path)
        return output_path

    if mode == "distant":
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")
    return None


def load_episode_dataset(episode_index: int) -> LeRobotDataset:
    return LeRobotDataset(
        repo_id=REPO_ID,
        episodes=[episode_index],
        root=ROOT,
        video_backend="pyav"
    )


if __name__ == "__main__":
    validate_local_dataset_root_and_repo_id(ROOT, REPO_ID)

    for episode_index in EPISODE_INDICES:
        print(f"Loading episode {episode_index} from {ROOT}...")
        dataset = load_episode_dataset(episode_index)
        print(f"Episode {episode_index} loaded with {len(dataset)} samples.")
        print(f"Checking first sample: {dataset[0].keys()}")

        visualize_dataset(
            dataset=dataset,
            episode_index=episode_index,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            mode=MODE,
            save=SAVE,
            output_dir=OUTPUT_DIR if SAVE else None,
            display_compressed_images=DISPLAY_COMPRESSED_IMAGES,
            display_debug_features=DISPLAY_DEBUG_FEATURES,
        )


# rerun --connect rerun+http://127.0.0.1:9876/proxy     # lerobot ssh调用数据显示

# Note:
