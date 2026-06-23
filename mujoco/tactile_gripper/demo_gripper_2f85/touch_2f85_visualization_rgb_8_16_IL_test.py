import argparse
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
import sys

import cv2
import numpy as np
import matplotlib
import h5py
import mujoco

# 训练环境里通常没有可用的 X11/Qt 图形窗口；强制切到非交互式后端，避免保存曲线时崩溃。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# 保证同目录下的 RL 脚本可以被稳定导入。
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# 直接复用 RL 脚本里的环境与启发式策略，避免重复维护同一套 MuJoCo 任务逻辑。
from touch_2f85_visualization_rgb_8_16_RL_test import (
    TASK_STAGE_AUTO,
    TASK_STAGE_FULL,
    TASK_STAGE_GRASP,
    TASK_STAGE_PLACE,
    Tactile2F85PickPlaceEnv,
    heuristic_action,
    resolve_task_stage,
)

logging.getLogger().setLevel(logging.ERROR)


# -----------------------------
# 默认路径与文件命名
# -----------------------------
DEFAULT_RUN_DIR = Path("runs/il_2f85")
DEFAULT_DATASET_DIR = DEFAULT_RUN_DIR / "dataset"
DEFAULT_MODEL_PATH = DEFAULT_RUN_DIR / "chunked_bc.pt"
DEFAULT_CONTEXT_LEN = 8
DEFAULT_CHUNK_LEN = 8
EPISODE_FILE_SUFFIX = ".hdf5"
DEFAULT_RGB_WIDTH = 640
DEFAULT_RGB_HEIGHT = 480
DEFAULT_RGB_TRAIN_WIDTH = 128
DEFAULT_RGB_TRAIN_HEIGHT = 96
OBS_CONTINUOUS_DIM = 30
OBS_TOTAL_DIM = 35
ACTION_DIM = 5
TACTILE_DIM = 2
PHASE_DIM = 5
CAMERA_NAMES = ("top_cam", "gripper_cam")
TACTILE_START = OBS_CONTINUOUS_DIM - TACTILE_DIM - 1
TACTILE_END = TACTILE_START + TACTILE_DIM
PHASE_START = OBS_CONTINUOUS_DIM


# -----------------------------
# 数据结构
# -----------------------------
@dataclass
class EpisodeData:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    images: dict[str, np.ndarray]
    success: bool
    seed: int
    task_stage: str
    source: str


# -----------------------------
# 通用工具
# -----------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def episode_file_path(dataset_dir: Path, index: int) -> Path:
    return dataset_dir / f"episode_{index:05d}{EPISODE_FILE_SUFFIX}"


def episode_meta_path(episode_path: Path) -> Path:
    return episode_path.with_suffix(".json")


def _decode_attr(value, default=None):
    if value is None:
        return default
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O"}:
        return [item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item) for item in value.tolist()]
    return value


def _stack_episode_images(images: dict[str, np.ndarray], camera_names: tuple[str, ...]) -> np.ndarray:
    stacked = []
    for name in camera_names:
        if name not in images:
            raise KeyError(f"Missing camera frames for {name}")
        stacked.append(images[name])
    return np.stack(stacked, axis=1)


def _pad_rgb_context(frames: np.ndarray, t: int, context_len: int) -> np.ndarray:
    start = max(0, t - context_len + 1)
    context = frames[start : t + 1]
    if len(context) < context_len:
        pad_count = context_len - len(context)
        pad_frame = context[0:1] if len(context) else frames[0:1]
        pad = np.repeat(pad_frame, pad_count, axis=0)
        context = np.concatenate([pad, context], axis=0)
    return context.astype(np.float32)


def _resize_rgb_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    interpolation = cv2.INTER_AREA if width < frame.shape[1] or height < frame.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def _display_rgb_frames(frame_map: dict[str, np.ndarray], step_index: int, window_prefix: str) -> None:
    panels = []
    for camera_name in CAMERA_NAMES:
        frames = frame_map.get(camera_name)
        if frames is None or len(frames) == 0:
            continue
        frame = frames[min(step_index, len(frames) - 1)]
        panels.append((camera_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))

    if not panels:
        return

    for camera_name, panel in panels:
        cv2.imshow(f"{window_prefix} {camera_name}", panel)
    cv2.waitKey(1)


def _create_rgb_renderer(model, width: int, height: int) -> mujoco.Renderer:
    return mujoco.Renderer(model, width=width, height=height)


def _capture_rgb_frames(renderer: mujoco.Renderer, data: mujoco.MjData, camera_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    for camera_name in camera_names:
        renderer.update_scene(data, camera=camera_name)
        frames[camera_name] = renderer.render().astype(np.uint8)
    return frames


def save_episode(episode_path: Path, episode: EpisodeData) -> None:
    if episode_path.suffix.lower() in {".h5", ".hdf5"}:
        if episode.observations.ndim != 2 or episode.observations.shape[1] != OBS_TOTAL_DIM:
            raise ValueError(
                f"Expected observations with shape [T, {OBS_TOTAL_DIM}], got {episode.observations.shape}"
            )
        if episode.actions.ndim != 2 or episode.actions.shape[1] != ACTION_DIM:
            raise ValueError(f"Expected actions with shape [T, {ACTION_DIM}], got {episode.actions.shape}")
        if episode.rewards.ndim != 1 or len(episode.rewards) != len(episode.actions):
            raise ValueError("Rewards must be a 1D array aligned with actions.")

        continuous = episode.observations[:, :OBS_CONTINUOUS_DIM].astype(np.float32)
        tactile = episode.observations[:, TACTILE_START:TACTILE_END].astype(np.float32)
        phase = episode.observations[:, PHASE_START:].astype(np.float32)
        has_rgb = bool(episode.images)
        if has_rgb:
            for camera_name in CAMERA_NAMES:
                if camera_name not in episode.images:
                    raise KeyError(f"Missing RGB frames for camera: {camera_name}")
                if episode.images[camera_name].shape[0] != len(episode.actions):
                    raise ValueError(
                        f"RGB frames for {camera_name} must align with actions; "
                        f"got {episode.images[camera_name].shape[0]} frames and {len(episode.actions)} actions"
                    )

        with h5py.File(episode_path, "w") as f:
            f.attrs["format"] = "tactile_il_episode"
            f.attrs["format_version"] = 2
            f.attrs["success"] = bool(episode.success)
            f.attrs["seed"] = int(episode.seed)
            f.attrs["task_stage"] = episode.task_stage
            f.attrs["source"] = episode.source
            f.attrs["num_steps"] = int(len(episode.actions))
            f.attrs["saved_at"] = float(time.time())
            f.attrs["has_rgb"] = has_rgb
            f.attrs["camera_names"] = np.array(CAMERA_NAMES, dtype=h5py.string_dtype(encoding="utf-8"))
            f.attrs["rgb_width"] = int(episode.images[CAMERA_NAMES[0]].shape[2]) if has_rgb else 0
            f.attrs["rgb_height"] = int(episode.images[CAMERA_NAMES[0]].shape[1]) if has_rgb else 0

            obs_group = f.create_group("observations")
            obs_group.create_dataset(
                "state",
                data=episode.observations.astype(np.float32),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            obs_group.create_dataset(
                "proprio",
                data=continuous,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            tactile_group = obs_group.create_group("tactile")
            tactile_group.create_dataset(
                "right_max",
                data=tactile[:, 0:1],
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            tactile_group.create_dataset(
                "left_max",
                data=tactile[:, 1:2],
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            obs_group.create_dataset(
                "phase",
                data=phase,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            action_group = f.create_group("action")
            action_group.create_dataset(
                "command",
                data=episode.actions.astype(np.float32),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            if has_rgb:
                images_group = obs_group.create_group("images")
                for camera_name in CAMERA_NAMES:
                    images_group.create_dataset(
                        camera_name,
                        data=episode.images[camera_name].astype(np.uint8),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )
            reward_group = f.create_group("rewards")
            reward_group.create_dataset(
                "value",
                data=episode.rewards.astype(np.float32),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
        return

    np.savez_compressed(
        episode_path,
        observations=episode.observations.astype(np.float32),
        actions=episode.actions.astype(np.float32),
        rewards=episode.rewards.astype(np.float32),
    )
    meta = {
        "success": bool(episode.success),
        "seed": int(episode.seed),
        "task_stage": episode.task_stage,
        "source": episode.source,
        "num_steps": int(len(episode.actions)),
        "saved_at": time.time(),
        "format": "npz",
    }
    with episode_meta_path(episode_path).open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def load_episode_from_hdf5(episode_path: Path) -> EpisodeData:
    with h5py.File(episode_path, "r") as f:
        if "observations" in f and "state" in f["observations"]:
            observations = f["observations"]["state"][:].astype(np.float32)
        else:
            proprio = f["observations"]["proprio"][:].astype(np.float32)
            phase = f["observations"]["phase"][:].astype(np.float32)
            observations = np.concatenate([proprio, phase], axis=1).astype(np.float32)

        if "action" in f and "command" in f["action"]:
            actions = f["action"]["command"][:].astype(np.float32)
        else:
            actions = f["actions"][:].astype(np.float32)

        if "rewards" in f and "value" in f["rewards"]:
            rewards = f["rewards"]["value"][:].astype(np.float32)
        else:
            rewards = f["rewards"][:].astype(np.float32)

        images: dict[str, np.ndarray] = {}
        if "observations" in f and "images" in f["observations"]:
            for camera_name in f["observations"]["images"].keys():
                images[camera_name] = f["observations"]["images"][camera_name][:].astype(np.uint8)

        success = bool(_decode_attr(f.attrs.get("success", False), False))
        seed = int(_decode_attr(f.attrs.get("seed", 0), 0))
        task_stage = str(_decode_attr(f.attrs.get("task_stage", TASK_STAGE_FULL), TASK_STAGE_FULL))
        source = str(_decode_attr(f.attrs.get("source", "unknown"), "unknown"))

    return EpisodeData(
        observations=observations,
        actions=actions,
        rewards=rewards,
        images=images,
        success=success,
        seed=seed,
        task_stage=task_stage,
        source=source,
    )


def load_episode(episode_path: Path) -> EpisodeData:
    meta_path = episode_meta_path(episode_path)
    if episode_path.suffix.lower() in {".h5", ".hdf5"}:
        episode = load_episode_from_hdf5(episode_path)
    else:
        with np.load(episode_path, allow_pickle=False) as data:
            observations = data["observations"].astype(np.float32)
            actions = data["actions"].astype(np.float32)
            rewards = data["rewards"].astype(np.float32)

        meta = {}
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

        episode = EpisodeData(
            observations=observations,
            actions=actions,
            rewards=rewards,
            images={},
            success=bool(meta.get("success", False)),
            seed=int(meta.get("seed", 0)),
            task_stage=str(meta.get("task_stage", TASK_STAGE_FULL)),
            source=str(meta.get("source", "unknown")),
        )

    return episode


def resize_episode_images(
    episodes: Iterable[EpisodeData],
    resize_width: int,
    resize_height: int,
    camera_names: tuple[str, ...],
) -> None:
    for episode in episodes:
        if not episode.images:
            continue
        resized_images: dict[str, np.ndarray] = {}
        for camera_name in camera_names:
            frames = episode.images.get(camera_name)
            if frames is None:
                raise KeyError(f"Missing RGB frames for camera {camera_name}")
            resized_images[camera_name] = np.stack(
                [_resize_rgb_frame(frame.astype(np.uint8), resize_width, resize_height) for frame in frames],
                axis=0,
            ).astype(np.uint8)
        episode.images = resized_images


def load_trusted_checkpoint(path: Path, device: str):
    # 这些 checkpoint 都是本地训练脚本生成的，需显式关闭 weights_only，避免 PyTorch 2.6 拦截。
    return torch.load(path, map_location=device, weights_only=False)


def save_loss_curve(output_dir: Path, train_losses: list[float], val_losses: list[float]) -> None:
    # 保存训练/验证 loss 曲线，便于判断是欠拟合、过拟合还是数据问题。
    ensure_dir(output_dir)
    epochs = np.arange(1, len(train_losses) + 1, dtype=np.int32)

    np.savetxt(
        output_dir / "loss_curve.csv",
        np.column_stack([epochs, np.asarray(train_losses, dtype=np.float32), np.asarray(val_losses, dtype=np.float32)]),
        delimiter=",",
        header="epoch,train_loss,val_loss",
        comments="",
    )

    fig = plt.figure(figsize=(8, 4.5), dpi=160)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, train_losses, label="train_loss", linewidth=2)
    ax.plot(epochs, val_losses, label="val_loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Imitation Learning Loss Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png")
    plt.close(fig)


def list_episode_files(dataset_dir: Path) -> list[Path]:
    files = list(dataset_dir.glob("episode_*.hdf5"))
    files.extend(dataset_dir.glob("episode_*.h5"))
    files.extend(dataset_dir.glob("episode_*.npz"))
    return sorted(files)


def split_episodes(episodes: list[EpisodeData], val_ratio: float, seed: int) -> tuple[list[EpisodeData], list[EpisodeData]]:
    if not episodes:
        return [], []

    rng = np.random.default_rng(seed)
    indices = np.arange(len(episodes))
    rng.shuffle(indices)

    val_count = max(1, int(round(len(episodes) * val_ratio))) if len(episodes) > 1 else 0
    val_indices = set(indices[:val_count].tolist())
    train_eps = [ep for i, ep in enumerate(episodes) if i not in val_indices]
    val_eps = [ep for i, ep in enumerate(episodes) if i in val_indices]
    if not train_eps:
        train_eps, val_eps = episodes[:-1], episodes[-1:]
    return train_eps, val_eps


def compute_obs_stats(episodes: Iterable[EpisodeData]) -> tuple[np.ndarray, np.ndarray]:
    all_obs = np.concatenate([ep.observations for ep in episodes if len(ep.observations) > 0], axis=0)
    mean = all_obs.mean(axis=0)
    std = all_obs.std(axis=0)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def compute_action_stats(episodes: Iterable[EpisodeData]) -> tuple[np.ndarray, np.ndarray]:
    all_actions = np.concatenate([ep.actions for ep in episodes if len(ep.actions) > 0], axis=0)
    mean = all_actions.mean(axis=0)
    std = all_actions.std(axis=0)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def pad_context(observations: np.ndarray, t: int, context_len: int) -> np.ndarray:
    start = max(0, t - context_len + 1)
    context = observations[start : t + 1]
    if len(context) < context_len:
        pad_count = context_len - len(context)
        pad_frame = context[0:1] if len(context) else observations[0:1]
        pad = np.repeat(pad_frame, pad_count, axis=0)
        context = np.concatenate([pad, context], axis=0)
    return context.astype(np.float32)


def pad_action_chunk(actions: np.ndarray, t: int, chunk_len: int) -> tuple[np.ndarray, np.ndarray]:
    start = t
    end = min(t + chunk_len, len(actions))
    chunk = actions[start:end]
    mask = np.zeros(chunk_len, dtype=np.float32)
    mask[: len(chunk)] = 1.0
    if len(chunk) < chunk_len:
        pad_action = chunk[-1: ] if len(chunk) else actions[-1:]
        pad = np.repeat(pad_action, chunk_len - len(chunk), axis=0)
        chunk = np.concatenate([chunk, pad], axis=0) if len(chunk) else pad
    return chunk.astype(np.float32), mask


def infer_rgb_shape(episodes: Iterable[EpisodeData], camera_names: tuple[str, ...]) -> tuple[int, int, int]:
    for episode in episodes:
        if not episode.images:
            continue
        for camera_name in camera_names:
            if camera_name in episode.images and len(episode.images[camera_name]) > 0:
                sample = episode.images[camera_name][0]
                if sample.ndim != 3 or sample.shape[-1] != 3:
                    raise ValueError(f"Expected RGB frame for {camera_name}, got {sample.shape}")
                return int(sample.shape[0]), int(sample.shape[1]), int(sample.shape[2])
    raise ValueError("Unable to infer RGB shape from dataset; no camera frames were found.")


def pad_rgb_chunk(
    images: dict[str, np.ndarray],
    t: int,
    context_len: int,
    camera_names: tuple[str, ...],
    resize_width: int | None = None,
    resize_height: int | None = None,
) -> np.ndarray:
    chunks = []
    for camera_name in camera_names:
        frames = images.get(camera_name)
        if frames is None:
            raise KeyError(f"Missing RGB frames for camera {camera_name}")
        context = _pad_rgb_context(frames, t, context_len)
        if resize_width is not None and resize_height is not None:
            context = np.stack(
                [_resize_rgb_frame(frame.astype(np.uint8), resize_width, resize_height) for frame in context],
                axis=0,
            )
        chunks.append(context)
    stacked = np.stack(chunks, axis=1)  # [T, Cams, H, W, 3]
    stacked = np.transpose(stacked, (0, 1, 4, 2, 3))  # [T, Cams, 3, H, W]
    return stacked.astype(np.float32) / 255.0


def build_rgb_context(
    rgb_buffers: dict[str, deque[np.ndarray]],
    context_len: int,
    camera_names: tuple[str, ...],
    resize_width: int | None = None,
    resize_height: int | None = None,
) -> np.ndarray:
    stacked = []
    for camera_name in camera_names:
        frames = np.stack(list(rgb_buffers[camera_name]), axis=0)
        if resize_width is not None and resize_height is not None:
            frames = np.stack(
                [_resize_rgb_frame(frame.astype(np.uint8), resize_width, resize_height) for frame in frames],
                axis=0,
            )
        stacked.append(frames)
    rgb_context = np.stack(stacked, axis=1).astype(np.float32) / 255.0
    return np.transpose(rgb_context, (0, 1, 4, 2, 3))


# -----------------------------
# 动作 chunk 策略
# -----------------------------
class TemporalBCPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        context_len: int,
        chunk_len: int,
        use_rgb: bool = False,
        camera_names: tuple[str, ...] = CAMERA_NAMES,
        rgb_height: int = DEFAULT_RGB_HEIGHT,
        rgb_width: int = DEFAULT_RGB_WIDTH,
        rgb_feature_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.context_len = context_len
        self.chunk_len = chunk_len
        self.use_rgb = use_rgb
        self.camera_names = tuple(camera_names)
        self.rgb_height = int(rgb_height)
        self.rgb_width = int(rgb_width)
        self.rgb_feature_dim = int(rgb_feature_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        if self.use_rgb:
            self.rgb_encoders = nn.ModuleDict(
                {
                    camera_name: nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(128, self.rgb_feature_dim),
                        nn.GELU(),
                    )
                    for camera_name in self.camera_names
                }
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim + len(self.camera_names) * self.rgb_feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        else:
            self.rgb_encoders = nn.ModuleDict()
            self.fusion = nn.Identity()
        self.pos_embed = nn.Parameter(torch.zeros(1, context_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, chunk_len * action_dim),
        )

    def forward(self, obs_seq: torch.Tensor, rgb_seq: torch.Tensor | None = None) -> torch.Tensor:
        # obs_seq shape: [B, T, D]
        if obs_seq.shape[1] != self.context_len:
            raise ValueError(f"Expected context length {self.context_len}, got {obs_seq.shape[1]}")
        features = self.obs_encoder(obs_seq)

        if self.use_rgb:
            if rgb_seq is None:
                raise ValueError("RGB input is required for this policy.")
            if rgb_seq.shape[1] != self.context_len:
                raise ValueError(f"Expected RGB context length {self.context_len}, got {rgb_seq.shape[1]}")
            if rgb_seq.shape[2] != len(self.camera_names):
                raise ValueError(f"Expected {len(self.camera_names)} cameras, got {rgb_seq.shape[2]}")

            rgb_feats = []
            for cam_idx, camera_name in enumerate(self.camera_names):
                cam_rgb = rgb_seq[:, :, cam_idx]  # [B, T, 3, H, W]
                cam_rgb = cam_rgb.reshape(-1, cam_rgb.shape[2], cam_rgb.shape[3], cam_rgb.shape[4])
                cam_feat = self.rgb_encoders[camera_name](cam_rgb)
                cam_feat = cam_feat.view(obs_seq.shape[0], self.context_len, -1)
                rgb_feats.append(cam_feat)

            rgb_features = torch.cat(rgb_feats, dim=-1)
            features = self.fusion(torch.cat([features, rgb_features], dim=-1))

        features = features + self.pos_embed
        encoded = self.transformer(features)
        pooled = encoded[:, -1, :]
        chunk = self.head(pooled).view(obs_seq.shape[0], self.chunk_len, self.action_dim)
        return chunk


class EpisodeWindowDataset(Dataset):
    def __init__(
        self,
        episodes: list[EpisodeData],
        context_len: int,
        chunk_len: int,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        act_mean: np.ndarray,
        act_std: np.ndarray,
        use_rgb: bool,
        camera_names: tuple[str, ...],
        rgb_width: int | None = None,
        rgb_height: int | None = None,
        success_only: bool = False,
    ) -> None:
        self.episodes = [ep for ep in episodes if (ep.success or not success_only)]
        self.context_len = context_len
        self.chunk_len = chunk_len
        self.use_rgb = use_rgb
        self.camera_names = tuple(camera_names)
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = obs_std.astype(np.float32)
        self.act_mean = act_mean.astype(np.float32)
        self.act_std = act_std.astype(np.float32)
        self.rgb_width = int(rgb_width) if rgb_width is not None else None
        self.rgb_height = int(rgb_height) if rgb_height is not None else None
        self.samples: list[tuple[int, int]] = []

        for ep_idx, ep in enumerate(self.episodes):
            for t in range(len(ep.actions)):
                self.samples.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        ep_idx, t = self.samples[index]
        episode = self.episodes[ep_idx]
        context = pad_context(episode.observations, t, self.context_len)
        context = (context - self.obs_mean) / self.obs_std
        action_chunk, mask = pad_action_chunk(episode.actions, t, self.chunk_len)
        action_chunk = (action_chunk - self.act_mean) / self.act_std
        if self.use_rgb:
            if not episode.images:
                raise ValueError("RGB conditioning is enabled, but this episode does not contain images.")
            rgb_context = pad_rgb_chunk(
                episode.images,
                t,
                self.context_len,
                self.camera_names,
                resize_width=self.rgb_width,
                resize_height=self.rgb_height,
            )
            return (
                torch.from_numpy(context),
                torch.from_numpy(rgb_context),
                torch.from_numpy(action_chunk),
                torch.from_numpy(mask),
            )
        return (
            torch.from_numpy(context),
            torch.empty(0),
            torch.from_numpy(action_chunk),
            torch.from_numpy(mask),
        )


# -----------------------------
# 策略封装
# -----------------------------
class TemporalBCAgent:
    def __init__(
        self,
        model: TemporalBCPolicy,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        act_mean: np.ndarray,
        act_std: np.ndarray,
        device: str,
    ):
        self.model = model.to(device)
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = obs_std.astype(np.float32)
        self.act_mean = act_mean.astype(np.float32)
        self.act_std = act_std.astype(np.float32)
        self.device = torch.device(device)
        self.action_cache: deque[np.ndarray] = deque()

    def _predict_chunk(self, context: np.ndarray, rgb_context: np.ndarray | None = None) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(((context - self.obs_mean) / self.obs_std).astype(np.float32)).unsqueeze(0).to(self.device)
            rgb_tensor = None
            if rgb_context is not None:
                rgb_tensor = torch.from_numpy(rgb_context.astype(np.float32)).unsqueeze(0).to(self.device)
            chunk = self.model(tensor, rgb_tensor).squeeze(0).cpu().numpy()
        chunk = chunk * self.act_std + self.act_mean
        return np.clip(chunk, -1.0, 1.0).astype(np.float32)

    def act(self, context: np.ndarray, rgb_context: np.ndarray | None = None) -> np.ndarray:
        if not self.action_cache:
            chunk = self._predict_chunk(context, rgb_context)
            for step_action in chunk:
                self.action_cache.append(step_action)
        return self.action_cache.popleft()

    def save(self, path: Path, extra_meta: dict | None = None) -> None:
        ensure_dir(path.parent)
        payload = {
            "state_dict": self.model.state_dict(),
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
            "act_mean": self.act_mean,
            "act_std": self.act_std,
            "obs_dim": int(self.obs_mean.shape[0]),
            "action_dim": int(self.act_mean.shape[0]),
            "context_len": int(self.model.context_len),
            "chunk_len": int(self.model.chunk_len),
            "use_rgb": bool(self.model.use_rgb),
            "camera_names": list(self.model.camera_names),
            "rgb_height": int(self.model.rgb_height),
            "rgb_width": int(self.model.rgb_width),
            "rgb_feature_dim": int(self.model.rgb_feature_dim),
            "hidden_dim": int(self.model.hidden_dim),
            "num_layers": int(self.model.num_layers),
            "num_heads": int(self.model.num_heads),
        }
        if extra_meta:
            payload["meta"] = extra_meta
        torch.save(payload, path)

    @staticmethod
    def load(path: Path, model_kwargs: dict, device: str) -> "TemporalBCAgent":
        payload = load_trusted_checkpoint(path, device)
        model = TemporalBCPolicy(**model_kwargs)
        model.load_state_dict(payload["state_dict"])
        return TemporalBCAgent(model, payload["obs_mean"], payload["obs_std"], payload["act_mean"], payload["act_std"], device=device)


# -----------------------------
# 采集 / 回放
# -----------------------------
def make_policy_function(
    source: str,
    env: Tactile2F85PickPlaceEnv,
    agent: TemporalBCAgent | None,
) -> Callable[[Tactile2F85PickPlaceEnv, np.ndarray, np.ndarray | None], np.ndarray]:
    if source == "heuristic":
        return lambda _env, _context, _rgb_context=None: heuristic_action(_env).astype(np.float32)
    if source == "model":
        if agent is None:
            raise ValueError("model source requires a loaded agent")
        return lambda _env, context, rgb_context=None: agent.act(context, rgb_context)
    raise ValueError(f"Unsupported collection source: {source}")


def collect_one_episode(
    source: str,
    dataset_dir: Path,
    episode_index: int,
    task_stage: str,
    seed: int,
    max_steps: int,
    render: bool,
    context_len: int,
    rgb_width: int,
    rgb_height: int,
    camera_names: tuple[str, ...] = CAMERA_NAMES,
    agent: TemporalBCAgent | None = None,
    keep_failed: bool = False,
) -> tuple[EpisodeData, Path]:
    env = Tactile2F85PickPlaceEnv(render_mode="human" if render else None, max_episode_steps=max_steps, seed=seed, task_stage=task_stage)
    obs, _ = env.reset(seed=seed)
    renderer = _create_rgb_renderer(env.model, rgb_width, rgb_height)
    policy_fn = make_policy_function(source, env, agent)
    obs_buffer = deque([obs.copy()] * context_len, maxlen=context_len)
    rgb_buffers = {camera_name: deque([None] * context_len, maxlen=context_len) for camera_name in camera_names}
    if agent is not None:
        agent.action_cache.clear()

    observations = []
    actions = []
    rewards = []
    rgb_frames = {camera_name: [] for camera_name in camera_names}
    done = False
    info = {}
    try:
        current_rgb = _capture_rgb_frames(renderer, env.data, camera_names)
        for camera_name in camera_names:
            for _ in range(context_len):
                rgb_buffers[camera_name].append(current_rgb[camera_name].copy())

        rgb_resize_width = int(agent.model.rgb_width) if (agent is not None and agent.model.use_rgb) else None
        rgb_resize_height = int(agent.model.rgb_height) if (agent is not None and agent.model.use_rgb) else None

        while not done:
            context = np.stack(list(obs_buffer), axis=0)
            rgb_context = None
            if source == "model" and agent is not None and agent.model.use_rgb:
                rgb_context = build_rgb_context(
                    rgb_buffers,
                    context_len,
                    camera_names,
                    resize_width=rgb_resize_width,
                    resize_height=rgb_resize_height,
                )

            action = policy_fn(env, context, rgb_context)
            next_obs, reward, terminated, truncated, info = env.step(action)

            observations.append(obs.copy())
            actions.append(action.copy())
            rewards.append(float(reward))
            for camera_name in camera_names:
                rgb_frames[camera_name].append(current_rgb[camera_name].copy())

            obs_buffer.append(next_obs.copy())
            obs = next_obs
            done = terminated or truncated

            current_rgb = _capture_rgb_frames(renderer, env.data, camera_names)
            for camera_name in camera_names:
                rgb_buffers[camera_name].append(current_rgb[camera_name].copy())

            if render:
                time.sleep(1.0 / 60.0)
    finally:
        renderer.close()

    success = bool(info.get("is_success", False) or info.get("task_success", False))
    episode = EpisodeData(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        images={camera_name: np.asarray(rgb_frames[camera_name], dtype=np.uint8) for camera_name in camera_names},
        success=success,
        seed=seed,
        task_stage=task_stage,
        source=source,
    )

    episode_path = episode_file_path(dataset_dir, episode_index)
    if success or keep_failed:
        save_episode(episode_path, episode)
        print(f"[collect] saved {episode_path.name} success={success} steps={len(actions)}")
    else:
        print(f"[collect] skipped failed episode {episode_index:05d} steps={len(actions)}")
    env.close()
    return episode, episode_path


def replay_episode(
    episode_path: Path,
    render: bool,
    task_stage: str,
    max_steps: int,
) -> None:
    episode = load_episode(episode_path)
    env = Tactile2F85PickPlaceEnv(render_mode="human" if render else None, max_episode_steps=max_steps, seed=episode.seed, task_stage=resolve_task_stage(task_stage, None))
    obs, _ = env.reset(seed=episode.seed)

    total_reward = 0.0
    for t, action in enumerate(episode.actions):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if render and episode.images:
            _display_rgb_frames(episode.images, t, window_prefix="[replay]")
        if render:
            time.sleep(1.0 / 60.0)
        if terminated or truncated:
            break

    success = bool(info.get("is_success", False) or info.get("task_success", False))
    print(f"[replay] file={episode_path.name} success={success} reward={total_reward:.2f} steps={t + 1}")
    env.close()


# -----------------------------
# 训练 / 验证 / 部署
# -----------------------------
def load_dataset(dataset_dir: Path, success_only: bool = False) -> list[EpisodeData]:
    episodes = []
    for episode_path in list_episode_files(dataset_dir):
        episode = load_episode(episode_path)
        if success_only and not episode.success:
            continue
        episodes.append(episode)
    return episodes


def train_policy(args: argparse.Namespace) -> None:
    episodes = load_dataset(Path(args.dataset_dir), success_only=args.success_only)
    if not episodes:
        raise RuntimeError(f"No episodes found in {args.dataset_dir}")

    train_eps, val_eps = split_episodes(episodes, args.val_ratio, args.seed)
    obs_mean, obs_std = compute_obs_stats(train_eps)
    act_mean, act_std = compute_action_stats(train_eps)
    use_rgb = bool(args.use_rgb)
    rgb_height = int(args.rgb_train_height)
    rgb_width = int(args.rgb_train_width)
    rgb_feature_dim = int(args.rgb_feature_dim)
    camera_names = CAMERA_NAMES

    resume_payload = None
    if args.resume and Path(args.resume).exists():
        resume_payload = load_trusted_checkpoint(Path(args.resume), args.device)
        use_rgb = bool(resume_payload.get("use_rgb", use_rgb))
        rgb_height = int(resume_payload.get("rgb_height", rgb_height))
        rgb_width = int(resume_payload.get("rgb_width", rgb_width))
        rgb_feature_dim = int(resume_payload.get("rgb_feature_dim", rgb_feature_dim))
        camera_names = tuple(resume_payload.get("camera_names", list(CAMERA_NAMES)))

    if use_rgb:
        if not all(ep.images for ep in train_eps):
            raise RuntimeError("RGB conditioning is enabled, but some training episodes do not contain images.")
        resize_episode_images(episodes, rgb_width, rgb_height, camera_names)

    train_set = EpisodeWindowDataset(
        train_eps,
        args.context_len,
        args.chunk_len,
        obs_mean,
        obs_std,
        act_mean,
        act_std,
        use_rgb=use_rgb,
        camera_names=camera_names,
        success_only=args.success_only,
    )
    val_set = EpisodeWindowDataset(
        val_eps if val_eps else train_eps,
        args.context_len,
        args.chunk_len,
        obs_mean,
        obs_std,
        act_mean,
        act_std,
        use_rgb=use_rgb,
        camera_names=camera_names,
        success_only=args.success_only,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=(args.device.startswith("cuda")),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=(args.device.startswith("cuda")),
    )

    obs_dim = int(train_eps[0].observations.shape[-1])
    action_dim = int(train_eps[0].actions.shape[-1])
    model = TemporalBCPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        context_len=args.context_len,
        chunk_len=args.chunk_len,
        use_rgb=use_rgb,
        camera_names=camera_names,
        rgb_height=rgb_height,
        rgb_width=rgb_width,
        rgb_feature_dim=rgb_feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )

    if resume_payload is not None:
        model.load_state_dict(resume_payload["state_dict"])
        obs_mean = resume_payload["obs_mean"].astype(np.float32)
        obs_std = resume_payload["obs_std"].astype(np.float32)
        act_mean = resume_payload["act_mean"].astype(np.float32)
        act_std = resume_payload["act_std"].astype(np.float32)
        args.context_len = int(resume_payload.get("context_len", args.context_len))
        args.chunk_len = int(resume_payload.get("chunk_len", args.chunk_len))
        if tuple(resume_payload.get("camera_names", list(camera_names))) != camera_names:
            print(
                f"[train] checkpoint camera order {resume_payload.get('camera_names')} "
                f"differs from dataset order {camera_names}"
            )
        print(f"[train] resumed from {args.resume}")

    device = torch.device(args.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    train_history: list[float] = []
    val_history: list[float] = []

    ensure_dir(Path(args.output_dir))
    checkpoint_path = Path(args.output_dir) / "temporal_bc_best.pt"

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="train", dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        total_count = 0
        batch_bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False, dynamic_ncols=True)
        for obs_batch, rgb_batch, act_batch, mask_batch in batch_bar:
            obs_batch = obs_batch.to(device, non_blocking=True)
            act_batch = act_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            rgb_batch = rgb_batch.to(device, non_blocking=True) if use_rgb else None
            pred = model(obs_batch, rgb_batch)
            mse = (pred - act_batch).pow(2).mean(dim=-1)
            loss = (mse * mask_batch).sum() / mask_batch.sum().clamp_min(1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item()) * float(mask_batch.sum().item())
            total_count += float(mask_batch.sum().item())
            batch_bar.set_postfix(loss=float(loss.item()))

        train_loss = total_loss / max(total_count, 1)
        val_loss = evaluate_loss(model, val_loader, device)
        train_history.append(train_loss)
        val_history.append(val_loss)
        print(f"[train] epoch={epoch}/{args.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            agent = TemporalBCAgent(model, obs_mean, obs_std, act_mean, act_std, args.device)
            agent.save(checkpoint_path, extra_meta={"best_val_loss": best_val, "context_len": args.context_len})
            print(f"[train] saved best checkpoint -> {checkpoint_path}")
        epoch_bar.set_postfix(train_loss=f"{train_loss:.6f}", val_loss=f"{val_loss:.6f}", best_val=f"{best_val:.6f}")

    final_path = Path(args.output_dir) / "temporal_bc_final.pt"
    agent = TemporalBCAgent(model, obs_mean, obs_std, act_mean, act_std, args.device)
    agent.save(final_path, extra_meta={"final_val_loss": best_val, "context_len": args.context_len})
    save_loss_curve(Path(args.output_dir), train_history, val_history)
    print(f"[train] saved final checkpoint -> {final_path}")


def evaluate_loss(model: TemporalBCPolicy, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for obs_batch, rgb_batch, act_batch, mask_batch in tqdm(loader, desc="val", leave=False, dynamic_ncols=True):
            obs_batch = obs_batch.to(device, non_blocking=True)
            act_batch = act_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            rgb_batch = rgb_batch.to(device, non_blocking=True) if rgb_batch.numel() > 0 else None
            pred = model(obs_batch, rgb_batch)
            mse = (pred - act_batch).pow(2).mean(dim=-1)
            loss = (mse * mask_batch).sum()
            total_loss += float(loss.item())
            total_count += float(mask_batch.sum().item())
    return total_loss / max(total_count, 1)


def deploy_policy(args: argparse.Namespace) -> None:
    checkpoint = Path(args.model_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Model not found: {checkpoint}")

    payload = load_trusted_checkpoint(checkpoint, args.device)
    obs_mean = payload["obs_mean"].astype(np.float32)
    obs_std = payload["obs_std"].astype(np.float32)
    act_mean = payload["act_mean"].astype(np.float32)
    act_std = payload["act_std"].astype(np.float32)
    obs_dim = int(payload["obs_dim"])
    action_dim = int(payload["action_dim"])
    context_len = int(payload.get("context_len", args.context_len))
    chunk_len = int(payload.get("chunk_len", args.chunk_len))
    hidden_dim = int(payload.get("hidden_dim", args.hidden_dim))
    num_layers = int(payload.get("num_layers", args.num_layers))
    num_heads = int(payload.get("num_heads", args.num_heads))
    use_rgb = bool(payload.get("use_rgb", args.use_rgb))
    camera_names = tuple(payload.get("camera_names", list(CAMERA_NAMES)))
    rgb_height = int(payload.get("rgb_height", args.rgb_height))
    rgb_width = int(payload.get("rgb_width", args.rgb_width))
    rgb_feature_dim = int(payload.get("rgb_feature_dim", args.rgb_feature_dim))

    model = TemporalBCPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        context_len=context_len,
        chunk_len=chunk_len,
        use_rgb=use_rgb,
        camera_names=camera_names,
        rgb_height=rgb_height,
        rgb_width=rgb_width,
        rgb_feature_dim=rgb_feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=args.dropout,
    )
    model.load_state_dict(payload["state_dict"])
    agent = TemporalBCAgent(model, obs_mean, obs_std, act_mean, act_std, args.device)

    successes = 0
    for episode in range(args.eval_episodes):
        env = Tactile2F85PickPlaceEnv(render_mode="human" if args.render else None, max_episode_steps=args.max_steps, seed=args.seed + episode, task_stage=resolve_task_stage(args.task_stage, checkpoint))
        obs, _ = env.reset(seed=args.seed + episode)
        obs_buffer = deque([obs.copy()] * context_len, maxlen=context_len)
        rgb_buffers = {camera_name: deque(maxlen=context_len) for camera_name in camera_names}
        renderer = _create_rgb_renderer(env.model, rgb_width, rgb_height) if use_rgb else None
        if use_rgb:
            current_rgb = _capture_rgb_frames(renderer, env.data, camera_names)
            for camera_name in camera_names:
                for _ in range(context_len):
                    rgb_buffers[camera_name].append(current_rgb[camera_name].copy())
        agent.action_cache.clear()

        total_reward = 0.0
        done = False
        info = {}
        try:
            while not done:
                context = np.stack(list(obs_buffer), axis=0)
                rgb_context = None
                if use_rgb:
                    rgb_context = build_rgb_context(
                        rgb_buffers,
                        context_len,
                        camera_names,
                        resize_width=rgb_width,
                        resize_height=rgb_height,
                    )
                action = agent.act(context, rgb_context)
                obs, reward, terminated, truncated, info = env.step(action)
                obs_buffer.append(obs.copy())
                total_reward += reward
                done = terminated or truncated
                if use_rgb and renderer is not None:
                    current_rgb = _capture_rgb_frames(renderer, env.data, camera_names)
                    for camera_name in camera_names:
                        rgb_buffers[camera_name].append(current_rgb[camera_name].copy())
                if args.render:
                    time.sleep(1.0 / 60.0)
        finally:
            if renderer is not None:
                renderer.close()

        success = bool(info.get("is_success", False) or info.get("task_success", False))
        successes += int(success)
        print(f"[deploy] episode={episode + 1}/{args.eval_episodes} success={success} reward={total_reward:.2f} steps={env.step_count}")
        env.close()

    print(f"[deploy] success_rate={successes / args.eval_episodes:.3f}")


def collect_dataset(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    ensure_dir(dataset_dir)

    agent = None
    rgb_height = int(args.rgb_height)
    rgb_width = int(args.rgb_width)
    camera_names = CAMERA_NAMES
    if args.collect_source == "model":
        if not args.collect_model:
            raise ValueError("--collect-model is required when --collect-source model")
        payload = load_trusted_checkpoint(Path(args.collect_model), args.device)
        obs_mean = payload["obs_mean"].astype(np.float32)
        obs_std = payload["obs_std"].astype(np.float32)
        act_mean = payload["act_mean"].astype(np.float32)
        act_std = payload["act_std"].astype(np.float32)
        obs_dim = int(payload["obs_dim"])
        action_dim = int(payload["action_dim"])
        context_len = int(payload.get("context_len", args.context_len))
        chunk_len = int(payload.get("chunk_len", args.chunk_len))
        hidden_dim = int(payload.get("hidden_dim", args.hidden_dim))
        num_layers = int(payload.get("num_layers", args.num_layers))
        num_heads = int(payload.get("num_heads", args.num_heads))
        use_rgb = bool(payload.get("use_rgb", args.use_rgb))
        camera_names = tuple(payload.get("camera_names", list(CAMERA_NAMES)))
        rgb_height = int(payload.get("rgb_height", rgb_height))
        rgb_width = int(payload.get("rgb_width", rgb_width))
        rgb_feature_dim = int(payload.get("rgb_feature_dim", args.rgb_feature_dim))
        model = TemporalBCPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            context_len=context_len,
            chunk_len=chunk_len,
            use_rgb=use_rgb,
            camera_names=camera_names,
            rgb_height=rgb_height,
            rgb_width=rgb_width,
            rgb_feature_dim=rgb_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=args.dropout,
        )
        model.load_state_dict(payload["state_dict"])
        agent = TemporalBCAgent(model, obs_mean, obs_std, act_mean, act_std, args.device)

    start_index = len(list_episode_files(dataset_dir))
    saved = 0
    attempts = 0
    while saved < args.episodes and attempts < args.max_attempts:
        attempts += 1
        episode_seed = args.seed + attempts
        episode, episode_path = collect_one_episode(
            source=args.collect_source,
            dataset_dir=dataset_dir,
            episode_index=start_index + saved,
            task_stage=resolve_task_stage(args.task_stage, None),
            seed=episode_seed,
            max_steps=args.max_steps,
            render=args.render,
            context_len=args.context_len,
            rgb_width=rgb_width,
            rgb_height=rgb_height,
            camera_names=camera_names,
            agent=agent,
            keep_failed=args.keep_failed,
        )
        if episode.success or args.keep_failed:
            saved += 1
    print(f"[collect] finished saved={saved} attempts={attempts} dataset_dir={dataset_dir}")


def replay_dataset(args: argparse.Namespace) -> None:
    episode_path = Path(args.episode_file)
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode file not found: {episode_path}")
    replay_episode(episode_path, render=args.render, task_stage=args.task_stage, max_steps=args.max_steps)


def print_dataset_summary(args: argparse.Namespace) -> None:
    episodes = load_dataset(Path(args.dataset_dir), success_only=False)
    if not episodes:
        print(f"[summary] no episodes in {args.dataset_dir}")
        return
    steps = [len(ep.actions) for ep in episodes]
    success_rate = sum(ep.success for ep in episodes) / len(episodes)
    print(f"[summary] episodes={len(episodes)} success_rate={success_rate:.3f} steps_mean={np.mean(steps):.1f} steps_std={np.std(steps):.1f}")


# -----------------------------
# 命令行
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Imitation learning for 2F85 tactile pick-and-place.")
    parser.add_argument("--mode", choices=["collect", "replay", "train", "deploy", "summary"], required=True)
    parser.add_argument("--task-stage", type=str, default=TASK_STAGE_FULL, choices=[TASK_STAGE_AUTO, TASK_STAGE_FULL, TASK_STAGE_GRASP, TASK_STAGE_PLACE])
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--episode-file", type=str, default="")
    parser.add_argument("--collect-source", type=str, default="heuristic", choices=["heuristic", "model"])
    parser.add_argument("--collect-model", type=str, default="")
    parser.add_argument("--keep-failed", action="store_true")
    parser.add_argument("--success-only", action="store_true", help="Train only on successful demos.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--context-len", type=int, default=DEFAULT_CONTEXT_LEN)
    parser.add_argument("--chunk-len", type=int, default=DEFAULT_CHUNK_LEN)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rgb-feature-dim", type=int, default=128)
    parser.add_argument("--rgb-train-width", type=int, default=DEFAULT_RGB_TRAIN_WIDTH, help="Resize RGB to this width before training and deployment.")
    parser.add_argument("--rgb-train-height", type=int, default=DEFAULT_RGB_TRAIN_HEIGHT, help="Resize RGB to this height before training and deployment.")
    parser.add_argument("--rgb-width", type=int, default=DEFAULT_RGB_WIDTH, help="Raw RGB width used for capture and storage.")
    parser.add_argument("--rgb-height", type=int, default=DEFAULT_RGB_HEIGHT, help="Raw RGB height used for capture and storage.")
    parser.add_argument("--no-rgb", action="store_true", help="Disable RGB conditioning and RGB storage.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", help="Use cpu, cuda, or cuda:0 explicitly.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.use_rgb = not args.no_rgb
    if args.mode == "collect":
        collect_dataset(args)
    elif args.mode == "replay":
        if not args.episode_file:
            raise ValueError("--episode-file is required for replay mode")
        replay_dataset(args)
    elif args.mode == "train":
        train_policy(args)
    elif args.mode == "deploy":
        deploy_policy(args)
    elif args.mode == "summary":
        print_dataset_summary(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

# python touch_2f85_visualization_rgb_8_16_IL_test.py --mode collect --episodes 20 --render
# python touch_2f85_visualization_rgb_8_16_IL_test.py --mode summary
# python touch_2f85_visualization_rgb_8_16_IL_test.py --mode replay --episode-file runs/il_2f85/dataset/episode_00000.hdf5 --render

# python touch_2f85_visualization_rgb_8_16_IL_test.py --mode train --epochs 200 --chunk-len 10 --context-len 10 --device cpu --batch-size 16 --rgb-train-width 128 --rgb-train-height 96
# python touch_2f85_visualization_rgb_8_16_IL_test.py --mode train --epochs 200 --no-rgb
# python touch_2f85_visualization_rgb_8_16_IL_test.py --mode deploy --render --model-path runs/il_2f85/temporal_bc_best.pt --eval-episodes 10
