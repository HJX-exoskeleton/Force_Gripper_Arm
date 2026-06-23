import argparse
import json
import logging
import time
from pathlib import Path

import mujoco  # pip install mujoco==3.3.0
import mujoco.viewer
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# 训练脚本依赖 Gymnasium；缺失时直接报错，避免运行到一半才失败。
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError("Install gymnasium first: pip install gymnasium") from exc

# Stable-Baselines3 负责 PPO 训练、评估和环境封装。
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as exc:
    raise ImportError("Install Stable-Baselines3 first: pip install stable-baselines3") from exc


logging.getLogger().setLevel(logging.ERROR)


# -----------------------------
# 场景/任务常量
# -----------------------------
# 任务相关的 MuJoCo 实体名称，必须和 XML 里的命名一致。
ALIGN_SITE_NAME = "ee_site"
GRIPPER_CENTER_SITE_NAME = "ee_site"
OBJECT_BODY_NAME = "red_cylinder"
OBJECT_GEOM_NAME = "red_cylinder"
OBJECT_JOINT_NAME = "red_cylinder_joint"
PLACE_TARGET_BODY_NAME = "target_box"
PLACE_TARGET_GEOM_NAME = "green_target_box"
PLACE_TARGET_JOINT_NAME = "target_box_joint"

TOUCH_ROWS = 8
TOUCH_COLS = 16
FORCE_MAX = 1.0
CONTACT_THRESHOLD = 0.02
NUM_TASK_PHASES = 5
PHASE_REACH_OBJECT = 0
PHASE_GRASP_OBJECT = 1
PHASE_LIFT_OBJECT = 2
PHASE_MOVE_TO_TARGET = 3
PHASE_PLACE_OBJECT = 4

REACH_OBJECT_XY_THRESHOLD = 0.025
GRASP_CENTER_XY_THRESHOLD = 0.0112  # 0.01
GRASP_SITE_OBJECT_THRESHOLD = 0.012
GRASP_DESCEND_Z = -0.135
GRASP_Z_THRESHOLD = 0.018
LIFT_OBJECT_Z_THRESHOLD = 0.03
MOVE_TARGET_XY_THRESHOLD = 0.05
PLACE_DESCEND_Z = -0.12
PLACE_Z_THRESHOLD = 0.02
PLACE_SITE_TARGET_Z_OFFSET = 0.02
OPEN_RELEASE_THRESHOLD = 80.0
GRASP_RETRY_HOME_Z_THRESHOLD = 0.015
HEURISTIC_REACH_GAIN = 0.10
HEURISTIC_TARGET_GAIN = 0.12
GRIPPER_ASSIST_STEP = 18.0
TASK_STAGE_FULL = "full"
TASK_STAGE_GRASP = "grasp"
TASK_STAGE_PLACE = "place"
TASK_STAGE_AUTO = "auto"
WARM_START_MAX_STEPS = 120
GRASP_SETTLE_STEPS = 10
PLACE_KEEP_CLOSED_THRESHOLD = 200.0
PLACE_LOST_GRASP_PENALTY = 25.0
# 放置阶段的奖励权重：对准目标、下降对位、张开夹爪。
PLACE_RELEASE_OPEN_WEIGHT = 6.0
PLACE_RELEASE_ALIGN_WEIGHT = 12.0
PLACE_RELEASE_HEIGHT_WEIGHT = 4.0
PLACE_TARGET_STABILITY_WEIGHT = 10.0
PLACE_CLOSED_NEAR_TARGET_PENALTY = 6.0
# 任务成功时的固定终止奖励，便于 PPO 明确区分成功轨迹。
PLACE_SUCCESS_BONUS = 800.0
LIFT_ASSIST_STEP = 0.006
TRANSPORT_ASSIST_STEP = 0.004

TARGET_X_RANGE = (0.15, 0.2)  # (0.1, 0.2)
TARGET_Y_RANGE = (0.25, 0.3)  # (0.2, 0.3)
TARGET_Z = 0.0
OBJECT_X_RANGE = (-0.05, 0.0)
OBJECT_Y_RANGE = (-0.05, 0.0)
OBJECT_Z = 0.0

HOME_Z = 0.0
DOWN_LIMIT_Z = -0.135
OPEN_GRIPPER = 0.0
CLOSED_GRIPPER = 230.0


# -----------------------------
# 低层工具函数
# -----------------------------
def resolve_xml_path():
    # 统一从脚本位置反推场景文件，避免绝对路径写死在代码里。
    script_dir = Path(__file__).resolve().parent
    return script_dir.parents[1] / "assets_robot_xml" / "gripper_2f85" / "scene_8_16.xml"


def build_touch_address_maps(model):
    # 8x16 触觉阵列在 MuJoCo 里是一串连续 sensor，这里提前建立索引映射。
    if model.nsensor < TOUCH_ROWS * TOUCH_COLS * 2:
        raise RuntimeError(
            f"Expected at least {TOUCH_ROWS * TOUCH_COLS * 2} touch sensors, got {model.nsensor}."
        )

    touch_point_adr_right = [[0] * TOUCH_COLS for _ in range(TOUCH_ROWS)]
    touch_point_adr_left = [[0] * TOUCH_COLS for _ in range(TOUCH_ROWS)]

    for x in range(TOUCH_ROWS):
        for y in range(TOUCH_COLS):
            idx = x + y * TOUCH_ROWS
            touch_point_adr_right[x][y] = model.sensor_adr[idx]
            touch_point_adr_left[x][y] = model.sensor_adr[idx + model.nsensor // 2]

    return touch_point_adr_right, touch_point_adr_left


def read_touch_arrays(data, touch_point_adr_right, touch_point_adr_left):
    # 将左右触觉传感器的三维力向量转换为 8x16 标量热图。
    touch_right = np.zeros((TOUCH_ROWS, TOUCH_COLS), dtype=np.float32)
    touch_left = np.zeros((TOUCH_ROWS, TOUCH_COLS), dtype=np.float32)

    for x in range(TOUCH_ROWS):
        for y in range(TOUCH_COLS):
            adr_right = touch_point_adr_right[x][y]
            adr_left = touch_point_adr_left[x][y]

            force_right = mujoco.mju_norm3(data.sensordata[adr_right:adr_right + 3])
            force_left = mujoco.mju_norm3(data.sensordata[adr_left:adr_left + 3])

            touch_right[x, y] = mujoco.mju_clip(force_right, 0.0, FORCE_MAX)
            touch_left[x, y] = mujoco.mju_clip(force_left, 0.0, FORCE_MAX)

    return touch_right, touch_left


def are_geoms_in_contact(data, geom_a_id, geom_b_id):
    # 直接遍历接触对，判断两个 geom 是否发生接触。
    for i in range(data.ncon):
        contact = data.contact[i]
        if {contact.geom1, contact.geom2} == {geom_a_id, geom_b_id}:
            return True
    return False


def random_free_joint_pose(model, data, joint_name, rng, x_range, y_range, z_value):
    # 给自由关节随机初始化位姿，用于重置时随机放置物体/目标。
    pos = np.array([rng.uniform(*x_range), rng.uniform(*y_range), z_value], dtype=np.float64)
    joint_id = model.joint(joint_name).id
    qpos_adr = model.jnt_qposadr[joint_id]
    dof_adr = model.jnt_dofadr[joint_id]

    data.qpos[qpos_adr:qpos_adr + 3] = pos
    data.qpos[qpos_adr + 3:qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[dof_adr:dof_adr + 6] = 0.0
    return pos


def world_xy_delta_to_ctrl_delta(delta_xy):
    # 当前控制空间和世界平面位移近似一致，保留一个显式转换函数便于后续扩展。
    return np.array([delta_xy[0], delta_xy[1]], dtype=np.float64)


def metadata_path_for_model(model_path):
    # 模型旁边保存一份元数据，记录该策略模型属于哪个任务阶段。
    model_path = Path(model_path)
    return model_path.with_name(model_path.name + ".meta.json")


def resolve_task_stage(requested_stage, model_path=None):
    # `auto` 模式下优先从模型元数据恢复任务阶段，否则回退到 full。
    if requested_stage != TASK_STAGE_AUTO:
        return requested_stage

    if model_path is not None:
        metadata_path = metadata_path_for_model(model_path)
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
                stage = metadata.get("task_stage", TASK_STAGE_FULL)
                if stage in {TASK_STAGE_FULL, TASK_STAGE_GRASP, TASK_STAGE_PLACE}:
                    print(f"[rl] auto task-stage resolved from metadata: {stage}")
                    return stage
            except Exception as exc:
                print(f"[rl] failed to read stage metadata from {metadata_path}: {exc}")

    print("[rl] auto task-stage fallback to full")
    return TASK_STAGE_FULL


def save_task_stage_metadata(model_path, task_stage):
    # 训练结束后把任务阶段写回模型旁边，方便评估时自动对齐。
    metadata_path = metadata_path_for_model(model_path)
    metadata = {
        "task_stage": task_stage,
        "saved_at": time.time(),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def collect_expert_dataset(task_stage, seed, episodes, max_episode_steps, max_attempts):
    # 用启发式策略采集成功轨迹，作为 BC warm start 的专家数据。
    obs_batches = []
    action_batches = []
    collected_episodes = 0
    attempts = 0

    while collected_episodes < episodes and attempts < max_attempts:
        attempts += 1
        env = Tactile2F85PickPlaceEnv(
            render_mode=None,
            max_episode_steps=max_episode_steps,
            seed=seed + attempts,
            task_stage=task_stage,
        )
        obs, _ = env.reset(seed=seed + attempts)
        done = False
        episode_obs = []
        episode_actions = []
        while not done:
            action = heuristic_action(env).astype(np.float32)
            episode_obs.append(obs.copy())
            episode_actions.append(action.copy())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        success = bool(info.get("is_success", False) or info.get("task_success", False))
        if success and episode_obs:
            obs_batches.append(np.asarray(episode_obs, dtype=np.float32))
            action_batches.append(np.asarray(episode_actions, dtype=np.float32))
            collected_episodes += 1
            print(
                f"[bc] collected expert episode {collected_episodes}/{episodes} "
                f"(attempt {attempts}, steps={len(episode_obs)})"
            )
        env.close()

    if not obs_batches:
        raise RuntimeError("No successful expert episodes were collected for BC pretraining.")

    observations = np.concatenate(obs_batches, axis=0)
    actions = np.concatenate(action_batches, axis=0)
    return observations, actions, collected_episodes, attempts


def bc_pretrain_policy(model, observations, actions, epochs, batch_size, lr):
    # 直接用 PPO policy 的动作分布做行为克隆预训练，给强化学习一个更好的起点。
    policy = model.policy
    device = policy.device
    policy.train()

    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    act_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    model.policy.optimizer = optimizer

    for epoch in range(epochs):
        total_loss = 0.0
        total_batches = 0
        for obs_batch, act_batch in loader:
            _, log_prob, entropy = policy.evaluate_actions(obs_batch, act_batch)
            loss = -log_prob.mean()
            if entropy is not None:
                loss = loss - 0.001 * entropy.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        print(f"[bc] epoch={epoch + 1}/{epochs}, loss={avg_loss:.4f}")

    policy.eval()


# -----------------------------
# 强化学习环境
# -----------------------------
class Tactile2F85PickPlaceEnv(gym.Env):
    """PPO environment for the 2F85 tactile pick-and-place task.

    Action is continuous delta control for [base_x, base_y, base_z, yaw, gripper].
    Success is defined as contact between red_cylinder and green_target_box.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, max_episode_steps=700, frame_skip=10, seed=None, task_stage=TASK_STAGE_FULL):
        super().__init__()
        # 每个环境实例都自己加载一份模型和数据，避免训练并行时共享状态。
        self.xml_path = str(resolve_xml_path())
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.rng = np.random.default_rng(seed)
        self.task_stage = task_stage
        self.viewer = None

        if self.task_stage not in {TASK_STAGE_FULL, TASK_STAGE_GRASP, TASK_STAGE_PLACE}:
            raise ValueError(f"Unsupported task_stage: {self.task_stage}")

        self.actuator_x = self.model.actuator("base_actuator_x").id
        self.actuator_y = self.model.actuator("base_actuator_y").id
        self.actuator_z = self.model.actuator("base_actuator_z").id
        self.actuator_yaw = self.model.actuator("base_actuator_yaw").id
        self.actuator_gripper = self.model.actuator("fingers_actuator").id

        self.align_site_id = self.model.site(ALIGN_SITE_NAME).id
        self.gripper_center_site_id = self.model.site(GRIPPER_CENTER_SITE_NAME).id
        self.object_body_id = self.model.body(OBJECT_BODY_NAME).id
        self.object_geom_id = self.model.geom(OBJECT_GEOM_NAME).id
        self.target_body_id = self.model.body(PLACE_TARGET_BODY_NAME).id
        self.target_geom_id = self.model.geom(PLACE_TARGET_GEOM_NAME).id

        self.touch_right_adr, self.touch_left_adr = build_touch_address_maps(self.model)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30 + NUM_TASK_PHASES,), dtype=np.float32)

        self.step_count = 0
        self.prev_object_target_dist = 0.0
        self.prev_gripper_object_dist = 0.0
        self.prev_site_object_dist = 0.0
        self.prev_place_object_target_dist = 0.0
        self.prev_grasp_z_error = 0.0
        self.prev_place_z_error = 0.0
        self.has_grasp_contact = False
        self.task_success = False
        self.phase = PHASE_REACH_OBJECT
        self.grasp_retry_count = 0
        self.grasp_settle_steps = 0

    def _get_base_xy_qpos(self):
        # 读取底盘在平面上的关节位姿。
        joint_x = self.model.joint("base_mount_joint_x").id
        joint_y = self.model.joint("base_mount_joint_y").id
        qpos_x = self.data.qpos[self.model.jnt_qposadr[joint_x]]
        qpos_y = self.data.qpos[self.model.jnt_qposadr[joint_y]]
        return np.array([qpos_x, qpos_y], dtype=np.float64)

    def _get_gripper_center_xy(self):
        # 用末端 site 作为夹爪中心在平面上的参考点。
        return self.data.site_xpos[self.gripper_center_site_id][:2].copy()

    def _get_base_z_qpos(self):
        # 读取底盘高度，用于抓取和放置阶段的下探/回升判断。
        joint_z = self.model.joint("base_mount_joint_z").id
        return float(self.data.qpos[self.model.jnt_qposadr[joint_z]])

    def _get_touch_state(self):
        # 返回左右触觉最大值以及是否两侧都接触到物体。
        touch_right, touch_left = read_touch_arrays(self.data, self.touch_right_adr, self.touch_left_adr)
        max_right = float(np.max(touch_right))
        max_left = float(np.max(touch_left))
        return max_right, max_left, max_right > CONTACT_THRESHOLD and max_left > CONTACT_THRESHOLD

    def _is_success(self, obj_pos, target_pos, gripper_ctrl):
        # 不同任务阶段的成功条件不同：
        # grasp 仍然看举升是否完成；full/place 则只要圆柱和目标几何接触，就认为放置成功。
        object_target_dist = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
        grasp_success = (
            self.task_stage == TASK_STAGE_GRASP
            and self.phase == PHASE_LIFT_OBJECT
            and obj_pos[2] > LIFT_OBJECT_Z_THRESHOLD
            and self.has_grasp_contact
        )
        placed_success = self.task_stage in {TASK_STAGE_FULL, TASK_STAGE_PLACE} and are_geoms_in_contact(
            self.data, self.object_geom_id, self.target_geom_id
        )
        if self.task_stage == TASK_STAGE_GRASP:
            return bool(grasp_success)
        return bool(placed_success)

    def _get_obs(self):
        # 观测向量 = 本体状态 + 目标相对量 + 触觉强度 + 阶段 one-hot。
        touch_right, touch_left = read_touch_arrays(self.data, self.touch_right_adr, self.touch_left_adr)
        max_right = float(np.max(touch_right))
        max_left = float(np.max(touch_left))

        base_xy = self._get_base_xy_qpos()
        gripper_xy = self._get_gripper_center_xy()
        base_z = self._get_base_z_qpos()
        ctrl = self.data.ctrl.copy()
        obj_pos = self.data.xpos[self.object_body_id].copy()
        target_pos = self.data.xpos[self.target_body_id].copy()
        site_pos = self.data.site_xpos[self.align_site_id].copy()

        obj_to_target = target_pos - obj_pos
        site_to_object = obj_pos - site_pos
        gripper_to_object_xy = obj_pos[:2] - gripper_xy
        gripper_to_target_xy = target_pos[:2] - gripper_xy

        continuous_obs = np.array(
            [
                base_xy[0],
                base_xy[1],
                base_z,
                ctrl[self.actuator_x],
                ctrl[self.actuator_y],
                ctrl[self.actuator_z],
                ctrl[self.actuator_yaw] / np.pi,
                ctrl[self.actuator_gripper] / 255.0,
                obj_pos[0],
                obj_pos[1],
                obj_pos[2],
                target_pos[0],
                target_pos[1],
                target_pos[2],
                site_pos[0],
                site_pos[1],
                site_pos[2],
                obj_to_target[0],
                obj_to_target[1],
                obj_to_target[2],
                site_to_object[0],
                site_to_object[1],
                site_to_object[2],
                gripper_to_object_xy[0],
                gripper_to_object_xy[1],
                gripper_to_target_xy[0],
                gripper_to_target_xy[1],
                np.clip(max_right, 0.0, FORCE_MAX),
                np.clip(max_left, 0.0, FORCE_MAX),
                float(self.has_grasp_contact),
            ],
            dtype=np.float32,
        )
        phase_obs = np.zeros(NUM_TASK_PHASES, dtype=np.float32)
        phase_obs[self.phase] = 1.0
        return np.concatenate([continuous_obs, phase_obs]).astype(np.float32)

    def _apply_action(self, action):
        # 把 [-1, 1] 动作映射成物理控制增量，再写入 MuJoCo ctrl。
        action = np.clip(action, -1.0, 1.0)
        if self.task_stage == TASK_STAGE_GRASP:
            delta = np.array([0.006, 0.006, 0.003, 0.02, 5.0], dtype=np.float64) * action
        elif self.task_stage == TASK_STAGE_PLACE:
            delta = np.array([0.004, 0.004, 0.003, 0.02, 4.0], dtype=np.float64) * action
        else:
            delta = np.array([0.008, 0.008, 0.004, 0.03, 6.0], dtype=np.float64) * action

        self.data.ctrl[self.actuator_x] = np.clip(self.data.ctrl[self.actuator_x] + delta[0], -1.0, 1.0)
        self.data.ctrl[self.actuator_y] = np.clip(self.data.ctrl[self.actuator_y] + delta[1], -1.0, 1.0)
        self.data.ctrl[self.actuator_z] = np.clip(self.data.ctrl[self.actuator_z] + delta[2], DOWN_LIMIT_Z, HOME_Z)
        self.data.ctrl[self.actuator_yaw] = np.clip(self.data.ctrl[self.actuator_yaw] + delta[3], -np.pi, np.pi)
        self.data.ctrl[self.actuator_gripper] = np.clip(
            self.data.ctrl[self.actuator_gripper] + delta[4], OPEN_GRIPPER, CLOSED_GRIPPER
        )

        obj_pos = self.data.xpos[self.object_body_id]
        target_pos = self.data.xpos[self.target_body_id]
        site_pos = self.data.site_xpos[self.align_site_id]
        base_z = self._get_base_z_qpos()
        site_object_xy_dist = float(np.linalg.norm(obj_pos[:2] - site_pos[:2]))
        object_target_dist = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))

        grasp_ready = (
            self.phase == PHASE_GRASP_OBJECT
            and site_object_xy_dist < GRASP_CENTER_XY_THRESHOLD
            and abs(base_z - GRASP_DESCEND_Z) < GRASP_Z_THRESHOLD
        )
        place_ready = (
            self.phase == PHASE_PLACE_OBJECT
            and object_target_dist < MOVE_TARGET_XY_THRESHOLD
            and abs(base_z - PLACE_DESCEND_Z) < PLACE_Z_THRESHOLD
        )

        if self.task_stage == TASK_STAGE_GRASP:
            if self.phase == PHASE_REACH_OBJECT and site_object_xy_dist < GRASP_CENTER_XY_THRESHOLD:
                self.data.ctrl[self.actuator_z] = np.clip(
                    self.data.ctrl[self.actuator_z] - 0.006, GRASP_DESCEND_Z, HOME_Z
                )
            if self.phase in {PHASE_REACH_OBJECT, PHASE_GRASP_OBJECT} and site_object_xy_dist < GRASP_CENTER_XY_THRESHOLD:
                self.data.ctrl[self.actuator_z] = np.clip(
                    min(self.data.ctrl[self.actuator_z], GRASP_DESCEND_Z),
                    GRASP_DESCEND_Z,
                    HOME_Z,
                )
        elif self.task_stage == TASK_STAGE_PLACE and self.phase in {PHASE_MOVE_TO_TARGET, PHASE_PLACE_OBJECT}:
            self.data.ctrl[self.actuator_gripper] = np.clip(
                max(self.data.ctrl[self.actuator_gripper], PLACE_KEEP_CLOSED_THRESHOLD),
                OPEN_GRIPPER,
                CLOSED_GRIPPER,
            )

        if grasp_ready:
            self.data.ctrl[self.actuator_gripper] = np.clip(
                self.data.ctrl[self.actuator_gripper] + GRIPPER_ASSIST_STEP,
                OPEN_GRIPPER,
                CLOSED_GRIPPER,
            )
        elif place_ready:
            self.data.ctrl[self.actuator_gripper] = np.clip(
                self.data.ctrl[self.actuator_gripper] - GRIPPER_ASSIST_STEP,
                OPEN_GRIPPER,
                CLOSED_GRIPPER,
            )

        if self.task_stage == TASK_STAGE_FULL and self.phase == PHASE_LIFT_OBJECT and self.has_grasp_contact:
            self.data.ctrl[self.actuator_z] = np.clip(
                self.data.ctrl[self.actuator_z] + LIFT_ASSIST_STEP,
                DOWN_LIMIT_Z,
                HOME_Z,
            )

        if self.task_stage == TASK_STAGE_FULL and self.phase in {PHASE_MOVE_TO_TARGET, PHASE_PLACE_OBJECT} and self.has_grasp_contact:
            transport_error = target_pos[:2] - site_pos[:2]
            transport_delta = np.clip(world_xy_delta_to_ctrl_delta(transport_error), -0.02, 0.02)
            self.data.ctrl[self.actuator_x] = np.clip(
                self.data.ctrl[self.actuator_x] + TRANSPORT_ASSIST_STEP * transport_delta[0],
                -1.0,
                1.0,
            )
            self.data.ctrl[self.actuator_y] = np.clip(
                self.data.ctrl[self.actuator_y] + TRANSPORT_ASSIST_STEP * transport_delta[1],
                -1.0,
                1.0,
            )

    def _scripted_grasp_warm_start(self, max_steps=WARM_START_MAX_STEPS):
        """Warm start place-stage episodes from a physically grasped object."""
        # 放置任务需要先有一个“已抓起”的初始状态，这里用脚本补出这个起点。
        for _ in range(max_steps):
            obj_pos = self.data.xpos[self.object_body_id].copy()
            ee_xy = self.data.site_xpos[self.align_site_id][:2].copy()
            base_z = self._get_base_z_qpos()
            xy_error = obj_pos[:2] - ee_xy
            _, _, both_touch = self._get_touch_state()

            if np.linalg.norm(xy_error) > REACH_OBJECT_XY_THRESHOLD:
                self.data.ctrl[self.actuator_x] = np.clip(
                    self.data.ctrl[self.actuator_x] + np.clip(xy_error[0], -0.03, 0.03),
                    -1.0,
                    1.0,
                )
                self.data.ctrl[self.actuator_y] = np.clip(
                    self.data.ctrl[self.actuator_y] + np.clip(xy_error[1], -0.03, 0.03),
                    -1.0,
                    1.0,
                )
                self.data.ctrl[self.actuator_z] = HOME_Z
                self.data.ctrl[self.actuator_gripper] = OPEN_GRIPPER
            elif base_z > GRASP_DESCEND_Z + GRASP_Z_THRESHOLD:
                self.data.ctrl[self.actuator_z] = np.clip(
                    self.data.ctrl[self.actuator_z] - 0.004, GRASP_DESCEND_Z, HOME_Z
                )
                self.data.ctrl[self.actuator_gripper] = OPEN_GRIPPER
            elif not both_touch:
                self.data.ctrl[self.actuator_z] = GRASP_DESCEND_Z
                self.data.ctrl[self.actuator_gripper] = np.clip(
                    self.data.ctrl[self.actuator_gripper] + GRIPPER_ASSIST_STEP,
                    OPEN_GRIPPER,
                    CLOSED_GRIPPER,
                )
            elif obj_pos[2] < LIFT_OBJECT_Z_THRESHOLD:
                self.data.ctrl[self.actuator_z] = HOME_Z
                self.data.ctrl[self.actuator_gripper] = CLOSED_GRIPPER
            else:
                self.data.ctrl[self.actuator_z] = HOME_Z
                self.data.ctrl[self.actuator_gripper] = CLOSED_GRIPPER

            mujoco.mj_step(self.model, self.data)

            obj_pos = self.data.xpos[self.object_body_id].copy()
            base_z = self._get_base_z_qpos()
            _, _, both_touch = self._get_touch_state()
            if both_touch and obj_pos[2] > LIFT_OBJECT_Z_THRESHOLD and abs(base_z - HOME_Z) < 0.05:
                return True

        return False

    def _compute_reward(self):
        # 奖励函数按任务阶段分开写，避免抓取和放置互相干扰。
        obj_pos = self.data.xpos[self.object_body_id]
        target_pos = self.data.xpos[self.target_body_id]
        site_pos = self.data.site_xpos[self.align_site_id]
        gripper_xy = self._get_gripper_center_xy()
        base_z = self._get_base_z_qpos()
        gripper_ctrl = float(self.data.ctrl[self.actuator_gripper])

        gripper_object_dist = float(np.linalg.norm(obj_pos[:2] - gripper_xy))
        site_object_xy_dist = float(np.linalg.norm(obj_pos[:2] - site_pos[:2]))
        site_object_dist = float(np.linalg.norm(obj_pos - site_pos))
        object_target_dist = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
        gripper_target_dist = float(np.linalg.norm(gripper_xy - target_pos[:2]))
        grasp_z_error = abs(base_z - GRASP_DESCEND_Z)
        place_z_error = abs(base_z - PLACE_DESCEND_Z)
        lifted = obj_pos[2] > LIFT_OBJECT_Z_THRESHOLD
        object_near_site = site_object_xy_dist < 0.045
        max_right, max_left, both_touch = self._get_touch_state()
        success = self._is_success(obj_pos, target_pos, gripper_ctrl)

        self.has_grasp_contact = self.has_grasp_contact or bool(both_touch)
        self.task_success = self.task_success or bool(success)

        reward = -0.02

        if self.task_stage == TASK_STAGE_GRASP:
            # 抓取阶段：先靠近，再下探，再闭合并建立双侧接触。
            if self.phase == PHASE_REACH_OBJECT:
                reward += 6.0 * (self.prev_gripper_object_dist - gripper_object_dist)
                reward += 2.0 * np.exp(-45.0 * gripper_object_dist)
                reward += 0.1 if gripper_ctrl < OPEN_RELEASE_THRESHOLD else -0.1
                reward += 2.5 * np.exp(-80.0 * site_object_xy_dist)
                if site_object_xy_dist < GRASP_CENTER_XY_THRESHOLD:
                    self.grasp_settle_steps += 1
                    reward += 0.8
                    reward += 1.2 * np.exp(-35.0 * abs(base_z - GRASP_DESCEND_Z))
                    reward += 0.8 * (self.prev_grasp_z_error - grasp_z_error)
                    if self.grasp_settle_steps >= GRASP_SETTLE_STEPS:
                        reward += 5.0
                        self.phase = PHASE_GRASP_OBJECT
                else:
                    self.grasp_settle_steps = 0

            elif self.phase == PHASE_GRASP_OBJECT:
                reward += 2.0 * np.exp(-45.0 * site_object_xy_dist)
                reward += 3.0 * (self.prev_grasp_z_error - grasp_z_error)
                reward += 2.0 * np.exp(-25.0 * grasp_z_error)
                if grasp_z_error < GRASP_Z_THRESHOLD and site_object_xy_dist < GRASP_CENTER_XY_THRESHOLD:
                    reward += 1.5 * (gripper_ctrl / CLOSED_GRIPPER)
                    reward += 0.5 if gripper_ctrl > 180.0 else -0.5
                else:
                    reward += 0.1 if gripper_ctrl < OPEN_RELEASE_THRESHOLD else -0.2
                if both_touch:
                    reward += 12.0
                    self.has_grasp_contact = True
                    self.phase = PHASE_LIFT_OBJECT

            elif self.phase == PHASE_LIFT_OBJECT:
                reward += 0.5 * (gripper_ctrl / CLOSED_GRIPPER)
                reward += 12.0 * max(0.0, obj_pos[2] - OBJECT_Z)
                reward += 1.0 if object_near_site else -0.5
                reward += 0.5 * np.exp(-20.0 * abs(base_z - HOME_Z))
                if lifted and object_near_site:
                    reward += 30.0
                elif abs(base_z - HOME_Z) < GRASP_RETRY_HOME_Z_THRESHOLD and not lifted:
                    reward -= 8.0
                    self.has_grasp_contact = False
                    self.grasp_retry_count += 1
                    self.phase = PHASE_REACH_OBJECT

            normalized_ctrl = np.array(
                [
                    self.data.ctrl[self.actuator_x],
                    self.data.ctrl[self.actuator_y],
                    self.data.ctrl[self.actuator_z] / max(abs(DOWN_LIMIT_Z), 1e-6),
                    self.data.ctrl[self.actuator_yaw] / np.pi,
                    self.data.ctrl[self.actuator_gripper] / CLOSED_GRIPPER,
                ],
                dtype=np.float64,
            )
            reward -= 0.02 * float(np.sum(np.square(normalized_ctrl)))

            self.prev_gripper_object_dist = gripper_object_dist
            self.prev_object_target_dist = object_target_dist
            self.prev_site_object_dist = site_object_dist
            self.prev_place_object_target_dist = object_target_dist
            self.prev_grasp_z_error = grasp_z_error
            self.prev_place_z_error = place_z_error
            return float(reward), bool(self.phase == PHASE_LIFT_OBJECT and lifted and object_near_site)

        if self.task_stage == TASK_STAGE_PLACE:
            # 放置阶段：先移动到目标上方，再下降，最后松开夹爪并让物体稳定落位。
            if self.phase == PHASE_MOVE_TO_TARGET:
                reward += 10.0 * (self.prev_object_target_dist - gripper_target_dist)
                reward += 5.0 * np.exp(-10.0 * gripper_target_dist)
                reward += 0.8 * (gripper_ctrl / CLOSED_GRIPPER)
                reward += 1.0 * np.exp(-20.0 * abs(base_z - HOME_Z))
                if lifted:
                    reward += 2.0
                else:
                    reward -= PLACE_LOST_GRASP_PENALTY
                if gripper_target_dist < MOVE_TARGET_XY_THRESHOLD:
                    reward += 8.0
                    self.phase = PHASE_PLACE_OBJECT

            elif self.phase == PHASE_PLACE_OBJECT:
                # 这部分是放置核心：对准目标 + 贴近地面 + 主动张开夹爪。
                # 这里的 shaping 目标是让策略学会三件事：
                # 1) 物体尽量对准目标区域；2) 下降到合适高度；3) 在对位后主动松夹爪。
                release_openness = np.clip((CLOSED_GRIPPER - gripper_ctrl) / CLOSED_GRIPPER, 0.0, 1.0)
                reward += PLACE_RELEASE_ALIGN_WEIGHT * np.exp(-25.0 * object_target_dist)
                reward += PLACE_TARGET_STABILITY_WEIGHT * (self.prev_place_object_target_dist - object_target_dist)
                reward += 4.0 * (self.prev_place_z_error - place_z_error)
                reward += PLACE_RELEASE_HEIGHT_WEIGHT * np.exp(-25.0 * place_z_error)
                reward += PLACE_RELEASE_OPEN_WEIGHT * release_openness
                reward += 6.0 * np.exp(-18.0 * object_target_dist) * release_openness
                reward -= PLACE_CLOSED_NEAR_TARGET_PENALTY * (1.0 - release_openness) * np.exp(-25.0 * object_target_dist)
                if not lifted:
                    # 只有在物体已经靠近目标时，才奖励“放下”这个动作。
                    reward += 20.0 * np.exp(-25.0 * object_target_dist)
                if success:
                    reward += PLACE_SUCCESS_BONUS

            else:
                # 如果阶段机走到了意外状态，给一个较强惩罚，避免策略停在错误阶段。
                reward -= PLACE_LOST_GRASP_PENALTY

            normalized_ctrl = np.array(
                [
                    self.data.ctrl[self.actuator_x],
                    self.data.ctrl[self.actuator_y],
                    self.data.ctrl[self.actuator_z] / max(abs(DOWN_LIMIT_Z), 1e-6),
                    self.data.ctrl[self.actuator_yaw] / np.pi,
                    self.data.ctrl[self.actuator_gripper] / CLOSED_GRIPPER,
                ],
                dtype=np.float64,
            )
            reward -= 0.02 * float(np.sum(np.square(normalized_ctrl)))

            self.prev_gripper_object_dist = gripper_object_dist
            self.prev_object_target_dist = gripper_target_dist
            self.prev_site_object_dist = site_object_dist
            self.prev_grasp_z_error = grasp_z_error
            self.prev_place_z_error = place_z_error
            return float(reward), bool(success)

        if self.phase == PHASE_REACH_OBJECT:
            # 默认完整任务流程：先抓取，再搬运，再放置。
            reward += 6.0 * (self.prev_gripper_object_dist - gripper_object_dist)
            reward += 2.0 * np.exp(-45.0 * gripper_object_dist)
            reward += 0.08 * (1.0 - gripper_ctrl / CLOSED_GRIPPER)
            reward += 2.5 * np.exp(-80.0 * site_object_xy_dist)
            if site_object_xy_dist < GRASP_CENTER_XY_THRESHOLD:
                reward += 4.0 * np.exp(-35.0 * abs(base_z - GRASP_DESCEND_Z))
                reward += 1.8 * (self.prev_grasp_z_error - grasp_z_error)
                reward += 2.0 * (1.0 - gripper_ctrl / CLOSED_GRIPPER)
                self.phase = PHASE_GRASP_OBJECT

        elif self.phase == PHASE_GRASP_OBJECT:
            # 抓取阶段继续奖励接触、下降和闭合。
            reward += 2.0 * np.exp(-45.0 * site_object_xy_dist)
            reward += 3.0 * (self.prev_grasp_z_error - grasp_z_error)
            reward += 2.0 * np.exp(-25.0 * grasp_z_error)
            if grasp_z_error < GRASP_Z_THRESHOLD and site_object_xy_dist < GRASP_CENTER_XY_THRESHOLD:
                reward += 2.2 * (gripper_ctrl / CLOSED_GRIPPER)
                reward += 0.8 if gripper_ctrl > 180.0 else -0.2
            else:
                reward -= 0.05 * (gripper_ctrl / CLOSED_GRIPPER)
            if both_touch:
                reward += 12.0
                self.has_grasp_contact = True
                self.phase = PHASE_LIFT_OBJECT

        elif self.phase == PHASE_LIFT_OBJECT:
            # 举升阶段强调物体离地和回到安全高度。
            lift_height = max(0.0, obj_pos[2] - OBJECT_Z)
            reward += 0.4 * (gripper_ctrl / CLOSED_GRIPPER)
            reward += 30.0 * lift_height
            reward += 8.0 * np.exp(-18.0 * abs(base_z - HOME_Z))
            reward += 2.0 if lifted else -2.0
            reward += 0.8 if object_near_site else -0.8
            reward += 1.0 * np.exp(-10.0 * gripper_target_dist)
            if lifted:
                reward += 20.0
                self.grasp_retry_count = 0
                self.phase = PHASE_MOVE_TO_TARGET
            elif abs(base_z - HOME_Z) < GRASP_RETRY_HOME_Z_THRESHOLD and not lifted:
                reward -= 8.0
                self.has_grasp_contact = False
                self.grasp_retry_count += 1
                self.phase = PHASE_REACH_OBJECT

        elif self.phase == PHASE_MOVE_TO_TARGET:
            # 搬运阶段强调把抓到的物体带到目标区域上方。
            if not object_near_site and obj_pos[2] < LIFT_OBJECT_Z_THRESHOLD:
                reward -= 8.0
                self.has_grasp_contact = False
                self.grasp_retry_count += 1
                self.phase = PHASE_REACH_OBJECT
            else:
                reward += 8.0 * (self.prev_object_target_dist - gripper_target_dist)
                reward += 3.0 * np.exp(-10.0 * gripper_target_dist)
                reward += 0.8 if object_near_site else -0.8
                reward += 0.4 * (gripper_ctrl / CLOSED_GRIPPER)
                reward += 0.8 * np.exp(-30.0 * abs(base_z - HOME_Z))
                if gripper_target_dist < MOVE_TARGET_XY_THRESHOLD:
                    reward += 10.0
                    self.phase = PHASE_PLACE_OBJECT

        elif self.phase == PHASE_PLACE_OBJECT:
            # 最终放置阶段，和上面的任务阶段分支保持一致。
            # 与训练分支保持同一套放置 shaping，避免评估时和训练时目标不一致。
            release_openness = np.clip((CLOSED_GRIPPER - gripper_ctrl) / CLOSED_GRIPPER, 0.0, 1.0)
            reward += PLACE_RELEASE_ALIGN_WEIGHT * np.exp(-25.0 * object_target_dist)
            reward += PLACE_TARGET_STABILITY_WEIGHT * (self.prev_place_object_target_dist - object_target_dist)
            reward += 4.0 * (self.prev_place_z_error - place_z_error)
            reward += PLACE_RELEASE_HEIGHT_WEIGHT * np.exp(-25.0 * place_z_error)
            reward += PLACE_RELEASE_OPEN_WEIGHT * release_openness
            reward += 6.0 * np.exp(-18.0 * object_target_dist) * release_openness
            reward -= PLACE_CLOSED_NEAR_TARGET_PENALTY * (1.0 - release_openness) * np.exp(-25.0 * object_target_dist)
            if gripper_target_dist < MOVE_TARGET_XY_THRESHOLD:
                reward += 5.0 * release_openness
            if success:
                reward += PLACE_SUCCESS_BONUS

        if success:
            self.phase = PHASE_PLACE_OBJECT

        normalized_ctrl = np.array(
            [
                self.data.ctrl[self.actuator_x],
                self.data.ctrl[self.actuator_y],
                self.data.ctrl[self.actuator_z] / max(abs(DOWN_LIMIT_Z), 1e-6),
                self.data.ctrl[self.actuator_yaw] / np.pi,
                self.data.ctrl[self.actuator_gripper] / CLOSED_GRIPPER,
            ],
            dtype=np.float64,
        )
        reward -= 0.02 * float(np.sum(np.square(normalized_ctrl)))

        self.prev_gripper_object_dist = gripper_object_dist
        self.prev_object_target_dist = gripper_target_dist
        self.prev_site_object_dist = site_object_dist
        self.prev_place_object_target_dist = object_target_dist
        self.prev_grasp_z_error = grasp_z_error
        self.prev_place_z_error = place_z_error
        return float(reward), bool(success)

    def reset(self, seed=None, options=None):
        # 每个 episode 重置时随机化物体和目标位置，并清空内部阶段变量。
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        random_free_joint_pose(
            self.model, self.data, OBJECT_JOINT_NAME, self.rng, OBJECT_X_RANGE, OBJECT_Y_RANGE, OBJECT_Z
        )
        random_free_joint_pose(
            self.model, self.data, PLACE_TARGET_JOINT_NAME, self.rng, TARGET_X_RANGE, TARGET_Y_RANGE, TARGET_Z
        )

        self.data.ctrl[self.actuator_x] = 0.0
        self.data.ctrl[self.actuator_y] = 0.0
        self.data.ctrl[self.actuator_z] = HOME_Z
        self.data.ctrl[self.actuator_yaw] = 0.0
        self.data.ctrl[self.actuator_gripper] = OPEN_GRIPPER
        mujoco.mj_forward(self.model, self.data)

        obj_pos = self.data.xpos[self.object_body_id]
        target_pos = self.data.xpos[self.target_body_id]
        gripper_xy = self._get_gripper_center_xy()
        base_z = self._get_base_z_qpos()
        self.prev_gripper_object_dist = float(np.linalg.norm(obj_pos[:2] - gripper_xy))
        self.prev_object_target_dist = float(np.linalg.norm(gripper_xy - target_pos[:2]))
        self.prev_site_object_dist = float(np.linalg.norm(obj_pos - self.data.site_xpos[self.align_site_id]))
        self.prev_grasp_z_error = abs(base_z - GRASP_DESCEND_Z)
        self.prev_place_z_error = abs(base_z - PLACE_DESCEND_Z)
        self.step_count = 0
        self.has_grasp_contact = False
        self.task_success = False
        self.phase = PHASE_REACH_OBJECT
        self.grasp_retry_count = 0
        self.grasp_settle_steps = 0

        if self.task_stage == TASK_STAGE_PLACE:
            success = self._scripted_grasp_warm_start()
            self.has_grasp_contact = bool(success)
            self.phase = PHASE_MOVE_TO_TARGET if success else PHASE_REACH_OBJECT
        return self._get_obs(), {}

    def world_xy_to_ctrl_xy(self, world_xy):
        """Convert world XY target to [base_actuator_x, base_actuator_y] control."""
        # 目前是直接线性映射，保留这个接口是为了后续接入更复杂的标定。
        return np.array([world_xy[0], world_xy[1]], dtype=np.float64)

    def step(self, action):
        # 标准 Gym step：先应用动作，再仿真若干子步，最后计算 reward 和终止条件。
        self._apply_action(action)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        reward, success = self._compute_reward()
        terminated = success
        truncated = self.step_count >= self.max_episode_steps
        info = {"is_success": success, "task_success": self.task_success, "phase": self.phase}

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        # 使用被动 viewer，同步当前仿真状态到窗口。
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        # 关闭 viewer，避免残留窗口句柄。
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# -----------------------------
# 训练 / 评估
# -----------------------------
def make_env(seed=None, render_mode=None, max_episode_steps=500, task_stage=TASK_STAGE_FULL):
    # Stable-Baselines3 的 DummyVecEnv 需要一个无参构造器，这里返回闭包。
    def _init():
        env = Tactile2F85PickPlaceEnv(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            seed=seed,
            task_stage=task_stage,
        )
        return Monitor(env)

    return _init


def train(args):
    # 训练入口：可选 BC 预训练 + PPO 主训练 + checkpoint/eval 保存。
    task_stage = resolve_task_stage(args.task_stage, args.init_model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log = None if args.no_tensorboard else str(output_dir / "tb")

    observations = actions = None
    if args.bc_pretrain_episodes > 0:
        observations, actions, collected_episodes, attempts = collect_expert_dataset(
            task_stage=task_stage,
            seed=args.seed,
            episodes=args.bc_pretrain_episodes,
            max_episode_steps=args.max_episode_steps,
            max_attempts=args.bc_max_attempts,
        )
        dataset_path = output_dir / "expert_bc_dataset.npz"
        np.savez_compressed(dataset_path, observations=observations, actions=actions)
        print(
            f"[bc] saved expert dataset to {dataset_path} "
            f"(episodes={collected_episodes}, attempts={attempts}, transitions={len(observations)})"
        )

    train_env = DummyVecEnv(
        [
            make_env(
                seed=args.seed + i,
                max_episode_steps=args.max_episode_steps,
                task_stage=task_stage,
            )
            for i in range(args.n_envs)
        ]
    )
    eval_env = DummyVecEnv(
        [make_env(seed=args.seed + 1, max_episode_steps=args.max_episode_steps, task_stage=task_stage)]
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_2f85",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    if args.init_model:
        model = PPO.load(str(args.init_model), env=train_env)
        model.tensorboard_log = tensorboard_log
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            tensorboard_log=tensorboard_log,
            seed=args.seed,
        )

    if observations is not None and actions is not None:
        bc_pretrain_policy(
            model,
            observations=observations,
            actions=actions,
            epochs=args.bc_pretrain_epochs,
            batch_size=args.bc_pretrain_batch_size,
            lr=args.bc_pretrain_lr,
        )

    model.learn(total_timesteps=args.total_timesteps, callback=[checkpoint_callback, eval_callback], reset_num_timesteps=not bool(args.init_model))

    final_path = output_dir / "ppo_2f85_final"
    model.save(str(final_path))
    save_task_stage_metadata(final_path.with_suffix(".zip"), task_stage)
    best_model_path = output_dir / "best_model" / "best_model.zip"
    if best_model_path.exists():
        save_task_stage_metadata(best_model_path, task_stage)
    train_env.close()
    eval_env.close()
    print(f"[rl] saved final model: {final_path}.zip")
    print(f"[rl] best model directory: {output_dir / 'best_model'}")


def evaluate(args):
    # 载入训练好的模型，跑固定回合数评估 success rate。
    model_path = Path(args.model)
    if model_path.suffix != ".zip" and not model_path.exists():
        zip_path = model_path.with_suffix(".zip")
        if zip_path.exists():
            model_path = zip_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    task_stage = resolve_task_stage(args.task_stage, model_path)

    env = Tactile2F85PickPlaceEnv(
        render_mode=None if args.no_render else "human",
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        task_stage=task_stage,
    )
    model = PPO.load(str(model_path), env=env)

    successes = 0
    for episode in range(args.eval_episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if not args.no_render:
                time.sleep(1.0 / 60.0)

        success = bool(info.get("is_success", False) or info.get("task_success", False))
        successes += int(success)
        print(
            f"[eval] episode={episode + 1}/{args.eval_episodes}, "
            f"success={success}, reward={total_reward:.2f}, steps={env.step_count}"
        )

    env.close()
    print(f"[eval] success_rate={successes / args.eval_episodes:.3f}")


def heuristic_action(env):
    # 启发式 baseline，用阶段机直接输出动作，既可调试也可采集 BC 数据。
    obj_pos = env.data.xpos[env.object_body_id].copy()
    target_pos = env.data.xpos[env.target_body_id].copy()
    gripper_xy = env._get_gripper_center_xy()
    base_z = env._get_base_z_qpos()

    _, _, both_touch = env._get_touch_state()
    env.has_grasp_contact = env.has_grasp_contact or bool(both_touch)

    if env.phase == PHASE_LIFT_OBJECT and abs(base_z - HOME_Z) < GRASP_RETRY_HOME_Z_THRESHOLD:
        if obj_pos[2] <= LIFT_OBJECT_Z_THRESHOLD:
            env.has_grasp_contact = False
            if env.task_stage != TASK_STAGE_PLACE:
                env.phase = PHASE_REACH_OBJECT

    if env.phase == PHASE_MOVE_TO_TARGET:
        object_near_site = np.linalg.norm(obj_pos[:2] - gripper_xy) < 0.045
        if not object_near_site and obj_pos[2] <= LIFT_OBJECT_Z_THRESHOLD:
            env.has_grasp_contact = False
            if env.task_stage != TASK_STAGE_PLACE:
                env.phase = PHASE_REACH_OBJECT

    action = np.zeros(5, dtype=np.float32)
    if env.phase == PHASE_REACH_OBJECT:
        xy_error = obj_pos[:2] - gripper_xy
        action[0:2] = np.clip(world_xy_delta_to_ctrl_delta(xy_error) / HEURISTIC_REACH_GAIN, -0.6, 0.6)
        action[2] = np.clip((HOME_Z - base_z) / 0.03, -1.0, 1.0)
        action[4] = -1.0

    elif env.phase == PHASE_GRASP_OBJECT:
        xy_error = obj_pos[:2] - gripper_xy
        if np.linalg.norm(xy_error) > GRASP_CENTER_XY_THRESHOLD:
            action[0:2] = np.clip(world_xy_delta_to_ctrl_delta(xy_error) / HEURISTIC_REACH_GAIN, -0.5, 0.5)
        if np.linalg.norm(xy_error) > GRASP_CENTER_XY_THRESHOLD:
            action[2] = 0.0
            action[4] = -1.0
        elif abs(base_z - GRASP_DESCEND_Z) > GRASP_Z_THRESHOLD:
            action[2] = np.clip((GRASP_DESCEND_Z - base_z) / 0.03, -1.0, 1.0)
            action[4] = -1.0
        else:
            action[4] = 1.0

    elif env.phase == PHASE_LIFT_OBJECT:
        action[2] = np.clip((HOME_Z - base_z) / 0.03, -1.0, 1.0)
        action[4] = 1.0

    elif env.phase == PHASE_MOVE_TO_TARGET:
        xy_error = target_pos[:2] - gripper_xy
        action[0:2] = np.clip(world_xy_delta_to_ctrl_delta(xy_error) / HEURISTIC_TARGET_GAIN, -0.7, 0.7)
        action[2] = np.clip((HOME_Z - base_z) / 0.03, -1.0, 1.0)
        action[4] = 1.0

    elif env.phase == PHASE_PLACE_OBJECT:
        xy_error = target_pos[:2] - gripper_xy
        if np.linalg.norm(xy_error) > MOVE_TARGET_XY_THRESHOLD:
            action[0:2] = np.clip(world_xy_delta_to_ctrl_delta(xy_error) / HEURISTIC_TARGET_GAIN, -0.4, 0.4)
        if abs(base_z - PLACE_DESCEND_Z) > PLACE_Z_THRESHOLD:
            action[2] = np.clip((PLACE_DESCEND_Z - base_z) / 0.03, -1.0, 1.0)
            action[4] = 1.0
        else:
            action[2] = 0.0
            action[4] = -1.0

    return action


def evaluate_heuristic(args):
    # 用脚本策略直接跑任务，用于 sanity check 和数据采集。
    task_stage = resolve_task_stage(args.task_stage, None)
    env = Tactile2F85PickPlaceEnv(
        render_mode=None if args.no_render else "human",
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        task_stage=task_stage,
    )
    successes = 0
    for episode in range(args.eval_episodes):
        env.reset(seed=args.seed + episode)
        done = False
        total_reward = 0.0
        while not done:
            obs, reward, terminated, truncated, info = env.step(heuristic_action(env))
            total_reward += reward
            done = terminated or truncated
            if not args.no_render:
                time.sleep(1.0 / 60.0)

        success = bool(info.get("is_success", False) or info.get("task_success", False))
        successes += int(success)
        print(
            f"[heuristic] episode={episode + 1}/{args.eval_episodes}, "
            f"success={success}, reward={total_reward:.2f}, steps={env.step_count}, "
            f"phase={info.get('phase')}"
        )
    env.close()
    print(f"[heuristic] success_rate={successes / args.eval_episodes:.3f}")


def parse_args():
    # 所有训练/评估参数都统一放在这里，避免散落在代码各处。
    parser = argparse.ArgumentParser(description="PPO training/evaluation for 2F85 tactile pick-and-place.")
    parser.add_argument("--eval", action="store_true", help="Run inference/evaluation instead of training.")
    parser.add_argument("--heuristic", action="store_true", help="Run scripted baseline evaluation for debugging.")
    parser.add_argument("--model", type=str, default="runs/ppo_2f85/ppo_2f85_final.zip")
    parser.add_argument("--output-dir", type=str, default="runs/ppo_2f85")
    parser.add_argument("--init-model", type=str, default=None, help="Optional checkpoint to resume from.")
    parser.add_argument(
        "--task-stage",
        type=str,
        default=TASK_STAGE_FULL,
        choices=[TASK_STAGE_AUTO, TASK_STAGE_FULL, TASK_STAGE_GRASP, TASK_STAGE_PLACE],
        help="Train/evaluate full task, grasp only, place only, or auto-load from metadata.",
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-render", action="store_true", help="Disable MuJoCo viewer during --eval.")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging.")
    parser.add_argument("--check-env", action="store_true", help="Run Stable-Baselines3 env checker and exit.")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel DummyVecEnv environments.")
    parser.add_argument("--bc-pretrain-episodes", type=int, default=5, help="Number of successful heuristic episodes for BC warm start.")
    parser.add_argument("--bc-pretrain-epochs", type=int, default=10, help="BC warm-start epochs on expert data.")
    parser.add_argument("--bc-pretrain-batch-size", type=int, default=256, help="BC warm-start batch size.")
    parser.add_argument("--bc-pretrain-lr", type=float, default=1e-4, help="Learning rate for BC warm-start.")
    parser.add_argument("--bc-max-attempts", type=int, default=20, help="Maximum heuristic rollout attempts to collect expert episodes.")
    return parser.parse_args()


def main():
    # 命令行入口：先处理环境检查，再分发到训练、评估或启发式模式。
    args = parse_args()
    if args.check_env:
        env = Tactile2F85PickPlaceEnv(seed=args.seed, task_stage=resolve_task_stage(args.task_stage))
        check_env(env, warn=True)
        env.close()
        print("[rl] environment check passed")
        return

    if args.heuristic:
        evaluate_heuristic(args)
    elif args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

# check: python touch_2f85_visualization_rgb_8_16_RL_test.py --check-env
# heuristic: python touch_2f85_visualization_rgb_8_16_RL_test.py --heuristic --task-stage full --eval-episodes 5

# train+bc: python touch_2f85_visualization_rgb_8_16_RL_test.py --task-stage full --bc-pretrain-episodes 5 --bc-pretrain-epochs 10 --total-timesteps 2000000 --n-envs 8
# eval: python touch_2f85_visualization_rgb_8_16_RL_test.py --model runs/ppo_2f85/ppo_2f85_final.zip --eval --eval-episodes 5


# 用tensorboard查看训练状态
# tensorboard --logdir runs/ppo_2f85/tb


"""

现在的流程是：

  1. 先跑 heuristic 专家采集，默认收 5 个成功回合
  2. 把这些轨迹保存成 runs/ppo_2f85/expert_bc_dataset.npz
  3. 用这批专家数据做 BC warm-start
  4. 再继续 PPO 训练

  你现在直接这样跑 full 任务就行：

  cd /home/hjx/hjx_file/STF/Force_Gripper_Arm/mujoco/tactile_gripper/demo_gripper_2f85
  python touch_2f85_visualization_rgb_8_16_RL_test.py \
    --task-stage full \
    --bc-pretrain-episodes 5 \
    --bc-pretrain-epochs 10 \
    --total-timesteps 1000000 \
    --n-envs 4

  如果你想只看专家轨迹本身，还是可以单独跑：

  python touch_2f85_visualization_rgb_8_16_RL_test.py --heuristic --task-stage full --eval-episodes 5

"""
