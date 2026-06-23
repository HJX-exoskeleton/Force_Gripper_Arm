import logging
import time
from enum import Enum
from pathlib import Path

import cv2
import mujoco  # pip install mujoco==3.3.0
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)


# 触觉阵列与主循环频率配置。当前 2F85 夹爪每侧有 8x16 个 force sensor。
TOUCH_ROWS = 8
TOUCH_COLS = 16
CONTROL_FREQUENCY = 60.0
FORCE_MAX = 1.0
CONTACT_THRESHOLD = 0.02

# MuJoCo XML 中使用到的 site/body/geom/joint 名称。
ALIGN_SITE_NAME = "ee_site"
OBJECT_BODY_NAME = "red_cylinder"
OBJECT_GEOM_NAME = "red_cylinder"
OBJECT_JOINT_NAME = "red_cylinder_joint"
PLACE_TARGET_BODY_NAME = "target_box"
PLACE_TARGET_GEOM_NAME = "green_target_box"
PLACE_TARGET_JOINT_NAME = "target_box_joint"
ALIGN_TOLERANCE = 0.003
SAFE_ALIGN_HEIGHT_OFFSET = 0.0
GRASP_DESCEND_Z = -0.135
GRASP_Z_TOLERANCE = 0.01
GRASP_DESCEND_TIMEOUT = 2.5

# 放置阶段的高度安全配置：以绿色目标体顶部高度为基准，避免夹爪下压碰撞目标区。
PLACE_TARGET_CLEARANCE = 0.001

PLACE_CONTACT_THRESHOLD = 0.005

# XY 到位判定阈值。ctrl/qpos/object 三者同时满足时才进入下一阶段，避免命令到位但实体未到位。
XY_TOLERANCE = 0.01
XY_CTRL_TOLERANCE = 0.006
XY_QPOS_TOLERANCE = 0.02
OBJECT_XY_TOLERANCE = 0.03
PICK_XY_TOLERANCE = 0.018
HOME_CTRL_TOLERANCE = 0.005
HOME_QPOS_TOLERANCE = 0.03
LIFT_TIMEOUT = 3.0
PICK_SETTLE_TIME = 0.6
PLACE_SETTLE_TIME = 0.6
XY_WAIT_LOG_INTERVAL = 2.0

# 每次运行随机生成绿色目标区和红色圆柱的初始位置。
TARGET_X_RANGE = (0.2, 0.5)
TARGET_Y_RANGE = (0.3, 0.6)
TARGET_Z = 0.0
OBJECT_X_RANGE = (-0.2, 0.0)
OBJECT_Y_RANGE = (-0.2, 0.0)
OBJECT_Z = 0.0

# 夹爪底座 z 方向和夹爪开合控制量。fingers_actuator 的 ctrlrange 为 0~255。
HOME_Z = 0.0
DOWN_LIMIT_Z = -0.135
OPEN_GRIPPER = 0.0
CLOSED_GRIPPER = 230.0

DESCEND_SPEED = 0.08
PLACE_DESCEND_SPEED = 0.035
LIFT_SPEED = 0.10
XY_SPEED = 0.035
GRIPPER_SPEED = 180.0
MAX_GRASP_RETRIES = 5
GRASP_VERIFY_OBJECT_Z = 0.015


class DemoState(Enum):
    """Pick-and-place 状态机。"""

    OPENING = "opening"
    MOVING_TO_OBJECT = "moving_to_object"
    SETTLING_OVER_OBJECT = "settling_over_object"
    DESCENDING = "descending"
    CLOSING = "closing"
    LIFTING = "lifting"
    MOVING_TO_TARGET = "moving_to_target"
    SETTLING_OVER_TARGET = "settling_over_target"
    DESCENDING_TO_PLACE = "descending_to_place"
    RELEASING = "releasing"
    FINAL_LIFTING = "final_lifting"
    FINISHED = "finished"


def resolve_xml_path():
    """基于脚本位置解析 XML，避免依赖硬编码绝对路径。"""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parents[1] / "assets_robot_xml" / "gripper_2f85" / "scene_8_16.xml"


def build_touch_address_maps(model):
    """预计算左右触觉阵列在 data.sensordata 中的起始地址。"""
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
    """读取左右触觉阵列，将每个 force sensor 的 3D 力向量转成力模长。"""
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


def show_touch_heatmaps(touch_right, touch_left, state, elapsed):
    """用 OpenCV 显示左右触觉热力图，并叠加当前状态名。"""
    def to_heatmap(touch):
        normalized = np.clip(touch, 0.0, FORCE_MAX) / FORCE_MAX * 255
        colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        resized = cv2.resize(colored, (480, 240))
        return np.ascontiguousarray(np.rot90(resized, k=-1))

    right_img = to_heatmap(touch_right)
    left_img = to_heatmap(touch_left)
    text = f"{state.value}  t={elapsed:.1f}s"

    cv2.putText(right_img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(left_img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imshow("Touch Heatmap - Sensor Right", right_img)
    cv2.imshow("Touch Heatmap - Sensor Left", left_img)


def move_towards(current, target, speed, dt):
    """按最大速度限制把 current 平滑推进到 target。"""
    max_step = speed * dt
    return current + np.clip(target - current, -max_step, max_step)


def get_joint_qpos(model, data, joint_name):
    joint_id = model.joint(joint_name).id
    return data.qpos[model.jnt_qposadr[joint_id]]


def get_site_z(data, site_id):
    return float(data.site_xpos[site_id][2])


def get_body_center_z(data, body_id):
    return float(data.xpos[body_id][2])


def get_body_xy(data, body_id):
    return data.xpos[body_id][:2].copy()


def get_base_xy_qpos(model, data):
    """读取夹爪底座 x/y 滑动关节的实际 qpos，用于判断实体是否到位。"""
    joint_x = model.joint("base_mount_joint_x").id
    joint_y = model.joint("base_mount_joint_y").id
    qpos_x = data.qpos[model.jnt_qposadr[joint_x]]
    qpos_y = data.qpos[model.jnt_qposadr[joint_y]]
    return np.array([qpos_x, qpos_y], dtype=np.float64)


def get_cylinder_top_z(model, data, geom_id):
    return float(data.geom_xpos[geom_id][2] + model.geom_size[geom_id][1])


def are_geoms_in_contact(data, geom_a_id, geom_b_id):
    """检测两个 geom 在当前 MuJoCo contact 列表中是否发生接触。"""
    for i in range(data.ncon):
        contact = data.contact[i]
        if {contact.geom1, contact.geom2} == {geom_a_id, geom_b_id}:
            return True
    return False


def randomize_free_joint_position(model, data, joint_name, x_range, y_range, z_value):
    """随机设置 free joint 的初始位姿，并清零速度。"""
    rng = np.random.default_rng()
    target_pos = np.array(
        [
            rng.uniform(*x_range),
            rng.uniform(*y_range),
            z_value,
        ],
        dtype=np.float64,
    )

    joint_id = model.joint(joint_name).id
    qpos_adr = model.jnt_qposadr[joint_id]
    dof_adr = model.jnt_dofadr[joint_id]

    data.qpos[qpos_adr:qpos_adr + 3] = target_pos
    data.qpos[qpos_adr + 3:qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[dof_adr:dof_adr + 6] = 0.0
    return target_pos


def z_home_reached(data, z_qpos, actuator_z, elapsed):
    """判断 z 方向是否回到空中高度，带控制量到位和超时兜底。"""
    z_ctrl_at_home = abs(float(data.ctrl[actuator_z]) - HOME_Z) < HOME_CTRL_TOLERANCE
    z_qpos_at_home = abs(z_qpos - HOME_Z) < HOME_QPOS_TOLERANCE
    return z_qpos_at_home or (z_ctrl_at_home and elapsed > 0.5) or elapsed > LIFT_TIMEOUT


def get_target_align_z(data, body_id):
    """抓取时让 ee_site 对齐到物体中心上方的安全高度。"""
    return get_body_center_z(data, body_id) + SAFE_ALIGN_HEIGHT_OFFSET


def move_xy_towards_target(data, actuator_x, actuator_y, current_xy, target_xy, dt):
    """按世界坐标误差增量更新底座 x/y 控制量。当前主流程保留该函数作调试备用。"""
    error_xy = target_xy - current_xy
    step_xy = np.clip(error_xy, -XY_SPEED * dt, XY_SPEED * dt)

    data.ctrl[actuator_x] += step_xy[0]
    data.ctrl[actuator_y] += step_xy[1]

    data.ctrl[actuator_x] = np.clip(data.ctrl[actuator_x], -1.0, 1.0)
    data.ctrl[actuator_y] = np.clip(data.ctrl[actuator_y], -1.0, 1.0)
    return float(np.linalg.norm(error_xy))


def world_xy_delta_to_ctrl_delta(delta_xy):
    """Convert world XY correction to base actuator XY correction."""
    return np.array([delta_xy[0], delta_xy[1]], dtype=np.float64)


def move_base_ctrl_xy_towards(data, actuator_x, actuator_y, target_ctrl_xy, dt):
    """直接把底座 x/y 控制量推进到目标控制量，避免夹持物体摆动导致闭环盘旋。"""
    current_ctrl_xy = np.array([data.ctrl[actuator_x], data.ctrl[actuator_y]], dtype=np.float64)
    error_xy = target_ctrl_xy - current_ctrl_xy
    step_xy = np.clip(error_xy, -XY_SPEED * dt, XY_SPEED * dt)

    data.ctrl[actuator_x] += step_xy[0]
    data.ctrl[actuator_y] += step_xy[1]

    data.ctrl[actuator_x] = np.clip(data.ctrl[actuator_x], -1.0, 1.0)
    data.ctrl[actuator_y] = np.clip(data.ctrl[actuator_y], -1.0, 1.0)
    return float(np.linalg.norm(error_xy))


def set_state(new_state, start_time):
    print(f"[demo] state -> {new_state.value}")
    return new_state, start_time


def main():
    xml_path = resolve_xml_path()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    # 较小的物理步长可以减轻接触和多连杆约束导致的微小抖动。
    model.opt.timestep = 0.002

    # 获取执行器和关键 body/site/geom 的 id，后续循环中直接用 id 访问更快。
    actuator_x = model.actuator("base_actuator_x").id
    actuator_y = model.actuator("base_actuator_y").id
    actuator_z = model.actuator("base_actuator_z").id
    actuator_yaw = model.actuator("base_actuator_yaw").id
    actuator_gripper = model.actuator("fingers_actuator").id
    align_site_id = model.site(ALIGN_SITE_NAME).id
    object_body_id = model.body(OBJECT_BODY_NAME).id
    object_geom_id = model.geom(OBJECT_GEOM_NAME).id
    place_target_body_id = model.body(PLACE_TARGET_BODY_NAME).id
    place_target_geom_id = model.geom(PLACE_TARGET_GEOM_NAME).id

    touch_right_adr, touch_left_adr = build_touch_address_maps(model)

    # 每次运行随机初始化红色圆柱和绿色目标区，再 forward 刷新世界坐标。
    randomized_object_pos = randomize_free_joint_position(
        model, data, OBJECT_JOINT_NAME, OBJECT_X_RANGE, OBJECT_Y_RANGE, OBJECT_Z
    )
    randomized_target_pos = randomize_free_joint_position(
        model, data, PLACE_TARGET_JOINT_NAME, TARGET_X_RANGE, TARGET_Y_RANGE, TARGET_Z
    )
    mujoco.mj_forward(model, data)
    place_target_xy = get_body_xy(data, place_target_body_id)
    place_target_center_z = get_body_center_z(data, place_target_body_id)
    place_target_top_z = get_cylinder_top_z(model, data, place_target_geom_id)

    print(f"[demo] xml: {xml_path}")
    print(f"[demo] sensor_num: {model.nsensor}")
    print(
        "[demo] sequence: pick red object -> return home height -> move to target -> place -> lift"
    )
    print(
        f"[demo] place target xy=({place_target_xy[0]:.3f}, {place_target_xy[1]:.3f}), "
        f"center.z={place_target_center_z:.3f}, top.z={place_target_top_z:.3f}"
    )
    print(
        f"[demo] randomized object pos=({randomized_object_pos[0]:.3f}, "
        f"{randomized_object_pos[1]:.3f}, {randomized_object_pos[2]:.3f})"
    )
    print(
        f"[demo] randomized target pos=({randomized_target_pos[0]:.3f}, "
        f"{randomized_target_pos[1]:.3f}, {randomized_target_pos[2]:.3f})"
    )

    data.ctrl[actuator_x] = 0.0
    data.ctrl[actuator_y] = 0.0
    data.ctrl[actuator_z] = HOME_Z
    data.ctrl[actuator_yaw] = 0.0
    data.ctrl[actuator_gripper] = OPEN_GRIPPER

    state = DemoState.OPENING
    state_start = time.time()
    last_time = state_start
    last_xy_wait_log = state_start
    close_success = False
    task_success = False
    grasp_z_ctrl = HOME_Z
    grasp_attempts = 0
    pick_target_ctrl_xy = np.zeros(2, dtype=np.float64)
    place_target_ctrl_xy = np.zeros(2, dtype=np.float64)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        rate = RateLimiter(frequency=CONTROL_FREQUENCY)

        while viewer.is_running():
            now = time.time()
            dt = max(now - last_time, model.opt.timestep)
            last_time = now
            elapsed = now - state_start

            touch_right, touch_left = read_touch_arrays(data, touch_right_adr, touch_left_adr)
            right_force = float(np.max(touch_right))
            left_force = float(np.max(touch_left))
            max_force = max(right_force, left_force)
            both_fingers_contact = right_force > CONTACT_THRESHOLD and left_force > CONTACT_THRESHOLD

            if state == DemoState.OPENING:
                # 初始阶段：保证夹爪在空中并处于张开状态。
                data.ctrl[actuator_z] = move_towards(data.ctrl[actuator_z], HOME_Z, LIFT_SPEED, dt)
                data.ctrl[actuator_gripper] = move_towards(
                    data.ctrl[actuator_gripper], OPEN_GRIPPER, GRIPPER_SPEED, dt
                )
                if elapsed > 0.8:
                    object_xy = get_body_xy(data, object_body_id)
                    ee_site_xy = data.site_xpos[align_site_id][:2].copy()
                    current_ctrl_xy = np.array([data.ctrl[actuator_x], data.ctrl[actuator_y]], dtype=np.float64)
                    pick_target_ctrl_xy = current_ctrl_xy + world_xy_delta_to_ctrl_delta(object_xy - ee_site_xy)
                    state, state_start = set_state(DemoState.MOVING_TO_OBJECT, now)

            elif state == DemoState.MOVING_TO_OBJECT:
                # 用固定目标控制量移动到红色圆柱上方，避免追踪 ee_site 造成闭环盘旋。
                data.ctrl[actuator_z] = HOME_Z
                data.ctrl[actuator_gripper] = OPEN_GRIPPER
                object_xy = get_body_xy(data, object_body_id)
                ee_site_xy = data.site_xpos[align_site_id][:2].copy()
                ee_site_delta = object_xy - ee_site_xy
                ee_site_error = float(np.linalg.norm(ee_site_delta))
                ctrl_error = move_base_ctrl_xy_towards(
                    data, actuator_x, actuator_y, pick_target_ctrl_xy, dt
                )

                if ctrl_error < XY_CTRL_TOLERANCE and ee_site_error < PICK_XY_TOLERANCE:
                    print(
                        "[demo] object area reached, settling before descent, "
                        f"ctrl_error={ctrl_error:.3f}, "
                        f"ee_site_delta=({ee_site_delta[0]:.3f}, {ee_site_delta[1]:.3f}), "
                        f"ee_site_error={ee_site_error:.3f}"
                    )
                    state, state_start = set_state(DemoState.SETTLING_OVER_OBJECT, now)
                elif now - last_xy_wait_log > XY_WAIT_LOG_INTERVAL:
                    print(
                        "[demo] waiting for object area, "
                        f"ctrl_error={ctrl_error:.3f}, "
                        f"ee_site_delta=({ee_site_delta[0]:.3f}, {ee_site_delta[1]:.3f}), "
                        f"ctrl=({float(data.ctrl[actuator_x]):.3f}, {float(data.ctrl[actuator_y]):.3f}), "
                        f"target_ctrl=({pick_target_ctrl_xy[0]:.3f}, {pick_target_ctrl_xy[1]:.3f}), "
                        f"ee_site_error={ee_site_error:.3f}"
                    )
                    last_xy_wait_log = now

            elif state == DemoState.SETTLING_OVER_OBJECT:
                data.ctrl[actuator_x] = pick_target_ctrl_xy[0]
                data.ctrl[actuator_y] = pick_target_ctrl_xy[1]
                data.ctrl[actuator_z] = HOME_Z
                data.ctrl[actuator_gripper] = OPEN_GRIPPER
                if elapsed > PICK_SETTLE_TIME:
                    object_xy = get_body_xy(data, object_body_id)
                    ee_site_xy = data.site_xpos[align_site_id][:2].copy()
                    ee_site_error = float(np.linalg.norm(object_xy - ee_site_xy))
                    print(f"[demo] settled over object, ee_site_error={ee_site_error:.3f}")
                    state, state_start = set_state(DemoState.DESCENDING, now)

            elif state == DemoState.DESCENDING:
                # 到达红色圆柱上方后锁定 x/y，只下降到固定抓取深度，再闭合。
                data.ctrl[actuator_x] = pick_target_ctrl_xy[0]
                data.ctrl[actuator_y] = pick_target_ctrl_xy[1]
                z_qpos = get_joint_qpos(model, data, "base_mount_joint_z")
                data.ctrl[actuator_z] = move_towards(
                    data.ctrl[actuator_z], GRASP_DESCEND_Z, DESCEND_SPEED, dt
                )
                data.ctrl[actuator_gripper] = OPEN_GRIPPER

                if abs(z_qpos - GRASP_DESCEND_Z) < GRASP_Z_TOLERANCE or elapsed > GRASP_DESCEND_TIMEOUT:
                    grasp_z_ctrl = float(data.ctrl[actuator_z])
                    print(
                        "[demo] grasp depth reached, "
                        f"z_qpos={z_qpos:.3f}, z_ctrl={float(data.ctrl[actuator_z]):.3f}"
                    )
                    state, state_start = set_state(DemoState.CLOSING, now)

            elif state == DemoState.CLOSING:
                # 保持当前高度闭合夹爪；两侧触觉均接触则认为抓取成功。
                data.ctrl[actuator_x] = pick_target_ctrl_xy[0]
                data.ctrl[actuator_y] = pick_target_ctrl_xy[1]
                data.ctrl[actuator_z] = grasp_z_ctrl
                data.ctrl[actuator_gripper] = move_towards(
                    data.ctrl[actuator_gripper], CLOSED_GRIPPER, GRIPPER_SPEED, dt
                )

                if both_fingers_contact and elapsed > 0.4:
                    close_success = True
                    print("[demo] grasp contact detected on both fingers")
                    state, state_start = set_state(DemoState.LIFTING, now)
                elif elapsed > 2.5:
                    grasp_attempts += 1
                    close_success = False
                    print(
                        "[demo] close timeout without stable two-finger contact, "
                        f"retrying grasp ({grasp_attempts}/{MAX_GRASP_RETRIES})"
                    )
                    if grasp_attempts >= MAX_GRASP_RETRIES:
                        print("[demo] max grasp retries reached, lifting anyway")
                        state, state_start = set_state(DemoState.LIFTING, now)
                    else:
                        state, state_start = set_state(DemoState.OPENING, now)

            elif state == DemoState.LIFTING:
                # 抓取后回到空中高度，再开始横向移动。
                data.ctrl[actuator_z] = move_towards(data.ctrl[actuator_z], HOME_Z, LIFT_SPEED, dt)
                data.ctrl[actuator_gripper] = CLOSED_GRIPPER

                z_qpos = get_joint_qpos(model, data, "base_mount_joint_z")
                if z_home_reached(data, z_qpos, actuator_z, elapsed):
                    obj_pos = data.xpos[object_body_id].copy()
                    ee_site_pos = data.site_xpos[align_site_id].copy()
                    object_lifted = obj_pos[2] > OBJECT_Z + GRASP_VERIFY_OBJECT_Z
                    object_near_site = np.linalg.norm(obj_pos[:2] - ee_site_pos[:2]) < 0.04
                    result = "success" if close_success else "unknown"
                    print(
                        f"[demo] returned to home height, grasp result: {result}, "
                        f"object_lifted={object_lifted}, object_near_site={object_near_site}, "
                        f"obj.z={obj_pos[2]:.3f}, z_qpos={z_qpos:.3f}, z_ctrl={float(data.ctrl[actuator_z]):.3f}"
                    )
                    if close_success and object_lifted and object_near_site:
                        grasp_attempts = 0
                        object_xy = get_body_xy(data, object_body_id)
                        current_ctrl_xy = np.array([data.ctrl[actuator_x], data.ctrl[actuator_y]], dtype=np.float64)
                        place_target_ctrl_xy = current_ctrl_xy + world_xy_delta_to_ctrl_delta(
                            place_target_xy - object_xy
                        )
                        state, state_start = set_state(DemoState.MOVING_TO_TARGET, now)
                    elif grasp_attempts < MAX_GRASP_RETRIES:
                        grasp_attempts += 1
                        close_success = False
                        print(f"[demo] grasp verification failed, retrying ({grasp_attempts}/{MAX_GRASP_RETRIES})")
                        state, state_start = set_state(DemoState.OPENING, now)
                    else:
                        print("[demo] grasp verification failed after max retries, moving anyway")
                        object_xy = get_body_xy(data, object_body_id)
                        current_ctrl_xy = np.array([data.ctrl[actuator_x], data.ctrl[actuator_y]], dtype=np.float64)
                        place_target_ctrl_xy = current_ctrl_xy + world_xy_delta_to_ctrl_delta(
                            place_target_xy - object_xy
                        )
                        state, state_start = set_state(DemoState.MOVING_TO_TARGET, now)

            elif state == DemoState.MOVING_TO_TARGET:
                # 用固定目标控制量移动到绿色目标区上方，避免夹持物体摆动导致盘旋。
                data.ctrl[actuator_z] = HOME_Z
                data.ctrl[actuator_gripper] = CLOSED_GRIPPER
                object_xy = get_body_xy(data, object_body_id)
                base_xy_qpos = get_base_xy_qpos(model, data)
                xy_error = move_base_ctrl_xy_towards(
                    data, actuator_x, actuator_y, place_target_ctrl_xy, dt
                )
                qpos_error = float(np.linalg.norm(place_target_ctrl_xy - base_xy_qpos))
                object_error = float(np.linalg.norm(place_target_xy - object_xy))

                if (
                    xy_error < XY_CTRL_TOLERANCE
                    and qpos_error < XY_QPOS_TOLERANCE
                    and object_error < OBJECT_XY_TOLERANCE
                ):
                    print(
                        "[demo] target area reached, "
                        f"ctrl_error={xy_error:.3f}, qpos_error={qpos_error:.3f}, "
                        f"object_error={object_error:.3f}"
                    )
                    state, state_start = set_state(DemoState.SETTLING_OVER_TARGET, now)
                elif now - last_xy_wait_log > XY_WAIT_LOG_INTERVAL:
                    print(
                        "[demo] waiting for target area, "
                        f"ctrl_error={xy_error:.3f}, qpos_error={qpos_error:.3f}, "
                        f"object_error={object_error:.3f}, "
                        f"target_ctrl=({place_target_ctrl_xy[0]:.3f}, {place_target_ctrl_xy[1]:.3f})"
                    )
                    last_xy_wait_log = now

            elif state == DemoState.SETTLING_OVER_TARGET:
                data.ctrl[actuator_x] = place_target_ctrl_xy[0]
                data.ctrl[actuator_y] = place_target_ctrl_xy[1]
                data.ctrl[actuator_z] = HOME_Z
                data.ctrl[actuator_gripper] = CLOSED_GRIPPER
                if elapsed > PLACE_SETTLE_TIME:
                    object_xy = get_body_xy(data, object_body_id)
                    object_error = float(np.linalg.norm(place_target_xy - object_xy))
                    print(f"[demo] settled over target, object_error={object_error:.3f}")
                    state, state_start = set_state(DemoState.DESCENDING_TO_PLACE, now)

            elif state == DemoState.DESCENDING_TO_PLACE:
                # 放置下探以绿色目标体顶部高度为基准，并用触觉接触做提前释放保护。
                data.ctrl[actuator_x] = place_target_ctrl_xy[0]
                data.ctrl[actuator_y] = place_target_ctrl_xy[1]
                data.ctrl[actuator_gripper] = CLOSED_GRIPPER
                site_z = get_site_z(data, align_site_id)
                target_place_z = place_target_top_z + PLACE_TARGET_CLEARANCE
                height_error = site_z - target_place_z
                z_qpos = get_joint_qpos(model, data, "base_mount_joint_z")

                if height_error > ALIGN_TOLERANCE:
                    data.ctrl[actuator_z] = move_towards(
                        data.ctrl[actuator_z], DOWN_LIMIT_Z, PLACE_DESCEND_SPEED, dt
                    )
                    if z_qpos < DOWN_LIMIT_Z + 0.01:
                        print("[demo] down limit reached before place height alignment")
                        state, state_start = set_state(DemoState.RELEASING, now)
                    elif max_force > PLACE_CONTACT_THRESHOLD:
                        print("[demo] touch contact detected during place descent, releasing early")
                        state, state_start = set_state(DemoState.RELEASING, now)
                else:
                    print(
                        "[demo] place height aligned "
                        f"{ALIGN_SITE_NAME}.z={site_z:.3f} with target.z={target_place_z:.3f}"
                    )
                    state, state_start = set_state(DemoState.RELEASING, now)

            elif state == DemoState.RELEASING:
                # 张开夹爪释放红色圆柱。
                data.ctrl[actuator_gripper] = move_towards(
                    data.ctrl[actuator_gripper], OPEN_GRIPPER, GRIPPER_SPEED, dt
                )
                if elapsed > 1.0:
                    print("[demo] released object at target area")
                    state, state_start = set_state(DemoState.FINAL_LIFTING, now)

            elif state == DemoState.FINAL_LIFTING:
                # 释放后抬起夹爪，完成一次 pick-and-place。
                data.ctrl[actuator_z] = move_towards(data.ctrl[actuator_z], HOME_Z, LIFT_SPEED, dt)
                data.ctrl[actuator_gripper] = OPEN_GRIPPER

                z_qpos = get_joint_qpos(model, data, "base_mount_joint_z")
                if z_home_reached(data, z_qpos, actuator_z, elapsed):
                    result = "SUCCESS" if task_success else "FAILED"
                    print(f"[demo] pick-and-place finished, task result: {result}")
                    state, state_start = set_state(DemoState.FINISHED, now)

            elif state == DemoState.FINISHED:
                data.ctrl[actuator_z] = HOME_Z
                data.ctrl[actuator_gripper] = OPEN_GRIPPER

            mujoco.mj_step(model, data)
            if not task_success and are_geoms_in_contact(data, object_geom_id, place_target_geom_id):
                task_success = True
                print("[demo] task success detected: red_cylinder is in contact with target_box")
            show_touch_heatmaps(touch_right, touch_left, state, now - state_start)
            viewer.sync()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            rate.sleep()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
