import mujoco
import mujoco.viewer
import numpy as np
import cv2
import glfw
import h5py
import mink
import logging
logging.getLogger().setLevel(logging.ERROR)

from pathlib import Path
from loop_rate_limiters import RateLimiter

# from tf.transformations import *
from transformations import *  # pip install transformations

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from tqdm import tqdm
from keyboard_control import *  # 键盘方向键输入监听


# === 夹爪开合映射 ===
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213


def map_gripper(x):
    # 将真实夹爪位置归一化映射到仿真夹爪 joint 区间
    norm = (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
    norm = np.clip(norm, 0.0, 1.0)
    return PUPPET_GRIPPER_JOINT_CLOSE + norm * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)


# === 获取双臂动作 ===
def get_action_arm(left_bot):
    action = np.zeros(7)
    action[:6] = left_bot.dxl.joint_states.position[:6]
    # action[7:13] = right_bot.dxl.joint_states.position[:6]
    action[6] = map_gripper(left_bot.dxl.joint_states.position[6])
    # action[13] = map_gripper(right_bot.dxl.joint_states.position[6])
    return action


# === 保存数据至 HDF5 文件 ===
def save_to_hdf5(path, data):
    print(f"Saving teleop data to {path}...")
    with h5py.File(path, 'w') as f:
        f.attrs['sim'] = True
        obs_grp = f.create_group('observations')
        obs_grp.create_dataset('qpos', data=np.stack(data['/observations/qpos']))
        obs_grp.create_dataset('qvel', data=np.stack(data['/observations/qvel']))
        image_grp = obs_grp.create_group('images')
        image_grp.create_dataset('top', data=np.stack(data['/observations/images/top']), dtype='uint8')
        f.create_dataset('action', data=np.stack(data['/action']))
    print(f"\u2705 Saved {len(data['/action'])} steps.")


# 加载 Mujoco 模型
xml_path = "/home/hjx/hjx_file/STF/Force_Gripper/mujoco/assets_robot_xml/single_aloha_tactile/single_viperx_cupboard.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
model.opt.timestep = 0.005  # 默认为0.001； 加快仿真步速（逻辑仿真更快）；注意：步长过大会影响物理精度，建议最多提升至 0.01
configuration = mink.Configuration(model)
model, data = configuration.model, configuration.data

# 初始化任务（例如底盘任务）
base_task = mink.FrameTask("vx300s_left", "body", 0.1, 1.0)
configuration.update_from_keyframe("home")
base_task.set_target_from_configuration(configuration)

# 获取 actuator ID（底盘、夹爪）
# left_wheel = model.actuator("diablo_left_wheel").id
# right_wheel = model.actuator("diablo_right_wheel").id
l_finger_l = model.actuator("vx300s_left/left_finger_link").id
l_finger_r = model.actuator("vx300s_left/right_finger_link").id
# r_finger_l = model.actuator("vx300s_right_left_finger").id
# r_finger_r = model.actuator("vx300s_right_right_finger").id

# 初始化真实机器人控制器（左、右）
left_bot = InterbotixManipulatorXS("wx250s", "arm", "gripper", 'master_left', init_node=True)
# right_bot = InterbotixManipulatorXS("wx250s", "arm", "gripper", 'master_right', init_node=False)

# 初始化 GLFW 与离屏渲染上下文
glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(640, 480, "offscreen", None, None)
glfw.make_context_current(window)

# 相机设置（top 视角）
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
cam = mujoco.MjvCamera()
cam.fixedcamid = cam_id
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
scene = mujoco.MjvScene(model, maxgeom=1000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)

# 初始化保存与仿真参数
save_path = "/home/hjx/hjx_file/STF/Force_Gripper/data/data_tactile_single_aloha/episode_0.hdf5"

data_dict = {
    '/observations/qpos': [],
    '/observations/qvel': [],
    '/observations/images/top': [],
    '/action': []
}


# === 主程序入口 ===
if __name__ == "__main__":

    # 初始化一个频率限制器，用于控制仿真循环运行频率为 200Hz（即每5ms一步）
    rate = RateLimiter(frequency=200.0)
    # # 启动 MuJoCo 的可视化窗口（被动模式），可显示仿真过程但不允许交互式控制 UI 面板
    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    # # 设置图像/状态数据保存的间隔步数。比如如果仿真步长是 0.005s，每 0.02s 采样一次，则每 0.02 / 0.005 = 4 步记录一次
    record_interval = int(0.02 / model.opt.timestep)

    step = 0  # 仿真总步数的初始化（step 是当前仿真进行到的步数）
    record_count = 0  # 记录图像和状态数据的计数器，每达到 record_interval 就记录一次数据
    max_steps = 6000  # 最大仿真步数，仿真到这个步数后自动退出主循环（例如 6000 步）
    wheel_base = 1.0  # 机器人底盘左右轮之间的距离（用于差速控制时计算左轮和右轮速度）, 可调节

    # 使用 tqdm 显示仿真进度条
    progress_bar = tqdm(total=max_steps, desc="\u4eff\u771f\u8fdb\u5ea6", ncols=100)

    while viewer.is_running():
        step += 1
        if step >= max_steps:
            break

        # 处理键盘控制逻辑
        # fwd, yaw = 0.0, 0.0
        # if key_states[keyboard.Key.up]: fwd -= 20
        # if key_states[keyboard.Key.down]: fwd += 20
        # if key_states[keyboard.Key.left]: yaw -= 20
        # if key_states[keyboard.Key.right]: yaw += 20
        # v_l = fwd - yaw * wheel_base / 2
        # v_r = fwd + yaw * wheel_base / 2
        # data.ctrl[left_wheel] = v_l
        # data.ctrl[right_wheel] = v_r

        # 读取真实机器人状态并映射到仿真控制器
        action = get_action_arm(left_bot)
        data.ctrl[0:6] = action[0:6]  # 左臂
        # data.ctrl[8:14] = action[7:13]  # 右臂
        data.ctrl[l_finger_l] = action[6]
        data.ctrl[l_finger_r] = -action[6]
        # data.ctrl[r_finger_l] = action[13]
        # data.ctrl[r_finger_r] = -action[13]

        mujoco.mj_step(model, data)
        viewer.sync()
        rate.sleep()

        progress_bar.update(1)  # 每步推进仿真后，更新 tqdm 进度条

        # 每隔一段时间采样并记录观测值 + 相机图像
        record_count += 1
        if record_count >= record_interval:
            record_count = 0
            qpos = data.qpos.copy()
            qvel = data.qvel.copy()

            viewport = mujoco.MjrRect(0, 0, 640, 480)
            mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(rgb, None, viewport, context)
            rgb = np.flipud(rgb)
            cv2.imshow("Top Camera View", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            data_dict['/observations/qpos'].append(qpos)
            data_dict['/observations/qvel'].append(qvel)
            data_dict['/action'].append(action)
            data_dict['/observations/images/top'].append(rgb.copy())

    save_to_hdf5(save_path, data_dict)  # 保存数据
    cv2.destroyAllWindows()
    viewer.close()
