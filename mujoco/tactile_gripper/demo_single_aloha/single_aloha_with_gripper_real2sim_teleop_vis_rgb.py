import mujoco
import mujoco.viewer
import numpy as np
import cv2
import glfw
import h5py
import logging
logging.getLogger().setLevel(logging.ERROR)
import threading
from pathlib import Path
from loop_rate_limiters import RateLimiter

# from tf.transformations import *
from transformations import *  # pip install transformations

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from tqdm import tqdm
from keyboard_control import *  # 键盘方向键输入监听


# === 初始化模型与数据 ===
model = mujoco.MjModel.from_xml_path("/home/hjx/hjx_file/STF/Force_Gripper/mujoco/assets_robot_xml/single_aloha_with_gripper/single_viperx_cupboard_tactile.xml")
data = mujoco.MjData(model)
# model.opt.timestep = 0.005  # 默认为0.001； 加快仿真步速（逻辑仿真更快）；注意：步长过大会影响物理精度，建议最多提升至 0.01


# 打印传感器数量
print(f"sensor_num: {model.nsensor}")

# 存储传感器的内存地址
touch_point_adr_right = [[0] * 16 for _ in range(16)]
touch_point_adr_left = [[0] * 16 for _ in range(16)]

for x in range(16):
    for y in range(16):
        idx = x * 16 + y
        touch_point_adr_right[x][y] = model.sensor_adr[idx]  # 第一个传感器
        touch_point_adr_left[x][y] = model.sensor_adr[idx + model.nsensor // 2]  # 第二个传感器在后半部分的地址


# === 获取双臂动作 ===
def get_action_arm(left_bot):

    action = np.zeros(7)
    action[:6] = left_bot.dxl.joint_states.position[:6]
    action[6] = left_bot.dxl.joint_states.position[6]

    return action


def map_gripper_control(actual_value):
    # 定义实际控制值的范围和仿真控制值的范围
    min_actual = -0.4
    max_actual = 0.285
    min_sim = 0
    max_sim = 255

    # 使用线性映射公式
    sim_value = ((actual_value - min_actual) / (max_actual - min_actual)) * (min_sim - max_sim) + max_sim

    return sim_value


# 机械臂夹爪
gripper_control = model.actuator("fingers_actuator").id

# 初始化真实机器人控制器（左、右）
left_bot = InterbotixManipulatorXS("wx250s", "arm", "gripper", 'master_left', init_node=True)

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


# === 主程序入口 ===
if __name__ == "__main__":

    # 初始化一个频率限制器，用于控制仿真循环运行频率为 200Hz（即每5ms一步）
    rate = RateLimiter(frequency=200.0)
    # # 启动 MuJoCo 的可视化窗口（被动模式），可显示仿真过程但不允许交互式控制 UI 面板
    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)

    while viewer.is_running():
        # 读取真实机器人状态并映射到仿真控制器
        action = get_action_arm(left_bot)
        data.ctrl[0:6] = action[0:6]  # 左臂
        # data.ctrl[gripper_control] = action[6]  # 主动端夹爪张开对应数值0.285; 主动端夹爪闭合对应数值(-0.4)

        # 将实际控制值映射到仿真值
        sim_gripper_value = map_gripper_control(action[6])
        # 将映射后的仿真值应用到仿真模型中
        data.ctrl[gripper_control] = sim_gripper_value

        mujoco.mj_step(model, data)

        # 创建两个16x16的矩阵来存储两个触觉传感器的数据
        touch_right = np.zeros((16, 16), dtype=np.float32)
        touch_left = np.zeros((16, 16), dtype=np.float32)

        force_max = 1

        for x in range(16):
            for y in range(16):
                adr_right = touch_point_adr_right[x][y]
                adr_left = touch_point_adr_left[x][y]

                # 获取第一个触觉传感器的数据
                data_right = mujoco.mju_norm3(data.sensordata[adr_right:adr_right + 3])
                touch_right[x, y] = mujoco.mju_clip(data_right, 0.0, force_max)

                # 获取第二个触觉传感器的数据
                data_left = mujoco.mju_norm3(data.sensordata[adr_left:adr_left + 3])
                touch_left[x, y] = mujoco.mju_clip(data_left, 0.0, force_max)

        # 将触觉传感器数据标准化到0到255之间
        touch_right_normalized = np.clip(touch_right, 0, force_max) / force_max * 255
        touch_left_normalized = np.clip(touch_left, 0, force_max) / force_max * 255

        # 使用OpenCV的COLORMAP_VIRIDIS来创建热力图效果
        touch_colored_right = cv2.applyColorMap(touch_right_normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        touch_colored_left = cv2.applyColorMap(touch_left_normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)

        # 调整热力图大小
        touch_colored_right_resized = cv2.resize(touch_colored_right, (480, 480))
        touch_colored_left_resized = cv2.resize(touch_colored_left, (480, 480))

        # 分别显示两个热力图在不同的窗口
        cv2.imshow("Touch Heatmap - Sensor Right", touch_colored_right_resized)
        cv2.imshow("Touch Heatmap - Sensor Left", touch_colored_left_resized)

        # cv2.waitKey(1)

        viewer.sync()
        rate.sleep()

        # top camera view
        viewport = mujoco.MjrRect(0, 0, 640, 480)
        mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, context)
        rgb = np.flipud(rgb)
        cv2.imshow("Top Camera View", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    viewer.close()
