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


# === 参数设置 ===
FPS = 50
touch_shape = (3, 16, 16)  # 原始 shape: (C，H, W)
max_shear = 0.05
max_pressure = 0.1
window_size = (480, 480)

# === 初始化模型与数据 ===
model = mujoco.MjModel.from_xml_path("/home/hjx/hjx_file/STF/Force_Gripper/mujoco/assets_robot_xml/single_aloha_with_gripper/single_viperx_cupboard_touch.xml")
data = mujoco.MjData(model)
model.opt.timestep = 0.005  # 默认为0.001； 加快仿真步速（逻辑仿真更快）；注意：步长过大会影响物理精度，建议最多提升至 0.01

# === 获取传感器索引 ===
sensor_id_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_left")
nsensor_adr_left = model.sensor_adr[sensor_id_left]
nsensor_dim_left = model.sensor_dim[sensor_id_left]
sensor_id_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_right")
nsensor_adr_right = model.sensor_adr[sensor_id_right]
nsensor_dim_right = model.sensor_dim[sensor_id_right]


# === 获取双臂动作 ===
def get_action_arm(left_bot):
    action = np.zeros(7)
    action[:6] = left_bot.dxl.joint_states.position[:6]
    # action[6] = map_gripper(left_bot.dxl.joint_states.position[6])
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


# === 改进后的触觉显示函数 ===
def show_tactile_arrowed(tactile, size=(480, 480), max_shear=0.05, max_pressure=1, name='tactile'):
    """
    显示触觉传感器的剪切力方向 + 压力强度
    :param tactile: 输入为形状 (3, H, W)
    """
    channels, ny, nx = tactile.shape
    assert channels == 3, "Tactile data must have 3 channels (shear_x, shear_y, pressure)"
    loc_x = np.linspace(0, size[1], nx)
    loc_y = np.linspace(size[0], 0, ny)
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    for i in range(nx):
        for j in range(ny):
            # === 剪切力方向（放大20倍，归一化到 [-1, 1]） ===
            dir_x = np.clip(tactile[0, j, i] / max_shear, -1, 1) * 20
            dir_y = np.clip(tactile[1, j, i] / max_shear, -1, 1) * 20
            # === 压力颜色（红绿通道） ===
            pressure = np.clip(tactile[2, j, i] / max_pressure, 0, 1)
            color = (0, int(255 * (1 - pressure)), int(255 * pressure))  # BGR: G→R
            start = (int(loc_y[i]), int(loc_x[j]))
            end = (int(loc_y[i] + dir_y), int(loc_x[j] - dir_x))
            cv2.arrowedLine(img, start, end, color, 2, tipLength=0.5)
    # 旋转图像 180 度
    img_rotated = cv2.rotate(img, cv2.ROTATE_180)
    # 显示旋转后的图像
    cv2.imshow(name, img_rotated)
    return img


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
        viewer.sync()
        rate.sleep()

        viewport = mujoco.MjrRect(0, 0, 640, 480)
        mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, context)
        rgb = np.flipud(rgb)
        cv2.imshow("Top Camera View", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        # === 实时显示触觉感知 ===
        raw_left = data.sensordata[nsensor_adr_left:nsensor_adr_left + nsensor_dim_left]
        raw_right = data.sensordata[nsensor_adr_right:nsensor_adr_right + nsensor_dim_right]
        tactile_left = raw_left.reshape(touch_shape)
        tactile_right = raw_right.reshape(touch_shape)
        show_tactile_arrowed(tactile_left, size=window_size, max_shear=max_shear, max_pressure=max_pressure, name='Touch Left')
        show_tactile_arrowed(tactile_right, size=window_size, max_shear=max_shear, max_pressure=max_pressure, name='Touch Right')
        if cv2.waitKey(int(1000 / FPS)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    viewer.close()
