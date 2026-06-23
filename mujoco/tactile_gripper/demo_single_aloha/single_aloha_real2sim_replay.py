import mujoco
import mujoco.viewer
import numpy as np
import h5py
import time
from loop_rate_limiters import RateLimiter

# === 加载模型 ===
xml_path = "/home/hjx/hjx_file/STF/Force_Gripper/mujoco/assets_robot_xml/single_aloha_tactile/single_viperx_cupboard.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# === 加载数据集 ===
hdf5_path = "/home/hjx/hjx_file/STF/Force_Gripper/data/data_tactile_single_aloha/episode_0.hdf5"
with h5py.File(hdf5_path, 'r') as f:
    qpos_data = f['observations/qpos'][:]
    qvel_data = f['observations/qvel'][:]
    actions = f['action'][:]

# === 启动可视化窗口 ===
viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)

# === 仿真主循环 ===
model.opt.timestep = 0.005
rate = RateLimiter(frequency=50.0)

for i in range(len(actions)):
    if not viewer.is_running():
        break

    # 设置状态（qpos/qvel）或直接设置控制（ctrl）
    data.qpos[:] = qpos_data[i]
    data.qvel[:] = qvel_data[i]
    data.ctrl[:7] = actions[i]

    # 前向动力学计算（确保渲染正确）
    mujoco.mj_forward(model, data)

    # 同步可视化并等待下一步
    viewer.sync()
    rate.sleep()

viewer.close()

