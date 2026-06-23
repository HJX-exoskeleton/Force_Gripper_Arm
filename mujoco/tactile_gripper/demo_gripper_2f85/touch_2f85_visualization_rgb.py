import time
import mujoco  # pip install mujoco==3.3.0
import mujoco.viewer
import cv2
import numpy as np
from loop_rate_limiters import RateLimiter
import logging
logging.getLogger().setLevel(logging.ERROR)

# 加载模型
m = mujoco.MjModel.from_xml_path('/home/hjx/hjx_file/STF/Force_Gripper/mujoco/assets_robot_xml/gripper_2f85/scene.xml')
d = mujoco.MjData(m)

# 打印传感器数量
print(f"sensor_num: {m.nsensor}")

# 存储传感器的内存地址
touch_point_adr_right = [[0] * 16 for _ in range(16)]
touch_point_adr_left = [[0] * 16 for _ in range(16)]

for x in range(16):
    for y in range(16):
        idx = x * 16 + y
        touch_point_adr_right[x][y] = m.sensor_adr[idx]  # 第一个传感器
        touch_point_adr_left[x][y] = m.sensor_adr[idx + m.nsensor // 2]  # 第二个传感器在后半部分的地址

# 启动渲染器并进入仿真循环
with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    rate = RateLimiter(frequency=200.0)
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(m, d)

        # 创建两个16x16的矩阵来存储两个触觉传感器的数据
        touch_right = np.zeros((16, 16), dtype=np.float32)
        touch_left = np.zeros((16, 16), dtype=np.float32)

        force_max = 1

        for x in range(16):
            for y in range(16):
                adr_right = touch_point_adr_right[x][y]
                adr_left = touch_point_adr_left[x][y]

                # 获取第一个触觉传感器的数据
                data_right = mujoco.mju_norm3(d.sensordata[adr_right:adr_right + 3])
                touch_right[x, y] = mujoco.mju_clip(data_right, 0.0, force_max)

                # 获取第二个触觉传感器的数据
                data_left = mujoco.mju_norm3(d.sensordata[adr_left:adr_left + 3])
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

        cv2.waitKey(1)

        # 同步仿真和渲染状态
        viewer.sync()

        rate.sleep()

        # 简单的时间控制，保持与物理仿真步长同步
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)