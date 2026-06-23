import time
import mujoco  # pip install mujoco==3.3.0
import mujoco.viewer
import cv2
import numpy as np
from loop_rate_limiters import RateLimiter  # 引入RateLimiter用于限制更新频率
import logging
logging.getLogger().setLevel(logging.ERROR)  # 设置日志级别，避免输出过多的日志信息

m = mujoco.MjModel.from_xml_path('/home/hjx/hjx_file/STF/Force_Gripper/mujoco/assets_robot_xml/gripper_2f85/scene_8_16.xml')
d = mujoco.MjData(m)

# 打印传感器数量
print(f"sensor_num: {m.nsensor}")  # 输出模型中的传感器总数量，用于调试和确认

# 创建两个二维列表，用于存储右侧和左侧触觉传感器的内存地址
# 每个触觉传感器将存储一个内存地址，共有8行16列
touch_point_adr_right = [[0] * 16 for _ in range(8)]  # 右侧触觉传感器
touch_point_adr_left = [[0] * 16 for _ in range(8)]  # 左侧触觉传感器

# 遍历每个触觉传感器的位置，填充左右传感器的内存地址
for x in range(8):  # 遍历8行
    for y in range(16):  # 遍历16列
        idx = x + y * 8  # 计算当前传感器的索引
        touch_point_adr_right[x][y] = m.sensor_adr[idx]  # 获取右侧传感器的内存地址
        # 左侧传感器的内存地址在 `sensor_adr` 数组的后半部分
        touch_point_adr_left[x][y] = m.sensor_adr[idx + m.nsensor // 2]  # 获取左侧传感器的内存地址

# 启动MuJoCo的被动渲染器，进入仿真循环
with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()  # 记录开始时间
    rate = RateLimiter(frequency=200.0)  # 创建一个限制更新频率的RateLimiter，确保更新频率为200次/秒
    while viewer.is_running():  # 循环，直到仿真结束
        step_start = time.time()  # 记录每个仿真步的开始时间
        mujoco.mj_step(m, d)  # 进行一步仿真，更新物理环境状态

        # 创建两个8x16矩阵，用于存储右侧和左侧触觉传感器的数据
        touch_right = np.zeros((8, 16), dtype=np.float32)  # 右侧触觉传感器的数据
        touch_left = np.zeros((8, 16), dtype=np.float32)  # 左侧触觉传感器的数据

        force_max = 1  # 定义力的最大值，用于数据标准化

        # 遍历每个触觉传感器，获取其测量数据
        for x in range(8):  # 遍历8行
            for y in range(16):  # 遍历16列
                adr_right = touch_point_adr_right[x][y]  # 获取右侧传感器的内存地址
                adr_left = touch_point_adr_left[x][y]  # 获取左侧传感器的内存地址

                # 获取第一个（右侧）触觉传感器的数据，`mju_norm3`用于计算传感器数据的范数（力向量的模）
                data_right = mujoco.mju_norm3(d.sensordata[adr_right:adr_right + 3])
                # 将数据限制在0到最大力值`force_max`之间
                touch_right[x, y] = mujoco.mju_clip(data_right, 0.0, force_max)

                # 获取第二个（左侧）触觉传感器的数据
                data_left = mujoco.mju_norm3(d.sensordata[adr_left:adr_left + 3])
                touch_left[x, y] = mujoco.mju_clip(data_left, 0.0, force_max)

        # 将触觉传感器数据标准化到0到255的范围，用于图像显示
        touch_right_normalized = np.clip(touch_right, 0, force_max) / force_max * 255
        touch_left_normalized = np.clip(touch_left, 0, force_max) / force_max * 255

        # 使用OpenCV的COLORMAP_VIRIDIS来创建热力图效果，`applyColorMap`将数据映射到颜色上
        touch_colored_right = cv2.applyColorMap(touch_right_normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        touch_colored_left = cv2.applyColorMap(touch_left_normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)

        # 调整热力图的显示大小
        touch_colored_right_resized = cv2.resize(touch_colored_right, (480, 240))  # 将右侧热力图调整为240x480大小
        touch_colored_left_resized = cv2.resize(touch_colored_left, (480, 240))  # 将左侧热力图调整为240x480大小

        # 旋转热力图 90 度，使用 np.rot90 或 cv2.rotate
        touch_colored_right_rotated = np.rot90(touch_colored_right_resized, k=-1)  # 顺时针旋转 90 度
        touch_colored_left_rotated = np.rot90(touch_colored_left_resized, k=-1)  # 顺时针旋转 90 度

        # 显示旋转后的热力图
        cv2.imshow("Touch Heatmap - Sensor Right", touch_colored_right_rotated)  # 显示右侧触觉传感器的热力图
        cv2.imshow("Touch Heatmap - Sensor Left", touch_colored_left_rotated)  # 显示左侧触觉传感器的热力图

        cv2.waitKey(1)  # 等待1毫秒，更新显示窗口

        # 同步仿真和渲染状态，确保仿真和图像更新的同步
        viewer.sync()

        # 使用RateLimiter控制更新频率，确保仿真步频率与RateLimiter同步
        rate.sleep()

        # 简单的时间控制，保持与物理仿真步长同步
        time_until_next_step = m.opt.timestep - (time.time() - step_start)  # 计算下一步的时间间隔
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)  # 如果时间间隔大于0，则休眠直到下一步仿真开始
