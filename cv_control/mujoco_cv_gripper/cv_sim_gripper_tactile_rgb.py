import os
import cv2
import numpy as np
import time
import math
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import logging

# 设置日志级别，避免输出过多的底层日志信息
logging.getLogger().setLevel(logging.ERROR)

# === 1. 初始化 MuJoCo 模型与数据 ===
# 注意：这里使用了带有 8x16 传感器的场景文件
XML_PATH = "/home/hjx/hjx_file/STF/Force_Gripper/mujoco/assets_robot_xml/gripper_2f85/scene_8_16.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
model.opt.timestep = 0.005
gripper_actuator_id = model.actuator("fingers_actuator").id

print(f"模型加载成功！当前模型中的传感器总数量: {model.nsensor}")

# === 2. 预先计算触觉传感器的内存地址映射 ===
# 创建两个 8x16 列表，存储右侧和左侧触觉传感器的内存地址
touch_point_adr_right = [[0] * 16 for _ in range(8)]
touch_point_adr_left = [[0] * 16 for _ in range(8)]

# 填充左右传感器的内存地址 (假设前一半是右侧，后一半是左侧)
for x in range(8):
    for y in range(16):
        idx = x + y * 8
        touch_point_adr_right[x][y] = model.sensor_adr[idx]
        touch_point_adr_left[x][y] = model.sensor_adr[idx + model.nsensor // 2]

# === 3. 主程序入口 ===
if __name__ == "__main__":
    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    print("MuJoCo 3D 渲染窗口已启动...")

    print("正在初始化手势识别模块...")
    import HandTrackingModule as htm

    detector = htm.handDetector(detectionCon=0.8)

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    pTime = 0
    # 设置更新频率，保持与传感器读取和画面渲染的流畅度匹配
    rate = RateLimiter(frequency=100.0)

    while viewer.is_running():
        success, img = cap.read()
        if not success:
            break

        # ==========================================
        # --- A. 视觉检测与手势控制核心逻辑 ---
        # ==========================================
        img = detector.findHands(img, draw=True)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # 1. 提取关键点坐标 (大拇指 4, 食指 8)
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            xc, yc = (x1 + x2) // 2, (y1 + y2) // 2

            # 2. 计算手指间距
            length = math.hypot(x2 - x1, y2 - y1)

            # 3. 在画面中绘制连线和指尖圆点
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (xc, yc), 8, (255, 0, 255), cv2.FILLED)

            cv2.putText(img, f'Dist: {int(length)}', (xc + 20, yc),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

            # 4. 映射到仿真控制 (根据手指距离控制夹爪)
            current_gripper_val = np.interp(length, [30, 180], [255, 0])
            data.ctrl[gripper_actuator_id] = current_gripper_val

            # 5. 闭合提示
            if length < 30:
                cv2.circle(img, (xc, yc), 12, (0, 255, 0), cv2.FILLED)

        # ==========================================
        # --- B. 仿真物理步进 ---
        # ==========================================
        mujoco.mj_step(model, data)

        # ==========================================
        # --- C. 触觉数据读取与热力图可视化 ---
        # ==========================================
        touch_right = np.zeros((8, 16), dtype=np.float32)
        touch_left = np.zeros((8, 16), dtype=np.float32)
        force_max = 1.0  # 数据标准化基准

        for x in range(8):
            for y in range(16):
                adr_right = touch_point_adr_right[x][y]
                adr_left = touch_point_adr_left[x][y]

                # 获取力向量的模并裁剪
                data_right = mujoco.mju_norm3(data.sensordata[adr_right:adr_right + 3])
                touch_right[x, y] = mujoco.mju_clip(data_right, 0.0, force_max)

                data_left = mujoco.mju_norm3(data.sensordata[adr_left:adr_left + 3])
                touch_left[x, y] = mujoco.mju_clip(data_left, 0.0, force_max)

        # 标准化映射到 0-255 并生成热力图
        touch_right_normalized = np.clip(touch_right, 0, force_max) / force_max * 255
        touch_left_normalized = np.clip(touch_left, 0, force_max) / force_max * 255

        touch_colored_right = cv2.applyColorMap(touch_right_normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        touch_colored_left = cv2.applyColorMap(touch_left_normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)

        # 调整大小与旋转
        touch_colored_right_resized = cv2.resize(touch_colored_right, (480, 240))
        touch_colored_left_resized = cv2.resize(touch_colored_left, (480, 240))

        touch_colored_right_rotated = np.rot90(touch_colored_right_resized, k=-1)
        touch_colored_left_rotated = np.rot90(touch_colored_left_resized, k=-1)

        # 显示热力图
        cv2.imshow("Touch Heatmap - Sensor Right", touch_colored_right_rotated)
        cv2.imshow("Touch Heatmap - Sensor Left", touch_colored_left_rotated)

        # ==========================================
        # --- D. 画面同步与显示 ---
        # ==========================================
        viewer.sync()  # 同步状态到 MuJoCo 渲染器

        # FPS 计算与显示
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # 显示手势控制监控画面
        cv2.imshow("Hand Control View", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        rate.sleep()

    cap.release()
    cv2.destroyAllWindows()
    viewer.close()
