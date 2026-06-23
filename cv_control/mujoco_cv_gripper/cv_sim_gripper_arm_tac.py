import os
import cv2
import numpy as np
import time
import math
import mujoco
import mujoco.viewer
import glfw
from loop_rate_limiters import RateLimiter


# === 参数设置 ===
FPS = 50
touch_shape = (3, 16, 16)
max_shear = 0.05
max_pressure = 0.1
window_size = (480, 480)

# === 1. 初始化 MuJoCo 模型与数据 ===
XML_PATH = "/home/hjx/hjx_file/STF/Force_Gripper/mujoco/assets_robot_xml/single_aloha_with_gripper/single_viperx_cupboard_touch.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
model.opt.timestep = 0.005
gripper_actuator_id = model.actuator("fingers_actuator").id

# 获取传感器索引变量
sensor_id_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_left")
nsensor_adr_left = model.sensor_adr[sensor_id_left]
nsensor_dim_left = model.sensor_dim[sensor_id_left]
sensor_id_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_right")
nsensor_adr_right = model.sensor_adr[sensor_id_right]
nsensor_dim_right = model.sensor_dim[sensor_id_right]

# === 2. 离屏渲染预初始化 (用于 Top 视角相机) ===
if not glfw.init():
    raise Exception("Could not initialize GLFW")

glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
offscreen_window = glfw.create_window(640, 480, "offscreen", None, None)
glfw.make_context_current(offscreen_window)

cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
cam = mujoco.MjvCamera()
cam.fixedcamid = cam_id
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
scene = mujoco.MjvScene(model, maxgeom=1000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)


# === 触觉显示辅助函数 ===
def show_tactile_arrowed(tactile, size=(480, 480), max_shear=0.05, max_pressure=1, name='tactile'):
    channels, ny, nx = tactile.shape
    loc_x = np.linspace(0, size[1], nx)
    loc_y = np.linspace(size[0], 0, ny)
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    for i in range(nx):
        for j in range(ny):
            dir_x = np.clip(tactile[0, j, i] / max_shear, -1, 1) * 20
            dir_y = np.clip(tactile[1, j, i] / max_shear, -1, 1) * 20
            pressure = np.clip(tactile[2, j, i] / max_pressure, 0, 1)
            color = (0, int(255 * (1 - pressure)), int(255 * pressure))
            start = (int(loc_y[i]), int(loc_x[j]))
            end = (int(loc_y[i] + dir_y), int(loc_x[j] - dir_x))
            cv2.arrowedLine(img, start, end, color, 2, tipLength=0.5)
    img_rotated = cv2.rotate(img, cv2.ROTATE_180)
    cv2.imshow(name, img_rotated)


# === 3. 主程序入口 ===
if __name__ == "__main__":
    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    print("MuJoCo 窗口已启动...")

    print("正在初始化手势识别模块...")
    import HandTrackingModule as htm

    detector = htm.handDetector(detectionCon=0.8)

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    pTime = 0
    rate = RateLimiter(frequency=100.0)

    while viewer.is_running():
        success, img = cap.read()
        if not success:
            break

        # --- A. 视觉检测核心逻辑 ---
        img = detector.findHands(img, draw=True)  # 重新开启画骨架功能
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # 1. 提取关键点坐标
            x1, y1 = lmList[4][1], lmList[4][2]  # 大拇指
            x2, y2 = lmList[8][1], lmList[8][2]  # 食指
            xc, yc = (x1 + x2) // 2, (y1 + y2) // 2

            # 2. 计算距离
            length = math.hypot(x2 - x1, y2 - y1)

            # 3. 在画面中绘制连线和指尖圆点（恢复消失的代码）
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (xc, yc), 8, (255, 0, 255), cv2.FILLED)

            # 4. 显示实时距离数值
            cv2.putText(img, f'Dist: {int(length)}', (xc + 20, yc),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

            # 5. 映射到仿真控制
            current_gripper_val = np.interp(length, [30, 180], [255, 0])
            data.ctrl[gripper_actuator_id] = current_gripper_val

            # 6. 特殊交互：闭合时变色
            if length < 30:
                cv2.circle(img, (xc, yc), 12, (0, 255, 0), cv2.FILLED)

        # --- B. 仿真步进 ---
        mujoco.mj_step(model, data)
        viewer.sync()

        # --- C. 离屏渲染 (Top View) ---
        glfw.make_context_current(offscreen_window)
        viewport = mujoco.MjrRect(0, 0, 640, 480)
        mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, context)
        cv2.imshow("Top Camera View", cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR))

        # --- D. 触觉反馈 ---
        raw_left = data.sensordata[nsensor_adr_left:nsensor_adr_left + nsensor_dim_left]
        raw_right = data.sensordata[nsensor_adr_right:nsensor_adr_right + nsensor_dim_right]
        if len(raw_left) > 0:
            show_tactile_arrowed(raw_left.reshape(touch_shape), size=window_size, name='Touch Left')
        if len(raw_right) > 0:
            show_tactile_arrowed(raw_right.reshape(touch_shape), size=window_size, name='Touch Right')

        # --- E. FPS 显示 ---
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # 显示手势控制画面
        cv2.imshow("Hand Control View", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        rate.sleep()

    cap.release()
    cv2.destroyAllWindows()
    viewer.close()
    glfw.terminate()