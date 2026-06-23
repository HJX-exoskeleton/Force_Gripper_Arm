import time
import math
import queue
import logging
import multiprocessing as mp

import cv2
import glfw
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter


logging.getLogger().setLevel(logging.ERROR)


# ============================================================
# 基本参数
# ============================================================

XML_PATH = "/home/hjx/hjx_file/STF/Force_Gripper_Arm/mujoco/assets_robot_xml/rebotarm_tactile/rebotarm_sim_cupboard.xml"

# 触觉形状
TOUCH_SHAPE = (3, 16, 16)
TACTILE_C, GRID_H, GRID_W = TOUCH_SHAPE

MAX_SHEAR = 0.05
MAX_PRESSURE = 0.1
WINDOW_SIZE = (480, 480)

# MuJoCo
MODEL_TIMESTEP = 0.005

# 主循环频率
SIM_HZ = 100.0

# 手势检测频率，摄像头一般 30Hz 左右即可
HAND_HZ = 30.0

# 触觉显示频率
TACTILE_VIS_HZ = 20.0

# MuJoCo viewer 同步频率
VIEWER_HZ = 20.0

# Top camera 渲染频率
TOP_CAMERA_HZ = 10.0

# 是否启用窗口
ENABLE_MUJOCO_VIEWER = True
ENABLE_TOP_CAMERA = True
ENABLE_HAND_CAMERA = True

# Top camera 参数
CAMERA_NAME = "top"
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

# 手势摄像头
HAND_CAMERA_ID = 0
HAND_CAMERA_WIDTH = 640
HAND_CAMERA_HEIGHT = 480

# 只保留最新帧，避免显示延迟堆积
QUEUE_MAXSIZE = 2

# force sensor 点阵恢复顺序
# 16x16 旧代码一般是 idx = x * 16 + y
SENSOR_ORDER = "row_major_xy"


# ============================================================
# 队列：只保留最新帧
# ============================================================

def put_latest(q: mp.Queue, item):
    """
    放入最新帧。
    队列满时丢弃旧帧，防止 OpenCV 显示延迟越来越大。
    """
    try:
        while q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                break
        q.put_nowait(item)
    except queue.Full:
        pass


# ============================================================
# Sensor 读取工具
# ============================================================

def print_sensor_info(model):
    print("\n========== Sensor Info ==========")
    print(f"model.nsensor     = {model.nsensor}")
    print(f"model.nsensordata = {model.nsensordata}")

    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        dim = int(model.sensor_dim[i])
        adr = int(model.sensor_adr[i])
        print(f"{i:04d} | name={name} | dim={dim} | adr={adr}")

    print("=================================\n")


def find_sensors_by_keyword(model, keyword):
    sensor_ids = []

    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)

        if name is None:
            continue

        if keyword in name:
            sensor_ids.append(i)

    return sensor_ids


def values_to_grid(values, grid_shape=(16, 16), order="row_major_xy"):
    h, w = grid_shape

    if order == "row_major_xy":
        return values.reshape(h, w)

    if order == "column_major_xy":
        return values.reshape(w, h).T

    raise ValueError(f"Unknown sensor order: {order}")


def build_tactile_reader(model, side):
    """
    构建触觉读取器。

    优先读取：
        touch_left / touch_right

    如果它们不存在，或者维度不是 3*16*16，
    自动尝试读取：
        touch_point_left / touch_point_right
    """
    assert side in ("left", "right")

    exact_name = f"touch_{side}"
    point_keyword = f"touch_point_{side}"
    expected_dim = TACTILE_C * GRID_H * GRID_W
    expected_points = GRID_H * GRID_W

    # 1. 优先找完整网格 sensor: touch_left / touch_right
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, exact_name)

    if sid >= 0:
        adr = int(model.sensor_adr[sid])
        dim = int(model.sensor_dim[sid])

        print(
            f"[INFO] Found sensor '{exact_name}': "
            f"id={sid}, dim={dim}, adr={adr}"
        )

        if dim == expected_dim:
            return {
                "mode": "grid_exact",
                "side": side,
                "sensor_id": sid,
                "adr": adr,
                "dim": dim,
            }

        if dim == 3:
            print(
                f"[WARNING] '{exact_name}' dim=3，不是完整 3x16x16 网格。"
                "将回退到 touch_point_* 搜索。"
            )

        else:
            print(
                f"[WARNING] '{exact_name}' dim={dim}，期望 {expected_dim}。"
                "将回退到 touch_point_* 搜索。"
            )

    # 2. 回退到 256 个 force sensor: touch_point_left/right
    sensor_ids = find_sensors_by_keyword(model, point_keyword)

    print(f"[INFO] keyword='{point_keyword}', found {len(sensor_ids)} sensors")

    if len(sensor_ids) == expected_points:
        adrs = np.array(
            [model.sensor_adr[sid] for sid in sensor_ids],
            dtype=np.int32
        )

        dims = np.array(
            [model.sensor_dim[sid] for sid in sensor_ids],
            dtype=np.int32
        )

        if not np.all(dims == 3):
            print(f"[WARNING] '{point_keyword}' 中并非所有 sensor 都是 dim=3")
            print("unique dims =", np.unique(dims))

        return {
            "mode": "points_force",
            "side": side,
            "sensor_ids": sensor_ids,
            "adrs": adrs,
            "dims": dims,
        }

    raise RuntimeError(
        f"无法构建 {side} 侧触觉读取器：\n"
        f"1. 未找到有效完整 sensor '{exact_name}'，或其维度不对；\n"
        f"2. '{point_keyword}' 匹配数量为 {len(sensor_ids)}，期望 {expected_points}。"
    )


def read_tactile(reader, data, sensor_order="row_major_xy"):
    """
    统一返回 shape=(3,16,16) 的触觉数据。

    channel 0: shear_x / Fx
    channel 1: shear_y / Fy
    channel 2: pressure / Fz
    """
    mode = reader["mode"]

    if mode == "grid_exact":
        adr = reader["adr"]
        dim = reader["dim"]
        raw = data.sensordata[adr:adr + dim]
        return raw.reshape(TOUCH_SHAPE).astype(np.float32)

    if mode == "points_force":
        adrs = reader["adrs"]
        dims = reader["dims"]

        if np.all(dims == 3):
            index = adrs[:, None] + np.array([0, 1, 2], dtype=np.int32)
            raw = data.sensordata[index]  # shape = (256, 3)

            fx = values_to_grid(raw[:, 0], (GRID_H, GRID_W), sensor_order)
            fy = values_to_grid(raw[:, 1], (GRID_H, GRID_W), sensor_order)
            fz = values_to_grid(raw[:, 2], (GRID_H, GRID_W), sensor_order)

            return np.stack([fx, fy, fz], axis=0).astype(np.float32)

        tactile = np.zeros(TOUCH_SHAPE, dtype=np.float32)

        for k, adr in enumerate(adrs):
            dim = int(dims[k])
            raw = data.sensordata[adr:adr + dim]

            if dim >= 3:
                fx, fy, fz = raw[0], raw[1], raw[2]
            elif dim == 1:
                fx, fy, fz = 0.0, 0.0, raw[0]
            else:
                fx, fy, fz = 0.0, 0.0, 0.0

            if sensor_order == "row_major_xy":
                x = k // GRID_W
                y = k % GRID_W
            else:
                x = k % GRID_H
                y = k // GRID_H

            tactile[0, x, y] = fx
            tactile[1, x, y] = fy
            tactile[2, x, y] = fz

        return tactile

    raise RuntimeError(f"Unknown reader mode: {mode}")


# ============================================================
# 触觉方向可视化
# ============================================================

def tactile_arrow_image_follow_reference(
    tactile,
    size=(480, 480),
    max_shear=0.05,
    max_pressure=0.1
):
    """
    严格遵循你给出的 show_tactile_arrowed() 可视化逻辑。

    注意：
    这里没有在函数内部 cv2.imshow，而是返回 img_rotated。
    这样可以放到独立可视化进程中显示。
    但箭头方向、坐标映射、颜色映射、旋转方式保持一致。
    """
    channels, ny, nx = tactile.shape

    if channels != 3:
        raise ValueError(
            f"Tactile data must have 3 channels, got shape={tactile.shape}"
        )

    loc_x = np.linspace(0, size[1], nx)
    loc_y = np.linspace(size[0], 0, ny)

    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    for i in range(nx):
        for j in range(ny):
            # === 剪切力方向，严格保留你的方向定义 ===
            dir_x = np.clip(tactile[0, j, i] / max_shear, -1, 1) * 20
            dir_y = np.clip(tactile[1, j, i] / max_shear, -1, 1) * 20

            # === 压力颜色，严格保留你的 BGR: G -> R ===
            pressure = np.clip(tactile[2, j, i] / max_pressure, 0, 1)
            color = (
                0,
                int(255 * (1 - pressure)),
                int(255 * pressure)
            )

            # === 起点和终点，严格保留你的坐标映射 ===
            start = (int(loc_y[i]), int(loc_x[j]))
            end = (int(loc_y[i] + dir_y), int(loc_x[j] - dir_x))

            cv2.arrowedLine(
                img,
                start,
                end,
                color,
                2,
                tipLength=0.5
            )

    # === 严格保留 180° 旋转 ===
    img_rotated = cv2.rotate(img, cv2.ROTATE_180)

    return img_rotated


# ============================================================
# Top camera 离屏渲染
# ============================================================

def init_offscreen_renderer(model, camera_name="top", width=320, height=240):
    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

    window = glfw.create_window(
        width,
        height,
        "offscreen",
        None,
        None
    )

    if window is None:
        glfw.terminate()
        raise RuntimeError("Could not create offscreen GLFW window")

    glfw.make_context_current(window)

    cam_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_name
    )

    if cam_id < 0:
        glfw.destroy_window(window)
        glfw.terminate()
        raise RuntimeError(f"Can not find camera '{camera_name}'")

    cam = mujoco.MjvCamera()
    cam.fixedcamid = cam_id
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    scene = mujoco.MjvScene(model, maxgeom=10000)

    context = mujoco.MjrContext(
        model,
        mujoco.mjtFontScale.mjFONTSCALE_150
    )

    mujoco.mjr_setBuffer(
        mujoco.mjtFramebuffer.mjFB_OFFSCREEN,
        context
    )

    viewport = mujoco.MjrRect(0, 0, width, height)

    return window, cam, scene, context, viewport


def render_top_camera(
    model,
    data,
    window,
    cam,
    scene,
    context,
    viewport,
    width=320,
    height=240
):
    glfw.make_context_current(window)

    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        cam,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene
    )

    mujoco.mjr_render(viewport, scene, context)

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb, None, viewport, context)

    rgb = np.flipud(rgb)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return bgr


# ============================================================
# 可视化进程
# ============================================================

def visualization_process(frame_queue: mp.Queue, stop_event: mp.Event):
    """
    独立 OpenCV 可视化进程。
    负责：
        1. Touch Left 箭头图
        2. Touch Right 箭头图
        3. Top Camera View
        4. Hand Control View
    """
    print("[VIS] visualization process started")

    cv2.namedWindow("Touch Left", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Touch Right", cv2.WINDOW_NORMAL)

    if ENABLE_TOP_CAMERA:
        cv2.namedWindow("Top Camera View", cv2.WINDOW_NORMAL)

    if ENABLE_HAND_CAMERA:
        cv2.namedWindow("Hand Control View", cv2.WINDOW_NORMAL)

    last_top_view = None
    last_hand_view = None

    while not stop_event.is_set():
        try:
            packet = frame_queue.get(timeout=0.05)
        except queue.Empty:
            key = cv2.waitKey(1)
            if key == 27 or key == ord("q"):
                stop_event.set()
            continue

        tactile_left = packet.get("tactile_left", None)
        tactile_right = packet.get("tactile_right", None)
        top_view = packet.get("top_view", None)
        hand_view = packet.get("hand_view", None)

        if tactile_left is not None:
            left_img = tactile_arrow_image_follow_reference(
                tactile_left,
                size=WINDOW_SIZE,
                max_shear=MAX_SHEAR,
                max_pressure=MAX_PRESSURE
            )
            cv2.imshow("Touch Left", left_img)

        if tactile_right is not None:
            right_img = tactile_arrow_image_follow_reference(
                tactile_right,
                size=WINDOW_SIZE,
                max_shear=MAX_SHEAR,
                max_pressure=MAX_PRESSURE
            )
            cv2.imshow("Touch Right", right_img)

        if top_view is not None:
            last_top_view = top_view

        if hand_view is not None:
            last_hand_view = hand_view

        if ENABLE_TOP_CAMERA and last_top_view is not None:
            cv2.imshow("Top Camera View", last_top_view)

        if ENABLE_HAND_CAMERA and last_hand_view is not None:
            cv2.imshow("Hand Control View", last_hand_view)

        key = cv2.waitKey(1)

        if key == 27 or key == ord("q"):
            stop_event.set()
            break

    cv2.destroyAllWindows()
    print("[VIS] visualization process stopped")


# ============================================================
# 仿真 + 手势控制进程
# ============================================================

def simulation_process(frame_queue: mp.Queue, stop_event: mp.Event):
    """
    MuJoCo 仿真进程。
    负责：
        1. 手势识别
        2. gripper 控制
        3. MuJoCo mj_step
        4. 触觉读取
        5. Top camera 低频渲染
        6. 发送数据给可视化进程
    """
    print("[SIM] loading model...")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = MODEL_TIMESTEP

    print(f"[SIM] model timestep = {model.opt.timestep}")
    print_sensor_info(model)

    # 触觉读取器
    left_reader = build_tactile_reader(model, side="left")
    right_reader = build_tactile_reader(model, side="right")

    # gripper actuator
    gripper_actuator_id = model.actuator("gripper").id
    print(f"[SIM] gripper actuator id = {gripper_actuator_id}")

    # 更稳妥：自动读取 actuator ctrlrange
    if model.actuator_ctrllimited[gripper_actuator_id]:
        gripper_ctrl_min = float(model.actuator_ctrlrange[gripper_actuator_id, 0])
        gripper_ctrl_max = float(model.actuator_ctrlrange[gripper_actuator_id, 1])
    else:
        gripper_ctrl_min = 0.0
        gripper_ctrl_max = 1.0

    print(
        f"[SIM] gripper ctrlrange = "
        f"[{gripper_ctrl_min}, {gripper_ctrl_max}]"
    )

    # MuJoCo viewer
    viewer = None
    if ENABLE_MUJOCO_VIEWER:
        viewer = mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=False,
            show_right_ui=False
        )
        print("[SIM] MuJoCo viewer started")

    # Top camera
    offscreen = None
    if ENABLE_TOP_CAMERA:
        try:
            offscreen = init_offscreen_renderer(
                model,
                camera_name=CAMERA_NAME,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT
            )
            print("[SIM] Top camera offscreen renderer initialized")
        except Exception as e:
            print("[SIM] WARNING: Top camera 初始化失败，自动关闭")
            print(e)
            offscreen = None

    # 手势识别
    detector = None
    cap = None
    if ENABLE_HAND_CAMERA:
        print("[SIM] initializing hand tracking module...")
        try:
            import HandTrackingModule as htm

            detector = htm.handDetector(detectionCon=0.8)

            cap = cv2.VideoCapture(HAND_CAMERA_ID)
            cap.set(3, HAND_CAMERA_WIDTH)
            cap.set(4, HAND_CAMERA_HEIGHT)

            if not cap.isOpened():
                print("[SIM] WARNING: 摄像头打开失败，关闭手势控制")
                cap.release()
                cap = None
                detector = None
            else:
                print("[SIM] hand camera started")

        except Exception as e:
            print("[SIM] WARNING: 手势识别模块初始化失败，关闭手势控制")
            print(e)
            detector = None
            cap = None

    sim_rate = RateLimiter(frequency=SIM_HZ)

    hand_interval = max(1, int(round(SIM_HZ / HAND_HZ)))
    tactile_interval = max(1, int(round(SIM_HZ / TACTILE_VIS_HZ)))
    viewer_interval = max(1, int(round(SIM_HZ / VIEWER_HZ)))
    top_camera_interval = max(1, int(round(SIM_HZ / TOP_CAMERA_HZ)))

    print("[SIM] simulation loop started")
    print(f"[SIM] SIM_HZ              = {SIM_HZ}")
    print(f"[SIM] hand every          = {hand_interval} steps")
    print(f"[SIM] tactile every       = {tactile_interval} steps")
    print(f"[SIM] viewer sync every   = {viewer_interval} steps")
    print(f"[SIM] top camera every    = {top_camera_interval} steps")

    step_count = 0
    p_time = time.time()
    last_hand_view = None

    try:
        while not stop_event.is_set():

            if viewer is not None and not viewer.is_running():
                stop_event.set()
                break

            step_count += 1

            # ====================================================
            # A. 手势识别与 gripper 控制，低频执行
            # ====================================================
            if (
                ENABLE_HAND_CAMERA
                and cap is not None
                and detector is not None
                and step_count % hand_interval == 0
            ):
                success, img = cap.read()

                if success:
                    img = detector.findHands(img, draw=True)
                    lm_list = detector.findPosition(img, draw=False)

                    if len(lm_list) != 0:
                        x1, y1 = lm_list[4][1], lm_list[4][2]
                        x2, y2 = lm_list[8][1], lm_list[8][2]
                        xc, yc = (x1 + x2) // 2, (y1 + y2) // 2

                        length = math.hypot(x2 - x1, y2 - y1)

                        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                        cv2.circle(img, (xc, yc), 8, (255, 0, 255), cv2.FILLED)

                        cv2.putText(
                            img,
                            f"Dist: {int(length)}",
                            (xc + 20, yc),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                        # 原代码是 [30,180] -> [255,0]
                        # 这里优化为映射到 MuJoCo actuator 的真实 ctrlrange。
                        # 通常小距离 = 闭合，大距离 = 张开。
                        current_gripper_val = np.interp(
                            length,
                            [30, 180],
                            [gripper_ctrl_min, gripper_ctrl_max]
                        )

                        data.ctrl[gripper_actuator_id] = current_gripper_val

                        if length < 30:
                            cv2.circle(img, (xc, yc), 12, (0, 255, 0), cv2.FILLED)

                    # FPS 显示
                    c_time = time.time()
                    fps = 1.0 / (c_time - p_time) if (c_time - p_time) > 0 else 0.0
                    p_time = c_time

                    cv2.putText(
                        img,
                        f"FPS: {int(fps)}",
                        (10, 40),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (255, 0, 0),
                        2
                    )

                    last_hand_view = img

            # ====================================================
            # B. MuJoCo 仿真步进
            # ====================================================
            mujoco.mj_step(model, data)

            # ====================================================
            # C. 低频读取触觉、渲染 Top View、发送给可视化进程
            # ====================================================
            if step_count % tactile_interval == 0:
                tactile_left = read_tactile(
                    left_reader,
                    data,
                    sensor_order=SENSOR_ORDER
                )

                tactile_right = read_tactile(
                    right_reader,
                    data,
                    sensor_order=SENSOR_ORDER
                )

                packet = {
                    "tactile_left": tactile_left,
                    "tactile_right": tactile_right,
                    "top_view": None,
                    "hand_view": last_hand_view,
                    "step": step_count,
                    "time": float(data.time),
                }

                if (
                    ENABLE_TOP_CAMERA
                    and offscreen is not None
                    and step_count % top_camera_interval == 0
                ):
                    window, cam, scene, context, viewport = offscreen

                    top_view = render_top_camera(
                        model,
                        data,
                        window,
                        cam,
                        scene,
                        context,
                        viewport,
                        width=CAMERA_WIDTH,
                        height=CAMERA_HEIGHT
                    )

                    packet["top_view"] = top_view

                put_latest(frame_queue, packet)

            # ====================================================
            # D. MuJoCo viewer 低频同步
            # ====================================================
            if viewer is not None and step_count % viewer_interval == 0:
                viewer.sync()

            sim_rate.sleep()

    except KeyboardInterrupt:
        stop_event.set()

    finally:
        if cap is not None:
            cap.release()

        if viewer is not None:
            viewer.close()

        if offscreen is not None:
            window, cam, scene, context, viewport = offscreen
            glfw.destroy_window(window)
            glfw.terminate()

        print("[SIM] simulation process stopped")


# ============================================================
# 主函数
# ============================================================

def main():
    """
    主进程启动：
        1. 仿真 + 手势控制进程
        2. OpenCV 可视化进程
    """
    mp.set_start_method("spawn", force=True)

    frame_queue = mp.Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = mp.Event()

    sim_proc = mp.Process(
        target=simulation_process,
        args=(frame_queue, stop_event),
        name="MuJoCoSimulationProcess"
    )

    vis_proc = mp.Process(
        target=visualization_process,
        args=(frame_queue, stop_event),
        name="OpenCVVisualizationProcess"
    )

    sim_proc.start()
    vis_proc.start()

    try:
        while sim_proc.is_alive() and vis_proc.is_alive():
            time.sleep(0.2)

    except KeyboardInterrupt:
        stop_event.set()

    finally:
        stop_event.set()

        sim_proc.join(timeout=3.0)
        vis_proc.join(timeout=3.0)

        if sim_proc.is_alive():
            sim_proc.terminate()

        if vis_proc.is_alive():
            vis_proc.terminate()

        print("[MAIN] all processes stopped")


if __name__ == "__main__":
    main()
