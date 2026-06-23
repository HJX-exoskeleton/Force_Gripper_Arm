import os
import time
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
logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)


# ============================================================
# 基本配置
# ============================================================

XML_PATH = "/home/hjx/hjx_file/STF/Force_Gripper_Arm/mujoco/assets_robot_xml/rebotarm_tactile/rebotarm_sim_cupboard.xml"

GRID_H = 16
GRID_W = 16

FORCE_MAX = 0.05  # 1

# 仿真频率：先不要直接 200 Hz，先用 100 Hz 测试稳定性
SIM_HZ = 100.0

# OpenCV 热力图显示频率
VIS_HZ = 50.0

# MuJoCo viewer 同步频率
VIEWER_HZ = 30.0

# top camera 渲染频率
CAMERA_HZ = 5.0

# 是否启用 MuJoCo viewer
ENABLE_MUJOCO_VIEWER = True

# 是否启用 top camera 离屏渲染。
# 离屏相机渲染通常是本脚本最重的步骤。触觉低延迟优先时建议保持 False。
ENABLE_TOP_CAMERA = False

# top camera 分辨率
# 640x480 会更清晰，但 readPixels 更慢
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_NAME = "top"

# OpenCV 热力图窗口显示大小
HEATMAP_DISPLAY_SIZE = 480

# 队列只保留最新帧，避免可视化延迟堆积
QUEUE_MAXSIZE = 1

# 每隔多少仿真步打印一次耗时统计。0 表示关闭。
PERF_LOG_INTERVAL = 0


# ============================================================
# 工具函数：队列只保留最新帧
# ============================================================

def put_latest(q: mp.Queue, item):
    """
    向 multiprocessing.Queue 放入最新数据。
    如果队列满了，丢掉旧帧，只保留最新帧。
    这样可以避免可视化窗口延迟越来越大。
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


def get_latest(q: mp.Queue, timeout=0.05):
    """
    读取最新帧。
    如果队列里已经积压了旧帧，直接丢弃旧帧，只返回最后一帧。
    """
    packet = q.get(timeout=timeout)

    while True:
        try:
            packet = q.get_nowait()
        except queue.Empty:
            return packet


# ============================================================
# sensor 扫描与触觉读取
# ============================================================

def print_sensor_info(model):
    """
    打印所有 sensor 的名称、维度和 sensordata 地址。
    用于确认 XML 展开后的实际 sensor 顺序。
    """
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
    """
    根据 sensor 名称关键词查找 sensor id。
    例如：
        keyword = "touch_point_right"
        keyword = "touch_point_left"
    """
    sensor_ids = []

    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        if name is None:
            continue

        if keyword in name:
            sensor_ids.append(i)

    return sensor_ids


def build_tactile_reader(model, keyword, grid_shape=(16, 16)):
    """
    构建触觉读取器。

    支持两种情况：

    情况 A：找到 256 个 sensor
        说明每个触觉点都有一个 force sensor。
        可以构建真正的 16x16 热力图。

    情况 B：只找到 1 个 sensor
        说明 XML 当前只提供整个触觉垫的总 force。
        这时无法恢复真实 16x16 分布，只能显示为均匀热力图。
    """
    h, w = grid_shape
    expected_num = h * w

    sensor_ids = find_sensors_by_keyword(model, keyword)

    print(f"[INFO] keyword='{keyword}', found {len(sensor_ids)} sensors")

    if len(sensor_ids) == expected_num:
        dims = np.array([model.sensor_dim[sid] for sid in sensor_ids], dtype=np.int32)
        adrs = np.array([model.sensor_adr[sid] for sid in sensor_ids], dtype=np.int32)

        if not np.all(dims == 3):
            print(f"[WARNING] '{keyword}' 中不是所有 sensor 都是 dim=3。")
            print("dims unique =", np.unique(dims))

        return {
            "mode": "grid_force",
            "keyword": keyword,
            "sensor_ids": sensor_ids,
            "adrs": adrs,
            "dims": dims,
            "grid_shape": grid_shape,
        }

    if len(sensor_ids) == 1:
        sid = sensor_ids[0]
        dim = int(model.sensor_dim[sid])
        adr = int(model.sensor_adr[sid])

        print(
            f"[WARNING] '{keyword}' 只找到 1 个 sensor: "
            f"id={sid}, dim={dim}, adr={adr}"
        )
        print(
            "[WARNING] 当前只能显示整块触觉垫的总力，"
            "不能得到真实 16x16 空间分布。"
        )

        return {
            "mode": "single_force",
            "keyword": keyword,
            "sensor_ids": sensor_ids,
            "adr": adr,
            "dim": dim,
            "grid_shape": grid_shape,
        }

    raise RuntimeError(
        f"触觉 sensor 数量异常：keyword='{keyword}', found={len(sensor_ids)}。"
        f"期望 {expected_num} 个，或者至少 1 个。"
    )


def read_tactile(reader, data, force_max=1.0):
    """
    根据 reader 配置读取触觉矩阵。
    返回 shape = (16, 16) 的 float32 矩阵。
    """
    mode = reader["mode"]
    h, w = reader["grid_shape"]

    if mode == "grid_force":
        adrs = reader["adrs"]
        dims = reader["dims"]

        # 最常见情况：每个触觉点是 3D force sensor
        if np.all(dims == 3):
            index = adrs[:, None] + np.array([0, 1, 2], dtype=np.int32)
            raw = data.sensordata[index]
            values = np.linalg.norm(raw, axis=1)
            values = np.clip(values, 0.0, force_max)
            return values.reshape(h, w).astype(np.float32)

        # 兼容其他 dim
        values = np.zeros(h * w, dtype=np.float32)
        for k, adr in enumerate(adrs):
            dim = int(dims[k])
            raw = data.sensordata[adr:adr + dim]
            if dim == 1:
                value = raw[0]
            else:
                value = np.linalg.norm(raw)
            values[k] = np.clip(value, 0.0, force_max)

        return values.reshape(h, w).astype(np.float32)

    if mode == "single_force":
        adr = reader["adr"]
        dim = reader["dim"]
        raw = data.sensordata[adr:adr + dim]

        if dim == 1:
            value = raw[0]
        else:
            value = np.linalg.norm(raw)

        value = float(np.clip(value, 0.0, force_max))

        # 注意：这不是空间分布，只是总力的均匀显示
        return np.full((h, w), value, dtype=np.float32)

    raise RuntimeError(f"未知 tactile reader mode: {mode}")


# ============================================================
# 可视化函数
# ============================================================

def tactile_to_colormap(tactile, force_max=1.0, display_size=480):
    """
    将 16x16 触觉矩阵转换为 OpenCV 热力图。
    """
    tactile_normalized = np.clip(tactile, 0.0, force_max) / force_max * 255.0
    tactile_uint8 = tactile_normalized.astype(np.uint8)

    colored = cv2.applyColorMap(tactile_uint8, cv2.COLORMAP_VIRIDIS)

    colored_resized = cv2.resize(
        colored,
        (display_size, display_size),
        interpolation=cv2.INTER_NEAREST
    )

    return colored_resized


# ============================================================
# top camera 离屏渲染
# ============================================================

def init_offscreen_renderer(model, camera_name="top", width=320, height=240):
    """
    初始化 MuJoCo 离屏渲染。
    注意：这个函数必须在仿真进程内部调用。
    """
    if not glfw.init():
        raise RuntimeError("GLFW 初始化失败")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(width, height, "offscreen", None, None)

    if window is None:
        glfw.terminate()
        raise RuntimeError("GLFW offscreen window 创建失败")

    glfw.make_context_current(window)

    cam_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_name
    )

    if cam_id < 0:
        glfw.destroy_window(window)
        glfw.terminate()
        raise RuntimeError(f"找不到名为 '{camera_name}' 的 camera。请检查 XML。")

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


def render_camera_view(model, data, window, cam, scene, context, viewport, width=320, height=240):
    """
    渲染 top camera，并返回 OpenCV BGR 图像。
    """
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
    独立可视化进程。
    只负责 OpenCV 显示，不参与 MuJoCo 仿真。
    """
    last_top_view = None

    cv2.namedWindow("Touch Heatmap - Sensor Right", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Touch Heatmap - Sensor Left", cv2.WINDOW_NORMAL)

    if ENABLE_TOP_CAMERA:
        cv2.namedWindow("Top Camera View", cv2.WINDOW_NORMAL)

    print("[VIS] visualization process started")

    while not stop_event.is_set():
        try:
            packet = get_latest(frame_queue, timeout=0.02)
        except queue.Empty:
            key = cv2.waitKey(1)
            if key == 27:
                stop_event.set()
            continue

        touch_right = packet.get("touch_right", None)
        touch_left = packet.get("touch_left", None)
        top_view = packet.get("top_view", None)

        if touch_right is not None:
            right_img = tactile_to_colormap(
                touch_right,
                force_max=FORCE_MAX,
                display_size=HEATMAP_DISPLAY_SIZE
            )
            cv2.imshow("Touch Heatmap - Sensor Right", right_img)

        if touch_left is not None:
            left_img = tactile_to_colormap(
                touch_left,
                force_max=FORCE_MAX,
                display_size=HEATMAP_DISPLAY_SIZE
            )
            cv2.imshow("Touch Heatmap - Sensor Left", left_img)

        if top_view is not None:
            last_top_view = top_view

        if ENABLE_TOP_CAMERA and last_top_view is not None:
            cv2.imshow("Top Camera View", last_top_view)

        key = cv2.waitKey(1)

        # ESC 退出
        if key == 27:
            stop_event.set()
            break

    cv2.destroyAllWindows()
    print("[VIS] visualization process stopped")


# ============================================================
# 仿真进程
# ============================================================

def simulation_process(frame_queue: mp.Queue, stop_event: mp.Event):
    """
    MuJoCo 仿真进程。
    负责：
        1. mj_step
        2. 触觉读取
        3. 可选 MuJoCo viewer
        4. 可选 top camera 低频渲染
        5. 将最新可视化数据发给 visualization_process
    """
    print("[SIM] loading model...")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    print_sensor_info(model)

    right_reader = build_tactile_reader(
        model,
        keyword="touch_point_right",
        grid_shape=(GRID_H, GRID_W)
    )

    left_reader = build_tactile_reader(
        model,
        keyword="touch_point_left",
        grid_shape=(GRID_H, GRID_W)
    )

    # 夹爪 actuator
    try:
        gripper_control = model.actuator("gripper").id
        print(f"[SIM] gripper actuator id = {gripper_control}")
    except Exception:
        gripper_control = None
        print("[SIM] WARNING: 找不到 actuator 'gripper'")

    # MuJoCo viewer
    viewer = None
    if ENABLE_MUJOCO_VIEWER:
        viewer = mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=False,
            show_right_ui=False
        )

    # top camera offscreen renderer
    offscreen = None
    if ENABLE_TOP_CAMERA:
        try:
            offscreen = init_offscreen_renderer(
                model,
                camera_name=CAMERA_NAME,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT
            )
            print("[SIM] top camera offscreen renderer initialized")
        except Exception as e:
            print("[SIM] WARNING: top camera 初始化失败，自动关闭。")
            print(e)
            offscreen = None

    sim_rate = RateLimiter(frequency=SIM_HZ)

    vis_interval = max(1, int(round(SIM_HZ / VIS_HZ)))
    viewer_interval = max(1, int(round(SIM_HZ / VIEWER_HZ)))
    camera_interval = max(1, int(round(SIM_HZ / CAMERA_HZ)))

    step_count = 0

    print("[SIM] simulation loop started")
    print(f"[SIM] SIM_HZ={SIM_HZ}, VIS every {vis_interval} steps")
    print(f"[SIM] VIEWER sync every {viewer_interval} steps")
    print(f"[SIM] CAMERA render every {camera_interval} steps")

    try:
        while not stop_event.is_set():
            loop_start = time.perf_counter()

            if viewer is not None and not viewer.is_running():
                stop_event.set()
                break

            step_count += 1

            # =====================================================
            # 这里预留 real2sim 控制映射
            #
            # 示例：
            # data.ctrl[0] = real_joint1
            # data.ctrl[1] = real_joint2
            # ...
            # if gripper_control is not None:
            #     data.ctrl[gripper_control] = real_gripper_value
            # =====================================================

            step_start = time.perf_counter()
            mujoco.mj_step(model, data)
            step_ms = (time.perf_counter() - step_start) * 1000.0

            # =====================================================
            # 低频读取触觉并发送给可视化进程
            # =====================================================
            tactile_ms = 0.0
            if step_count % vis_interval == 0:
                tactile_start = time.perf_counter()
                touch_right = read_tactile(
                    right_reader,
                    data,
                    force_max=FORCE_MAX
                )

                touch_left = read_tactile(
                    left_reader,
                    data,
                    force_max=FORCE_MAX
                )

                packet = {
                    "touch_right": touch_right,
                    "touch_left": touch_left,
                    "top_view": None,
                    "step": step_count,
                    "time": float(data.time),
                }

                put_latest(frame_queue, packet)
                tactile_ms = (time.perf_counter() - tactile_start) * 1000.0

            # =====================================================
            # top camera 更低频渲染。单独发包，避免拖慢触觉帧显示。
            # =====================================================
            camera_ms = 0.0
            if (
                ENABLE_TOP_CAMERA
                and offscreen is not None
                and step_count % camera_interval == 0
            ):
                camera_start = time.perf_counter()
                window, cam, scene, context, viewport = offscreen

                top_view = render_camera_view(
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

                put_latest(
                    frame_queue,
                    {
                        "touch_right": None,
                        "touch_left": None,
                        "top_view": top_view,
                        "step": step_count,
                        "time": float(data.time),
                    }
                )
                camera_ms = (time.perf_counter() - camera_start) * 1000.0

            # =====================================================
            # MuJoCo viewer 降频同步
            # =====================================================
            viewer_ms = 0.0
            if viewer is not None and step_count % viewer_interval == 0:
                viewer_start = time.perf_counter()
                viewer.sync()
                viewer_ms = (time.perf_counter() - viewer_start) * 1000.0

            loop_ms = (time.perf_counter() - loop_start) * 1000.0
            if PERF_LOG_INTERVAL and step_count % PERF_LOG_INTERVAL == 0:
                print(
                    "[PERF] "
                    f"loop={loop_ms:.2f} ms, "
                    f"mj_step={step_ms:.2f} ms, "
                    f"tactile={tactile_ms:.2f} ms, "
                    f"camera={camera_ms:.2f} ms, "
                    f"viewer={viewer_ms:.2f} ms"
                )

            sim_rate.sleep()

    except KeyboardInterrupt:
        stop_event.set()

    finally:
        if viewer is not None:
            viewer.close()

        if offscreen is not None:
            window, cam, scene, context, viewport = offscreen
            glfw.destroy_window(window)
            glfw.terminate()

        print("[SIM] simulation process stopped")


# ============================================================
# 主入口
# ============================================================

def main():
    """
    主进程只负责启动两个子进程：
        1. simulation_process
        2. visualization_process
    """

    # spawn 比 fork 更安全，尤其是涉及 MuJoCo / OpenGL / GLFW 时
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
