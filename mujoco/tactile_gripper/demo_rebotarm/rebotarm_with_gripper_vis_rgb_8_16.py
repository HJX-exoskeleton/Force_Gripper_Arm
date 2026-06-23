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


# ============================================================
# 基本配置
# ============================================================

XML_PATH = "/home/hjx/hjx_file/STF/Force_Gripper_Arm/mujoco/assets_robot_xml/rebotarm_tactile/rebotarm_sim_cupboard.xml"

# 你的这版代码是 8 x 16 触觉阵列
GRID_H = 8
GRID_W = 16
FORCE_MAX = 1.0

# 注意：
# 你原代码 idx = x + y * 8
# 这说明 sensor 顺序是：
# y 先分组，每一列里 x 从 0 到 7
# 因此从一维 values 恢复成矩阵时应使用 values.reshape(GRID_W, GRID_H).T
SENSOR_ORDER = "column_major_xy"

# 仿真步长
MODEL_TIMESTEP = 0.005

# 仿真频率
# 原代码是 200 Hz，但显示和渲染太重，建议先 100 Hz
SIM_HZ = 100.0

# OpenCV 触觉显示频率
VIS_HZ = 20.0

# MuJoCo viewer 同步频率
VIEWER_HZ = 20.0

# top camera 渲染频率
CAMERA_HZ = 10.0

# 是否启用 MuJoCo viewer
ENABLE_MUJOCO_VIEWER = True

# 是否启用 top camera
# 如果还卡，优先改成 False
ENABLE_TOP_CAMERA = True

# top camera 设置
CAMERA_NAME = "top"
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

# 触觉热力图显示尺寸
# 原代码 resize 到 480 x 240，然后旋转 90 度
HEATMAP_RESIZE_W = 480
HEATMAP_RESIZE_H = 240
ROTATE_HEATMAP_CLOCKWISE = True

# 队列只保存最新几帧，避免可视化延迟堆积
QUEUE_MAXSIZE = 2


# ============================================================
# 队列工具：只保留最新帧
# ============================================================

def put_latest(q: mp.Queue, item):
    """
    放入最新帧。
    如果队列已满，先丢弃旧帧，避免显示延迟越来越大。
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
# Sensor 工具函数
# ============================================================

def print_sensor_info(model):
    """
    打印 MuJoCo 展开后的 sensor 信息。
    用于确认 sensor 顺序、维度和地址。
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
    根据 sensor 名称关键词筛选 sensor id。
    """
    sensor_ids = []

    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)

        if name is None:
            continue

        if keyword in name:
            sensor_ids.append(i)

    return sensor_ids


def build_tactile_reader(model, keyword, grid_shape=(8, 16)):
    """
    构建触觉读取器。

    期望：
        8 x 16 = 128 个 force sensor

    如果只找到 1 个 sensor，则说明 XML 当前只提供整块触觉垫总力，
    无法恢复真实 8x16 空间分布，只能显示为均匀热力图。
    """
    h, w = grid_shape
    expected_num = h * w

    sensor_ids = find_sensors_by_keyword(model, keyword)

    print(f"[INFO] keyword='{keyword}', found {len(sensor_ids)} sensors")

    if len(sensor_ids) == expected_num:
        dims = np.array(
            [model.sensor_dim[sid] for sid in sensor_ids],
            dtype=np.int32
        )

        adrs = np.array(
            [model.sensor_adr[sid] for sid in sensor_ids],
            dtype=np.int32
        )

        if not np.all(dims == 3):
            print(f"[WARNING] '{keyword}' 中不是所有 sensor 都是 dim=3")
            print("unique dims =", np.unique(dims))

        return {
            "mode": "grid_force",
            "keyword": keyword,
            "sensor_ids": sensor_ids,
            "dims": dims,
            "adrs": adrs,
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
            "[WARNING] 当前只能显示整块触觉垫总力，"
            "无法得到真实 8x16 空间分布。"
        )

        return {
            "mode": "single_force",
            "keyword": keyword,
            "sensor_ids": sensor_ids,
            "dim": dim,
            "adr": adr,
            "grid_shape": grid_shape,
        }

    raise RuntimeError(
        f"触觉 sensor 数量异常：keyword='{keyword}', found={len(sensor_ids)}。"
        f" 当前代码期望 {expected_num} 个，也就是 {h}x{w}。"
        f" 请检查 XML 是否真的使用了 8x16 触觉垫。"
    )


def values_to_grid(values, grid_shape=(8, 16), order="column_major_xy"):
    """
    将一维触觉值恢复成 8x16 矩阵。

    你的原代码使用：
        idx = x + y * 8

    所以：
        x = idx % 8
        y = idx // 8

    因此应使用：
        values.reshape(16, 8).T
    """
    h, w = grid_shape

    if order == "column_major_xy":
        return values.reshape(w, h).T

    if order == "row_major_xy":
        return values.reshape(h, w)

    raise ValueError(f"未知 SENSOR_ORDER: {order}")


def read_tactile(reader, data, force_max=1.0, sensor_order="column_major_xy"):
    """
    读取触觉矩阵，返回 shape = (8, 16) 的 float32 数组。
    """
    mode = reader["mode"]
    h, w = reader["grid_shape"]

    if mode == "grid_force":
        adrs = reader["adrs"]
        dims = reader["dims"]

        # 常规情况：每个触觉点是 3D force sensor
        if np.all(dims == 3):
            index = adrs[:, None] + np.array([0, 1, 2], dtype=np.int32)
            raw = data.sensordata[index]
            values = np.linalg.norm(raw, axis=1)
            values = np.clip(values, 0.0, force_max)
            grid = values_to_grid(
                values,
                grid_shape=(h, w),
                order=sensor_order
            )
            return grid.astype(np.float32)

        # 兼容非 3D sensor
        values = np.zeros(h * w, dtype=np.float32)

        for k, adr in enumerate(adrs):
            dim = int(dims[k])
            raw = data.sensordata[adr:adr + dim]

            if dim == 1:
                value = raw[0]
            else:
                value = np.linalg.norm(raw)

            values[k] = np.clip(value, 0.0, force_max)

        grid = values_to_grid(
            values,
            grid_shape=(h, w),
            order=sensor_order
        )

        return grid.astype(np.float32)

    if mode == "single_force":
        adr = reader["adr"]
        dim = reader["dim"]
        raw = data.sensordata[adr:adr + dim]

        if dim == 1:
            value = raw[0]
        else:
            value = np.linalg.norm(raw)

        value = float(np.clip(value, 0.0, force_max))

        # 注意：这是总力均匀显示，不是真实空间分布
        return np.full((h, w), value, dtype=np.float32)

    raise RuntimeError(f"未知 tactile reader mode: {mode}")


# ============================================================
# 热力图转换
# ============================================================

def tactile_to_colormap(
    tactile,
    force_max=1.0,
    resize_w=480,
    resize_h=240,
    rotate_clockwise=True
):
    """
    将 8x16 触觉矩阵转换成 OpenCV 热力图。
    保持你原来的显示方式：
        8x16 → resize(480, 240) → 顺时针旋转 90 度
    """
    tactile_normalized = np.clip(tactile, 0.0, force_max) / force_max * 255.0
    tactile_uint8 = tactile_normalized.astype(np.uint8)

    colored = cv2.applyColorMap(tactile_uint8, cv2.COLORMAP_VIRIDIS)

    colored_resized = cv2.resize(
        colored,
        (resize_w, resize_h),
        interpolation=cv2.INTER_NEAREST
    )

    if rotate_clockwise:
        colored_resized = cv2.rotate(
            colored_resized,
            cv2.ROTATE_90_CLOCKWISE
        )

    return colored_resized


# ============================================================
# top camera 离屏渲染
# ============================================================

def init_offscreen_renderer(model, camera_name="top", width=320, height=240):
    """
    初始化 MuJoCo 离屏渲染。
    必须在仿真进程内部调用。
    """
    if not glfw.init():
        raise RuntimeError("GLFW 初始化失败")

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
        raise RuntimeError(f"找不到名为 '{camera_name}' 的 camera")

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


def render_camera_view(
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
    独立 OpenCV 可视化进程。
    只负责显示，不负责 MuJoCo 仿真。
    """
    print("[VIS] visualization process started")

    cv2.namedWindow("Touch Heatmap - Sensor Right", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Touch Heatmap - Sensor Left", cv2.WINDOW_NORMAL)

    if ENABLE_TOP_CAMERA:
        cv2.namedWindow("Top Camera View", cv2.WINDOW_NORMAL)

    last_top_view = None

    while not stop_event.is_set():
        try:
            packet = frame_queue.get(timeout=0.05)
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
                resize_w=HEATMAP_RESIZE_W,
                resize_h=HEATMAP_RESIZE_H,
                rotate_clockwise=ROTATE_HEATMAP_CLOCKWISE
            )
            cv2.imshow("Touch Heatmap - Sensor Right", right_img)

        if touch_left is not None:
            left_img = tactile_to_colormap(
                touch_left,
                force_max=FORCE_MAX,
                resize_w=HEATMAP_RESIZE_W,
                resize_h=HEATMAP_RESIZE_H,
                rotate_clockwise=ROTATE_HEATMAP_CLOCKWISE
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
        3. MuJoCo viewer 低频同步
        4. top camera 低频渲染
        5. 向可视化进程发送最新帧
    """
    print("[SIM] loading model...")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    model.opt.timestep = MODEL_TIMESTEP

    print(f"[SIM] model timestep = {model.opt.timestep}")
    print_sensor_info(model)

    # 构建左右触觉读取器
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

    # top camera
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
            print("[SIM] WARNING: top camera 初始化失败，自动关闭")
            print(e)
            offscreen = None

    sim_rate = RateLimiter(frequency=SIM_HZ)

    vis_interval = max(1, int(round(SIM_HZ / VIS_HZ)))
    viewer_interval = max(1, int(round(SIM_HZ / VIEWER_HZ)))
    camera_interval = max(1, int(round(SIM_HZ / CAMERA_HZ)))

    print("[SIM] simulation loop started")
    print(f"[SIM] SIM_HZ={SIM_HZ}")
    print(f"[SIM] tactile vis every {vis_interval} steps")
    print(f"[SIM] viewer sync every {viewer_interval} steps")
    print(f"[SIM] camera render every {camera_interval} steps")

    step_count = 0

    try:
        while not stop_event.is_set():

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

            mujoco.mj_step(model, data)

            # =====================================================
            # 低频读取触觉并发送给可视化进程
            # =====================================================
            if step_count % vis_interval == 0:
                touch_right = read_tactile(
                    right_reader,
                    data,
                    force_max=FORCE_MAX,
                    sensor_order=SENSOR_ORDER
                )

                touch_left = read_tactile(
                    left_reader,
                    data,
                    force_max=FORCE_MAX,
                    sensor_order=SENSOR_ORDER
                )

                packet = {
                    "touch_right": touch_right,
                    "touch_left": touch_left,
                    "top_view": None,
                    "step": step_count,
                    "time": float(data.time),
                }

                # top camera 低频渲染
                if (
                    ENABLE_TOP_CAMERA
                    and offscreen is not None
                    and step_count % camera_interval == 0
                ):
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

                    packet["top_view"] = top_view

                put_latest(frame_queue, packet)

            # =====================================================
            # MuJoCo viewer 低频同步
            # =====================================================
            if viewer is not None and step_count % viewer_interval == 0:
                viewer.sync()

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
# 主函数
# ============================================================

def main():
    """
    主进程启动：
        1. MuJoCo 仿真进程
        2. OpenCV 可视化进程
    """

    # spawn 对 MuJoCo / GLFW / OpenGL 更安全
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
