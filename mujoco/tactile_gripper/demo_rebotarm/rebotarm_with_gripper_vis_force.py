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
# 基本参数
# ============================================================

XML_PATH = "/home/hjx/hjx_file/STF/Force_Gripper_Arm/mujoco/assets_robot_xml/rebotarm_tactile/rebotarm_sim_cupboard.xml"

# 触觉数据形状：C, H, W
TACTILE_C = 3
GRID_H = 16
GRID_W = 16
TOUCH_SHAPE = (TACTILE_C, GRID_H, GRID_W)

# 触觉显示参数
MAX_SHEAR = 0.01  # 0.05
MAX_PRESSURE = 0.02  # 0.1
WINDOW_SIZE = (480, 480)

# MuJoCo timestep
MODEL_TIMESTEP = 0.005

# 仿真频率
SIM_HZ = 100.0

# 触觉可视化频率
VIS_HZ = 30.0

# MuJoCo viewer 同步频率
VIEWER_HZ = 30.0

# top camera 渲染频率
CAMERA_HZ = 5.0

# 是否启用 MuJoCo viewer
ENABLE_MUJOCO_VIEWER = True

# 是否启用 top camera。
# 离屏相机渲染通常是本脚本最重的步骤。触觉低延迟优先时建议保持 False。
ENABLE_TOP_CAMERA = False

# top camera 设置
CAMERA_NAME = "top"
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

# 队列只保留最新帧，避免显示延迟堆积
QUEUE_MAXSIZE = 1

# 每隔多少仿真步打印一次耗时统计。0 表示关闭。
PERF_LOG_INTERVAL = 0

# sensor 点阵恢复顺序
# 16x16 代码一般对应 idx = x * 16 + y
SENSOR_ORDER = "row_major_xy"


# ============================================================
# 队列工具：只保留最新帧
# ============================================================

def put_latest(q: mp.Queue, item):
    """
    向队列放入最新帧。
    如果队列满了，丢弃旧帧，避免可视化滞后。
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
# Sensor 工具函数
# ============================================================

def print_sensor_info(model):
    """
    打印所有 sensor 信息，方便检查 XML 展开后的 sensor 结构。
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


def find_exact_sensor(model, name):
    """
    按精确名称查找 sensor。
    找不到返回 -1。
    """
    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        name
    )


def find_sensors_by_keyword(model, keyword):
    """
    按关键词查找 sensor id。
    """
    sensor_ids = []

    for i in range(model.nsensor):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            i
        )

        if name is None:
            continue

        if keyword in name:
            sensor_ids.append(i)

    return sensor_ids


def values_to_grid(values, grid_shape=(16, 16), order="row_major_xy"):
    """
    将一维 values 恢复为 H x W。

    row_major_xy:
        idx = x * W + y
        grid[x, y] = values[idx]

    column_major_xy:
        idx = x + y * H
        grid[x, y] = values[idx]
    """
    h, w = grid_shape

    if order == "row_major_xy":
        return values.reshape(h, w)

    if order == "column_major_xy":
        return values.reshape(w, h).T

    raise ValueError(f"未知 SENSOR_ORDER: {order}")


def build_tactile3_reader(model, side):
    """
    构建 3x16x16 触觉读取器。

    side:
        "left" 或 "right"

    优先尝试：
        touch_left / touch_right

    如果没有完整网格 sensor，则回退到：
        touch_point_left / touch_point_right force sensors
    """
    assert side in ("left", "right")

    expected_dim = TACTILE_C * GRID_H * GRID_W
    exact_name = f"touch_{side}"
    point_keyword = f"touch_point_{side}"

    # ------------------------------------------------------------
    # 模式 1：完整网格 sensor，例如 touch_left / touch_right
    # ------------------------------------------------------------
    sid = find_exact_sensor(model, exact_name)

    if sid >= 0:
        adr = int(model.sensor_adr[sid])
        dim = int(model.sensor_dim[sid])

        print(
            f"[INFO] Found exact sensor '{exact_name}': "
            f"id={sid}, dim={dim}, adr={adr}"
        )

        if dim == expected_dim:
            return {
                "mode": "grid3_exact",
                "side": side,
                "sensor_id": sid,
                "adr": adr,
                "dim": dim,
                "shape": TOUCH_SHAPE,
            }

        if dim == 3:
            print(
                f"[WARNING] sensor '{exact_name}' 只有 dim=3，"
                f"不是完整的 3x16x16 网格。"
            )
            print(
                "[WARNING] 将把它作为整块触觉垫的总力显示，"
                "无法恢复真实空间分布。"
            )

            return {
                "mode": "single_force",
                "side": side,
                "sensor_id": sid,
                "adr": adr,
                "dim": dim,
                "shape": TOUCH_SHAPE,
            }

        print(
            f"[WARNING] sensor '{exact_name}' 维度异常：dim={dim}，"
            f"期望 {expected_dim} 或 3。将继续尝试按点 force sensor。"
        )

    # ------------------------------------------------------------
    # 模式 2：256 个 touch_point_left/right force sensor
    # ------------------------------------------------------------
    sensor_ids = find_sensors_by_keyword(model, point_keyword)

    print(
        f"[INFO] keyword='{point_keyword}', "
        f"found {len(sensor_ids)} sensors"
    )

    expected_num = GRID_H * GRID_W

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
            print(
                f"[WARNING] '{point_keyword}' 中不是所有 sensor 都是 dim=3"
            )
            print("unique dims =", np.unique(dims))

        return {
            "mode": "points_force",
            "side": side,
            "sensor_ids": sensor_ids,
            "dims": dims,
            "adrs": adrs,
            "shape": TOUCH_SHAPE,
        }

    if len(sensor_ids) == 1:
        sid = sensor_ids[0]
        adr = int(model.sensor_adr[sid])
        dim = int(model.sensor_dim[sid])

        print(
            f"[WARNING] '{point_keyword}' 只找到 1 个 sensor: "
            f"id={sid}, dim={dim}, adr={adr}"
        )
        print(
            "[WARNING] 将作为整块触觉垫总力显示，"
            "不能恢复 16x16 空间分布。"
        )

        return {
            "mode": "single_force",
            "side": side,
            "sensor_id": sid,
            "adr": adr,
            "dim": dim,
            "shape": TOUCH_SHAPE,
        }

    raise RuntimeError(
        f"无法构建 {side} 侧触觉读取器。\n"
        f"既没有找到完整 sensor '{exact_name}'，"
        f"也没有找到 {expected_num} 个 '{point_keyword}' force sensor。\n"
        f"当前 '{point_keyword}' 匹配数量 = {len(sensor_ids)}。"
    )


def read_tactile3(reader, data, sensor_order="row_major_xy"):
    """
    读取触觉数据，统一返回 shape=(3,16,16)。

    channel 含义：
        0: shear_x 或 Fx
        1: shear_y 或 Fy
        2: pressure 或 Fz
    """
    mode = reader["mode"]

    if mode == "grid3_exact":
        adr = reader["adr"]
        dim = reader["dim"]
        raw = data.sensordata[adr:adr + dim]
        tactile = raw.reshape(TOUCH_SHAPE).astype(np.float32)
        return tactile

    if mode == "points_force":
        adrs = reader["adrs"]
        dims = reader["dims"]

        # 常规情况：每个点 3 维 force
        if np.all(dims == 3):
            index = adrs[:, None] + np.array([0, 1, 2], dtype=np.int32)
            raw = data.sensordata[index]  # shape = (256, 3)

            fx = values_to_grid(
                raw[:, 0],
                grid_shape=(GRID_H, GRID_W),
                order=sensor_order
            )

            fy = values_to_grid(
                raw[:, 1],
                grid_shape=(GRID_H, GRID_W),
                order=sensor_order
            )

            fz = values_to_grid(
                raw[:, 2],
                grid_shape=(GRID_H, GRID_W),
                order=sensor_order
            )

            tactile = np.stack([fx, fy, fz], axis=0).astype(np.float32)
            return tactile

        # 兼容非 3 维 sensor
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

    if mode == "single_force":
        adr = reader["adr"]
        dim = reader["dim"]
        raw = data.sensordata[adr:adr + dim]

        tactile = np.zeros(TOUCH_SHAPE, dtype=np.float32)

        if dim >= 3:
            fx, fy, fz = raw[0], raw[1], raw[2]
        elif dim == 1:
            fx, fy, fz = 0.0, 0.0, raw[0]
        else:
            fx, fy, fz = 0.0, 0.0, 0.0

        tactile[0, :, :] = fx
        tactile[1, :, :] = fy
        tactile[2, :, :] = fz

        return tactile

    raise RuntimeError(f"未知 tactile reader mode: {mode}")


# ============================================================
# 触觉矢量场绘制
# ============================================================

def tactile_to_arrow_image(
    tactile,
    size=(480, 480),
    max_shear=0.05,
    max_pressure=0.1,
    arrow_scale=20.0,
    rotate_180=True
):
    """
    触觉方向箭头可视化。

    这个函数严格遵循你原始 show_tactile_arrowed() 的方向逻辑：

        dir_x = tactile[0, j, i]
        dir_y = tactile[1, j, i]

        start = (loc_y[i], loc_x[j])
        end   = (loc_y[i] + dir_y, loc_x[j] - dir_x)

        img_rotated = cv2.rotate(img, cv2.ROTATE_180)

    不使用普通 OpenCV 图像坐标解释，否则箭头方向会错。
    """
    channels, ny, nx = tactile.shape

    if channels != 3:
        raise ValueError(
            f"Tactile data must have 3 channels, got shape={tactile.shape}"
        )

    # 严格保留原始代码的 loc_x / loc_y 定义
    loc_x = np.linspace(0, size[1], nx)
    loc_y = np.linspace(size[0], 0, ny)

    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    for i in range(nx):
        for j in range(ny):
            # === 剪切力方向，严格保留原始定义 ===
            dir_x = np.clip(
                tactile[0, j, i] / max_shear,
                -1.0,
                1.0
            ) * arrow_scale

            dir_y = np.clip(
                tactile[1, j, i] / max_shear,
                -1.0,
                1.0
            ) * arrow_scale

            # === 压力颜色，严格保留原始逻辑 ===
            # 注意：这里不取 abs，否则会改变原始压力方向逻辑
            pressure = np.clip(
                tactile[2, j, i] / max_pressure,
                0.0,
                1.0
            )

            color = (
                0,
                int(255 * (1.0 - pressure)),
                int(255 * pressure)
            )

            # === 关键：严格保留原始 start/end 坐标映射 ===
            start = (
                int(loc_y[i]),
                int(loc_x[j])
            )

            end = (
                int(loc_y[i] + dir_y),
                int(loc_x[j] - dir_x)
            )

            cv2.arrowedLine(
                img,
                start,
                end,
                color,
                2,
                tipLength=0.5
            )

    # 严格保留原始 180° 旋转
    if rotate_180:
        img = cv2.rotate(img, cv2.ROTATE_180)

    return img


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
    """
    print("[VIS] visualization process started")

    cv2.namedWindow("Touch Left", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Touch Right", cv2.WINDOW_NORMAL)

    if ENABLE_TOP_CAMERA:
        cv2.namedWindow("Top Camera View", cv2.WINDOW_NORMAL)

    last_top_view = None

    while not stop_event.is_set():
        try:
            packet = get_latest(frame_queue, timeout=0.02)
        except queue.Empty:
            key = cv2.waitKey(1)
            if key == 27 or key == ord("q"):
                stop_event.set()
            continue

        tactile_left = packet.get("tactile_left", None)
        tactile_right = packet.get("tactile_right", None)
        top_view = packet.get("top_view", None)

        if tactile_left is not None:
            left_img = tactile_to_arrow_image(
                tactile_left,
                size=WINDOW_SIZE,
                max_shear=MAX_SHEAR,
                max_pressure=MAX_PRESSURE,
                rotate_180=True
            )
            cv2.imshow("Touch Left", left_img)

        if tactile_right is not None:
            right_img = tactile_to_arrow_image(
                tactile_right,
                size=WINDOW_SIZE,
                max_shear=MAX_SHEAR,
                max_pressure=MAX_PRESSURE,
                rotate_180=True
            )
            cv2.imshow("Touch Right", right_img)

        if top_view is not None:
            last_top_view = top_view

        if ENABLE_TOP_CAMERA and last_top_view is not None:
            cv2.imshow("Top Camera View", last_top_view)

        key = cv2.waitKey(1)

        if key == 27 or key == ord("q"):
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
    """
    print("[SIM] loading model...")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    model.opt.timestep = MODEL_TIMESTEP

    print(f"[SIM] model timestep = {model.opt.timestep}")
    print_sensor_info(model)

    # 构建左右触觉读取器
    left_reader = build_tactile3_reader(model, side="left")
    right_reader = build_tactile3_reader(model, side="right")

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
            # 低频读取触觉并发给可视化进程
            # =====================================================
            tactile_ms = 0.0
            if step_count % vis_interval == 0:
                tactile_start = time.perf_counter()
                tactile_left = read_tactile3(
                    left_reader,
                    data,
                    sensor_order=SENSOR_ORDER
                )

                tactile_right = read_tactile3(
                    right_reader,
                    data,
                    sensor_order=SENSOR_ORDER
                )

                packet = {
                    "tactile_left": tactile_left,
                    "tactile_right": tactile_right,
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
                        "tactile_left": None,
                        "tactile_right": None,
                        "top_view": top_view,
                        "step": step_count,
                        "time": float(data.time),
                    }
                )
                camera_ms = (time.perf_counter() - camera_start) * 1000.0

            # =====================================================
            # MuJoCo viewer 低频同步
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
# 主函数
# ============================================================

def main():
    """
    主进程启动：
        1. MuJoCo 仿真进程
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
