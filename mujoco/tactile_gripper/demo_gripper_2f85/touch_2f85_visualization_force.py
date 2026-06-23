import numpy as np
import threading
import cv2
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

# === 参数设置 ===
FPS = 50
touch_shape = (3, 16, 16)  # 原始 shape: (C，H, W)
max_shear = 0.05
max_pressure = 0.1
window_size = (480, 480)

# === 初始化模型与数据 ===
model = mujoco.MjModel.from_xml_path("/home/hjx/hjx_file/STF/Force_Gripper/mujoco/assets_robot_xml/gripper_2f85_insertion/scene.xml")
data = mujoco.MjData(model)
# model.opt.timestep = 0.005

# === 获取传感器索引 ===
sensor_id_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_left")
nsensor_adr_left = model.sensor_adr[sensor_id_left]
nsensor_dim_left = model.sensor_dim[sensor_id_left]

sensor_id_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_right")
nsensor_adr_right = model.sensor_adr[sensor_id_right]
nsensor_dim_right = model.sensor_dim[sensor_id_right]


# === 控制器线程 ===
def mujoco_thread():
    rate = RateLimiter(frequency=200.0)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            rate.sleep()


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

            # start = (int(loc_x[i]), int(loc_y[j]))
            # end = (int(loc_x[i] + dir_x), int(loc_y[j] - dir_y))
            start = (int(loc_y[i]), int(loc_x[j]))
            end = (int(loc_y[i] + dir_y), int(loc_x[j] - dir_x))

            cv2.arrowedLine(img, start, end, color, 2, tipLength=0.5)

    # 旋转图像 180 度
    img_rotated = cv2.rotate(img, cv2.ROTATE_180)
    # 显示旋转后的图像
    cv2.imshow(name, img_rotated)
    # cv2.imshow(name, img)
    return img


# === 实时显示线程 ===
def start_touch_visualizer_cv_arrowed():
    while True:
        raw_left = data.sensordata[nsensor_adr_left:nsensor_adr_left + nsensor_dim_left]
        raw_right = data.sensordata[nsensor_adr_right:nsensor_adr_right + nsensor_dim_right]

        tactile_left = raw_left.reshape(touch_shape)
        tactile_right = raw_right.reshape(touch_shape)

        show_tactile_arrowed(tactile_left, size=window_size, max_shear=max_shear, max_pressure=max_pressure, name='Touch Left')
        show_tactile_arrowed(tactile_right, size=window_size, max_shear=max_shear, max_pressure=max_pressure, name='Touch Right')

        if cv2.waitKey(int(1000 / FPS)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# === 主程序入口 ===
if __name__ == "__main__":
    thread = threading.Thread(target=mujoco_thread)
    thread.daemon = True
    thread.start()

    start_touch_visualizer_cv_arrowed()
