import os
# 如果你之前遇到 OpenGL 报错，可以取消下一行注释
# os.environ["MUJOCO_GL"] = "egl"

import time
import cv2
import mujoco
import numpy as np


XML_PATH = "/home/hjx/hjx_file/STF/Force_Gripper_Arm/mujoco/assets_robot_xml/gripper_2f85/scene_8_16.xml"
CAMERA_NAME = "top_cam"

WIDTH = 640
HEIGHT = 480


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam_id < 0:
        raise ValueError(f"XML 中没有找到相机: {CAMERA_NAME}")

    print(f"已找到相机: {CAMERA_NAME}, id = {cam_id}")
    print("按 q 退出实时相机窗口。")

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    while True:
        # 推进一步仿真
        mujoco.mj_step(model, data)

        # 从 top_cam 渲染 RGB 画面
        renderer.update_scene(data, camera=CAMERA_NAME)
        rgb = renderer.render()

        # OpenCV 使用 BGR，所以要转换
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow("MuJoCo top_cam RGB", bgr)

        # 按 q 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # 控制显示速度，避免 CPU 占满
        time.sleep(0.01)

    cv2.destroyAllWindows()
    renderer.close()


if __name__ == "__main__":
    main()
