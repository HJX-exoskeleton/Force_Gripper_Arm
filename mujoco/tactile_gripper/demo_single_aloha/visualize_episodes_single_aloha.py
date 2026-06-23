import h5py
import cv2
import numpy as np
import os


def hdf5_to_video(hdf5_path, output_video_path, fps=25):
    # 打开 HDF5 文件
    with h5py.File(hdf5_path, 'r') as f:
        # 读取图像数据（top 视角）
        images = f['observations/images/top'][:]

    # 确定图像尺寸
    height, width = images.shape[1:3]

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可选: 'XVID', 'avc1', 'mp4v'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"开始写入视频，共 {len(images)} 帧...")
    for i, img in enumerate(images):
        # 如果图像为 RGB 格式，需转为 BGR 供 OpenCV 写入
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(bgr_img)
        if i % 50 == 0:
            print(f"已处理 {i} 帧")

    out.release()
    print(f"✅ 视频保存完成: {output_video_path}")


# 示例用法
if __name__ == "__main__":
    hdf5_file = '/home/hjx/hjx_file/STF/Force_Gripper/data/data_tactile_single_aloha/episode_0.hdf5'
    output_video = '/home/hjx/hjx_file/STF/Force_Gripper/data/data_tactile_single_aloha/episode_0_video.mp4'
    hdf5_to_video(hdf5_file, output_video, fps=50)
