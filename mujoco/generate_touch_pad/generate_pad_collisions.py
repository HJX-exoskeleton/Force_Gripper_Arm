from absl import app, flags  # 引入 absl 库中的 app 和 flags 模块，用于命令行参数解析
import shutil                # 引入 shutil 模块，用于文件复制操作
import os                    # 引入 os 模块，用于路径操作

FLAGS = flags.FLAGS          # 定义全局 FLAGS 变量，便于后续使用定义的命令行参数
flags.DEFINE_integer('nx', 16, 'Number of pads in x direction')  # 定义命令行参数 nx，表示 x 方向的 pad 数量
flags.DEFINE_integer('ny', 16, 'Number of pads in y direction')  # 定义命令行参数 ny，表示 y 方向的 pad 数量


def main(_):  # 主函数，absl 框架要求的格式（main 接收一个参数 _，通常是 argv）

    dx = FLAGS.nx  # 获取 x 方向 pad 数量
    dy = FLAGS.ny  # 获取 y 方向 pad 数量

    size_x = 0.011 / dx         # 每个 pad 在 x 方向上的尺寸（宽度），pad 区域总宽 0.011m
    size_y = 0.009375 * 2 / dy  # 每个 pad 在 y 方向上的尺寸（高度），pad 区域总高 0.009375×2

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建文件路径
    right_pad_path = os.path.join(current_dir, "right_pad_collisions.xml")
    left_pad_path = os.path.join(current_dir, "left_pad_collisions.xml")

    # 打开输出的 XML 文件（写入右手触觉 pad 的碰撞体定义）
    f = open(right_pad_path, "w")
    f.write("<mujoco>\n")  # 写入文件头 <mujoco>

    # 双层嵌套循环，依次生成 dy 行、dx 列的 pad 几何体
    for i in range(dy):  # 遍历 y 方向上的行
        pos_y = size_y + 2 * size_y * i  # 计算每一行 pad 的中心位置 y 坐标（含间隔）

        for j in range(dx):  # 遍历 x 方向上的列
            pos_x = -0.011 + size_x + 2 * size_x * j  # 计算每个 pad 的 x 坐标（-0.011 是左边界）

            rgb = 0.6 + 0.1 * (i * dx + j) / (dx * dy - 1)  # 根据 pad 编号生成灰度色调（从0.6~0.7之间变化）

            # 创建 XML 元素字符串，每个 pad 是一个 box geom，尺寸为 size_x 和 size_y
            xml_string = "<geom class=\"pad\" pos=\"{} -0.0026 {}\" size=\"{} 0.004 {}\" rgba=\"{} {} {} 1\"/>".format(
                pos_x, pos_y, size_x, size_y, rgb, rgb, rgb)

            f.write(xml_string + '\n')  # 将该 pad 写入 XML 文件

    f.write("</mujoco>")  # 写入文件尾
    f.close()             # 关闭文件写入

    # 将生成的右手 pad 定义复制一份为左手使用（文件内容完全相同）
    shutil.copyfile(right_pad_path, left_pad_path)

    # 生成触觉传感器 XML 片段，使用 mujoco.sensor.touch_grid 插件，挂载在左右 site 上
    # touch_sensor_string = """
    # <mujoco>
    # <sensor>
    #     <plugin name="touch_right" plugin="mujoco.sensor.touch_grid" objtype="site" objname="touch_right">
    #     <config key="size" value="{} {}"/>  <!-- 定义触觉图像尺寸为 dx × dy -->
    #         <config key="fov" value="14 23"/>  <!-- 视场角，可调控制 pad 区域 -->
    #     <config key="gamma" value="0"/>        <!-- gamma 设为 0 表示无非线性映射 -->
    #     <config key="nchannel" value="3"/>     <!-- 输出为 3 通道 RGB 图像 -->
    #     </plugin>
    # </sensor>
    # <sensor>
    #     <plugin name="touch_left" plugin="mujoco.sensor.touch_grid" objtype="site" objname="touch_left">
    #     <config key="size" value="{} {}"/>  <!-- 同样配置左手触觉 -->
    #         <config key="fov" value="14 23"/>
    #     <config key="gamma" value="0"/>
    #     <config key="nchannel" value="3"/>
    #     </plugin>
    # </sensor>
    # </mujoco>
    # """.format(dx, dy, dx, dy)
    #
    # # 写入触觉传感器 XML 到文件
    # f = open("/home/hjx/hjx_file/3D-ViTac_Tactile_Hardware/python/touch_sensors.xml", "w")
    # f.write(touch_sensor_string)
    # f.close()


# 使用 absl 提供的 app.run 接口启动程序（调用 main 函数）
if __name__ == "__main__":
    app.run(main)
