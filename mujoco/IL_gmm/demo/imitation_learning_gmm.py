import numpy as np
import pickle
import argparse
from pathlib import Path
from spatialmath import SE3, SO3
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

import sys
sys.path.append("/home/hjx/hjx_file/STF/Force_Gripper/mujoco/IL_gmm")
from imitation_learning.gaussian_mixture_model import GMM
from src.motion_planning import *
from src.robot import IIWA14

from tqdm import tqdm  # pip install tqdm

BASE_DIR = Path("/home/hjx/hjx_file/STF/Force_Gripper/mujoco/IL_gmm")
COLLECT_PATH = BASE_DIR / "demo/collect_data"
MODEL_PATH = BASE_DIR / "demo/gmm_model/gmm_model.pkl"
SCENE_PATH = BASE_DIR / "assets/kuka_iiwa_14/scene.xml"


def generate_gmm_plot():
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    gmm = GaussianMixture(n_components=4)
    gmm.fit(X)
    labels = gmm.predict(X)

    plt.figure(1)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', alpha=0.6)
    plt.colorbar()
    plt.title('GMM Clustering')
    plt.savefig(BASE_DIR / "demo/GMM_Clustering.png")
    plt.show()

def collect_trajectory_data():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    q0 = np.zeros(model.nq)
    q0[:7] = [0.0, np.pi / 4, 0.0, -np.pi / 4, 0.0, np.pi / 2, 0.0]
    mujoco.mj_setState(model, data, q0, mujoco.mjtState.mjSTATE_QPOS)
    mujoco.mj_forward(model, data)

    data.mocap_pos[0, :] = [np.random.random() * 0.3 + 0.5, np.random.random() * 0.6 - 0.3,
                            np.random.random() * 0.3 + 0.1]

    robot = IIWA14(tool=np.array([0.0, 0.0, 0.1488]))
    robot.set_joint(q0[:7])
    T0 = robot.fkine(q0[:7])
    Te = T0
    gripper_joint = 0.0

    motion_time = 2.0

    time0 = motion_time

    time1 = motion_time
    t0 = T0.t
    R0 = SO3.Ry(np.pi)
    t1 = data.mocap_pos[0, :] + [0.0, 0.0, 0.05]
    R1 = R0
    position_parameter1 = LinePositionParameter(t0, t1)
    attitude_parameter1 = OneAttitudeParameter(R0, R1)
    cartesian_parameter1 = CartesianParameter(position_parameter1, attitude_parameter1)
    velocity_parameter1 = QuinticVelocityParameter(time1)
    trajectory_parameter1 = TrajectoryParameter(cartesian_parameter1, velocity_parameter1)
    trajectory_planner1 = TrajectoryPlanner(trajectory_parameter1)

    time2 = motion_time
    t2 = data.mocap_pos[0, :]
    R2 = R1
    position_parameter2 = LinePositionParameter(t1, t2)
    attitude_parameter2 = OneAttitudeParameter(R1, R2)
    cartesian_parameter2 = CartesianParameter(position_parameter2, attitude_parameter2)
    velocity_parameter2 = QuinticVelocityParameter(time2)
    trajectory_parameter2 = TrajectoryParameter(cartesian_parameter2, velocity_parameter2)
    trajectory_planner2 = TrajectoryPlanner(trajectory_parameter2)

    time3 = motion_time

    times = np.array([time0, time1, time2, time3])
    trajectory_planners = [trajectory_planner1, trajectory_planner2]

    collect_data = []
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            if data.time > 10:
                break

            if time0 <= data.time < np.sum(times):

                if data.time < np.sum(times[:-1]):
                    for i in range(1, times.size - 1):
                        if data.time < np.sum(times[: i + 1]):
                            Te = trajectory_planners[i - 1].interpolate(data.time - np.sum(times[: i]))
                            gripper_joint = 0.0
                            break
                else:
                    gripper_joint = (data.time - np.sum(times[: -1])) / times[-1] * 255

                collect_data.append(np.hstack((data.time - times[0], Te.t, gripper_joint)))

            robot.move_cartesian(Te)
            qe = robot.get_joint()

            ctrl = np.hstack((qe, gripper_joint))
            mujoco.mj_setState(model, data, ctrl, mujoco.mjtState.mjSTATE_CTRL)

            mujoco.mj_step(model, data)

            viewer.sync()

    COLLECT_PATH.mkdir(parents=True, exist_ok=True)
    np.savetxt(COLLECT_PATH / 'collect_data1.csv', np.array(collect_data), delimiter=',')

def train_gmm():
    all_data, ps = [], []
    print("[INFO] 加载采集数据并计算坐标变换...")
    for i in tqdm(range(10), desc="加载数据"):
        data_i = np.genfromtxt(COLLECT_PATH / f'collect_data{i + 1}.csv', delimiter=',')
        A0 = np.eye(5)
        n = (data_i[1, 1:4] - data_i[0, 1:4]) / np.linalg.norm(data_i[1, 1:4] - data_i[0, 1:4])
        o = np.cross([0, 0, 1], n)
        a = np.cross(n, o)
        A0[1:4, 1:4] = np.vstack((n, o / np.linalg.norm(o), a)).T
        ps.append([[A0, data_i[0, :]], [np.eye(5), np.hstack((0, data_i[-1, 1:]))]])
        all_data.append(data_i)
    print("[INFO] 构造统一训练数据张量...")
    collect_data = np.hstack(all_data).T
    nb_data = 3001
    data = np.zeros((5, 2, nb_data * 10))
    for n in tqdm(range(10), desc="转换数据"):
        for m in range(2):
            data[:, m, n * nb_data:(n + 1) * nb_data] = np.linalg.inv(ps[n][m][0]) @ (
                collect_data[n * 5:(n + 1) * 5, :].T - ps[n][m][1]
            ).T
    print("[INFO] 训练GMM模型中...")
    gmm = GMM(nb_states=6, nb_frames=2, nb_var=5)
    gmm.train(data)
    print("[INFO] 保存模型...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(gmm, f)
    print(f"[DONE] 模型已保存至：{MODEL_PATH}")

def reproduce_gmm():
    with open(MODEL_PATH, 'rb') as f:
        gmm = pickle.load(f)

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    q0_robot = [0.0, np.pi / 4, 0.0, -np.pi / 4, 0.0, np.pi / 2, 0.0]
    robot = IIWA14(tool=np.array([0.0, 0.0, 0.1488]))
    robot.set_joint(q0_robot)
    T0 = robot.fkine(q0_robot)

    mujoco.mj_resetData(model, data)
    q0 = np.zeros(model.nq)
    q0[:7] = q0_robot
    mujoco.mj_setState(model, data, q0, mujoco.mjtState.mjSTATE_QPOS)
    mujoco.mj_forward(model, data)

    start = T0.t
    goal = np.array(
        [np.random.random() * 0.3 + 0.4, np.random.random() * 0.6 - 0.3, np.random.random() * 0.3 + 0.1])
    data.mocap_pos[0, :] = goal

    reproduce_trajectory = gmm.reproduce(start=start, goal=goal)

    time_num = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            Te = SE3(reproduce_trajectory[:3, time_num]) * SE3(SO3(T0.R))
            robot.move_cartesian(Te)
            qe = robot.get_joint()

            gripper_joint = reproduce_trajectory[3, time_num]
            ctrl = np.hstack((qe, gripper_joint))

            mujoco.mj_setState(model, data, ctrl, mujoco.mjtState.mjSTATE_CTRL)

            mujoco.mj_step(model, data)

            if data.time > 2.0 and time_num < 3000:
                time_num += 1

            viewer.sync()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        choices=['plot', 'data_collection', 'train', 'eval'], required=True, help='Mode to run')
    args = parser.parse_args()

    if args.mode == 'plot':
        generate_gmm_plot()
    elif args.mode == 'data_collection':
        collect_trajectory_data()
    elif args.mode == 'train':
        train_gmm()
    elif args.mode == 'eval':
        reproduce_gmm()


# -----------演示示例-----------
# python imitation_learning_gmm.py --mode plot
# python imitation_learning_gmm.py --mode data_collection
# python imitation_learning_gmm.py --mode train
# python imitation_learning_gmm.py --mode eval

