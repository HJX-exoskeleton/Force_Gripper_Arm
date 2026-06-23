import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3, SO3, UnitQuaternion
import modern_robotics as mr

from .robot import Robot, get_transformation_mdh, wrap
from .robot_config import RobotConfig


class IIWA14(Robot):
    def __init__(self, tool=np.zeros(3)) -> None:
        super().__init__()
        self.d1 = 0.1575 + 0.2025
        self.d3 = 0.2045 + 0.2155
        self.d5 = 0.1845 + 0.2155
        self.d7 = 0.081 + 0.045

        self._dof = 7
        self.q0 = [0.0 for _ in range(self._dof)]

        self.alpha_array = [0.0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2]
        self.a_array = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.d_array = [self.d1, 0.0, self.d3, 0.0, self.d5, 0.0, self.d7]
        self.theta_array = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.sigma_array = [0, 0, 0, 0, 0, 0, 0]
        self.tool = SE3.Trans(tool)

        m1 = 5.76
        r1 = np.array([0.0, -0.03, -0.2025 + 0.12])
        Ixx1 = 0.0333
        Iyy1 = 0.033
        Izz1 = 0.0123
        I1 = np.diag([Ixx1, Iyy1, Izz1])
        R1 = SO3().R

        m2 = 6.35
        r2 = np.array([-0.0003, -0.059, 0.042])
        Ixx2 = 0.0305
        Iyy2 = 0.0304
        Izz2 = 0.011
        I2 = np.diag([Ixx2, Iyy2, Izz2])
        R2 = SO3.Rx(np.pi / 2).R

        m3 = 3.5
        r3 = np.array([0.0, 0.03, -0.2155 + 0.13])
        Ixx3 = 0.025
        Iyy3 = 0.0238
        Izz3 = 0.0076
        I3 = np.diag([Ixx3, Iyy3, Izz3])
        R3 = SO3().R

        m4 = 3.5
        r4 = np.array([0.0, 0.067, 0.034])
        Ixx4 = 0.017
        Iyy4 = 0.0164
        Izz4 = 0.006
        I4 = np.diag([Ixx4, Iyy4, Izz4])
        R4 = UnitQuaternion([1, 1, 0, 0]).R

        m5 = 3.5
        r5 = np.array([-0.0001, -0.021, -0.2155 + 0.076])
        Ixx5 = 0.01
        Iyy5 = 0.0087
        Izz5 = 0.00449
        I5 = np.diag([Ixx5, Iyy5, Izz5])
        R5 = SE3.Rz(np.pi).R

        m6 = 1.8
        r6 = np.array([0.0, -0.0006, 0.0004])
        Ixx6 = 0.0049
        Iyy6 = 0.0047
        Izz6 = 0.0036
        I6 = np.diag([Ixx6, Iyy6, Izz6])
        R6 = UnitQuaternion([0, 0, -1, -1]).R

        m7 = 1.2
        r7 = np.array([0.0, 0.0, -0.045 + 0.02])
        Ixx7 = 0.001
        Iyy7 = 0.001
        Izz7 = 0.001
        I7 = np.diag([Ixx7, Iyy7, Izz7])
        R7 = SO3().R

        ms = [m1, m2, m3, m4, m5, m6, m7]
        rs = [r1, r2, r3, r4, r5, r6, r7]
        Is = [I1, I2, I3, I4, I5, I6, I7]
        Rs = [R1, R2, R3, R4, R5, R6, R7]

        T = SE3()
        self.Slist = np.zeros((6, self._dof))
        self.Glist = []
        self.Jms = []
        self.Mlist = []
        for i in range(self._dof):
            Ti = get_transformation_mdh(self.alpha_array[i], self.a_array[i], self.d_array[i], self.theta_array[i],
                                        self.sigma_array[i], 0.0)
            if i == 0:
                self.Mlist.append(T.A @ Ti.A)
            else:
                self.Mlist.append(Ti.A)

            T: SE3 = T * Ti
            self.Slist[:, i] = np.hstack((T.a, np.cross(T.t, T.a)))

            Gm = np.zeros((6, 6))
            Gm[:3, :3] = Is[i]
            Gm[3:, 3:] = ms[i] * np.eye(3)
            Tab = mr.RpToTrans(Rs[i], rs[i])
            Tba = mr.TransInv(Tab)
            AdT = mr.Adjoint(Tba)
            self.Glist.append(AdT.T @ Gm @ AdT)

        self.M = T.A
        self.Mlist.append(SE3().A)

        links = []
        for i in range(self._dof):
            links.append(
                rtb.DHLink(d=self.d_array[i], alpha=self.alpha_array[i], a=self.a_array[i], offset=self.theta_array[i],
                           mdh=True, m=ms[i], r=rs[i], I=(Rs[i] @ Is[i] @ Rs[i].T)))
            self.robot = rtb.DHRobot(links, tool=self.tool)

            self.robot_config = RobotConfig()

    def ikine(self, Tep: SE3) -> np.ndarray:
        T07: SE3 = Tep * self.tool.inv()

        a = T07.a
        t = T07.t

        S = np.array([0, 0, self.d1])
        W = t - self.d7 * a

        L_sw = np.linalg.norm(W - S)
        V_sw = (W - S) / L_sw

        qs = np.zeros(self._dof)

        # solve q4
        q4_condition = np.power(L_sw, 2) - np.power(self.d3, 2) - np.power(self.d5, 2)
        if np.abs(q4_condition) > (2 * self.d3 * self.d5):
            return np.array([])
        if self.robot_config.inline == 0:
            qs[3] = np.arccos(q4_condition / (2 * self.d3 * self.d5))
        elif self.robot_config.inline == 1:
            qs[3] = -np.arccos(q4_condition / (2 * self.d3 * self.d5))
        else:
            raise ValueError("Wrong inline value")

        x = (L_sw * L_sw + self.d3 * self.d3 - self.d5 * self.d5) / (2 * L_sw)
        r = np.sqrt(self.d3 * self.d3 - x * x)

        F = S + x * V_sw

        L_fe = np.array([(- V_sw[0] * V_sw[2]) / (V_sw[0] * V_sw[0] + V_sw[1] * V_sw[1]),
                         (- V_sw[1] * V_sw[2]) / (V_sw[0] * V_sw[0] + V_sw[1] * V_sw[1]),
                         1])

        E = L_fe / np.linalg.norm(L_fe) * r + F

        V_se = (E - S) / np.linalg.norm(E - S)
        V_ew = (W - E) / np.linalg.norm(W - E)

        R30_z = V_se
        R30_y = - np.sign(qs[3]) * np.cross(V_se, V_ew)
        R30_x = np.cross(R30_y, R30_z)
        R30 = np.vstack((R30_x, R30_y, R30_z)).T

        u_hat = np.array([[0, -V_sw[2], V_sw[1]],
                          [V_sw[2], 0, -V_sw[0]],
                          [-V_sw[1], V_sw[0], 0]])
        R3 = (np.eye(3) + u_hat * np.sin(self._phi) + u_hat @ u_hat * (1 - np.cos(self._phi))) @ R30

        # solve q2
        if self.robot_config.overhead == 0:
            qs[1] = np.arccos(R3[2, 2])
        elif self.robot_config.overhead == 1:
            qs[1] = -np.arccos(R3[2, 2])
        else:
            raise ValueError("Wrong overhead value")

        qs[0] = np.arctan2(R3[1, 2] / np.sin(qs[1]), R3[0, 2] / np.sin(qs[1]))
        qs[2] = np.arctan2(R3[2, 1] / np.sin(qs[1]), -R3[2, 0] / np.sin(qs[1]))

        T01 = get_transformation_mdh(self.alpha_array[0], self.a_array[0], self.d_array[0], self.theta_array[0],
                                     self.sigma_array[0], qs[0])
        T12 = get_transformation_mdh(self.alpha_array[1], self.a_array[1], self.d_array[1], self.theta_array[1],
                                     self.sigma_array[1], qs[1])
        T23 = get_transformation_mdh(self.alpha_array[2], self.a_array[2], self.d_array[2], self.theta_array[2],
                                     self.sigma_array[2], qs[2])
        T34 = get_transformation_mdh(self.alpha_array[3], self.a_array[3], self.d_array[3], self.theta_array[3],
                                     self.sigma_array[3], qs[3])

        T04: SE3 = T01 * T12 * T23 * T34
        T47 = (T04.inv() * Tep).A

        # solve q6
        if self.robot_config.wrist == 0:
            qs[5] = np.arccos(T47[1, 2])
        elif self.robot_config.wrist == 1:
            qs[5] = - np.arccos(T47[1, 2])
        else:
            raise ValueError("Wrong wrist configuration")
        qs[4] = np.arctan2(-T47[2, 2] / np.sin(qs[5]), T47[0, 2] / np.sin(qs[5]))
        qs[6] = np.arctan2(T47[1, 1] / np.sin(qs[5]), -T47[1, 0] / np.sin(qs[5]))

        q0_s = list(map(wrap, self.q0))

        for i in range(self._dof):
            if qs[i] - q0_s[i][0] > np.pi:
                qs[i] += (q0_s[i][1] - 1) * 2 * np.pi
            elif qs[i] - q0_s[i][0] < -np.pi:
                qs[i] += (q0_s[i][1] + 1) * 2 * np.pi
            else:
                qs[i] += q0_s[i][1] * 2 * np.pi

        return qs

    def set_robot_config(self, q: np.ndarray):
        # overhead
        if wrap(q[1])[0] >= 0:
            self.robot_config.overhead = 0
        else:
            self.robot_config.overhead = 1

        # inline
        if wrap(q[3])[0] >= 0:
            self.robot_config.inline = 0
        else:
            self.robot_config.inline = 1

        # wrist
        if wrap(q[5])[0] >= 0:
            self.robot_config.wrist = 0
        else:
            self.robot_config.wrist = 1

        # phi
        T01 = get_transformation_mdh(self.alpha_array[0], self.a_array[0], self.d_array[0], self.theta_array[0],
                                     self.sigma_array[0], q[0])
        T12 = get_transformation_mdh(self.alpha_array[1], self.a_array[1], self.d_array[1], self.theta_array[1],
                                     self.sigma_array[1], q[1])
        T23 = get_transformation_mdh(self.alpha_array[2], self.a_array[2], self.d_array[2], self.theta_array[2],
                                     self.sigma_array[2], q[2])
        T34 = get_transformation_mdh(self.alpha_array[3], self.a_array[3], self.d_array[3], self.theta_array[3],
                                     self.sigma_array[3], q[3])
        T45 = get_transformation_mdh(self.alpha_array[4], self.a_array[4], self.d_array[4], self.theta_array[4],
                                     self.sigma_array[4], q[4])

        T03 = T01 * T12 * T23
        T04 = T03 * T34
        T05 = T04 * T45

        S = T01.t
        W = T05.t

        L_sw = np.linalg.norm(W - S)
        V_sw = (W - S) / L_sw

        x = (L_sw * L_sw + self.d3 * self.d3 - self.d5 * self.d5) / (2 * L_sw)
        r = np.sqrt(self.d3 * self.d3 - x * x)

        F = S + x * V_sw

        L_fe = np.array([(- V_sw[0] * V_sw[2]) / (V_sw[0] * V_sw[0] + V_sw[1] * V_sw[1]),
                       (- V_sw[1] * V_sw[2]) / (V_sw[0] * V_sw[0] + V_sw[1] * V_sw[1]),
                       1])

        E = L_fe / np.linalg.norm(L_fe) * r + F

        V_se = (E - S) / np.linalg.norm(E - S)
        V_ew = (W - E) / np.linalg.norm(W - E)

        R30_z = V_se
        R30_y = -np.sign(q[3]) * np.cross(V_se, V_ew)
        R30_y = R30_y / np.linalg.norm(R30_y)
        R30_x = np.cross(R30_y, R30_z)
        R30 = np.vstack((R30_x, R30_y, R30_z)).T

        R3 = T03.R

        R_phi = R3 @ np.linalg.inv(R30)

        vec = mr.so3ToVec(mr.MatrixLog3(R_phi))
        self._phi = np.sign(np.dot(vec, V_sw)) * np.linalg.norm(vec)

    def move_cartesian(self, T: SE3):
        q = self.ikine(T)

        if q.size != 0:
            self.q0 = q[:]
