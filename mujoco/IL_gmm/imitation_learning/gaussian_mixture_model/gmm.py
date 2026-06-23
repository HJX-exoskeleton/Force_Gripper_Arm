import numpy as np

REAL_MIN = 1e-12


def gauss_pdf(data: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    if len(data.shape) == 1:
        nb_var = 1
        data = data.reshape((1, -1))
        sigma = sigma.reshape((1, -1))
    else:
        nb_var = data.shape[0]

    data = (data.T - mu).T
    prob = np.sum(np.linalg.pinv(sigma) @ data * data, axis=0)
    prob = np.exp(-0.5 * prob) / np.sqrt(np.power(2 * np.pi, nb_var) * np.abs(np.linalg.det(sigma))) + REAL_MIN
    return prob


class GMM:
    def __init__(self, nb_states: int, nb_frames: int, nb_var: int) -> None:
        super().__init__()

        self._nb_states = nb_states
        self._nb_frames = nb_frames
        self._nb_var = nb_var

        self._params_diag_reg_fact = 1e-4

        self._diag_regularization_factor = 1e-4

        self.priors = np.array([])
        self.mu = np.array([])
        self.sigma = np.array([])

        self.params_nb_min_steps = 4
        self.params_nb_max_steps = 100
        self.params_max_diffLL = 1e-5
        self.params_update_comp = np.ones(3)

        self.nbData = 0
        self.in_ = np.array([])
        self.out = np.array([])
        self.MuGMR = np.array([])
        self.SigmaGMR = np.array([])

    @property
    def nb_states(self):
        return self._nb_states

    @property
    def nb_frames(self):
        return self._nb_frames

    @property
    def nb_var(self):
        return self._nb_var

    @property
    def params_diagRegFact(self):
        return self._params_diag_reg_fact

    def train(self, data: np.ndarray):

        data_all = data.reshape((data.shape[0] * data.shape[1], data.shape[2]), order='F')

        timing_sep = np.linspace(np.min(data_all[0, :]), np.max(data_all[0, :]), self._nb_states + 1)

        mu = np.zeros((self._nb_frames * self._nb_var, self._nb_states))
        sigma = np.zeros((self._nb_frames * self._nb_var, self._nb_frames * self._nb_var, self._nb_states))

        self.mu = np.zeros((self._nb_var, self._nb_frames, self._nb_states))
        self.sigma = np.zeros((self._nb_var, self._nb_var, self._nb_frames, self._nb_states))
        self.priors = np.zeros(self._nb_states)

        for i in range(self._nb_states):
            idtmp: np.ndarray = np.where((data_all[0, :] >= timing_sep[i]) & (data_all[0, :] < timing_sep[i + 1]))[0]
            mu[:, i] = np.mean(data_all[:, idtmp], 1)
            sigma[:, :, i] = np.cov(data_all[:, idtmp]) + np.eye(data_all.shape[0]) * self._diag_regularization_factor
            self.priors[i] = idtmp.size
        self.priors = self.priors / np.sum(self.priors)

        for m in range(self._nb_frames):
            for i in range(self._nb_states):
                self.mu[:, m, i] = mu[m * self._nb_var: (m + 1) * self._nb_var, i]
                self.sigma[:, :, m, i] = sigma[m * self._nb_var: (m + 1) * self._nb_var,
                                         m * self._nb_var: (m + 1) * self._nb_var, i]

        nb_data = data.shape[2]
        LL = np.zeros(self.params_nb_max_steps)

        for nb_iter in range(self.params_nb_max_steps):

            L, gamma, gamma0 = self.compute_gamma(data)
            gamma2 = (gamma.T / np.sum(gamma, axis=1)).T
            self.pix = gamma2

            for i in range(self._nb_states):

                if self.params_update_comp[0]:
                    self.priors[i] = np.sum(gamma[i, :]) / nb_data

                for m in range(self._nb_frames):
                    data_mat = data[:, m, :]

                    if self.params_update_comp[1]:
                        self.mu[:, m, i] = data_mat @ gamma2[i, :]

                    if self.params_update_comp[2]:
                        data_tmp = (data_mat.T - self.mu[:, m, i]).T
                        self.sigma[:, :, m, i] = data_tmp @ np.diag(gamma2[i, :]) @ data_tmp.T + np.eye(
                            data_tmp.shape[0]) * self._params_diag_reg_fact

            LL[nb_iter] = np.sum(np.log(np.sum(L, axis=0))) / L.shape[1]

            if nb_iter > self.params_nb_min_steps:
                if (LL[nb_iter] - LL[nb_iter - 1]) < self.params_max_diffLL or nb_iter == self.params_nb_max_steps - 1:
                    break

        self.inv_sigma = np.zeros_like(self.sigma)
        for m in range(self._nb_frames):
            for i in range(self._nb_states):
                self.inv_sigma[:, :, m, i] = np.linalg.inv(self.sigma[:, :, m, i])

        self.update_para()

    def compute_gamma(self, data) -> tuple:
        nb_data = data.shape[2]
        Lik = np.ones((self._nb_states, nb_data))
        gamma0 = np.zeros((self._nb_states, self._nb_frames, nb_data))
        for i in range(self._nb_states):
            for m in range(self._nb_frames):
                data_mat = data[:, m, :]
                gamma0[i, m, :] = gauss_pdf(data_mat, self.mu[:, m, i], self.sigma[:, :, m, i])
                Lik[i, :] = Lik[i, :] * gamma0[i, m, :]
            Lik[i, :] = Lik[i, :] * self.priors[i]
        gamma = Lik / (np.sum(Lik, axis=0) + REAL_MIN)
        return Lik, gamma, gamma0

    def update_para(self):
        self.nbData = 3001
        end_time = 6.0
        DataIn = np.linspace(0.0, end_time, self.nbData)

        self.in_ = np.array([0])
        self.out = np.arange(1, self._nb_var)
        self.MuGMR = np.zeros((self.out.size, self.nbData, self._nb_frames))
        self.SigmaGMR = np.zeros((self.out.size, self.out.size, self.nbData, self._nb_frames))
        H = np.zeros((self._nb_states, self.nbData))

        for m in range(self._nb_frames):
            for i in range(self.nb_states):
                H[i, :] = self.priors[i] * gauss_pdf(DataIn, self.mu[self.in_, m, i],
                                                     self.sigma[self.in_, self.in_, m, i])
            H = H / np.sum(H, axis=0)

            for t in range(self.nbData):
                for i in range(self._nb_states):
                    MuTmp = self.mu[self.out, m, i] + self.sigma[self.out][:, self.in_, m, i] @ np.linalg.inv(
                        self.sigma[self.in_][:, self.in_, m, i]) @ (
                                        np.reshape(DataIn, (len(self.in_), -1))[:, t] - self.mu[self.in_, m, i])
                    self.MuGMR[:, t, m] = self.MuGMR[:, t, m] + H[i, t] * MuTmp

                    SigmaTmp = self.sigma[self.out][:, self.out, m, i] - self.sigma[self.out][:, self.in_, m,
                                                                         i] @ np.linalg.inv(
                        self.sigma[self.in_][:, self.in_, m, i]) @ self.sigma[self.in_][:, self.out, m, i]
                    self.SigmaGMR[:, :, t, m] = self.SigmaGMR[:, :, t, m] + H[i, t] * (
                            SigmaTmp + np.dot(MuTmp.reshape(1, -1).T, MuTmp.reshape(1, -1)))
                self.SigmaGMR[:, :, t, m] = self.SigmaGMR[:, :, t, m] - np.dot(self.MuGMR[:, t, m].reshape(1, -1).T,
                                                                               self.MuGMR[:, t, m].reshape(1,
                                                                                                           -1)) + np.eye(
                    self.out.size) * self._params_diag_reg_fact

    def reproduce(self, start: np.ndarray, goal: np.ndarray):
        A0 = np.eye(5)

        goal_after = goal + np.array([0.0, 0.0, 0.05])

        n = (goal_after - start) / np.linalg.norm(goal_after - start)
        a = np.array([0, 0, 1])
        o = np.cross(a, n) / np.linalg.norm(np.cross(a, n))
        a = np.cross(n, o)
        A0[1:4, 1:4] = np.vstack((n, o, a)).T

        b0 = np.array([0.0, start[0], start[1], start[2], 0.0])

        A1 = np.eye(5)
        b1 = np.array([0.0, goal[0], goal[1], goal[2], 255])

        p = [[A0, b0], [A1, b1]]

        data_out = np.zeros((self._nb_var - 1, self.nbData))
        MuTmp = np.zeros((self.out.size, self.nbData, self._nb_frames))
        SigmaTmp = np.zeros((self.out.size, self.out.size, self.nbData, self._nb_frames))

        for m in range(self._nb_frames):
            MuTmp[:, :, m] = ((p[m][0][1:, 1:] @ self.MuGMR[:, :, m]).T + p[m][1][1:]).T
            for t in range(self.nbData):
                SigmaTmp[:, :, t, m] = p[m][0][1:, 1:] @ self.SigmaGMR[:, :, t, m] @ p[m][0][1:, 1:].T

        for t in range(self.nbData):
            SigmaP = np.zeros((self.out.size, self.out.size))
            MuP = np.zeros(self.out.size)
            for m in range(self._nb_frames):
                SigmaP = SigmaP + np.linalg.inv(SigmaTmp[:, :, t, m])
                MuP = MuP + np.linalg.inv(SigmaTmp[:, :, t, m]) @ MuTmp[:, t, m]
            Sigma_out = np.linalg.inv(SigmaP)
            data_out[:, t] = Sigma_out @ MuP

        return data_out
