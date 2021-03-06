import numpy as np
import matplotlib.pyplot as plt
import warnings


class SWSolver:
    def __init__(self, grid_x, delta_t, time, method='SW1', to_plot=False, epsilon=0.1):
        self.gamma = 1.4
        self.R = 287.05  # Specific gas constant for air
        self.x_length = 10
        self.n = grid_x
        self.delta_x = self.x_length / self.n
        self.delta_t = delta_t  # [s]
        self.v_x = np.arange(0, self.x_length + self.delta_x, self.delta_x)
        self.to_plot = to_plot
        self.epsilon = epsilon  # 0 <= epsilon <= 0.125
        # Initial condition in region R
        p_r = 0.1
        rho_r = 0.125
        u_r = 0

        # Initial condition in region L
        p_l = 1
        rho_l = 1
        u_l = 0

        self.list_U = []
        m_U_0_l = self.calc_U_i(u_l, rho_l, p_l) * np.ones([1, int(np.ceil((self.n + 1)/2))])
        m_U_0_r = self.calc_U_i(u_r, rho_r, p_r) * np.ones([1, int(np.floor((self.n + 1)/2))])
        m_U_0 = np.hstack((m_U_0_l, m_U_0_r))

        self.list_U.append(self.format_u(m_U_0))
        if method is 'SW1':
            m_U_n = self.calc_U_np1_by_first_order_SW(m_U_0)
        elif method is 'SW2':
            m_U_n = self.calc_U_np1_by_second_order_SW(m_U_0)
        else:
            m_U_n = self.calc_U_TVD2(m_U_0)
        self.list_U.append(self.format_u(m_U_n))
        for i in range(int((time - delta_t) / delta_t)):
            print('t={}'.format(2*delta_t + i*delta_t))
            if method is 'SW1':
                m_U_np1 = self.calc_U_np1_by_first_order_SW(m_U_n)
            elif method is 'SW2':
                m_U_np1 = self.calc_U_np1_by_second_order_SW(m_U_n)
            else:
                m_U_np1 = self.calc_U_TVD2(m_U_n)
            self.list_U.append(self.format_u(m_U_np1))
            m_U_n = m_U_np1
        if to_plot:
            self.plot_u(self.list_U, "bla")

    def format_u(self, m_u):
        v_p = (self.gamma - 1) * (m_u[2, :] - m_u[1, :] ** 2 / (2 * m_u[0, :]))
        v_rho = m_u[0, :]
        v_velocity = m_u[1, :] / v_rho
        v_temp = v_p / (self.R * v_rho)

        c = self.delta_t / self.delta_x
        # print('c: {}'.format(c))

        return np.vstack((v_p, v_rho, v_velocity, v_temp))

    def calc_T(self, v_U):
        m_T = np.eye(3)
        p = self.calc_p(v_U)
        u = v_U[1] / v_U[0]
        a = self.calc_sound_velocity(v_U[0], p)
        alpha = v_U[0] / (np.sqrt(2) * a)
        m_T[0, 1] = alpha
        m_T[0, 2] = alpha
        m_T[1, 0] = u
        m_T[1, 1] = alpha * (u + a)
        m_T[1, 2] = alpha * (u - a)
        m_T[2, 0] = 0.5 * u ** 2
        m_T[2, 1] = alpha * (0.5 * u ** 2 + u*a + a**2 / (self.gamma - 1))
        m_T[2, 2] = alpha * (0.5 * u ** 2 - u*a + a**2 / (self.gamma - 1))
        return m_T

    def calc_T_inverse(self, v_U):
        m_T = np.eye(3)
        p = self.calc_p(v_U)
        u = v_U[1] / v_U[0]
        a = self.calc_sound_velocity(v_U[0], p)
        beta = 1 / (v_U[0] * np.sqrt(2) * a)
        m_T[0, 0] = 1 - ((self.gamma - 1)*u**2) / (2*a**2)
        m_T[0, 1] = ((self.gamma - 1)*u) / (a**2)
        m_T[0, 2] = -(self.gamma - 1) / (a ** 2)
        m_T[1, 0] = beta * (0.5*(self.gamma - 1)*u**2 - u*a)
        m_T[1, 1] = beta * (a - (self.gamma - 1)*u)
        m_T[1, 2] = beta * (self.gamma - 1)
        m_T[2, 0] = beta * (0.5*(self.gamma - 1)*u**2 + u*a)
        m_T[2, 1] = -beta * (a + (self.gamma - 1)*u)
        m_T[2, 2] = beta * (self.gamma - 1)
        return m_T

    def calc_A_p(self, v_U):
        p = self.calc_p(v_U)
        u = v_U[1] / v_U[0]
        a = self.calc_sound_velocity(v_U[0], p)
        m_D = np.zeros((3, 3))
        m_D[0, 0] = u
        m_D[1, 1] = u + a
        if u > a:
            m_D[2, 2] = u - a
        m_T = self.calc_T(v_U)
        m_T_inverse = self.calc_T_inverse(v_U)
        return np.matmul(np.matmul(m_T, m_D), m_T_inverse)

    def calc_A_m(self, v_U):
        p = self.calc_p(v_U)
        u = v_U[1] / v_U[0]
        a = self.calc_sound_velocity(v_U[0], p)
        m_D = np.zeros((3, 3))
        if u >= a:
            return m_D
        m_D[2, 2] = u - a
        m_T = self.calc_T(v_U)
        m_T_inverse = self.calc_T_inverse(v_U)
        return np.matmul(np.matmul(m_T, m_D), m_T_inverse)

    def calc_E_m(self, v_U, i):
        v_U = np.array([v_U]).T
        p = self.calc_p(v_U)
        u = v_U[1] / v_U[0]
        a = self.calc_sound_velocity(v_U[0], p)
        if u >= a:
            return np.zeros(3)
        return np.matmul(self.calc_A_m(v_U), v_U).reshape(3)

    def calc_E_p(self, v_U, i):
        v_U = np.array([v_U]).T
        p = self.calc_p(v_U)
        u = v_U[1] / v_U[0]
        a = self.calc_sound_velocity(v_U[0], p)
        if u >= a:
            return self.calc_E(v_U, i)
        return np.matmul(self.calc_A_p(v_U), v_U).reshape(3)

    def calc_E(self, v_U, i):
        v_E = np.zeros(3)
        v_E[0] = v_U[1]
        p = self.calc_p(v_U)
        v_E[1] = (v_U[1] ** 2 / v_U[0]) + p
        v_E[2] = ((v_U[2] + p) * v_U[1]) / v_U[0]
        return v_E

    def calc_p(self, v_U):
        return (self.gamma - 1) * (v_U[2] - v_U[1] ** 2 / (2 * v_U[0]))

    def calc_s(self, i):
        return 0.13 + 0.032 * np.tanh(2.666*i*self.delta_x - 4)

    def calc_U_np1_by_first_order_SW(self, m_U_n):
        m_U_np1 = np.copy(m_U_n)
        for i in range(1, self.n):
            v_E_p_i = self.calc_E_p(m_U_n[:, i], i)
            v_E_p_im1 = self.calc_E_p(m_U_n[:, i - 1], i - 1)
            v_E_m_i = self.calc_E_m(m_U_n[:, i], i)
            v_E_m_ip1 = self.calc_E_m(m_U_n[:, i + 1], i + 1)
            v_H = np.array([0, self.calc_p(m_U_n[:, i]), 0])
            m_U_np1[:, i] = m_U_n[:, i] - (self.delta_t / self.delta_x) * (v_E_p_i - v_E_p_im1 + v_E_m_ip1 - v_E_m_i)

        m_U_np1[:, self.n] = m_U_np1[:, self.n - 1]
        return m_U_np1

    def calc_U_np1_by_second_order_SW(self, m_U_n):
        m_U_np1 = np.copy(m_U_n)
        for i in range(2, self.n-1):
            v_E_p_i = self.calc_E_p(m_U_n[:, i], i)
            v_E_p_im1 = self.calc_E_p(m_U_n[:, i - 1], i - 1)
            v_E_p_im2 = self.calc_E_p(m_U_n[:, i - 2], i - 2)
            v_E_m_i = self.calc_E_m(m_U_n[:, i], i)
            v_E_m_ip1 = self.calc_E_m(m_U_n[:, i + 1], i + 1)
            v_E_m_ip2 = self.calc_E_m(m_U_n[:, i + 2], i + 2)
            v_H = np.array([0, self.calc_p(m_U_n[:, i]), 0])
            m_U_np1[:, i] = m_U_n[:, i] - 0.5*(self.delta_t / self.delta_x) * (3 * v_E_p_i - 4 * v_E_p_im1 + v_E_p_im2 - v_E_m_ip2 + 4 * v_E_m_ip1 - 3*v_E_m_i)

        m_U_np1[:, self.n - 1] = m_U_np1[:, self.n - 2]
        m_U_np1[:, self.n] = m_U_np1[:, self.n - 2]
        return m_U_np1

    def calc_U_TVD(self, m_U_n):
        '''
        Use the TVD equations to calculate U
        '''

        def calc_u_p_half(U_next, U, i):
            if i == self.n:
                'First iteration'
                return U
            else:
                return U_next - U

        def calc_u_m_half(U, U_prev, i):
            if i == 0:
                'First iteration'
                return U
            else:
                return U - U_prev

        def calc_alpha_p_half(U_next, U, E_next, E, delta_u):
            alpha_p_half = U * (delta_u == 0)
            tmp = (E_next - E) / (U_next - U) * (delta_u != 0)
            alpha_p_half += np.nan_to_num(tmp)  # Replace nan to zero (When dividing  by zero)
            return alpha_p_half

        def calc_alpha_m_half(U, U_prev, E, E_prev, delta_u):
            alpha_m_half = U * (delta_u == 0)
            tmp = (E - E_prev) / (U - U_prev) * (delta_u != 0)
            alpha_m_half += np.nan_to_num(tmp)  # Replace nan to zero (When dividing  by zero)
            return alpha_m_half

        m_U_np1 = np.copy(m_U_n)
        for i in range(1, self.n):
            'Calculate deltaU+0.5, deltaU-0.5'
            u_p_half = calc_u_p_half(m_U_n[:, i + 1], m_U_n[:, i], i)
            u_m_half = calc_u_m_half(m_U_n[:, i], m_U_n[:, i - 1], i)

            v_Ep1 = self.calc_E(np.array([m_U_n[:, i + 1]]).T, i)
            v_E = self.calc_E(np.array([m_U_n[:, i]]).T, i)
            v_Em1 = self.calc_E(np.array([m_U_n[:, i - 1]]).T, i)

            alpha_p_half = calc_alpha_p_half(m_U_n[:, i + 1], m_U_n[:, i], v_Ep1, v_E, u_p_half)
            alpha_m_half = calc_alpha_m_half(m_U_n[:, i], m_U_n[:, i - 1], v_E, v_Em1, u_m_half)

            phi_p_half = np.abs(alpha_p_half) * u_p_half
            phi_m_half = np.abs(alpha_m_half) * u_m_half

            m_U_star = m_U_n[:, i] - (self.delta_t / 2 * self.delta_x) * (v_Ep1 - v_Em1)
            m_U_np1[:, i] = m_U_star + (self.delta_t / 2 * self.delta_x) * (phi_p_half - phi_m_half)

        m_U_np1[:, self.n] = m_U_np1[:, self.n - 1]
        return m_U_np1

    def calc_U_TVD2(self, m_U_n):
        '''
        Use the secound-order TVD equations to calculate U
        '''

        def calc_alpha(U_next, U):
            p_next = self.calc_p(U_next)
            a_next = self.calc_sound_velocity(U_next[0], p_next)
            v_next = U_next[1] / U_next[0]

            p = self.calc_p(U)
            a = self.calc_sound_velocity(U[0], p)
            v = U[1] / U[0]
            return 0.5 * np.array([v_next + v, v_next + a_next + v + a, v_next - a_next + v - a])

        def calc_psi(y, epsilon):
            psi = np.abs(y) * (np.abs(y) >= epsilon)
            psi += ((y**2 + epsilon**2) / (2 * epsilon)) * (np.abs(y) < epsilon)
            return psi

        def calc_G(v_U_ip1, v_U_i, v_U_im1):
            m_X_p_half_inv = (self.calc_T_inverse(v_U_ip1) + self.calc_T_inverse(v_U_i)) * 0.5
            m_X_m_half_inv = (self.calc_T_inverse(v_U_i) + self.calc_T_inverse(v_U_im1)) * 0.5

            delta_p_half = (m_X_p_half_inv @ (v_U_ip1 - v_U_i)).T
            delta_m_half = (m_X_m_half_inv @ (v_U_i - v_U_im1)).T

            v_G = np.zeros(v_U_i.shape).T
            v_G += delta_p_half * np.logical_and(np.abs(delta_p_half) < np.abs(delta_m_half), delta_p_half * delta_m_half > 0.0)
            v_G += delta_m_half * np.logical_and(np.abs(delta_m_half) < np.abs(delta_p_half), delta_p_half * delta_m_half > 0.0)
            return v_G

        def calc_beta(sigma, delta_half, v_G_ip1, v_G_i):
            v_beta = sigma * (v_G_ip1 - v_G_i) / delta_half
            return np.nan_to_num(v_beta)  # Convert nan value (due to zero divisiontion) to zero

        m_U_np1 = np.copy(m_U_n)
        for i in range(2, self.n - 2):
            v_Ep1 = self.calc_E(np.array([m_U_n[:, i + 1]]).T, i)
            v_E = self.calc_E(np.array([m_U_n[:, i]]).T, i)
            v_Em1 = self.calc_E(np.array([m_U_n[:, i - 1]]).T, i)

            alpha_p_half = calc_alpha(m_U_n[:, i + 1], m_U_n[:, i])
            alpha_m_half = calc_alpha(m_U_n[:, i], m_U_n[:, i - 1])

            sigma_p_half = 0.5 * calc_psi(alpha_p_half, self.epsilon) + (self.delta_t / self.delta_x) * alpha_p_half **2
            sigma_m_half = 0.5 * calc_psi(alpha_m_half, self.epsilon) + (self.delta_t / self.delta_x) * alpha_m_half **2

            m_X_p_half_inv = (self.calc_T_inverse(np.array([m_U_n[:, i + 1]]).T) + self.calc_T_inverse(np.array([m_U_n[:, i]]).T)) * 0.5
            m_X_m_half_inv = (self.calc_T_inverse(np.array([m_U_n[:, i]]).T) + self.calc_T_inverse(np.array([m_U_n[:, i - 1]]).T)) * 0.5

            delta_p_half = (m_X_p_half_inv @ (np.array([m_U_n[:, i + 1]]).T - np.array([m_U_n[:, i]]).T)).T
            delta_m_half = (m_X_m_half_inv @ (np.array([m_U_n[:, i]]).T - np.array([m_U_n[:, i - 1]]).T)).T

            v_G_ip1 = calc_G(np.array([m_U_n[:, i + 2]]).T, np.array([m_U_n[:, i + 1]]).T, np.array([m_U_n[:, i]]).T)
            v_G_i = calc_G(np.array([m_U_n[:, i + 1]]).T, np.array([m_U_n[:, i]]).T, np.array([m_U_n[:, i - 1]]).T)
            v_G_im1 = calc_G(np.array([m_U_n[:, i]]).T, np.array([m_U_n[:, i - 1]]).T, np.array([m_U_n[:, i - 2]]).T)

            beta_p_half = calc_beta(sigma_p_half, delta_p_half, v_G_ip1, v_G_i)
            beta_m_half = calc_beta(sigma_m_half, delta_m_half, v_G_i, v_G_im1)

            phi_p_half = sigma_p_half * (v_G_ip1 + v_G_i) - calc_psi(alpha_p_half + beta_p_half, self.epsilon) * delta_p_half
            phi_m_half = sigma_m_half * (v_G_i + v_G_im1) - calc_psi(alpha_m_half + beta_m_half, self.epsilon) * delta_m_half

            v_R_p_half = 0.5 * (v_Ep1 + v_E + (0.5*(self.calc_T(np.array([m_U_n[:, i + 1]]).T) + self.calc_T(np.array([m_U_n[:, i]]).T)) @ phi_p_half.T).T)
            v_R_m_half = 0.5 * (v_Em1 + v_E + (0.5*(self.calc_T(np.array([m_U_n[:, i]]).T) + self.calc_T(np.array([m_U_n[:, i - 1]]).T)) @ phi_m_half.T).T)

            m_U_np1[:, i] = m_U_n[:, i] - (self.delta_t / self.delta_x) * (v_R_p_half - v_R_m_half)

        m_U_np1[:, self.n] = m_U_np1[:, self.n - 1]
        return m_U_np1

    def convert_U(self, m_U):
        m_F = np.zeros((3, 1))
        for v_U in m_U.T:
            p = self.calc_p(v_U)
            M = v_U[1] / (v_U[0] * self.calc_sound_velocity(v_U[0], p))
            v_F = np.array([[v_U[0], M, p]])
            m_F = np.hstack((m_F, v_F.T))
        return m_F[:, 1:]

    def calc_sound_velocity(self, rho, p):
        return np.sqrt(self.gamma * p / rho)

    def calc_U_i(self, u, rho, p):
        v_U = np.zeros((3, 1))
        v_U[0] = rho
        v_U[1] = rho * u
        a = self.calc_sound_velocity(rho, p)
        v_U[2] = ((rho * a ** 2) / (self.gamma * (self.gamma - 1))) + (v_U[1] ** 2 / (2 * rho))
        return v_U

    def build_legend_labels(self, arr):
        labels = []
        for i in arr:
            labels.append('n={}'.format(i))
        return labels

    def plot(self, m_U_n, labels):
        warnings.filterwarnings("ignore")
        m_U_n = self.convert_U(m_U_n)
        # plt.subplot(131)
        plt.figure(1)
        plt.plot(self.v_x[:-1], m_U_n[0, :], marker='o')
        plt.ylabel('$rho$')
        plt.xlabel('x')
        plt.grid(linestyle='dashed')
        plt.legend(labels)
        # plt.subplot(132)
        plt.figure(2)
        plt.plot(self.v_x[:-1], m_U_n[1, :] / m_U_n[0, :], marker='o')
        plt.ylabel('V')
        plt.xlabel('x')
        plt.grid(linestyle='dashed')
        plt.legend(labels)
        # plt.subplot(133)
        plt.figure(3)
        plt.plot(self.v_x[:-1], m_U_n[2, :], marker='o')
        plt.ylabel('p')
        plt.xlabel('x')
        plt.grid(linestyle='dashed')
        plt.legend(labels)
        # plt.savefig('figure_2.png', dpi=600)
        # plt.show()

    def plot_u(self, list_U, fig_name, save_fig=False):
        labels = []
        slice = int(len(list_U)/9)
        # slice = 1
        for i, u_n in enumerate(list_U[::slice]):
            labels.append('n={}'.format(i*slice))
            self.plot(u_n, labels)
        # plt.legend(labels)
        if save_fig:
            plt.savefig('figure_{}.png'.format(fig_name), dpi=600)
        if self.to_plot:
            plt.show()


if __name__ == "__main__":
    sw = SWSolver()