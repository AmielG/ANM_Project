import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


class RiemannSolver:

    def __init__(self, v_x, t):
        self.gamma = 1.4
        self.tube_length = 10
        self.R = 287.05
        self.v_x = v_x
        self.t = t

        # Flow properties in region R
        self.p_r = 0.1
        self.rho_r = 0.125
        self.u_r = 0
        self.temp_r = self.calc_ideal_gas_temp(self.p_r, self.rho_r)
        self.a_r = self.calc_speed_of_sound(self.temp_r)

        # Flow properties in region L
        self.p_l = 1
        self.rho_l = 1
        self.u_l = 0
        self.temp_l = self.calc_ideal_gas_temp(self.p_l, self.rho_l)
        self.a_l = self.calc_speed_of_sound(self.temp_l)

        # Flow properties in region 1
        self.p_1 = self.calc_p_1(self.p_r * 2)
        self.temp_1 = self.calc_temp_1()
        self.rho_1 = self.calc_rho_1()
        self.velocity_1 = self.calc_velocity_1()

        self.velocity_s = self.calc_velocity_s()
        self.mach_s = self.calc_mach_s()

        # Flow properties in region 2
        self.p_2 = self.p_1
        self.velocity_2 = self.velocity_1
        self.temp_2 = self.calc_temp_2()
        self.rho_2 = self.calc_rho_2()
        self.a_2 = self.calc_a_2()

        # Split x to regions
        self.region_l, self.region_3, self.region_2, self.region_1, self.region_r = self.split_to_regions()

        # Flow properties in region 3
        self.velocity_3 = self.calc_velocity_3(self.region_3)
        self.p_3 = self.calc_p_3()
        self.temp_3 = self.calc_temp_3()
        self.rho_3 = self.calc_rho_3()

        self.v_p = self.build_p()
        self.v_rho = self.build_rho()
        self.v_velocity = self.build_velocity()
        self.v_temp = self.build_temp()

    def plot(self):
        v_x = np.hstack((self.region_l, self.region_3, self.region_2, self.region_1, self.region_r))
        plt.subplot(4, 1, 1)
        plt.plot(v_x, self.v_p, markersize=4)
        plt.legend(['time={}'.format(self.t)])
        plt.ylabel('$Pressure$')
        plt.xlabel('time [s]')
        plt.grid(linestyle='dashed')

        plt.subplot(4, 1, 2)
        plt.plot(v_x, self.v_rho, markersize=4)
        plt.legend(['time={}'.format(self.t)])
        plt.ylabel('Density')
        plt.xlabel('time [s]')
        plt.grid(linestyle='dashed')

        plt.subplot(4, 1, 3)
        plt.plot(v_x, self.v_velocity, markersize=4)
        plt.legend(['time={}'.format(self.t)])
        plt.ylabel('Velocity')
        plt.xlabel('time [s]')
        plt.grid(linestyle='dashed')

        plt.subplot(4, 1, 4)
        plt.plot(v_x, self.v_temp, markersize=4)
        plt.legend(['time={}'.format(self.t)])
        plt.ylabel('Temperature')
        plt.xlabel('time [s]')
        plt.grid(linestyle='dashed')
        plt.show()

    def build_rho(self):
        v_rho_l = np.ones(self.region_l.shape) * self.rho_l
        v_rho_2 = np.ones(self.region_2.shape) * self.rho_2
        v_rho_1 = np.ones(self.region_1.shape) * self.rho_1
        v_rho_r = np.ones(self.region_r.shape) * self.rho_r
        return np.hstack((v_rho_l, self.rho_3, v_rho_2, v_rho_1, v_rho_r))

    def build_velocity(self):
        v_velocity_l = np.zeros(self.region_l.shape)
        v_velocity_2 = np.ones(self.region_2.shape) * self.velocity_2
        v_velocity_1 = np.ones(self.region_1.shape) * self.velocity_1
        v_velocity_r = np.zeros(self.region_r.shape)
        return np.hstack((v_velocity_l, self.velocity_3, v_velocity_2, v_velocity_1, v_velocity_r))

    def build_p(self):
        v_p_l = np.ones(self.region_l.shape) * self.p_l
        v_p_2 = np.ones(self.region_2.shape) * self.p_2
        v_p_1 = np.ones(self.region_1.shape) * self.p_1
        v_p_r = np.ones(self.region_r.shape) * self.p_r
        return np.hstack((v_p_l, self.p_3, v_p_2, v_p_1, v_p_r))

    def build_temp(self):
        v_temp_l = np.ones(self.region_l.shape) * self.temp_l
        v_temp_2 = np.ones(self.region_2.shape) * self.temp_2
        v_temp_1 = np.ones(self.region_1.shape) * self.temp_1
        v_temp_r = np.ones(self.region_r.shape) * self.temp_r
        return np.hstack((v_temp_l, self.temp_3, v_temp_2, v_temp_1, v_temp_r))

    def split_to_regions(self):
        x_l = -self.a_l * self.t
        region_l = self.v_x[self.v_x <= x_l]
        region_l = np.hstack((region_l, x_l))
        x_3 = (self.velocity_2 - self.a_2) * self.t
        region_3 = self.v_x[np.logical_and(x_l <= self.v_x, self.v_x <= x_3)]
        region_3 = np.hstack((x_l, region_3, x_3))
        x_2 = self.velocity_2 * self.t
        region_2 = self.v_x[np.logical_and(x_3 <= self.v_x, self.v_x <= x_2)]
        region_2 = np.hstack((x_3, region_2, x_2))
        x_1 = self.mach_s * self.t
        region_1 = self.v_x[np.logical_and(x_2 <= self.v_x, self.v_x <= x_1)]
        region_1 = np.hstack((x_2, region_1, x_1))
        region_r = self.v_x[x_1 <= self.v_x]
        region_r = np.hstack((x_1, region_r))
        return region_l, region_3, region_2, region_1, region_r

    def calc_speed_of_sound(self, temp):
        return np.sqrt(self.gamma * self.R * temp)

    def calc_ideal_gas_temp(self, p, rho):
        return p / (self.R * rho)

    def calc_velocity_s(self):
        return self.a_r * (1 + ((self.gamma + 1)/(2 * self.gamma)) * ((self.p_1 / self.p_r) - 1)) ** 0.5

    def calc_mach_s(self):
        return self.velocity_s / self.a_r

    def calc_p_1(self, p_1_guss):
        return optimize.newton(self.implicit_function_p_1, p_1_guss)

    def implicit_function_p_1(self, p_1):
        return p_1 - self.p_l * (1 - (self.gamma - 1) * (self.a_r / self.a_l) * ((p_1 / self.p_r) - 1) / (
                4 * self.gamma ** 2 + 2 * self.gamma * (self.gamma + 1) * ((p_1 / self.p_r) - 1)))

    def calc_temp_1(self):
        a = self.p_1 / self.p_r
        b = (self.gamma + 1) / (self.gamma - 1)
        return self.temp_r * a * ((a + b) / (1 + a * b))

    def calc_rho_1(self):
        a = self.p_1 / self.p_r
        b = (self.gamma + 1) / (self.gamma - 1)
        return self.rho_r * ((1 + a * b) / (a + b))

    def calc_velocity_1(self):
        a = self.p_1 / self.p_r
        b = (self.gamma - 1) / (self.gamma + 1)
        return (self.a_r / self.gamma) * (a -1) * ((2*self.gamma/(self.gamma+1))/(b + a)) ** 0.5

    def calc_temp_2(self):
        a = self.p_1 / self.p_l
        return self.temp_l * a ** ((self.gamma - 1) / self.gamma)

    def calc_rho_2(self):
        a = self.p_1 / self.p_l
        return self.temp_l * a ** (1 / self.gamma)

    def calc_a_2(self):
        a = self.p_l / self.p_2
        return self.a_l * a ** -((self.gamma - 1) / (2 * self.gamma))

    def calc_velocity_3(self, v_x):
        return (2 / (self.gamma + 1)) * (self.a_l + v_x / self.t)

    def calc_temp_3(self):
        a = 1 - ((self.gamma - 1) / 2) * (self.velocity_3 / self.a_l)
        return self.temp_l * a ** 2

    def calc_rho_3(self):
        a = 1 - ((self.gamma - 1) / 2) * (self.velocity_3 / self.a_l)
        return self.rho_l * a ** (2 / (self.gamma - 1))

    def calc_p_3(self):
        a = 1 - ((self.gamma - 1) / 2) * (self.velocity_3 / self.a_l)
        return self.p_l * a ** ((2 * self.gamma) / (self.gamma - 1))

