import numpy as np
import matplotlib.pyplot as plt
from mpmath.libmp.libelefun import machin
from scipy import optimize


class RiemannSolver:

    def __init__(self, grid_x, t, x_0=5):
        """
        Solve the exact solution for the Riemannn problem for a given time.
        The solution is accessible in the internal variables: self.v_p, self.v_rho, self.v_velocity, self.v_temp
        :param v_x: A row vector of the x axis grid.
        :param t: Scalar. The desire solution time.
        :param x_0: The position of the diaphragm.
        """
        self.gamma = 1.4
        self.tube_length = 10
        self.R = 287.05  # Specific gas constant for air
        self.delta_x = self.tube_length / grid_x
        self.v_x = np.arange(0, self.tube_length + self.delta_x, self.delta_x)
        self.x_0 = x_0
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

        self.mach_s = self.calc_mach_s(1)

        # Flow properties in region 1
        self.p_1 = self.calc_p_1()
        self.temp_1 = self.calc_temp_1()
        self.rho_1 = self.calc_rho_1()
        self.velocity_1 = self.calc_velocity_1()

        # Flow properties in region 2
        self.p_2 = self.p_1
        self.velocity_2 = self.velocity_1
        self.temp_2 = self.calc_temp_2()
        self.rho_2 = self.calc_rho_2()
        self.a_2 = self.calc_a_2()

        # Split x to regions
        self.region_l, self.region_3, self.region_2, self.region_1, self.region_r = self.split_to_regions()
        self.v_x = np.hstack((self.region_l, self.region_3, self.region_2, self.region_1, self.region_r))

        # Flow properties in region 3
        self.velocity_3 = self.calc_velocity_3(self.region_3)
        self.p_3 = self.calc_p_3()
        self.temp_3 = self.calc_temp_3()
        self.rho_3 = self.calc_rho_3()

        # Smooth the transition between region l and region 3
        self.velocity_2 = self.velocity_1 = self.velocity_3[-1]
        self.p_2 = self.p_1 = self.p_3[-1]
        self.temp_2 = self.temp_3[-1]
        self.rho_2 = self.rho_3[-1]

        self.v_p = self.build_p()
        self.v_rho = self.build_rho()
        self.v_velocity = self.build_velocity()
        self.v_temp = self.build_temp()

        self.m_u = np.vstack((self.v_p, self.v_rho, self.v_velocity, self.v_temp))

    def plot(self, to_show=True):
        v_x = np.hstack((self.region_l, self.region_3, self.region_2, self.region_1, self.region_r))
        plt.subplot(4, 1, 1)
        plt.plot(v_x, self.v_p, markersize=4, label='time={}'.format(self.t))
        plt.legend()
        plt.ylabel('$Pressure$')
        plt.xlabel('x')
        plt.grid(linestyle='dashed')

        plt.subplot(4, 1, 2)
        plt.plot(v_x, self.v_rho, markersize=4, label='time={}'.format(self.t))
        plt.legend()
        plt.ylabel('Density')
        plt.xlabel('x')
        plt.grid(linestyle='dashed')

        plt.subplot(4, 1, 3)
        plt.plot(v_x, self.v_velocity, markersize=4, label='time={}'.format(self.t))
        plt.legend()
        plt.ylabel('Velocity')
        plt.xlabel('x')
        plt.grid(linestyle='dashed')

        plt.subplot(4, 1, 4)
        plt.plot(v_x, self.v_temp, markersize=4, label='time={}'.format(self.t))
        plt.legend()
        plt.ylabel('Temperature')
        plt.xlabel('x')
        plt.grid(linestyle='dashed')
        if to_show:
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
        x_l = self.x_0-self.a_l * self.t
        region_l = self.v_x[self.v_x <= x_l]
        region_l = np.hstack((region_l, x_l))
        x_3 = self.x_0 + (self.velocity_2 - self.a_2) * self.t
        region_3 = self.v_x[np.logical_and(x_l <= self.v_x, self.v_x <= x_3)]
        region_3 = np.hstack((x_l, region_3, x_3))
        x_2 = self.x_0 + self.velocity_2 * self.t
        region_2 = self.v_x[np.logical_and(x_3 <= self.v_x, self.v_x <= x_2)]
        region_2 = np.hstack((x_3, region_2, x_2))
        x_1 = self.x_0 + self.mach_s * self.t
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

    def calc_mach_s(self, mach_s_guess):
        return optimize.newton(self.implicit_function_mach_s, mach_s_guess)

    def calc_p_1(self):
        a = (2*self.gamma*self.mach_s**2)/(self.gamma+1)
        b = ((self.gamma - 1) / (self.gamma + 1))
        return self.p_r * (a-b)

    def implicit_function_mach_s(self, mach_s):
        a = self.p_r / self.p_l
        b = ((self.gamma + 1)/(self.gamma - 1))
        f = (1 / mach_s) - mach_s + self.a_l * b * (1 - (a*((2*self.gamma*mach_s**2)/(self.gamma+1)-1/b))**((self.gamma - 1)/(2*self.gamma)))
        return f

    def calc_temp_1(self):
        a = self.p_1 / self.p_r
        b = (self.gamma + 1) / (self.gamma - 1)
        return self.temp_r * a * ((a + b) / (1 + a * b))

    def calc_rho_1(self):
        a = 2 / ((self.gamma + 1) * self.mach_s ** 2)
        b = ((self.gamma - 1) / (self.gamma + 1))
        return self.rho_r / (a + b)

    def calc_velocity_1(self):
        a = self.p_1 / self.p_r
        b = (self.gamma - 1) / (self.gamma + 1)
        return (self.a_r / self.gamma) * (a -1) * ((2*self.gamma/(self.gamma+1))/(b + a)) ** 0.5

    def calc_temp_2(self):
        a = self.p_1 / self.p_l
        return self.temp_l * a ** ((self.gamma - 1) / self.gamma)

    def calc_rho_2(self):
        a = self.p_1 / self.p_l
        return self.rho_l * a ** (1 / self.gamma)

    def calc_a_2(self):
        a = self.p_l / self.p_2
        return self.a_l * a ** -((self.gamma - 1) / (2 * self.gamma))

    def calc_velocity_3(self, v_x):
        return (2 / (self.gamma + 1)) * (self.a_l + (v_x - self.x_0) / self.t)

    def calc_temp_3(self):
        a = 1 - ((self.gamma - 1) / 2) * (self.velocity_3 / self.a_l)
        return self.temp_l * a ** 2

    def calc_rho_3(self):
        a = 1 - ((self.gamma - 1) / 2) * (self.velocity_3 / self.a_l)
        return self.rho_l * a ** (2 / (self.gamma - 1))

    def calc_p_3(self):
        a = 1 - ((self.gamma - 1) / 2) * (self.velocity_3 / self.a_l)
        return self.p_l * a ** ((2 * self.gamma) / (self.gamma - 1))

