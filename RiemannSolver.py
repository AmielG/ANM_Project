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

        # Flow properties in region 2
        self.p_2 = self.p_1
        self.velocity_2 = self.velocity_1
        self.temp_2 = self.calc_temp_2()
        self.rho_2 = self.calc_rho_2()

        # Flow properties in region 3
        self.velocity_3 = self.calc_velocity_3()
        self.p_3 = self.calc_p_3()
        self.temp_3 = self.calc_temp_3()
        self.rho_3 = self.calc_rho_3()



    def calc_speed_of_sound(self, temp):
        return np.sqrt(self.gamma * self.R * temp)

    def calc_ideal_gas_temp(self, p, rho):
        return p / (self.R * rho)

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

    def calc_velocity_3(self):
        return (2 / (self.gamma + 1)) * (self.a_l + self.v_x / self.t)

    def calc_temp_3(self):
        a = 1 - ((self.gamma - 1) / 2) * (self.velocity_3 / self.a_l)
        return self.temp_l * a ** 2

    def calc_rho_3(self):
        a = 1 - ((self.gamma - 1) / 2) * (self.velocity_3 / self.a_l)
        return self.rho_l * a ** (2 / (self.gamma - 1))

    def calc_p_3(self):
        a = 1 - ((self.gamma - 1) / 2) * (self.velocity_3 / self.a_l)
        return self.p_l * a ** ((2 * self.gamma) / (self.gamma - 1))

