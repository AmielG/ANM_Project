import numpy as np
import matplotlib.pyplot as plt


class RiemannSolver:

    def __init__(self, v_t):
        self.p_l = 1
        self.rho_l = 1
        self.p_r = 0.1
        self.rho_r = 0.125
        self.u_r = 0
        self.u_l = 0
        self.gamma = 1.4
        self.tube_length = 10
        self.R = 287.05
        self.v_t = v_t

        self.temp_l = self.calc_ideal_gas_temp(self.p_l, self.rho_l)
        self.a_l = self.calc_speed_of_sound(self.temp_l)
        self.temp_r = self.calc_ideal_gas_temp(self.p_r, self.rho_r)
        self.a_r = self.calc_speed_of_sound(self.temp_r)
        self.p_1 = self.calc_p_1(self.p_r * 2)

    def calc_speed_of_sound(self, temp):
        return np.sqrt(self.gamma * self.R * temp)

    def calc_ideal_gas_temp(self, p, rho):
        return p / (self.R * rho)

    def calc_p_1(self, p_1_guss):
        delta = 1
        p_1 = p_1_guss
        while delta > 0.001:
            p_1_new = self.p_l * (1 - (self.gamma - 1) * (self.a_r / self.a_l) * ((p_1 / self.p_r) - 1) / (
                        4 * self.gamma ** 2 + 2 * self.gamma * (self.gamma + 1) * ((p_1 / self.p_r) - 1)))
            delta = abs(p_1_new - p_1)
            p_1 = p_1_new
        return p_1

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
