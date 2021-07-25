import numpy as np
import matplotlib.pyplot as plt
from RiemannSolver import RiemannSolver
from SWSolver import SWSolver


def plot(v_x, m_u, time, method='Steger-Warming', marker='-', figure_id=0):
    plt.figure(figure_id)
    plt.subplot(2, 2, 1)
    plt.plot(v_x, m_u[0, :], marker, markersize=2, label='{}'.format(method))
    plt.legend()
    plt.ylabel('$Pressure$')
    plt.xlabel('x')
    plt.title('t={}'.format(time))
    plt.grid(linestyle='dashed')

    plt.subplot(2, 2, 2)
    plt.plot(v_x, m_u[1, :], marker, markersize=2, label='{}'.format(method))
    plt.legend()
    plt.ylabel('Density')
    plt.xlabel('x')
    plt.grid(linestyle='dashed')

    plt.subplot(2, 2, 3)
    plt.plot(v_x, m_u[2, :], marker, markersize=2, label='{}'.format(method))
    plt.legend()
    plt.ylabel('Velocity')
    plt.xlabel('x')
    plt.grid(linestyle='dashed')

    plt.subplot(2, 2, 4)
    plt.plot(v_x, m_u[3, :], marker, markersize=2, label='{}'.format(method))
    plt.legend()
    plt.ylabel('Temperature')
    plt.xlabel('x')
    plt.grid(linestyle='dashed')
    # plt.show()


if __name__ == "__main__":
    time = 0.5
    delta_t = 0.001
    grid_x = 1000

    rs = RiemannSolver(grid_x, time)
    plot(rs.v_x, rs.m_u, time, method='Analytic solution', marker='-', figure_id=1)
    sw1 = SWSolver(1000, 0.001, time, first_order=True)
    plot(sw1.v_x, sw1.list_U[-1], time, method='S-W order 1 - c=0.1', marker='-o', figure_id=1)
    sw1 = SWSolver(500, 0.0001, time, first_order=True)
    plot(sw1.v_x, sw1.list_U[-1], time, method='S-W order 1 - c=0.005', marker='-o', figure_id=1)
    sw1 = SWSolver(100, 0.0001, time, first_order=True)
    plot(sw1.v_x, sw1.list_U[-1], time, method='S-W order 1 - c=0.001', marker='-o', figure_id=1)

    plt.show()
    print("main")
