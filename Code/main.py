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
    sw1 = SWSolver(100, 0.01, time, method='SW1')
    plot(sw1.v_x, sw1.list_U[-1], time, method='S-W order 1', marker='-o', figure_id=1)
    sw2 = SWSolver(100, 0.01, time, method='SW2')
    plot(sw2.v_x, sw2.list_U[-1], time, method='S-W order 2', marker='-o', figure_id=1)
    tvd = SWSolver(100, 0.01, time, method='TVD')
    plot(tvd.v_x, tvd.list_U[-1], time, method='TVD', marker='-o', figure_id=1)

    plt.show()
    print("main")
