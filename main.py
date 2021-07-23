import numpy as np
import matplotlib.pyplot as plt
from RiemannSolver import RiemannSolver
from SWSolver import SWSolver


def plot(m_u, rs, time, grid_x, numerical_method='Steger-Warming'):
    x_length = 10
    delta_x = x_length / grid_x
    v_x = np.arange(0, x_length + delta_x, delta_x)

    plt.title('Time={}'.format(time))
    plt.subplot(4, 1, 1)
    plt.plot(v_x[:-1], m_u[0, :], markersize=4, label='{}'.format(numerical_method))
    plt.plot(rs.v_x, rs.m_u[0, :], markersize=4, label='Analytic solution')
    plt.legend()
    plt.ylabel('$Pressure$')
    plt.xlabel('x')
    plt.grid(linestyle='dashed')

    plt.subplot(4, 1, 2)
    plt.plot(v_x[:-1], m_u[1, :], markersize=4, label='{}'.format(numerical_method))
    plt.plot(rs.v_x, rs.m_u[1, :], markersize=4, label='Analytic solution')
    plt.legend()
    plt.ylabel('Density')
    plt.xlabel('x')
    plt.grid(linestyle='dashed')

    plt.subplot(4, 1, 3)
    plt.plot(v_x[:-1], m_u[2, :], markersize=4, label='{}'.format(numerical_method))
    plt.plot(rs.v_x, rs.m_u[2, :], markersize=4, label='Analytic solution')
    plt.legend()
    plt.ylabel('Velocity')
    plt.xlabel('x')
    plt.grid(linestyle='dashed')

    plt.subplot(4, 1, 4)
    plt.plot(v_x[:-1], m_u[3, :], markersize=4, label='{}'.format(numerical_method))
    plt.plot(rs.v_x, rs.m_u[3, :], markersize=4, label='Analytic solution')
    plt.legend()
    plt.ylabel('Temperature')
    plt.xlabel('x')
    plt.grid(linestyle='dashed')
    plt.show()


if __name__ == "__main__":
    delta_t = 0.01
    time_steps = 150
    grid_x = 81
    time = 1.4

    rs = RiemannSolver(grid_x, time)
    sw = SWSolver(grid_x, delta_t, time_steps)

    plot(sw.list_U[int(time / delta_t)], rs, time, grid_x)

    print("main")
