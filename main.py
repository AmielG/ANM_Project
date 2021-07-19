import numpy as np

from RiemannSolver import RiemannSolver

if __name__ == "__main__":
    v_x = np.linspace(-5, 5, 10)
    t = 0
    r = RiemannSolver(v_x, 0.2)

    print(r.calc_p_1(0.5))

    print("main")
