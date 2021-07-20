import numpy as np

from RiemannSolver import RiemannSolver

if __name__ == "__main__":
    v_x = np.linspace(-5, 5, 20)
    r = RiemannSolver(v_x, 1.3)


    print("main")
