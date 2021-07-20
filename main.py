import numpy as np
import matplotlib.pyplot as plt
from RiemannSolver import RiemannSolver

if __name__ == "__main__":
    v_x = np.linspace(-5, 5, 20)
    r = RiemannSolver(v_x, 1)
    r.plot()

    print("main")
