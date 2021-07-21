import numpy as np
import matplotlib.pyplot as plt
from RiemannSolver import RiemannSolver
from SWSolver import SWSolver

if __name__ == "__main__":
    v_x = np.linspace(0, 10, 80)
    r = RiemannSolver(v_x, 1.5)
    sw = SWSolver()
    r.plot(to_show=False)
    plt.show()

    print("main")
