import numpy as np
import matplotlib.pyplot as plt

from load_data import load_data
from error_fcns import mse


def create_static_model(x: np.ndarray, y: np.ndarray, polynomial_order: int):
    M = np.array([np.power(x, i) for i in range(polynomial_order+1)])

    w = np.linalg.lstsq(M.T, y)[0]      # numpy's equivalent to MATLAB left division
    model = lambda u: np.sum([w*np.power(u, i) for i, w in enumerate(w)], axis=0)
    return model


if __name__ == "__main__":
    data = load_data("data/danestat48.txt")

    sorted_data = np.array(sorted(data, key=lambda x: x[0]))
    x = sorted_data[:, 0]
    y = sorted_data[:, 1]

    # Test polynomials
    for i in range(2, 3):
        model = create_static_model(x, y, i)
        u = np.linspace(-1, 1, 200)
        y_mod = model(u)
        print(f"Poly order: {i}\tMSE: {mse(y, y_mod):.5f}")
        plt.scatter(x, y)
        plt.scatter(u, y_mod)
        plt.show()
