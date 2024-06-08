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

    train = data[::2]
    test = data[1::2]
    x_train = train[:, 0]
    y_train = train[:, 1]

    x_test = test[:, 0]
    y_test = test[:, 1]

    # Train data
    plt.scatter(x_train)


    """# Test polynomials
    for i in range(1, 6):
        model = create_static_model(x_train, y_train, i)
        u = np.linspace(-1, 1, 200)
        y_mod = model(x_test)
        print(f"Poly order: {i}\tMSE: {mse(y_test, y_mod):.5f}")
        plt.scatter(x_test, y_mod)
        plt.scatter(x_test, y_test)
        plt.show()"""
