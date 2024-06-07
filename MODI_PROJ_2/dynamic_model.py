import numpy as np
import matplotlib.pyplot as plt

from load_data import load_data
from error_fcns import mse


def matrix_part(data: np.ndarray, polynomial_order: int):
    m_part = np.array([np.power(data, i) for i in range(1, polynomial_order+1)])
    return m_part


def get_model_coefficients(x: np.ndarray, y: np.ndarray, polynomial_order: int, rank: int) -> np.ndarray:
    Y = y[rank:]

    M_u = np.array([matrix_part(x[rank - i:-i], polynomial_order) for i in range(1, rank+1)]).reshape((polynomial_order*rank, len(Y)))
    M_y = np.array([matrix_part(y[rank - i:- i], polynomial_order) for i in range(1, rank+1)]).reshape((polynomial_order*rank, len(Y)))
    M = np.concatenate((M_u.T, M_y.T), axis=1)

    w = np.linalg.lstsq(M, Y, rcond=None)[0]      # numpy's equivalent to MATLAB left division
    return w


def get_model_output(x: np.ndarray, y: np.ndarray, weights: np.ndarray, polynomial_order: int, rank: int, recursive=False) -> np.ndarray:
    y_mod = np.zeros(len(x))

    if recursive:
        for i in range(rank):
            y_mod[i] = y[i]

    for k in range(rank, len(y)):
        u_part = 0
        for j in range(rank):
            batch_weights = weights[j*polynomial_order: (j+1)*polynomial_order]
            u_part += np.sum([w * pow(x[k-j-1], i) for i, w in enumerate(batch_weights, start=1)])

        y_part = 0
        for j in range(rank):
            batch_weights = weights[rank * polynomial_order + j*polynomial_order: rank * polynomial_order + (j+1)*polynomial_order]
            if recursive:
                y_part += np.sum(
                    [w * pow(y_mod[k - j - 1], i) for i, w in enumerate(batch_weights, start=1)])
            else:
                y_part += np.sum(
                    [w * pow(y[k - j - 1], i) for i, w in enumerate(batch_weights, start=1)])
        y_mod[k] = u_part + y_part

    return y_mod


if __name__ == "__main__":
    dyn_ucz_data = load_data("data/danedynucz48.txt")
    dyn_wer_data = load_data("data/danedynwer48.txt")

    train_x = dyn_ucz_data[:, 0]
    train_y = dyn_ucz_data[:, 1]

    valid_x = dyn_wer_data[:, 0]
    valid_y = dyn_wer_data[:, 1]

    k = np.linspace(0, len(valid_x), len(valid_x))

    for i in range(3):
        poly_order = 4
        rank = 1 + i

        weights = get_model_coefficients(train_x, train_y, poly_order, rank)
        y_mod = get_model_output(train_x, train_y, weights, poly_order, rank)
        y_mod_recursive = get_model_output(train_x, train_y, weights, poly_order, rank, recursive=True)

        plt.plot(k, train_y, label="Validation data")
        plt.plot(k, y_mod, label="Non-Recursive model")

        plt.plot(k, y_mod_recursive, label="Recursive model")
        plt.title(f"Train Poly: {poly_order}    Rank: {rank}")
        plt.legend()
        plt.show()

        y_mod_valid = get_model_output(valid_x, valid_y, weights, poly_order, rank)
        y_mod_recursive_valid = get_model_output(valid_x, valid_y, weights, poly_order, rank, recursive=True)

        plt.plot(k, valid_y, label="Validation data")
        plt.plot(k, y_mod_valid, label="Non-Recursive model")

        plt.plot(k, y_mod_recursive_valid, label="Recursive model", linestyle="--")
        plt.title(f"Validation Poly: {poly_order}    Rank: {rank}")
        plt.legend()
        plt.show()

        print(f"Train\t            Poly order: {poly_order}\t Rank: {rank}\tMSE: {mse(train_y, y_mod):.5f}")
        print(f"Validate\t        Poly order: {poly_order}\t Rank: {rank}\tMSE: {mse(valid_y, y_mod_valid):.5f}")
        print(f"Validate recursive\tPoly order: {poly_order}\t Rank: {rank}\tMSE: {mse(valid_y, y_mod_recursive_valid):.5f}")