from dynamic_model import get_model_coefficients, get_model_output
from load_data import load_data
import numpy as np
import matplotlib.pyplot as plt
from error_fcns import mse


if __name__ == "__main__":
    dyn_ucz_data = load_data("data/danedynucz48.txt")
    dyn_wer_data = load_data("data/danedynwer48.txt")

    train_x = dyn_ucz_data[:, 0]
    train_y = dyn_ucz_data[:, 1]

    valid_x = dyn_wer_data[:, 0]
    valid_y = dyn_wer_data[:, 1]

    k = np.linspace(0, len(valid_x), len(valid_x))

    poly_order = 1
    for rank in range(1, 4):

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
