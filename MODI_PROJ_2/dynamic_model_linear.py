from dynamic_model import get_model_coefficients, get_model_output
from load_data import load_data
import numpy as np
import matplotlib.pyplot as plt
from error_fcns import mse


def make_plot(x: np.ndarray, y: np.ndarray, y_mod: np.ndarray,
              filename: str, save=False, resolution=200, data_label=None, model_label=None, dots=True) -> None:
    fig, ax = plt.subplots(figsize=(1280/resolution, 720/resolution))
    ax.plot(x, y, label=data_label, marker='o', markersize='2')
    if dots:
        ax.plot(x, y_mod, label=model_label, marker='o', markersize='2')
    else:
        ax.plot(x, y_mod, label=model_label)
    ax.grid(which="major")
    ax.ticklabel_format(useLocale=True)

    plt.xlabel(r"$k$", fontsize=14)
    plt.ylabel(r"$y$", fontsize=14)
    plt.tight_layout(pad=0.15)
    ax.legend(fontsize=10, loc='best')

    if save:
        fig.savefig(filename, dpi=resolution)


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

        # Train data non recursive
        make_plot(k, train_y, y_mod,
                  f"images/dynamic/ex_b_{rank}_train_nonrecursive.png",
                  data_label="Dane uczące", model_label="Tryb nierekurencyjny", save=False)

        # Train data recursive
        make_plot(k, train_y, y_mod_recursive,
                  f"images/dynamic/ex_b_rank_{rank}_train_recursive.png",
                  data_label="Dane uczące", model_label="Tryb rekurencyjny", save=True, dots=False)

        y_mod_valid = get_model_output(valid_x, valid_y, weights, poly_order, rank)
        y_mod_recursive_valid = get_model_output(valid_x, valid_y, weights, poly_order, rank, recursive=False)

        # Validation data non recursive
        make_plot(k, valid_y, y_mod_valid,
                  f"images/dynamic/ex_b_rank_{rank}_valid_nonrecursive.png",
                  data_label="Dane weryfikujące", model_label="Tryb nierekurencyjny", save=False)

        # Validation data recursive
        make_plot(k, valid_y, y_mod_recursive_valid,
                  f"images/dynamic/ex_b_rank_{rank}_valid_recursive.png",
                  data_label="Dane weryfikujące", model_label="Tryb rekurencyjny", save=False, dots=False)


        print(f"Train\t            Poly order: {poly_order}\t Rank: {rank}\tMSE: {mse(train_y, y_mod):.5f}")
        print(f"Train recursive\t    Poly order: {poly_order}\t Rank: {rank}\tMSE: {mse(train_y, y_mod_recursive):.5f}")
        print(f"Validate\t        Poly order: {poly_order}\t Rank: {rank}\tMSE: {mse(valid_y, y_mod_valid):.5f}")
        print(f"Validate recursive\tPoly order: {poly_order}\t Rank: {rank}\tMSE: {mse(valid_y, y_mod_recursive_valid):.5f}")

    plt.show()
