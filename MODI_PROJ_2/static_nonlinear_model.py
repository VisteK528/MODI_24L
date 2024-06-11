from static_model import create_static_model
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data
from error_fcns import mse
import locale


def make_plot(x: np.ndarray, y: np.ndarray, y_mod: np.ndarray,
              filename: str, save=False, resolution=200, data_label=None, model_label=None, lim=True, compare_labels=False) -> None:
    fig, ax = plt.subplots(figsize=(1280/resolution, 720/resolution))
    ax.scatter(x, y, label=data_label, s=5)
    if y_mod is not None:
        ax.scatter(x, y_mod, label=model_label, s=5)
    ax.grid(which="major")
    ax.ticklabel_format(useLocale=True)

    if lim:
        plt.xlim([-1.1, 1.1])
        plt.ylim([-0.55, 2.5])

    if compare_labels:
        plt.xlabel(r"$u$", fontsize=14)
        plt.ylabel(r"$y$", fontsize=14)
    else:
        plt.xlabel(r"$y$", fontsize=14)
        plt.ylabel(r"$y_{mod}$", fontsize=14)

    plt.tight_layout(pad=0.15)

    if data_label and model_label:
        fig.legend(fontsize=10, loc='lower right', bbox_to_anchor=(1, 0.2))

    if save:
        fig.savefig(filename, dpi=resolution)


def make_plot_double_x(x_1: np.ndarray, y_1: np.ndarray, x_2: np.ndarray, y_2: np.ndarray,
                       filename: str, save=False, resolution=200, data_label=None, model_label=None) -> None:
    fig, ax = plt.subplots(figsize=(1280/resolution, 720/resolution))
    ax.scatter(x_1, y_1, label=data_label, s=5)
    ax.plot(x_2, y_2, label=model_label, color="orange")
    ax.grid(which="major")
    # ax.tick_params(axis='both', labelsize=20)
    ax.ticklabel_format(useLocale=True)

    plt.xlim([-1.1, 1.1])
    plt.ylim([-0.55, 2.5])

    plt.xlabel(r"$u$", fontsize=14)
    plt.ylabel(r"$y$", fontsize=14)
    plt.tight_layout(pad=0.15)

    if data_label and model_label:
         fig.legend(fontsize=10, loc='lower right', bbox_to_anchor=(1, 0.2))

    if save:
        fig.savefig(filename, dpi=resolution)


if __name__ == "__main__":
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage[utf8]{inputenc} \usepackage{polski}'
    locale.setlocale(locale.LC_NUMERIC, "pl_PL.UTF-8")
    data = load_data("data/danestat48.txt")

    train = data[::2]
    valid = data[1::2]
    x_train = train[:, 0]
    y_train = train[:, 1]

    x_valid = valid[:, 0]
    y_valid = valid[:, 1]

    for poly_order in range(2, 6):
        # Train model
        model = create_static_model(x_train, y_train, poly_order)

        # Predict for train and test data
        y_train_mod = model(x_train)
        y_valid_mod = model(x_valid)

        # Train data
        make_plot(x_train, y_train, y_train_mod, f"images/static/ex_1c_static_nonlinear_poly_{poly_order}_model_train.png",
                  data_label=r"Dane uczące", model_label=r"Wyjście dla danych uczących", save=True)

        # Validation data
        make_plot(x_valid, y_valid, y_valid_mod, f"images/static/ex_1c_static_nonlinear_poly_{poly_order}_model_valid.png",
                  data_label="Dane weryfikujące", model_label="Wyjście dla danych weryfikujących", save=True)

        # Model
        u_linspace = np.arange(-1, 1, 0.01)
        y_mod_fcn = model(u_linspace)
        make_plot_double_x(x_valid, y_valid, u_linspace, y_mod_fcn,
                           f"images/static/ex_1c_static_nonlinear_poly_{poly_order}_model_y_u.png",
                           data_label="Dane weryfikujące", model_label=r"Charakterystyka $y(u)$", save=True)

        # Fit check
        make_plot(y_valid, y_valid_mod, None, f"images/static/ex_1c_static_nonlinear_poly_{poly_order}_model_fit_check.png",
                  save=True, lim=False)

        print(f"Polynomial order: {poly_order}")
        print(f"Train data MSE: {mse(y_train, y_train_mod)}")
        print(f"Validation data MSE: {mse(y_valid, y_valid_mod)}\n")
    plt.show()
