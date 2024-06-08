from static_model import create_static_model
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data
from error_fcns import mse
import locale


def make_plot(x: np.ndarray, y: np.ndarray, y_mod: np.ndarray, filename: str, save=False, resolution=400) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.scatter(x, y)
    ax.scatter(x, y_mod)
    ax.grid(which="major")
    ax.tick_params(axis='both', labelsize=20)
    ax.ticklabel_format(useLocale=True)

    if save:
        fig.savefig(filename, dpi=resolution)


if __name__ == "__main__":
    locale.setlocale(locale.LC_NUMERIC, "de_DE.UTF-8")
    data = load_data("data/danestat48.txt")

    train = data[::2]
    test = data[1::2]
    x_train = train[:, 0]
    y_train = train[:, 1]

    x_test = test[:, 0]
    y_test = test[:, 1]

    for poly_order in range(2, 6):
        # Train model
        model = create_static_model(x_train, y_train, poly_order)

        # Predict for train and test data
        y_train_mod = model(x_train)
        y_test_mod = model(x_test)

        # Train data
        make_plot(x_train, y_train, y_train_mod, f"images/static/ex_1c_static_poly_{poly_order}_model_train.png", save=True)

        # Test data
        make_plot(x_test, y_test, y_test_mod, f"images/static/ex_1c_static_poly_{poly_order}_model_test.png", save=True)

        print(f"Polynomial order: {poly_order}")
        print(f"Train data MSE: {mse(y_train, y_train_mod)}")
        print(f"Test data MSE: {mse(y_test, y_test_mod)}")

    plt.show()
