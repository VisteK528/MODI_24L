from static_model import create_static_model
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data
from error_fcns import mse
import locale


def make_plot(x: np.ndarray, y: np.ndarray, y_mod: np.ndarray, filename: str, save=False,
              resolution=400, data_label=None, model_label=None) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.scatter(x, y, label=data_label)
    ax.scatter(x, y_mod, label=model_label)
    ax.grid(which="major")
    ax.tick_params(axis='both', labelsize=20)
    ax.ticklabel_format(useLocale=True)

    if data_label and model_label:
        fig.legend(fontsize=20)

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

    # Train model
    model = create_static_model(x_train, y_train, 1)

    # Predict for train and test data
    y_train_mod = model(x_train)
    y_test_mod = model(x_test)

    # Train data
    make_plot(x_train, y_train, y_train_mod, "images/static/exb_static_linear_model_train.png",
              data_label="Dane uczące", model_label="Model liniowy", save=True)

    # Test data
    make_plot(x_test, y_test, y_test_mod, "images/static/exb_static_linear_model_test.png",
              data_label="Dane weryfikujące", model_label="Model liniowy", save=True)

    print(f"Train data MSE: {mse(y_train, y_train_mod)}")
    print(f"Test data MSE: {mse(y_test, y_test_mod)}")

    plt.show()
