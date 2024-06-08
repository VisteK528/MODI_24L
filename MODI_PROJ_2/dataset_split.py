import numpy as np

from load_data import load_data
import matplotlib.pyplot as plt
import locale


def make_plot(x: np.ndarray, y: np.ndarray, filename: str, save=False, resolution=400) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.scatter(x, y)
    ax.grid(which="major")
    ax.tick_params(axis='both', labelsize=20)
    ax.ticklabel_format(useLocale=True)

    if save:
        fig.savefig(filename, dpi=resolution)


if __name__ == "__main__":
    locale.setlocale(locale.LC_NUMERIC, "de_DE.UTF-8")
    data = load_data("data/danestat48.txt")

    # Whole dataset
    make_plot(data[:, 0], data[:, 1], "images/static/static_dataset.png", save=True)

    train = data[::2]
    test = data[1::2]
    x_train = train[:, 0]
    y_train = train[:, 1]

    x_test = test[:, 0]
    y_test = test[:, 1]

    # Train data
    make_plot(x_train, y_train, "images/static/train_dataset.png", save=True)

    # Test data
    make_plot(x_test, y_test, "images/static/test_data.png", save=True)
    plt.show()