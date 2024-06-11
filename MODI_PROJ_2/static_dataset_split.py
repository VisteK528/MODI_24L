import numpy as np
from load_data import load_data
import matplotlib.pyplot as plt
import locale


def make_plot(x: np.ndarray, y: np.ndarray, filename: str, save=False, resolution=100) -> None:
    fig, ax = plt.subplots(figsize=(1280 / resolution, 720 / resolution))
    ax.scatter(x, y)
    ax.grid(which="major")
    ax.tick_params(axis='both', labelsize=14)
    ax.ticklabel_format(useLocale=True)

    plt.xlim([-1.1, 1.1])
    plt.ylim([-0.55, 2.5])

    plt.xlabel(r"$u$", fontsize=20)
    plt.ylabel(r"$y$", fontsize=20)
    plt.tight_layout(pad=0.15)

    if save:
        fig.savefig(filename, dpi=resolution)


if __name__ == "__main__":
    plt.rcParams['text.usetex'] = True
    plt.rcParams['lines.linewidth'] = 1
    locale.setlocale(locale.LC_NUMERIC, "pl_PL.UTF-8")
    data = load_data("data/danestat48.txt")

    # Whole dataset
    make_plot(data[:, 0], data[:, 1], "images/static/static_data.png", save=True)

    train = data[::2]
    valid = data[1::2]
    x_train = train[:, 0]
    y_train = train[:, 1]

    x_valid = valid[:, 0]
    y_valid = valid[:, 1]

    # Train data
    make_plot(x_train, y_train, "images/static/static_train_data.png", save=True)

    # Validation data
    make_plot(x_valid, y_valid, "images/static/static_validation_data.png", save=True)
    plt.show()