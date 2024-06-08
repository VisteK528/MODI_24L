from load_data import load_data
import numpy as np
import locale
import matplotlib.pyplot as plt


def make_plot(x: np.ndarray, y: np.ndarray, filename: str, save=False,
              resolution=400, data_label=None) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.scatter(np.linspace(0, len(y), len(y)), y, label=data_label)
    ax.grid(which="major")
    ax.tick_params(axis='both', labelsize=20)
    ax.ticklabel_format(useLocale=True)

    if data_label:
        fig.legend(fontsize=20)

    if save:
        fig.savefig(filename, dpi=resolution)


if __name__ == "__main__":
    locale.setlocale(locale.LC_NUMERIC, "de_DE.UTF-8")

    dyn_ucz_data = load_data("data/danedynucz48.txt")
    dyn_wer_data = load_data("data/danedynwer48.txt")

    train_x = dyn_ucz_data[:, 0]
    train_y = dyn_ucz_data[:, 1]

    valid_x = dyn_wer_data[:, 0]
    valid_y = dyn_wer_data[:, 1]

    # Train plot
    make_plot(train_x, train_y, "images/dynamic/ex_a_train.png", save=True)

    # Valid plot
    make_plot(valid_x, valid_y, "images/dynamic/ex_a_valid.png", save=True)

    plt.show()
