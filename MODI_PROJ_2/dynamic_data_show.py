from load_data import load_data
import numpy as np
import matplotlib.pyplot as plt


def make_plot(x: np.ndarray, y: np.ndarray, filename: str, save=False,
              resolution=100, data_label=None) -> None:
    fig, ax = plt.subplots(figsize=(1280/resolution, 720/resolution))
    ax.plot(x, y, label=data_label, marker='o', markersize='2')
    ax.grid(which="major")
    ax.tick_params(axis='both')
    ax.ticklabel_format(useLocale=True)

    plt.xlabel(r"k", fontsize=14)
    plt.ylabel(r"y", fontsize=14)
    plt.tight_layout(pad=0.15)

    if save:
        fig.savefig(filename, dpi=resolution)


if __name__ == "__main__":
    dyn_ucz_data = load_data("data/danedynucz48.txt")
    dyn_wer_data = load_data("data/danedynwer48.txt")

    train_x = dyn_ucz_data[:, 0]
    train_y = dyn_ucz_data[:, 1]

    valid_x = dyn_wer_data[:, 0]
    valid_y = dyn_wer_data[:, 1]
    print(max(train_x))
    print(min(train_x))

    # Train x plot
    make_plot(np.linspace(0, len(train_x), len(train_x)), train_x, "images/dynamic/ex_a_train_x.png", save=True)

    # Train y plot
    make_plot(np.linspace(0, len(train_y), len(train_y)), train_y, "images/dynamic/ex_a_train_y.png", save=True)

    # Valid x plot
    make_plot(np.linspace(0, len(valid_x), len(valid_x)), valid_x, "images/dynamic/ex_a_valid_x.png", save=True)

    # Valid y plot
    make_plot(np.linspace(0, len(valid_y), len(valid_y)), valid_y, "images/dynamic/ex_a_valid_y.png", save=True)

    plt.show()
