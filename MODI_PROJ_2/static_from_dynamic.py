from dynamic_model import get_model_coefficients, get_model_output
from load_data import load_data
import numpy as np
import matplotlib.pyplot as plt


def make_plot_double_x(x_1: np.ndarray, y_1: np.ndarray, x_2: np.ndarray, y_2: np.ndarray,
                       filename: str, save=False, resolution=100, data_label=None, model_label=None) -> None:
    fig, ax = plt.subplots(figsize=(1280 / resolution, 720 / resolution))
    ax.scatter(x_1, y_1, label=data_label)
    ax.plot(x_2, y_2, label=model_label, color="orange", linewidth=5)
    ax.grid(which="major")
    ax.tick_params(axis='both', labelsize=14)
    ax.ticklabel_format(useLocale=True)

    plt.xlim([-1.1, 1.1])
    plt.ylim([-0.55, 2.5])

    plt.xlabel(r"$u$", fontsize=20)
    plt.ylabel(r"$y$", fontsize=20)
    plt.tight_layout(pad=0.15)

    if data_label and model_label:
        fig.legend(fontsize=20, loc='lower right', bbox_to_anchor=(1, 0.2))

    if save:
        fig.savefig(filename, dpi=resolution)


if __name__ == "__main__":
    static_data = load_data("data/danestat48.txt")
    dyn_ucz_data = load_data("data/danedynucz48.txt")

    train_x = dyn_ucz_data[:, 0]
    train_y = dyn_ucz_data[:, 1]
    rank = 1
    poly = 3
    samples = 1000
    iterations = 1000

    weights = get_model_coefficients(train_x, train_y, poly, rank)
    u_linspace = np.linspace(-1, 1, samples)
    y_stable = []
    for i in range(len(u_linspace)):
        iterations = iterations
        u_vec = np.ones(iterations,) * u_linspace[i]
        y = np.array([0 for _ in range(rank)])

        predictions = get_model_output(u_vec, y, weights, poly, rank, recursive=True)
        y_stable.append(predictions[-1])

    make_plot_double_x(static_data[:, 0], static_data[:, 1], u_linspace, np.array(y_stable),
                       filename="images/static/static_from_dynamic.png", save=True)
    plt.show()