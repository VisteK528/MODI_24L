from keras import Sequential
from keras.api.optimizers import Adam
from keras.api.layers import Dense
from load_data import load_data
from keras.api.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from error_fcns import mse


def make_plot(
    x: np.ndarray,
    y: np.ndarray,
    y_mod: np.ndarray,
    filename: str,
    save=False,
    resolution=200,
    data_label=None,
    model_label=None,
    dots=True,
) -> None:
    fig, ax = plt.subplots(
        figsize=(1280 / resolution, 720 / resolution)
    )
    ax.plot(x, y, label=data_label, marker="o", markersize="2")
    if dots:
        ax.plot(
            x, y_mod, label=model_label, marker="o", markersize="2"
        )
    else:
        ax.plot(x, y_mod, label=model_label)
    ax.grid(which="major")
    ax.ticklabel_format(useLocale=True)

    plt.xlabel(r"$k$", fontsize=14)
    plt.ylabel(r"$y$", fontsize=14)
    plt.tight_layout(pad=0.15)
    ax.legend(fontsize=10, loc="best")

    if save:
        fig.savefig(filename, dpi=100)


def create_input_matrix(data: np.ndarray):
    u = data[:, 0]
    y = data[:, 1]

    m = np.array([u[:-1], y[:-1]])
    return m


def get_neural_model_output(x: np.ndarray, model, recursive=False):
    if recursive:
        u = x[:, 0]
        y_mod = np.zeros(len(u))

        for i in range(2):
            y_mod[i] = x[i, 1]

        for k in range(2, len(x)):
            input = np.array([[u[k - 1], y_mod[k - 1]]])
            prediction = model.__call__(input)
            y_mod[k] = prediction
    else:
        y_mod = model.predict(x)
    return y_mod


if __name__ == "__main__":
    dyn_ucz_data = load_data("data/danedynucz48.txt")
    dyn_wer_data = load_data("data/danedynwer48.txt")

    # Create datasets
    x_train_full = create_input_matrix(dyn_ucz_data).T
    y_train_full = dyn_ucz_data[:, 1][1:]
    x_test = create_input_matrix(dyn_wer_data).T
    y_test = dyn_wer_data[:, 1][1:]

    x_train = x_train_full[200:]
    valid_x = x_train[:200]
    y_train = y_train_full[200:]
    valid_y = y_train[:200]

    activation = "relu"
    for i in range(1, 2, 2):
        model = Sequential(
            [
                Dense(i, activation=activation, input_shape=(2,)),
                Dense(1),
            ]
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        model.compile(optimizer=Adam(), loss="mse")

        model.fit(
            x_train,
            y_train,
            epochs=1000,
            validation_data=(valid_x, valid_y),
            callbacks=[early_stopping],
            verbose=1,
        )

        # Predict train data
        y_mod_train = get_neural_model_output(x_train_full, model)
        y_mod_train_recursive = get_neural_model_output(
            x_train_full, model, recursive=True
        )

        # Predict test data
        y_mod_test = get_neural_model_output(x_test, model)
        y_mod_test_recursive = get_neural_model_output(
            x_test, model, recursive=True
        )

        k = np.linspace(0, len(y_mod_train), len(y_mod_train))

        # Train nonrecursive
        make_plot(
            k,
            dyn_ucz_data[1:, 1],
            y_mod_train,
            f"images/extra/extra/extra_ex_neurons_{i}_activation_{activation}_train.png",
            data_label="Dane uczące",
            model_label="Tryb bez rekurencji",
            save=True,
        )

        # Train recursive
        make_plot(
            k,
            dyn_ucz_data[1:, 1],
            y_mod_train_recursive,
            f"images/extra/extra/extra_ex_neurons_{i}_activation_{activation}_train_recursive.png",
            data_label="Dane uczące",
            model_label="Tryb rekurencyjny",
            save=True,
            dots=False,
        )

        # Test nonrecursive
        make_plot(
            k,
            dyn_wer_data[1:, 1],
            y_mod_test,
            f"images/extra/extra/extra_ex_neurons_{i}_activation_{activation}_test.png",
            data_label="Dane testujące",
            model_label="Tryb bez rekurencji",
            save=True,
        )

        # Test recursive
        make_plot(
            k,
            dyn_wer_data[1:, 1],
            y_mod_test_recursive,
            f"images/extra/extra/extra_ex_neurons_{i}_activation_{activation}_test_recursive.png",
            data_label="Dane testujące",
            model_label="Tryb rekurencyjny",
            save=True,
            dots=False,
        )

        print(f"Neurons: {i}\tActivation: {activation}")
        print(
            f"Train\t            MSE: {mse(y_train_full, y_mod_train):.7f}"
        )
        print(
            f"Train recursive\t    MSE: {mse(y_train_full, y_mod_train_recursive):.7f}"
        )
        print(f"Validate\t        MSE: {mse(y_test, y_mod_test):.7f}")
        print(
            f"Validate recursiveMSE: {mse(y_test, y_mod_test_recursive):.7f}"
        )
    plt.show()
