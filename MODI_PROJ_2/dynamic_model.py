import numpy as np


def matrix_part(data: np.ndarray, polynomial_order: int):
    m_part = np.array(
        [np.power(data, i) for i in range(1, polynomial_order + 1)]
    )
    return m_part


def get_model_coefficients(
    x: np.ndarray,
    y: np.ndarray,
    polynomial_order: int,
    rank: int,
) -> np.ndarray:
    Y = y[rank:]

    M_u = np.array(
        [
            matrix_part(x[rank - i : -i], polynomial_order)
            for i in range(1, rank + 1)
        ]
    ).reshape((polynomial_order * rank, len(Y)))
    M_y = np.array(
        [
            matrix_part(y[rank - i : -i], polynomial_order)
            for i in range(1, rank + 1)
        ]
    ).reshape((polynomial_order * rank, len(Y)))
    M = np.concatenate((M_u.T, M_y.T), axis=1)

    # Manual least squares implementation
    w = np.linalg.inv(M.T @ M) @ M.T @ Y
    return w


def get_model_output(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    polynomial_order: int,
    rank: int,
    recursive=False,
) -> np.ndarray:
    y_mod = np.zeros(len(x))

    if recursive:
        for i in range(rank):
            y_mod[i] = y[i]

    for k in range(rank, len(x)):
        u_part = 0
        for j in range(rank):
            batch_weights = weights[
                j * polynomial_order : (j + 1) * polynomial_order
            ]
            u_part += np.sum(
                [
                    w * pow(x[k - j - 1], i)
                    for i, w in enumerate(batch_weights, start=1)
                ]
            )

        y_part = 0
        for j in range(rank):
            batch_weights = weights[
                rank * polynomial_order
                + j * polynomial_order : rank * polynomial_order
                + (j + 1) * polynomial_order
            ]
            if recursive:
                y_part += np.sum(
                    [
                        w * pow(y_mod[k - j - 1], i)
                        for i, w in enumerate(batch_weights, start=1)
                    ]
                )
            else:
                y_part += np.sum(
                    [
                        w * pow(y[k - j - 1], i)
                        for i, w in enumerate(batch_weights, start=1)
                    ]
                )
        y_mod[k] = u_part + y_part

    return y_mod
