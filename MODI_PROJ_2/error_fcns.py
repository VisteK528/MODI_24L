import numpy as np


def mse(y_data: np.ndarray, y_mod_data: np.ndarray) -> float:
    assert len(y_data) == len(y_mod_data)
    return np.sum([pow(y_mod - y, 2) for y_mod, y in zip(y_mod_data, y_data)]) / len(y_data)
