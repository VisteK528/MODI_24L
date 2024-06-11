import numpy as np
from typing import Callable


def create_static_model(x: np.ndarray, y: np.ndarray,
                        polynomial_order: int) -> Callable:
    M = np.array([np.power(x, i) for i in range(polynomial_order+1)]).T
    w = np.linalg.inv(M.T@M)@M.T@y

    model = lambda u: np.sum([w*np.power(u, i) for i,
    w in enumerate(w)], axis=0)
    return model
