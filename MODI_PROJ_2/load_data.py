import numpy as np


def load_data(path: str) -> np.ndarray:
    data = []
    with open(path, "r") as f:
        while line := f.readline():
            stripped_line = line.strip('\n').strip('\r')
            x, y = stripped_line.split()
            data.append([float(x), float(y)])
    return np.array(data)

