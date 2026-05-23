import numpy as np


def mahalanobis_d2(y, s):
    return float(y.T @ np.linalg.inv(s) @ y)
