import numpy as np
from association.hungarian_matcher import hungarian_with_gate


def euclidean_match(a, b, gate, large_cost):
    if len(a) == 0 or len(b) == 0:
        return [], np.zeros((len(a), len(b)))
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return hungarian_with_gate(cost, gate, large_cost), cost
