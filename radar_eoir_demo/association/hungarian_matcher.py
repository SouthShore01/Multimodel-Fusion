import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_with_gate(cost, gate, large_cost):
    if cost.size == 0:
        return []
    c = cost.copy()
    c[c > gate] = large_cost
    rows, cols = linear_sum_assignment(c)
    return [(r, col) for r, col in zip(rows, cols) if c[r, col] < large_cost]
