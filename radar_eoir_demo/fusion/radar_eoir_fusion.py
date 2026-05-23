import numpy as np
from scipy.optimize import linear_sum_assignment


def associate_radar_eoir(radar_tracks, eoir_tracks, r_eoir, gate, large_cost):
    if len(radar_tracks) == 0 or len(eoir_tracks) == 0:
        return []
    h = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    cost = np.full((len(radar_tracks), len(eoir_tracks)), large_cost, float)
    for i, rt in enumerate(radar_tracks):
        for j, et in enumerate(eoir_tracks):
            y = et["x"][:2] - h @ rt["x"]
            s = h @ rt["P"] @ h.T + r_eoir
            d2 = float(y.T @ np.linalg.inv(s) @ y)
            if d2 <= gate:
                cost[i, j] = d2
    rows, cols = linear_sum_assignment(cost)
    return [{"radar_idx": i, "eoir_idx": j, "cost": float(cost[i, j])} for i, j in zip(rows, cols) if cost[i, j] < large_cost]
