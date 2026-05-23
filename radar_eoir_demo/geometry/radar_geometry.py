import numpy as np


def xy_to_range_bearing(x, y):
    r = np.hypot(x, y)
    b = np.degrees(np.arctan2(y, x))
    return r, b


def velocity_to_speed_heading(vx, vy):
    speed = np.hypot(vx, vy)
    heading = np.degrees(np.arctan2(vy, vx))
    return speed, heading
