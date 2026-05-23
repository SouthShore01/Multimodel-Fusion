import numpy as np


class CVKalman:
    def __init__(self, dt, q):
        self.f = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)
        self.h = np.array([[1,0,0,0],[0,1,0,0]], float)
        g = np.array([[dt**2/2,0],[0,dt**2/2],[dt,0],[0,dt]])
        self.q = g @ (q*np.eye(2)) @ g.T

    def predict(self, x, p):
        return self.f @ x, self.f @ p @ self.f.T + self.q

    def update(self, x, p, z, r):
        y = z - self.h @ x
        s = self.h @ p @ self.h.T + r
        k = p @ self.h.T @ np.linalg.inv(s)
        x_new = x + k @ y
        p_new = (np.eye(4) - k @ self.h) @ p
        return x_new, p_new
