import numpy as np
from simulation.vessel import Vessel


class SceneSimulator:
    def __init__(self, num_vessels, dt, rng):
        self.num_vessels = num_vessels
        self.dt = dt
        self.rng = rng
        self.vessels = self._init_vessels()

    def _init_vessels(self):
        vessels = []
        for i in range(self.num_vessels):
            px = self.rng.uniform(-600, 600)
            py = self.rng.uniform(-600, 600)
            vx = self.rng.uniform(-4, 4)
            vy = self.rng.uniform(-4, 4)
            vessels.append(Vessel(i, np.array([px, py, vx, vy], dtype=float)))
        return vessels

    def step(self, t):
        gt = []
        for v in self.vessels:
            v.step(self.dt, process_noise_std=0.2, rng=self.rng)
            gt.append({
                "timestamp": t,
                "vessel_id": v.vessel_id,
                "px": float(v.state[0]),
                "py": float(v.state[1]),
                "vx": float(v.state[2]),
                "vy": float(v.state[3]),
            })
        return gt
