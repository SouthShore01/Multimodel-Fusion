from dataclasses import dataclass
import numpy as np


@dataclass
class Vessel:
    vessel_id: int
    state: np.ndarray  # [px, py, vx, vy]

    def step(self, dt: float, process_noise_std: float, rng: np.random.Generator):
        noise = rng.normal(0.0, process_noise_std, size=2)
        self.state[0] += self.state[2] * dt + noise[0]
        self.state[1] += self.state[3] * dt + noise[1]
