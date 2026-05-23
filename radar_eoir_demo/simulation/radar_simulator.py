import numpy as np
from geometry.coordinates import enu_to_latlon
from geometry.radar_geometry import xy_to_range_bearing, velocity_to_speed_heading


class RadarSimulator:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng

    def simulate(self, gt_list, t):
        dets = []
        for gt in gt_list:
            if self.rng.random() < self.cfg.RADAR_DROP_PROB:
                continue
            nx = gt["px"] + self.rng.normal(0, self.cfg.RADAR_SIGMA_POSITION)
            ny = gt["py"] + self.rng.normal(0, self.cfg.RADAR_SIGMA_POSITION)
            lat, lon = enu_to_latlon(nx, ny, self.cfg.ORIGIN_LAT, self.cfg.ORIGIN_LON)
            r, b = xy_to_range_bearing(nx, ny)
            spd, hdg = velocity_to_speed_heading(gt["vx"], gt["vy"])
            spd += self.rng.normal(0, self.cfg.RADAR_SIGMA_SPEED)
            hdg += self.rng.normal(0, self.cfg.RADAR_SIGMA_HEADING_DEG)
            dets.append({"timestamp": t, "lat": lat, "lon": lon, "range": r, "bearing": b,
                         "speed": spd, "heading": hdg, "vessel_id": gt["vessel_id"]})
        if self.rng.random() < self.cfg.RADAR_CLUTTER_PROB:
            cx, cy = self.rng.uniform(-800, 800, size=2)
            lat, lon = enu_to_latlon(cx, cy, self.cfg.ORIGIN_LAT, self.cfg.ORIGIN_LON)
            r, b = xy_to_range_bearing(cx, cy)
            dets.append({"timestamp": t, "lat": lat, "lon": lon, "range": r, "bearing": b,
                         "speed": abs(self.rng.normal(2, 1)), "heading": self.rng.uniform(-180, 180),
                         "vessel_id": -1})
        return dets
