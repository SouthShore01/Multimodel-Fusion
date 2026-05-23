import numpy as np
from geometry.camera_geometry import look_at_rotation, project_point


class VisibleCameraSimulator:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng

    def _uav_pose(self, t):
        cam = np.array([200.0, -300.0 + 3.0 * t, 120.0])
        target = np.array([0.0, 0.0, 0.0])
        r_cw = look_at_rotation(cam, target)
        return cam, r_cw

    def simulate(self, gt_list, t):
        cam, r_cw = self._uav_pose(t)
        intr = (self.cfg.FX, self.cfg.FY, self.cfg.CX, self.cfg.CY)
        dets = []
        for gt in gt_list:
            if self.rng.random() < self.cfg.VISIBLE_DROP_PROB:
                continue
            uv = project_point(np.array([gt["px"], gt["py"], 0.0]), cam, r_cw, intr)
            if uv is None:
                continue
            if not (0 <= uv[0] < self.cfg.IMAGE_WIDTH and 0 <= uv[1] < self.cfg.IMAGE_HEIGHT):
                continue
            c = uv + self.rng.normal(0, self.cfg.VISIBLE_PIXEL_NOISE, size=2)
            w, h = 40.0, 25.0
            bbox = [float(c[0]-w/2), float(c[1]-h/2), float(w), float(h)]
            dets.append({"timestamp": t, "bbox": bbox, "confidence": float(self.rng.uniform(0.7,0.99)),
                         "vessel_id": gt["vessel_id"]})
        return dets, {"cam_pos": cam, "r_cw": r_cw}
