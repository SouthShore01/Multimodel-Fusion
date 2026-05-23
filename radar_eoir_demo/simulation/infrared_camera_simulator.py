import numpy as np
from simulation.visible_camera_simulator import VisibleCameraSimulator


class InfraredCameraSimulator(VisibleCameraSimulator):
    def simulate(self, gt_list, t):
        cam, r_cw = self._uav_pose(t)
        intr = (self.cfg.FX, self.cfg.FY, self.cfg.CX, self.cfg.CY)
        from geometry.camera_geometry import project_point
        dets = []
        for gt in gt_list:
            if self.rng.random() < self.cfg.IR_DROP_PROB:
                continue
            uv = project_point(np.array([gt["px"], gt["py"], 0.0]), cam, r_cw, intr)
            if uv is None:
                continue
            if not (0 <= uv[0] < self.cfg.IMAGE_WIDTH and 0 <= uv[1] < self.cfg.IMAGE_HEIGHT):
                continue
            c = uv + self.rng.normal(0, self.cfg.IR_PIXEL_NOISE, size=2)
            w, h = 36.0, 22.0
            bbox = [float(c[0]-w/2), float(c[1]-h/2), float(w), float(h)]
            dets.append({"timestamp": t, "bbox": bbox, "confidence": float(self.rng.uniform(0.75,0.995)),
                         "vessel_id": gt["vessel_id"]})
        return dets, {"cam_pos": cam, "r_cw": r_cw}
