import numpy as np
from datasets.base_adapter import BaseDatasetAdapter
from simulation.scene import SceneSimulator
from simulation.radar_simulator import RadarSimulator
from simulation.visible_camera_simulator import VisibleCameraSimulator
from simulation.infrared_camera_simulator import InfraredCameraSimulator
from geometry.coordinates import latlon_to_enu


class SyntheticDatasetAdapter(BaseDatasetAdapter):
    def __init__(self, data_root, config):
        super().__init__(data_root, config)
        self.rng = np.random.default_rng(config.SEED)
        self.scene = SceneSimulator(config.NUM_VESSELS, config.DT, self.rng)
        self.radar = RadarSimulator(config, self.rng)
        self.vis = VisibleCameraSimulator(config, self.rng)
        self.ir = InfraredCameraSimulator(config, self.rng)
        self.cache = {}
        self.load_metadata()

    def load_metadata(self):
        self.frames = self.config.NUM_FRAMES

    def num_frames(self): return self.frames
    def get_timestamp(self, frame_idx): return float(frame_idx * self.config.DT)
    def has_ground_truth_ids(self): return True
    def coordinate_frame(self): return "ENU"

    def _frame(self, frame_idx):
        if frame_idx not in self.cache:
            t = self.get_timestamp(frame_idx)
            gt = self.scene.step(t)
            radar = self.radar.simulate(gt, t)
            vis, vis_pose = self.vis.simulate(gt, t)
            ir, ir_pose = self.ir.simulate(gt, t)

            radar_dets = []
            for d in radar:
                x, y = latlon_to_enu(d["lat"], d["lon"], self.config.ORIGIN_LAT, self.config.ORIGIN_LON)
                radar_dets.append({"sensor_type": "radar", "timestamp": t, "position": np.array([x, y]),
                                   "velocity": None, "bbox": None, "confidence": 1.0,
                                   "target_id": d["vessel_id"], "raw": d})
            vis_dets = [{"sensor_type":"visible","timestamp":t,"position":None,"velocity":None,
                         "bbox":d["bbox"],"confidence":d["confidence"],"target_id":d["vessel_id"],"raw":d} for d in vis]
            ir_dets = [{"sensor_type":"infrared","timestamp":t,"position":None,"velocity":None,
                        "bbox":d["bbox"],"confidence":d["confidence"],"target_id":d["vessel_id"],"raw":d} for d in ir]
            gt_items = [{"target_id":g["vessel_id"],"position":np.array([g["px"], g["py"]])} for g in gt]
            self.cache[frame_idx] = {"radar": radar_dets, "visible": vis_dets, "infrared": ir_dets,
                                     "ground_truth": gt_items, "vis_pose": vis_pose, "ir_pose": ir_pose}
        return self.cache[frame_idx]

    def get_radar_detections(self, frame_idx): return self._frame(frame_idx)["radar"]
    def get_visible_detections(self, frame_idx): return self._frame(frame_idx)["visible"]
    def get_infrared_detections(self, frame_idx): return self._frame(frame_idx)["infrared"]
    def get_ground_truth(self, frame_idx): return self._frame(frame_idx)["ground_truth"]
    def get_pose(self, frame_idx, modality):
        f = self._frame(frame_idx)
        return f["vis_pose"] if modality == "visible" else f["ir_pose"]
