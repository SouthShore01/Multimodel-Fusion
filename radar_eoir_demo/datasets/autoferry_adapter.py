import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from datasets.base_adapter import BaseDatasetAdapter


class AutoferryDatasetAdapter(BaseDatasetAdapter):
    def __init__(self, data_root, config):
        super().__init__(data_root, config)
        self.frames = []
        self.gt_has_id = False
        self.load_metadata()

    def load_metadata(self):
        root = Path(self.data_root)
        if not root.exists():
            raise FileNotFoundError(f"Autoferry data root not found: {self.data_root}")
        files = list(root.rglob("*.json")) + list(root.rglob("*.mat"))
        radar_f = [f for f in files if "radar" in f.name.lower()]
        eo_f = [f for f in files if "eo" in f.name.lower() or "visible" in f.name.lower()]
        ir_f = [f for f in files if "ir" in f.name.lower() or "infrared" in f.name.lower()]
        gt_f = [f for f in files if any(k in f.name.lower() for k in ["ground_truth", "gt", "truth"])]
        self.data = {"radar": self._load_any(radar_f), "eo": self._load_any(eo_f), "ir": self._load_any(ir_f), "gt": self._load_any(gt_f)}
        self.frames = self._build_frames()

    def _load_any(self, file_list):
        out = []
        for f in file_list:
            if f.suffix == ".json":
                out.append(json.loads(f.read_text()))
            elif f.suffix == ".mat":
                out.append(loadmat(f))
        return out

    def _to_det(self, d, sensor, ts):
        n = d.get("north", d.get("x", d.get("px", 0.0)))
        e = d.get("east", d.get("y", d.get("py", 0.0)))
        tid = d.get("target_id", d.get("id", None))
        return {"sensor_type": sensor, "timestamp": ts, "position": np.array([float(n), float(e)]),
                "velocity": None, "bbox": d.get("bbox", None), "confidence": float(d.get("confidence", 1.0)),
                "target_id": tid, "raw": d}

    def _extract_frames_from_json(self, obj, sensor):
        res = {}
        if isinstance(obj, dict) and "frames" in obj:
            for fr in obj["frames"]:
                idx = int(fr.get("frame_idx", fr.get("frame", 0)))
                ts = float(fr.get("timestamp", idx))
                res.setdefault(idx, {"ts": ts, "dets": []})
                for d in fr.get("detections", []):
                    res[idx]["dets"].append(self._to_det(d, sensor, ts))
        return res

    def _build_frames(self):
        frame_map = {}
        for sensor_key, s_name in [("radar", "radar"), ("eo", "visible"), ("ir", "infrared")]:
            for obj in self.data[sensor_key]:
                if isinstance(obj, dict):
                    parsed = self._extract_frames_from_json(obj, s_name)
                    for k, v in parsed.items():
                        frame_map.setdefault(k, {"timestamp": v["ts"], "radar": [], "visible": [], "infrared": [], "ground_truth": []})
                        frame_map[k][s_name].extend(v["dets"])
        # GT from json frames if present
        for obj in self.data["gt"]:
            if isinstance(obj, dict) and "frames" in obj:
                for fr in obj["frames"]:
                    idx = int(fr.get("frame_idx", fr.get("frame", 0)))
                    ts = float(fr.get("timestamp", idx))
                    frame_map.setdefault(idx, {"timestamp": ts, "radar": [], "visible": [], "infrared": [], "ground_truth": []})
                    for g in fr.get("targets", fr.get("ground_truth", [])):
                        tid = g.get("target_id", g.get("id", None))
                        if tid is not None: self.gt_has_id = True
                        n = g.get("north", g.get("x", 0.0)); e = g.get("east", g.get("y", 0.0))
                        frame_map[idx]["ground_truth"].append({"target_id": tid, "position": np.array([float(n), float(e)])})
        if not frame_map:
            raise RuntimeError("Autoferry adapter could not parse frames. Please map dataset-specific files to common Detection schema.")
        return [frame_map[k] for k in sorted(frame_map.keys())]

    def num_frames(self): return len(self.frames)
    def get_timestamp(self, frame_idx): return self.frames[frame_idx]["timestamp"]
    def get_radar_detections(self, frame_idx): return self.frames[frame_idx]["radar"]
    def get_visible_detections(self, frame_idx): return self.frames[frame_idx]["visible"]
    def get_infrared_detections(self, frame_idx): return self.frames[frame_idx]["infrared"]
    def get_ground_truth(self, frame_idx): return self.frames[frame_idx]["ground_truth"]
    def has_ground_truth_ids(self): return self.gt_has_id
    def coordinate_frame(self): return "NED_OWN_FIXED"
