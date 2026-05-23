import numpy as np
from collections import Counter
from tracking.kalman import CVKalman
from association.matcher import euclidean_match


class SimpleTracker:
    def __init__(self, cfg, meas_noise, sensor_type):
        self.cfg = cfg
        self.kf = CVKalman(cfg.DT, cfg.PROCESS_NOISE)
        self.R = meas_noise
        self.sensor_type = sensor_type
        self.tracks = []
        self.next_id = 1

    def _majority(self, ids):
        ids = [i for i in ids if i is not None]
        return Counter(ids).most_common(1)[0][0] if ids else None

    def update(self, detections):
        for trk in self.tracks:
            trk["x"], trk["P"] = self.kf.predict(trk["x"], trk["P"])
            trk["age"] += 1
            trk["missed_count"] += 1
        tpos = np.array([t["x"][:2] for t in self.tracks]) if self.tracks else np.zeros((0, 2))
        mpos = np.array([d["position"][:2] for d in detections]) if detections else np.zeros((0, 2))
        pairs, _ = euclidean_match(tpos, mpos, self.cfg.NN_GATE, self.cfg.LARGE_COST)
        used = set()
        for ti, di in pairs:
            trk, det = self.tracks[ti], detections[di]
            trk["x"], trk["P"] = self.kf.update(trk["x"], trk["P"], det["position"][:2], self.R)
            trk["history"].append(trk["x"][:2].copy())
            trk["assigned_target_ids"].append(det.get("target_id"))
            trk["majority_target_id"] = self._majority(trk["assigned_target_ids"])
            trk["missed_count"] = 0
            used.add(di)
        for i, det in enumerate(detections):
            if i in used:
                continue
            x0 = np.array([det["position"][0], det["position"][1], 0.0, 0.0])
            trk = {
                "track_id": self.next_id,
                "sensor_type": self.sensor_type,
                "x": x0,
                "P": self.cfg.TRACK_INIT_COV.copy(),
                "history": [x0[:2].copy()],
                "assigned_target_ids": [det.get("target_id")],
                "majority_target_id": det.get("target_id"),
                "missed_count": 0,
                "age": 1,
            }
            self.next_id += 1
            self.tracks.append(trk)
        self.tracks = [t for t in self.tracks if t["missed_count"] <= self.cfg.TRACK_MAX_MISSED]
        return self.tracks
