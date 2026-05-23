import argparse
import os
import numpy as np
import config as cfg
from datasets.synthetic_adapter import SyntheticDatasetAdapter
from datasets.autoferry_adapter import AutoferryDatasetAdapter
from datasets.whut_msfvessel_adapter import WHUTMSFVesselAdapter
from datasets.mtdsp_adapter import MTDSPAdapter
from datasets.mit_marine_perception_adapter import MITMarinePerceptionAdapter
from geometry.camera_geometry import backproject_to_sea
from fusion.visible_ir_fusion import fuse_visible_ir
from fusion.radar_eoir_fusion import associate_radar_eoir
from tracking.radar_tracker import SimpleTracker
from tracking.eoir_tracker import EOIRTracker
from evaluation.metrics import compute_match_accuracy, compute_precision_recall_f1, compute_id_switches, compute_per_frame_metrics, save_metrics_csv, save_summary_json
from visualization.plotter import plot_visible_ir, plot_radar_eoir, plot_association_result


def get_adapter(name, data_root):
    m = {
        "synthetic": SyntheticDatasetAdapter,
        "autoferry": AutoferryDatasetAdapter,
        "whut_msfvessel": WHUTMSFVesselAdapter,
        "mtdsp": MTDSPAdapter,
        "mit_marine_perception": MITMarinePerceptionAdapter,
    }
    return m[name](data_root, cfg)


def nearest_gt_id(pos, gt, thr):
    if len(gt) == 0:
        return None
    d = [(g["target_id"], np.linalg.norm(pos[:2] - g["position"][:2])) for g in gt]
    gid, dist = min(d, key=lambda x: x[1])
    return gid if dist <= thr else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "autoferry", "whut_msfvessel", "mtdsp", "mit_marine_perception"])
    p.add_argument("--data-root", default="")
    args = p.parse_args()

    os.makedirs("outputs", exist_ok=True)
    adapter = get_adapter(args.dataset, args.data_root)
    radar_tracker = SimpleTracker(cfg, cfg.RADAR_R, sensor_type="radar")
    eoir_tracker = EOIRTracker(cfg, cfg.EOIR_R)

    stats = {"radar": 0, "visible": 0, "infrared": 0, "fused": 0}
    per_frame, pred_matches_all, match_history = [], [], []
    gt_hist = {}
    last_vis_ir = ([], [], [])
    last_re = []

    for fi in range(adapter.num_frames()):
        ts = adapter.get_timestamp(fi)
        radar = adapter.get_radar_detections(fi)
        vis = adapter.get_visible_detections(fi)
        ir = adapter.get_infrared_detections(fi)
        gt = adapter.get_ground_truth(fi)
        stats["radar"] += len(radar); stats["visible"] += len(vis); stats["infrared"] += len(ir)
        for g in gt:
            gt_hist.setdefault(g.get("target_id", "unknown"), []).append(tuple(g["position"][:2]))

        def to_world(pixel, modality):
            if args.dataset != "synthetic":
                return None
            pose = adapter.get_pose(fi, modality)
            intr = (cfg.FX, cfg.FY, cfg.CX, cfg.CY)
            return backproject_to_sea(pixel, pose["cam_pos"], pose["r_cw"], intr)

        if args.dataset == "synthetic":
            fused, links, v_world, i_world = fuse_visible_ir(vis, ir, to_world, cfg.VISIBLE_IR_DISTANCE_GATE, cfg.LARGE_COST, cfg.EOIR_W_VISIBLE, cfg.EOIR_W_IR)
            eoir_dets = [{"sensor_type": "eoir", "timestamp": ts, "position": f["z"], "confidence": 1.0, "target_id": f["vessel_id"], "raw": f} for f in fused]
            last_vis_ir = (v_world, i_world, links)
        else:
            eoir_dets = []
            for d in vis + ir:
                if d.get("position") is not None:
                    eoir_dets.append({"sensor_type": "eoir", "timestamp": ts, "position": d["position"][:2], "confidence": d.get("confidence", 1.0), "target_id": d.get("target_id"), "raw": d})
            links = []
        stats["fused"] += len(eoir_dets)

        r_tracks = radar_tracker.update(radar)
        e_tracks = eoir_tracker.update(eoir_dets)
        matches = associate_radar_eoir(r_tracks, e_tracks, cfg.EOIR_R, cfg.MATCH_GATE_THRESHOLD, cfg.LARGE_COST)
        last_re = matches

        gt_assignments = {}
        coobs = set()
        radar_ids = {d.get("target_id") for d in radar if d.get("target_id") is not None}
        eoir_ids = {d.get("target_id") for d in eoir_dets if d.get("target_id") is not None}
        coobs = radar_ids.intersection(eoir_ids)

        for m in matches:
            rt, et = r_tracks[m["radar_idx"]], e_tracks[m["eoir_idx"]]
            rid = rt["majority_target_id"]
            eid = et["majority_target_id"]
            if rid is None and gt:
                rid = nearest_gt_id(rt["x"][:2], gt, cfg.EVAL_GT_DISTANCE_THRESHOLD)
            if eid is None and gt:
                eid = nearest_gt_id(et["x"][:2], gt, cfg.EVAL_GT_DISTANCE_THRESHOLD)
            label = "unknown" if (rid is None or eid is None) else ("correct" if rid == eid else "false")
            gt_assignments[(rt["track_id"], et["track_id"])] = label
            pred_matches_all.append({"frame": fi, "radar_track_id": rt["track_id"], "eoir_track_id": et["track_id"], "cost": m["cost"]})
            match_history.append({"radar_track_id": rt["track_id"], "eoir_track_id": et["track_id"], "gt_id": rid if rid == eid else None})

        frame_acc = compute_match_accuracy([{"radar_track_id":k[0],"eoir_track_id":k[1]} for k in gt_assignments.keys()], gt_assignments)
        matched_ids = {r_tracks[m["radar_idx"]]["majority_target_id"] for m in matches}.intersection({e_tracks[m["eoir_idx"]]["majority_target_id"] for m in matches})
        missed = len([x for x in coobs if x not in matched_ids]) if coobs else 0
        avg_cost = float(np.mean([m["cost"] for m in matches])) if matches else 0.0
        per_frame.append(compute_per_frame_metrics(fi, ts, frame_acc["predicted_matches"], frame_acc["correct_matches"], frame_acc["false_matches"], missed, avg_cost))

    total_correct = sum(r["correct_matches"] for r in per_frame)
    total_false = sum(r["false_matches"] for r in per_frame)
    total_missed = sum(r["missed_matches"] for r in per_frame)
    total_pred = len(pred_matches_all)
    precision, recall, f1 = compute_precision_recall_f1(total_correct, total_false, total_missed)
    acc = total_correct / max(total_pred, 1)
    avg_cost = float(np.mean([m["cost"] for m in pred_matches_all])) if pred_matches_all else 0.0
    ids = compute_id_switches(match_history)

    if not any(adapter.get_ground_truth(i) for i in range(adapter.num_frames())):
        print("Cannot compute real matching accuracy because ground truth is unavailable.")

    summary = {
        "dataset": args.dataset, "frames": adapter.num_frames(),
        "num_radar_detections": stats["radar"], "num_visible_detections": stats["visible"], "num_infrared_detections": stats["infrared"],
        "num_eoir_fused_observations": stats["fused"], "predicted_matches": total_pred, "correct_matches": total_correct,
        "false_matches": total_false, "missed_matches": total_missed, "accuracy": acc, "precision": precision,
        "recall": recall, "f1": f1, "avg_cost": avg_cost, "id_switches": ids,
    }

    save_metrics_csv(f"outputs/results_{args.dataset}.csv", per_frame)
    save_summary_json(f"outputs/matching_summary_{args.dataset}.json", summary)

    print(f"dataset name: {args.dataset}")
    print(f"number of frames: {summary['frames']}")
    print(f"number of radar detections: {summary['num_radar_detections']}")
    print(f"number of visible detections: {summary['num_visible_detections']}")
    print(f"number of infrared detections: {summary['num_infrared_detections']}")
    print(f"number of EO/IR fused observations: {summary['num_eoir_fused_observations']}")
    print(f"total predicted radar–EO/IR matches: {summary['predicted_matches']}")
    print(f"correct matches: {summary['correct_matches']}")
    print(f"false matches: {summary['false_matches']}")
    print(f"missed matches: {summary['missed_matches']}")
    print(f"matching accuracy: {summary['accuracy']:.3f}")
    print(f"precision: {summary['precision']:.3f}")
    print(f"recall: {summary['recall']:.3f}")
    print(f"F1 score: {summary['f1']:.3f}")
    print(f"average matching cost: {summary['avg_cost']:.3f}")
    print(f"ID switches if available: {summary['id_switches']}")

    plot_visible_ir(*last_vis_ir, "outputs/visible_ir_matching.png")
    plot_radar_eoir(radar_tracker.tracks, eoir_tracker.tracks, [(m['radar_idx'], m['eoir_idx'], m['cost']) for m in last_re], "outputs/radar_eoir_matching.png")
    plot_association_result(gt_hist, radar_tracker.tracks, eoir_tracker.tracks, "outputs/association_result.png")


if __name__ == "__main__":
    main()
