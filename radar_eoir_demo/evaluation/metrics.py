import csv
import json


def compute_match_accuracy(pred_matches, gt_assignments):
    correct = false = unknown = 0
    for m in pred_matches:
        key = (m["radar_track_id"], m["eoir_track_id"])
        label = gt_assignments.get(key, "unknown")
        if label == "correct":
            correct += 1
        elif label == "false":
            false += 1
        else:
            unknown += 1
    total = len(pred_matches)
    accuracy = correct / max(total, 1)
    return {"predicted_matches": total, "correct_matches": correct, "false_matches": false, "unknown_gt_matches": unknown, "accuracy": accuracy}


def compute_precision_recall_f1(correct_matches, false_matches, missed_matches):
    precision = correct_matches / max(correct_matches + false_matches, 1)
    recall = correct_matches / max(correct_matches + missed_matches, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def compute_id_switches(match_history):
    last = {}
    switches = 0
    for item in match_history:
        rid, eid, gid = item["radar_track_id"], item["eoir_track_id"], item.get("gt_id")
        if gid is None:
            continue
        k = (rid, eid)
        if k in last and last[k] != gid:
            switches += 1
        last[k] = gid
    return switches


def compute_per_frame_metrics(frame_idx, timestamp, predicted_matches, correct_matches, false_matches, missed_matches, avg_cost):
    precision, recall, f1 = compute_precision_recall_f1(correct_matches, false_matches, missed_matches)
    acc = correct_matches / max(predicted_matches, 1)
    return {
        "frame": frame_idx,
        "timestamp": timestamp,
        "predicted_matches": predicted_matches,
        "correct_matches": correct_matches,
        "false_matches": false_matches,
        "missed_matches": missed_matches,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_cost": avg_cost,
    }


def save_metrics_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_summary_json(path, summary):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
