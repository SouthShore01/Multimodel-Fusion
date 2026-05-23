import numpy as np
from association.matcher import euclidean_match


def bbox_bottom_center(b):
    return np.array([b[0] + b[2] / 2.0, b[1] + b[3]])


def fuse_visible_ir(visible, infrared, to_world_fn, gate, large_cost, wv=0.5, wi=0.5):
    v_world = [to_world_fn(bbox_bottom_center(d["bbox"]), "visible") for d in visible]
    i_world = [to_world_fn(bbox_bottom_center(d["bbox"]), "infrared") for d in infrared]
    v_idx = [i for i,p in enumerate(v_world) if p is not None]
    i_idx = [i for i,p in enumerate(i_world) if p is not None]
    v_arr = np.array([v_world[i] for i in v_idx]) if v_idx else np.zeros((0,2))
    i_arr = np.array([i_world[i] for i in i_idx]) if i_idx else np.zeros((0,2))
    pairs, _ = euclidean_match(v_arr, i_arr, gate, large_cost)
    fused, links = [], []
    used_v, used_i = set(), set()
    for a,b in pairs:
        vi, ii = v_idx[a], i_idx[b]
        z = wv * v_world[vi] + wi * i_world[ii]
        vid = visible[vi]["vessel_id"] if visible[vi]["vessel_id"] == infrared[ii]["vessel_id"] else visible[vi]["vessel_id"]
        fused.append({"z": z, "vessel_id": vid})
        links.append((v_world[vi], i_world[ii]))
        used_v.add(vi); used_i.add(ii)
    for vi, p in enumerate(v_world):
        if p is not None and vi not in used_v:
            fused.append({"z": p, "vessel_id": visible[vi]["vessel_id"]})
    for ii, p in enumerate(i_world):
        if p is not None and ii not in used_i:
            fused.append({"z": p, "vessel_id": infrared[ii]["vessel_id"]})
    return fused, links, v_world, i_world
