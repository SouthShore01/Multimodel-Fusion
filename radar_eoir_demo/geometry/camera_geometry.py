import numpy as np


def look_at_rotation(camera_pos, target):
    fwd = target - camera_pos
    fwd = fwd / np.linalg.norm(fwd)
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_world)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    down = down / np.linalg.norm(down)
    return np.vstack([right, down, fwd])


def project_point(world_pt, cam_pos, r_cw, intrinsics):
    p_rel = world_pt - cam_pos
    p_cam = r_cw @ p_rel
    if p_cam[2] <= 1e-6:
        return None
    fx, fy, cx, cy = intrinsics
    u = fx * (p_cam[0] / p_cam[2]) + cx
    v = fy * (p_cam[1] / p_cam[2]) + cy
    return np.array([u, v])


def backproject_to_sea(pixel, cam_pos, r_cw, intrinsics):
    fx, fy, cx, cy = intrinsics
    x = (pixel[0] - cx) / fx
    y = (pixel[1] - cy) / fy
    ray_cam = np.array([x, y, 1.0])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)
    ray_world = r_cw.T @ ray_cam
    if abs(ray_world[2]) < 1e-8:
        return None
    s = -cam_pos[2] / ray_world[2]
    if s <= 0:
        return None
    p = cam_pos + s * ray_world
    return p[:2]
