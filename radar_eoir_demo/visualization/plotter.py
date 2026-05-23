import matplotlib.pyplot as plt


def plot_visible_ir(v_world, i_world, links, path):
    plt.figure(figsize=(8, 6))
    for p in v_world:
        if p is not None: plt.scatter(p[0], p[1], c='b', s=12)
    for p in i_world:
        if p is not None: plt.scatter(p[0], p[1], c='r', s=12)
    for a, b in links:
        plt.plot([a[0], b[0]], [a[1], b[1]], 'k--', lw=0.8)
    plt.title('Visible-Infrared Matching (world plane)')
    plt.legend(['match link', 'visible', 'infrared'])
    plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.grid(True); plt.axis('equal'); plt.tight_layout(); plt.savefig(path); plt.close()


def plot_radar_eoir(radar_tracks, eoir_tracks, matches, path):
    plt.figure(figsize=(8, 6))
    for t in radar_tracks:
        if t['history']:
            hs = list(t['history']); xs=[p[0] for p in hs]; ys=[p[1] for p in hs]; plt.plot(xs, ys, 'g-')
    for t in eoir_tracks:
        if t['history']:
            hs = list(t['history']); xs=[p[0] for p in hs]; ys=[p[1] for p in hs]; plt.plot(xs, ys, 'm-')
    for i, j, _ in matches:
        rp = radar_tracks[i]['x'][:2]; ep = eoir_tracks[j]['x'][:2]
        plt.plot([rp[0], ep[0]], [rp[1], ep[1]], 'k--', lw=1.0)
    plt.title('Radar-EO/IR Track Matching')
    plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.grid(True); plt.axis('equal'); plt.tight_layout(); plt.savefig(path); plt.close()


def plot_association_result(gt_hist, radar_tracks, eoir_tracks, path):
    plt.figure(figsize=(9, 7))
    for vid, pts in gt_hist.items():
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; plt.plot(xs, ys, '--', lw=1.2)
    for t in radar_tracks:
        if t['history']:
            xs=[p[0] for p in t['history']]; ys=[p[1] for p in t['history']]; plt.plot(xs, ys, 'g-', alpha=0.8)
    for t in eoir_tracks:
        if t['history']:
            xs=[p[0] for p in t['history']]; ys=[p[1] for p in t['history']]; plt.plot(xs, ys, 'm-', alpha=0.8)
    plt.title('Ground truth vs Radar and EO/IR tracks')
    plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.grid(True); plt.axis('equal'); plt.tight_layout(); plt.savefig(path); plt.close()
