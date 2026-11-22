import glob
import json

import cv2
import numpy as np
import pandas as pd

from clip_config import CLIP, out

Hs = np.load(out("pitch_mapping") / "homography_all_frames.npy")
df = pd.read_csv(out("player_tracking") / "tracking_data_botsort_s1280.csv")
akeys = sorted(int(json.load(open(p))["frame"])
               for p in glob.glob(str(out("pitch_mapping") / "homography_frame_*.json")))
total = len(Hs)


def nearest(f):
    return min(akeys, key=lambda k: abs(k - f))


def pos(f):
    g = df[df.frame == f]
    p = cv2.perspectiveTransform(g[["x", "y"]].to_numpy(np.float32).reshape(-1, 1, 2),
                                 Hs[f - 1].astype(np.float64)).reshape(-1, 2)
    return dict(zip(g.track_id, map(tuple, p)))


def jumps(f):
    a, b = pos(f), pos(f + 1)
    return np.array([np.hypot(a[t][0] - b[t][0], a[t][1] - b[t][1]) for t in a if t in b])


print("CLIP:", CLIP, "| jangkar:", akeys)
for f in range(2, total - 1):
    if nearest(f) != nearest(f + 1):
        j, base = jumps(f), jumps(f - 1)
        if len(j) and len(base):
            print(f"Pergantian jangkar {nearest(f)} -> {nearest(f + 1)} di frame {f}/{f + 1}: "
                  f"lompat median {np.median(j):.2f} m, maks {j.max():.2f} m "
                  f"(pembanding tanpa pergantian: median {np.median(base):.2f} m)")