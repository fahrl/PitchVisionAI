import cv2
import numpy as np


def jersey_lab(frame, box):
    """Median Lab dari area torso, piksel rumput dibuang. None jika tidak valid."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 15:
        return None
    ty1, ty2 = y1 + int(bh * 0.20), y1 + int(bh * 0.60)
    tx1, tx2 = x1 + int(bw * 0.20), x2 - int(bw * 0.20)
    if tx2 <= tx1 or ty2 <= ty1:
        return None
    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    grass = (h >= 30) & (h <= 95) & (s > 35)
    valid = ~grass & (v > 25)
    if valid.sum() < 10:
        return None
    return np.median(lab[valid], axis=0).astype(float)


def split_teams(X, min_dist=30.0):
    """KMeans 2 cluster, outlier (median + 3 MAD) jadi -1. Return (labels, dist, light_cluster)."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
    lab0 = km.labels_
    d0 = np.linalg.norm(X - km.cluster_centers_[lab0], axis=1)
    thr = []
    for c in range(2):
        d = d0[lab0 == c]
        med = np.median(d)
        mad = np.median(np.abs(d - med)) * 1.4826
        thr.append(max(min_dist, med + 3 * mad))
    inlier = np.array([d0[i] <= thr[lab0[i]] for i in range(len(X))])

    centers = np.array([X[inlier & (lab0 == c)].mean(axis=0) for c in range(2)])
    dist_all = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    labels = dist_all.argmin(axis=1)
    dist = dist_all.min(axis=1)
    for i in range(len(X)):
        if dist[i] > thr[labels[i]]:
            labels[i] = -1
    light = int(np.argmax(centers[:, 0]))
    return labels, dist, light