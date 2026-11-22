import cv2
import numpy as np

# Referensi warna Lab (skala OpenCV) dari median track clip1. Dibekukan.
REF_LIGHT = np.array([248.0, 129.0, 128.0])   # Spanyol (putih)
REF_DARK = np.array([110.0, 131.0, 132.0])    # Kroasia (hitam)


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
    """Tugaskan ke pusat warna beku. Jarak > min_dist ke kedua pusat = -1 (Other).
    Return (labels, dist, light_cluster); cluster 0 = putih, 1 = hitam."""
    centers = np.stack([REF_LIGHT, REF_DARK])
    d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    labels = d.argmin(axis=1)
    dist = d.min(axis=1)
    labels = np.where(dist > min_dist, -1, labels)
    return labels, dist, 0