import json

import cv2
import numpy as np
import pandas as pd

from clip_config import CLIP, VIDEO, out

OUT = out("pitch_mapping")
TRK_CSV = out("player_tracking") / "tracking_data_botsort_s1280.csv"
OUT_H = OUT / "homography_all_frames.npy"
L, W = 105.0, 68.0


def green_mask(frame, boxes):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (30, 35, 25), (95, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(m, (int(x1) - 12, int(y1) - 12), (int(x2) + 12, int(y2) + 12), 0, -1)
    return m


def pitch_lines():
    t = np.linspace(0, 2 * np.pi, 120)
    lines = [
        [(0, 0), (L, 0), (L, W), (0, W), (0, 0)],
        [(52.5, 0), (52.5, W)],
        [(52.5 + 9.15 * np.cos(a), W / 2 + 9.15 * np.sin(a)) for a in t],
        [(0, 13.84), (16.5, 13.84), (16.5, 54.16), (0, 54.16)],
        [(L, 13.84), (L - 16.5, 13.84), (L - 16.5, 54.16), (L, 54.16)],
        [(0, 24.84), (5.5, 24.84), (5.5, 43.16), (0, 43.16)],
        [(L, 24.84), (L - 5.5, 24.84), (L - 5.5, 43.16), (L, 43.16)],
    ]
    res = []
    for ln in lines:
        pts = []
        for a, b in zip(ln[:-1], ln[1:]):
            for s in np.linspace(0, 1, 40, endpoint=False):
                pts.append((a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])))
        pts.append(ln[-1])
        res.append(np.array(pts, np.float32))
    return res


def main():
    print("CLIP:", CLIP, flush=True)
    anchors = {}
    for p in OUT.glob("homography_frame_*.json"):
        d = json.loads(p.read_text())
        anchors[int(d["frame"])] = np.array(d["H"], np.float64)
    if 1 not in anchors:
        raise RuntimeError("homography_frame_1.json belum ada")
    akeys = sorted(anchors)
    print("Jangkar di frame:", akeys, flush=True)

    df = pd.read_csv(TRK_CSV)
    by_frame = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in df.groupby("frame")}

    cap = cv2.VideoCapture(str(VIDEO))
    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("Cannot read video")
    frames_all = [prev.copy()]          # hanya frame yang dibutuhkan disimpan nanti
    keep_idx = set()
    cum = {1: np.eye(3)}                # memetakan frame 1 -> frame n
    inl_log = []
    n = 1
    store = {1: prev.copy()}
    total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = set(akeys) | {1, total_est // 4, total_est // 2, 3 * total_est // 4, total_est}
    while True:
        ok, cur = cap.read()
        if not ok:
            break
        n += 1
        mask = green_mask(prev, by_frame.get(n - 1, []))
        g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(g0, 1500, 0.01, 8, mask=mask)
        M = None
        if p0 is not None and len(p0) >= 20:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(31, 31), maxLevel=3)
            good = st.ravel() == 1
            if good.sum() >= 20:
                M, inl = cv2.findHomography(p0[good], p1[good], cv2.RANSAC, 2.0)
                inl_log.append(int(inl.sum()) if inl is not None else 0)
        if M is None:
            M = np.eye(3)
            inl_log.append(0)
            print(f"PERINGATAN: frame {n} gagal, dianggap diam", flush=True)
        cum[n] = M @ cum[n - 1]
        if n in want:
            store[n] = cur.copy()
        prev = cur
        if n % 100 == 0:
            print(f"Frame {n}", flush=True)
    cap.release()
    total = n
    print(f"\nFrame: {total} | inlier antar-frame: min {min(inl_log)}, median {int(np.median(inl_log))}")
    print("Frame inlier < 50:", sum(1 for v in inl_log if v < 50))

    def H_from(a, f):
        # frame a -> piksel frame f -> lapangan, memakai jangkar a
        return anchors[a] @ cum[a] @ np.linalg.inv(cum[f])

    Hs = []
    for f in range(1, total + 1):
        a = min(akeys, key=lambda k: abs(k - f))
        Hs.append(H_from(a, f))
    np.save(OUT_H, np.stack(Hs))

    # drift: bandingkan klik jangkar dengan prediksi rantai dari jangkar lain
    grid = np.array([[x, y] for x in np.linspace(300, 1600, 5)
                     for y in np.linspace(300, 900, 4)], np.float32).reshape(-1, 1, 2)
    if len(akeys) > 1:
        print("\nDRIFT (selisih klik vs prediksi rantai, di tengah gambar):")
        for a in akeys:
            for b in akeys:
                if a == b:
                    continue
                pa = cv2.perspectiveTransform(grid, anchors[a]).reshape(-1, 2)
                pb = cv2.perspectiveTransform(grid, H_from(b, a)).reshape(-1, 2)
                d = np.linalg.norm(pa - pb, axis=1)
                print(f"  jangkar {b} -> frame {a}: rata-rata {d.mean():.2f} m, maks {d.max():.2f} m")

    lines = pitch_lines()
    for f in sorted(store):
        img = store[f]
        Hinv = np.linalg.inv(Hs[f - 1])
        for ln in lines:
            q = cv2.perspectiveTransform(ln.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
            cv2.polylines(img, [q.astype(np.int32).reshape(-1, 1, 2)], False, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"overlay_frame_{f}.png"), img)
    print("Overlay disimpan di:", OUT, "untuk frame", sorted(store))


if __name__ == "__main__":
    main()