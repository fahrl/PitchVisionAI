import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[2]
VIDEO = P / "datasets" / "test_videos" / "football_test.mp4"
H1_JSON = P / "outputs" / "pitch_mapping" / "homography_frame_1.json"
TEAM_CSV = P / "outputs" / "team_classification" / "team_classification_botsort.csv"
OUT = P / "outputs" / "pitch_mapping"
OUT_CSV = OUT / "tracking_pitch_coords.csv"
OUT_H = OUT / "homography_all_frames.npy"
CHECK_FRAMES = [1, 86, 172, 258, 344]
L, W = 105.0, 68.0


def green_mask(frame, boxes):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (30, 35, 25), (95, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(m, (int(x1) - 12, int(y1) - 12), (int(x2) + 12, int(y2) + 12), 0, -1)
    return m


def pitch_lines():
    """Daftar polyline (meter) garis lapangan."""
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
    out = []
    for ln in lines:
        pts = []
        for a, b in zip(ln[:-1], ln[1:]):
            for s in np.linspace(0, 1, 40, endpoint=False):
                pts.append((a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])))
        pts.append(ln[-1])
        out.append(np.array(pts, np.float32))
    return out


def main():
    H1 = np.array(json.loads(H1_JSON.read_text())["H"], np.float64)
    df = pd.read_csv(TEAM_CSV)
    by_frame = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in df.groupby("frame")}

    cap = cv2.VideoCapture(str(VIDEO))
    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("Cannot read video")
    frames = {1: prev.copy()}
    Hs = {1: H1}
    M_1t = np.eye(3)
    inl_log = []
    n = 1
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
        M_1t = M @ M_1t
        Hs[n] = H1 @ np.linalg.inv(M_1t)
        if n in CHECK_FRAMES:
            frames[n] = cur.copy()
        prev = cur
        if n % 50 == 0:
            print(f"Frame {n}", flush=True)
    cap.release()

    total = n
    np.save(OUT_H, np.stack([Hs[i] for i in range(1, total + 1)]))
    print(f"\nInlier antar-frame: min {min(inl_log)}, median {int(np.median(inl_log))}")

    px = np.zeros(len(df))
    py = np.zeros(len(df))
    for f, idx in df.groupby("frame").indices.items():
        pts = df.iloc[idx][["x", "y"]].to_numpy(np.float32).reshape(-1, 1, 2)
        q = cv2.perspectiveTransform(pts, Hs[f]).reshape(-1, 2)
        px[idx], py[idx] = q[:, 0], q[:, 1]
    df["px"], df["py"] = px, py
    df.to_csv(OUT_CSV, index=False)

    pl = df[df.team.isin(["Team A", "Team B"])]
    inside = (pl.px > -3) & (pl.px < L + 3) & (pl.py > -3) & (pl.py < W + 3)
    print("\nPemain Team A/B di dalam lapangan (toleransi 3 m) per frame check:")
    for f in CHECK_FRAMES:
        s = pl[pl.frame == f]
        ins = ((s.px > -3) & (s.px < L + 3) & (s.py > -3) & (s.py < W + 3)).sum()
        print(f"  frame {f:>3}: {ins}/{len(s)}")
    print(f"Seluruh klip: {inside.mean() * 100:.1f}% di dalam lapangan")

    lines = pitch_lines()
    for f in CHECK_FRAMES:
        img = frames[f]
        Hinv = np.linalg.inv(Hs[f])
        for ln in lines:
            q = cv2.perspectiveTransform(ln.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
            cv2.polylines(img, [q.astype(np.int32).reshape(-1, 1, 2)], False, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"overlay_frame_{f}.png"), img)
    print("\nCSV    :", OUT_CSV)
    print("Overlay:", OUT)


if __name__ == "__main__":
    main()