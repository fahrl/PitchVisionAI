import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import numpy as np
import pandas as pd

from clip_config import CLIP, VIDEO, out
from team_utils import jersey_lab, split_teams

CSV_IN = out("player_tracking") / "tracking_data_botsort_s1280.csv"
H_NPY = out("pitch_mapping") / "homography_all_frames.npy"
OUT = out("pipeline")
OUT_CSV = OUT / "tracking_final.csv"

L, W = 105.0, 68.0
TOL = 2.0
EDGE = 1.0
MIN_FRAMES = 10
MIN_INSIDE = 0.5
MIN_SAMPLES = 5
MIN_OUTLIER_DIST = 30.0
DUP_DIST = 0.5
COLORS = {"Team A": (255, 100, 0), "Team B": (0, 80, 255),
          "Other": (160, 160, 160), "Unknown": (0, 255, 255)}


def main():
    print("CLIP:", CLIP, flush=True)
    df = pd.read_csv(CSV_IN)
    Hs = np.load(H_NPY)
    print(f"Records: {len(df)} | tracks: {df.track_id.nunique()} | H: {len(Hs)} frame", flush=True)
    if df.frame.max() > len(Hs):
        raise RuntimeError("Jumlah frame homography lebih sedikit dari CSV")
    total = len(Hs)

    px = np.zeros(len(df))
    py = np.zeros(len(df))
    for f, idx in df.groupby("frame").indices.items():
        pts = df.iloc[idx][["x", "y"]].to_numpy(np.float32).reshape(-1, 1, 2)
        q = cv2.perspectiveTransform(pts, Hs[f - 1]).reshape(-1, 2)
        px[idx], py[idx] = q[:, 0], q[:, 1]
    df["px"], df["py"] = px, py
    df["inside"] = ((df.px > -TOL) & (df.px < L + TOL) &
                    (df.py > -TOL) & (df.py < W + TOL))

    st = df.groupby("track_id").agg(n=("frame", "size"), ins=("inside", "mean"),
                                    mpy=("py", "median"), spy=("py", "std"))
    edge_ref = ((st.mpy < EDGE) | (st.mpy > W + 0.5)) & (st.spy.fillna(0) < 0.5)
    keep = st[(st.n >= MIN_FRAMES) & (st.ins >= MIN_INSIDE) & ~edge_ref].index
    print(f"Track: {len(st)} -> {len(keep)} setelah filter", flush=True)
    df = df[df.track_id.isin(keep) & df.inside].copy()
    by_frame = {f: g for f, g in df.groupby("frame")}

    cap = cv2.VideoCapture(str(VIDEO))
    feats = {}
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        g = by_frame.get(n)
        if g is not None:
            for r in g.itertuples(index=False):
                f = jersey_lab(frame, (r.x1, r.y1, r.x2, r.y2))
                if f is not None:
                    feats.setdefault(int(r.track_id), []).append(f)
        if n % 100 == 0:
            print(f"Warna: frame {n}", flush=True)
    cap.release()

    ids = [t for t, v in feats.items() if len(v) >= MIN_SAMPLES]
    X = np.array([np.median(feats[t], axis=0) for t in ids])
    labels, dist, light = split_teams(X, MIN_OUTLIER_DIST)
    assign = {}
    for t, lb in zip(ids, labels):
        assign[t] = "Other" if lb == -1 else ("Team A" if lb == light else "Team B")
    for t in df.track_id.unique():
        assign.setdefault(int(t), "Unknown")
    df["team"] = df.track_id.map(assign)

    df = df.sort_values(["frame", "confidence"], ascending=[True, False])
    drop = []
    pl = df[df.team.isin(["Team A", "Team B"])]
    for (f, t), s in pl.groupby(["frame", "team"]):
        pts = s[["px", "py"]].to_numpy()
        chosen = []
        for i in range(len(s)):
            if all(np.hypot(*(pts[i] - pts[j])) >= DUP_DIST for j in chosen):
                chosen.append(i)
            else:
                drop.append(s.index[i])
    print(f"Kotak ganda dibuang: {len(drop)}", flush=True)
    df = df.drop(index=drop).sort_values(["frame", "track_id"])
    df.drop(columns=["inside"]).to_csv(OUT_CSV, index=False)

    n_obs = df.groupby("track_id").size()
    med = df.groupby("track_id")[["px", "py"]].median()
    row = {t: x for t, x in zip(ids, X)}
    print("\nTabel track:")
    print(f"{'id':>4} {'n':>4} {'L':>5} {'a':>5} {'b':>5} {'px':>6} {'py':>5}  team")
    present = sorted(int(t) for t in df.track_id.unique())
    for t in sorted(present, key=lambda k: (assign[k], k)):
        x = row[t] if t in row else [np.nan] * 3
        print(f"{t:>4} {n_obs[t]:>4} {x[0]:>5.0f} {x[1]:>5.0f} {x[2]:>5.0f} "
              f"{med.loc[t, 'px']:>6.1f} {med.loc[t, 'py']:>5.1f}  {assign[t]}")
    pf = df.groupby(["frame", "team"]).size().unstack(fill_value=0)
    print("\nPemain per frame:")
    print(pf.describe().loc[["min", "mean", "max"]].round(1).to_string())

    cap = cv2.VideoCapture(str(VIDEO))
    for k in sorted({1, total // 2, total}):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k - 1)
        ok, frame = cap.read()
        if not ok:
            continue
        for r in df[df.frame == k].itertuples(index=False):
            c = COLORS[r.team]
            cv2.rectangle(frame, (r.x1, r.y1), (r.x2, r.y2), c, 3)
            tag = {"Team A": "A", "Team B": "B"}.get(r.team, "X")
            cv2.putText(frame, f"{tag}{r.track_id}", (r.x1, max(24, r.y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 3, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"final_frame_{k}.png"), frame)
    cap.release()
    print("\nCSV   :", OUT_CSV)


if __name__ == "__main__":
    main()