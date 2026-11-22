import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from team_utils import jersey_lab, split_teams

PROJECT_DIR = Path(__file__).resolve().parents[2]
VIDEO_PATH = PROJECT_DIR / "datasets" / "test_videos" / "football_test.mp4"
TRACKING_CSV = PROJECT_DIR / "outputs" / "player_tracking" / "tracking_data_botsort.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "team_classification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "team_classification_botsort.csv"

EXCLUDE_IDS = {3, 24, 26}   # 3 = hakim garis dekat, 24 = wasit, 26 = hakim garis jauh
MIN_SAMPLES = 5
MIN_OUTLIER_DIST = 30.0
SAMPLE_FRAMES = [1, 115, 230, 340]
COLORS = {"Team A": (255, 100, 0), "Team B": (0, 80, 255),
          "Other": (160, 160, 160), "Unknown": (160, 160, 160)}


def main():
    df = pd.read_csv(TRACKING_CSV)
    print(f"Records: {len(df)} | tracks: {df['track_id'].nunique()}", flush=True)
    by_frame = {f: g for f, g in df.groupby("frame")}

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_PATH}")
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
                if r.track_id in EXCLUDE_IDS:
                    continue
                f = jersey_lab(frame, (r.x1, r.y1, r.x2, r.y2))
                if f is not None:
                    feats.setdefault(int(r.track_id), []).append(f)
        if n % 100 == 0:
            print(f"Pass 1: frame {n}", flush=True)
    cap.release()

    ids = [t for t, v in feats.items() if len(v) >= MIN_SAMPLES]
    X = np.array([np.median(feats[t], axis=0) for t in ids])
    print(f"Track valid: {len(ids)}", flush=True)

    labels, dist, light = split_teams(X, MIN_OUTLIER_DIST)
    assign = {}
    for t, lb in zip(ids, labels):
        assign[t] = "Other" if lb == -1 else ("Team A" if lb == light else "Team B")
    for t in EXCLUDE_IDS:
        assign[t] = "Other"
    for t in df["track_id"].unique():
        assign.setdefault(int(t), "Unknown")

    n_obs = df.groupby("track_id").size()
    print("\nTabel track:")
    print(f"{'id':>4} {'n':>4} {'L':>5} {'a':>5} {'b':>5} {'dist':>6}  team")
    for t, x, d in sorted(zip(ids, X, dist), key=lambda r: (assign[r[0]], r[2])):
        print(f"{t:>4} {n_obs[t]:>4} {x[0]:>5.0f} {x[1]:>5.0f} {x[2]:>5.0f} {d:>6.1f}  {assign[t]}")

    df["team"] = df["track_id"].map(assign)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nRINGKASAN TRACK")
    print(pd.Series(assign).value_counts().to_string())
    pf = df.groupby(["frame", "team"]).size().unstack(fill_value=0)
    print("\nPemain per frame:")
    print(pf.describe().loc[["min", "mean", "max"]].round(1).to_string())

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    for k in SAMPLE_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k - 1)
        ok, frame = cap.read()
        if not ok:
            continue
        for r in df[df["frame"] == k].itertuples(index=False):
            c = COLORS[r.team]
            cv2.rectangle(frame, (r.x1, r.y1), (r.x2, r.y2), c, 3)
            tag = {"Team A": "A", "Team B": "B"}.get(r.team, "X")
            cv2.putText(frame, f"{tag}{r.track_id}", (r.x1, max(24, r.y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 3, cv2.LINE_AA)
        cv2.imwrite(str(OUTPUT_DIR / f"team_frame_{k}.png"), frame)
    cap.release()
    print("\nCSV   :", OUTPUT_CSV)
    print("Gambar:", OUTPUT_DIR)


if __name__ == "__main__":
    main()