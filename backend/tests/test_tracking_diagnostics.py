from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
VIDEO_PATH = PROJECT_DIR / "datasets" / "test_videos" / "football_test.mp4"
CSV_IN = PROJECT_DIR / "outputs" / "player_tracking" / "tracking_data_botsort.csv"
CSV_OUT = PROJECT_DIR / "outputs" / "player_tracking" / "tracking_data_botsort_merged.csv"
OUT_DIR = PROJECT_DIR / "outputs" / "tracking_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_GAP = 45            # frame maksimum antara akhir track lama dan awal track baru
MIN_DIST_PX = 120       # jarak minimum yang selalu diizinkan
DIST_FACTOR = 2.5       # atau 2.5 x tinggi box, mana yang lebih besar
SAMPLE_FRAMES = [1, 115, 230, 340]


def build_tracks(df):
    df = df.sort_values("frame")
    tracks = df.groupby("track_id").agg(
        start=("frame", "min"), end=("frame", "max"), n=("frame", "size"),
        h=("h", "median"), conf=("confidence", "mean"),
    )
    first = df.groupby("track_id")[["x", "y"]].first().rename(columns={"x": "sx", "y": "sy"})
    last = df.groupby("track_id")[["x", "y"]].last().rename(columns={"x": "ex", "y": "ey"})
    return tracks.join(first).join(last).sort_values("start")


def merge_tracks(tracks):
    """Sambung track yang berakhir lalu muncul lagi di dekatnya. Heuristik, bukan kebenaran."""
    root = {}
    used_as_pred = set()
    pairs = []
    for tid, t in tracks.iterrows():
        root[tid] = tid
        best = None
        for pid, p in tracks.iterrows():
            if pid == tid or pid in used_as_pred or p["end"] >= t["start"]:
                continue
            gap = t["start"] - p["end"]
            if gap > MAX_GAP:
                continue
            dist = float(np.hypot(t["sx"] - p["ex"], t["sy"] - p["ey"]))
            limit = max(MIN_DIST_PX, DIST_FACTOR * max(t["h"], p["h"]))
            if dist > limit:
                continue
            score = dist + 2.0 * gap
            if best is None or score < best[0]:
                best = (score, pid, gap, dist)
        if best is not None:
            _, pid, gap, dist = best
            used_as_pred.add(pid)
            root[tid] = root[pid]
            pairs.append((pid, tid, int(gap), dist))
    return root, pairs


def color_for(i):
    rng = np.random.RandomState(int(i) * 7 + 3)
    return tuple(int(c) for c in rng.randint(60, 255, 3))


def main():
    df = pd.read_csv(CSV_IN)
    df["h"] = df["y2"] - df["y1"]

    conc = df.groupby("frame")["track_id"].nunique()
    print(f"ID bersamaan per frame: min {conc.min()}, median {int(conc.median())}, max {conc.max()}")

    tracks = build_tracks(df)
    print(f"\nTrack pendek (<= 30 frame): {int((tracks['n'] <= 30).sum())}")
    print(tracks[tracks["n"] <= 30][["start", "end", "n", "sx", "sy", "ex", "ey"]]
          .round(0).astype(int).to_string())

    root, pairs = merge_tracks(tracks)
    print(f"\nPenyambungan: {len(pairs)} pasang")
    for pid, tid, gap, dist in pairs:
        print(f"  {pid:>3} -> {tid:>3}  gap {gap:>2} frame, jarak {dist:>5.0f} px")

    df["orig_track_id"] = df["track_id"]
    df["track_id"] = df["orig_track_id"].map(root)
    df = df.drop(columns=["h"])
    df.to_csv(CSV_OUT, index=False)

    lengths = df.groupby("track_id").size()
    print("\nSETELAH PENYAMBUNGAN")
    print(f"ID unik            : {df['track_id'].nunique()} (sebelumnya {tracks.shape[0]})")
    print(f"Track > 100 frame  : {int((lengths > 100).sum())}")
    print(f"Track <= 30 frame  : {int((lengths <= 30).sum())}")
    print("CSV:", CSV_OUT)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    for n in SAMPLE_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
        ok, frame = cap.read()
        if not ok:
            continue
        for r in df[df["frame"] == n].itertuples(index=False):
            c = color_for(r.track_id)
            cv2.rectangle(frame, (r.x1, r.y1), (r.x2, r.y2), c, 2)
            cv2.putText(frame, str(r.track_id), (r.x1, max(24, r.y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 3, cv2.LINE_AA)
        cv2.imwrite(str(OUT_DIR / f"merged_frame_{n}.png"), frame)
    cap.release()
    print("Gambar contoh:", OUT_DIR)


if __name__ == "__main__":
    main()