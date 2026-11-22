from collections import Counter

import cv2
import numpy as np
import pandas as pd

from clip_config import CLIP, VIDEO, out

CSV_IN = out("pipeline") / "tracking_repaired.csv"
OUT = out("formation_lines")
OUT_VIDEO = OUT / "formation_pitch.mp4"
OUT_CSV = OUT / "formation_pitch.csv"

# Arah serang di sumbu x lapangan. CEK di awal klip dan ubah SEBELUM menjalankan
# jika Spanyol (Team A) menyerang ke arah sebaliknya.
DIR = {"Team A": 1, "Team B": -1}
COLORS = {"Team A": (255, 100, 0), "Team B": (0, 80, 255)}
MIN_PLAYERS = 10   # tepat 10 pemain luar
WINDOW = 31
SMOOTH_POS = 31
SEG = 90


def split_lines(depth):
    n = len(depth)
    order = np.argsort(depth)
    d = depth[order]
    best = None
    for i in range(2, n - 2):
        for j in range(i + 2, n):
            sse = 0.0
            for seg in (d[:i], d[i:j], d[j:]):
                sse += float(((seg - seg.mean()) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, i, j)
    if best is None:
        return [np.arange(n), np.array([], int), np.array([], int)]
    _, i, j = best
    sl = np.zeros(n, int)
    sl[i:] += 1
    sl[j:] += 1
    labels = np.empty(n, int)
    labels[order] = sl
    return [np.where(labels == k)[0] for k in range(3)]


def main():
    print("CLIP:", CLIP, flush=True)
    df = pd.read_csv(CSV_IN)
    df = df.sort_values(["track_id", "frame"])
    for c in ("px", "py"):
        df[c] = df.groupby("track_id")[c].transform(
            lambda s: s.rolling(SMOOTH_POS, center=True, min_periods=1).mean())
    df = df.sort_values(["frame", "track_id"])

    res = {}
    for f, g in df.groupby("frame"):
        for team, d in DIR.items():
            s = g[g.team == team].reset_index(drop=True)
            if len(s) != MIN_PLAYERS:
                continue
            lines = split_lines(d * s.px.to_numpy())
            res[(f, team)] = {"s": s, "lines": lines,
                              "raw": "-".join(str(len(l)) for l in lines)}
    half = WINDOW // 2
    for team in DIR:
        raws = {f: r["raw"] for (f, t), r in res.items() if t == team}
        for f in raws:
            win = [raws[w] for w in range(f - half, f + half + 1) if w in raws]
            res[(f, team)]["smooth"] = Counter(win).most_common(1)[0][0]

    out_df = pd.DataFrame([{"frame": f, "team": t, "n": len(r["s"]),
                            "raw": r["raw"], "smooth": r["smooth"]}
                           for (f, t), r in sorted(res.items())])
    out_df.to_csv(OUT_CSV, index=False)

    total = int(df.frame.max())
    for t in DIR:
        o = out_df[out_df.team == t]
        print(t, "| formasi smooth:", o.smooth.value_counts().head(4).to_dict())
        print(f"Frame valid {t}: {o.frame.nunique()}/{total} ({o.frame.nunique() / total * 100:.0f}%)")

    print("\nFormasi dominan per segmen 3 detik:")
    out_df["seg"] = (out_df.frame - 1) // SEG
    for t in DIR:
        row = []
        for sg, o in out_df[out_df.team == t].groupby("seg"):
            row.append(f"{sg * SEG + 1}-{sg * SEG + SEG}: {o.raw.value_counts().index[0]} ({len(o)}f)")
        print(f"  {t}: " + " | ".join(row))

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(str(OUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    by_frame = {f: g for f, g in df.groupby("frame")}
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        yt = 35
        g = by_frame.get(n)
        for team in DIR:
            c = COLORS[team]
            gt = g[g.team == team] if g is not None else pd.DataFrame(columns=["x", "y"])
            for x, y in gt[["x", "y"]].to_numpy(int):
                cv2.circle(frame, (int(x), int(y)), 6, c, -1)
            r = res.get((n, team))
            if r is None:
                cv2.putText(frame, f"{team}: n={len(gt)} (tidak valid)", (20, yt),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
            else:
                s = r["s"]
                for idx in r["lines"]:
                    pts = s.iloc[idx].sort_values("py")[["x", "y"]].to_numpy(np.int32)
                    if len(pts) >= 2:
                        cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, c, 3, cv2.LINE_AA)
                cv2.putText(frame, f"{team}: {r['smooth']} (n={len(s)})", (20, yt),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
            yt += 32
        wr.write(frame)
    cap.release()
    wr.release()
    print("\nCSV  :", OUT_CSV)
    print("Video:", OUT_VIDEO)


if __name__ == "__main__":
    main()