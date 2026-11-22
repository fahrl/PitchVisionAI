from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[2]
VIDEO = P / "datasets" / "test_videos" / "football_test.mp4"
CSV_IN = P / "outputs" / "pipeline" / "tracking_final.csv"
OUT = P / "outputs" / "formation_lines"
OUT.mkdir(parents=True, exist_ok=True)
OUT_VIDEO = OUT / "formation_pitch.mp4"
OUT_CSV = OUT / "formation_pitch.csv"

DIR = {"Team A": 1, "Team B": -1}   # arah serang di sumbu x lapangan
COLORS = {"Team A": (255, 100, 0), "Team B": (0, 80, 255)}
MIN_PLAYERS = 10   # evaluasi (CSV, angka paper): tepat 10 pemain
MIN_DRAW = 8       # tampilan video: minimal pemain untuk menggambar garis
HOLD = 0          # frame maksimal memakai garis terakhir yang tersedia
WINDOW = 31        # mode formasi per jendela (frame)
SMOOTH_POS = 31    # haluskan posisi per pemain (frame)
SEG = 90           # panjang segmen untuk tabel per 3 detik


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
    df = pd.read_csv(CSV_IN)
    df = df.sort_values(["track_id", "frame"])
    for c in ("px", "py"):
        df[c] = df.groupby("track_id")[c].transform(
            lambda s: s.rolling(SMOOTH_POS, center=True, min_periods=1).mean())
    df = df.sort_values(["frame", "track_id"])

    # --- evaluasi ketat: tepat MIN_PLAYERS pemain ---
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

    # --- tampilan video: lebih longgar, celah diisi garis terakhir ---
    disp = {}
    for f, g in df.groupby("frame"):
        for team, d in DIR.items():
            s = g[g.team == team].reset_index(drop=True)
            if len(s) != MIN_PLAYERS:
                continue
            disp[(f, team)] = {"s": s, "lines": split_lines(d * s.px.to_numpy()), "age": 0}
    for team in DIR:
        last = None
        for f in sorted(df.frame.unique()):
            if (f, team) in disp:
                last = (f, disp[(f, team)])
            elif last is not None and f - last[0] <= HOLD:
                disp[(f, team)] = {**last[1], "age": f - last[0]}

    out = pd.DataFrame([{"frame": f, "team": t, "n": len(r["s"]),
                         "raw": r["raw"], "smooth": r["smooth"]}
                        for (f, t), r in sorted(res.items())])
    out.to_csv(OUT_CSV, index=False)

    print("VERSI: tampilan longgar + garis ditahan, evaluasi tetap ketat")
    for t in DIR:
        o = out[out.team == t]
        print(t, "n rata-rata", round(o.n.mean(), 1),
              "| formasi smooth:", o.smooth.value_counts().head(4).to_dict())
    total = df.frame.nunique()
    for t in DIR:
        v = out[out.team == t].frame.nunique()
        print(f"Frame valid {t}: {v}/{total} ({v / total * 100:.0f}%)")
    drawn = {t: sum(1 for (f, tt) in disp if tt == t) for t in DIR}
    print("Frame bergaris di video:", {t: f"{drawn[t]}/{total}" for t in DIR})

    print("\nFormasi dominan per segmen 3 detik (bandingkan dengan label manual):")
    out["seg"] = (out.frame - 1) // SEG
    for t in DIR:
        row = []
        for sg, o in out[out.team == t].groupby("seg"):
            row.append(f"{sg * SEG + 1}-{sg * SEG + SEG}: {o.raw.value_counts().index[0]}")
        print(f"  {t}: " + " | ".join(row))

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(str(OUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        yt = 35
        g = df[df.frame == n]
        for team in DIR:
            c = COLORS[team]
            gt = g[g.team == team]
            for x, y in gt[["x", "y"]].to_numpy(int):
                cv2.circle(frame, (int(x), int(y)), 6, c, -1)
            r = disp.get((n, team))
            if r is None:
                cv2.putText(frame, f"{team}: n={len(gt)} (data kurang)",
                            (20, yt), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
            else:
                s = r["s"]
                for idx in r["lines"]:
                    pts = s.iloc[idx].sort_values("py")[["x", "y"]].to_numpy(np.int32)
                    if len(pts) >= 2:
                        cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, c, 3, cv2.LINE_AA)
                tag = "" if r["age"] == 0 else f" [ditahan {r['age']}f]"
                valid = res.get((n, team))
                name = valid["smooth"] if valid else "-".join(str(len(l)) for l in r["lines"])
                cv2.putText(frame, f"{team}: {name} (n={len(gt)}){tag}", (20, yt),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
            yt += 32
        wr.write(frame)
    cap.release()
    wr.release()
    print("\nVideo:", OUT_VIDEO)


if __name__ == "__main__":
    main()