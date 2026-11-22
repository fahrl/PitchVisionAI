from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[2]
CSV_IN = P / "outputs" / "pipeline" / "tracking_final.csv"
CSV_OUT = P / "outputs" / "pipeline" / "tracking_repaired.csv"
TEAMS = ("Team A", "Team B")
MAX_GAP = 20       # frame maksimum untuk menyambung/interpolasi
V_MAX = 0.4        # m/frame (sekitar 12 m/s pada 30 fps)
SLACK = 1.5        # toleransi posisi (m)
DUP_DIST = 0.5     # baris interpolasi dekat pemain asli dibuang (m)
COLS = ["x", "y", "x1", "y1", "x2", "y2", "px", "py"]


def report(d, label):
    total = int(d.frame.max())
    print(label)
    for t in TEAMS:
        n = d[d.team == t].groupby("frame").size().reindex(range(1, total + 1), fill_value=0)
        print(f"  {t}: tepat 10 pemain di {(n == 10).sum()}/{total} frame ({(n == 10).mean() * 100:.0f}%)")


def stitch(df):
    root = {int(t): int(t) for t in df.track_id.unique()}
    pairs = []
    for team in TEAMS:
        g = df[df.team == team].sort_values("frame")
        st = g.groupby("track_id").agg(s=("frame", "min"), e=("frame", "max"))
        first = g.groupby("track_id")[["px", "py"]].first()
        last = g.groupby("track_id")[["px", "py"]].last()
        used = set()
        for tid in st.sort_values("s").index:
            best = None
            for pid in st.index:
                if pid == tid or pid in used or st.loc[pid, "e"] >= st.loc[tid, "s"]:
                    continue
                gap = st.loc[tid, "s"] - st.loc[pid, "e"]
                if gap > MAX_GAP:
                    continue
                d = float(np.hypot(*(first.loc[tid].to_numpy() - last.loc[pid].to_numpy())))
                if d > V_MAX * gap + SLACK:
                    continue
                if best is None or d < best[0]:
                    best = (d, pid, gap)
            if best is not None:
                used.add(best[1])
                root[int(tid)] = root[int(best[1])]
                pairs.append((team, int(best[1]), int(tid), best[2], round(best[0], 2)))
    return root, pairs


def interpolate(df):
    new = []
    for tid, g in df[df.team.isin(TEAMS)].groupby("track_id"):
        g = g.sort_values("frame").reset_index(drop=True)
        fr = g.frame.to_numpy()
        for i in range(len(fr) - 1):
            a, b = int(fr[i]), int(fr[i + 1])
            if 1 < b - a <= MAX_GAP + 1:
                ra, rb = g.iloc[i], g.iloc[i + 1]
                for f in range(a + 1, b):
                    w = (f - a) / (b - a)
                    row = {c: float(ra[c]) + (float(rb[c]) - float(ra[c])) * w for c in COLS}
                    row.update(frame=f, track_id=int(tid), team=ra.team, interp=1,
                               confidence=0.0, timestamp=round((f - 1) / 30.0, 4))
                    new.append(row)
    return pd.DataFrame(new)


def main():
    df = pd.read_csv(CSV_IN)
    df["interp"] = 0
    report(df, "SEBELUM:")

    root, pairs = stitch(df)
    print(f"\nPenyambungan: {len(pairs)} pasang")
    for p in pairs:
        print(f"  {p[0]}: {p[1]} -> {p[2]} (celah {p[3]}f, jarak {p[4]} m)")
    df["orig_track_id"] = df.track_id
    df["track_id"] = df.track_id.map(root)

    new = interpolate(df)
    print(f"\nBaris interpolasi sebelum dedup: {len(new)}")
    if len(new):
        real = {k: g[["px", "py"]].to_numpy() for k, g in df[df.team.isin(TEAMS)].groupby(["frame", "team"])}
        keep = []
        for r in new.itertuples(index=False):
            pts = real.get((r.frame, r.team))
            keep.append(pts is None or np.hypot(pts[:, 0] - r.px, pts[:, 1] - r.py).min() >= DUP_DIST)
        new = new[np.array(keep)]
        print(f"Baris interpolasi dipakai: {len(new)}")
        df = pd.concat([df, new], ignore_index=True)
    df = df.sort_values(["frame", "track_id"]).reset_index(drop=True)
    df.to_csv(CSV_OUT, index=False)

    print()
    report(df, "SESUDAH:")
    n_team = int(df[df.team.isin(TEAMS)].shape[0])
    print(f"\nBaris hasil interpolasi: {int(df.interp.sum())}/{n_team} ({df.interp.sum() / n_team * 100:.1f}%)")
    print("CSV:", CSV_OUT)


if __name__ == "__main__":
    main()