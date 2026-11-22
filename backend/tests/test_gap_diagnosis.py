from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[2]
RAW = pd.read_csv(P / "outputs" / "player_tracking" / "tracking_data_botsort_s1280.csv")
FIN = pd.read_csv(P / "outputs" / "pipeline" / "tracking_final.csv")

print("Rata-rata kotak mentah/frame :", round(RAW.groupby("frame").size().mean(), 2))
print("Rata-rata setelah filter     :", round(FIN.groupby("frame").size().mean(), 2))

for team in ("Team A", "Team B"):
    n = FIN[FIN.team == team].groupby("frame").size().reindex(range(1, 345), fill_value=0)
    print(f"\n{team}: distribusi jumlah pemain per frame")
    print(n.value_counts().sort_index().to_string())
    bad = n[n != 10].index.to_numpy()
    if len(bad) == 0:
        continue
    runs = np.split(bad, np.where(np.diff(bad) != 1)[0] + 1)
    print(f"  {len(runs)} rentang frame bermasalah; 5 terpanjang:")
    for r in sorted(runs, key=len, reverse=True)[:5]:
        print(f"   frame {r[0]}-{r[-1]} ({len(r)}f), n={n[r].min()}..{n[r].max()}")

# track tim yang hilang-muncul: celah di dalam satu track
g = FIN[FIN.team.isin(["Team A", "Team B"])].sort_values(["track_id", "frame"])
gap = g.groupby("track_id").frame.diff().fillna(1) - 1
print("\nCelah di dalam track (frame hilang lalu muncul lagi dengan ID sama):",
      int((gap > 0).sum()), "kejadian, total", int(gap[gap > 0].sum()), "frame")

# ID baru yang muncul tepat setelah ID lain hilang (indikasi ID putus)
st = g.groupby("track_id").agg(s=("frame", "min"), e=("frame", "max"), team=("team", "first"))
cnt = 0
for tid, r in st.iterrows():
    prev = st[(st.team == r.team) & (st.e < r.s) & (r.s - st.e <= 20)]
    if len(prev):
        cnt += 1
print("Track yang mulai <=20 frame setelah track lain satu tim berakhir:", cnt, "dari", len(st))