from pathlib import Path

import cv2
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[2]
VIDEO = P / "datasets" / "test_videos" / "football_test.mp4"
CSV = P / "outputs" / "team_classification" / "team_classification_botsort.csv"
OUT = P / "outputs" / "team_classification" / "check_tracks.png"
CHECK = [10, 90, 73, 46, 26, 60, 7, 100]

df = pd.read_csv(CSV)

print("Rentang frame track yang diragukan:")
for t in CHECK:
    g = df[df.track_id == t]
    if len(g):
        print(f"  ID {t:>3}: frame {g.frame.min()}-{g.frame.max()}, n={len(g)}, {g.team.iloc[0]}")

b = df[df.team == "Team B"].groupby("frame").track_id.apply(sorted)
over = b[b.apply(len) >= 11]
print(f"\nFrame dengan Team B >= 11: {len(over)} dari {df.frame.nunique()}")
for f, ids in list(over.items())[:6]:
    print(f"  frame {f}: {ids}")
if len(over):
    from collections import Counter
    cnt = Counter(i for ids in over for i in ids)
    print("ID paling sering muncul saat Team B = 11:", cnt.most_common(6))

cap = cv2.VideoCapture(str(VIDEO))
tiles = []
for t in CHECK:
    g = df[df.track_id == t].sort_values("frame")
    if g.empty:
        continue
    r = g.iloc[len(g) // 2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame) - 1)
    ok, frame = cap.read()
    if not ok:
        continue
    pad = 25
    x1, y1 = max(0, int(r.x1) - pad), max(0, int(r.y1) - pad)
    x2, y2 = int(r.x2) + pad, int(r.y2) + pad
    crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(crop, (int(crop.shape[1] * 260 / crop.shape[0]), 260))
    cv2.putText(crop, f"ID {t} {r.team} f{int(r.frame)}", (4, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    tiles.append(crop)
cap.release()

W = sum(t.shape[1] for t in tiles) + 6 * (len(tiles) - 1)
sheet = np.zeros((260, W, 3), np.uint8)
x = 0
for t in tiles:
    sheet[:, x:x + t.shape[1]] = t
    x += t.shape[1] + 6
cv2.imwrite(str(OUT), sheet)
print("\nGambar:", OUT)