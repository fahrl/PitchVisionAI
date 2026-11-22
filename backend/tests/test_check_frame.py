from pathlib import Path

import cv2
import pandas as pd

P = Path(__file__).resolve().parents[2]
VIDEO = P / "datasets" / "test_videos" / "football_test.mp4"
CSV = P / "outputs" / "player_tracking" / "tracking_data_botsort_s1280.csv"
OUT = P / "outputs" / "player_tracking"
FRAMES = [335]

df = pd.read_csv(CSV)
cap = cv2.VideoCapture(str(VIDEO))
for n in FRAMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    ok, frame = cap.read()
    if not ok:
        continue
    g = df[df.frame == n]
    print(f"frame {n}: {len(g)} kotak")
    for r in g.itertuples(index=False):
        cv2.rectangle(frame, (r.x1, r.y1), (r.x2, r.y2), (0, 255, 0), 2)
        cv2.putText(frame, str(r.track_id), (r.x1, max(24, r.y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
    cv2.imwrite(str(OUT / f"check_s1280_frame_{n}.png"), frame)
cap.release()