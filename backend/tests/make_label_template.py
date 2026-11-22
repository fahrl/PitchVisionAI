import sys

import cv2
import pandas as pd

from clip_config import P

CLIP = sys.argv[1] if len(sys.argv) > 1 else "clip3"
SEG = 90

video = P / "datasets" / "test_videos" / (
    "football_test.mp4" if CLIP == "clip1" else f"football_test_{CLIP}.mp4")
cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise SystemExit(f"Tidak bisa membuka {video}")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

eval_dir = P / "outputs" / "evaluation"
eval_dir.mkdir(parents=True, exist_ok=True)
path = eval_dir / f"segment_labels_{CLIP}.csv"
if path.exists():
    raise SystemExit(f"{path} sudah ada, tidak ditimpa. Hapus dulu jika memang ingin dibuat ulang.")

nseg = (total + SEG - 1) // SEG
rows = []
for s in range(nseg):
    start, end = s * SEG + 1, min((s + 1) * SEG, total)
    for t in ("Team A", "Team B"):
        rows.append({"clip": CLIP, "seg": s, "frames": f"{start}-{end}",
                     "team": t, "label_manual": ""})
pd.DataFrame(rows).to_csv(path, index=False)
print(f"{CLIP}: {total} frame, {nseg} segmen, {len(rows)} baris")
print("Dibuat:", path)
print("Segmen terakhir hanya", total - (nseg - 1) * SEG, "frame, kosongkan jika kurang dari 30.")