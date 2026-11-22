from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from clip_config import CLIP, VIDEO, out

P = Path(__file__).resolve().parents[2]
MODEL_PATH = P / "yolo26s.pt"
TRACKER_CFG = P / "backend" / "config" / "botsort_football.yaml"
OUT = out("player_tracking")
OUT_CSV = OUT / "tracking_data_botsort_s1280.csv"
OUT_VIDEO = OUT / "football_tracking_botsort_s1280.mp4"

CONF, IMGSZ, IOU = 0.25, 1280, 0.70   # sama persis dengan clip1


def main():
    print("CLIP:", CLIP, "|", VIDEO, flush=True)
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    writer = cv2.VideoWriter(str(OUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    model = YOLO(str(MODEL_PATH))
    rows = []

    results = model.track(
        source=str(VIDEO), stream=True, persist=True,
        tracker=str(TRACKER_CFG), classes=[0], conf=CONF, imgsz=IMGSZ, iou=IOU,
        device="cpu", verbose=False,
    )

    for i, r in enumerate(results, start=1):
        frame = r.orig_img.copy()
        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, confs):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                rows.append({
                    "frame": i, "timestamp": round((i - 1) / fps, 4), "track_id": int(tid),
                    "x": (x1 + x2) // 2, "y": y2,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": round(float(c), 4),
                })
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, str(tid), (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        writer.write(frame)
        if i % 25 == 0:
            print(f"Frame {i}/{total}", flush=True)

    writer.release()
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    lengths = df.groupby("track_id").size()
    print("\nTRACKING SUMMARY")
    print(f"Frames with detections : {df['frame'].nunique()}")
    print(f"Total detections       : {len(df)}")
    print(f"Unique track IDs       : {df['track_id'].nunique()}")
    print(f"Avg detections/frame   : {len(df) / df['frame'].nunique():.2f}")
    print(f"Tracks > 100 frames    : {int((lengths > 100).sum())}")
    print("CSV  :", OUT_CSV)


if __name__ == "__main__":
    main()