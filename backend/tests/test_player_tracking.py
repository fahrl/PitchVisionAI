from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parents[2]
VIDEO_PATH = PROJECT_DIR / "datasets" / "test_videos" / "football_test.mp4"
MODEL_PATH = PROJECT_DIR / "yolo26n.pt"
TRACKER_CFG = PROJECT_DIR / "backend" / "config" / "botsort_football.yaml"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "player_tracking"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "tracking_data_botsort_s1280.csv"
OUTPUT_VIDEO = OUTPUT_DIR / "football_tracking_botsort_s1280.mp4"

CONF, IMGSZ, IOU = 0.25, 1280, 0.70


def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    model = YOLO(str(MODEL_PATH))
    rows = []

    results = model.track(
        source=str(VIDEO_PATH), stream=True, persist=True,
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
    df.to_csv(OUTPUT_CSV, index=False)

    lengths = df.groupby("track_id").size()
    bins = pd.cut(lengths, [0, 10, 30, 100, 299, 10**6],
                  labels=["1-10", "11-30", "31-100", "101-299", "300+"])
    print("\nTRACKING SUMMARY")
    print(f"Frames with detections : {df['frame'].nunique()}")
    print(f"Total detections       : {len(df)}")
    print(f"Unique track IDs       : {df['track_id'].nunique()}")
    print(f"Avg detections/frame   : {len(df) / df['frame'].nunique():.2f}")
    print(f"Avg confidence         : {df['confidence'].mean():.4f}")
    print(f"Tracks > 100 frames    : {int((lengths > 100).sum())}")
    print("Track duration distribution:")
    print(bins.value_counts().sort_index().to_string())
    print("CSV  :", OUTPUT_CSV)
    print("Video:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()