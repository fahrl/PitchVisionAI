
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# PITCHVISION AI - AUTOMATIC TEAM IDENTIFICATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

VIDEO_PATH = PROJECT_DIR / "datasets" / "test_videos" / "football_test.mp4"
TRACKING_CSV = PROJECT_DIR / "outputs" / "player_tracking" / "tracking_data_v3.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "team_classification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "football_team_classification.mp4"
OUTPUT_CSV = OUTPUT_DIR / "team_classification.csv"
OUTPUT_JSON = OUTPUT_DIR / "team_colors.json"

MIN_VALID_SAMPLES = 5
MIN_TRACK_OBSERVATIONS = 5
RANDOM_STATE = 42


def find_column(df, candidates):
    """Find a column using alternative possible names."""
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(
        f"Column not found. Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def load_tracking_data():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    if not TRACKING_CSV.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {TRACKING_CSV}")

    df = pd.read_csv(TRACKING_CSV)

    columns = {
        "frame": find_column(df, ["frame", "frame_number"]),
        "track": find_column(df, ["track_id", "tracker_id", "id"]),
        "x1": find_column(df, ["x1", "xmin", "left"]),
        "y1": find_column(df, ["y1", "ymin", "top"]),
        "x2": find_column(df, ["x2", "xmax", "right"]),
        "y2": find_column(df, ["y2", "ymax", "bottom"]),
    }

    df = df.rename(columns={v: k for k, v in columns.items()})

    for col in ["frame", "track", "x1", "y1", "x2", "y2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["frame", "track", "x1", "y1", "x2", "y2"])
    df["frame"] = df["frame"].astype(int)
    df["track"] = df["track"].astype(int)

    return df


def extract_jersey_feature(frame, box):
    """
    Extract jersey color from the upper-body region.
    Returns a compact HSV + Lab feature vector.
    """
    height, width = frame.shape[:2]

    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w < 8 or box_h < 15:
        return None

    # Focus on torso: exclude head and legs.
    torso_y1 = y1 + int(box_h * 0.20)
    torso_y2 = y1 + int(box_h * 0.65)
    torso_x1 = x1 + int(box_w * 0.20)
    torso_x2 = x2 - int(box_w * 0.20)

    if torso_x2 <= torso_x1 or torso_y2 <= torso_y1:
        return None

    crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

    pixels_hsv = hsv.reshape(-1, 3)
    pixels_lab = lab.reshape(-1, 3)

    h = pixels_hsv[:, 0]
    s = pixels_hsv[:, 1]
    v = pixels_hsv[:, 2]

    # Remove green pitch and very dark/bright background pixels.
    valid = (
        ~((h >= 30) & (h <= 95) & (s > 35))
        & (v > 25)
        & (v < 250)
    )

    selected_hsv = pixels_hsv[valid]
    selected_lab = pixels_lab[valid]

    if len(selected_hsv) < 10:
        return None

    hsv_median = np.median(selected_hsv, axis=0)
    lab_median = np.median(selected_lab, axis=0)

    return np.concatenate([hsv_median, lab_median]).astype(float)


def main():
    print("=" * 60)
    print("PITCHVISION AI - AUTOMATIC TEAM IDENTIFICATION")
    print("=" * 60)

    df = load_tracking_data()

    print(f"Tracking records : {len(df)}")
    print(f"Unique tracks    : {df['track'].nunique()}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Cannot open football_test.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution       : {width}x{height}")
    print(f"FPS              : {fps:.2f}")
    print(f"Total frames     : {total_frames}")

    # Group tracking records by frame.
    frame_records = {
        frame: group
        for frame, group in df.groupby("frame")
    }

    # Collect jersey features per track.
    track_features = {}

    print("\nPASS 1: Extracting jersey colors...")

    frame_number = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_number += 1

        records = frame_records.get(frame_number)

        if records is not None:
            for row in records.itertuples(index=False):
                feature = extract_jersey_feature(
                    frame,
                    (row.x1, row.y1, row.x2, row.y2)
                )

                if feature is not None:
                    track_id = int(row.track)
                    track_features.setdefault(track_id, []).append(feature)

        if frame_number % 50 == 0:
            print(f"Processed {frame_number}/{total_frames}")

    cap.release()

    # Aggregate features using median to reduce noise.
    valid_tracks = {}
    observation_counts = df.groupby("track").size()

    for track_id, features in track_features.items():
        if (
            len(features) >= MIN_VALID_SAMPLES
            and observation_counts.get(track_id, 0) >= MIN_TRACK_OBSERVATIONS
        ):
            valid_tracks[track_id] = np.median(
                np.array(features),
                axis=0
            )

    print("\nValid tracks:", len(valid_tracks))

    if len(valid_tracks) < 2:
        raise RuntimeError(
            "Not enough valid tracks for team classification."
        )

    track_ids = list(valid_tracks.keys())
    feature_matrix = np.array(
        [valid_tracks[track_id] for track_id in track_ids]
    )

    # Normalize features before clustering.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    print("\nRunning K-Means clustering...")

    kmeans = KMeans(
        n_clusters=2,
        random_state=RANDOM_STATE,
        n_init=20
    )

    cluster_labels = kmeans.fit_predict(scaled_features)

    # Identify lighter and darker jersey clusters using Lab L.
    cluster_brightness = {}

    for cluster in range(2):
        members = feature_matrix[cluster_labels == cluster]
        cluster_brightness[cluster] = float(np.median(members[:, 3]))

    light_cluster = max(cluster_brightness, key=cluster_brightness.get)
    dark_cluster = 1 - light_cluster

    assignments = {}

    for track_id, cluster in zip(track_ids, cluster_labels):
        assignments[track_id] = (
            "Team A" if cluster == light_cluster else "Team B"
        )

    # Tracks without reliable jersey features remain unknown.
    for track_id in df["track"].unique():
        assignments.setdefault(int(track_id), "Unknown")

    counts = pd.Series(list(assignments.values())).value_counts()

    print("\nTEAM CLASSIFICATION SUMMARY")
    print("-" * 45)
    print("Team A  :", counts.get("Team A", 0))
    print("Team B  :", counts.get("Team B", 0))
    print("Unknown :", counts.get("Unknown", 0))

    # Render classified video.
    print("\nPASS 2: Rendering classified video...")

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create output video.")

    colors = {
        "Team A": (255, 100, 0),
        "Team B": (0, 80, 255),
        "Unknown": (160, 160, 160),
    }

    frame_number = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_number += 1
        records = frame_records.get(frame_number)

        if records is not None:
            for row in records.itertuples(index=False):
                track_id = int(row.track)
                team = assignments.get(track_id, "Unknown")
                color = colors[team]

                x1, y1, x2, y2 = map(
                    int,
                    (row.x1, row.y1, row.x2, row.y2)
                )

                # Draw bounding box.
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Draw foot position.
                center_x = (x1 + x2) // 2
                bottom_y = y2

                cv2.circle(
                    frame,
                    (center_x, bottom_y),
                    6,
                    color,
                    -1
                )

                label = f"{team} | ID {track_id}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA
                )

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        writer.write(frame)

        if frame_number % 50 == 0:
            print(f"Rendered {frame_number}/{total_frames}")

    cap.release()
    writer.release()

    # Save classification CSV.
    df["team"] = df["track"].map(assignments)
    df.to_csv(OUTPUT_CSV, index=False)

    # Save team color information.
    color_info = {
        "Team A": {
            "description": "Lighter jersey cluster",
            "median_lab_brightness": cluster_brightness[light_cluster]
        },
        "Team B": {
            "description": "Darker jersey cluster",
            "median_lab_brightness": cluster_brightness[dark_cluster]
        },
        "method": "HSV and Lab torso features + K-Means"
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(color_info, file, indent=4)

    print("\n" + "=" * 60)
    print("TEAM IDENTIFICATION COMPLETED")
    print("=" * 60)
    print("Video :", OUTPUT_VIDEO)
    print("CSV   :", OUTPUT_CSV)
    print("JSON  :", OUTPUT_JSON)
    print("=" * 60)


if __name__ == "__main__":
    main()
