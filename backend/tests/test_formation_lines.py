from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
VIDEO_PATH = PROJECT_DIR / "datasets" / "test_videos" / "football_test.mp4"
TEAM_CSV = PROJECT_DIR / "outputs" / "team_classification" / "team_classification.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "formation_lines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO = OUTPUT_DIR / "football_formation_lines.mp4"
OUTPUT_CSV = OUTPUT_DIR / "formation_per_frame.csv"

# +1 = tim menyerang ke arah x membesar di gambar, -1 = ke arah x mengecil.
# CEK MANUAL pada video, lalu sesuaikan.
ATTACK_DIR = {"Team A": 1, "Team B": -1}
MIN_PLAYERS = 8
K_RANGE = (3, 4, 5)
SMOOTH_WINDOW = 25
COLORS = {"Team A": (255, 100, 0), "Team B": (0, 80, 255)}


def silhouette_1d(x, labels):
    scores = []
    for i in range(len(x)):
        same = labels == labels[i]
        if same.sum() < 2:
            scores.append(0.0)
            continue
        a = np.abs(x[same] - x[i]).sum() / (same.sum() - 1)
        b = min(
            np.abs(x[labels == c] - x[i]).mean()
            for c in set(labels) if c != labels[i]
        )
        scores.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(scores))


def detect_lines(depth):
    """Potong 1D di (k-1) jarak antar-pemain terbesar.
    Mengembalikan (lines, gk). Baris terdalam (dekat gawang sendiri) lebih dulu."""
    n = len(depth)
    order = np.argsort(depth)
    gaps = np.diff(depth[order])
    best = None
    for k in K_RANGE:
        if n <= k:
            continue
        cuts = np.sort(np.argsort(gaps)[-(k - 1):])
        sorted_labels = np.zeros(n, dtype=int)
        for c in cuts:
            sorted_labels[c + 1:] += 1
        labels = np.empty(n, dtype=int)
        labels[order] = sorted_labels
        score = silhouette_1d(depth, labels)
        if best is None or score > best[0]:
            best = (score, k, labels)
    if best is None:
        return None, None
    _, k, labels = best
    lines = [np.where(labels == c)[0] for c in range(k)]
    gk = None
    if len(lines[0]) == 1:
        gk, lines = lines[0], lines[1:]
    return lines, gk


def analyze_frames(df):
    """Hitung garis per frame dan tim, lalu haluskan string formasinya."""
    result = {}  # (frame, team) -> dict
    for frame, g in df.groupby("frame"):
        if frame % 50 == 0:
            print(f"Analyzing frame {frame}", flush=True)
        for team, direction in ATTACK_DIR.items():
            sub = g[g["team"] == team].reset_index(drop=True)
            if len(sub) < MIN_PLAYERS:
                continue
            depth = direction * sub["x"].to_numpy(float)
            lines, gk = detect_lines(depth)
            if lines is None:
                continue
            result[(frame, team)] = {
                "sub": sub,
                "lines": lines,
                "gk": gk,
                "raw": "-".join(str(len(line)) for line in lines),
            }

    frames = sorted(df["frame"].unique())
    half = SMOOTH_WINDOW // 2
    for team in ATTACK_DIR:
        raws = {f: result[(f, team)]["raw"] for f in frames if (f, team) in result}
        for f in frames:
            if (f, team) not in result:
                continue
            window = [raws[w] for w in range(f - half, f + half + 1) if w in raws]
            result[(f, team)]["smooth"] = Counter(window).most_common(1)[0][0]
    return result


def main():
    df = pd.read_csv(TEAM_CSV)
    print(f"Records: {len(df)} | teams: {df['team'].value_counts().to_dict()}", flush=True)

    result = analyze_frames(df)
    print(f"Frame-tim dengan formasi terdeteksi: {len(result)}", flush=True)

    rows = [
        {"frame": f, "team": t, "n_players": len(r["sub"]),
         "raw_formation": r["raw"], "smoothed_formation": r["smooth"]}
        for (f, t), r in sorted(result.items())
    ]
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create output video.")

    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if frame_no % 50 == 0:
            print(f"Rendering frame {frame_no}", flush=True)

        y_text = 35
        for team in ATTACK_DIR:
            r = result.get((frame_no, team))
            if r is None:
                cv2.putText(frame, f"{team}: n/a", (20, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[team], 2)
                y_text += 30
                continue

            sub, color = r["sub"], COLORS[team]

            # Garis tiap baris pemain, disambung dari atas ke bawah gambar.
            for idx in r["lines"]:
                pts = sub.iloc[idx].sort_values("y")[["x", "y"]].to_numpy(np.int32)
                if len(pts) >= 2:
                    cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False,
                                  color, 3, cv2.LINE_AA)

            for x, y in sub[["x", "y"]].to_numpy(int):
                cv2.circle(frame, (int(x), int(y)), 6, color, -1)

            # Kiper ditandai lingkaran kuning, tidak ikut garis.
            if r["gk"] is not None:
                gx, gy = sub.iloc[r["gk"][0]][["x", "y"]].astype(int)
                cv2.circle(frame, (int(gx), int(gy)), 10, (0, 255, 255), 2)

            label = f"{team}: {r['smooth']} (raw {r['raw']}, n={len(sub)})"
            cv2.putText(frame, label, (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_text += 30

        writer.write(frame)

    cap.release()
    writer.release()
    print("Video:", OUTPUT_VIDEO)
    print("CSV  :", OUTPUT_CSV)


if __name__ == "__main__":
    main()