from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
VIDEO_PATH = PROJECT_DIR / "datasets" / "test_videos" / "football_test.mp4"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "camera_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_frame(cap, n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {n}")
    return frame


def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_PATH}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    targets = [1, total // 4, total // 2, 3 * total // 4, total]

    ref = read_frame(cap, 1)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    print(f"Total frames: {total}")
    print(f"{'frame':>6} {'dx(px)':>8} {'dy(px)':>8} {'scale':>7} {'inliers':>8}")
    for n in targets:
        frame = read_frame(cap, n)
        cv2.imwrite(str(OUTPUT_DIR / f"frame_{n}.png"), frame)
        if n == 1:
            print(f"{n:>6} {0:>8.1f} {0:>8.1f} {1:>7.3f} {'ref':>8}")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(gray, None)
        matches = matcher.match(des1, des2)
        if len(matches) < 10:
            print(f"{n:>6} terlalu sedikit kecocokan fitur")
            continue
        p1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        p2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        M, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0)
        if M is None:
            print(f"{n:>6} estimasi gagal")
            continue
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        print(f"{n:>6} {M[0, 2]:>8.1f} {M[1, 2]:>8.1f} {scale:>7.3f} {int(inl.sum()):>8}")
    cap.release()
    print(f"Frame disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()