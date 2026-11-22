import json
from pathlib import Path

import cv2
import numpy as np

P = Path(__file__).resolve().parents[2]
VIDEO = P / "datasets" / "test_videos" / "football_test.mp4"
OUT = P / "outputs" / "pitch_mapping"
OUT.mkdir(parents=True, exist_ok=True)

FRAME_NUMBER = 1      # 1-based
SCALE = 0.6
L, W = 105.0, 68.0

# (nama prompt, (x sepanjang lapangan, y lebar) dalam meter)
LANDMARKS = [
    ("1. Garis tengah bertemu garis samping JAUH (atas)", (52.5, 0.0)),
    ("2. Titik tengah lapangan (center spot)", (52.5, W / 2)),
    ("3. Lingkaran tengah x garis tengah, sisi ATAS", (52.5, W / 2 - 9.15)),
    ("4. Lingkaran tengah x garis tengah, sisi BAWAH", (52.5, W / 2 + 9.15)),
    ("5. Kotak penalti kanan: sudut luar ATAS", (L - 16.5, 13.84)),
    ("6. Kotak penalti kanan: sudut luar BAWAH", (L - 16.5, 54.16)),
    ("7. Kotak penalti kanan: garis atas bertemu garis gawang", (L, 13.84)),
    ("8. Kotak 6 yard kanan: sudut luar ATAS", (L - 5.5, 24.84)),
    ("9. Corner kanan atas (bendera)", (L, 0.0)),
]


def main():
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Tidak bisa membaca frame")

    disp = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
    state = {"pt": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pt"] = (x / SCALE, y / SCALE)

    win = "Klik landmark | s = lewati | q = selesai"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    img_pts, pitch_pts = [], []
    done = False
    for name, xy in LANDMARKS:
        if done:
            break
        state["pt"] = None
        while True:
            view = disp.copy()
            for p in img_pts:
                cv2.circle(view, (int(p[0] * SCALE), int(p[1] * SCALE)), 5, (0, 255, 255), -1)
            cv2.rectangle(view, (0, 0), (view.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(view, f"{name}   [terklik: {len(img_pts)}]", (10, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow(win, view)
            k = cv2.waitKey(20) & 0xFF
            if state["pt"] is not None:
                img_pts.append(state["pt"])
                pitch_pts.append(xy)
                break
            if k == ord("s"):
                break
            if k == ord("q"):
                done = True
                break
    cv2.destroyAllWindows()

    if len(img_pts) < 4:
        raise RuntimeError("Minimal 4 titik diperlukan")

    img_pts = np.array(img_pts, np.float32)
    pitch_pts = np.array(pitch_pts, np.float32)
    H, mask = cv2.findHomography(img_pts, pitch_pts, cv2.RANSAC, 1.5)
    if H is None:
        raise RuntimeError("Homography gagal")

    proj = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - pitch_pts, axis=1)
    print(f"Titik: {len(img_pts)} | inlier: {int(mask.sum())}")
    print(f"Error reproyeksi: rata-rata {err.mean():.2f} m, maks {err.max():.2f} m")
    for (name, _), e in zip([l for l in LANDMARKS][:len(err)], err):
        print(f"  {e:5.2f} m  {name}")

    path = OUT / f"homography_frame_{FRAME_NUMBER}.json"
    path.write_text(json.dumps({"frame": FRAME_NUMBER, "H": H.tolist()}, indent=2))
    print("Disimpan:", path)


if __name__ == "__main__":
    main()