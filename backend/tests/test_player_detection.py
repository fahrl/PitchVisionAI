
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# ==========================================
# PITCHVISION AI - PLAYER DETECTION TEST
# ==========================================

# Project directory
PROJECT_DIR = Path(__file__).resolve().parents[2]

# Input image
IMAGE_PATH = PROJECT_DIR / "datasets" / "test_images" / "football_test.png"

# Output directory
OUTPUT_DIR = PROJECT_DIR / "outputs" / "player_detection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "football_detection.png"

# Model configuration
MODEL_PATH = PROJECT_DIR / "yolo26n.pt"
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640

def main():
    print("=" * 50)
    print("PITCHVISION AI - PLAYER DETECTION")
    print("=" * 50)

    # Check input image
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(str(MODEL_PATH))

    # Read image
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise ValueError("Failed to read input image.")

    # Run detection
    print("Running player detection...")
    results = model.predict(
        source=image,
        classes=[0],
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        device=device,
        verbose=False
    )

    result = results[0]
    detections = result.boxes

    # Draw player markers
    player_count = 0

    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0])

        # Bottom-center coordinate
        center_x = int((x1 + x2) / 2)
        bottom_y = int(y2)

        player_count += 1

        # Draw bounding box (thin)
        cv2.rectangle(
            image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 0, 0),
            2
        )

        # Draw foot-position marker
        cv2.circle(
            image,
            (center_x, bottom_y),
            6,
            (0, 255, 255),
            -1
        )

        # Draw compact confidence label
        cv2.putText(
            image,
            f"P {confidence:.2f}",
            (int(x1), max(20, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # Save output
    success = cv2.imwrite(str(OUTPUT_PATH), image)

    if not success:
        raise RuntimeError("Failed to save detection output.")

    print("-" * 50)
    print(f"Players detected: {player_count}")
    print(f"Output saved: {OUTPUT_PATH}")
    print("Detection completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
