import torch
import cv2
import ultralytics
import supervision as sv

print("=" * 45)
print("PITCHVISION AI - ENVIRONMENT CHECK")
print("=" * 45)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("OpenCV:", cv2.__version__)
print("Ultralytics:", ultralytics.__version__)
print("Supervision:", sv.__version__)

print("-" * 45)
print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")
print("Environment check completed!")