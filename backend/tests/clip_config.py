import os
from pathlib import Path

P = Path(__file__).resolve().parents[2]
CLIP = os.environ.get("CLIP", "clip1")
VIDEO = P / "datasets" / "test_videos" / (
    "football_test.mp4" if CLIP == "clip1" else f"football_test_{CLIP}.mp4")


def out(sub):
    """clip1 tetap di folder lama. Klip lain di outputs/<sub>/<clip>."""
    d = P / "outputs" / sub if CLIP == "clip1" else P / "outputs" / sub / CLIP
    d.mkdir(parents=True, exist_ok=True)
    return d