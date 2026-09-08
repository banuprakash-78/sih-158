"""
Saarpolygon Best 100 Frames Video Generator
Extracts the 100 sharpest, evenly-spaced 360-degree keyframes from the top-ranked
flight sequence (DJI_0255) and encodes them into an optimized cinematic MP4.
"""

import os
import glob
import subprocess
import cv2
import numpy as np
import imageio_ffmpeg

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
FRAMES_DIR = r"C:\Users\BANU PRAKASH\Saarpolygon"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "saarpolygon_best_100_frames.mp4")

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
TARGET_FPS = 12  # 12 FPS gives ~8.3s smooth, clear showcase of all 100 keyframes


def select_best_100_frames():
    files_255 = sorted(glob.glob(os.path.join(FRAMES_DIR, "DJI_0255_*.jpg")))
    if not files_255:
        raise RuntimeError(f"No DJI_0255 frames found in {FRAMES_DIR}")

    total = len(files_255)
    print(f"Found {total} raw frames in DJI_0255.")

    # Evenly distribute 100 sample positions across the 360 orbit
    ideal_indices = np.linspace(0, total - 1, 100, dtype=int)
    best_100 = []

    print("Selecting sharpest frame in each local neighborhood...")
    for step, idx in enumerate(ideal_indices):
        # Check adjacent frames within a window of [-2, +2] to avoid any momentary motion blur
        window = range(max(0, idx - 2), min(total, idx + 3))
        best_fpath = files_255[idx]
        best_sharpness = -1.0

        for w in window:
            fpath = files_255[w]
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Laplacian variance measures visual edge sharpness
                sharpness = float(cv2.Laplacian(img, cv2.CV_64F).var())
                if sharpness > best_sharpness:
                    best_sharpness = sharpness
                    best_fpath = fpath

        best_100.append((best_fpath, best_sharpness))

    return best_100


def build_video():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_frames = select_best_100_frames()
    print(f"Selected {len(best_frames)} optimal keyframes.")

    avg_sharpness = np.mean([s for _, s in best_frames])
    print(f"Average keyframe sharpness score: {avg_sharpness:.1f}")

    # Launch FFmpeg encoder
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        "-pix_fmt", "bgr24",
        "-r", str(TARGET_FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",  # High quality visually lossless
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        OUTPUT_VIDEO
    ]

    print(f"Encoding to {OUTPUT_VIDEO} at {TARGET_FPS} FPS...")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    written = 0
    for idx, (fpath, score) in enumerate(best_frames):
        img = cv2.imread(fpath)
        if img is None:
            continue
        if img.shape[1] != TARGET_WIDTH or img.shape[0] != TARGET_HEIGHT:
            img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        proc.stdin.write(img.tobytes())
        written += 1

    proc.stdin.close()
    proc.wait()

    file_size_mb = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)
    duration = written / TARGET_FPS
    print(f"\n[SUCCESS] Generated 100-frame video:")
    print(f"  Path: {OUTPUT_VIDEO}")
    print(f"  Frames Written: {written}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Framerate: {TARGET_FPS} FPS")
    print(f"  Resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Avg Sharpness: {avg_sharpness:.1f}")


if __name__ == "__main__":
    build_video()
