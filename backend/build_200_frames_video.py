"""
Saarpolygon First 200 Consecutive Frames Video Generator (30 FPS)
Takes the first 200 frames in sequential order (DJI_0255_frame_0000.jpg to 0199.jpg)
and encodes an optimized 1080p Full HD video at 30 FPS.
"""

import os
import glob
import subprocess
import cv2
import imageio_ffmpeg

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
FRAMES_DIR = r"C:\Users\BANU PRAKASH\Saarpolygon"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
TARGET_FPS = 30


def encode_frames(file_list, output_path, fps=30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]

    print(f"Encoding {len(file_list)} frames to {output_path} at {fps} FPS...")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    written = 0
    for fpath in file_list:
        img = cv2.imread(fpath)
        if img is None:
            continue
        if img.shape[1] != TARGET_WIDTH or img.shape[0] != TARGET_HEIGHT:
            img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        proc.stdin.write(img.tobytes())
        written += 1

    proc.stdin.close()
    proc.wait()

    cap = cv2.VideoCapture(output_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    dur = count / actual_fps if actual_fps > 0 else 0
    cap.release()
    size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"[SUCCESS] Encoded {output_path}:")
    print(f"  Frames: {count}")
    print(f"  FPS: {actual_fps}")
    print(f"  Duration: {dur:.2f}s")
    print(f"  Resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"  Size: {size_mb:.2f} MB")
    return output_path


def main():
    # 1. Primary: First 200 consecutive frames in order from the main orbit sequence (DJI_0255)
    f_255 = sorted(glob.glob(os.path.join(FRAMES_DIR, "DJI_0255_*.jpg")))[:200]
    out_255 = os.path.join(OUTPUT_DIR, "saarpolygon_first_200_frames_30fps.mp4")
    encode_frames(f_255, out_255, fps=TARGET_FPS)

    # 2. Literal first 200 files in alphabetical order from folder
    f_raw = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))[:200]
    out_raw = os.path.join(OUTPUT_DIR, "saarpolygon_raw_first_200_files_30fps.mp4")
    encode_frames(f_raw, out_raw, fps=TARGET_FPS)


if __name__ == "__main__":
    main()
