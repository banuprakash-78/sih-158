"""
Saarpolygon Dataset Video Compiler
Assembles multi-angle drone videos from HuggingFace dataset frames
(datasets/naruone90/gsplat-training-frames_Saarpolygon)
"""

import os
import sys
import glob
import subprocess
import imageio_ffmpeg

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
FRAMES_DIR = r"C:\Users\BANU PRAKASH\Saarpolygon"
OUTPUT_DIR = os.path.abspath("datasets")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_sequence(input_pattern: str, output_path: str, fps: int = 30, crf: int = 22, width: int = 1920, height: int = 1080):
    """Encode an image sequence pattern to an H.264 MP4 video."""
    print(f"--> Encoding {os.path.basename(output_path)} from {input_pattern}...")
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-vf", f"scale={width}:{height}:flags=lanczos",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(cmd, check=True)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    [DONE] {output_path} ({size_mb:.2f} MB)")

def create_concat_list_video(file_list: list, output_path: str, fps: int = 30, crf: int = 22, width: int = 1920, height: int = 1080):
    """Encode a video from an explicit list of file paths."""
    print(f"--> Encoding {os.path.basename(output_path)} from {len(file_list)} selected frames...")
    
    # Write a temporary concat file
    concat_txt = os.path.join(OUTPUT_DIR, "temp_concat.txt")
    duration_per_frame = 1.0 / fps
    with open(concat_txt, "w", encoding="utf-8") as f:
        for p in file_list:
            norm = p.replace("\\", "/")
            f.write(f"file '{norm}'\n")
            f.write(f"duration {duration_per_frame:.4f}\n")
        # repeat last file once per ffmpeg concat demuxer spec
        if file_list:
            norm = file_list[-1].replace("\\", "/")
            f.write(f"file '{norm}'\n")

    cmd = [
        FFMPEG_EXE,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_txt,
        "-vf", f"scale={width}:{height}:flags=lanczos",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(cmd, check=True)
    if os.path.exists(concat_txt):
        os.remove(concat_txt)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    [DONE] {output_path} ({size_mb:.2f} MB)")

def build_all_videos():
    # 1. Video 1: Single Complete 360-Degree Continuous Orbit (DJI_0255)
    # Perfect for SfM feature continuity (416 frames)
    out_360 = os.path.join(OUTPUT_DIR, "saarpolygon_orbit_360.mp4")
    pattern_255 = os.path.join(FRAMES_DIR, "DJI_0255_frame_%04d.jpg")
    encode_sequence(pattern_255, out_360, fps=30, crf=22)

    # 2. Video 2: Multi-Angle Master Tour
    # Combines High Orbit (DJI_0255), Mid-Altitude Orbit (DJI_0256), and Eye-Level Overpass (DJI_0257)
    f255 = sorted(glob.glob(os.path.join(FRAMES_DIR, "DJI_0255_*.jpg")))[::2]
    f256 = sorted(glob.glob(os.path.join(FRAMES_DIR, "DJI_0256_*.jpg")))[::2]
    f257 = sorted(glob.glob(os.path.join(FRAMES_DIR, "DJI_0257_*.jpg")))[::2]
    all_multi = f255 + f256 + f257
    print(f"Total multi-angle frames selected: {len(all_multi)} (High: {len(f255)}, Mid: {len(f256)}, Low: {len(f257)})")
    
    out_master = os.path.join(OUTPUT_DIR, "saarpolygon_all_angles_master.mp4")
    create_concat_list_video(all_multi, out_master, fps=30, crf=22)

    # 3. Video 3: Fast Prototype Testing Video (100 evenly distributed 360-degree keyframes)
    f255_lite = sorted(glob.glob(os.path.join(FRAMES_DIR, "DJI_0255_*.jpg")))[::4] # 104 frames
    out_lite = os.path.join(OUTPUT_DIR, "saarpolygon_lite_orbit.mp4")
    create_concat_list_video(f255_lite, out_lite, fps=24, crf=22)

if __name__ == "__main__":
    build_all_videos()
