"""
Saarpolygon 2-Minute All-Angles Cinematic Master Video Generator
Combines all key drone flight paths into a continuous ~2-minute video:
1. High-Altitude 360° Panoramic Orbit (DJI_0255)
2. Mid-Altitude 360° Orbit (DJI_0256)
3. Low-Altitude & Roof Overpass (DJI_0257)
4. Overhead Apex Orbit (DJI_0253)
5. Shadow & Ground Base Orbit (DJI_0260)
6. Scenic Landscape Approach (DJI_0261)
Includes smooth cross-dissolve transitions between sequence cuts.
"""

import os
import sys
import glob
import subprocess
import cv2
import numpy as np
import imageio_ffmpeg

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
FRAMES_DIR = r"C:\Users\BANU PRAKASH\Saarpolygon"
OUTPUT_VIDEO = os.path.abspath(r"datasets\saarpolygon_complete_all_angles_2min.mp4")

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
FPS = 15
TRANSITION_FRAMES = 12  # ~0.8s smooth cross-dissolve between sequences


def get_sequence_files(prefix: str) -> list:
    files = sorted(glob.glob(os.path.join(FRAMES_DIR, f"{prefix}_*.jpg")))
    return files


def main():
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    
    # 6 cinematic flight sequences in aesthetic narrative order
    sequence_names = [
        ("DJI_0255", "High-Altitude 360 Panoramic Orbit"),
        ("DJI_0256", "Mid-Altitude 360 Geometric Orbit"),
        ("DJI_0257", "Low-Altitude Roof Overpass"),
        ("DJI_0253", "Overhead Apex Orbit"),
        ("DJI_0260", "Base & Shadow Structural Orbit"),
        ("DJI_0261", "Panoramic Vista Orbit")
    ]
    
    sequences = []
    total_raw_frames = 0
    for prefix, desc in sequence_names:
        flist = get_sequence_files(prefix)
        sequences.append((prefix, desc, flist))
        total_raw_frames += len(flist)
        print(f"Loaded {prefix} ({desc}): {len(flist)} frames")
        
    print(f"\nTotal raw frames: {total_raw_frames}")
    est_duration = total_raw_frames / FPS
    print(f"Target FPS: {FPS}, estimated duration: {est_duration:.1f}s ({int(est_duration//60):02d}:{int(est_duration%60):02d})")

    # Start FFmpeg subprocess accepting raw BGR frames on stdin
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        "-pix_fmt", "bgr24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "21",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        OUTPUT_VIDEO
    ]
    
    print(f"Starting FFmpeg encoder...")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    frames_written = 0

    for s_idx, (prefix, desc, flist) in enumerate(sequences):
        print(f"Processing sequence {s_idx+1}/{len(sequences)}: {prefix} ({len(flist)} frames)...")
        
        # Read frames of this sequence
        seq_frames = []
        for fpath in flist:
            img = cv2.imread(fpath)
            if img is None:
                continue
            if img.shape[1] != TARGET_WIDTH or img.shape[0] != TARGET_HEIGHT:
                img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
            seq_frames.append(img)
            
        if not seq_frames:
            continue

        # If not first sequence, do cross-dissolve with previous sequence's tail
        if s_idx > 0 and 'prev_tail' in locals() and len(prev_tail) >= TRANSITION_FRAMES and len(seq_frames) >= TRANSITION_FRAMES:
            # Create crossfade
            for i in range(TRANSITION_FRAMES):
                alpha = (i + 1) / float(TRANSITION_FRAMES + 1)
                blended = cv2.addWeighted(prev_tail[i], 1.0 - alpha, seq_frames[i], alpha, 0)
                proc.stdin.write(blended.tobytes())
                frames_written += 1
            # Write remaining body of current sequence
            for frame in seq_frames[TRANSITION_FRAMES:]:
                proc.stdin.write(frame.tobytes())
                frames_written += 1
        else:
            # First sequence: write all frames
            for frame in seq_frames:
                proc.stdin.write(frame.tobytes())
                frames_written += 1
                
        # Store tail for next transition
        prev_tail = seq_frames[-TRANSITION_FRAMES:]

    # Finish stream
    proc.stdin.close()
    proc.wait()
    
    final_size_mb = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)
    duration_sec = frames_written / FPS
    mins = int(duration_sec // 60)
    secs = int(duration_sec % 60)
    print(f"\n[SUCCESS] Successfully compiled:")
    print(f"  File: {OUTPUT_VIDEO}")
    print(f"  Duration: {mins:02d}:{secs:02d} ({duration_sec:.1f}s)")
    print(f"  Total Frames: {frames_written}")
    print(f"  Resolution: {TARGET_WIDTH}x{TARGET_HEIGHT} @ {FPS} FPS")
    print(f"  File Size: {final_size_mb:.2f} MB")


if __name__ == "__main__":
    main()
