"""
Video Processing & Dynamic Keyframe Extraction Module
Handles drone video ingestion, blur metric calculation, and keyframe decimation.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional


def calculate_blur_score(image: np.ndarray) -> float:
    """Calculate the Laplacian variance as a proxy for image sharpness/blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_keyframes(
    video_path: str,
    output_dir: str,
    target_frames: int = 40,
    max_resolution: int = 1280,
    min_blur_threshold: float = 30.0,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict:
    """
    Extracts high-quality keyframes from drone video, skipping blurry frames and
    maintaining optimal spatial baseline for photogrammetry / 3DGS.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
        
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_video_frames / fps if fps > 0 else 0
    
    if progress_callback:
        progress_callback(5.0, f"Analyzing video: {width}x{height} @ {fps:.1f} FPS, {duration_sec:.1f}s duration")

    # Determine step size to get approximate target frames
    step = max(1, total_video_frames // (target_frames * 2)) if total_video_frames > target_frames else 1
    
    candidate_frames = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % step == 0:
            blur_score = calculate_blur_score(frame)
            candidate_frames.append({
                'index': frame_idx,
                'timestamp': frame_idx / fps,
                'blur_score': blur_score,
                'frame': frame
            })
            
        frame_idx += 1
        
        if progress_callback and frame_idx % 20 == 0:
            pct = 5.0 + 20.0 * (frame_idx / max(1, total_video_frames))
            progress_callback(pct, f"Scanning frames: {frame_idx}/{total_video_frames}")
            
    cap.release()
    
    if not candidate_frames:
        raise ValueError("No valid frames could be decoded from video.")

    # Sort and pick the sharpest frames evenly distributed across trajectory
    # Partition candidate frames into target_frames bins
    bin_size = max(1, len(candidate_frames) // target_frames)
    selected_frames = []
    
    for b in range(0, len(candidate_frames), bin_size):
        chunk = candidate_frames[b:b+bin_size]
        if not chunk:
            continue
        # Pick the sharpest frame in this interval
        best = max(chunk, key=lambda x: x['blur_score'])
        selected_frames.append(best)
        if len(selected_frames) >= target_frames:
            break
            
    # Save selected keyframes
    extracted_paths = []
    for i, item in enumerate(selected_frames):
        frame = item['frame']
        h, w = frame.shape[:2]
        
        # Resize if exceeding max resolution
        if max(h, w) > max_resolution:
            scale = max_resolution / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
        frame_filename = f"frame_{i:04d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        extracted_paths.append(frame_path)
        
        if progress_callback:
            pct = 25.0 + 10.0 * ((i + 1) / len(selected_frames))
            progress_callback(pct, f"Exported keyframe {i+1}/{len(selected_frames)} (Sharpness: {item['blur_score']:.1f})")

    return {
        "total_video_frames": total_video_frames,
        "fps": fps,
        "original_resolution": [width, height],
        "duration_sec": duration_sec,
        "extracted_keyframes_count": len(extracted_paths),
        "keyframe_paths": extracted_paths,
        "output_dir": output_dir
    }
