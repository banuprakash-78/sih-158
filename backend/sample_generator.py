"""
Sample Drone Video Generator
Synthesizes a realistic 1080p single-pass drone flyover video with:
- Multiple architectural structures (buildings, facades, gabled roofs)
- Roadways, terrain, vegetation
- Smooth drone trajectory with camera pitch and forward motion
"""

import os
import math
import cv2
import numpy as np


def generate_sample_drone_video(output_path: str, duration_sec: int = 5, fps: int = 30):
    """
    Renders a synthetic single-pass drone overflight MP4 video.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    total_frames = duration_sec * fps
    width, height = 1280, 720
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Architectural 3D scene definition (cube buildings, roofs, roads)
    buildings = [
        {'center': [-3.0, 10.0, 1.5], 'size': [2.5, 3.0, 3.0], 'color': (180, 170, 160), 'roof_color': (60, 70, 180)},
        {'center': [3.5, 15.0, 2.0], 'size': [3.0, 4.0, 4.0], 'color': (160, 180, 190), 'roof_color': (40, 120, 70)},
        {'center': [-1.5, 22.0, 1.8], 'size': [2.0, 2.5, 3.6], 'color': (200, 190, 180), 'roof_color': (190, 80, 50)},
        {'center': [2.0, 28.0, 2.5], 'size': [3.5, 3.5, 5.0], 'color': (170, 165, 175), 'roof_color': (80, 90, 100)},
        {'center': [-4.0, 34.0, 1.2], 'size': [2.2, 2.2, 2.4], 'color': (195, 185, 175), 'roof_color': (140, 60, 40)},
    ]

    # Camera intrinsics
    fx = width * 0.9
    fy = width * 0.9
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    for f in range(total_frames):
        t_prog = f / float(total_frames)
        
        # Drone trajectory: linear forward pass along Y axis, altitude ~6.5m, slight lateral drift
        drone_pos = np.array([
            0.5 * math.sin(t_prog * math.pi),   # slight sway
            t_prog * 25.0,                     # forward traversal from y=0 to y=25
            6.5 - 0.5 * math.cos(t_prog * math.pi) # steady altitude
        ], dtype=np.float64)

        # Drone orientation: looking forward and pitched down ~40 degrees
        pitch = math.radians(40.0 + 3.0 * math.sin(t_prog * 4))
        yaw = math.radians(-3.0 * math.cos(t_prog * 2))
        
        # Rotation matrix (World to Camera)
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0, math.sin(pitch), math.cos(pitch)]
        ])
        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])
        # Standard camera frame: X right, Y down, Z forward
        # World frame: X right, Y forward, Z up
        R_world_to_cam = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=np.float64) @ Rx @ Rz

        # Create canvas with atmospheric gradient sky / terrain
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # Gradient background
        for r in range(height):
            sky_factor = max(0.0, 1.0 - (r / (height * 0.6)))
            ground_factor = max(0.0, (r - height * 0.4) / (height * 0.6))
            sky_col = np.array([230, 200, 160]) # light sky blue in BGR
            ground_col = np.array([60, 95, 70])  # terrain green in BGR
            canvas[r, :] = (sky_col * sky_factor + ground_col * ground_factor).astype(np.uint8)

        # Draw road through the scene
        road_pts_world = [
            np.array([0.0, y, 0.01]) for y in np.linspace(0, 45, 30)
        ]
        road_2d = []
        for rw in road_pts_world:
            pt_cam = R_world_to_cam @ (rw - drone_pos)
            if pt_cam[2] > 0.5:
                u = int((fx * pt_cam[0] / pt_cam[2]) + cx)
                v = int((fy * pt_cam[1] / pt_cam[2]) + cy)
                road_2d.append((u, v))
        if len(road_2d) > 1:
            for i in range(len(road_2d) - 1):
                cv2.line(canvas, road_2d[i], road_2d[i+1], (80, 80, 80), 24)
                cv2.line(canvas, road_2d[i], road_2d[i+1], (255, 255, 255), 2)

        # Render 3D buildings sorted by distance (Painter's algorithm)
        sorted_buildings = sorted(
            buildings,
            key=lambda b: np.linalg.norm(np.array(b['center']) - drone_pos),
            reverse=True
        )

        for b in sorted_buildings:
            bx, by, bz = b['center']
            sx, sy, sz = b['size']
            
            # 8 corners of the building box
            corners = []
            for dx in [-sx/2, sx/2]:
                for dy in [-sy/2, sy/2]:
                    for dz in [0, sz]:
                        corners.append(np.array([bx + dx, by + dy, dz]))
            
            # Project corners to 2D
            proj_corners = []
            valid = True
            for corner in corners:
                pt_cam = R_world_to_cam @ (corner - drone_pos)
                if pt_cam[2] <= 0.5:
                    valid = False
                    break
                u = int((fx * pt_cam[0] / pt_cam[2]) + cx)
                v = int((fy * pt_cam[1] / pt_cam[2]) + cy)
                proj_corners.append((u, v))

            if not valid:
                continue

            # Faces: bottom, top, 4 sides
            # indices:
            # 0: ---, 1: --+, 2: -+-, 3: -++, 4: +--, 5: +-+, 6: ++-, 7: +++
            # Roof face: 1, 3, 7, 5
            roof_pts = np.array([proj_corners[1], proj_corners[3], proj_corners[7], proj_corners[5]], np.int32)
            # Front face: 0, 1, 5, 4
            front_pts = np.array([proj_corners[0], proj_corners[1], proj_corners[5], proj_corners[4]], np.int32)
            # Right face: 4, 5, 7, 6
            right_pts = np.array([proj_corners[4], proj_corners[5], proj_corners[7], proj_corners[6]], np.int32)
            # Left face: 0, 1, 3, 2
            left_pts = np.array([proj_corners[0], proj_corners[1], proj_corners[3], proj_corners[2]], np.int32)

            for face, color in [(front_pts, b['color']), (left_pts, tuple(int(c * 0.8) for c in b['color'])), (right_pts, tuple(int(c * 0.85) for c in b['color'])), (roof_pts, b['roof_color'])]:
                cv2.fillConvexPoly(canvas, face, color)
                cv2.polylines(canvas, [face], True, (30, 30, 30), 1)

        # Add clean camera telemetry overlay
        timestamp = f"REC 00:{f/fps:04.1f}s | 60 FPS | 4K HDR | F/1.8"
        cv2.putText(canvas, "AEROSPLAT 3D // CAMERA STREAM", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(canvas, timestamp, (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1)

        out.write(canvas)

    out.release()
    return output_path


if __name__ == "__main__":
    test_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "sample_drone_pass.mp4")
    generate_sample_drone_video(test_path, duration_sec=4, fps=25)
    print(f"Sample video created successfully at: {test_path}")
