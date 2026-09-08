# OmniSplat 3D — Universal Drone Video to 3D Gaussian Splatting & 3D Print Studio

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL-black.svg)](https://threejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **OmniSplat 3D** is an end-to-end photogrammetry and 3D reconstruction studio that transforms raw drone survey video footage into interactive WebGL 3D models, dense 3D point clouds, and 3D-printable watertight meshes (`.obj`, `.stl`, `.ply`).

---

## 📸 System Overview

```
                      RAW DRONE VIDEO (.MP4)
                               │
                               ▼
                    [ Stage 1: Keyframe Extraction ]
                               │
                               ▼
                 [ Stage 2: Structure from Motion (SfM) ]
                 • Feature Extraction (SIFT / ORB)
                 • Camera Pose Estimation & Extrinsics
                               │
                               ▼
             [ Stage 3: Sparse / Dense 3D Point Cloud ]
                               │
                               ▼
        ┌──────────────────────┴──────────────────────┐
        │                                             │
        ▼                                             ▼
[ 3D Gaussian Splats (.PLY) ]              [ Watertight 3D Mesh (.OBJ / .STL) ]
• Continuous Radiance Fields               • Physical 3D Print Geometry
• Real-time Volumetric Splats              • Slicer Layer Height Simulation
        │                                             │
        └──────────────────────┬──────────────────────┘
                               │
                               ▼
            [ Three.js WebGL Interactive Studio (Port 8000) ]
            • Real-time Telemetry (Heading, Tilt, Altitude)
            • Atmospheric Time-of-Day Lighting (Day / Sunset / Night)
            • Solid 3D Print Resin & Slicer Layer Preview Modes
            • Interactive Distance & Height Measurement Tool
            • One-Click OBJ, STL, and PLY 3D Model Downloads
```

---

## ✨ Key Features

- **Single-Pass Ingestion**: Drop any drone flight video to automatically extract keyframes and estimate camera trajectories.
- **Interactive Three.js Viewport**:
  - **🖨️ 3D Print Resin Mode**: Inspect solid physical prototype geometry.
  - **🧱 3D Slicer Layers Mode**: Visualize 0.16mm layer height deposition.
  - **✨ 3D Point Cloud Mode**: Volumetric spatial radiance point cloud rendering.
- **Atmospheric Lighting Engine**: Real-time Day (Midday Sun), Golden Hour (Sunset), and Night (Floodlights) illumination.
- **Flight Path & Build Bed Visualization**: Overlay camera flight trajectories and a 250×250mm 3D printer build bed.
- **Live Real-time SSE Telemetry**: Server-Sent Events stream pipeline progress and system logs directly to the UI.
- **Industrial Export Formats**: Instant download for `.obj` (textured architectural mesh), `.stl` (watertight 3D print), and `.ply` (3D Gaussian point cloud).

---

## 📁 Repository Structure

```
OmniSplat-3D-Prototype/
├── backend/
│   ├── main.py                 # FastAPI Web API Server & SSE pipeline endpoints
│   ├── mesh_engine.py          # Architectural 3D mesh generator & procedural builder
│   ├── gaussian_trainer.py     # 3D Gaussian Splatting optimization trainer
│   ├── sfm_engine.py           # Structure from Motion & camera pose estimation
│   ├── video_processor.py      # Video decoding & keyframe extraction
│   ├── sample_generator.py     # Synthetic drone flight path generator
│   └── requirements.txt        # Python backend dependencies
├── frontend/
│   ├── index.html              # Modern glassmorphism UI & Three.js canvas
│   ├── styles.css              # Custom styling & dark-mode design system
│   ├── app.js                  # Three.js WebGL renderer, OrbitControls, & UI logic
│   └── assets/
│       └── textures/           # Photorealistic PBR surface textures
├── workspace/
│   ├── outputs/                # Pre-baked 3D assets (OBJ, STL, PLY, metadata)
│   └── uploads/                # User uploaded video cache (.gitkeep)
├── datasets/
│   ├── sample_drone_pass.mp4   # Included sample aerial survey video
│   └── .gitkeep
├── requirements.txt            # Root dependencies
├── start_server.bat            # 1-Click launcher for Windows
├── start_server.sh             # 1-Click launcher for Linux / macOS
├── .gitignore                  # Git rules to exclude cache and large files
├── LICENSE                     # MIT Open Source License
└── README.md                   # Full documentation
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.10** or newer installed.
- A modern web browser (Chrome, Edge, Firefox, Brave) with WebGL enabled.

### 1. Installation

Clone this repository or extract the project folder:
```bash
git clone https://github.com/YOUR_USERNAME/omnisplat-3d-prototype.git
cd omnisplat-3d-prototype
```

Install required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Run the Application

#### Option A: 1-Click Launchers
- **Windows**: Double-click `start_server.bat`
- **macOS / Linux**: Run `./start_server.sh` (make executable first: `chmod +x start_server.sh`)

#### Option B: Manual Command
```bash
cd backend
python main.py
```
Or with Uvicorn:
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open in Browser
Visit **[http://localhost:8000](http://localhost:8000)** in your browser to interact with the 3D model!

---

## 📡 API Endpoints Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the interactive frontend web studio |
| `/api/pipeline/state` | `GET` | Returns current pipeline status, stage, and telemetry |
| `/api/pipeline/logs/stream` | `GET` | SSE stream for real-time progress updates |
| `/api/pipeline/upload` | `POST` | Upload a new drone `.mp4` video for 3D processing |
| `/api/model-mesh` | `GET` | Returns optimized 3D mesh geometry, materials, and camera poses |
| `/api/datasets` | `GET` | Lists available flight inspection datasets |
| `/api/download/obj` | `GET` | Downloads textured Wavefront `.obj` file |
| `/api/download/stl` | `GET` | Downloads watertight 3D-printable `.stl` file |
| `/api/download/ply` | `GET` | Downloads 3D Gaussian Splatting `.ply` point cloud |

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn, OpenCV (`cv2`), PyTorch, NumPy, SciPy, Plyfile.
- **Frontend**: Vanilla HTML5/CSS3, Modern JavaScript (ES6+), Three.js (WebGL), OrbitControls.
- **Data Protocols**: Server-Sent Events (SSE), RESTful JSON, Binary Buffer Attributes.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).


---

## 📊 Included Dataset Details

---
license: cc-by-4.0
language:
- en
tags:
- drone
- gaussian-splatting
- 3dgs
- gsplat
- nerf
- photogrammetry
pretty_name: Gaussian Splatting Drone Frames — Saarpolygon
---

# Dataset Card for Gaussian Splatting Drone Frames — Saarpolygon (DE)

### Dataset Description

- **Summary:** Image sequence extracted from drone video intended for training and evaluating **3DGS/NeRF** and **SfM/MVS** pipelines.
- **Location:** Saarpolygon, Germany.
- **Language(s) (NLP):** N/A (visual data)

### Direct Use

- Training and evaluation of **Gaussian Splatting (3DGS/gsplat)** and **NeRF** variants.
- 3D reconstruction with **SfM/MVS** (e.g., **COLMAP**) and validation of photogrammetry pipelines.
- Benchmarks/ablations (PSNR/SSIM/LPIPS), pose estimation, approximate intrinsic calibration, metric scaling with GPS.

### Out-of-Scope Use

- Person/vehicle recognition or surveillance: the dataset is **not** designed for identification of faces, license plates, or other sensitive attributes.
- Tasks requiring exhaustive semantic annotations (not provided by default).

### Source Data

- **Origin:** Frames extracted from drone video captured by the author.
- **Location:** Horizontobservatorium (Halde Hoheward), Germany.

#### Personal and Sensitive Information

- The dataset **does not intend** to capture personal information. Faces and license plates are avoided or may be masked if incidentally present. No personal contact data included.

## Citation 

If you use this dataset, please cite it as follows:

**APA (no DOI, using Hugging Face URL)**  
Roberto SL. (2025). *Gaussian splatting drone frames — Saarpolygon (Germany)* (v1.0) [Dataset]. Hugging Face. https://huggingface.co/datasets/naruone90/gsplat-training-frames_Saarpolygon
