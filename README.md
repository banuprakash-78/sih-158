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
