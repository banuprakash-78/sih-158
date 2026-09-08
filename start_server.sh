#!/bin/bash
echo "========================================================"
echo "  OmniSplat 3D: Drone Video to 3D Gaussian Splatting"
echo "  Starting Local Server at http://localhost:8000 ..."
echo "========================================================"
echo ""

cd "$(dirname "$0")"
pip install -r requirements.txt
cd backend
python main.py
