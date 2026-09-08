@echo off
title OmniSplat 3D Prototype - Port 8000
echo ========================================================
echo   OmniSplat 3D: Drone Video to 3D Gaussian Splatting
echo   Starting Local Server at http://localhost:8000 ...
echo ========================================================
echo.

cd /d "%~dp0"
python -m pip install -r requirements.txt
cd backend
python main.py

pause
