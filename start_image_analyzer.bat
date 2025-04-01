@echo off
title AWDigitalworld - Image Analyzer Validation
color 0A

echo ================================================
echo  AWDigitalworld - Starting Validation Sequence
echo ================================================
python validate_code.py
echo.
pause

echo ================================================
echo  Running pip installs (if needed)...
echo ================================================
pip install -r requirements.txt
echo.
pause

echo ================================================
echo  Launching Image Analyzer...
echo ================================================
python main.py

echo.
pause
