@echo off
REM Run once on a fresh machine to install everything the model and web UI need.
REM Not needed on the machine this was built on -- those packages are already there.

cd /d "%~dp0"
title Urdu STT - Install requirements

echo.
echo   Installing Python packages. This can take several minutes.
echo.

REM Torch must come from the CUDA index; the default PyPI wheel is CPU-only on Windows
REM and everything would silently fall back to running on the CPU.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install faster-whisper ctranslate2 fastapi uvicorn python-multipart soundfile numpy

echo.
echo   Checking ffmpeg (needed to read mp3/m4a and browser mic recordings)...
where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo   ffmpeg NOT FOUND. Install it with:
  echo       winget install Gyan.FFmpeg
  echo   then close and reopen this window.
) else (
  echo   ffmpeg found.
)

echo.
echo   Done. You can now run START_WEBAPP.bat
echo.
pause
