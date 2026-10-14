@echo off
REM Double-click to start the Urdu Speech web interface (transcribe + speak).
REM Listens on all network interfaces so other devices on the LAN can reach it too.
REM Opens http://127.0.0.1:8095 locally once both models are loaded.

cd /d "%~dp0"
title Urdu Speech - Web Interface

echo.
echo   ============================================
echo     URDU SPEECH - starting up
echo   ============================================
echo.
echo   Loading the transcriber and the voice (about 20 seconds)...
echo   Your browser will open automatically.
echo.
echo   Other devices on this network can reach it at:
echo     http://192.168.0.10:8095
echo   (requires the firewall rule set up once -- see README)
echo.
echo   Leave this window OPEN while using it.
echo   Close it when you are finished.
echo.

REM Give the server a moment to bind the port before the browser opens, otherwise
REM the first page load races the startup and shows a connection error.
start "" /b cmd /c "timeout /t 25 /nobreak >nul & start http://127.0.0.1:8095"

python webapp\server.py --host 0.0.0.0 --preload

REM Only reached if the server exits or fails to start.
echo.
echo   The server has stopped.
echo.
echo   If you saw an error above:
echo     - "No module named ..."   ^-^> pip install fastapi uvicorn transformers faster-whisper scipy
echo     - "No voice model at ..." ^-^> run export_model.py to create urdu_tts_final
echo     - ffmpeg errors           ^-^> winget install Gyan.FFmpeg
echo.
pause
