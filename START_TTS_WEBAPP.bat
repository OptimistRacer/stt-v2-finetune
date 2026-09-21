@echo off
REM Double-click to start the Urdu TTS web interface.
REM Opens http://127.0.0.1:8090 in your browser once the voice is loaded.

cd /d "%~dp0"
title Urdu TTS - Web Interface

echo.
echo   ============================================
echo     URDU TEXT TO SPEECH - starting up
echo   ============================================
echo.
echo   Loading the voice (about 10 seconds)...
echo   Your browser will open automatically.
echo.
echo   Leave this window OPEN while using it.
echo   Close it when you are finished.
echo.

REM Give the server a moment to bind the port before the browser opens, otherwise
REM the first page load races the startup and shows a connection error.
start "" /b cmd /c "timeout /t 12 /nobreak >nul & start http://127.0.0.1:8090"

python webapp\server.py --preload

REM Only reached if the server exits or fails to start.
echo.
echo   The server has stopped.
echo.
echo   If you saw an error above:
echo     - "No module named ..."  ^-^> pip install fastapi uvicorn transformers torch scipy
echo     - "No model at ..."      ^-^> run export_model.py to create urdu_tts_final
echo.
pause
