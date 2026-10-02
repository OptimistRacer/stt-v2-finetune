@echo off
REM Double-click to start the Urdu STT web interface.
REM Opens http://127.0.0.1:8080 in your browser once the model is loaded.

cd /d "%~dp0"
title Urdu STT - Web Interface

echo.
echo   ============================================
echo     URDU SPEECH TO TEXT - starting up
echo   ============================================
echo.
echo   Loading the model (about 5 seconds)...
echo   Your browser will open automatically.
echo.
echo   Leave this window OPEN while using it.
echo   Close it when you are finished.
echo.

REM Give the server a moment to bind the port before the browser opens, otherwise
REM the first page load races the startup and shows a connection error.
start "" /b cmd /c "timeout /t 8 /nobreak >nul & start http://127.0.0.1:8080"

python webapp\server.py --preload

REM Only reached if the server exits or fails to start.
echo.
echo   The server has stopped.
echo.
echo   If you saw an error above:
echo     - "No module named ..."  ^-^> run INSTALL_REQUIREMENTS.bat first
echo     - "No model at ..."      ^-^> the model folder on D: is missing
echo.
pause
