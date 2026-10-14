@echo off
REM Starts the Parvaaz v2 pipeline (download wait, dataset, measure, describe, train) detached.
cd /d "%~dp0"
"C:\Program Files\Git\bin\bash.exe" -c "bash /d/Audion-Data/TTS/rasa_voice/run_v2_pipeline.sh" >> v2_pipeline.log 2>&1
echo [%date% %time%] pipeline exited with code %errorlevel% >> v2_pipeline.log
