@echo off
REM Starts the full Parvaaz fine-tune detached from any terminal, logging to train_full.log.
REM Re-running resumes nothing on its own: it starts the run in parvaaz_train\ (see train_parvaaz.sh).
cd /d "%~dp0"
echo [%date% %time%] starting full training >> train_full.log
"C:\Program Files\Git\bin\bash.exe" -c "bash /d/Audion-Data/TTS/rasa_voice/train_parvaaz.sh full" >> train_full.log 2>&1
echo [%date% %time%] training exited with code %errorlevel% >> train_full.log
