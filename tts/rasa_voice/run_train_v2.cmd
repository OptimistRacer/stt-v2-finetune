@echo off
REM Starts the Parvaaz v2 fine-tune (male + female) detached from any terminal, logging to train_v2.log.
REM Re-running resumes nothing on its own: it starts the run in parvaaz_v2_train\ (see train_parvaaz.sh).
cd /d "%~dp0"
echo [%date% %time%] starting v2 training >> train_v2.log
"C:\Program Files\Git\bin\bash.exe" -c "bash /d/Audion-Data/TTS/rasa_voice/train_parvaaz.sh v2" >> train_v2.log 2>&1
echo [%date% %time%] training exited with code %errorlevel% >> train_v2.log
