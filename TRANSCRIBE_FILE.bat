@echo off
REM Drag an audio file onto this .bat to transcribe it.
REM Writes <filename>.txt and <filename>.srt next to the audio file.

cd /d "%~dp0"
title Urdu STT - Transcribe a file

if "%~1"=="" (
  echo.
  echo   ============================================
  echo     URDU SPEECH TO TEXT - file transcription
  echo   ============================================
  echo.
  echo   Drag an audio file ONTO this .bat file to transcribe it.
  echo.
  echo   Accepts mp3, wav, m4a, mp4, ogg, webm.
  echo   Writes a .txt and a .srt beside the audio file.
  echo.
  pause
  exit /b
)

echo.
echo   Transcribing: %~nx1
echo   This takes roughly 5 seconds per minute of audio.
echo.

python scripts\transcribe.py "%~1" --codeswitch_prompt --fix_codeswitch --format text --out "%~dpn1.txt"
if errorlevel 1 goto failed

python scripts\transcribe.py "%~1" --codeswitch_prompt --fix_codeswitch --format srt --out "%~dpn1.srt"
if errorlevel 1 goto failed

echo.
echo   Done. Created:
echo     %~dpn1.txt
echo     %~dpn1.srt
echo.
pause
exit /b

:failed
echo.
echo   Transcription failed - see the error above.
echo.
pause
