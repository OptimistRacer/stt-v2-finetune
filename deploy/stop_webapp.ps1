# Stops the merged speech web app completely: the scheduled task (so it is not restarted)
# and the server process. Stop-ScheduledTask alone is not enough -- it ends the supervisor
# but leaves the server it started running.
#
#     powershell -File D:\Audion-Data\TTS\deploy\stop_webapp.ps1
#
# Start it again with:  Start-ScheduledTask -TaskName UrduSpeechWebapp

Stop-ScheduledTask -TaskName UrduSpeechWebapp -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

$owner = Get-NetTCPConnection -State Listen -LocalPort 8095 -ErrorAction SilentlyContinue | Select-Object -First 1
if ($owner) {
    $p = Get-Process -Id $owner.OwningProcess -ErrorAction SilentlyContinue
    if ($p -and $p.ProcessName -eq 'python') {
        Stop-Process -Id $p.Id -Force
        Write-Host "Stopped the server (pid $($p.Id))."
    } else {
        Write-Host "Port 8095 is held by '$($p.ProcessName)', not python -- left alone."
    }
} else {
    Write-Host "Nothing was listening on port 8095."
}

# The OmniVoice, Parvaaz and Arabic voices run as separate helper processes (ports 8096-8098)
# that the app starts on first use and normally stops on exit. A forced kill of the app skips that, so end it here
# too, or it stays running and holds GPU memory.
# Python processes only: a bare filename match would also hit any shell or editor whose
# command line merely mentions the script.
Get-CimInstance Win32_Process |
    Where-Object { $_.ProcessId -ne $PID -and $_.Name -match '^pythonw?\.exe$' -and $_.CommandLine -match '(omnivoice|parvaaz|chatterbox)_worker\.py' } |
    ForEach-Object {
        taskkill /PID $_.ProcessId /T /F | Out-Null
        Write-Host "Stopped a voice helper (pid $($_.ProcessId))."
    }
