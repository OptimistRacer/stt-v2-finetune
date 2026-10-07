# Supervisor for the merged speech web app (transcribe + speak), port 8095.
# Started at Windows login by the scheduled task "UrduSpeechWebapp" (see deploy\README.txt).
#
# Restarts the server whenever it exits. Task Scheduler's own restart-on-failure was not
# used: the task starts this through "conhost --headless", which reports success to the
# scheduler whatever the program inside returned, so the scheduler never sees a failure.
#
# Crash-loop guard: if the server dies within 2 minutes of starting, 5 times in a row,
# this gives up (exit 1) rather than hammering the GPU. A run that lasts longer than
# 2 minutes resets the count.

Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force logs | Out-Null
$log = Join-Path (Get-Location) 'logs\webapp.log'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

function Log($msg) { "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg | Add-Content -Path $log }

# Task Scheduler ends this script when the task is stopped, but not the server it launched,
# which is left running. Clear such a leftover so a fresh start gets port 8095 instead of
# silently falling back to another port. Only a python process is touched.
function Clear-StaleServer {
    $owner = Get-NetTCPConnection -State Listen -LocalPort 8095 -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $owner) { return }
    $p = Get-Process -Id $owner.OwningProcess -ErrorAction SilentlyContinue
    if ($p -and $p.ProcessName -eq 'python') {
        Log "stopping leftover server (pid $($p.Id)) on port 8095"
        Stop-Process -Id $p.Id -Force
        Start-Sleep -Seconds 3
    }
}

$quickFailures = 0
while ($true) {
    Clear-StaleServer
    $t0 = Get-Date
    Log 'starting server'
    # cmd does the redirection so python's stdout and stderr both land in the log as-is.
    & cmd.exe /c "C:\Python314\python.exe webapp\server.py --host 0.0.0.0 --preload >> logs\webapp.log 2>&1"
    $rc = $LASTEXITCODE
    $ran = [int]((Get-Date) - $t0).TotalSeconds
    Log "server exited with code $rc after ${ran}s"

    if ($ran -lt 120) { $quickFailures++ } else { $quickFailures = 0 }
    if ($quickFailures -ge 5) {
        Log 'giving up: 5 quick failures in a row. Fix the cause, then run Start-ScheduledTask -TaskName UrduSpeechWebapp'
        exit 1
    }
    Start-Sleep -Seconds 15
}
