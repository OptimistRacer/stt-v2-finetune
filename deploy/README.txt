Deployment of the merged speech web app (transcribe + speak) on this PC
=======================================================================

What runs:   D:\Audion-Data\TTS\webapp\server.py on port 8095, all network interfaces
Started by:  Windows scheduled task "UrduSpeechWebapp", when user Sistems-AI signs in
             (it does NOT run before anyone has signed in to Windows)
Supervisor:  deploy\run_webapp.ps1 keeps the server alive. If the server exits it is
             started again after 15 seconds. If it dies within 2 minutes of starting,
             5 times in a row, the supervisor gives up (see the log) and stays down until
             the next login or Start-ScheduledTask.
Window:      hidden (a console may flash for a moment at login)
Log:         D:\Audion-Data\TTS\logs\webapp.log
Open:        this PC      http://localhost:8095
             other PCs    http://192.168.0.10:8095   (this PC's address on "Ethernet 2")
Firewall:    the existing inbound rule "Urdu Speech Webapp" (Private networks, port 8095).
             Nothing was changed there. There is no login: anyone on that network can use it.
GPU memory:  both models stay loaded, roughly 7-8 GB of GPU memory, while it runs.

Start / stop (PowerShell)
-------------------------
    Start-ScheduledTask -TaskName UrduSpeechWebapp
    powershell -File D:\Audion-Data\TTS\deploy\stop_webapp.ps1

Use stop_webapp.ps1 to stop it. Stop-ScheduledTask on its own ends the supervisor but
leaves the server running; the next Start-ScheduledTask clears that leftover, so it is
harmless, but the app stays up until then.

Stop it from starting at login:   Disable-ScheduledTask    -TaskName UrduSpeechWebapp
Remove it completely:             Unregister-ScheduledTask -TaskName UrduSpeechWebapp -Confirm:$false

Verified on 2026-10-02: start; health check on localhost and on 192.168.0.10; speech to
text and text to speech; recovery after the server process was killed (back in ~25 s);
a second Start does not create a duplicate; Start clears a leftover process on 8095;
stop_webapp.ps1 leaves nothing running.
NOT tested: the start at login itself (needs a sign-out), and access from another device
(the checks ran from this PC, which does not go through the firewall).

Change something
----------------
Only the hidden window, the log path or the restart rules need the task or
run_webapp.ps1 touched. To bind to this PC only, change --host 0.0.0.0 to --host 127.0.0.1
in run_webapp.ps1 and run stop_webapp.ps1 then Start-ScheduledTask.
