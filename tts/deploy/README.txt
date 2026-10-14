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
Voices:      only our own (commercially usable) voices are offered since 2026-10-14:
             Urdu = Parvaaz v2 (helper on port 8097, model D:\Audion-Data\TTS\parvaaz_voice_v2)
             Arabic = our Arabic voice (Chatterbox helper on port 8098, D:\Audion-Data\TTS\arabic_voice)
             Meta and OmniVoice are refused (non-commercial licences).
             Each helper starts on the first request for its language (~26 s Urdu, ~60 s Arabic).
GPU memory:  the speech-to-text model stays loaded while it runs; with both voice helpers
             loaded the total is about 18-19 GB. Stop the app (stop_webapp.ps1) before any
             training on this GPU, or training spills into system RAM and runs ~7x slower.

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
