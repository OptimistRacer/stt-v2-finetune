# Web UI

Local web interface for the Urdu STT model: upload a file or record from the mic.

```bash
python webapp/server.py --preload
```

Then open **http://127.0.0.1:8080**

`--preload` loads the model at startup (~2s) so the first transcription isn't slow.
Without it the model loads on the first request instead.

## Options

| flag | default | |
|---|---|---|
| `--port` | 8080 | 8000 is often unbindable on Windows (Hyper-V/WSL reserve it) |
| `--host` | 127.0.0.1 | use `0.0.0.0` to reach it from other machines on the LAN |
| `--model_dir` | the augmented CT2 export | point at another model to compare |
| `--device` | cuda | `cpu` works but is far slower |

## Notes

- Everything runs on this machine. Audio is written to a temp file, transcribed, and
  deleted; nothing is uploaded anywhere.
- Browser mic recordings arrive as webm/ogg, so ffmpeg must be on PATH.
- The "Write English words in English script" checkbox runs the chunked code-switch
  prompt plus the dictionary pass. It re-applies the prompt every 60s -- a whole-file
  prompt only conditions Whisper's first 30s window and silently does nothing on longer
  audio.
