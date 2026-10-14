"""Phase 1 audit of the Parvaaz shorts: is there enough clean, single-speaker Urdu speech
in them to fine-tune a voice we own? (Plan: C:\\Users\\Sistems-AI\\.claude\\plans\\
for-urdu-omni-voices-glowing-wirth.md)

For each file in shorts_raw/ it measures what decides that:
- music bed: Demucs splits voice from accompaniment (in its own venv, venv_demucs/);
  the accompaniment level under speech, relative to the voice, says whether the clip
  is clean, salvageable, or unusable. Everything after this works on the voice stem.
- speech time (faster-whisper's Silero VAD), noise floor and speech-to-background gap,
  clipping
- speakers: pyannote diarization (how many voices, main speaker's share, overlap) and a
  wespeaker embedding of the main speaker, compared across all shorts to catch ones
  where someone else is the main voice
- a draft transcript with word timings from the served STT model (code-switch prompt,
  English kept in Latin script) -- the starting point for the human correction pass
- training clips: the main speaker's speech cut at pauses into 2-15 s pieces, dropping
  pieces that overlap another voice or sit on heavy music

and writes audit/report.json (everything, including per-clip draft text) and
audit/report.html (totals, where minutes were lost, per-file table, listening samples,
and a go / no-go). Thresholds are starting proposals, to be tuned by ear.

    python audit_shorts.py --limit 3        # smoke test on the first 3 files
    python audit_shorts.py                  # everything in shorts_raw/
    python audit_shorts.py --no_demucs      # skip music separation (no music numbers)

Inputs are never modified. Re-runs reuse extracted audio, stems and per-file results
already on disk, so an interrupted run resumes.
"""
import argparse
import base64
import html
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
RAW = HERE / "shorts_raw"
WORK = HERE / "work"
STEMS = HERE / "stems"
AUDIT = HERE / "audit"
DEMUCS_PY = HERE / "venv_demucs" / "Scripts" / "python.exe"


def set_root(root: Path):
    """Point every input/output folder at another root (used for smoke tests, so test
    files never mix with the real shorts or their cached results)."""
    global RAW, WORK, STEMS, AUDIT
    RAW, WORK, STEMS, AUDIT = root / "shorts_raw", root / "work", root / "stems", root / "audit"
STT_REPO = Path(r"C:\Users\Sistems-AI\stt-v2-finetune")
AUDIO_EXT = {".mp4", ".m4a", ".mp3", ".wav", ".webm", ".mkv", ".mov", ".aac", ".ogg", ".flac", ".opus"}

SR = 16000
FRAME = SR // 50  # 20 ms analysis frames

# Starting thresholds (plan): tune after listening.
MUSIC_NONE_DB, MUSIC_HEAVY_DB = -30.0, -20.0
CLIP_MIN_S, CLIP_MAX_S, CLIP_JOIN_GAP_S = 2.0, 15.0, 0.8
OVERLAP_TOL_S = 0.2
GO_MIN, BORDER_MIN = 45.0, 20.0


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- audio helpers

def ffprobe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                         capture_output=True, text=True, encoding="utf-8")
    info = json.loads(out.stdout or "{}")
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {})
    fmt = info.get("format", {})
    return {
        "container": fmt.get("format_name"),
        "duration_s": float(fmt.get("duration") or a.get("duration") or 0.0),
        "codec": a.get("codec_name"),
        "sample_rate": int(a["sample_rate"]) if a.get("sample_rate") else None,
        "channels": a.get("channels"),
        "bitrate_kbps": round(int(a["bit_rate"]) / 1000) if a.get("bit_rate") else None,
        "has_audio": bool(a),
    }


def extract(src: Path, dst: Path):
    """44.1 kHz stereo wav for Demucs; the original stays untouched."""
    if not dst.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "2", "-ar", "44100", str(dst)],
                       check=True)


def to_mono16k(path: Path) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    if sr != SR:
        g = np.gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype("float32")
    return x


def frame_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    if n == 0:
        return np.array([-120.0])
    f = x[: n * FRAME].reshape(n, FRAME)
    return 20 * np.log10(np.sqrt((f ** 2).mean(axis=1)) + 1e-9)


def level_db(db: np.ndarray) -> float | None:
    """Average power of a set of frames, in dB. A median of frame levels is dragged down
    by the quiet gaps between words and misjudges how loud speech (or music) really is."""
    if db.size == 0:
        return None
    return float(10 * np.log10(np.mean(10 ** (db / 10)) + 1e-12))


def wav_b64(x: np.ndarray) -> str:
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1, 1), SR, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- intervals

def total(iv):
    return sum(e - s for s, e in iv)


def intersect(a, b):
    out, i, j = [], 0, 0
    a, b = sorted(a), sorted(b)
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def overlap_regions(turns):
    """Where two or more speakers talk at once."""
    pts = sorted([(t["start"], 1) for t in turns] + [(t["end"], -1) for t in turns])
    out, depth, start = [], 0, None
    for p, d in pts:
        prev = depth
        depth += d
        if prev < 2 <= depth:
            start = p
        elif prev >= 2 > depth and start is not None:
            out.append((start, p))
    return out


# ---------------------------------------------------------------- stages

def run_demucs(wavs: list[Path]):
    pending = [w for w in wavs if not (STEMS / "htdemucs" / w.stem / "vocals.wav").exists()]
    if not pending:
        return
    if not DEMUCS_PY.exists():
        sys.exit(f"Demucs venv missing at {DEMUCS_PY} -- build it, or run with --no_demucs")
    log(f"[demucs] separating {len(pending)} file(s) ...")
    for i in range(0, len(pending), 20):  # batches keep the command line short
        batch = pending[i:i + 20]
        subprocess.run([str(DEMUCS_PY), "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs", "-d", "cuda",
                        "-o", str(STEMS), *map(str, batch)], check=True)


class Models:
    def __init__(self):
        import torch
        from faster_whisper import WhisperModel
        from huggingface_hub import get_token
        from pyannote.audio import Pipeline
        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

        sys.path.insert(0, str(STT_REPO))
        from scripts.transcribe import CODESWITCH_PROMPT, DEFAULT_MODEL

        self.torch = torch
        token = get_token()
        log("[models] loading diarization, speaker embedding and STT ...")
        self.diar = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
        self.diar.to(torch.device("cuda"))
        self.emb = PretrainedSpeakerEmbedding("pyannote/wespeaker-voxceleb-resnet34-LM",
                                              device=torch.device("cuda"), token=token)
        self.stt = WhisperModel(DEFAULT_MODEL, device="cuda", compute_type="float16")
        self.prompt = CODESWITCH_PROMPT

    def diarize(self, x):
        out = self.diar({"waveform": self.torch.from_numpy(x)[None, :], "sample_rate": SR})
        ann = getattr(out, "speaker_diarization", out)  # pyannote 4.x wraps the Annotation
        return [{"start": t.start, "end": t.end, "speaker": s} for t, _, s in ann.itertracks(yield_label=True)]

    def embed(self, x):
        e = np.asarray(self.emb(self.torch.from_numpy(x)[None, None, :])).reshape(-1)
        n = np.linalg.norm(e)
        return e / n if n > 0 and np.isfinite(n) else None

    def transcribe(self, x, speech):
        """Decode in <=60 s pieces cut at VAD gaps (so cuts land in pauses), re-applying
        the code-switch prompt to each piece; returns words with absolute times."""
        pieces, cur = [], None
        for s, e in speech:
            if cur and e - cur[0] > 60.0:
                pieces.append(cur)
                cur = None
            cur = (cur[0], e) if cur else (s, e)
        if cur:
            pieces.append(cur)
        words, logprobs = [], []
        for s, e in pieces:
            a, b = max(0, int((s - 0.2) * SR)), min(len(x), int((e + 0.2) * SR))
            off = a / SR
            segs, _ = self.stt.transcribe(x[a:b], language="ur", beam_size=5, vad_filter=False,
                                          initial_prompt=self.prompt, word_timestamps=True)
            for seg in segs:
                logprobs.append(seg.avg_logprob)
                for w in seg.words or []:
                    if w.word.strip():
                        words.append({"start": w.start + off, "end": w.end + off, "word": w.word.strip()})
        return words, (float(np.mean(logprobs)) if logprobs else None)


def split_long(region, words, voc_db):
    """Split a stretch longer than CLIP_MAX_S with no VAD pause in it: cut at the widest
    gap between words (at least CLIP_MIN_S from either end), or failing that at the
    quietest 20 ms frame in its middle half. Recursive until every piece fits."""
    s, e = region
    if e - s <= CLIP_MAX_S:
        return [region]
    inside = [w for w in words if s <= w["start"] and w["end"] <= e]
    gaps = [((a["end"] + b["start"]) / 2, b["start"] - a["end"]) for a, b in zip(inside, inside[1:])
            if s + CLIP_MIN_S <= (a["end"] + b["start"]) / 2 <= e - CLIP_MIN_S]
    if gaps:
        mid = (s + e) / 2
        cut = max(gaps, key=lambda g: (g[1], -abs(g[0] - mid)))[0]
    else:
        a, b = int((s + (e - s) / 4) * 50), int((e - (e - s) / 4) * 50)
        cut = (a + int(np.argmin(voc_db[a:b]))) / 50 if b > a else (s + e) / 2
    return split_long((s, cut), words, voc_db) + split_long((cut, e), words, voc_db)


def build_clips(regions, words, voc_db, acc_db, others):
    """Group the main speaker's speech regions into 2-15 s clips cut at pauses."""
    regions = [p for r in sorted(regions) for p in split_long(r, words, voc_db)]
    clips, cur = [], None
    for s, e in sorted(regions):
        if cur and (s - cur[1] <= CLIP_JOIN_GAP_S) and (e - cur[0] <= CLIP_MAX_S):
            cur = (cur[0], e)
        else:
            if cur:
                clips.append(cur)
            cur = (s, e)
    if cur:
        clips.append(cur)

    out = []
    for s, e in clips:
        d = e - s
        c = {"start": round(s, 2), "end": round(e, 2), "dur": round(d, 2), "drop": None}
        if d < CLIP_MIN_S:
            c["drop"] = "too_short"
        elif d > CLIP_MAX_S:
            c["drop"] = "too_long"  # a single unbroken stretch with no pause to cut at
        elif total(intersect([(s, e)], others)) > OVERLAP_TOL_S:
            c["drop"] = "other_voice"
        a, b = int(s * 50), max(int(s * 50) + 1, int(e * 50))
        if acc_db is not None:
            m = level_db(acc_db[a:b]) - level_db(voc_db[a:b])
            c["music_db"] = round(m, 1)
            if c["drop"] is None and m > MUSIC_HEAVY_DB:
                c["drop"] = "heavy_music"
        c["draft"] = " ".join(w["word"] for w in words if s <= (w["start"] + w["end"]) / 2 < e)
        if c["drop"] is None and not c["draft"]:
            c["drop"] = "no_words"
        out.append(c)
    return out


def analyse(fid, src, models, use_demucs):
    meta = {"id": fid, "file": src.name, **ffprobe(src)}
    if not meta["has_audio"]:
        return {**meta, "error": "no audio stream"}
    wav = WORK / f"{fid}.wav"
    extract(src, wav)
    orig = to_mono16k(wav)
    if use_demucs:
        sd = STEMS / "htdemucs" / fid
        voice, acc = to_mono16k(sd / "vocals.wav"), to_mono16k(sd / "no_vocals.wav")
    else:
        voice, acc = orig, None
    n = min(len(voice), len(orig), len(acc) if acc is not None else len(voice))
    voice, orig = voice[:n], orig[:n]
    acc = acc[:n] if acc is not None else None

    from faster_whisper.vad import VadOptions, get_speech_timestamps
    ts = get_speech_timestamps(voice, VadOptions(min_silence_duration_ms=300, speech_pad_ms=100), sampling_rate=SR)
    speech = [(t["start"] / SR, t["end"] / SR) for t in ts]

    voc_db = frame_db(voice)
    acc_db = frame_db(acc) if acc is not None else None
    mask = np.zeros(len(voc_db), bool)
    for s, e in speech:
        mask[int(s * 50):int(e * 50)] = True
    voice_level = level_db(voc_db[mask]) if mask.any() else None
    music = None
    if acc_db is not None and mask.any():
        music = level_db(acc_db[mask]) - voice_level
    # Noise is judged on the original: Demucs moves background noise out of the voice
    # stem, so a floor measured there looks unrealistically clean.
    orig_db = frame_db(orig)[: len(mask)]
    om = mask[: len(orig_db)]
    floor = float(np.percentile(orig_db[~om], 10)) if (~om).sum() > 25 else float(np.percentile(orig_db, 5))
    speech_level_orig = level_db(orig_db[om]) if om.any() else None

    turns = models.diarize(voice) if speech else []
    talk = {}
    for t in turns:
        talk[t["speaker"]] = talk.get(t["speaker"], 0.0) + t["end"] - t["start"]
    main = max(talk, key=talk.get) if talk else None
    main_turns = [(t["start"], t["end"]) for t in turns if t["speaker"] == main]
    other_turns = [(t["start"], t["end"]) for t in turns if t["speaker"] != main]
    overlaps = overlap_regions(turns)

    emb = None
    long_turns = [(s, e) for s, e in main_turns if e - s >= 1.0]
    if long_turns:
        chunks, got = [], 0.0
        for s, e in sorted(long_turns, key=lambda t: t[1] - t[0], reverse=True):
            chunks.append(voice[int(s * SR):int(e * SR)])
            got += e - s
            if got >= 30:
                break
        emb = models.embed(np.concatenate(chunks))

    words, conf = models.transcribe(voice, speech) if speech else ([], None)
    latin = sum(1 for w in words if re.search(r"[A-Za-z]", w["word"]))

    regions = intersect(speech, main_turns) if main_turns else []
    clips = build_clips(regions, words, voc_db, acc_db, other_turns + overlaps)

    return {
        **meta,
        "speech_s": round(total(speech), 1),
        "voice_db": round(voice_level, 1) if voice_level is not None else None,
        "noise_floor_db": round(floor, 1),
        "gap_db": round(speech_level_orig - floor, 1) if speech_level_orig is not None else None,
        "clipped_pct": round(100 * float((np.abs(orig) >= 0.999).mean()), 3),
        "music_db": round(music, 1) if music is not None else None,
        "music_class": (None if music is None else "none" if music < MUSIC_NONE_DB
                        else "light" if music <= MUSIC_HEAVY_DB else "heavy"),
        "n_speakers": len(talk),
        "main_share": round(talk[main] / sum(talk.values()), 3) if talk else None,
        "overlap_s": round(total(overlaps), 1),
        "other_voice_s": round(total(other_turns), 1),
        "n_words": len(words),
        "english_pct": round(100 * latin / len(words), 1) if words else None,
        "stt_logprob": round(conf, 3) if conf is not None else None,
        "transcript": " ".join(w["word"] for w in words),
        "clips": clips,
        "_emb": emb.tolist() if emb is not None else None,
    }


# ---------------------------------------------------------------- report

def speaker_check(results):
    have = [r for r in results if r.get("_emb")]
    if len(have) < 2:
        for r in results:
            r["speaker_sim"], r["speaker_flag"] = None, False
        return None
    E = np.array([r["_emb"] for r in have])
    c = E.mean(axis=0)
    c /= np.linalg.norm(c)
    sims = E @ c
    keep = sims >= 0.4  # recompute without obvious outliers so they don't pull the centroid
    if keep.sum() >= 2:
        c = E[keep].mean(axis=0)
        c /= np.linalg.norm(c)
        sims = E @ c
    med = float(np.median(sims))
    cut = min(med - 0.15, 0.6)
    for r, s in zip(have, sims):
        r["speaker_sim"] = round(float(s), 3)
        r["speaker_flag"] = bool(s < cut)
    for r in results:
        r.setdefault("speaker_sim", None)
        r.setdefault("speaker_flag", False)
    return {"median_sim": round(med, 3), "flag_below": round(cut, 3)}


def summarise(results):
    S = {"files": len(results), "raw_min": 0.0, "speech_min": 0.0, "lost_min": {}, "usable_min": 0.0, "usable_clips": 0}
    for r in results:
        S["raw_min"] += r.get("duration_s", 0) / 60
        S["speech_min"] += r.get("speech_s", 0) / 60
        for c in r.get("clips", []):
            reason = "other_main_speaker" if (r.get("speaker_flag") and c["drop"] is None) else c["drop"]
            if reason is None:
                S["usable_min"] += c["dur"] / 60
                S["usable_clips"] += 1
            else:
                S["lost_min"][reason] = S["lost_min"].get(reason, 0) + c["dur"] / 60
    main_speech = sum(c["dur"] for r in results for c in r.get("clips", [])) / 60
    S["lost_min"]["other_voices_or_silence_gaps"] = S["speech_min"] - main_speech
    for k in ("raw_min", "speech_min", "usable_min"):
        S[k] = round(S[k], 1)
    S["lost_min"] = {k: round(v, 1) for k, v in sorted(S["lost_min"].items(), key=lambda kv: -kv[1])}
    u = S["usable_min"]
    S["verdict"] = ("GO" if u >= GO_MIN else "BORDERLINE" if u >= BORDER_MIN else "NO-GO (shorts alone)")
    return S


def excerpt(fid, which, start, secs=8.0):
    base = STEMS / "htdemucs" / fid / "vocals.wav" if which == "voice" else WORK / f"{fid}.wav"
    if not base.exists():
        return None
    x = to_mono16k(base)
    a = int(start * SR)
    return wav_b64(x[a:a + int(secs * SR)])


def write_html(results, S, spk, path):
    def best_clip(r):
        ok = [c for c in r.get("clips", []) if c["drop"] is None]
        return (ok or r.get("clips") or [{"start": 0}])[0]["start"]

    samples = []
    usable = sorted([r for r in results if r.get("clips")], key=lambda r: -sum(c["dur"] for c in r["clips"] if c["drop"] is None))
    for r in usable[:2]:
        samples.append((f"Clean usable speech: {r['file']}", [("original", excerpt(r["id"], "orig", best_clip(r)))]))
    music = sorted([r for r in results if r.get("music_db") is not None], key=lambda r: -r["music_db"])
    for r in music[:2]:
        if r["music_db"] > MUSIC_NONE_DB:
            st = best_clip(r)
            samples.append((f"Music bed ({r['music_db']} dB): {r['file']} -- is the voice-only version clean, or does it sound damaged?",
                            [("original", excerpt(r["id"], "orig", st)), ("music removed", excerpt(r["id"], "voice", st))]))
    for r in [r for r in results if r.get("speaker_flag")][:2]:
        samples.append((f"Flagged as a different main voice (similarity {r['speaker_sim']}): {r['file']}",
                        [("original", excerpt(r["id"], "orig", best_clip(r)))]))

    rows = []
    for r in results:
        usable_m = sum(c["dur"] for c in r.get("clips", []) if c["drop"] is None and not r.get("speaker_flag")) / 60
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in [
            r["id"], r["file"], round(r.get("duration_s", 0), 1), round(r.get("speech_s", 0), 1),
            r.get("music_db"), r.get("music_class"), r.get("gap_db"), r.get("n_speakers"), r.get("main_share"),
            r.get("speaker_sim"), "yes" if r.get("speaker_flag") else "", r.get("english_pct"),
            round(usable_m, 2), r.get("error", "")]) + "</tr>")

    lost = "".join(f"<li>{html.escape(k.replace('_', ' '))}: {v} min</li>" for k, v in S["lost_min"].items())
    samp = "".join(
        f"<div class='s'><p>{html.escape(t)}</p>" + "".join(
            f"<figure><figcaption>{l}</figcaption><audio controls preload='none' src='{a}'></audio></figure>"
            for l, a in pairs if a) + "</div>" for t, pairs in samples)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Shorts audit</title>
<style>
:root{{--bg:#fff;--fg:#1b1f27;--mut:#5b6472;--line:#dde1e8;--card:#f6f7f9}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f1115;--fg:#e7e9ee;--mut:#9aa3b2;--line:#262b36;--card:#171a21}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,"Segoe UI",sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px 48px}} h1{{font-size:22px;margin:0}}
.v{{font-size:20px;font-weight:700;margin:14px 0}} .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}}
.wrap{{overflow-x:auto}} figure{{margin:4px 0}} figcaption{{font-size:12px;color:var(--mut)}} audio{{width:100%;max-width:420px}}
.mut{{color:var(--mut);font-size:13px}}
</style></head><body><main>
<h1>Parvaaz shorts: audit for training a voice</h1>
<p class="mut">Generated {time.strftime('%Y-%m-%d %H:%M')}. Thresholds are starting proposals; your ears decide.</p>
<div class="v">Verdict: {S['verdict']} -- {S['usable_min']} usable minutes in {S['usable_clips']} clips</div>
<div class="card"><b>From {S['files']} files:</b> {S['raw_min']} min of audio, {S['speech_min']} min of speech.
<br>Where minutes were lost:<ul>{lost}</ul>
<span class="mut">Go: 45+ usable min. Borderline (add podcast host turns): 20-45. Below 20: shorts alone aren't enough.
Speaker check: median similarity {spk['median_sim'] if spk else '-'}, flagged below {spk['flag_below'] if spk else '-'}.</span></div>
<h2>Listen</h2>{samp or '<p class="mut">No samples.</p>'}
<h2>Per file</h2><div class="wrap"><table><tr><th>id</th><th>file</th><th>dur s</th><th>speech s</th><th>music dB</th>
<th>music</th><th>gap dB</th><th>speakers</th><th>main share</th><th>voice sim</th><th>other voice?</th><th>English %</th>
<th>usable min</th><th>error</th></tr>{''.join(rows)}</table></div>
</main></body></html>"""
    path.write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_demucs", action="store_true")
    ap.add_argument("--root", default=None, help="use <root>/shorts_raw etc. instead of this folder")
    a = ap.parse_args()
    if a.root:
        set_root(Path(a.root))

    files = sorted(p for p in RAW.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if a.limit:
        files = files[:a.limit]
    if not files:
        sys.exit(f"No audio/video files in {RAW}")
    WORK.mkdir(exist_ok=True)
    AUDIT.mkdir(exist_ok=True)
    per_file = AUDIT / "per_file"
    per_file.mkdir(exist_ok=True)

    # Stable ids (s0001...) so odd characters in titles never reach a command line.
    idmap_path = AUDIT / "ids.json"
    idmap = json.loads(idmap_path.read_text(encoding="utf-8")) if idmap_path.exists() else {}
    for f in files:
        key = str(f.relative_to(RAW))
        if key not in idmap:
            idmap[key] = f"s{len(idmap) + 1:04d}"
    idmap_path.write_text(json.dumps(idmap, ensure_ascii=False, indent=1), encoding="utf-8")

    log(f"[audit] {len(files)} file(s)")
    for f in files:
        extract(f, WORK / f"{idmap[str(f.relative_to(RAW))]}.wav")
    if not a.no_demucs:
        run_demucs([WORK / f"{idmap[str(f.relative_to(RAW))]}.wav" for f in files])

    models = None
    results = []
    for i, f in enumerate(files, 1):
        fid = idmap[str(f.relative_to(RAW))]
        cache = per_file / f"{fid}.json"
        if cache.exists():
            results.append(json.loads(cache.read_text(encoding="utf-8")))
            continue
        models = models or Models()
        t0 = time.time()
        try:
            r = analyse(fid, f, models, not a.no_demucs)
        except Exception as e:  # one bad file shouldn't sink the audit; it's reported
            r = {"id": fid, "file": f.name, "error": f"{type(e).__name__}: {e}"}
        cache.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        results.append(r)
        log(f"[{i}/{len(files)}] {fid} {r.get('duration_s', 0):.0f}s  speech {r.get('speech_s', 0):.0f}s  "
            f"music {r.get('music_db')}  speakers {r.get('n_speakers')}  ({time.time() - t0:.0f}s)"
            + (f"  ERROR {r['error']}" if r.get("error") else ""))

    spk = speaker_check(results)
    S = summarise(results)
    out = {"summary": S, "speaker_check": spk,
           "thresholds": {"music_none_db": MUSIC_NONE_DB, "music_heavy_db": MUSIC_HEAVY_DB, "clip_s": [CLIP_MIN_S, CLIP_MAX_S],
                          "go_min": GO_MIN, "borderline_min": BORDER_MIN},
           "files": [{k: v for k, v in r.items() if k != "_emb"} for r in results]}
    (AUDIT / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    write_html(results, S, spk, AUDIT / "report.html")
    log(f"[done] {S['verdict']}: {S['usable_min']} usable min of {S['speech_min']} speech min "
        f"({S['raw_min']} min audio) -> {AUDIT / 'report.html'}")


if __name__ == "__main__":
    main()
