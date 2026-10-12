"""Blind listening test: Parvaaz vs OmniVoice vs Meta on the 6 shared sentences.

Per sentence the three voices play in a random order as A / B / C. The listener rates
each 1-5 for naturalness (5 = sounds like a real person, 1 = robotic), then can reveal
which was which and download the ratings as JSON. Ratings stay in the browser
(localStorage) so a reload doesn't lose them.

Every clip is loudness-matched (same RMS) before embedding: louder clips tend to be
preferred, which would muddy a naturalness judgement. Sample rates are left native
(Meta 16 kHz, OmniVoice 24 kHz, Parler 44.1 kHz) -- that is what each would serve.

    python build_blind_test.py s300            (Urdu: Parvaaz vs OmniVoice vs Meta; main environment)
    python build_blind_test.py arabic --lang ar --voices voxcpm=VoxCPM2 chatterbox=Chatterbox ...
                                               (any voices under candidates/out/<dir>/)
Writes C:\\Users\\Sistems-AI\\stt-v2-finetune\\outputs\\blind_test_<tag>.html
"""
import base64
import html
import io
import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
CAND = HERE.parent / "candidates"
OUT_DIR = Path(r"C:\Users\Sistems-AI\stt-v2-finetune\outputs")
TARGET_RMS_DB = -20.0


def clip_uri(path: Path) -> str:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    x = np.clip(x * (10 ** (TARGET_RMS_DB / 20) / rms), -1, 1)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--lang", default="ur")
    ap.add_argument("--voices", nargs="+", default=None, help="dir=Label pairs, dirs under candidates/out/")
    a = ap.parse_args()
    tag, lang = a.tag, a.lang
    if a.voices:
        voices = dict(v.split("=", 1) for v in a.voices)
    else:
        voices = {f"parvaaz_{tag}": "Parvaaz (ours, commercial OK)", "omnivoice": "OmniVoice (non-commercial)",
                  "meta": "Meta Standard (non-commercial)"}
    sentences = json.loads((CAND / "sentences.json").read_text(encoding="utf-8"))[lang]
    rng = random.Random(2026)
    key, blocks = {}, []
    for i, text in enumerate(sentences):
        order = list(voices)
        rng.shuffle(order)
        key[i] = {chr(65 + j): voices[v] for j, v in enumerate(order)}
        rows = []
        for j, v in enumerate(order):
            letter = chr(65 + j)
            radios = "".join(f'<label><input type="radio" name="s{i}{letter}" value="{n}">{n}</label>' for n in range(1, 6))
            rows.append(f'<div class="clip"><b>{letter}</b><audio controls preload="none" src="{clip_uri(CAND / "out" / v / f"{lang}_{i}.wav")}"></audio>'
                        f'<div class="rate" data-s="{i}" data-l="{letter}">{radios}</div><span class="who" data-s="{i}" data-l="{letter}"></span></div>')
        blocks.append(f'<div class="item"><p class="sent" dir="rtl" lang="{lang}">{html.escape(text)}</p>{"".join(rows)}</div>')

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind voice test</title><style>
:root{{--bg:#fff;--fg:#1b1f27;--mut:#5b6472;--line:#dde1e8;--card:#f6f7f9;--acc:#2f6fdf}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f1115;--fg:#e7e9ee;--mut:#9aa3b2;--line:#262b36;--card:#171a21;--acc:#6ea0ff}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,"Segoe UI",sans-serif}}
main{{max-width:760px;margin:0 auto;padding:24px 16px 64px}} h1{{font-size:22px;margin:0 0 6px}} .lead{{color:var(--mut)}}
.item{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:14px 0}}
.sent{{font-size:20px;line-height:1.9;margin:0 0 8px}} .clip{{display:grid;grid-template-columns:24px 1fr;gap:4px 10px;align-items:center;margin:10px 0}}
audio{{width:100%}} .rate{{grid-column:2;display:flex;gap:14px;font-size:14px}} .rate label{{display:flex;gap:4px;align-items:center}}
.who{{grid-column:2;font-size:13px;font-weight:600;color:var(--acc)}} button{{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);
background:var(--card);color:var(--fg);cursor:pointer;margin-right:8px}} #summary{{margin-top:14px}}
</style></head><body><main>
<h1>Blind test: which voice sounds most natural?</h1>
<p class="lead">For each sentence, {len(voices)} voices play as {', '.join(chr(65 + j) for j in range(len(voices)))}, shuffled every time. Rate each one:
<b>5</b> = sounds like a real person, <b>1</b> = robotic. When you're done, press Reveal, then Download and send me the file.</p>
{''.join(blocks)}
<button id="reveal">Reveal which is which</button><button id="dl">Download my ratings</button><div id="summary"></div>
<script>
const KEY = {json.dumps(key, ensure_ascii=False)};
const STORE = "blind_{tag}";
let saved = {{}}; try {{ saved = JSON.parse(localStorage.getItem(STORE) || "{{}}"); }} catch (e) {{}}
document.querySelectorAll(".rate input").forEach(inp => {{
  if (saved[inp.name] == inp.value) inp.checked = true;
  inp.addEventListener("change", () => {{ saved[inp.name] = inp.value; try {{ localStorage.setItem(STORE, JSON.stringify(saved)); }} catch (e) {{}} }});
}});
function results() {{
  const per = {{}};
  for (const [s, map] of Object.entries(KEY)) for (const [l, voice] of Object.entries(map)) {{
    const v = saved["s" + s + l]; if (!v) continue; (per[voice] = per[voice] || []).push(+v);
  }}
  return per;
}}
document.getElementById("reveal").onclick = () => {{
  document.querySelectorAll(".who").forEach(el => el.textContent = KEY[el.dataset.s][el.dataset.l]);
  const per = results();
  document.getElementById("summary").innerHTML = Object.entries(per).map(([v, xs]) =>
    `<div>${{v}}: average <b>${{(xs.reduce((a, b) => a + b, 0) / xs.length).toFixed(2)}}</b> over ${{xs.length}} ratings</div>`).join("") || "No ratings yet.";
}};
document.getElementById("dl").onclick = () => {{
  const blob = new Blob([JSON.stringify({{ratings: saved, key: KEY, averages: results()}}, null, 1)], {{type: "application/json"}});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "blind_test_{tag}_ratings.json"; a.click();
}};
</script></main></body></html>"""
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"blind_test_{tag}.html"
    out.write_text(page, encoding="utf-8")
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
