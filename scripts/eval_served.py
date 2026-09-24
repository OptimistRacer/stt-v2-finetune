"""WER/CER for the model as actually served: the CT2 export through faster-whisper.

scripts/run_day4_eval.py scores the *training* runtime (HF `generate()`), and the
README notes the two runtimes aren't comparable -- so the thing that actually ships has
never been measured. This closes that gap and makes decoding settings testable, since
`transcribe.py` leaves temperature fallback, condition_on_previous_text, VAD, patience
and the thresholds at faster-whisper defaults.

Scores per domain as well as overall. FLEURS-only numbers once hid a severe
code-switching regression on podcast clips (README: 56.2% vs base 26.2%), so a change
that helps read-aloud speech has to be checked against conversational speech too.

    # baseline, current defaults
    python scripts/eval_served.py --split test

    # try one variant
    python scripts/eval_served.py --split test --params '{"condition_on_previous_text": false}'

    # compare several variants on a subset, then verify the winners on everything
    python scripts/eval_served.py --split test --sweep --limit 120
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faster_whisper import WhisperModel

from scripts.transcribe import CODESWITCH_PROMPT, DEFAULT_MODEL
from src.evaluation.transcribe_eval import score
from src.training.dataset import load_manifest

DEFAULT_MANIFEST = r"D:\Audion-Data\Urdu\open_datasets\manifest_combined.jsonl"

# What transcribe.py effectively asks for today: beam 5, VAD off, everything else left
# at faster-whisper's defaults.
BASELINE = {
    "beam_size": 5,
    "vad_filter": False,
    "condition_on_previous_text": True,
}

# One knob at a time, so a win can be attributed. condition_on_previous_text is first
# because carrying previous text forward is the usual cause of repetition/drift.
SWEEP = [
    ("baseline", {}),
    ("no_context", {"condition_on_previous_text": False}),
    ("beam8", {"beam_size": 8}),
    ("beam8_no_context", {"beam_size": 8, "condition_on_previous_text": False}),
    ("vad", {"vad_filter": True}),
    ("no_temp_fallback", {"temperature": [0.0]}),
    ("patience2", {"beam_size": 5, "patience": 2.0}),
    ("best_of5_no_context", {"best_of": 5, "condition_on_previous_text": False}),
]


def transcribe_all(model, records, params, out_path, prompt=None):
    """Streams each clip's hypothesis to `out_path` and resumes from whatever's already
    there -- same crash-resilience as the day-4 harness, which matters because this
    machine has hard-crashed mid-eval before."""
    done = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["clip_id"]] = rec
    pending = [r for r in records if r["clip_id"] not in done]
    if done:
        print(f"  resuming: {len(done)} done, {len(pending)} left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as out_f:
        for i, rec in enumerate(pending, 1):
            segments, _ = model.transcribe(rec["path"], language="ur",
                                           initial_prompt=prompt, **params)
            text = " ".join(s.text.strip() for s in segments).strip()
            out_f.write(json.dumps({**rec, "hypothesis": text}, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                out_f.flush()
                print(f"  {i}/{len(pending)}")
    with open(out_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_domain(results):
    fleurs = [r for r in results if not r["source_id"].startswith("podcast")]
    podcast = [r for r in results if r["source_id"].startswith("podcast")]
    out = {"all": score(results)}
    if fleurs:
        out["fleurs"] = score(fleurs)
    if podcast:
        out["podcast"] = score(podcast)
    return out


def fmt(name, s):
    line = f"  {name:22s} WER {s['all']['wer']:6.2f}%  CER {s['all']['cer']:6.2f}%"
    if "fleurs" in s and "podcast" in s:
        line += f"   | read-aloud {s['fleurs']['wer']:6.2f}%  conversational {s['podcast']['wer']:6.2f}%"
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=DEFAULT_MODEL)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--limit", type=int, default=None, help="cap clips (sweeps run faster on a subset)")
    ap.add_argument("--params", default="{}", help="JSON of decoding params to override the baseline")
    ap.add_argument("--sweep", action="store_true", help="run the built-in list of variants and compare")
    ap.add_argument("--codeswitch_prompt", action="store_true", help="bias English words toward Latin script")
    ap.add_argument("--out_dir", default=r"D:\Audion-Data\Urdu\eval_served")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    records = load_manifest(args.manifest, args.split)
    if args.limit:
        records = records[: args.limit]
    n_pod = sum(1 for r in records if r["source_id"].startswith("podcast"))
    print(f"[eval] {len(records)} clips ({len(records) - n_pod} read-aloud, {n_pod} conversational) "
          f"from split={args.split}")
    print(f"[eval] model: {args.model_dir}")

    model = WhisperModel(args.model_dir, device="cuda", compute_type="float16")
    prompt = CODESWITCH_PROMPT if args.codeswitch_prompt else None
    out_dir = Path(args.out_dir)

    variants = SWEEP if args.sweep else [("custom" + args.tag, json.loads(args.params))]
    report = {}
    for name, overrides in variants:
        params = {**BASELINE, **overrides}
        print(f"\n[eval] {name}: {json.dumps(overrides) if overrides else 'current defaults'}")
        t0 = time.time()
        raw = out_dir / f"{args.split}{'_' + str(args.limit) if args.limit else ''}_{name}.jsonl"
        results = transcribe_all(model, records, params, raw, prompt)
        report[name] = {**by_domain(results), "params": params, "elapsed_s": round(time.time() - t0, 1)}
        print(fmt(name, report[name]))

    summary = out_dir / f"summary_{args.split}{'_' + str(args.limit) if args.limit else ''}{args.tag}.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n[eval] ranked by overall WER:")
    for name, s in sorted(report.items(), key=lambda kv: kv[1]["all"]["wer"]):
        print(fmt(name, s))
    print(f"\n[done] -> {summary}")


if __name__ == "__main__":
    main()
