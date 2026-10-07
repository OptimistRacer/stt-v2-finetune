"""Measure each clip of rasa_male_hf/ the way DataSpeech's main.py does, minus the parts
that don't apply, and save the result for describe_rasa.py.

Same as main.py: pitch mean and spread (penn, on 16 kHz audio) and brouhaha's SNR, C50
(room reverberation) and speech duration -- DataSpeech's own functions are called.
Different, and why:
- penn decodes with "argmax" instead of its default Viterbi: the Viterbi decoder comes
  from a compiled add-on (torbi) that has no kernels for this GPU (RTX 5090, sm_120) and
  fails with "no kernel image is available".
- no speaking rate here: main.py's counts English phonemes, which is meaningless for
  Urdu; describe_rasa.py computes an Urdu one instead.
- no SQUIM quality scores: SNR covers noise for this purpose.

    ..\\venv_dataspeech\\Scripts\\python.exe measure_rasa.py                      # full set
    ..\\venv_dataspeech\\Scripts\\python.exe measure_rasa.py --src smoke_hf --out smoke_tags
"""
import argparse
import sys
from pathlib import Path

import penn
import torch
from datasets import Audio, load_dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dataspeech"))
import dataspeech.gpu_enrichments.pitch as pitch_mod  # noqa: E402
from dataspeech import snr_apply  # noqa: E402


class _ArgmaxPenn:
    """penn, but decoding with argmax (see module docstring)."""

    def __getattr__(self, name):
        return getattr(penn, name)

    def from_audio(self, *args, **kwargs):
        kwargs.setdefault("decoder", "argmax")
        return penn.from_audio(*args, **kwargs)


pitch_mod.penn = _ArgmaxPenn()

# torch >= 2.6 loads checkpoints with weights_only=True and refuses unknown classes. The
# brouhaha checkpoint DataSpeech uses (ylacombe/brouhaha-best) stores a torch version tag
# and pyannote task descriptors; allow exactly those rather than turning the check off.
from pyannote.audio.core.model import Introspection  # noqa: E402
from pyannote.audio.core.task import Problem, Resolution, Specifications  # noqa: E402

torch.serialization.add_safe_globals([torch.torch_version.TorchVersion, Introspection, Problem, Resolution, Specifications])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="rasa_male_hf")
    ap.add_argument("--out", default="rasa_male_tags")
    ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args()
    src, out = HERE / a.src, HERE / a.out

    ds = load_dataset(str(src))
    gpu = torch.cuda.device_count() > 0
    common = dict(batched=True, batch_size=a.batch_size, with_rank=gpu, num_proc=None, remove_columns=["audio"])

    print("pitch ...", flush=True)
    pitch = ds.cast_column("audio", Audio(sampling_rate=16_000)).map(
        pitch_mod.pitch_apply, fn_kwargs={"audio_column_name": "audio", "penn_batch_size": 4096}, **common)
    print("snr, reverberation, speech duration ...", flush=True)
    snr = ds.map(snr_apply, fn_kwargs={"audio_column_name": "audio"}, **common)

    for split in ds.keys():
        pitch[split] = (pitch[split].add_column("snr", snr[split]["snr"]).add_column("c50", snr[split]["c50"])
                        .add_column("speech_duration", snr[split]["speech_duration"]))
    pitch.save_to_disk(str(out))
    for split in pitch.keys():
        d = pitch[split].to_pandas()
        print(split, len(d), d[["utterance_pitch_mean", "utterance_pitch_std", "snr", "c50", "speech_duration"]]
              .describe().loc[["mean", "min", "max"]].round(2).to_string(), sep="\n", flush=True)
    print(f"[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
