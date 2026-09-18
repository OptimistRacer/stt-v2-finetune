"""Remove the background noise from the Iqbal recordings before training.

The recordings carry steady low-frequency rumble plus broadband background (quiet floor
~-49 dB, speech-to-background gap 38.5 dB), and a TTS model copies whatever noise is in
its training audio. High-pass at 80 Hz takes out the rumble, stationary spectral gating
takes out the steady background, then each clip is peak-normalized so levels are
consistent. Tuned to reach the base MMS model's gap (~48.5 dB) at gate strength 0.8;
stronger gating risked "musical noise" artifacts. Verified not to hurt intelligibility
on real clips (STT word error 0.213 raw -> 0.190 cleaned).

Reads iqbal_dataset/ (already text-normalized, 16 kHz) and writes iqbal_dataset_clean/
with identical layout and metadata.
"""
import shutil
from pathlib import Path

import noisereduce as nr
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

SRC = Path("D:/Audion-Data/TTS/iqbal_dataset")
DST = Path("D:/Audion-Data/TTS/iqbal_dataset_clean")
SR = 16000
HIGHPASS_HZ = 80
GATE_STRENGTH = 0.8
PEAK = 0.9

SOS = butter(4, HIGHPASS_HZ, "highpass", fs=SR, output="sos")


def clean(x):
    y = sosfiltfilt(SOS, x).astype(np.float32)
    y = nr.reduce_noise(y=y, sr=SR, stationary=True, prop_decrease=GATE_STRENGTH)
    return (y / (np.max(np.abs(y)) + 1e-9) * PEAK).astype(np.float32)


def main():
    n = 0
    for split in ["train", "validation"]:
        (DST / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(SRC / split / "metadata.csv", DST / split / "metadata.csv")
        for wav in sorted((SRC / split).glob("*.wav")):
            x, sr = sf.read(wav, dtype="float32")
            assert sr == SR, f"{wav} is {sr} Hz"
            sf.write(DST / split / wav.name, clean(x), SR, subtype="PCM_16")
            n += 1
    shutil.copy(SRC / "manifest.json", DST / "manifest.json")
    print(f"[clean] {n} clips -> {DST}")


if __name__ == "__main__":
    main()
