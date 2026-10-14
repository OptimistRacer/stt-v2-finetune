"""Are the LJ* and 1LJ* groups in the Iqbal dataset the same reciter? Mixing two voices
in one TTS training set blurs the result, so compare speaker embeddings across groups
against the within-group baseline.
"""
import glob
import json
import random

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import get_token
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

random.seed(0)
emb_model = PretrainedSpeakerEmbedding(
    "pyannote/wespeaker-voxceleb-resnet34-LM", device=torch.device("cuda"), token=get_token()
)


def embed(path):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    e = np.asarray(emb_model(torch.from_numpy(data)[None, None, :])).reshape(-1)
    return e / np.linalg.norm(e)


groups = {}
for g in ["LJ", "1LJ"]:
    files = [f for f in glob.glob("D:/Audion-Data/TTS/iqbal_dataset/train/*.wav")
             if f.replace("\\", "/").rsplit("/", 1)[1].startswith(g)
             and not (g == "LJ" and f.replace("\\", "/").rsplit("/", 1)[1].startswith("1LJ"))]
    groups[g] = [embed(f) for f in random.sample(files, 20)]


def mean_cos(a, b, same):
    sims = [float(x @ y) for i, x in enumerate(a) for j, y in enumerate(b) if not (same and i >= j)]
    return float(np.mean(sims))


result = {
    "within_LJ": round(mean_cos(groups["LJ"], groups["LJ"], True), 3),
    "within_1LJ": round(mean_cos(groups["1LJ"], groups["1LJ"], True), 3),
    "across_LJ_vs_1LJ": round(mean_cos(groups["LJ"], groups["1LJ"], False), 3),
}
json.dump(result, open("D:/Audion-Data/TTS/iqbal_speaker_check.json", "w"), indent=2)
print(result)
