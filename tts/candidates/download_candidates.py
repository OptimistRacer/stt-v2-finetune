"""Download the candidate voice models into the normal Hugging Face cache, so each
model's own loader finds them later without re-downloading. Network/disk only, no GPU."""
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError

MODELS = [
    ("ai4bharat/indic-parler-tts", None),        # Urdu, Apache-2.0
    ("k2-fsa/OmniVoice", None),                  # Urdu + Arabic, CC-BY-NC
    ("openbmb/VoxCPM2", None),                   # Arabic, Apache-2.0
    # Chatterbox: only the multilingual (23-language) v3 weights, not the English-only
    # and duplicate .pt copies that make the full repo 13.9 GB.
    ("ResembleAI/chatterbox", ["t3_mtl23ls_v3.safetensors", "s3gen_v3.safetensors",
                               "s3gen.safetensors", "ve.safetensors", "conds.pt",
                               "*.json", "README.md"]),
]

for repo, patterns in MODELS:
    print(f"[dl] {repo} ...", flush=True)
    try:
        path = snapshot_download(repo, allow_patterns=patterns)
    except GatedRepoError:
        # Gated repos need their terms accepted on huggingface.co by the account owner.
        print(f"[dl] {repo} GATED -- accept its terms on huggingface.co, then rerun", flush=True)
        continue
    print(f"[dl] {repo} -> {path}", flush=True)
print("[dl] all done", flush=True)
