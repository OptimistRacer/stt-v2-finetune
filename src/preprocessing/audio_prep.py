import subprocess
from pathlib import Path


def resample_and_normalize(input_path: Path, output_path: Path, sample_rate: int = 16000) -> None:
    """Denoise (FFT noise gate), loudness-normalize (EBU R128), and convert to mono WAV at sample_rate."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", str(sample_rate), "-ac", "1",
        "-af", "afftdn=nf=-25,loudnorm=I=-23:TP=-2:LRA=7",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
