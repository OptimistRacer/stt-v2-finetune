from faster_whisper import WhisperModel


def load_model(model_size: str = "base", device: str = "cuda", compute_type: str = "float16"):
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_clip(model, audio_path: str, language: str = "ur") -> str:
    segments, _ = model.transcribe(audio_path, language=language)
    return " ".join(seg.text.strip() for seg in segments)
