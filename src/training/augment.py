"""Audio augmentation for robustness to noisy/degraded recordings.

Why this exists: both training corpora are clean. FLEURS is crowd-sourced read speech
(median ~28 dB SNR, worst ~14 dB) and the podcasts are studio-grade (~46-50 dB raw).
Neither contains traffic, cafe noise, phone-line compression, or room reverb, so a model
fine-tuned on them has never seen degraded audio -- and a demo run on a laptop mic or
over a phone would land well outside anything it was trained on.

Deliberately no external noise corpus (MUSAN and friends are multi-GB downloads). The
degradations here are synthesized or drawn from the training audio itself:
- colored noise (white/pink/brown) covers broadband hiss, fans, AC
- babble is mixed from other clips in this very corpus, which is a better stand-in for
  background conversation than any synthetic signal
- reverb via an exponentially-decaying-noise impulse response, a cheap but serviceable
  approximation of a small room
- telephone band-limiting plus mu-law quantization for phone-call audio

Applied ONLY to the training split. Augmenting eval would make the numbers unreadable --
you could not tell a real improvement from an easier test set.
"""
import numpy as np
from scipy.signal import butter, fftconvolve, lfilter


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def _mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale `noise` so the mixture sits at `snr_db`, then add it."""
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    noise = noise[: len(speech)]
    scale = _rms(speech) / (_rms(noise) * (10 ** (snr_db / 20)))
    return speech + noise * scale


def colored_noise(n: int, rng: np.random.Generator, kind: str = "white") -> np.ndarray:
    white = rng.standard_normal(n).astype(np.float32)
    if kind == "white":
        return white
    # 1/f (pink) and 1/f^2 (brown) via spectral shaping
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    exponent = 0.5 if kind == "pink" else 1.0
    shaped = spec / (freqs**exponent)
    out = np.fft.irfft(shaped, n).astype(np.float32)
    return out / (np.abs(out).max() + 1e-9)


def synth_rir(sr: int, rng: np.random.Generator, rt60: float) -> np.ndarray:
    """Exponentially decaying noise burst as a crude room impulse response."""
    n = int(sr * rt60)
    if n < 2:
        return np.array([1.0], dtype=np.float32)
    t = np.arange(n) / sr
    rir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0  # keep the direct path dominant
    return rir / (np.abs(rir).max() + 1e-9)


def telephone(x: np.ndarray, sr: int) -> np.ndarray:
    """Band-limit to roughly the telephone band and mu-law quantize."""
    low = max(300 / (sr / 2), 1e-4)
    high = min(3400 / (sr / 2), 0.99)
    b, a = butter(4, [low, high], btype="band")
    y = lfilter(b, a, x).astype(np.float32)
    mu = 255.0
    peak = np.abs(y).max() + 1e-9
    companded = np.sign(y) * np.log1p(mu * np.abs(y) / peak) / np.log1p(mu)
    quantized = np.round(companded * 128) / 128
    expanded = np.sign(quantized) * ((1 + mu) ** np.abs(quantized) - 1) / mu
    return (expanded * peak).astype(np.float32)


class Augmenter:
    """Randomly degrades a waveform. `p_clean` leaves audio untouched so the model keeps
    seeing pristine input too -- training only on degraded audio would trade one blind
    spot for another."""

    def __init__(
        self,
        sample_rate: int = 16000,
        p_clean: float = 0.35,
        p_noise: float = 0.6,
        p_babble: float = 0.3,
        p_reverb: float = 0.3,
        p_telephone: float = 0.15,
        p_gain: float = 0.4,
        snr_range: tuple[float, float] = (5.0, 25.0),
        seed: int | None = None,
    ):
        self.sr = sample_rate
        self.p_clean = p_clean
        self.p_noise = p_noise
        self.p_babble = p_babble
        self.p_reverb = p_reverb
        self.p_telephone = p_telephone
        self.p_gain = p_gain
        self.snr_range = snr_range
        self.rng = np.random.default_rng(seed)
        self._babble_pool: list[np.ndarray] = []

    def set_babble_pool(self, clips: list[np.ndarray]) -> None:
        """Short waveforms used to synthesize background conversation."""
        self._babble_pool = clips

    def _babble(self, n: int) -> np.ndarray | None:
        if len(self._babble_pool) < 2:
            return None
        k = int(self.rng.integers(2, min(5, len(self._babble_pool) + 1)))
        idx = self.rng.choice(len(self._babble_pool), size=k, replace=False)
        mix = np.zeros(n, dtype=np.float32)
        for i in idx:
            c = self._babble_pool[i]
            if len(c) < n:
                c = np.tile(c, int(np.ceil(n / len(c))))
            off = int(self.rng.integers(0, max(1, len(c) - n + 1)))
            mix += c[off : off + n]
        return mix / (np.abs(mix).max() + 1e-9)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.p_clean:
            return x
        y = x.astype(np.float32).copy()

        if self.rng.random() < self.p_reverb:
            rir = synth_rir(self.sr, self.rng, rt60=float(self.rng.uniform(0.15, 0.6)))
            y = fftconvolve(y, rir)[: len(x)].astype(np.float32)

        if self.rng.random() < self.p_babble:
            b = self._babble(len(y))
            if b is not None:
                y = _mix_at_snr(y, b, float(self.rng.uniform(10.0, 25.0)))

        if self.rng.random() < self.p_noise:
            kind = str(self.rng.choice(["white", "pink", "brown"]))
            y = _mix_at_snr(y, colored_noise(len(y), self.rng, kind), float(self.rng.uniform(*self.snr_range)))

        if self.rng.random() < self.p_telephone:
            y = telephone(y, self.sr)

        if self.rng.random() < self.p_gain:
            y = y * float(self.rng.uniform(0.3, 1.4))

        peak = np.abs(y).max()
        if peak > 1.0:  # keep it in range; clipping here would be an unintended extra degradation
            y = y / peak
        return y.astype(np.float32)
