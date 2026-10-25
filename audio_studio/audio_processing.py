"""Mix a converted vocal and an optional accompaniment without changing pitch."""

from pathlib import Path


def mix_accompaniment(vocal_path: Path, accompaniment_path: Path, output_path: Path,
                      vocal_gain: float = 1, accompaniment_gain: float = .7) -> dict:
    import librosa
    import numpy as np
    import soundfile as sf
    vocal, sample_rate = sf.read(vocal_path, dtype="float32", always_2d=True)
    accompaniment, other_rate = sf.read(accompaniment_path, dtype="float32", always_2d=True)
    if not np.isfinite(vocal).all() or not np.isfinite(accompaniment).all():
        raise ValueError("音频含有无效数值，无法混音")
    if other_rate != sample_rate:
        accompaniment = librosa.resample(accompaniment.T, orig_sr=other_rate, target_sr=sample_rate).T
    channels = max(vocal.shape[1], accompaniment.shape[1])
    if vocal.shape[1] != channels:
        vocal = np.repeat(vocal, channels, axis=1)
    if accompaniment.shape[1] != channels:
        accompaniment = np.repeat(accompaniment, channels, axis=1)
    length = max(len(vocal), len(accompaniment))
    mixed = np.zeros((length, channels), dtype="float32")
    mixed[:len(vocal)] += vocal * vocal_gain
    mixed[:len(accompaniment)] += accompaniment * accompaniment_gain
    peak = float(np.max(np.abs(mixed), initial=0))
    if peak > .98:
        mixed *= .98 / peak
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, mixed, sample_rate, subtype="PCM_16")
    return {"path": str(output_path), "duration_seconds": length / sample_rate, "sample_rate": sample_rate,
            "duration": length / sample_rate, "peak_normalized": peak > .98}
