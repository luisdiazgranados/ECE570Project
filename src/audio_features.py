import io
import numpy as np
import requests
import librosa

def download_preview(url: str) -> bytes:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.content

def extract_librosa_features(audio_bytes: bytes, sr_target: int = 22050) -> dict:
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr_target, mono=True)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms       = librosa.feature.rms(y=y)[0]
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr       = librosa.feature.zero_crossing_rate(y)[0]
    contrast  = librosa.feature.spectral_contrast(y=y, sr=sr)
    mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma    = librosa.feature.chroma_stft(y=y, sr=sr)

    features = {
        "tempo":          float(np.squeeze(tempo)),
        "rms_mean":       float(rms.mean()),
        "rms_std":        float(rms.std()),
        "centroid_mean":  float(centroid.mean() / sr),
        "rolloff_mean":   float(rolloff.mean() / sr),
        "zcr_mean":       float(zcr.mean()),
        **{f"mfcc_{i}_mean": float(mfccs[i].mean()) for i in range(13)},
        **{f"mfcc_{i}_std":  float(mfccs[i].std())  for i in range(13)},
        **{f"chroma_{i}":    float(chroma[i].mean()) for i in range(12)},
        **{f"contrast_{i}":  float(contrast[i].mean()) for i in range(7)},
    }
    return features

# Krumhansl-Schmuckler key profiles (major and minor)
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                            2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                            2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def _major_score(chroma: np.ndarray) -> float:
    """Returns 1.0 if best key match is major, 0.0 if minor."""
    chroma = chroma / (chroma.sum() + 1e-8)
    maj = _MAJOR_PROFILE / _MAJOR_PROFILE.sum()
    minn = _MINOR_PROFILE / _MINOR_PROFILE.sum()
    major_corrs = [np.dot(chroma, np.roll(maj,  i)) for i in range(12)]
    minor_corrs = [np.dot(chroma, np.roll(minn, i)) for i in range(12)]
    best_major, best_minor = max(major_corrs), max(minor_corrs)
    return float(best_major / (best_major + best_minor + 1e-8))

def compute_arousal_valence(features: dict) -> tuple[float, float]:
    """Compute arousal and valence proxies from librosa features."""
    # Arousal: weighted energy, tempo, zero-crossing rate, brightness
    tempo_norm    = min(features["tempo"] / 200.0, 1.0)
    rms_norm      = min(features["rms_mean"] / 0.15, 1.0)
    centroid_norm = min(features["centroid_mean"] / 0.5, 1.0)
    zcr_norm      = min(features["zcr_mean"] / 0.15, 1.0)
    arousal = 0.35 * rms_norm + 0.30 * tempo_norm + 0.20 * zcr_norm + 0.15 * centroid_norm

    # Valence: major/minor key (Krumhansl-Schmuckler) + low-band spectral contrast
    chroma        = np.array([features[f"chroma_{i}"] for i in range(12)])
    mode_score    = _major_score(chroma)
    contrast_norm = np.clip((features["contrast_0"] + 30) / 60.0, 0.0, 1.0)
    valence = 0.65 * mode_score + 0.35 * float(contrast_norm)

    return float(arousal), float(valence)
