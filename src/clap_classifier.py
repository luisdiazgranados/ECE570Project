from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import librosa
from transformers import ClapModel, ClapProcessor

from data import MOOD_LABELS

MOOD_PROMPTS: dict[str, list[str]] = {
    "blue": [
        "sad melancholic music",
        "dark sorrowful song",
        "slow depressing music with minor key",
    ],
    "calm": [
        "calm peaceful relaxing music",
        "quiet ambient music",
        "gentle soothing instrumental",
    ],
    "focus": [
        "focused instrumental music for concentration",
        "atmospheric electronic music",
        "steady rhythmic background music",
    ],
    "love": [
        "romantic love song",
        "warm tender ballad",
        "sweet emotional music",
    ],
    "energetic": [
        "high energy aggressive rock music",
        "fast loud intense music",
        "powerful driving heavy music",
    ],
    "feel good": [
        "happy upbeat feel good music",
        "joyful danceable pop song",
        "fun cheerful uplifting music",
    ],
}

_MOOD_NAMES  = list(MOOD_PROMPTS.keys())
_ALL_PROMPTS = [p for m in _MOOD_NAMES for p in MOOD_PROMPTS[m]]
_PROMPTS_PER_MOOD = [len(MOOD_PROMPTS[m]) for m in _MOOD_NAMES]


class ClapMoodClassifier:
    """Wraps the CLAP model for zero-shot mood classification."""

    def __init__(self, model_id: str = "laion/larger_clap_music"):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model     = ClapModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(model_id)

        # Pre-encode all text prompts (done once)
        text_inputs = self.processor(
            text=_ALL_PROMPTS, return_tensors="pt", padding=True
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_embeds = self.model.get_text_features(**text_inputs)
        self._text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    def classify(self, audio_path: str | Path, sr: int = 48000) -> tuple[str, float, dict]:
        """Returns (mood_name, confidence, scores_per_mood)."""
        y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
        audio_inputs = self.processor(
            audios=y, sampling_rate=sr, return_tensors="pt"
        )
        audio_inputs = {k: v.to(self.device) for k, v in audio_inputs.items()}

        with torch.no_grad():
            audio_embed = self.model.get_audio_features(**audio_inputs)  # (1, D)
        audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)

        sims = (audio_embed @ self._text_embeds.T).squeeze(0).cpu().numpy()

        mood_scores: dict[str, float] = {}
        offset = 0
        for mood, n_prompts in zip(_MOOD_NAMES, _PROMPTS_PER_MOOD):
            mood_scores[mood] = float(sims[offset:offset + n_prompts].mean())
            offset += n_prompts

        raw = np.array([mood_scores[m] for m in _MOOD_NAMES])
        exp  = np.exp(raw - raw.max())
        probs = exp / exp.sum()
        prob_dict = {m: float(p) for m, p in zip(_MOOD_NAMES, probs)}

        best_mood   = _MOOD_NAMES[int(np.argmax(probs))]
        confidence  = float(probs.max())
        return best_mood, confidence, prob_dict
