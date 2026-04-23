from __future__ import annotations
from pathlib import Path

import numpy as np
import opensmile
import pandas as pd

_SMILE = opensmile.Smile(
    feature_set=opensmile.FeatureSet.IS09,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

# maps DEAM column names to opensmile Python package IS09 feature names
DEAM_TO_IS09 = {
    "pcm_RMSenergy_sma_amean":         "pcm_RMSenergy_sma",
    "pcm_zcr_sma_amean":               "pcm_zcr_sma",
    "voicingFinalUnclipped_sma_amean":  "voiceProb_sma",
    "F0final_sma_amean":               "F0_sma",
    **{
        f"pcm_fftMag_mfcc_sma[{i}]_amean": f"pcm_fftMag_mfcc_sma[{i}]"
        for i in range(1, 13)
    },
}

COMMON_FEAT_COLS = list(DEAM_TO_IS09.keys())


def extract_is09_features(audio_path: str | Path) -> dict[str, float]:
    audio_path = str(audio_path)
    try:
        frame_df = _SMILE.process_file(audio_path)
    except Exception as exc:
        raise RuntimeError(f"opensmile failed on {audio_path}: {exc}") from exc

    song_means = frame_df.mean()

    return {
        deam_col: float(song_means[is09_col])
        for deam_col, is09_col in DEAM_TO_IS09.items()
    }


def select_common_deam_cols(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COMMON_FEAT_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"DEAM feature DataFrame is missing expected columns: {missing}\n"
            "Ensure you are using the correct DEAM openSMILE feature files."
        )
    return df[COMMON_FEAT_COLS]
