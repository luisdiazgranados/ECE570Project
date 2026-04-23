from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

def load_songlevel_annotations(song_level_dir: str | Path) -> pd.DataFrame:
    song_level_dir = Path(song_level_dir)
    files = sorted(song_level_dir.glob("static_annotations*.csv"))
    
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        df.columns = df.columns.str.strip()
        dfs.append(df)

    ann = pd.concat(dfs, ignore_index=True)

    song_col = [c for c in ann.columns if "song" in c.lower()][0]
    val_col  = [c for c in ann.columns if "valence" in c.lower()][0]
    aro_col  = [c for c in ann.columns if "arousal" in c.lower()][0]

    out = ann[[song_col, val_col, aro_col]].rename(
        columns={song_col: "SongId", val_col: "valence", aro_col: "arousal"}
    )

    out["SongId"] = pd.to_numeric(out["SongId"], errors="coerce").astype("Int64")
    out["valence"] = pd.to_numeric(out["valence"], errors="coerce")
    out["arousal"] = pd.to_numeric(out["arousal"], errors="coerce")
    out = out.dropna(subset=["SongId", "valence", "arousal"]).astype({"SongId": int})

    return out

def make_arousal_3class(arousal: pd.Series) -> np.ndarray:
    # 0=calm, 1=medium, 2=energetic based on 33rd/66th percentile thresholds
    low_thr = arousal.quantile(0.33)
    high_thr = arousal.quantile(0.66)

    labels = []
    for val in arousal:
        if val <= low_thr:
            labels.append(0) #calm
        elif val >= high_thr:
            labels.append(2) #energetic
        else:
            labels.append(1) #medium

    return np.array(labels)

MOOD_LABELS = {0: "blue", 1: "calm", 2: "focus", 3: "love", 4: "energetic", 5: "feel good"}

def make_mood_6class(arousal: pd.Series, valence: pd.Series) -> np.ndarray:
    # splits valence x arousal into 6 regions using 33rd/66th percentile cutoffs
    a_low  = arousal.quantile(0.33)
    a_high = arousal.quantile(0.66)
    v_low  = valence.quantile(0.33)
    v_high = valence.quantile(0.66)

    a, v = arousal.values, valence.values
    conditions = [
        (a <= a_low)  & (v <= v_low),                   # 0: blue
        (a <= a_low)  & (v >  v_low),                   # 1: calm
        (a >  a_low)  & (a < a_high) & (v <  v_high),  # 2: focus
        (a >  a_low)  & (a < a_high) & (v >= v_high),  # 3: love
        (a >= a_high) & (v <  v_high),                  # 4: energetic
        (a >= a_high) & (v >= v_high),                  # 5: feel good
    ]
    return np.select(conditions, [0, 1, 2, 3, 4, 5]).astype(int)

def load_song_features_dir(features_dir: str | Path) -> pd.DataFrame:
    # each file is <SongId>.csv with per-frame openSMILE features; we average across frames
    features_dir = Path(features_dir)
    files = sorted(features_dir.glob("*.csv"))

    all_rows = []

    for file in files:
        song_id = int(file.stem)

        df = pd.read_csv(file, sep=";")
        df.columns = df.columns.str.strip()

        if "frameTime" in df.columns:
            df = df.drop(columns=["frameTime"])

        mean_features = df.mean()

        row = mean_features.to_dict()
        row["SongId"] = song_id
        all_rows.append(row)

    return pd.DataFrame(all_rows)