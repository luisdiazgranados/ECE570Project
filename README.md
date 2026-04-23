# Mood-Based Music Recommendation Engine

A music recommendation prototype that classifies songs by mood using the
valence × arousal model of emotion, trained on the DEAM dataset.

---

## Project Structure

```
music-mood-engine/
├── data/
│   ├── annotations/          # DEAM song-level annotation CSVs (see Dataset section)
│   ├── features/             # DEAM per-song openSMILE feature CSVs (see Dataset section)
│   ├── spotify_catalog.csv   # Pre-built 48-song catalog with mood labels
│   └── model.joblib          # Trained SVM classifier (produced by train_unified.py)
├── src/
│   ├── data.py               # DEAM data loading and mood label generation
│   ├── audio_features.py     # librosa feature extraction and arousal/valence proxy
│   ├── features_opensmile.py # IS09 LLD feature extraction via opensmile package
│   ├── train.py              # Logistic Regression baseline (CP1 reference)
│   ├── compare_models.py     # 4-model comparison on full DEAM feature set
│   ├── train_unified.py      # Trains on IS09-compatible DEAM features, saves model
│   ├── build_catalog.py      # Downloads audio, extracts features, builds catalog CSV
│   ├── recommend.py          # CLI mood-based song retrieval
│   ├── clap_classifier.py    # Experimental: CLAP zero-shot classifier (non-functional on Python 3.13)
│   └── spotify_client.py     # Spotify API helper (optional metadata enrichment)
└── requirements.txt
```

---

## Dependencies

Python 3.9+ is required. Install all dependencies with:

```bash
pip install -r requirements.txt
```

`yt-dlp` must also be available on your PATH (used by `build_catalog.py` to
download audio clips). Install it separately if needed:

```bash
pip install yt-dlp
```

---

## Dataset

This project uses the **DEAM dataset** (Database for Emotional Analysis in
Music). It is not bundled here due to size and licensing constraints.

**To obtain DEAM:**
1. Register and download from the official page:
   http://cvml.unige.ch/databases/DEAM/
2. Download the following archives:
   - `DEAM_audio.zip` — 1,802 audio clips (45 s each)
   - `DEAM_Annotations.zip` — per-song valence/arousal ratings
   - `DEAM_features.zip` — pre-extracted openSMILE features
3. Extract so the directory layout matches:
   ```
   data/
   ├── annotations/annotations averaged per song/song_level/
   │   ├── static_annotations_averaged_songs_1_2000.csv
   │   └── static_annotations_averaged_songs_2000_2058.csv
   └── features/
       ├── 10.csv
       ├── 11.csv
       └── ...
   ```

The pre-built `spotify_catalog.csv` and `model.joblib` are included so the
recommender can be run without downloading DEAM.

---

## Running the Project

All commands are run from the `src/` directory.

### 1. Train the mood classifier (requires DEAM)

Compare all four models on the full openSMILE feature set:

```bash
python compare_models.py \
    --features_dir   ../data/features \
    --song_level_dir "../data/annotations/annotations averaged per song/song_level"
```

Train on the IS09-compatible feature subset and save the best model:

```bash
python train_unified.py \
    --features_dir   ../data/features \
    --song_level_dir "../data/annotations/annotations averaged per song/song_level" \
    --output         ../data/model.joblib
```

### 2. Build the song catalog (requires internet + yt-dlp)

Downloads 30-second audio clips from YouTube for 48 curated songs, extracts
librosa features, computes arousal/valence proxies, and assigns mood labels.

```bash
python build_catalog.py
```

This overwrites `../data/spotify_catalog.csv`. Runtime is approximately
10–20 minutes depending on network speed.

### 3. Get song recommendations

```bash
python recommend.py --mood calm --k 5
python recommend.py --mood energetic --k 10
python recommend.py --mood blue --k 5
```

Available moods: `blue`, `calm`, `focus`, `love`, `energetic`, `feel good`

---

## Mood Taxonomy

Songs are placed in a 6-class mood space derived from Russell's (1980)
circumplex model of affect, using two axes:

| | Low valence | High valence |
|---|---|---|
| **Low arousal** | blue | calm |
| **Mid arousal** | focus | love |
| **High arousal** | energetic | feel good |

Class boundaries are set at the 33rd and 66th percentiles of the DEAM
arousal and valence distributions.

---

## Code Authorship

All code in `src/` was written from scratch for this project. The following
notes clarify the origin of specific design choices:

| File | Lines | Notes |
|---|---|---|
| `data.py` | 1–96 | Entirely original. Mood label functions use Russell (1980) taxonomy. |
| `audio_features.py` | 39–53 | Krumhansl-Schmuckler key profiles are standard music theory constants, not derived from a codebase. |
| `features_opensmile.py` | 1–75 | Entirely original. Uses the `opensmile` Python package API. |
| `train.py` | 1–40 | Entirely original. Uses scikit-learn Pipeline API per library docs. |
| `compare_models.py` | 1–68 | Entirely original. |
| `train_unified.py` | 1–80 | Entirely original. |
| `build_catalog.py` | 1–end | Entirely original. yt-dlp subprocess calls follow yt-dlp CLI documentation. |
| `recommend.py` | 1–56 | Entirely original. |
| `clap_classifier.py` | 1–123 | Entirely original. Written as an experimental zero-shot classifier using the CLAP model; non-functional due to Python 3.13 / transformers v5 incompatibilities. Retained for documentation of attempted approaches. |
| `spotify_client.py` | 1–41 | Entirely original. Uses spotipy library API per library docs. |

No code was copied from external repositories.

---

## Notes on Reproducibility

- **Catalog mood labels** are derived from 30-second YouTube audio clips and
  depend on which video yt-dlp selects. Minor variation across runs is
  expected due to YouTube search results changing over time.
- **Model training** is deterministic given `random_state=0` in all
  stochastic estimators.
- **MFCC and librosa features** depend on `librosa` version; results were
  produced with `librosa==0.11.0`.
