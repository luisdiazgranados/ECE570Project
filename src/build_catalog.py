from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from audio_features import extract_librosa_features, compute_arousal_valence
from data import make_mood_6class, MOOD_LABELS

# Hardcoded demo catalog — 48 well-known songs spanning all 6 moods
SONGS = [
    # calm
    ("Miles Davis",         "So What"),
    ("Norah Jones",         "Come Away with Me"),
    ("Nick Drake",          "Pink Moon"),
    ("Brian Eno",           "Music for Airports"),
    ("Mazzy Star",          "Fade Into You"),
    ("Elliott Smith",       "Between the Bars"),
    ("Sufjan Stevens",      "Death With Dignity"),
    ("Bon Iver",            "Skinny Love"),
    # blue
    ("Billie Holiday",      "Strange Fruit"),
    ("Johnny Cash",         "Hurt"),
    ("Amy Winehouse",       "Back to Black"),
    ("The National",        "Sorrow"),
    ("Radiohead",           "How to Disappear Completely"),
    ("Nina Simone",         "Feeling Good"),
    ("Jeff Buckley",        "Hallelujah"),
    ("Lana Del Rey",        "Video Games"),
    # focus
    ("Daft Punk",           "Veridis Quo"),
    ("Explosions in the Sky","Your Hand in Mine"),
    ("Moby",                "Porcelain"),
    ("Air",                 "La Femme d'Argent"),
    ("Boards of Canada",    "Roygbiv"),
    ("Hans Zimmer",         "Time"),
    ("Max Richter",         "On the Nature of Daylight"),
    ("Aphex Twin",          "Ageispolis"),
    # love
    ("Frank Sinatra",       "The Way You Look Tonight"),
    ("Marvin Gaye",         "Let's Get It On"),
    ("Al Green",            "Let's Stay Together"),
    ("John Legend",         "All of Me"),
    ("Stevie Wonder",       "Isn't She Lovely"),
    ("Elvis Presley",       "Can't Help Falling in Love"),
    ("Sam Cooke",           "You Send Me"),
    ("Luther Vandross",     "A House Is Not a Home"),
    # energetic
    ("Nirvana",             "Smells Like Teen Spirit"),
    ("Led Zeppelin",        "Whole Lotta Love"),
    ("AC/DC",               "Thunderstruck"),
    ("Metallica",           "Enter Sandman"),
    ("The Clash",           "London Calling"),
    ("Rage Against the Machine", "Killing in the Name"),
    ("Black Sabbath",       "Paranoid"),
    ("Guns N Roses",        "Welcome to the Jungle"),
    # feel good
    ("Michael Jackson",     "Don't Stop 'Til You Get Enough"),
    ("Daft Punk",           "Get Lucky"),
    ("Bruno Mars",          "Uptown Funk"),
    ("Pharrell Williams",   "Happy"),
    ("Earth Wind and Fire", "September"),
    ("Kool and the Gang",   "Celebration"),
    ("Stevie Wonder",       "Superstition"),
    ("Prince",              "Kiss"),
]

def get_duration(query: str) -> float | None:
    """Return track duration in seconds from the first YouTube result."""
    cmd = ["yt-dlp", "--quiet", "--no-warnings", "--print", "duration", f"ytsearch1:{query}"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        return float(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None

def download_audio(artist: str, title: str, out_dir: Path) -> Path | None:
    query    = f"{artist} {title} official audio"
    out_tmpl = str(out_dir / "%(id)s.%(ext)s")

    duration = get_duration(query)
    start    = int(duration * 0.40) if duration and duration > 60 else 45
    section  = f"*{start}-{start + 30}"

    cmd = [
        "yt-dlp", "--quiet", "--no-warnings",
        "--format", "bestaudio/best",
        "--extract-audio", "--audio-format", "mp3",
        "--audio-quality", "5",
        "--download-sections", section,
        "--output", out_tmpl,
        "--print", "after_move:filepath",
        f"ytsearch1:{query}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    filepath = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    path = Path(filepath) if filepath and Path(filepath).exists() else None
    return path

def build_catalog(output_path: str = "../data/spotify_catalog.csv") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for artist, title in SONGS:
            print(f"  Processing: {artist} — {title}")
            try:
                audio_path = download_audio(artist, title, tmp_dir)
                if audio_path is None:
                    print(f"    [skip] no audio found")
                    continue
                feats            = extract_librosa_features(audio_path.read_bytes())
                arousal, valence = compute_arousal_valence(feats)
                rows.append({"artist": artist, "track_name": title,
                              "arousal": arousal, "valence": valence, **feats})
                print(f"    [ok] arousal={arousal:.3f} valence={valence:.3f}")
            except Exception as e:
                print(f"    [error] {e}")

    df            = pd.DataFrame(rows)
    df["mood_id"] = make_mood_6class(df["arousal"], df["valence"])
    df["mood"]    = df["mood_id"].map(MOOD_LABELS)

    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} tracks to {output_path}")
    print(df["mood"].value_counts().to_string())


if __name__ == "__main__":
    build_catalog()
