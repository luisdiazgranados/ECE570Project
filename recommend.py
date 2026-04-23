import argparse
import pandas as pd
from data import MOOD_LABELS

MOOD_NAME_TO_ID = {v: k for k, v in MOOD_LABELS.items()}


def recommend(catalog: pd.DataFrame, mood: str, k: int = 10) -> pd.DataFrame:
    mood = mood.lower().strip()
    if mood not in MOOD_NAME_TO_ID:
        valid = ", ".join(MOOD_LABELS.values())
        raise ValueError(f"Unknown mood '{mood}'. Choose from: {valid}")

    matches = catalog[catalog["mood"] == mood].copy()

    if "confidence" in matches.columns:
        matches = matches.sort_values("confidence", ascending=False)
    elif "rms_mean" in matches.columns:
        matches = matches.sort_values("rms_mean", ascending=False)

    keep = [c for c in ["track_name", "artist", "mood", "confidence"] if c in matches.columns]
    return matches[keep].head(k)


def main(args: argparse.Namespace) -> None:
    catalog = pd.read_csv(args.catalog)
    results = recommend(catalog, args.mood, args.k)

    if results.empty:
        print(f"No songs found for mood: {args.mood}")
        return

    print(f"\nTop {len(results)} '{args.mood}' songs:\n")
    for _, row in results.iterrows():
        conf = f"  [confidence={row['confidence']:.2f}]" if "confidence" in row else ""
        print(f"  {row['track_name']} — {row['artist']}{conf}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Recommend songs by mood from the pre-built catalog.")
    p.add_argument("--catalog", default="../data/spotify_catalog.csv")
    p.add_argument("--mood",    required=True, choices=list(MOOD_LABELS.values()))
    p.add_argument("--k",       type=int, default=10)
    main(p.parse_args())
