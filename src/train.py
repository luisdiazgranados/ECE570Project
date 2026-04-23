import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from data import load_songlevel_annotations, load_song_features_dir, make_mood_6class, MOOD_LABELS

def main(args):
    ann = load_songlevel_annotations(args.song_level_dir)
    feat = load_song_features_dir(args.features_dir)

    df = feat.merge(ann, on="SongId", how="inner")

    y = make_mood_6class(df["arousal"], df["valence"])
    X = df.drop(columns=["SongId", "valence", "arousal"]).to_numpy(dtype=np.float32)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, class_weight="balanced"))
    ])

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    target_names = [MOOD_LABELS[i] for i in sorted(MOOD_LABELS)]
    print("Confusion matrix:\n", confusion_matrix(yte, pred))
    print("\nClassification report:\n", classification_report(yte, pred, target_names=target_names, digits=3))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", required=True, help="Path to folder with {SongId}.csv")
    p.add_argument("--song_level_dir", required=True, help="Path to annotations averaged per song/song_level/")
    main(p.parse_args())