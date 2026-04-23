import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from data import load_songlevel_annotations, load_song_features_dir, make_mood_6class, MOOD_LABELS

MODELS = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, class_weight="balanced")),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0)),
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=0)),
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=300, random_state=0)),
    ]),
}

def main(args):
    ann  = load_songlevel_annotations(args.song_level_dir)
    feat = load_song_features_dir(args.features_dir)
    df   = feat.merge(ann, on="SongId", how="inner")

    y = make_mood_6class(df["arousal"], df["valence"])
    X = df.drop(columns=["SongId", "valence", "arousal"]).to_numpy(dtype=np.float32)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    target_names = [MOOD_LABELS[i] for i in sorted(MOOD_LABELS)]

    print(f"{'Model':<22} {'Accuracy':>9} {'Macro F1':>9}")
    print("-" * 42)
    for name, model in MODELS.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc  = (pred == yte).mean()
        f1   = f1_score(yte, pred, average="macro")
        print(f"{name:<22} {acc:>9.3f} {f1:>9.3f}")

    print("\n--- Full report for best model ---")
    best_name = max(
        MODELS,
        key=lambda n: f1_score(yte, MODELS[n].predict(Xte), average="macro")
    )
    print(f"Model: {best_name}")
    print(classification_report(yte, MODELS[best_name].predict(Xte),
                                target_names=target_names, digits=3))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir",   required=True)
    p.add_argument("--song_level_dir", required=True)
    main(p.parse_args())
