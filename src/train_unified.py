import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import load_songlevel_annotations, load_song_features_dir, make_mood_6class, MOOD_LABELS
from features_opensmile import select_common_deam_cols, COMMON_FEAT_COLS

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


def main(args: argparse.Namespace) -> None:
    ann  = load_songlevel_annotations(args.song_level_dir)
    feat = load_song_features_dir(args.features_dir)
    df   = feat.merge(ann, on="SongId", how="inner")

    X_df = select_common_deam_cols(df)
    X    = X_df.to_numpy(dtype=np.float32)
    y    = make_mood_6class(df["arousal"], df["valence"])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    target_names = [MOOD_LABELS[i] for i in sorted(MOOD_LABELS)]

    print(f"\n{'Model':<22} {'Accuracy':>9} {'Macro F1':>9}")
    print("-" * 42)

    results = {}
    for name, model in MODELS.items():
        model.fit(Xtr, ytr)
        pred       = model.predict(Xte)
        acc        = float((pred == yte).mean())
        macro_f1   = float(f1_score(yte, pred, average="macro"))
        results[name] = macro_f1
        print(f"{name:<22} {acc:>9.3f} {macro_f1:>9.3f}")

    best_name  = max(results, key=results.__getitem__)
    best_model = MODELS[best_name]

    print(f"\n--- Best model: {best_name} (macro F1 = {results[best_name]:.3f}) ---")
    pred_best = best_model.predict(Xte)
    print(classification_report(yte, pred_best, target_names=target_names, digits=3))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "feature_cols": COMMON_FEAT_COLS}, out_path)
    print(f"Saved {best_name} → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train unified IS09-feature mood classifier on DEAM.")
    p.add_argument("--features_dir",   required=True, help="Path to DEAM per-song openSMILE CSVs")
    p.add_argument("--song_level_dir", required=True, help="Path to DEAM song-level annotation CSVs")
    p.add_argument("--output",         default="../data/model.joblib", help="Where to save the model")
    main(p.parse_args())
