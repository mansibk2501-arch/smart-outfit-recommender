import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from color_utils import COLOR_PALETTE, hex_to_rgb, color_features, harmony_rule_score

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "outfit_pairs_dataset.csv"
MODEL_PATH = ROOT / "outfit_color_model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"

def make_dataset():
    # Generate reproducible color pairs. Labels are derived from the documented
    # color-harmony heuristic; they are not human-rated fashion judgments.
    rng = np.random.default_rng(2026)
    palette = list(COLOR_PALETTE.items())
    rows = []
    for _ in range(6000):
        name1, hex1 = palette[int(rng.integers(0, len(palette)))]
        name2, hex2 = palette[int(rng.integers(0, len(palette)))]
        rgb1, rgb2 = hex_to_rgb(hex1), hex_to_rgb(hex2)
        features = color_features(rgb1, rgb2)
        score = harmony_rule_score(rgb1, rgb2)
        # A transparent threshold turns the rule score into a binary training label.
        label = int(score >= 0.67)
        rows.append({
            "base_color": name1,
            "candidate_color": name2,
            "hue_distance": features[0],
            "saturation_difference": features[1],
            "value_difference": features[2],
            "brightness_contrast": features[3],
            "neutral_pair": features[4],
            "minimum_saturation": features[5],
            "minimum_value": features[6],
            "rule_score": score,
            "compatible_label": label,
        })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_PATH, index=False)
    return df

def main():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = make_dataset()

    feature_cols = [
        "hue_distance", "saturation_difference", "value_difference",
        "brightness_contrast", "neutral_pair", "minimum_saturation", "minimum_value"
    ]
    X = df[feature_cols].astype(float)
    y = df["compatible_label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=180, max_depth=10, min_samples_leaf=2,
        class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=["not-match", "match"], output_dict=True, zero_division=0
        ),
        "note": "These metrics measure agreement with synthetic labels created by a color-harmony rule. They do not establish human fashion preference or real-world accuracy."
    }
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Model saved to:", MODEL_PATH.name)
    print("Dataset saved to:", DATA_PATH.name)
    print("Metrics saved to:", METRICS_PATH.name)
    print(f"Test accuracy: {metrics['accuracy']:.3f}")
    print(f"Test precision: {metrics['precision']:.3f}")
    print(f"Test recall: {metrics['recall']:.3f}")
    print("\nImportant: scores are against synthetic rule-generated labels, not human fashion ratings.")

if __name__ == "__main__":
    main()
