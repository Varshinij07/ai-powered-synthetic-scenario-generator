
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/network_traffic.csv"
MODEL_DIR = "models"
RESULTS_DIR = "data"

FEATURES = [
    "duration",
    "packets",
    "bytes",
    "failed_connections",
    "connection_rate",
    "bytes_per_packet",
    "failed_connection_ratio",
    "traffic_intensity",
]

TARGET = "label"


def evaluate_model(model, X_test, y_test, labels):
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(
            precision_score(
                y_test, predictions, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                y_test, predictions, average="macro", zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(
                y_test, predictions, average="macro", zero_division=0
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_test, predictions, average="weighted", zero_division=0
            )
        ),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        try:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_test,
                    probabilities,
                    labels=labels,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError:
            metrics["roc_auc_ovr_macro"] = None

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    false_positive_rates = []
    for i in range(len(labels)):
        tp = matrix[i, i]
        fn = matrix[i, :].sum() - tp
        fp = matrix[:, i].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        false_positive_rates.append(
            fp / (fp + tn) if (fp + tn) else 0.0
        )

    metrics["false_positive_rate_macro"] = float(
        np.mean(false_positive_rates)
    )

    return metrics, matrix, report


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run python data_generator.py first."
        )

    df = pd.read_csv(DATA_PATH)

    missing_columns = [
        col for col in FEATURES + [TARGET] if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    if df[FEATURES + [TARGET]].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    X = df[FEATURES]
    y = df[TARGET]
    labels = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    results = {}
    best_model = None
    best_name = None
    best_score = -1.0
    best_matrix = None
    best_report = None

    for name, model in models.items():
        print(f"\n{'=' * 55}\n{name}\n{'=' * 55}")

        cv_start = time.perf_counter()
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
        )
        cv_elapsed = time.perf_counter() - cv_start

        print("CV macro F1 scores:", np.round(cv_scores, 4))
        print(f"CV mean macro F1: {cv_scores.mean():.4f}")
        print(f"CV standard deviation: {cv_scores.std():.4f}")

        fit_start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_elapsed = time.perf_counter() - fit_start

        metrics, matrix, report = evaluate_model(
            model, X_test, y_test, labels
        )

        metrics["cv_f1_macro_mean"] = float(cv_scores.mean())
        metrics["cv_f1_macro_std"] = float(cv_scores.std())
        metrics["cv_f1_macro_scores"] = [
            float(value) for value in cv_scores
        ]
        metrics["cv_seconds"] = round(cv_elapsed, 3)
        metrics["training_seconds"] = round(fit_elapsed, 3)

        results[name] = metrics

        print("\nHeld-out test results:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            elif metric != "cv_f1_macro_scores":
                print(f"{metric}: {value}")

        # Select by cross-validation mean, not test-set performance.
        if cv_scores.mean() > best_score:
            best_score = float(cv_scores.mean())
            best_model = model
            best_name = name
            best_matrix = matrix
            best_report = report

    joblib.dump(
        best_model,
        os.path.join(MODEL_DIR, "best_model.joblib"),
    )

    with open(
        os.path.join(RESULTS_DIR, "model_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    pd.DataFrame(
        best_matrix, index=labels, columns=labels
    ).to_csv(os.path.join(RESULTS_DIR, "confusion_matrix.csv"))

    with open(
        os.path.join(RESULTS_DIR, "classification_report.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(best_report, file, indent=4)

    print(f"\nBest model by CV mean macro F1: {best_name}")
    print(f"Mean CV macro F1-score: {best_score:.4f}")
    print("Saved model: models/best_model.joblib")
    print("Saved metrics: data/model_metrics.json")
    print("Saved confusion matrix: data/confusion_matrix.csv")
    print("Saved classification report: data/classification_report.json")


if __name__ == "__main__":
    main()

