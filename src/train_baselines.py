from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from transformer_pm.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN_NAME,
    TEST_RATIO,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)
from transformer_pm.data import load_dga_dataset, split_dataset

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline multiclass models for transformer DGA fault classification."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to source dataset (.xlsx).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics, figures, and saved models.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE oversampling in training pipelines.",
    )
    return parser.parse_args()


def make_pipeline(
    model,
    *,
    use_smote: bool,
    scale_features: bool,
    random_state: int,
) -> Pipeline:
    steps = []
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("model", model))
    return Pipeline(steps)


def build_models(*, use_smote: bool, random_state: int) -> dict[str, Pipeline]:
    return {
        "svm_rbf": make_pipeline(
            SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
            ),
            use_smote=use_smote,
            scale_features=True,
            random_state=random_state,
        ),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
            use_smote=use_smote,
            scale_features=False,
            random_state=random_state,
        ),
        "knn": make_pipeline(
            KNeighborsClassifier(n_neighbors=7, weights="distance"),
            use_smote=use_smote,
            scale_features=True,
            random_state=random_state,
        ),
        "decision_tree": make_pipeline(
            DecisionTreeClassifier(class_weight="balanced", random_state=random_state),
            use_smote=use_smote,
            scale_features=False,
            random_state=random_state,
        ),
        "mlp_ann": make_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=500,
                early_stopping=False,
                random_state=random_state,
            ),
            use_smote=use_smote,
            scale_features=True,
            random_state=random_state,
        ),
    }


def align_probabilities(
    model: Pipeline,
    X: pd.DataFrame,
    labels: list[str],
) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None

    proba = model.predict_proba(X)
    classes = list(model.named_steps["model"].classes_)
    proba_df = pd.DataFrame(proba, columns=classes)
    return proba_df.reindex(columns=labels, fill_value=0.0).to_numpy()


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    labels: list[str],
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    roc_auc = float("nan")
    if y_proba is not None and len(labels) > 2:
        lb = LabelBinarizer()
        lb.fit(labels)
        y_true_bin = lb.transform(y_true)
        if y_true_bin.ndim == 1:
            y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])
        try:
            roc_auc = float(
                roc_auc_score(
                    y_true_bin,
                    y_proba,
                    average="macro",
                    multi_class="ovr",
                )
            )
        except ValueError:
            roc_auc = float("nan")

    metrics["roc_auc_ovr_macro"] = roc_auc
    return metrics


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()

    np.random.seed(args.random_state)

    output_dir = args.output_dir
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"

    for directory in (metrics_dir, figures_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    df = load_dga_dataset(args.data_path)
    splits = split_dataset(
        df,
        train_ratio=TRAIN_RATIO,
        val_ratio=VALIDATION_RATIO,
        test_ratio=TEST_RATIO,
        random_state=args.random_state,
    )

    labels = sorted(df[TARGET_COLUMN_NAME].unique().tolist())
    use_smote = not args.no_smote
    models = build_models(use_smote=use_smote, random_state=args.random_state)

    summary_rows: list[dict[str, float | str]] = []
    trained_models: dict[str, Pipeline] = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        try:
            start = time.perf_counter()
            model.fit(splits.X_train, splits.y_train)
            train_seconds = time.perf_counter() - start

            trained_models[model_name] = model

            val_pred = model.predict(splits.X_val)
            val_proba = align_probabilities(model, splits.X_val, labels)
            val_metrics = compute_metrics(splits.y_val, val_pred, val_proba, labels)

            test_pred = model.predict(splits.X_test)
            test_proba = align_probabilities(model, splits.X_test, labels)
            test_metrics = compute_metrics(splits.y_test, test_pred, test_proba, labels)

            val_report = classification_report(
                splits.y_val,
                val_pred,
                labels=labels,
                zero_division=0,
            )
            test_report = classification_report(
                splits.y_test,
                test_pred,
                labels=labels,
                zero_division=0,
            )

            save_text(
                metrics_dir / f"classification_report_val_{model_name}.txt",
                val_report,
            )
            save_text(
                metrics_dir / f"classification_report_test_{model_name}.txt",
                test_report,
            )

            save_confusion_matrix(
                splits.y_val,
                val_pred,
                labels,
                figures_dir / f"confusion_matrix_val_{model_name}.png",
                f"Validation Confusion Matrix - {model_name}",
            )
            save_confusion_matrix(
                splits.y_test,
                test_pred,
                labels,
                figures_dir / f"confusion_matrix_test_{model_name}.png",
                f"Test Confusion Matrix - {model_name}",
            )

            joblib.dump(model, models_dir / f"{model_name}.joblib")

            row: dict[str, float | str] = {
                "model": model_name,
                "train_seconds": round(train_seconds, 4),
            }
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            row.update({f"test_{k}": v for k, v in test_metrics.items()})
            summary_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            err = (
                f"Model {model_name} failed with {type(exc).__name__}: {exc}\n\n"
                + traceback.format_exc()
            )
            print(err)
            save_text(metrics_dir / f"error_{model_name}.txt", err)

    if not summary_rows:
        raise RuntimeError("All models failed. Check outputs/metrics/error_*.txt")

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by="val_f1_macro", ascending=False)
    summary_path = metrics_dir / "model_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    best_model_name = str(summary_df.iloc[0]["model"])
    best_model = trained_models[best_model_name]

    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN_NAME,
            "labels": labels,
            "use_smote": use_smote,
        },
        models_dir / "best_model_bundle.joblib",
    )

    split_summary = {
        "dataset_path": str(args.data_path),
        "samples_total": int(len(df)),
        "features": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN_NAME,
        "classes": labels,
        "class_distribution": df[TARGET_COLUMN_NAME].value_counts().to_dict(),
        "split_sizes": {
            "train": int(len(splits.X_train)),
            "validation": int(len(splits.X_val)),
            "test": int(len(splits.X_test)),
        },
        "random_state": args.random_state,
        "use_smote": use_smote,
        "primary_metric": "val_f1_macro",
        "best_model": best_model_name,
    }

    (metrics_dir / "experiment_config.json").write_text(
        json.dumps(split_summary, indent=2),
        encoding="utf-8",
    )

    print("\nTraining complete.")
    print(f"Best model by validation macro F1: {best_model_name}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
