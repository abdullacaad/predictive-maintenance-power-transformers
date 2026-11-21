from __future__ import annotations

from pathlib import Path

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

matplotlib.use("Agg")


def make_pipeline(
    model: object,
    *,
    use_smote: bool,
    scale_features: bool,
    random_state: int,
) -> Pipeline:
    steps: list[tuple[str, object]] = []
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("model", model))
    return Pipeline(steps)


def build_baseline_models(*, use_smote: bool, random_state: int) -> dict[str, Pipeline]:
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


def build_tuning_search_space(*, use_smote: bool, random_state: int) -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    return {
        "svm_rbf": (
            make_pipeline(
                SVC(kernel="rbf", probability=False, class_weight="balanced"),
                use_smote=use_smote,
                scale_features=True,
                random_state=random_state,
            ),
            {
                "model__C": [1.0, 3.0, 10.0, 30.0],
                "model__gamma": ["scale", 0.01, 0.1, 1.0],
            },
        ),
        "random_forest": (
            make_pipeline(
                RandomForestClassifier(random_state=random_state, n_jobs=-1),
                use_smote=use_smote,
                scale_features=False,
                random_state=random_state,
            ),
            {
                "model__n_estimators": [150, 250, 350],
                "model__max_depth": [None, 12, 20],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt"],
                "model__class_weight": ["balanced_subsample"],
            },
        ),
        "knn": (
            make_pipeline(
                KNeighborsClassifier(),
                use_smote=use_smote,
                scale_features=True,
                random_state=random_state,
            ),
            {
                "model__n_neighbors": [3, 5, 7, 9, 11, 15],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
        "decision_tree": (
            make_pipeline(
                DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
                use_smote=use_smote,
                scale_features=False,
                random_state=random_state,
            ),
            {
                "model__criterion": ["gini", "entropy", "log_loss"],
                "model__max_depth": [None, 5, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        "mlp_ann": (
            make_pipeline(
                MLPClassifier(
                    max_iter=100,
                    early_stopping=False,
                    random_state=random_state,
                ),
                use_smote=use_smote,
                scale_features=True,
                random_state=random_state,
            ),
            {
                "model__hidden_layer_sizes": [(64,), (128,), (128, 64)],
                "model__alpha": [1e-5, 1e-4, 1e-3],
                "model__learning_rate_init": [0.001, 0.005],
            },
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
