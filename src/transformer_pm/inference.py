from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import (
    FEATURE_COLUMNS,
    LABEL_TRANSLATION_MAP,
    SOURCE_LABEL_COLUMN_CANDIDATES,
    TARGET_COLUMN_NAME,
)
from .data import prepare_input_frame


@dataclass
class LoadedModelBundle:
    model: object
    model_name: str
    feature_columns: list[str]
    target_column: str
    labels: list[str]
    use_smote: bool


def probability_column_name(label: str) -> str:
    normalized = label.lower().replace("-", " ").replace("/", " ")
    normalized = "_".join(normalized.split())
    return f"prob_{normalized}"


def load_model_bundle(bundle_path: str | Path) -> LoadedModelBundle:
    path = Path(bundle_path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")

    payload = joblib.load(path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("Model bundle is not in the expected dictionary format.")

    model = payload["model"]
    labels = list(payload.get("labels", []))
    if not labels:
        inner_model = model.named_steps.get("model") if hasattr(model, "named_steps") else model
        labels = list(getattr(inner_model, "classes_", []))

    if not labels:
        raise ValueError("Could not determine class labels from the model bundle.")

    return LoadedModelBundle(
        model=model,
        model_name=str(payload.get("model_name", "unknown_model")),
        feature_columns=list(payload.get("feature_columns", FEATURE_COLUMNS)),
        target_column=str(payload.get("target_column", TARGET_COLUMN_NAME)),
        labels=labels,
        use_smote=bool(payload.get("use_smote", False)),
    )


def _find_optional_label_column(columns: list[str]) -> str | None:
    column_list = [str(col).strip() for col in columns]
    for candidate in SOURCE_LABEL_COLUMN_CANDIDATES:
        if candidate in column_list:
            return candidate
    return None


def _align_probabilities(model: object, X: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        inner_model = model.named_steps.get("model") if hasattr(model, "named_steps") else model
        classes = list(getattr(inner_model, "classes_", labels))
        probability_df = pd.DataFrame(probabilities, columns=classes, index=X.index)
        return probability_df.reindex(columns=labels, fill_value=0.0)

    predicted = pd.Series(model.predict(X), index=X.index)
    return pd.get_dummies(predicted, dtype=float).reindex(columns=labels, fill_value=0.0)


def _runner_up_confidence(probability_df: pd.DataFrame) -> pd.Series:
    if probability_df.shape[1] < 2:
        return pd.Series(0.0, index=probability_df.index)

    values = probability_df.to_numpy(copy=False)
    top_two = np.partition(values, -2, axis=1)[:, -2:]
    return pd.Series(top_two.min(axis=1), index=probability_df.index)


def score_input_frame(bundle: LoadedModelBundle, raw_df: pd.DataFrame) -> pd.DataFrame:
    working_df = raw_df.copy().reset_index(drop=True)
    if "Source Index" not in working_df.columns:
        working_df.insert(0, "Source Index", working_df.index)

    prepared_df = prepare_input_frame(working_df)
    X = prepared_df[bundle.feature_columns].copy()
    probability_df = _align_probabilities(bundle.model, X, bundle.labels)
    predicted = bundle.model.predict(X)
    confidence = probability_df.max(axis=1)
    runner_up = _runner_up_confidence(probability_df)

    scored_df = prepared_df.copy()
    label_col = _find_optional_label_column(list(prepared_df.columns))
    if label_col is not None:
        scored_df["Actual Fault Type"] = (
            prepared_df[label_col].astype(str).str.strip().replace(LABEL_TRANSLATION_MAP)
        )

    scored_df["Predicted Fault Type"] = predicted
    scored_df["Prediction Confidence"] = confidence.astype(float)
    scored_df["Confidence Gap"] = (confidence - runner_up).astype(float)

    for label in bundle.labels:
        scored_df[probability_column_name(label)] = probability_df[label].astype(float)

    if "Actual Fault Type" in scored_df.columns:
        scored_df["Correct Prediction"] = (
            scored_df["Actual Fault Type"] == scored_df["Predicted Fault Type"]
        )

    return scored_df
