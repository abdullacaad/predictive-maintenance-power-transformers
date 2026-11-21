from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    FEATURE_COLUMNS,
    LABEL_TRANSLATION_MAP,
    SOURCE_LABEL_COLUMN_CANDIDATES,
    TARGET_COLUMN_NAME,
)


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def _find_label_column(columns: Iterable[str]) -> str:
    column_list = [str(col).strip() for col in columns]
    for candidate in SOURCE_LABEL_COLUMN_CANDIDATES:
        if candidate in column_list:
            return candidate

    # Fallback: assume the last column is the target if known candidates are absent.
    if column_list:
        return column_list[-1]
    raise ValueError("Could not detect a label column in the dataset.")


def read_input_table(data_path: str | Path) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw_df = pd.read_excel(path)
    raw_df.columns = [str(col).strip() for col in raw_df.columns]
    return raw_df


def prepare_input_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(
            "Missing required feature columns: "
            + ", ".join(missing_features)
        )

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows remain after dropping missing feature values.")

    return df


def load_dga_dataset(data_path: str | Path) -> pd.DataFrame:
    prepared_df = prepare_input_frame(read_input_table(data_path))

    label_col = _find_label_column(prepared_df.columns)

    df = prepared_df[FEATURE_COLUMNS + [label_col]].copy()
    df = df.rename(columns={label_col: TARGET_COLUMN_NAME})

    df[TARGET_COLUMN_NAME] = (
        df[TARGET_COLUMN_NAME]
        .astype(str)
        .str.strip()
        .replace(LABEL_TRANSLATION_MAP)
    )

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN_NAME]).reset_index(drop=True)

    if df[TARGET_COLUMN_NAME].nunique() < 2:
        raise ValueError("Label column must contain at least two classes.")

    return df


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> DatasetSplits:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN_NAME].copy()

    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=y,
    )

    test_fraction_of_temp = test_ratio / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_fraction_of_temp,
        random_state=random_state,
        stratify=y_temp,
    )

    return DatasetSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
