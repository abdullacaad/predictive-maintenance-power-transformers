from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from transformer_pm.config import (
    DEFAULT_COST_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_BUNDLE_PATH,
    DEFAULT_PREDICTIONS_DIR,
)
from transformer_pm.data import read_input_table
from transformer_pm.inference import load_model_bundle, score_input_frame
from transformer_pm.maintenance import build_maintenance_plan, load_cost_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a DGA spreadsheet and produce a prioritized maintenance plan."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the input spreadsheet (.xlsx).",
    )
    parser.add_argument(
        "--model-bundle",
        type=Path,
        default=DEFAULT_MODEL_BUNDLE_PATH,
        help="Path to the saved best-model bundle (.joblib).",
    )
    parser.add_argument(
        "--cost-config",
        type=Path,
        default=DEFAULT_COST_CONFIG_PATH,
        help="Path to maintenance cost assumptions (.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
        help="Directory for scored prediction artifacts.",
    )
    return parser.parse_args()


def evaluate_predictions(scored_df: pd.DataFrame, labels: list[str]) -> tuple[dict | None, str | None]:
    if "Actual Fault Type" not in scored_df.columns:
        return None, None

    y_true = scored_df["Actual Fault Type"]
    y_pred = scored_df["Predicted Fault Type"]
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            4,
        ),
        "recall_macro": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            4,
        ),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            4,
        ),
    }
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    return metrics, report


def round_float_columns(df: pd.DataFrame) -> pd.DataFrame:
    rounded = df.copy()
    float_columns = rounded.select_dtypes(include=["float32", "float64"]).columns
    rounded[float_columns] = rounded[float_columns].round(6)
    return rounded


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_model_bundle(args.model_bundle)
    raw_df = read_input_table(args.input_path)
    scored_df = score_input_frame(bundle, raw_df)
    cost_config = load_cost_config(args.cost_config)
    plan_df, summary = build_maintenance_plan(
        scored_df,
        labels=bundle.labels,
        cost_config=cost_config,
    )

    scored_path = args.output_dir / "scored_samples.csv"
    plan_path = args.output_dir / "maintenance_plan.csv"
    summary_path = args.output_dir / "maintenance_summary.json"

    round_float_columns(scored_df).to_csv(scored_path, index=False)
    round_float_columns(plan_df).to_csv(plan_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metrics, report = evaluate_predictions(plan_df, bundle.labels)
    if metrics is not None and report is not None:
        (args.output_dir / "scoring_evaluation.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        (args.output_dir / "classification_report_scoring.txt").write_text(
            report,
            encoding="utf-8",
        )

    print(f"Scored samples saved to: {scored_path}")
    print(f"Maintenance plan saved to: {plan_path}")
    print(f"Maintenance summary saved to: {summary_path}")
    if metrics is not None:
        print(f"Scoring accuracy on labeled input: {metrics['accuracy']}")
        print(f"Scoring macro F1 on labeled input: {metrics['f1_macro']}")


if __name__ == "__main__":
    main()
