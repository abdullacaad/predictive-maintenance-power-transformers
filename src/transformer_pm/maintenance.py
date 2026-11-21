from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DEFAULT_COST_CONFIG_PATH
from .inference import probability_column_name


def load_cost_config(config_path: str | Path = DEFAULT_COST_CONFIG_PATH) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Cost configuration not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_score(row: pd.Series, labels: list[str], mapping: dict[str, float]) -> float:
    total = 0.0
    for label in labels:
        probability = float(row.get(probability_column_name(label), 0.0))
        total += probability * float(mapping.get(label, 0.0))
    return total


def _priority_from_risk(risk_score: float, thresholds: dict[str, float]) -> str:
    if risk_score >= float(thresholds["critical"]):
        return "Critical"
    if risk_score >= float(thresholds["urgent"]):
        return "Urgent"
    if risk_score >= float(thresholds["planned"]):
        return "Planned"
    return "Routine"


def build_maintenance_plan(
    scored_df: pd.DataFrame,
    *,
    labels: list[str],
    cost_config: dict,
) -> tuple[pd.DataFrame, dict]:
    required_columns = {"Predicted Fault Type", "Prediction Confidence"}
    missing_columns = required_columns.difference(scored_df.columns)
    if missing_columns:
        raise ValueError(
            "Scored dataframe is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    df = scored_df.copy()
    severity_by_class = cost_config["severity_by_class"]
    fault_event_multiplier = cost_config["fault_event_multiplier_by_class"]
    actions_by_class = cost_config["recommended_actions_by_class"]
    priority_thresholds = cost_config["priority_thresholds"]
    response_target_hours = cost_config["response_target_hours"]
    planned_intervention_cost = cost_config["planned_intervention_cost_by_priority"]

    reactive_strategy = cost_config["reactive_strategy"]
    predictive_strategy = cost_config["predictive_strategy"]
    reactive_cost_ceiling = float(reactive_strategy["repair_cost"]) + float(
        reactive_strategy["downtime_cost"]
    )
    monitoring_cost_per_asset = float(predictive_strategy["monitoring_cost_per_asset"])
    preventive_effectiveness = float(predictive_strategy["preventive_effectiveness"])

    if not 0.0 <= preventive_effectiveness <= 1.0:
        raise ValueError("preventive_effectiveness must be between 0 and 1.")

    df["Predicted Severity Score"] = (
        df["Predicted Fault Type"].map(severity_by_class).fillna(0.0).astype(float)
    )
    df["Expected Fault Event Multiplier"] = df.apply(
        lambda row: _expected_score(row, labels, fault_event_multiplier),
        axis=1,
    )
    df["Expected Severity Score"] = df.apply(
        lambda row: _expected_score(row, labels, severity_by_class),
        axis=1,
    )
    df["Risk Score"] = (
        0.75 * df["Expected Fault Event Multiplier"]
        + 0.25 * df["Prediction Confidence"] * df["Predicted Severity Score"]
    ).clip(lower=0.0, upper=1.0)
    df["Priority"] = df["Risk Score"].apply(
        lambda score: _priority_from_risk(float(score), priority_thresholds)
    )
    df["Recommended Action"] = (
        df["Predicted Fault Type"]
        .map(actions_by_class)
        .fillna("Manual engineering review required")
    )
    df["Response Target Hours"] = (
        df["Priority"]
        .map(response_target_hours)
        .fillna(response_target_hours.get("Routine", 720))
        .astype(float)
    )

    df["Expected Reactive Cost"] = df["Risk Score"] * reactive_cost_ceiling
    df["Expected Predictive Cost"] = (
        monitoring_cost_per_asset
        + df["Priority"].map(planned_intervention_cost).fillna(0.0).astype(float)
        + df["Expected Reactive Cost"] * (1.0 - preventive_effectiveness)
    )
    df["Estimated Savings"] = (
        df["Expected Reactive Cost"] - df["Expected Predictive Cost"]
    )

    priority_order = {"Critical": 0, "Urgent": 1, "Planned": 2, "Routine": 3}
    df["Priority Rank"] = df["Priority"].map(priority_order).fillna(99)
    df = df.sort_values(
        by=["Priority Rank", "Risk Score", "Prediction Confidence"],
        ascending=[True, False, False],
    ).drop(columns=["Priority Rank"])
    df = df.reset_index(drop=True)

    summary = {
        "currency": cost_config.get("currency", "GBP"),
        "sample_count": int(len(df)),
        "average_risk_score": round(float(df["Risk Score"].mean()), 4),
        "priority_counts": df["Priority"].value_counts().to_dict(),
        "predicted_class_counts": df["Predicted Fault Type"].value_counts().to_dict(),
        "total_expected_reactive_cost": round(float(df["Expected Reactive Cost"].sum()), 2),
        "total_expected_predictive_cost": round(float(df["Expected Predictive Cost"].sum()), 2),
        "estimated_net_savings": round(float(df["Estimated Savings"].sum()), 2),
        "assumptions": {
            "reactive_repair_cost": float(reactive_strategy["repair_cost"]),
            "reactive_downtime_cost": float(reactive_strategy["downtime_cost"]),
            "monitoring_cost_per_asset": monitoring_cost_per_asset,
            "preventive_effectiveness": preventive_effectiveness,
        },
    }

    return df, summary
