from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from .maintenance import build_maintenance_plan


def derive_priority_thresholds(
    scored_df: pd.DataFrame,
    *,
    labels: list[str],
    cost_config: dict,
    target_priority_mix: dict[str, float],
) -> dict[str, float]:
    base_plan, _ = build_maintenance_plan(scored_df, labels=labels, cost_config=cost_config)
    risk_scores = base_plan["Risk Score"].to_numpy(copy=True)

    critical_share = float(target_priority_mix["Critical"])
    urgent_share = float(target_priority_mix["Urgent"])
    planned_share = float(target_priority_mix["Planned"])

    if critical_share <= 0 or urgent_share <= 0 or planned_share <= 0:
        raise ValueError("Target priority shares must be positive.")
    if critical_share + urgent_share + planned_share >= 1.0:
        raise ValueError("Critical + Urgent + Planned target shares must sum to less than 1.")

    critical_threshold = float(np.quantile(risk_scores, 1.0 - critical_share))
    urgent_threshold = float(np.quantile(risk_scores, 1.0 - (critical_share + urgent_share)))
    planned_threshold = float(np.quantile(risk_scores, 1.0 - (critical_share + urgent_share + planned_share)))

    urgent_threshold = min(urgent_threshold, critical_threshold - 1e-6)
    planned_threshold = min(planned_threshold, urgent_threshold - 1e-6)

    return {
        "critical": round(max(critical_threshold, 0.0), 6),
        "urgent": round(max(urgent_threshold, 0.0), 6),
        "planned": round(max(planned_threshold, 0.0), 6),
    }


def with_updated_thresholds(cost_config: dict, thresholds: dict[str, float]) -> dict:
    updated = deepcopy(cost_config)
    updated["priority_thresholds"] = {
        "critical": float(thresholds["critical"]),
        "urgent": float(thresholds["urgent"]),
        "planned": float(thresholds["planned"]),
    }
    return updated


def summarize_priority_mix(plan_df: pd.DataFrame) -> dict[str, float]:
    counts = plan_df["Priority"].value_counts(normalize=True).to_dict()
    return {
        "Critical": round(float(counts.get("Critical", 0.0)), 4),
        "Urgent": round(float(counts.get("Urgent", 0.0)), 4),
        "Planned": round(float(counts.get("Planned", 0.0)), 4),
        "Routine": round(float(counts.get("Routine", 0.0)), 4),
    }


def build_cost_sensitivity_table(
    scored_df: pd.DataFrame,
    *,
    labels: list[str],
    cost_config: dict,
    monitoring_cost_options: list[float],
    preventive_effectiveness_options: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for monitoring_cost in monitoring_cost_options:
        for effectiveness in preventive_effectiveness_options:
            scenario_config = deepcopy(cost_config)
            scenario_config["predictive_strategy"]["monitoring_cost_per_asset"] = float(monitoring_cost)
            scenario_config["predictive_strategy"]["preventive_effectiveness"] = float(effectiveness)
            _, summary = build_maintenance_plan(scored_df, labels=labels, cost_config=scenario_config)
            rows.append(
                {
                    "monitoring_cost_per_asset": float(monitoring_cost),
                    "preventive_effectiveness": float(effectiveness),
                    "estimated_net_savings": float(summary["estimated_net_savings"]),
                    "total_expected_predictive_cost": float(summary["total_expected_predictive_cost"]),
                    "total_expected_reactive_cost": float(summary["total_expected_reactive_cost"]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        by=["estimated_net_savings", "preventive_effectiveness"],
        ascending=[False, False],
    )
