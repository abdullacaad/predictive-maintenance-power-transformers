from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from transformer_pm.calibration import (
    build_cost_sensitivity_table,
    derive_priority_thresholds,
    summarize_priority_mix,
    with_updated_thresholds,
)
from transformer_pm.config import (
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_COST_CONFIG_PATH,
    DEFAULT_PREDICTIONS_DIR,
)
from transformer_pm.maintenance import build_maintenance_plan, load_cost_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate maintenance thresholds and generate cost sensitivity analysis."
    )
    parser.add_argument(
        "--scored-path",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR / "scored_samples.csv",
        help="Path to scored sample predictions CSV.",
    )
    parser.add_argument(
        "--cost-config",
        type=Path,
        default=DEFAULT_COST_CONFIG_PATH,
        help="Path to base maintenance cost assumptions JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="Directory for calibration outputs.",
    )
    parser.add_argument(
        "--critical-share",
        type=float,
        default=0.15,
        help="Target share of assets marked Critical.",
    )
    parser.add_argument(
        "--urgent-share",
        type=float,
        default=0.20,
        help="Target share of assets marked Urgent.",
    )
    parser.add_argument(
        "--planned-share",
        type=float,
        default=0.25,
        help="Target share of assets marked Planned.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scored_df = pd.read_csv(args.scored_path)
    cost_config = load_cost_config(args.cost_config)
    labels = list(cost_config["severity_by_class"].keys())

    target_priority_mix = {
        "Critical": float(args.critical_share),
        "Urgent": float(args.urgent_share),
        "Planned": float(args.planned_share),
    }
    thresholds = derive_priority_thresholds(
        scored_df,
        labels=labels,
        cost_config=cost_config,
        target_priority_mix=target_priority_mix,
    )

    calibrated_config = with_updated_thresholds(cost_config, thresholds)
    calibrated_plan, calibrated_summary = build_maintenance_plan(
        scored_df,
        labels=labels,
        cost_config=calibrated_config,
    )
    achieved_mix = summarize_priority_mix(calibrated_plan)

    calibrated_config_path = args.output_dir / "calibrated_maintenance_config.json"
    calibrated_plan_path = args.output_dir / "calibrated_maintenance_plan.csv"
    calibration_report_path = args.output_dir / "calibration_report.json"
    sensitivity_path = args.output_dir / "cost_sensitivity_analysis.csv"

    calibrated_config_path.write_text(json.dumps(calibrated_config, indent=2), encoding="utf-8")
    calibrated_plan.round(6).to_csv(calibrated_plan_path, index=False)

    monitoring_values = [250.0, 500.0, 750.0, 1000.0, 1500.0]
    effectiveness_values = [0.25, 0.35, 0.45, 0.55, 0.65]
    sensitivity_df = build_cost_sensitivity_table(
        scored_df,
        labels=labels,
        cost_config=calibrated_config,
        monitoring_cost_options=monitoring_values,
        preventive_effectiveness_options=effectiveness_values,
    )
    sensitivity_df.round(2).to_csv(sensitivity_path, index=False)

    report = {
        "base_thresholds": deepcopy(cost_config["priority_thresholds"]),
        "calibrated_thresholds": thresholds,
        "target_priority_mix": target_priority_mix,
        "achieved_priority_mix": achieved_mix,
        "sample_count": int(len(calibrated_plan)),
        "estimated_net_savings": calibrated_summary["estimated_net_savings"],
        "total_expected_predictive_cost": calibrated_summary["total_expected_predictive_cost"],
        "total_expected_reactive_cost": calibrated_summary["total_expected_reactive_cost"],
        "sensitivity_analysis_rows": int(len(sensitivity_df)),
        "recommended_use": "Use the calibrated thresholds for dissertation sensitivity analysis unless utility-specific workload targets are available.",
    }
    calibration_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Calibrated config saved to: {calibrated_config_path}")
    print(f"Calibrated maintenance plan saved to: {calibrated_plan_path}")
    print(f"Calibration report saved to: {calibration_report_path}")
    print(f"Cost sensitivity analysis saved to: {sensitivity_path}")


if __name__ == "__main__":
    main()
