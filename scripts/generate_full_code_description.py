from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import re

ROOT = Path(r"d:\MY PROJECTS\THESIS data science")
CODE_ROOT = ROOT / "code"
OUT_PATH = ROOT / "report" / "thesis_content" / "full_code_description_for_thesis.txt"

FILES = [
    CODE_ROOT / "README.md",
    CODE_ROOT / "requirements.txt",
    CODE_ROOT / "run_app.ps1",
    CODE_ROOT / "run_app.bat",
    CODE_ROOT / "configs" / "maintenance_defaults.json",
    CODE_ROOT / "src" / "train_baselines.py",
    CODE_ROOT / "src" / "tune_models.py",
    CODE_ROOT / "src" / "score_maintenance.py",
    CODE_ROOT / "src" / "calibrate_maintenance.py",
    CODE_ROOT / "src" / "transformer_pm" / "__init__.py",
    CODE_ROOT / "src" / "transformer_pm" / "config.py",
    CODE_ROOT / "src" / "transformer_pm" / "data.py",
    CODE_ROOT / "src" / "transformer_pm" / "modeling.py",
    CODE_ROOT / "src" / "transformer_pm" / "inference.py",
    CODE_ROOT / "src" / "transformer_pm" / "maintenance.py",
    CODE_ROOT / "src" / "transformer_pm" / "calibration.py",
    CODE_ROOT / "app" / "streamlit_app.py",
]

ARTIFACTS = [
    CODE_ROOT / "outputs" / "metrics" / "experiment_config.json",
    CODE_ROOT / "outputs" / "metrics" / "model_summary.csv",
    CODE_ROOT / "outputs" / "predictions" / "scoring_evaluation.json",
    CODE_ROOT / "outputs" / "predictions" / "maintenance_summary.json",
    CODE_ROOT / "outputs" / "calibration" / "calibration_report.json",
    CODE_ROOT / "outputs" / "calibration" / "cost_sensitivity_analysis.csv",
]

FILE_DESCRIPTIONS = {
    "code/README.md": {
        "role": "Master technical narrative of the project.",
        "inputs": "Repository outputs and implementation decisions.",
        "outputs": "Human-readable guidance for thesis alignment and reproducibility.",
        "thesis_use": "Use as narrative source for methodology and implementation consistency checks.",
        "risks": "If outdated, thesis claims can drift from code behavior.",
    },
    "code/requirements.txt": {
        "role": "Dependency lock guidance for environment setup.",
        "inputs": "Package names and minimum versions.",
        "outputs": "Consistent runtime environment for scripts and app.",
        "thesis_use": "Cite as evidence of software stack and tools used.",
        "risks": "Version mismatches can alter model behavior or break scripts.",
    },
    "code/run_app.ps1": {
        "role": "PowerShell launcher for Streamlit app execution.",
        "inputs": "Port and optional startup flags.",
        "outputs": "Interactive dashboard process with predictable startup behavior.",
        "thesis_use": "Supports artifact deployment and practical usability claim.",
        "risks": "Port conflicts can prevent app launch.",
    },
    "code/run_app.bat": {
        "role": "Batch launcher convenience for Windows execution.",
        "inputs": "Command-line options and port choices.",
        "outputs": "Simplified app startup and stop workflow.",
        "thesis_use": "Demonstrates accessibility of the implemented artifact.",
        "risks": "Hardcoded assumptions about shell context may require local adjustment.",
    },
    "code/configs/maintenance_defaults.json": {
        "role": "Central decision-policy and cost assumptions configuration.",
        "inputs": "Threshold values, severity mappings, and financial parameters.",
        "outputs": "Deterministic maintenance plan behavior and cost estimation policy.",
        "thesis_use": "Use directly when reporting thresholds and economic assumptions.",
        "risks": "Any edits here materially change output costs and priorities.",
    },
    "code/src/train_baselines.py": {
        "role": "Primary training pipeline for baseline models and report generation.",
        "inputs": "Preprocessed DGA data and reproducibility settings.",
        "outputs": "Model artifacts, confusion matrices, classification reports, summary metrics.",
        "thesis_use": "Core source for Chapter 3 experimental protocol and Chapter 5 baseline comparison.",
        "risks": "Changing model defaults or metric logic will shift all benchmark conclusions.",
    },
    "code/src/tune_models.py": {
        "role": "Hyperparameter search workflow with held-out test preservation.",
        "inputs": "Train+validation pool, search spaces, CV settings.",
        "outputs": "Best parameter records, tuned model bundle, comparison tables.",
        "thesis_use": "Use to justify optimization methodology and tuning limits.",
        "risks": "Small search budgets may under-explore complex model spaces.",
    },
    "code/src/score_maintenance.py": {
        "role": "Inference entrypoint and maintenance-plan generation script.",
        "inputs": "Input spreadsheet, trained model bundle, cost config.",
        "outputs": "Scored rows, maintenance plan, JSON summaries, optional labeled evaluation.",
        "thesis_use": "Use for the operationalization narrative from prediction to action.",
        "risks": "Input schema mismatch can prevent scoring or cause dropped rows.",
    },
    "code/src/calibrate_maintenance.py": {
        "role": "Workload-target calibration and scenario sensitivity analysis.",
        "inputs": "Scored prediction file and cost assumptions.",
        "outputs": "Calibrated thresholds, calibrated plan, calibration report, sensitivity table.",
        "thesis_use": "Supports policy-level interpretation and decision robustness analysis.",
        "risks": "Unrealistic target shares can generate impractical threshold configurations.",
    },
    "code/src/transformer_pm/__init__.py": {
        "role": "Package export surface for reusable components.",
        "inputs": "Module-level imports.",
        "outputs": "Cleaner import ergonomics across scripts.",
        "thesis_use": "Minor but useful to document package modularization.",
        "risks": "Export drift can break script imports.",
    },
    "code/src/transformer_pm/config.py": {
        "role": "Global constants for paths, schema, labels, and split ratios.",
        "inputs": "Static definitions.",
        "outputs": "Shared reproducibility contract across workflows.",
        "thesis_use": "Authoritative source for feature columns, class map, and split settings.",
        "risks": "Any inconsistency between text and this file is a thesis contradiction.",
    },
    "code/src/transformer_pm/data.py": {
        "role": "Data loading, normalization, validation, and stratified splitting utilities.",
        "inputs": "Raw tables from xlsx/csv files.",
        "outputs": "Cleaned feature matrix and target labels with robust split objects.",
        "thesis_use": "Use for preprocessing and leakage prevention description.",
        "risks": "Silent coercion and row dropping must be clearly reported.",
    },
    "code/src/transformer_pm/modeling.py": {
        "role": "Shared model-search and metric support logic.",
        "inputs": "Model specs, scoring settings, CV parameters.",
        "outputs": "Configured search spaces, evaluated summaries, reusable helpers.",
        "thesis_use": "Use to explain why tuning pipeline is systematic and reproducible.",
        "risks": "Metric misalignment can lead to selecting models with the wrong objective.",
    },
    "code/src/transformer_pm/inference.py": {
        "role": "Model bundle loading and robust scoring transformation.",
        "inputs": "Serialized bundle and incoming tabular data.",
        "outputs": "Predicted labels, probabilities, confidence, and optional correctness fields.",
        "thesis_use": "Core basis for discussing confidence-aware inference.",
        "risks": "Label-order and probability-alignment errors can corrupt downstream plans.",
    },
    "code/src/transformer_pm/maintenance.py": {
        "role": "Risk, priority, action, and cost computation engine.",
        "inputs": "Scored predictions and maintenance config.",
        "outputs": "Maintenance plan table and summary cost metrics.",
        "thesis_use": "Principal source for formulas and decision logic.",
        "risks": "Small config changes can significantly alter expected savings.",
    },
    "code/src/transformer_pm/calibration.py": {
        "role": "Threshold derivation and sensitivity-table utilities.",
        "inputs": "Risk distribution, target shares, scenario options.",
        "outputs": "Calibrated thresholds and scenario-level cost comparisons.",
        "thesis_use": "Use when discussing operational policy tuning.",
        "risks": "Poorly chosen targets may optimize workload mix but reduce service realism.",
    },
    "code/app/streamlit_app.py": {
        "role": "Human-facing operational UI for scoring and plan review.",
        "inputs": "Model path, config path, uploaded/default dataset.",
        "outputs": "Visual summaries, charts, tables, and downloadable CSV outputs.",
        "thesis_use": "Evidence of practical artifact and deployment readiness at prototype level.",
        "risks": "UI layer may hide errors if input-validation messaging is unclear.",
    },
}


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def write_header(lines: list[str]) -> None:
    lines.extend(
        [
            "FULL CODE DESCRIPTION FOR THESIS (MASTER VERSION)",
            "Project: Transformer DGA Predictive Maintenance",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "1. DOCUMENT INTENT",
            "This document is intentionally exhaustive and thesis-ready.",
            "It includes architecture, workflows, formulas, artifact mapping, and full code snippets.",
            "It is designed so you can extract text directly for Chapters 3, 4, 5, and reproducibility appendices.",
            "",
            "2. HIGH-LEVEL SYSTEM SUMMARY",
            "Implemented pipeline stages:",
            "2.1 Data ingestion and schema validation from Excel.",
            "2.2 Label harmonization and preprocessing.",
            "2.3 Stratified train/validation/test split (70/15/15).",
            "2.4 Baseline model training (5 model families).",
            "2.5 Hyperparameter tuning with RandomizedSearchCV.",
            "2.6 Batch inference with probabilities and confidence metrics.",
            "2.7 Maintenance risk scoring and priority assignment.",
            "2.8 Cost projection (reactive vs predictive).",
            "2.9 Threshold calibration to target workload shares.",
            "2.10 Cost sensitivity analysis across scenario grids.",
            "2.11 Streamlit UI for practical operation.",
            "",
            "3. THESIS ALIGNMENT GUIDE",
            "3.1 Chapter 1: Motivation from asset criticality, DGA relevance, and diagnostics-to-decision gap.",
            "3.2 Chapter 2: DGA, classical methods, class imbalance, and ML rationale.",
            "3.3 Chapter 3: Data contract, split protocol, models, metrics, cost equations, calibration method.",
            "3.4 Chapter 4: Repository architecture, script responsibilities, execution flow.",
            "3.5 Chapter 5: Report only values present in generated artifacts.",
            "3.6 Chapter 6: Discuss limitations and realistic deployment constraints.",
            "",
            "4. IMPLEMENTED FORMULAS",
            "4.1 RiskScore = 0.75 * ExpectedFaultEventMultiplier + 0.25 * PredictionConfidence * PredictedSeverityScore",
            "4.2 ReactiveCostCeiling = repair_cost + downtime_cost",
            "4.3 ExpectedReactiveCost = RiskScore * ReactiveCostCeiling",
            "4.4 ExpectedPredictiveCost = monitoring_cost_per_asset + planned_intervention_cost(priority) + ExpectedReactiveCost * (1 - preventive_effectiveness)",
            "4.5 EstimatedSavings = ExpectedReactiveCost - ExpectedPredictiveCost",
            "",
            "5. SOURCE FILE INVENTORY",
        ]
    )


def add_inventory(lines: list[str]) -> None:
    total = 0
    for idx, path in enumerate(FILES, start=1):
        if path.exists():
            count = line_count(path)
            total += count
            rel = path.relative_to(ROOT)
            lines.append(f"5.{idx} {rel.as_posix()} | lines={count}")
    lines.append(f"5.total_core_lines {total}")
    lines.append("")


def add_artifact_snapshot(lines: list[str]) -> None:
    lines.append("6. ARTIFACT SNAPSHOT")
    for idx, path in enumerate(ARTIFACTS, start=1):
        rel = path.relative_to(ROOT).as_posix()
        lines.append(f"6.{idx} {rel}")
        if not path.exists():
            lines.append("  - missing")
            continue
        text = path.read_text(encoding="utf-8")
        preview = text.splitlines()[:40]
        lines.append("  - preview:")
        for pl in preview:
            lines.append(f"    {pl}")
    lines.append("")


def extract_symbols(content: list[str]) -> tuple[list[str], list[str], list[str]]:
    funcs: list[str] = []
    classes: list[str] = []
    constants: list[str] = []
    for raw in content:
        line = raw.strip()
        m_func = re.match(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
        if m_func:
            funcs.append(m_func.group(1))
        m_class = re.match(r"class\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if m_class:
            classes.append(m_class.group(1))
        m_const = re.match(r"([A-Z][A-Z0-9_]+)\s*=", line)
        if m_const:
            constants.append(m_const.group(1))
    return funcs, classes, constants


def add_pipeline_walkthrough(lines: list[str]) -> None:
    lines.extend(
        [
            "7. END-TO-END EXECUTION WALKTHROUGH",
            "7.1 Stage A - Data loading",
            "The system loads spreadsheet data, resolves label columns, coerces numeric features, and drops invalid rows.",
            "7.2 Stage B - Split protocol",
            "Data is split stratified into train/validation/test with fixed random seed to maintain reproducibility.",
            "7.3 Stage C - Baseline training",
            "Five model families are trained through optional scaling and optional SMOTE wrapped pipelines.",
            "7.4 Stage D - Selection",
            "Models are compared using validation macro F1 and best model is bundled for inference.",
            "7.5 Stage E - Inference",
            "Scoring creates predicted class, class probabilities, confidence metrics, and optional correctness fields.",
            "7.6 Stage F - Maintenance mapping",
            "Predictions are mapped to severity, expected fault-event behavior, risk score, and actionable priority.",
            "7.7 Stage G - Economic projection",
            "Reactive and predictive expected costs are computed per row and aggregated for summary decisions.",
            "7.8 Stage H - Calibration and sensitivity",
            "Thresholds are re-derived for workload targets and cost scenarios are tested across monitoring/effectiveness grids.",
            "",
        ]
    )


def add_file_section(lines: list[str], section_no: int, path: Path) -> None:
    rel = path.relative_to(ROOT).as_posix()
    content = path.read_text(encoding="utf-8").splitlines()
    funcs, classes, constants = extract_symbols(content)
    meta = FILE_DESCRIPTIONS.get(
        rel,
        {
            "role": "Core project file.",
            "inputs": "Context-dependent runtime inputs.",
            "outputs": "Context-dependent outputs.",
            "thesis_use": "Use as implementation evidence.",
            "risks": "Behavior changes if edited.",
        },
    )

    lines.append(f"{section_no}. FILE DETAIL: {rel}")
    lines.append(f"{section_no}.1 Purpose")
    lines.append(meta["role"])
    lines.append(f"{section_no}.2 Detailed behavior summary")
    lines.append(
        "This section explains what the file does operationally, how it interacts with other modules, and why it matters for reproducibility."
    )
    lines.append(f"{section_no}.3 Inputs")
    lines.append(meta["inputs"])
    lines.append(f"{section_no}.4 Outputs")
    lines.append(meta["outputs"])
    lines.append(f"{section_no}.5 Key implementation symbols")
    lines.append(f"- Functions detected: {', '.join(funcs) if funcs else 'None'}")
    lines.append(f"- Classes detected: {', '.join(classes) if classes else 'None'}")
    lines.append(f"- Constants detected: {', '.join(constants) if constants else 'None'}")
    lines.append(f"{section_no}.6 Thesis writing interpretation")
    lines.append(meta["thesis_use"])
    lines.append(f"{section_no}.7 Risks and assumptions")
    lines.append(meta["risks"])
    lines.append(f"{section_no}.8 Full code snippet with line numbers")
    lines.append("BEGIN_SNIPPET")
    for i, line in enumerate(content, start=1):
        lines.append(f"{i:04d}: {line}")
    lines.append("END_SNIPPET")
    lines.append(f"{section_no}.9 Notes")
    lines.append("- Use this snippet when describing module behavior in implementation chapters.")
    lines.append("- Preserve parameter names and defaults exactly as listed in code.")
    lines.append("- Do not claim behavior that is absent in this snippet.")
    lines.append("")


def add_contradictions_and_checks(lines: list[str], start_no: int) -> None:
    lines.append(f"{start_no}. CONTRADICTION CHECKLIST")
    checks = [
        "Do not claim customer-churn domain; this project is transformer DGA diagnosis.",
        "Do not claim 80/20 split; implemented split is 70/15/15.",
        "Do not claim TensorFlow/Keras baseline ANN; executed baseline ANN is sklearn MLPClassifier.",
        "Do not claim GridSearchCV 10-fold as default executed tuning; current saved run uses RandomizedSearchCV with 3 folds and n_iter=4.",
        "Do not state 7 gases in current baseline feature set; implemented features are H2, CH4, C2H6, C2H4, C2H2.",
        "Treat full-file labeled scoring metrics as workflow verification unless explicitly isolated from training/validation overlap.",
    ]
    for i, check in enumerate(checks, start=1):
        lines.append(f"{start_no}.{i} {check}")
    lines.append("")


def add_runbook(lines: list[str], section_no: int) -> None:
    lines.extend(
        [
            f"{section_no}. REPRODUCIBILITY RUNBOOK",
            f"{section_no}.1 Activate environment",
            ".venv\\Scripts\\Activate.ps1",
            f"{section_no}.2 Install dependencies",
            "pip install -r code\\requirements.txt",
            f"{section_no}.3 Train baselines",
            "python code\\src\\train_baselines.py",
            f"{section_no}.4 Tune models",
            "python code\\src\\tune_models.py",
            f"{section_no}.5 Score and build maintenance plan",
            "python code\\src\\score_maintenance.py",
            f"{section_no}.6 Calibrate thresholds and sensitivity",
            "python code\\src\\calibrate_maintenance.py",
            f"{section_no}.7 Launch app",
            "python -m streamlit run code\\app\\streamlit_app.py --server.port 8502",
            "",
        ]
    )


def add_footer(lines: list[str]) -> None:
    lines.extend(
        [
            "FINAL NOTES",
            "- This file is intentionally long and explicit for dissertation-grade traceability.",
            "- All code snippets are copied from source files in the workspace.",
            "- If implementation changes, regenerate this document before final submission.",
            "END OF DOCUMENT",
        ]
    )


def main() -> None:
    lines: list[str] = []
    write_header(lines)
    add_inventory(lines)
    add_artifact_snapshot(lines)
    add_pipeline_walkthrough(lines)

    section_no = 8
    for path in FILES:
        if path.exists():
            add_file_section(lines, section_no, path)
            section_no += 1

    add_contradictions_and_checks(lines, section_no)
    section_no += 1
    add_runbook(lines, section_no)
    add_footer(lines)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Created: {OUT_PATH}")
    print(f"Lines: {len(lines)}")


if __name__ == "__main__":
    main()
