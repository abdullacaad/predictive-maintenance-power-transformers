from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from transformer_pm.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TUNING_DIR,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN_NAME,
    TEST_RATIO,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)
from transformer_pm.data import load_dga_dataset, split_dataset
from transformer_pm.modeling import (
    align_probabilities,
    build_tuning_search_space,
    compute_metrics,
    save_confusion_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run randomized-search hyperparameter tuning with cross-validation for DGA models."
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
        default=DEFAULT_TUNING_DIR,
        help="Directory for tuning outputs.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of CV folds for randomized search.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=4,
        help="Maximum randomized-search iterations per model.",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE oversampling in training pipelines.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional subset of model names to tune. Example: --models random_forest knn",
    )
    return parser.parse_args()


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def parameter_grid_size(param_grid: dict[str, list[object]]) -> int:
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


def upsert_summary_row(summary_df: pd.DataFrame, row: dict[str, float | str]) -> pd.DataFrame:
    row_df = pd.DataFrame([row])
    if summary_df.empty:
        return row_df

    summary_df = summary_df[summary_df["model"] != row["model"]]
    return pd.concat([summary_df, row_df], ignore_index=True)


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)

    output_dir = args.output_dir
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    search_dir = output_dir / "search"
    for directory in (metrics_dir, figures_dir, models_dir, search_dir):
        directory.mkdir(parents=True, exist_ok=True)

    df = load_dga_dataset(args.data_path)
    splits = split_dataset(
        df,
        train_ratio=TRAIN_RATIO,
        val_ratio=VALIDATION_RATIO,
        test_ratio=TEST_RATIO,
        random_state=args.random_state,
    )

    X_train_val = pd.concat([splits.X_train, splits.X_val], axis=0).reset_index(drop=True)
    y_train_val = pd.concat([splits.y_train, splits.y_val], axis=0).reset_index(drop=True)
    X_test = splits.X_test.reset_index(drop=True)
    y_test = splits.y_test.reset_index(drop=True)

    labels = sorted(df[TARGET_COLUMN_NAME].unique().tolist())
    use_smote = not args.no_smote
    all_search_space = build_tuning_search_space(use_smote=use_smote, random_state=args.random_state)
    if args.models:
        invalid_models = sorted(set(args.models).difference(all_search_space.keys()))
        if invalid_models:
            raise ValueError(f"Unknown model names: {', '.join(invalid_models)}")
        search_space = {name: all_search_space[name] for name in args.models}
    else:
        search_space = all_search_space
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

    summary_path = metrics_dir / "tuned_model_summary.csv"
    partial_summary_path = metrics_dir / "tuned_model_summary_partial.csv"
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
    elif partial_summary_path.exists():
        summary_df = pd.read_csv(partial_summary_path)
    else:
        summary_df = pd.DataFrame()

    tuned_models: dict[str, object] = {}

    for model_name, (pipeline, param_grid) in search_space.items():
        print(f"Tuning {model_name}...", flush=True)
        try:
            start = time.perf_counter()
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                n_iter=min(args.n_iter, parameter_grid_size(param_grid)),
                scoring="f1_macro",
                n_jobs=1,
                cv=cv,
                verbose=1,
                refit=True,
                random_state=args.random_state,
                return_train_score=True,
            )
            search.fit(X_train_val, y_train_val)
            train_seconds = time.perf_counter() - start

            best_model = search.best_estimator_
            tuned_models[model_name] = best_model

            test_pred = best_model.predict(X_test)
            test_proba = align_probabilities(best_model, X_test, labels)
            test_metrics = compute_metrics(y_test, test_pred, test_proba, labels)

            report = classification_report(y_test, test_pred, labels=labels, zero_division=0)
            save_text(metrics_dir / f"classification_report_test_{model_name}.txt", report)
            save_confusion_matrix(
                y_test,
                test_pred,
                labels,
                figures_dir / f"confusion_matrix_test_{model_name}.png",
                f"Tuned Test Confusion Matrix - {model_name}",
            )

            cv_results = pd.DataFrame(search.cv_results_).sort_values(
                by="rank_test_score", ascending=True
            )
            cv_results.to_csv(search_dir / f"cv_results_{model_name}.csv", index=False)
            joblib.dump(best_model, models_dir / f"{model_name}.joblib")
            save_text(
                metrics_dir / f"best_params_{model_name}.json",
                json.dumps(search.best_params_, indent=2),
            )

            row: dict[str, float | str] = {
                "model": model_name,
                "search_iterations": int(min(args.n_iter, parameter_grid_size(param_grid))),
                "cv_best_f1_macro": float(search.best_score_),
                "cv_rank": int(cv_results.iloc[0]["rank_test_score"]),
                "train_seconds": round(train_seconds, 4),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            }
            row.update({f"test_{k}": v for k, v in test_metrics.items()})
            summary_df = upsert_summary_row(summary_df, row)
            summary_df.sort_values(
                by="cv_best_f1_macro",
                ascending=False,
            ).to_csv(metrics_dir / "tuned_model_summary_partial.csv", index=False)
        except Exception as exc:  # noqa: BLE001
            err = (
                f"Model {model_name} failed with {type(exc).__name__}: {exc}\n\n"
                + traceback.format_exc()
            )
            print(err, flush=True)
            save_text(metrics_dir / f"error_{model_name}.txt", err)

    if summary_df.empty:
        raise RuntimeError("All tuning runs failed. Check outputs/tuning/metrics/error_*.txt")

    summary_df = summary_df.sort_values(by="cv_best_f1_macro", ascending=False)
    summary_df.to_csv(summary_path, index=False)

    best_model_name = str(summary_df.iloc[0]["model"])
    if best_model_name in tuned_models:
        best_model = tuned_models[best_model_name]
    else:
        best_model = joblib.load(models_dir / f"{best_model_name}.joblib")
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN_NAME,
            "labels": labels,
            "use_smote": use_smote,
            "source": "randomized_search_cv",
        },
        models_dir / "best_tuned_model_bundle.joblib",
    )

    baseline_summary_path = DEFAULT_OUTPUT_DIR / "metrics" / "model_summary.csv"
    if baseline_summary_path.exists():
        baseline_df = pd.read_csv(baseline_summary_path)
        comparison_df = summary_df[["model", "cv_best_f1_macro", "test_f1_macro", "test_accuracy"]].merge(
            baseline_df[["model", "val_f1_macro", "test_f1_macro", "test_accuracy"]],
            on="model",
            how="left",
            suffixes=("_tuned", "_baseline"),
        )
        comparison_df.to_csv(metrics_dir / "comparison_vs_baseline.csv", index=False)

    tuning_config = {
        "dataset_path": str(args.data_path),
        "samples_total": int(len(df)),
        "train_val_samples": int(len(X_train_val)),
        "test_samples": int(len(X_test)),
        "cv_folds": args.cv_folds,
        "n_iter_per_model": args.n_iter,
        "use_smote": use_smote,
        "random_state": args.random_state,
        "best_model": best_model_name,
        "primary_metric": "cv_best_f1_macro",
    }
    (metrics_dir / "tuning_config.json").write_text(json.dumps(tuning_config, indent=2), encoding="utf-8")

    print("\nTuning complete.", flush=True)
    print(f"Best tuned model by CV macro F1: {best_model_name}", flush=True)
    print(f"Summary saved to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
