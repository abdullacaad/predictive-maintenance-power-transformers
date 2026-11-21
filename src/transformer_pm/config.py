from pathlib import Path

# Default experiment paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "transformer-fault-dataset-main" / "data.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODEL_BUNDLE_PATH = DEFAULT_OUTPUT_DIR / "models" / "best_model_bundle.joblib"
DEFAULT_PREDICTIONS_DIR = DEFAULT_OUTPUT_DIR / "predictions"
DEFAULT_TUNING_DIR = DEFAULT_OUTPUT_DIR / "tuning"
DEFAULT_CALIBRATION_DIR = DEFAULT_OUTPUT_DIR / "calibration"
DEFAULT_COST_CONFIG_PATH = PROJECT_ROOT / "configs" / "maintenance_defaults.json"

# Model input schema
FEATURE_COLUMNS = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]
TARGET_COLUMN_NAME = "Fault Type"

# Source label column candidates in incoming spreadsheets
SOURCE_LABEL_COLUMN_CANDIDATES = (
    TARGET_COLUMN_NAME,
    "\u6545\u969c\u7c7b\u578b",  # Chinese: Fault Type
)

# Standardized class names used in reports and figures
LABEL_TRANSLATION_MAP = {
    "\u6b63\u5e38": "Normal",
    "\u9ad8\u6e29\u8fc7\u70ed": "High-temperature Overheating",
    "\u4e2d\u6e29\u8fc7\u70ed": "Medium-temperature Overheating",
    "\u4f4e\u6e29\u8fc7\u70ed": "Low-temperature Overheating",
    "\u9ad8\u80fd\u653e\u7535": "High-energy Discharge",
    "\u4f4e\u80fd\u653e\u7535": "Low-energy Discharge",
    "\u5c40\u90e8\u653e\u7535": "Partial Discharge",
}

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
