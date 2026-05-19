from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATASET_PATH = RAW_DIR / DATASET_NAME

MODEL_FILENAME = "churn_logreg_pipeline.joblib"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

RANDOM_STATE = 42
TEST_SIZE = 0.2