
import logging
from pathlib import Path

# ----------------------------------------------------------------------------
# Folders and files
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
LOG_DIR = BASE_DIR / "logs"

DEFAULT_CSV = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DB_PATH = OUTPUT_DIR / "telco.db"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
TOKENS_PATH = OUTPUT_DIR / "report_tokens.json"
SQL_SUMMARY_PATH = OUTPUT_DIR / "sql_summary.json"
SCORED_PATH = OUTPUT_DIR / "customers_scored.csv"
COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"
IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"

# ----------------------------------------------------------------------------
# Machine-learning settings
# ----------------------------------------------------------------------------
SEED = 42                 # fixed seed -> same results on every run
TEST_SIZE = 0.20          # 80:20 train/test split
CV_FOLDS = 5              # cross-validation folds
RISK_THRESHOLDS = (0.30, 0.60)   # <=0.30 Low, <=0.60 Medium, else High
RISK_LABELS = ("Low", "Medium", "High")

# ----------------------------------------------------------------------------
# Dataset schema (IBM Telco Customer Churn)
# ----------------------------------------------------------------------------
ID_COL = "customerID"
TARGET_COL = "Churn"
TARGET_FLAG = "ChurnFlag"

RAW_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]
ENGINEERED_FEATURES = ["NumServices", "AvgMonthlySpend", "TenureGroup"]
FEATURE_COLUMNS = RAW_FEATURES + ENGINEERED_FEATURES
REQUIRED_COLUMNS = [ID_COL] + RAW_FEATURES + [TARGET_COL]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "NumServices", "AvgMonthlySpend"]
CATEGORICAL_FEATURES = [c for c in FEATURE_COLUMNS if c not in NUMERIC_FEATURES]

SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

# Columns for which the SQL layer computes churn rate (also a safe-list for queries)
SQL_GROUP_COLUMNS = ["Contract", "InternetService", "PaymentMethod"]

# Drop-down options used by the dashboard's prediction form
FORM_OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}


def ensure_dirs() -> None:
    """Create every output folder if it does not exist yet."""
    for folder in (DATA_DIR, MODELS_DIR, OUTPUT_DIR, FIGURES_DIR, LOG_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str = "churnguard") -> logging.Logger:
    """Return a logger that writes to the console and to logs/app.log."""
    ensure_dirs()
    logger = logging.getLogger(name)
    if logger.handlers:                      # already configured -> reuse
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logfile = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    logfile.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(logfile)
    return logger
