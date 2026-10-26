
import sqlite3
from pathlib import Path
from typing import Dict

import pandas as pd

import config

logger = config.setup_logging()


def load_raw_data(path=config.DEFAULT_CSV) -> pd.DataFrame:
    """Read the Telco CSV and check that all required columns exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download 'Telco Customer Churn' (WA_Fn-UseC_-Telco-Customer-Churn.csv) from Kaggle "
            "and place it in the 'data' folder, or pass --csv <path>."
        )
    df = pd.read_csv(path)
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"The CSV is missing required columns: {missing}")
    logger.info("Loaded %d rows x %d columns from %s", df.shape[0], df.shape[1], path.name)
    return df


def load_to_sqlite(df: pd.DataFrame, db_path=config.DB_PATH, table: str = "customers") -> None:
    """Write the raw data into a SQLite table (replaces it if it already exists)."""
    config.ensure_dirs()
    with sqlite3.connect(db_path) as con:
        df.to_sql(table, con, if_exists="replace", index=False)
    logger.info("Raw data stored in SQLite table '%s' (%s)", table, Path(db_path).name)


def churn_rate_by(column: str, db_path=config.DB_PATH, table: str = "customers") -> pd.DataFrame:
    """SQL: number of customers and churn % for every value of a categorical column."""
    if column not in config.SQL_GROUP_COLUMNS:           # prevents SQL injection
        raise ValueError(f"Column '{column}' is not allowed. Use one of {config.SQL_GROUP_COLUMNS}")
    query = f"""
        SELECT {column} AS category,
               COUNT(*) AS customers,
               ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_pct
        FROM {table}
        GROUP BY {column}
        ORDER BY churn_pct DESC;
    """
    with sqlite3.connect(db_path) as con:
        return pd.read_sql(query, con)


def run_sql_summaries(db_path=config.DB_PATH) -> Dict[str, pd.DataFrame]:
    """Run the churn-rate query for every column listed in config.SQL_GROUP_COLUMNS."""
    result = {col: churn_rate_by(col, db_path) for col in config.SQL_GROUP_COLUMNS}
    logger.info("SQL summaries computed for %s", list(result))
    return result
