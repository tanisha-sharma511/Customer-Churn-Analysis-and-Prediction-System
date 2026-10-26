"""
preprocessing.py
----------------
Processing layer: cleaning, feature engineering and the scikit-learn preprocessor.

prepare_features() is used for BOTH training and for single-customer prediction,
so the model always sees data prepared in exactly the same way.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def _normalise_senior(series: pd.Series) -> pd.Series:
    """Accept 0/1 or No/Yes and always return 'No'/'Yes'."""
    mapping = {"0": "No", "1": "Yes", "0.0": "No", "1.0": "Yes", "No": "No", "Yes": "Yes"}
    return series.astype(str).map(mapping).fillna("No")


def clean_raw(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean the raw data. Returns the clean frame and a data-quality report."""
    df = df.copy()
    report = {"rows_in": int(len(df))}

    report["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    numeric_tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
    report["invalid_total_charges"] = int(numeric_tc.isna().sum())   # blanks (new customers)
    df["TotalCharges"] = numeric_tc.fillna(0.0)

    df["SeniorCitizen"] = _normalise_senior(df["SeniorCitizen"])
    if config.TARGET_COL in df.columns:
        df[config.TARGET_FLAG] = (df[config.TARGET_COL] == "Yes").astype(int)

    report["rows_out"] = int(len(df))
    return df.reset_index(drop=True), report


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add NumServices, AvgMonthlySpend and TenureGroup."""
    df = df.copy()
    df["NumServices"] = (df[config.SERVICE_COLS] == "Yes").sum(axis=1)
    tenure = df["tenure"].astype(float)
    safe_tenure = tenure.where(tenure > 0, np.nan)
    df["AvgMonthlySpend"] = np.where(tenure > 0, df["TotalCharges"] / safe_tenure, df["MonthlyCharges"])
    df["TenureGroup"] = pd.cut(tenure, bins=[-1, 12, 24, 48, 72],
                               labels=["0-12 m", "13-24 m", "25-48 m", "49-72 m"]).astype(str)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw-like frame -> model-ready feature frame (same columns, same order, always)."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = _normalise_senior(df["SeniorCitizen"])
    df = add_features(df)
    return df[config.FEATURE_COLUMNS]


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categorical columns."""
    return ColumnTransformer([
        ("num", StandardScaler(), config.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
    ])
