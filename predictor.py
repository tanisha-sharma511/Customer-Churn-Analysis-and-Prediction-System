"""
predictor.py
------------
Loads the saved model and predicts churn for ONE customer (used by the dashboard).
"""
from typing import Dict

import joblib
import pandas as pd

import config
from insights import recommendations, risk_band
from preprocessing import prepare_features

logger = config.setup_logging()


class ChurnPredictor:
    """Thin wrapper around the trained scikit-learn pipeline."""

    def __init__(self, model_path=config.MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}\nRun  python train.py  first to create it.")
        self.bundle = joblib.load(model_path)
        self.pipeline = self.bundle["pipeline"]
        self.model_name = self.bundle["model_name"]
        logger.info("Model loaded: %s", self.model_name)

    def predict_one(self, customer: Dict) -> Dict:
        """customer = dict with the 19 raw feature fields (TotalCharges optional)."""
        data = dict(customer)
        if data.get("TotalCharges") in (None, ""):
            data["TotalCharges"] = float(data["tenure"]) * float(data["MonthlyCharges"])
        X = prepare_features(pd.DataFrame([data]))
        prob = float(self.pipeline.predict_proba(X)[0, 1])
        return {
            "probability": prob,
            "risk_band": risk_band(prob),
            "will_churn": prob >= 0.5,
            "recommendations": recommendations(data, prob),
        }
