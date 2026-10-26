"""
modeling.py
-----------
Analytics & ML layer: train several classifiers, compare them, tune the Random Forest,
explain the best model and score every customer.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline

import config
from insights import risk_band
from preprocessing import build_preprocessor

logger = config.setup_logging()


def get_models() -> Dict[str, Tuple[str, object]]:
    """Key -> (display name, estimator). Class weights handle the churn/no-churn imbalance."""
    s = config.SEED
    return {
        "LR": ("Logistic Regression",
               LogisticRegression(max_iter=1000, class_weight="balanced", random_state=s)),
        "RF": ("Random Forest",
               RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                      class_weight="balanced_subsample", random_state=s, n_jobs=-1)),
        "GB": ("Gradient Boosting", GradientBoostingClassifier(random_state=s)),
    }


def make_pipeline(estimator) -> Pipeline:
    """Preprocessing + model in ONE object, so test data never leaks into training."""
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def evaluate_model(pipe: Pipeline, X_tr, y_tr, X_te, y_te, cv) -> Tuple[dict, np.ndarray]:
    """Fit on the training set, score on the test set (+ cross-validated AUC)."""
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
    metrics = {
        "ACC": accuracy_score(y_te, pred), "PREC": precision_score(y_te, pred, zero_division=0),
        "REC": recall_score(y_te, pred), "F1": f1_score(y_te, pred),
        "AUC": roc_auc_score(y_te, proba), "CVAUC": cv_auc,
    }
    return metrics, proba


def tune_random_forest(X_tr, y_tr) -> Tuple[Pipeline, dict]:
    """3-fold grid search over the main Random Forest settings (scored by ROC-AUC)."""
    pipe = make_pipeline(RandomForestClassifier(class_weight="balanced_subsample",
                                                random_state=config.SEED, n_jobs=-1))
    grid = GridSearchCV(
        pipe,
        param_grid={"model__n_estimators": [200, 400], "model__max_depth": [8, 12, None],
                    "model__min_samples_leaf": [3, 5, 10]},
        cv=3, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_tr, y_tr)
    logger.info("Best RF parameters: %s", grid.best_params_)
    return grid.best_estimator_, grid.best_params_


def best_f1_threshold(y_true, proba) -> Tuple[float, float, float]:
    """Decision threshold that maximises F1 -> (threshold, precision, recall)."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    i = int(np.argmax(f1))
    return float(thr[i]), float(prec[i]), float(rec[i])


def feature_importance(pipe: Pipeline, X_te, y_te) -> pd.DataFrame:
    """Permutation importance: how much does ROC-AUC drop if a feature is shuffled?"""
    perm = permutation_importance(pipe, X_te, y_te, scoring="roc_auc", n_repeats=5,
                                  random_state=config.SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": X_te.columns, "mean": perm.importances_mean,
                        "std": perm.importances_std})
    return imp.sort_values("mean", ascending=False).reset_index(drop=True)


def score_customers(pipe: Pipeline, X, y, cv) -> np.ndarray:
    """Out-of-fold churn probabilities: every customer is scored by a model that never saw them."""
    return cross_val_predict(clone(pipe), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def train_all(X: pd.DataFrame, y: pd.Series, tune: bool = True) -> dict:
    """Run the whole modelling stage and return everything the other modules need."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)

    fitted, results, probas = {}, {}, {}
    for key, (name, est) in get_models().items():
        pipe = make_pipeline(est)
        results[key], probas[key] = evaluate_model(pipe, X_tr, y_tr, X_te, y_te, cv)
        results[key]["name"] = name
        fitted[key] = pipe
        logger.info("%-20s AUC=%.3f  F1=%.3f", name, results[key]["AUC"], results[key]["F1"])

    best_params = None
    if tune:
        tuned, best_params = tune_random_forest(X_tr, y_tr)
        results["RFT"], probas["RFT"] = evaluate_model(tuned, X_tr, y_tr, X_te, y_te, cv)
        results["RFT"]["name"] = "Tuned Random Forest"
        fitted["RFT"] = tuned
        logger.info("%-20s AUC=%.3f  F1=%.3f", "Tuned Random Forest",
                    results["RFT"]["AUC"], results["RFT"]["F1"])

    best_key = max(results, key=lambda k: results[k]["AUC"])
    best = fitted[best_key]
    best_proba = probas[best_key]
    logger.info("Best model: %s", results[best_key]["name"])

    tn, fp, fn, tp = confusion_matrix(y_te, (best_proba >= 0.5).astype(int)).ravel()
    thr, thr_prec, thr_rec = best_f1_threshold(y_te, best_proba)
    importance = feature_importance(best, X_te, y_te)

    oof = score_customers(best, X, y, cv)
    final_model = clone(best).fit(X, y)           # refit on all data for the dashboard

    return {
        "X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te,
        "results": results, "probas": probas, "fitted": fitted,
        "best_key": best_key, "best": best, "best_proba": best_proba, "best_params": best_params,
        "confusion": (int(tn), int(fp), int(fn), int(tp)),
        "threshold": (thr, thr_prec, thr_rec), "importance": importance,
        "oof_proba": oof, "final_model": final_model,
    }


def risk_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Customers, actual churn rate and monthly revenue for each risk band."""
    scored = scored.copy()
    scored["RiskBand"] = scored["ChurnProbability"].apply(risk_band)
    band = scored.groupby("RiskBand").agg(
        customers=("customerID", "count"),
        actual_churn=("Churn", lambda s: (s == "Yes").mean()),
        monthly_rev=("MonthlyCharges", "sum"),
    ).reindex(list(config.RISK_LABELS)).fillna(0)
    return band
