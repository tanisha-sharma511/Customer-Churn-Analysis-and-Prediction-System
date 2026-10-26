"""
train.py  -  run the WHOLE analytics pipeline with one command

    python train.py                       # full run (with hyper-parameter tuning)
    python train.py --skip-tuning         # faster run (3 models, no grid search)
    python train.py --csv path/to/file.csv

Stages: load -> SQLite + SQL -> clean -> features -> EDA charts -> train & compare models
        -> explain -> risk-score customers -> save everything.
"""
import argparse
import json
import sys

import joblib
import pandas as pd
import sklearn

import config
import modeling
import reporting
import visualize
from insights import risk_band
from data_loader import load_raw_data, load_to_sqlite, run_sql_summaries
from preprocessing import add_features, clean_raw, prepare_features

logger = config.setup_logging()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ChurnGuard training pipeline")
    ap.add_argument("--csv", default=str(config.DEFAULT_CSV), help="path to the Telco CSV")
    ap.add_argument("--skip-tuning", action="store_true", help="skip GridSearchCV (faster)")
    ap.add_argument("--no-figures", action="store_true", help="do not draw charts")
    args = ap.parse_args(argv)

    config.ensure_dirs()
    if "SYNTHETIC" in args.csv.upper():
        logger.warning("=" * 70)
        logger.warning("SYNTHETIC DEMO DATA IN USE - results are NOT real. Do not put them in your report.")
        logger.warning("=" * 70)

    # 1. data layer
    try:
        raw = load_raw_data(args.csv)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    load_to_sqlite(raw)
    sql = run_sql_summaries()
    (config.SQL_SUMMARY_PATH).write_text(
        json.dumps({k: v.to_dict("records") for k, v in sql.items()}, indent=1), encoding="utf-8")

    # 2. processing layer
    df, quality = clean_raw(raw)
    logger.info("Cleaning: %s", quality)
    X, y = prepare_features(df), df[config.TARGET_FLAG]

    # 3. analytics layer
    if not args.no_figures:
        visualize.eda_charts(add_features(df), sql)
    res = modeling.train_all(X, y, tune=not args.skip_tuning)

    # 4. risk scoring
    scored = df[[config.ID_COL, "Contract", "InternetService", "PaymentMethod",
                 "tenure", "MonthlyCharges", config.TARGET_COL]].copy()
    scored["ChurnProbability"] = res["oof_proba"].round(4)
    band = modeling.risk_table(scored)
    scored["RiskBand"] = scored["ChurnProbability"].apply(risk_band)
    if not args.no_figures:
        visualize.result_charts(res, band)

    # 5. save everything
    scored.to_csv(config.SCORED_PATH, index=False)
    pd.DataFrame(res["results"]).T.to_csv(config.COMPARISON_PATH)
    res["importance"].to_csv(config.IMPORTANCE_PATH, index=False)
    best = res["results"][res["best_key"]]
    joblib.dump({"pipeline": res["final_model"], "model_name": best["name"],
                 "feature_columns": config.FEATURE_COLUMNS, "sklearn_version": sklearn.__version__},
                config.MODEL_PATH)
    tokens = reporting.build_tokens(sql, quality, df, res, band, raw.shape[1])
    config.TOKENS_PATH.write_text(json.dumps(tokens, indent=1), encoding="utf-8")
    config.METRICS_PATH.write_text(json.dumps(
        {k: {m: (round(v, 4) if isinstance(v, float) else v) for m, v in r.items()}
         for k, r in res["results"].items()} | {"best_model": best["name"]}, indent=1), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f" Best model : {best['name']}")
    print(f" ROC-AUC    : {best['AUC']:.3f}   Recall: {best['REC']:.3f}   F1: {best['F1']:.3f}")
    print(f" Top drivers: {', '.join(res['importance'].feature[:3])}")
    print(f" High-risk customers: {int(band.loc['High', 'customers'])}  "
          f"(monthly revenue ${band.loc['High', 'monthly_rev']:,.0f})")
    print("=" * 62)
    print(" Saved: models/churn_model.joblib, outputs/*.csv|json, reports/figures/*.png")
    print(" Next : streamlit run app.py   (opens the dashboard)")
    return 0


if __name__ == "__main__":          # this guard is required on Windows
    sys.exit(main())
