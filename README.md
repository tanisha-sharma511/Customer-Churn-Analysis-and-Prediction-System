# ChurnGuard: Telecom Customer Churn Analysis & Prediction System

A complete data-analytics project built with **Python, SQL (SQLite) and Machine Learning**.
It answers three business questions about a telecom company:

1. **Who** is likely to leave (churn)?
2. **Why** are they leaving?
3. **How much revenue** is at risk, and what should the retention team do?

Dataset: IBM *Telco Customer Churn* (7,043 customers, 21 columns), free on Kaggle.

---------------------------------------------------------------------------------------------

## 1. Project structure

```
churnguard/
├── app.py               Streamlit dashboard (4 pages)           <- you RUN this
├── train.py             Runs the whole pipeline with 1 command  <- you RUN this first
├── config.py            All paths, constants, logging
├── data_loader.py       Data layer: CSV -> SQLite -> SQL queries
├── preprocessing.py     Cleaning, feature engineering, preprocessor
├── modeling.py          Train/compare/tune models, explain, risk-score
├── visualize.py         All charts (saved as PNG)
├── insights.py          Risk bands + retention recommendations
├── predictor.py         Predict ONE customer (used by dashboard)
├── reporting.py         Collects every number for your report (report_tokens.json)
├── tests/               8 automated tests
├── tools/               make_synthetic_data.py (fake data, only for testing)
├── data/                <- put the Kaggle CSV here
├── models/              created by train.py  (churn_model.joblib)
├── outputs/             created by train.py  (scored CSV, metrics, SQLite DB, tokens)
├── reports/figures/     created by train.py  (all charts for the report / screenshots)
├── logs/                created by train.py  (app.log)
├── requirements.txt
└── README.md
```

Architecture (4 layers): **Data** (CSV + SQLite) -> **Processing** (clean + features) ->
**Analytics/ML** (EDA + models) -> **Presentation** (charts + dashboard + CSV for Power BI).

---------------------------------------------------------------------------------------------

## 2. One-time setup

You need **Python 3.10 or newer** (check with `python --version`).

### Windows (Command Prompt or VS Code terminal)
```
cd path\to\churnguard
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux
```
cd path/to/churnguard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You will see `(venv)` at the start of the terminal line when the environment is active.
Activate it again (`venv\Scripts\activate`) every time you open a new terminal.

### Get the dataset
1. Go to Kaggle and search **"Telco Customer Churn"** (by *blastchar*, IBM sample data).
2. Download and unzip. The file is `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
3. Put it inside the **`data/`** folder.

---------------------------------------------------------------------------------------------

## 3. Run the project (3 commands)

```
python train.py            # 1. runs everything (takes about 1-3 minutes)
streamlit run app.py       # 2. opens the dashboard in your browser (http://localhost:8501)
python -m unittest discover -s tests -v    # 3. optional: run the automated tests
```

Faster practice run (skips hyper-parameter tuning): `python train.py --skip-tuning`

**What `train.py` does, step by step**

| Step | What happens | Output |
|---|---|---|
| 1 | Reads the CSV and checks the columns | |
| 2 | Loads it into SQLite and runs SQL `GROUP BY` queries | `outputs/telco.db`, `sql_summary.json` |
| 3 | Cleans data (blank TotalCharges, duplicates) and adds 3 new features | |
| 4 | Draws 6 exploratory charts | `reports/figures/eda_*.png` |
| 5 | Trains Logistic Regression, Random Forest, Gradient Boosting + tuned Random Forest | `outputs/model_comparison.csv` |
| 6 | Evaluates (accuracy, precision, recall, F1, ROC-AUC), explains with permutation importance | `reports/figures/model_*.png` |
| 7 | Scores every customer, assigns Low/Medium/High risk, computes revenue at risk | `outputs/customers_scored.csv` |
| 8 | Saves the final model and every number for your report | `models/churn_model.joblib`, `outputs/report_tokens.json` |

**The dashboard pages**
- **Dashboard**: KPI cards (customers, churn rate, high-risk customers, revenue at risk) + key charts.
- **Customer Explorer**: filter customers by risk band and contract, download the list as CSV.
- **Predict a Customer**: fill a form -> churn probability, risk band and retention advice.
- **Model Performance**: comparison table, ROC curves, confusion matrix, top churn drivers.

Take your **screenshots for the report appendix** from these pages.

---------------------------------------------------------------------------------------------

## 4. Test that your setup works (before downloading Kaggle data)

```
python tools/make_synthetic_data.py
python train.py --csv data/SYNTHETIC_demo_data.csv --skip-tuning
```
This uses **FAKE data**. The program prints a warning. **Never use these numbers in your report.**
Delete `data/SYNTHETIC_demo_data.csv` and run `python train.py` on the real CSV when done.

---------------------------------------------------------------------------------------------

## 5. How to manage the project (good practice)

**Daily workflow**
1. Open the folder in VS Code, activate `venv`.
2. Change code -> run `python train.py --skip-tuning` for a quick check.
3. Run the tests -> `python -m unittest discover -s tests -v`.
4. Commit to Git.

**Where to change things**
| I want to... | Edit this file |
|---|---|
| Change the risk band limits (0.30 / 0.60) | `config.py` -> `RISK_THRESHOLDS` |
| Change train/test split, CV folds, random seed | `config.py` |
| Add a new engineered feature | `preprocessing.py` -> `add_features()` and `config.py` -> `ENGINEERED_FEATURES`, `NUMERIC_FEATURES` |
| Add a new model (e.g. XGBoost) | `modeling.py` -> `get_models()` |
| Change retention advice | `insights.py` -> `recommendations()` |
| Add a chart | `visualize.py` |
| Add a dashboard page | `app.py` -> new `page_xxx()` and add it to `PAGES` |

**Git and GitHub (so you can put the link in your report)**
```
git init
git add .
git commit -m "ChurnGuard v1.0 - churn analysis and prediction"
git branch -M main
git remote add origin https://github.com/<your-username>/churnguard.git
git push -u origin main
```
`.gitignore` already stops the dataset, model and generated outputs from being uploaded.
Anyone who clones your repo only needs: setup (section 2) -> add CSV -> `python train.py`.

**Versioning idea (like "Version 1, 2, 3" in a report):** v1.0 = this project.
v1.1 = add XGBoost. v2.0 = deploy the model as an API (FastAPI).

---------------------------------------------------------------------------------------------

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `Dataset not found` | Put the CSV in `data/` with the exact name, or run `python train.py --csv "full\path\file.csv"` |
| `ModuleNotFoundError` | Activate the venv, then `pip install -r requirements.txt` |
| Dashboard says "model has not been trained yet" | Run `python train.py` first, then refresh the page |
| `streamlit` not recognised | `python -m streamlit run app.py` |
| Training is slow | Use `--skip-tuning` |
| Charts look different from the report | Fine. Re-run `train.py`, the report uses `outputs/report_tokens.json` |

---------------------------------------------------------------------------------------------

## 7. Viva cheat-sheet (be ready to explain)

- **Why not just accuracy?** Only ~26% of customers churn, so a model predicting "nobody leaves"
  is 74% accurate but useless. We use **recall** and **ROC-AUC**.
- **Class imbalance?** Handled with `class_weight="balanced"`.
- **Data leakage?** Scaling/encoding live inside a scikit-learn **Pipeline**, fitted only on training data.
- **Why SQL?** To profile churn by group (`GROUP BY` + `CASE WHEN`) before modelling.
- **Why a Random Forest AND a Logistic Regression?** One is interpretable, the other captures
  non-linear patterns. We pick the best by ROC-AUC on unseen data.
- **Permutation importance?** Shuffle one feature; if ROC-AUC drops a lot, the model relied on it.
- **Out-of-fold probabilities?** Each customer is scored by a model that never saw them, so risk bands are honest.
- **Limitation?** One public dataset, correlation is not causation, batch (not real-time).
