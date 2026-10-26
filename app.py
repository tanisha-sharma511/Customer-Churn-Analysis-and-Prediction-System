"""
app.py  -  ChurnGuard dashboard (Streamlit)

    streamlit run app.py

Pages:  Dashboard | Customer Explorer | Predict a Customer | Model Performance
Run  python train.py  once BEFORE starting the dashboard (it creates the model and data files).
"""
import json

import pandas as pd
import streamlit as st

import config
from predictor import ChurnPredictor

st.set_page_config(page_title="ChurnGuard", page_icon="📉", layout="wide")


# ----------------------------------------------------------------------------
# Loading helpers (cached so the files are read only once)
# ----------------------------------------------------------------------------
@st.cache_data
def load_scored() -> pd.DataFrame:
    return pd.read_csv(config.SCORED_PATH)


@st.cache_data
def load_json(path_str: str) -> dict:
    with open(path_str, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def get_predictor() -> ChurnPredictor:
    return ChurnPredictor()


def show_figure(name: str, caption: str = "") -> None:
    path = config.FIGURES_DIR / f"{name}.png"
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Chart '{name}' not found. Run train.py again.")


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_dashboard() -> None:
    st.title("📉 ChurnGuard: Customer Churn Dashboard")
    st.caption("Telecom customer churn analysis and prediction (Python, SQL, Machine Learning)")
    scored = load_scored()
    tokens = load_json(str(config.TOKENS_PATH))
    high = scored[scored["RiskBand"] == "High"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(scored):,}")
    c2.metric("Overall churn rate", tokens["CHURN_RATE"])
    c3.metric("High-risk customers", f"{len(high):,}")
    c4.metric("Monthly revenue at risk", f"${high['MonthlyCharges'].sum():,.0f}")

    left, right = st.columns(2)
    with left:
        show_figure("eda_churn_by_contract", "Churn rate by contract type")
        show_figure("eda_tenure_distribution", "Tenure of churned vs retained customers")
    with right:
        show_figure("eda_internet_payment", "Churn by internet service and payment method")
        show_figure("risk_bands", "Actual churn rate in each predicted risk band")


def page_explorer() -> None:
    st.title("🔎 Customer Explorer")
    scored = load_scored()
    bands = st.multiselect("Risk band", ["High", "Medium", "Low"], default=["High"])
    contracts = st.multiselect("Contract", sorted(scored["Contract"].unique()),
                               default=sorted(scored["Contract"].unique()))
    view = scored[scored["RiskBand"].isin(bands) & scored["Contract"].isin(contracts)]
    view = view.sort_values("ChurnProbability", ascending=False)
    st.write(f"**{len(view):,}** customers match. Monthly revenue: **${view['MonthlyCharges'].sum():,.0f}**")
    st.dataframe(view, use_container_width=True, height=420)
    st.download_button("⬇ Download this list (CSV)", view.to_csv(index=False).encode("utf-8"),
                       file_name="customers_filtered.csv", mime="text/csv")


def page_predict() -> None:
    st.title("🎯 Predict a Customer")
    st.write("Enter a customer's details to get the churn probability, risk band and retention advice.")
    opts = config.FORM_OPTIONS
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", opts["gender"])
            senior = st.selectbox("Senior citizen", opts["SeniorCitizen"])
            partner = st.selectbox("Partner", opts["Partner"])
            dependents = st.selectbox("Dependents", opts["Dependents"])
            tenure = st.slider("Tenure (months)", 0, 72, 6)
            monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 80.0, step=0.5)
        with c2:
            phone = st.selectbox("Phone service", opts["PhoneService"])
            lines = st.selectbox("Multiple lines", opts["MultipleLines"])
            internet = st.selectbox("Internet service", opts["InternetService"])
            security = st.selectbox("Online security", opts["OnlineSecurity"])
            backup = st.selectbox("Online backup", opts["OnlineBackup"])
            protection = st.selectbox("Device protection", opts["DeviceProtection"])
        with c3:
            support = st.selectbox("Tech support", opts["TechSupport"])
            tv = st.selectbox("Streaming TV", opts["StreamingTV"])
            movies = st.selectbox("Streaming movies", opts["StreamingMovies"])
            contract = st.selectbox("Contract", opts["Contract"])
            paperless = st.selectbox("Paperless billing", opts["PaperlessBilling"])
            payment = st.selectbox("Payment method", opts["PaymentMethod"])
        submitted = st.form_submit_button("Predict churn risk")

    if submitted:
        customer = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
            "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": None,
        }
        out = get_predictor().predict_one(customer)
        st.subheader(f"Churn probability: {out['probability'] * 100:.1f}%")
        st.progress(min(max(out["probability"], 0.0), 1.0))
        band = out["risk_band"]
        if band == "High":
            st.error(f"Risk band: {band}")
        elif band == "Medium":
            st.warning(f"Risk band: {band}")
        else:
            st.success(f"Risk band: {band}")
        st.markdown("**Suggested retention actions**")
        for tip in out["recommendations"]:
            st.markdown(f"- {tip}")


def page_performance() -> None:
    st.title("📊 Model Performance")
    comp = pd.read_csv(config.COMPARISON_PATH, index_col=0)
    show = comp[["name", "ACC", "PREC", "REC", "F1", "AUC", "CVAUC"]].copy()
    show.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV AUC"]
    st.dataframe(show.round(3), use_container_width=True)
    left, right = st.columns(2)
    with left:
        show_figure("model_roc_curves", "ROC curves")
        show_figure("model_confusion_matrix", "Confusion matrix of the best model")
    with right:
        show_figure("model_comparison", "Metric comparison")
        show_figure("model_feature_importance", "Top churn drivers")


# ----------------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------------
if not config.MODEL_PATH.exists() or not config.SCORED_PATH.exists():
    st.error("The model has not been trained yet. Open a terminal in the project folder and run:  "
             "`python train.py`  then refresh this page.")
    st.stop()

PAGES = {"Dashboard": page_dashboard, "Customer Explorer": page_explorer,
         "Predict a Customer": page_predict, "Model Performance": page_performance}
choice = st.sidebar.radio("Navigation", list(PAGES))
st.sidebar.markdown("---")
st.sidebar.caption("ChurnGuard v1.0 - Data Analytics Internship Project")
PAGES[choice]()
