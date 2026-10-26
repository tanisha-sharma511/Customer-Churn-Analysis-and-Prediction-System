"""
visualize.py
------------
Presentation layer (charts). Every chart is saved as a PNG in reports/figures.
All charts have the same size so they sit neatly in a report.
"""
import matplotlib

matplotlib.use("Agg")                      # no pop-up windows; just save files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

import config

FIGSIZE = (7, 4)
PALETTE = {"No": "#2a9d8f", "Yes": "#e76f51"}
sns.set_theme(style="whitegrid", font="DejaVu Sans")


def _save(fig, name: str) -> None:
    config.ensure_dirs()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{name}.png", dpi=150)
    plt.close(fig)


def eda_charts(df: pd.DataFrame, sql: dict) -> None:
    """Six exploratory charts. `df` is the cleaned data, `sql` the SQL summary tables."""
    # 1. churn distribution
    fig, ax = plt.subplots(figsize=FIGSIZE)
    cnt = df["Churn"].value_counts().reindex(["No", "Yes"])
    bars = ax.bar(["Retained (No)", "Churned (Yes)"], cnt.values, color=[PALETTE["No"], PALETTE["Yes"]])
    for b, v in zip(bars, cnt.values):
        ax.text(b.get_x() + b.get_width() / 2, v + cnt.max() * 0.01,
                f"{v}  ({v / len(df) * 100:.1f}%)", ha="center", fontsize=10)
    ax.set_ylabel("Number of customers"); ax.set_title("Customer Churn Distribution")
    ax.set_ylim(0, cnt.max() * 1.15)
    _save(fig, "eda_churn_distribution")

    # 2. churn by contract (SQL result)
    q = sql["Contract"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(data=q, x="category", y="churn_pct", color="#457b9d", ax=ax)
    for i, r in q.iterrows():
        ax.text(i, r.churn_pct + 0.8, f"{r.churn_pct:.1f}%", ha="center", fontsize=10)
    ax.set_ylabel("Churn rate (%)"); ax.set_xlabel("")
    ax.set_title("Churn Rate by Contract Type (SQL result)"); ax.set_ylim(0, q.churn_pct.max() * 1.18)
    _save(fig, "eda_churn_by_contract")

    # 3. tenure distribution
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.histplot(data=df, x="tenure", hue="Churn", bins=24, palette=PALETTE, element="step", ax=ax)
    ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Customers")
    ax.set_title("Tenure Distribution: Churned vs Retained")
    _save(fig, "eda_tenure_distribution")

    # 4. charges boxplots
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn", palette=PALETTE, legend=False, ax=axes[0])
    axes[0].set_title("Monthly Charges ($)")
    sns.boxplot(data=df, x="Churn", y="TotalCharges", hue="Churn", palette=PALETTE, legend=False, ax=axes[1])
    axes[1].set_title("Total Charges ($)")
    _save(fig, "eda_charges_boxplot")

    # 5. internet service and payment method
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    qi = sql["InternetService"].sort_values("churn_pct")
    axes[0].barh(qi["category"], qi["churn_pct"], color="#e9c46a")
    axes[0].set_title("By Internet Service", fontsize=10); axes[0].set_xlabel("Churn rate (%)")
    for i, v in enumerate(qi["churn_pct"]):
        axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    axes[0].grid(axis="y", visible=False); axes[0].set_xlim(0, qi["churn_pct"].max() * 1.25)
    qp = sql["PaymentMethod"].sort_values("churn_pct")
    axes[1].barh(qp["category"].str.replace(" (automatic)", "\n(auto)", regex=False), qp["churn_pct"], color="#f4a261")
    axes[1].set_title("By Payment Method", fontsize=10); axes[1].set_xlabel("Churn rate (%)")
    axes[1].tick_params(axis="y", labelsize=7)
    for i, v in enumerate(qp["churn_pct"]):
        axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    axes[1].grid(axis="y", visible=False); axes[1].set_xlim(0, qp["churn_pct"].max() * 1.25)
    _save(fig, "eda_internet_payment")

    # 6. correlation heatmap
    fig, ax = plt.subplots(figsize=FIGSIZE)
    cols = ["tenure", "MonthlyCharges", "TotalCharges", "NumServices", "AvgMonthlySpend", config.TARGET_FLAG]
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"shrink": .8}, annot_kws={"size": 8})
    ax.set_title("Correlation Heatmap of Numerical Features")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    _save(fig, "eda_correlation")


def result_charts(res: dict, band: pd.DataFrame) -> None:
    """Five charts about the models and the risk bands."""
    results, probas = res["results"], res["probas"]
    y_test = res["y_test"]

    m = pd.DataFrame(results).T.reset_index(drop=True)
    mm = m.melt(id_vars="name", value_vars=["ACC", "PREC", "REC", "F1", "AUC"], var_name="Metric", value_name="Score")
    mm["Score"] = mm["Score"].astype(float)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(data=mm, x="Metric", y="Score", hue="name", ax=ax, palette="Set2")
    ax.set_ylim(0, 1); ax.set_title("Model Comparison on the Hold-out Test Set"); ax.legend(fontsize=7, loc="lower right")
    _save(fig, "model_comparison")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for k in results:
        fpr, tpr, _ = roc_curve(y_test, probas[k])
        ax.plot(fpr, tpr, label=f"{results[k]['name']} (AUC = {results[k]['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curves")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, "model_roc_curves")

    tn, fp, fn, tp = res["confusion"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Predicted: Stay", "Predicted: Churn"], yticklabels=["Actual: Stay", "Actual: Churn"])
    ax.set_title(f"Confusion Matrix - {results[res['best_key']]['name']}")
    _save(fig, "model_confusion_matrix")

    top10 = res["importance"].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.barh(top10["feature"], top10["mean"], xerr=top10["std"], color="#457b9d")
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    ax.set_title("Top 10 Churn Drivers (Permutation Importance)")
    _save(fig, "model_feature_importance")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
    bars = ax.bar(band.index.astype(str), band["actual_churn"] * 100, color=colors)
    top = max(band["actual_churn"].max() * 100, 1)
    for b, (_, r) in zip(bars, band.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, r.actual_churn * 100 + top * 0.03,
                f"{r.actual_churn * 100:.1f}%\n({int(r.customers)} customers)", ha="center", fontsize=9)
    ax.set_ylabel("Actual churn rate (%)"); ax.set_title("Actual Churn Rate by Predicted Risk Band")
    ax.set_ylim(0, top * 1.3)
    _save(fig, "risk_bands")
