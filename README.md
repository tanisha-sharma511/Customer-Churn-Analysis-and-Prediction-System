# ChurnGuard: Telecom Customer Churn Analysis & Prediction System

A complete data-analytics project built with **Python, SQL (SQLite) and Machine Learning**.
The business question: which telecom customers are about to leave, why, and how much revenue is at risk?
It answers three business questions about a telecom company:

1. **Who** is likely to leave (churn)?
2. **Why** are they leaving?
3. **How much revenue** is at risk, and what should the retention team do?

Dataset: IBM *Telco Customer Churn* (7,043 customers, 21 columns), free on Kaggle.

---------------------------------------------------------------------------------------------

## 🎯 Business Problem

Customer churn is a major challenge for telecommunications companies because losing existing customers can affect revenue and long-term business growth.

The objective of this project is to understand the factors associated with customer churn and build a machine learning model to identify customers who may be at risk of leaving the company.

## 🎯 Project Objectives

- Analyze customer churn patterns and trends.
- Identify factors associated with customer attrition.
- Explore relationships between tenure, contract type, monthly charges, and churn.
- Develop a machine learning model to predict customer churn.
- Generate actionable insights to support customer retention strategies.

## 📊 Dataset

- **Dataset:** IBM Telco Customer Churn
- **Source:** Kaggle
- **Link:**[IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains customer information, including demographics, account details, subscribed services, contract types, payment methods, monthly charges, total charges, and churn status.

## 🛠️ Tools & Technologies

- **Python** — Data analysis and machine learning
- **Pandas & NumPy** — Data manipulation and preprocessing
- **Matplotlib & Seaborn** — Exploratory data analysis and visualization
- **Scikit-learn** — Machine learning and model evaluation
- **Jupyter Notebook** — Development and experimentation

## 🔄 Project Workflow

1. **Data Collection:** Obtained the IBM Telco Customer Churn dataset from Kaggle.
2. **Data Cleaning:** Handled missing values, invalid entries, and data type inconsistencies where applicable.
3. **Exploratory Data Analysis:** Analyzed churn distribution and customer characteristics.
4. **Data Preprocessing:** Prepared numerical and categorical features for machine learning.
5. **Model Development:** Trained a classification model to predict customer churn.
6. **Model Evaluation:** Evaluated the model using appropriate classification metrics.
7. **Business Insights:** Identified potential customer retention opportunities based on the analysis.

## 📈 Key Analysis Areas

- Overall customer churn rate
- Churn across different contract types
- Relationship between customer tenure and churn
- Monthly charges and customer churn
- Payment methods and service subscriptions
- Customer segments with higher churn rates

## 🤖 Machine Learning Model

The project uses a supervised machine learning classification approach to predict whether a customer is likely to churn.

**Model Used:** Add the name of your actual trained model here.

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC, if calculated

Actual model performance metrics can be added after evaluating the trained model on test data.

## 💡 Key Findings

Add the actual insights discovered during your analysis, such as:

- Customer groups with higher observed churn rates.
- Contract types associated with higher churn.
- The relationship between tenure and customer retention.
- The model's ability to identify customers at risk of leaving.

## 💼 Business Recommendations

Based on the analysis, potential strategies for reducing customer churn include:

- Developing targeted retention campaigns for high-risk customers.
- Reviewing contract options and pricing for customers with elevated churn rates.
- Improving onboarding and engagement for newer customers.
- Monitoring service usage and customer satisfaction.
- Using predictive insights to prioritize customer retention efforts.

These recommendations should be refined based on the actual findings of the project.

## 📁 Project Structure

churnguard/
├── app.py               Streamlit dashboard           
├── train.py             Runs the whole pipeline with 1 command  
├── config.py            All paths, constants, logging
├── data_loader.py       Data layer: CSV -> SQLite -> SQL queries
├── preprocessing.py     Cleaning, feature engineering, preprocessor
├── modeling.py          Train/compare/tune models, explain, risk-score
├── visualize.py         All charts (saved as PNG)
├── insights.py          Risk bands + retention recommendations
├── predictor.py         Predict ONE customer 
├── reporting.py         Collects every number for your report (report_tokens.json)
├── tests/               automated tests
├── tools/               make_synthetic_data.py (fake data, only for testing)
├── data/                <- put the Kaggle CSV here
├── models/              created by train.py  (churn_model.joblib)
├── outputs/             created by train.py  (scored CSV, metrics, SQLite DB, tokens)
├── reports/figures/     created by train.py  (all charts for the report / screenshots)
├── logs/                created by train.py  (app.log)
├── requirements.txt
└── README.md
```

## 🚀 How to Run the Project

1. Clone or download this repository.
2. Install the required Python libraries.
3. Open the Jupyter Notebook.
4. Run the notebook cells in sequence to reproduce the analysis and model evaluation.

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## 🏁 Conclusion

This project demonstrates an end-to-end approach to telecom customer churn analysis and prediction by combining data cleaning, exploratory data analysis, machine learning, and business interpretation.

The goal is to help businesses better understand customer attrition, identify customers at risk of leaving, and explore data-driven customer retention strategies.

