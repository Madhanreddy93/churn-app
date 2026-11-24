# Keepwise: Customer Churn Prediction & Retention Intelligence

Streamlit dashboard that predicts which customers will churn, explains why, and suggests retention actions.

## Run
```
pip install -r requirements.txt
python generate_sample_data.py     # creates data/sample_churn.csv
streamlit run app.py
```

## Structure
| File | Purpose |
|---|---|
| app.py | Streamlit UI with 4 pages |
| ml.py | Cleaning, pipeline, training, metrics, feature importance |
| retention.py | Rule-based retention actions |
| generate_sample_data.py | Synthetic Telco-style dataset |

## Pages
1. **Upload data**: CSV upload (or sample), pick churn column, ID column and model.
2. **Churn prediction**: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve, single-customer predictor, scored CSV download.
3. **Risk analysis**: risk tiers, revenue at risk, churn drivers, segment and distribution charts.
4. **Retention suggestions**: per-customer reasons and actions, downloadable action list.

## Pipeline
CSV -> clean -> impute/scale/one-hot -> Logistic Regression | Random Forest | Gradient Boosting -> probability -> risk tier -> retention rules.

## Ideas to extend
SMOTE for imbalance, SHAP explanations, hyperparameter tuning, saving models with joblib, login and a database.
