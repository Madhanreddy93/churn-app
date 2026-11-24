"""Preprocessing, training and scoring helpers."""
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=42),
}

def to_binary(s: pd.Series) -> pd.Series:
    m = {"yes": 1, "y": 1, "true": 1, "1": 1, "churned": 1, "no": 0, "n": 0, "false": 0, "0": 0, "retained": 0}
    out = s.astype(str).str.strip().str.lower().map(m)
    if out.isna().any():
        raise ValueError("Target column must contain Yes/No, True/False or 1/0 values.")
    return out.astype(int)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:  # numbers stored as text (e.g. blank TotalCharges)
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().mean() > 0.9:
                df[c] = conv
    return df.drop_duplicates()

def split_columns(X: pd.DataFrame):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return num, cat

def build_pipeline(X: pd.DataFrame, model_name: str) -> Pipeline:
    num, cat = split_columns(X)
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    return Pipeline([("pre", pre), ("clf", MODELS[model_name]())])

def train(df, target, id_col, model_name, test_size=0.2):
    df = clean(df)
    y = to_binary(df[target])
    X = df.drop(columns=[c for c in [target, id_col] if c and c in df.columns])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    pipe = build_pipeline(X, model_name).fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(yte, proba)
    metrics = {"Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred, zero_division=0),
               "Recall": recall_score(yte, pred), "F1": f1_score(yte, pred), "ROC-AUC": roc_auc_score(yte, proba)}
    return dict(pipe=pipe, X=X, y=y, df=df, metrics=metrics, cm=confusion_matrix(yte, pred),
                roc=(fpr, tpr), target=target, id_col=id_col, model_name=model_name)

def importances(pipe) -> pd.DataFrame:
    names = pipe.named_steps["pre"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    vals = np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else clf.feature_importances_
    out = pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names], "importance": vals})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)

def risk_tier(p, high=0.6, med=0.3):
    return np.where(p >= high, "High", np.where(p >= med, "Medium", "Low"))
